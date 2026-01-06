#!/usr/bin/env python3
"""Self-Improving Trading Agent with Aggressive Targets.

This is the main entry point for the self-improving agent that:
1. Uses Multi-Armed Bandit to select best strategies dynamically
2. Automatically retrains models when performance degrades
3. Exits positions when pre-game assumptions are invalidated
4. Tracks progress towards 10-12% daily return targets
5. Respects 20% max drawdown limit

Usage:
    # Paper trading with aggressive targets
    python scripts/run_self_improving_agent.py --mode paper --interval 30

    # Live trading (CAREFUL!)
    python scripts/run_self_improving_agent.py --mode live --interval 15

    # Backtest the self-improving system
    python scripts/run_self_improving_agent.py --mode backtest --start-date 2025-01-01
"""

import argparse
import asyncio
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.agent.live_game import LiveGameStateProvider
from footbe_trader.agent.telegram import NarrativeGenerator, TelegramNotifier
from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.execution.position_invalidator import PositionInvalidator
from footbe_trader.football.client import FootballApiClient
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.nba.client import NBAApiClient
from footbe_trader.reporting.daily_performance_tracker import DailyPerformanceTracker
from footbe_trader.self_improvement.model_lifecycle import ModelLifecycleManager
from footbe_trader.self_improvement.strategy_bandit import StrategyBandit
from footbe_trader.storage.database import Database
from footbe_trader.strategy.paper_trading import PaperTradingSimulator
from footbe_trader.strategy.trading_strategy import StrategyConfig

logger = get_logger(__name__)

_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global _shutdown_requested
    logger.info("shutdown_requested", signal=signum)
    _shutdown_requested = True


class SelfImprovingAgent:
    """Self-improving trading agent with aggressive targets."""

    def __init__(
        self,
        db: Database,
        config: any,
        kalshi_client: KalshiClient | None = None,
        mode: str = "paper",
        dry_run: bool = False,
        initial_bankroll: float = 1000.0,
    ):
        """Initialize self-improving agent.

        Args:
            db: Database connection.
            config: System configuration.
            kalshi_client: Kalshi API client.
            mode: Trading mode (paper/live).
            dry_run: If True, don't execute trades.
            initial_bankroll: Starting bankroll.
        """
        self.db = db
        self.config = config
        self.kalshi_client = kalshi_client
        self.mode = mode
        self.dry_run = dry_run

        # Initialize components

        # 1. Strategy Bandit (selects best strategy dynamically)
        self.strategy_bandit = StrategyBandit(db)
        self.strategy_bandit.load_state()

        # 2. Paper trading simulator
        self.simulator = PaperTradingSimulator(
            config=StrategyConfig(initial_bankroll=initial_bankroll),
            initial_bankroll=initial_bankroll,
        )

        # 3. Live game provider
        football_client = None
        nba_client = None
        if hasattr(config, "football_api") and config.football_api.api_key:
            football_client = FootballApiClient(config.football_api)
            nba_client = NBAApiClient(config.football_api)

        self.live_game_provider = None
        if football_client or nba_client:
            self.live_game_provider = LiveGameStateProvider(football_client, nba_client)

        # 4. Position invalidator (exits positions when game state changes)
        self.position_invalidator = PositionInvalidator(
            db=db,
            kalshi_client=kalshi_client,
            simulator=self.simulator,
            live_game_provider=self.live_game_provider,
            dry_run=dry_run,
        )

        # 5. Daily performance tracker
        self.performance_tracker = DailyPerformanceTracker(
            db=db,
            starting_bankroll=initial_bankroll,
        )

        # 6. Model lifecycle manager (automated retraining)
        self.model_lifecycle = ModelLifecycleManager(db, config)

        # 7. Telegram notifier
        self.telegram = None
        if config.telegram.is_configured and config.telegram.enabled:
            self.telegram = TelegramNotifier(config.telegram)

        self.narrative = NarrativeGenerator()

    async def run_loop(self, interval_minutes: int = 30):
        """Run continuous self-improving loop.

        Args:
            interval_minutes: Minutes between iterations.
        """
        logger.info(
            "self_improving_agent_started",
            mode=self.mode,
            interval=interval_minutes,
            bankroll=self.simulator.bankroll,
        )

        # Start background tasks
        asyncio.create_task(self.model_lifecycle.start_monitoring_loop(interval_hours=24))

        # Send startup message
        if self.telegram and self.mode == "live":
            await self.telegram.send_message(
                f"🚀 *Self-Improving Agent Started*\n"
                f"Mode: {self.mode}\n"
                f"Bankroll: ${self.simulator.bankroll:.2f}\n"
                f"Target: 10-12% daily\n"
                f"Max DD: 20%\n"
                f"🤖 Using Multi-Armed Bandit strategy selection\n"
                f"🔄 Automated model retraining enabled"
            )

        while not _shutdown_requested:
            try:
                await self._run_iteration()

                # Check drawdown limit
                if self.performance_tracker.is_drawdown_limit_breached():
                    logger.critical("drawdown_limit_breached", msg="STOPPING AGENT")
                    if self.telegram:
                        await self.telegram.send_message(
                            "🛑 *EMERGENCY STOP*\n"
                            "Drawdown limit breached (20%)\n"
                            "Agent has been halted."
                        )
                    break

                # Generate and send performance report
                await self._send_performance_update()

            except Exception as e:
                logger.error("iteration_error", error=str(e), exc_info=True)
                if self.telegram and self.mode == "live":
                    await self.telegram.send_error_alert(str(e))

            if not _shutdown_requested:
                await asyncio.sleep(interval_minutes * 60)

        logger.info("self_improving_agent_stopped")

    async def _run_iteration(self):
        """Run one iteration of the trading loop."""
        logger.info("iteration_started")

        # Step 1: Invalidate positions with changed game state
        invalidations = await self.position_invalidator.scan_and_invalidate_positions()
        logger.info("position_scan_complete", invalidations=len(invalidations))

        # Step 2: Get available fixtures
        fixtures = await self._get_tradeable_fixtures()
        logger.info("fixtures_loaded", count=len(fixtures))

        # Step 3: For each fixture, select strategy and evaluate
        decisions_made = 0
        orders_placed = 0

        for fixture in fixtures:
            if _shutdown_requested:
                break

            # Use Strategy Bandit to select best strategy
            strategy, strategy_name = self.strategy_bandit.select_strategy(fixture)

            # Evaluate fixture with selected strategy
            # (This would integrate with your existing evaluate_fixture logic)
            # For now, logging the selection
            logger.info(
                "strategy_selected_for_fixture",
                fixture_id=fixture.fixture_id,
                strategy=strategy_name,
            )

            decisions_made += 1

        # Step 4: Update performance tracking
        self._update_performance_metrics()

        # Step 5: Persist state
        self.strategy_bandit.persist_state()
        self.performance_tracker.persist_daily_snapshot()

        logger.info(
            "iteration_complete",
            decisions=decisions_made,
            orders=orders_placed,
            bankroll=self.simulator.bankroll,
        )

    async def _get_tradeable_fixtures(self):
        """Get fixtures available for trading."""
        # Similar to existing logic in run_agent.py
        # Would query database for upcoming fixtures with market mappings
        return []

    def _update_performance_metrics(self):
        """Update daily performance tracking."""
        # Calculate metrics
        realized_pnl = self.simulator.total_realized_pnl
        unrealized_pnl = sum(p.unrealized_pnl for p in self.simulator.positions.values())
        current_exposure = self.simulator.total_exposure

        # Count positions
        positions_opened_today = 0  # Would track this
        positions_closed_today = 0
        positions_pending = len([p for p in self.simulator.positions.values() if p.is_open])

        # Update tracker
        self.performance_tracker.update_daily_progress(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            current_exposure=current_exposure,
            positions_opened=positions_opened_today,
            positions_closed=positions_closed_today,
            positions_pending=positions_pending,
        )

    async def _send_performance_update(self):
        """Send performance update via Telegram."""
        if not self.telegram or self.mode != "live":
            return

        # Generate report
        report = self.performance_tracker.generate_daily_report()

        # Generate alert if needed
        alert = self.performance_tracker.generate_pace_alert()

        # Format message
        message = (
            f"📊 *Daily Performance Update*\n\n"
            f"💰 *Bankroll*: ${report['current_bankroll']:.2f}\n"
            f"📈 *Today's P&L*: ${report['today']['total_pnl']:.2f} ({report['today']['return_pct']:.1%})\n"
            f"🎯 *Target Progress*: {report['today']['completion_pct']:.0%}\n"
            f"📊 *Status*: {report['today']['status']}\n\n"
            f"📅 *Week*: {report['weekly']['week_to_date_pnl']:.2f} "
            f"({'✅ On pace' if report['weekly']['on_pace'] else '⚠️ Behind pace'})\n\n"
            f"⚠️ *Risk*:\n"
            f"  Drawdown: {report['risk']['drawdown_pct']}\n"
            f"  Remaining capacity: ${report['risk']['remaining_capacity']:.2f}\n\n"
            f"📦 *Positions*:\n"
            f"  Open: {report['positions']['pending_settlement']}\n"
            f"  Exposure: ${report['positions']['current_exposure']:.2f}\n"
        )

        if alert:
            message = f"{message}\n\n{alert}"

        await self.telegram.send_message(message)

        # Also send strategy performance
        bandit_report = self.strategy_bandit.get_performance_report()
        strategy_msg = "🤖 *Strategy Performance*\n\n"
        for name, perf in bandit_report["strategies"].items():
            strategy_msg += (
                f"*{name}*: "
                f"Win rate: {perf['win_rate']:.1%}, "
                f"Sharpe: {perf['sharpe_ratio']:.2f}, "
                f"Trades: {perf['total_trades']}\n"
            )

        await self.telegram.send_message(strategy_msg)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Self-Improving Trading Agent")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Minutes between iterations",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute trades",
    )

    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration
    config = load_config(args.config)

    # Initialize database
    db = Database(config.database.path)
    db.connect()
    db.migrate()

    # Add new tables for self-improvement
    cursor = db.connection.cursor()

    # Model versions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            training_window_days INTEGER,
            training_samples INTEGER,
            hyperparameters TEXT,
            validation_accuracy REAL,
            validation_log_loss REAL,
            validation_sharpe REAL,
            status TEXT,
            deployed_at TEXT,
            retired_at TEXT,
            artifact_path TEXT
        )
    """)

    # Strategy bandit state table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_bandit_state (
            strategy_name TEXT PRIMARY KEY,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0.0,
            times_selected INTEGER DEFAULT 0,
            last_updated TEXT
        )
    """)

    # Daily performance snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_performance_snapshots (
            date TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            current_bankroll REAL NOT NULL,
            total_return_pct REAL NOT NULL,
            daily_target_met INTEGER,
            weekly_on_pace INTEGER,
            current_drawdown REAL,
            report_json TEXT
        )
    """)

    db.connection.commit()

    # Initialize Kalshi client
    kalshi_client = None
    if config.kalshi.api_key_id and config.kalshi.private_key_path:
        kalshi_client = KalshiClient(config.kalshi)

    # Create and run agent
    agent = SelfImprovingAgent(
        db=db,
        config=config,
        kalshi_client=kalshi_client,
        mode=args.mode,
        dry_run=args.dry_run,
        initial_bankroll=args.bankroll,
    )

    try:
        # Enter async context managers for clients
        clients = []
        if kalshi_client:
            clients.append(kalshi_client)
        if agent.live_game_provider and agent.live_game_provider.football_client:
            clients.append(agent.live_game_provider.football_client)
        if agent.live_game_provider and agent.live_game_provider.nba_client:
            clients.append(agent.live_game_provider.nba_client)

        try:
            for client in clients:
                await client.__aenter__()

            await agent.run_loop(args.interval)

        finally:
            for client in reversed(clients):
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing client: {e}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
