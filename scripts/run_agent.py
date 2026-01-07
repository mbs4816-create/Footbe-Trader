#!/usr/bin/env python3
"""Trading Agent CLI.

Run the trading agent in paper or live mode to evaluate fixtures,
generate trading signals, and execute trades.

Usage:
    python scripts/run_agent.py --mode paper --interval 30
    python scripts/run_agent.py --mode paper --once
    python scripts/run_agent.py --mode live --interval 60 --dry-run
"""

import argparse
import asyncio
import json
import numpy as np
import pickle
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def json_serialize(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.kalshi.interfaces import MarketData, OrderbookData, OrderbookLevel, OrderData
from footbe_trader.modeling.interfaces import PredictionResult
from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.poisson_model import PoissonModel
from footbe_trader.storage.database import Database
from footbe_trader.strategy.decision_record import (
    AgentRunSummary,
    DecisionRecord,
    ModelPrediction,
)
from footbe_trader.strategy.mapping import (
    FixtureMarketMapping,
    FixtureMarketMapper,
    MappingConfig,
)
from footbe_trader.strategy.paper_trading import (
    PaperFill,
    PaperOrder,
    PaperPosition,
    PaperTradingSimulator,
    PnlSnapshot,
)
from footbe_trader.strategy.trading_strategy import (
    EdgeStrategy,
    FixtureContext,
    OutcomeContext,
    StaleOrderDetector,
    StrategyConfig,
    create_agent_run_summary,
)
from footbe_trader.self_improvement.strategy_bandit import StrategyBandit
from footbe_trader.self_improvement.model_lifecycle import ModelLifecycleManager
from footbe_trader.execution.position_invalidator import PositionInvalidator
from footbe_trader.reporting.daily_performance_tracker import DailyPerformanceTracker
from footbe_trader.agent.live_game import (
    GamePhase,
    LiveGameState,
    LiveGameStateProvider,
)
from footbe_trader.agent.telegram import (
    NarrativeGenerator,
    TelegramNotifier,
)

logger = get_logger(__name__)

# Global shutdown flag
_shutdown_requested = False


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    global _shutdown_requested
    logger.info("shutdown_requested", signal=signum)
    _shutdown_requested = True


class TradingAgent:
    """Trading agent that runs the strategy loop."""

    def __init__(
        self,
        db: Database,
        kalshi_client: KalshiClient | None,
        strategy: EdgeStrategy,
        simulator: PaperTradingSimulator,
        mode: str = "paper",
        dry_run: bool = False,
        live_game_provider: LiveGameStateProvider | None = None,
        telegram_notifier: TelegramNotifier | None = None,
        narrative_generator: NarrativeGenerator | None = None,
        use_bandit: bool = False,
        starting_bankroll: float = 1000.0,
        config: Any = None,
        poisson_model: PoissonModel | None = None,
    ):
        """Initialize trading agent.

        Args:
            db: Database connection.
            kalshi_client: Kalshi API client (None for testing).
            strategy: Trading strategy (used as fallback if bandit disabled).
            simulator: Paper trading simulator.
            mode: "paper" or "live".
            dry_run: If True, don't execute trades even in live mode.
            live_game_provider: Provider for live game state (optional).
            telegram_notifier: Telegram notification sender (optional).
            narrative_generator: Generator for human-readable narratives (optional).
            use_bandit: If True, use StrategyBandit for multi-strategy selection.
            starting_bankroll: Starting bankroll for performance tracking.
            config: Main config object for lifecycle manager.
            poisson_model: Trained Poisson model for predictions (optional).
        """
        self.db = db
        self.kalshi_client = kalshi_client
        self.strategy = strategy
        self.simulator = simulator
        self.mode = mode
        self.dry_run = dry_run
        self._current_run: AgentRunSummary | None = None
        self.live_game_provider = live_game_provider
        self.telegram = telegram_notifier
        self.narrative = narrative_generator or NarrativeGenerator()
        self.poisson_model = poisson_model

        # Cache for live game states during a run
        self._live_game_states: dict[int, LiveGameState] = {}

        # Self-improvement components
        self.use_bandit = use_bandit
        self.strategy_bandit = StrategyBandit(db) if use_bandit else None
        self.position_invalidator = PositionInvalidator(
            db=db,
            kalshi_client=kalshi_client,
            simulator=simulator,
            live_game_provider=live_game_provider,
            dry_run=dry_run,
        ) if live_game_provider else None
        self.performance_tracker = DailyPerformanceTracker(db, starting_bankroll)
        self.model_lifecycle = ModelLifecycleManager(db, config) if config else None

    async def run_once(self) -> AgentRunSummary:
        """Run a single iteration of the trading loop.

        Returns:
            Run summary with statistics.
        """
        # Create run record
        run_summary = create_agent_run_summary(
            run_type=self.mode,
            config=self.strategy.config,
        )

        # Store run in database
        run_id = self._create_agent_run(run_summary)
        run_summary.run_id = run_id
        self.strategy.set_run_id(run_id)
        self.simulator.set_run_id(run_id)
        self._current_run = run_summary

        logger.info(
            "agent_run_started",
            run_id=run_id,
            mode=self.mode,
            dry_run=self.dry_run,
        )

        try:
            # Clear live game state cache for fresh data
            self._live_game_states.clear()
            if self.live_game_provider:
                self.live_game_provider.clear_cache()
            
            # Step 1: Get mapped fixtures
            fixtures = await self._get_mapped_fixtures()
            run_summary.fixtures_evaluated = len(fixtures)

            logger.info("fixtures_loaded", count=len(fixtures))

            # Step 2: Process each fixture
            for fixture_ctx in fixtures:
                if _shutdown_requested:
                    break

                await self._process_fixture(fixture_ctx, run_summary)

            # Step 2.5: Cancel stale resting orders (in-game safeguard)
            if not self.dry_run and self.kalshi_client is not None:
                stale_cancelled = await self._cancel_stale_orders()
                if stale_cancelled > 0:
                    logger.info("stale_orders_cancelled", count=stale_cancelled)

            # Step 3: Check exit conditions for open positions
            await self._check_exits(run_summary)

            # Step 4: Take P&L snapshot
            pnl_snapshot = self.simulator.take_pnl_snapshot()
            self._store_pnl_snapshot(pnl_snapshot)

            # Step 4.5: Update bandit with settled position outcomes
            self._update_bandit_outcomes()

            # Step 5: Update summary
            # In LIVE mode, fetch real portfolio data from Kalshi
            # In PAPER mode, use simulator data
            if self.mode == "live" and self.kalshi_client:
                await self._update_summary_from_kalshi(run_summary)
            else:
                self.simulator.update_run_summary(run_summary)

            run_summary.status = "completed"
            run_summary.completed_at = utc_now()

            # Decision records are stored inline during _process_fixture
            # to satisfy foreign key constraints with paper_orders

            # Update run in database
            self._complete_agent_run(run_summary)

            # Step 6: Track daily performance (self-improvement)
            if self.performance_tracker:
                report = self.performance_tracker.generate_daily_report()
                pace_alert = self.performance_tracker.generate_pace_alert()

                # Log performance tracking
                logger.info(
                    "daily_performance",
                    current_bankroll=report["current_bankroll"],
                    target_met=report["today"]["status"],
                    completion_pct=f"{report['today']['completion_pct']:.1%}",
                )

                if pace_alert and self.telegram:
                    await self.telegram.send_message(pace_alert)

            logger.info(
                "agent_run_completed",
                run_id=run_id,
                fixtures=run_summary.fixtures_evaluated,
                markets=run_summary.markets_evaluated,
                decisions=run_summary.decisions_made,
                orders_placed=run_summary.orders_placed,
                orders_filled=run_summary.orders_filled,
                realized_pnl=run_summary.total_realized_pnl,
                unrealized_pnl=run_summary.total_unrealized_pnl,
            )

        except Exception as e:
            run_summary.status = "failed"
            run_summary.error_message = str(e)
            run_summary.error_count += 1
            run_summary.completed_at = utc_now()
            self._complete_agent_run(run_summary)
            logger.error(
                "agent_run_failed",
                run_id=run_id,
                error=str(e),
                exc_info=True,
            )
            raise

        return run_summary

    def _get_market_data_from_db(
        self, ticker: str
    ) -> tuple[MarketData | None, OrderbookData | None]:
        """Fetch market data from database for paper trading.
        
        Args:
            ticker: Market ticker.
            
        Returns:
            Tuple of (MarketData, OrderbookData) or (None, None).
        """
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            SELECT 
                ticker, event_ticker, title, subtitle, status,
                close_time, yes_bid, yes_ask, no_bid, no_ask, 
                last_price, volume
            FROM kalshi_markets
            WHERE ticker = ?
            """,
            (ticker,),
        )
        row = cursor.fetchone()
        
        if not row:
            return None, None
        
        # Parse close_time string to datetime
        close_time = None
        if row["close_time"]:
            try:
                close_time_str = row["close_time"]
                if close_time_str.endswith("Z"):
                    close_time_str = close_time_str[:-1] + "+00:00"
                close_time = datetime.fromisoformat(close_time_str)
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=UTC)
            except ValueError:
                pass
        
        market_data = MarketData(
            ticker=row["ticker"],
            event_ticker=row["event_ticker"],
            title=row["title"],
            subtitle=row["subtitle"] or "",
            status=row["status"],
            close_time=close_time,
            yes_bid=row["yes_bid"] or 0,
            yes_ask=row["yes_ask"] or 0,
            volume=row["volume"] or 0,
            open_interest=0,
        )
        
        # Create synthetic orderbook from bid/ask using OrderbookLevel
        yes_bid = row["yes_bid"] or 0
        yes_ask = row["yes_ask"] or 0
        no_bid = row["no_bid"] or 0
        no_ask = row["no_ask"] or 0
        
        orderbook = OrderbookData(
            ticker=row["ticker"],
            yes_bids=[OrderbookLevel(price=yes_bid, quantity=1000)] if yes_bid > 0 else [],
            yes_asks=[OrderbookLevel(price=yes_ask, quantity=1000)] if yes_ask > 0 else [],
            no_bids=[OrderbookLevel(price=no_bid, quantity=1000)] if no_bid > 0 else [],
            no_asks=[OrderbookLevel(price=no_ask, quantity=1000)] if no_ask > 0 else [],
        )
        
        return market_data, orderbook

    async def run_loop(self, interval_minutes: int) -> None:
        """Run the trading loop continuously.

        Args:
            interval_minutes: Minutes between iterations.
        """
        logger.info(
            "agent_loop_started",
            mode=self.mode,
            interval_minutes=interval_minutes,
        )

        # Send startup notification (live mode only)
        if self.telegram and self.mode == "live":
            await self.telegram.send_message(
                f"🚀 *Live Agent Started*\n"
                f"Interval: {interval_minutes} minutes\n"
                f"Dry Run: {self.dry_run}\n"
                f"Trading real money on Kalshi"
            )

        while not _shutdown_requested:
            try:
                summary = await self.run_once()

                # Log summary
                print(f"\n{'=' * 60}")
                print(f"Run {summary.run_id} completed at {summary.completed_at}")
                print(f"  Fixtures evaluated: {summary.fixtures_evaluated}")
                print(f"  Markets evaluated: {summary.markets_evaluated}")
                print(f"  Decisions made: {summary.decisions_made}")
                print(f"  Orders placed: {summary.orders_placed}")
                print(f"  Orders filled: {summary.orders_filled}")
                print(f"  Realized P&L: ${summary.total_realized_pnl:.2f}")
                print(f"  Unrealized P&L: ${summary.total_unrealized_pnl:.2f}")
                print(f"  Open positions: {summary.position_count}")
                print(f"  Exposure: ${summary.total_exposure:.2f}")
                print(f"{'=' * 60}\n")

                # Send Telegram notification with narrative (live mode only)
                if self.telegram and self.mode == "live":
                    # Generate human-readable narrative
                    narrative = self.narrative.generate_run_narrative(
                        fixtures_evaluated=summary.fixtures_evaluated,
                        markets_evaluated=summary.markets_evaluated,
                        decisions_made=summary.decisions_made,
                        orders_placed=summary.orders_placed,
                        orders_filled=summary.orders_filled,
                        skipped_reasons={},  # TODO: Track skip reasons
                        trades_by_outcome={},  # TODO: Track trade outcomes
                        live_games=[],  # TODO: Track live games
                        cancelled_stale=0,  # TODO: Track cancelled orders
                        edge_summary={},  # TODO: Track edge stats
                    )
                    
                    # Get position narrative
                    positions_data = [
                        {
                            "ticker": p.ticker,
                            "unrealized_pnl": p.unrealized_pnl,
                            "quantity": p.quantity,
                        }
                        for p in self.simulator.positions.values()
                    ]
                    position_narrative = self.narrative.generate_position_narrative(
                        positions=positions_data,
                        total_pnl=summary.total_unrealized_pnl,
                    )
                    
                    full_narrative = f"{narrative}\n\n{position_narrative}"
                    
                    await self.telegram.send_run_summary(
                        run_id=summary.run_id or 0,
                        mode=self.mode,
                        fixtures_evaluated=summary.fixtures_evaluated,
                        markets_evaluated=summary.markets_evaluated,
                        decisions_made=summary.decisions_made,
                        orders_placed=summary.orders_placed,
                        orders_filled=summary.orders_filled,
                        realized_pnl=summary.total_realized_pnl,
                        unrealized_pnl=summary.total_unrealized_pnl,
                        total_exposure=summary.total_exposure,
                        position_count=summary.position_count,
                        narrative=full_narrative,
                    )

            except Exception as e:
                logger.error("run_loop_error", error=str(e), exc_info=True)
                # Send error notification (live mode only - paper errors are silent)
                if self.telegram and self.mode == "live":
                    await self.telegram.send_error_alert(
                        error_message=str(e),
                    )
                # Continue loop despite errors

            # Sleep until next iteration
            if not _shutdown_requested:
                logger.info("sleeping", minutes=interval_minutes)
                await asyncio.sleep(interval_minutes * 60)

        # Send shutdown notification (live mode only)
        if self.telegram and self.mode == "live":
            await self.telegram.send_message("🛑 *Live Agent Stopped*")
        
        logger.info("agent_loop_stopped")

    async def _get_mapped_fixtures(self) -> list[FixtureContext]:
        """Get fixtures with valid market mappings (soccer and NBA)."""
        fixtures: list[FixtureContext] = []

        # Query soccer fixtures with mappings from database
        cursor = self.db.connection.cursor()

        # Get soccer fixtures in the next 7 days with active mappings
        now = utc_now()
        cutoff = now + timedelta(days=7)

        cursor.execute(
            """
            SELECT
                f.fixture_id,
                f.home_team_id,
                f.away_team_id,
                f.kickoff_utc,
                f.league_id,
                m.mapping_version,
                m.structure_type,
                m.ticker_home_win,
                m.ticker_draw,
                m.ticker_away_win,
                m.event_ticker,
                m.confidence_score,
                th.name as home_team_name,
                ta.name as away_team_name,
                'soccer' as sport_type
            FROM fixtures_v2 f
            JOIN fixture_market_map m ON f.fixture_id = m.fixture_id
            LEFT JOIN teams th ON f.home_team_id = th.team_id
            LEFT JOIN teams ta ON f.away_team_id = ta.team_id
            WHERE f.kickoff_utc >= ?
            AND f.kickoff_utc <= ?
            AND m.status = 'AUTO'
            AND m.confidence_score >= 0.70
            ORDER BY f.kickoff_utc ASC
            """,
            (now.isoformat(), cutoff.isoformat()),
        )

        for row in cursor.fetchall():
            mapping = FixtureMarketMapping(
                fixture_id=row["fixture_id"],
                mapping_version=row["mapping_version"],
                home_team_id=row["home_team_id"],
                away_team_id=row["away_team_id"],
                structure_type=row["structure_type"],
                ticker_home_win=row["ticker_home_win"],
                ticker_draw=row["ticker_draw"],
                ticker_away_win=row["ticker_away_win"],
                event_ticker=row["event_ticker"],
                confidence_score=row["confidence_score"],
            )

            kickoff = (
                datetime.fromisoformat(row["kickoff_utc"])
                if row["kickoff_utc"]
                else now
            )

            fixture_ctx = FixtureContext(
                fixture_id=row["fixture_id"],
                home_team=row["home_team_name"] or str(row["home_team_id"]),
                away_team=row["away_team_name"] or str(row["away_team_id"]),
                kickoff_time=kickoff,
                league=str(row["league_id"]),
                mapping=mapping,
                current_exposure=self.simulator.get_fixture_exposure(row["fixture_id"]),
            )
            fixtures.append(fixture_ctx)

        # Get NBA games with mappings
        nba_fixtures = await self._get_mapped_nba_games()
        fixtures.extend(nba_fixtures)

        return fixtures

    async def _get_mapped_nba_games(self) -> list[FixtureContext]:
        """Get NBA games with valid market mappings."""
        fixtures: list[FixtureContext] = []
        cursor = self.db.connection.cursor()
        
        now = utc_now()
        cutoff = now + timedelta(days=7)
        
        cursor.execute(
            """
            SELECT
                g.game_id,
                g.home_team_id,
                g.away_team_id,
                g.date_utc,
                'NBA' as league,
                m.mapping_version,
                'MONEYLINE' as structure_type,
                m.ticker_home_win,
                NULL as ticker_draw,
                m.ticker_away_win,
                m.event_ticker,
                m.confidence_score,
                h.nickname as home_team_name,
                a.nickname as away_team_name,
                'basketball' as sport_type
            FROM nba_games g
            JOIN nba_game_market_map m ON g.game_id = m.game_id
            LEFT JOIN nba_teams h ON g.home_team_id = h.team_id
            LEFT JOIN nba_teams a ON g.away_team_id = a.team_id
            WHERE g.date_utc >= ?
            AND g.date_utc <= ?
            AND g.status = 1  -- NOT_STARTED
            AND m.status = 'AUTO'
            AND m.confidence_score >= 0.50
            ORDER BY g.date_utc ASC
            """,
            (now.isoformat(), cutoff.isoformat()),
        )
        
        for row in cursor.fetchall():
            # Use game_id but prefix with NBA to avoid ID collisions with soccer
            game_id = row["game_id"]
            fixture_id = 1000000000 + game_id  # NBA games have IDs > 1 billion
            
            mapping = FixtureMarketMapping(
                fixture_id=fixture_id,
                mapping_version=row["mapping_version"],
                structure_type=row["structure_type"],
                ticker_home_win=row["ticker_home_win"],
                ticker_draw=None,  # No draw in basketball
                ticker_away_win=row["ticker_away_win"],
                event_ticker=row["event_ticker"],
                confidence_score=row["confidence_score"],
            )
            
            kickoff = (
                datetime.fromisoformat(row["date_utc"])
                if row["date_utc"]
                else now
            )
            
            fixture_ctx = FixtureContext(
                fixture_id=fixture_id,
                home_team=row["home_team_name"] or str(row["home_team_id"]),
                away_team=row["away_team_name"] or str(row["away_team_id"]),
                kickoff_time=kickoff,
                league="NBA",
                mapping=mapping,
                current_exposure=self.simulator.get_fixture_exposure(fixture_id),
            )
            fixtures.append(fixture_ctx)
        
        return fixtures

    async def _process_fixture(
        self,
        fixture: FixtureContext,
        summary: AgentRunSummary,
    ) -> None:
        """Process a single fixture."""
        logger.info(
            "processing_fixture",
            fixture_id=fixture.fixture_id,
            match=f"{fixture.home_team} vs {fixture.away_team}",
        )
        
        # Check live game state for in-game awareness
        live_state = await self._get_live_game_state(fixture)
        
        if live_state:
            # Log current game state
            if live_state.is_live:
                score_str = "N/A"
                if live_state.score:
                    score_str = f"{live_state.score.home_score}-{live_state.score.away_score}"
                logger.info(
                    "fixture_is_live",
                    fixture_id=fixture.fixture_id,
                    phase=live_state.phase.value,
                    score=score_str,
                    home_adj=f"{live_state.home_win_adjustment:+.2%}",
                    away_adj=f"{live_state.away_win_adjustment:+.2%}",
                )
            
            # Skip if game is finished or not tradeable
            if not live_state.is_tradeable:
                logger.info(
                    "skipping_untradeable_fixture",
                    fixture_id=fixture.fixture_id,
                    reason=live_state.stale_reason,
                )
                return

        # Build outcome contexts
        outcomes = await self._build_outcome_contexts(fixture)
        summary.markets_evaluated += len(outcomes)

        if not outcomes:
            logger.debug("no_valid_outcomes", fixture_id=fixture.fixture_id)
            return

        # Get model predictions (placeholder - would integrate with actual model)
        model_prediction = await self._get_model_prediction(fixture)

        # Evaluate fixture with strategy (bandit or single strategy)
        if self.use_bandit and self.strategy_bandit:
            # Use multi-armed bandit to select best strategy
            selected_strategy, strategy_name = self.strategy_bandit.select_strategy(fixture)
            logger.info(
                "strategy_selected",
                fixture_id=fixture.fixture_id,
                strategy=strategy_name,
            )
            decisions = selected_strategy.evaluate_fixture(
                fixture=fixture,
                outcomes=outcomes,
                model_prediction=model_prediction,
                global_exposure=self.simulator.total_exposure,
                bankroll=self.simulator.bankroll,
            )
            # Store strategy name for later outcome tracking
            for decision in decisions:
                decision.strategy_name = strategy_name
        else:
            # Use single strategy
            decisions = self.strategy.evaluate_fixture(
                fixture=fixture,
                outcomes=outcomes,
                model_prediction=model_prediction,
                global_exposure=self.simulator.total_exposure,
                bankroll=self.simulator.bankroll,
            )

        summary.decisions_made += len(decisions)

        # Execute decisions
        for decision in decisions:
            # Store decision record first (before order, due to FK constraint)
            self._store_decision_record(decision)
            
            if decision.order_params is not None:
                if not self.dry_run and self.kalshi_client is not None:
                    # Live trading: place real order via Kalshi API
                    order_data = await self._execute_live_order(decision, summary)
                    if order_data:
                        decision.order_placed = True
                        decision.order_id = order_data.order_id
                        self._update_decision_with_order(decision)
                        
                        # Also update simulator for position tracking
                        orderbook = None
                        for outcome in outcomes:
                            if outcome.ticker == decision.market_ticker:
                                orderbook = outcome.orderbook
                                break
                        self.simulator.execute_decision(decision, orderbook)
                else:
                    # Paper trading: simulate order locally
                    orderbook = None
                    for outcome in outcomes:
                        if outcome.ticker == decision.market_ticker:
                            orderbook = outcome.orderbook
                            break

                    order = self.simulator.execute_decision(decision, orderbook)

                    if order:
                        decision.order_placed = True
                        decision.order_id = order.order_id
                        
                        # Update the decision record with order info
                        self._update_decision_with_order(decision)

                        # Store order in database
                        self._store_paper_order(order)

                        # Store fills
                        for fill in self.simulator.fills:
                            if fill.order_id == order.order_id:
                                self._store_paper_fill(fill)
        
        # Clear decision records since we stored them inline
        self.strategy.clear_decision_records()

    async def _build_outcome_contexts(
        self,
        fixture: FixtureContext,
    ) -> list[OutcomeContext]:
        """Build outcome contexts for a fixture."""
        outcomes: list[OutcomeContext] = []
        mapping = fixture.mapping

        # Define outcome -> ticker mapping
        outcome_tickers = [
            ("home_win", mapping.ticker_home_win),
            ("draw", mapping.ticker_draw),
            ("away_win", mapping.ticker_away_win),
        ]

        for outcome_type, ticker in outcome_tickers:
            if not ticker:
                continue

            market_data: MarketData | None = None
            orderbook: OrderbookData | None = None

            if self.kalshi_client:
                try:
                    market_data = await self.kalshi_client.get_market(ticker)
                    orderbook = await self.kalshi_client.get_orderbook(ticker)
                except Exception as e:
                    logger.warning(
                        "failed_to_fetch_market",
                        ticker=ticker,
                        error=str(e),
                    )
                    continue
            else:
                # Paper trading without live client - fetch from database
                market_data, orderbook = self._get_market_data_from_db(ticker)

            # Get existing position
            position = self.simulator.get_position(ticker)

            outcome_ctx = OutcomeContext(
                outcome=outcome_type,
                ticker=ticker,
                market_data=market_data,
                orderbook=orderbook,
                current_position=position.quantity if position else 0,
                average_entry_price=position.average_entry_price if position else 0.0,
            )
            outcomes.append(outcome_ctx)

        return outcomes

    async def _get_model_prediction(
        self,
        fixture: FixtureContext,
    ) -> ModelPrediction:
        """Get model prediction for a fixture.

        This integrates with live game state to adjust pre-match predictions
        based on current score and time remaining.
        """
        # Get base predictions from Poisson model or fallback to priors
        if self.poisson_model and self.poisson_model.is_trained and fixture.league != "NBA":
            # Use trained Poisson model for soccer leagues
            feature = MatchFeatureVector(
                fixture_id=fixture.fixture_id,
                kickoff_utc=fixture.kickoff_time,
                home_team_id=fixture.mapping.home_team_id,
                away_team_id=fixture.mapping.away_team_id,
                season=2024,  # TODO: extract from fixture
                round_str="",
            )

            try:
                params = self.poisson_model.predict_params(feature)
                base_home = params.prob_home
                base_draw = params.prob_draw
                base_away = params.prob_away
                model_name = "poisson_v1"

                logger.debug(
                    "poisson_prediction",
                    fixture_id=fixture.fixture_id,
                    home=fixture.home_team,
                    away=fixture.away_team,
                    prob_home=base_home,
                    prob_draw=base_draw,
                    prob_away=base_away,
                    lambda_home=params.lambda_home,
                    lambda_away=params.lambda_away,
                )
            except Exception as e:
                logger.warning(
                    "poisson_prediction_failed",
                    fixture_id=fixture.fixture_id,
                    error=str(e),
                )
                # Fallback to prior
                base_home = 0.45
                base_draw = 0.25
                base_away = 0.30
                model_name = "home_advantage_prior"
        elif fixture.league == "NBA":
            base_home = 0.58
            base_draw = 0.0  # No draws in basketball
            base_away = 0.42
            model_name = "nba_home_advantage_prior"
        else:
            base_home = 0.45
            base_draw = 0.25
            base_away = 0.30
            model_name = "home_advantage_prior"
        
        # Check for live game state adjustments
        live_state = await self._get_live_game_state(fixture)
        
        if live_state and live_state.is_live:
            # Apply live game adjustments to base probabilities
            adj_home = base_home + live_state.home_win_adjustment
            adj_draw = base_draw + live_state.draw_adjustment
            adj_away = base_away + live_state.away_win_adjustment
            
            # Normalize to ensure probabilities sum to 1.0
            total = adj_home + adj_draw + adj_away
            if total > 0:
                adj_home /= total
                adj_draw /= total
                adj_away /= total
            
            # Clamp to valid probability range
            adj_home = max(0.01, min(0.99, adj_home))
            adj_draw = max(0.0, min(0.99, adj_draw))
            adj_away = max(0.01, min(0.99, adj_away))
            
            logger.info(
                "live_adjusted_prediction",
                fixture_id=fixture.fixture_id,
                phase=live_state.phase.value,
                score=f"{live_state.score.home_score}-{live_state.score.away_score}" if live_state.score else "N/A",
                base_probs=f"H:{base_home:.2f} D:{base_draw:.2f} A:{base_away:.2f}",
                adj_probs=f"H:{adj_home:.2f} D:{adj_draw:.2f} A:{adj_away:.2f}",
            )
            
            return ModelPrediction(
                model_name=f"{model_name}_live_adjusted",
                model_version="1.0.0",
                prob_home_win=adj_home,
                prob_draw=adj_draw,
                prob_away_win=adj_away,
                confidence=0.75,  # Higher confidence with live data
            )
        
        # Pre-match: return base predictions
        return ModelPrediction(
            model_name=model_name,
            model_version="1.0.0",
            prob_home_win=base_home,
            prob_draw=base_draw,
            prob_away_win=base_away,
            confidence=0.6 if fixture.league == "NBA" else 0.7,
        )
    
    async def _get_live_game_state(
        self,
        fixture: FixtureContext,
    ) -> LiveGameState | None:
        """Get live game state for a fixture.
        
        Caches results during a run to reduce API calls.
        """
        # Check cache first
        if fixture.fixture_id in self._live_game_states:
            return self._live_game_states[fixture.fixture_id]
        
        if not self.live_game_provider:
            return None
        
        try:
            if fixture.league == "NBA":
                # For NBA, we need the game_id which might be stored differently
                # For now, use fixture_id as game_id (needs proper mapping)
                state = await self.live_game_provider.get_nba_game_state(fixture.fixture_id)
            else:
                state = await self.live_game_provider.get_football_game_state(fixture.fixture_id)
            
            self._live_game_states[fixture.fixture_id] = state
            return state
            
        except Exception as e:
            logger.warning(
                "live_game_state_fetch_failed",
                fixture_id=fixture.fixture_id,
                error=str(e),
            )
            return None

    async def _check_exits(self, summary: AgentRunSummary) -> None:
        """Check exit conditions for open positions."""
        open_positions = self.simulator.open_positions

        # First, check for position invalidations (stale/adverse positions)
        if self.position_invalidator:
            invalidations = await self.position_invalidator.scan_and_invalidate_positions()
            # Track invalidations count (position invalidator handles execution internally)
            if invalidations:
                summary.decisions_made += len(invalidations)
                logger.info("positions_invalidated", count=len(invalidations))

        # Then, check normal exit conditions
        for ticker, position in open_positions.items():
            if not position.is_open:
                continue

            # Get current market data
            market_data: MarketData | None = None
            orderbook: OrderbookData | None = None

            if self.kalshi_client:
                try:
                    market_data = await self.kalshi_client.get_market(ticker)
                    orderbook = await self.kalshi_client.get_orderbook(ticker)
                except Exception as e:
                    logger.warning(
                        "failed_to_fetch_market_for_exit",
                        ticker=ticker,
                        error=str(e),
                    )
                    continue

            # Mark position to market
            if orderbook and orderbook.mid_price:
                self.simulator.mark_to_market(ticker, orderbook.mid_price)

            # Get fixture context
            fixture_id = position.fixture_id or 0
            fixture_ctx = await self._get_fixture_by_id(fixture_id)
            if not fixture_ctx:
                continue

            # Build outcome context
            outcome_ctx = OutcomeContext(
                outcome=position.outcome,
                ticker=ticker,
                market_data=market_data,
                orderbook=orderbook,
                current_position=position.quantity,
                average_entry_price=position.average_entry_price,
            )

            # Get updated model prediction
            model_prediction = await self._get_model_prediction(fixture_ctx)

            # Evaluate exit
            decision = self.strategy.evaluate_exit(
                outcome=outcome_ctx,
                model_prediction=model_prediction,
                entry_price=position.average_entry_price,
                fixture_id=fixture_id,
            )

            summary.decisions_made += 1

            # Store decision record first (before order, due to FK constraint)
            self._store_decision_record(decision)

            # Execute exit if needed
            if decision.order_params is not None:
                if not self.dry_run and self.kalshi_client is not None:
                    # Live trading: place real order via Kalshi API
                    order_data = await self._execute_live_order(decision, summary)
                    if order_data:
                        decision.order_placed = True
                        decision.order_id = order_data.order_id
                        self._update_decision_with_order(decision)
                        self.simulator.execute_decision(decision, orderbook)
                else:
                    # Paper trading
                    order = self.simulator.execute_decision(decision, orderbook)

                    if order:
                        decision.order_placed = True
                        decision.order_id = order.order_id
                        self._update_decision_with_order(decision)
                        self._store_paper_order(order)

    async def _get_fixture_by_id(self, fixture_id: int) -> FixtureContext | None:
        """Get fixture context by ID."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            SELECT
                f.fixture_id,
                f.home_team_id,
                f.away_team_id,
                f.kickoff_utc,
                f.league_id,
                m.mapping_version,
                m.structure_type,
                m.ticker_home_win,
                m.ticker_draw,
                m.ticker_away_win,
                m.event_ticker,
                m.confidence_score,
                th.name as home_team_name,
                ta.name as away_team_name
            FROM fixtures_v2 f
            JOIN fixture_market_map m ON f.fixture_id = m.fixture_id
            LEFT JOIN teams th ON f.home_team_id = th.team_id
            LEFT JOIN teams ta ON f.away_team_id = ta.team_id
            WHERE f.fixture_id = ?
            LIMIT 1
            """,
            (fixture_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        mapping = FixtureMarketMapping(
            fixture_id=row["fixture_id"],
            mapping_version=row["mapping_version"],
            structure_type=row["structure_type"],
            ticker_home_win=row["ticker_home_win"],
            ticker_draw=row["ticker_draw"],
            ticker_away_win=row["ticker_away_win"],
            event_ticker=row["event_ticker"],
            confidence_score=row["confidence_score"],
        )

        kickoff = (
            datetime.fromisoformat(row["kickoff_utc"])
            if row["kickoff_utc"]
            else utc_now()
        )

        return FixtureContext(
            fixture_id=row["fixture_id"],
            home_team=row["home_team_name"] or str(row["home_team_id"]),
            away_team=row["away_team_name"] or str(row["away_team_id"]),
            kickoff_time=kickoff,
            league=str(row["league_id"]),
            mapping=mapping,
        )

    async def _update_summary_from_kalshi(self, summary: AgentRunSummary) -> None:
        """Update run summary with real Kalshi portfolio data (LIVE mode only).

        Args:
            summary: Run summary to update with live data.
        """
        if not self.kalshi_client:
            return

        try:
            # Get real positions from Kalshi
            positions, _ = await self.kalshi_client.get_positions(limit=200)

            # Get resting orders count
            orders, _ = await self.kalshi_client.list_orders(status="resting", limit=200)

            # Calculate P&L from positions
            total_realized = 0.0
            total_unrealized = 0.0
            total_exposure = 0.0

            for pos in positions:
                total_realized += pos.realized_pnl
                total_unrealized += pos.market_exposure - pos.total_cost
                total_exposure += abs(pos.market_exposure)

            # Update summary with real data
            summary.position_count = len(positions)
            summary.total_realized_pnl = total_realized
            summary.total_unrealized_pnl = total_unrealized
            summary.total_exposure = total_exposure

            # Note: orders_placed/filled are still from this run only (simulator tracking)
            # We keep those as-is since they track activity during THIS run

            logger.debug(
                "kalshi_portfolio_fetched",
                positions=len(positions),
                resting_orders=len(orders),
                realized_pnl=total_realized,
                unrealized_pnl=total_unrealized,
            )

        except Exception as e:
            logger.warning("failed_to_fetch_kalshi_portfolio", error=str(e))
            # Fall back to simulator data if Kalshi fetch fails
            self.simulator.update_run_summary(summary)

    async def _cancel_stale_orders(self) -> int:
        """Cancel stale resting orders where market has moved significantly.
        
        This is an in-game safeguard that cancels orders when:
        1. The market price has diverged significantly from our limit price
        2. The game has started (for pre-game only strategy)
        
        Returns:
            Number of orders cancelled.
        """
        if not self.kalshi_client:
            return 0
        
        detector = StaleOrderDetector(self.strategy.config)
        cancelled = 0
        
        try:
            # Get all resting orders from Kalshi
            orders, _ = await self.kalshi_client.list_orders(status="resting")
            
            if not orders:
                return 0
            
            logger.info("checking_stale_orders", count=len(orders))
            
            for order in orders:
                try:
                    # Get current orderbook for this market
                    orderbook = await self.kalshi_client.get_orderbook(order.ticker)
                    if not orderbook:
                        continue
                    
                    # Get kickoff time for this fixture (if we can find it)
                    kickoff_time = self._get_kickoff_for_ticker(order.ticker)
                    
                    # Check if order is stale
                    stale_info = detector.check_order(
                        order=order,
                        current_ask=orderbook.best_yes_ask,
                        current_bid=orderbook.best_yes_bid,
                        kickoff_time=kickoff_time,
                    )
                    
                    if stale_info:
                        logger.warning(
                            "cancelling_stale_order",
                            order_id=order.order_id,
                            ticker=order.ticker,
                            limit_price=order.price,
                            current_market=stale_info.current_market_price,
                            divergence=stale_info.price_divergence,
                            reason=stale_info.reason,
                        )
                        
                        success = await self.kalshi_client.cancel_order(order.order_id)
                        if success:
                            cancelled += 1
                            
                            # Update local tracking in database
                            self._mark_order_cancelled_in_db(order.order_id)
                
                except Exception as e:
                    logger.warning(
                        "stale_order_check_failed",
                        order_id=order.order_id,
                        error=str(e),
                    )
                    continue
            
            return cancelled
            
        except Exception as e:
            logger.error("cancel_stale_orders_failed", error=str(e))
            return 0
    
    def _get_kickoff_for_ticker(self, ticker: str) -> datetime | None:
        """Get kickoff time for a market ticker from our mappings."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            SELECT f.kickoff_utc
            FROM fixtures_v2 f
            JOIN fixture_market_map m ON f.fixture_id = m.fixture_id
            WHERE m.ticker_home_win = ? OR m.ticker_draw = ? OR m.ticker_away_win = ?
            LIMIT 1
            """,
            (ticker, ticker, ticker),
        )
        row = cursor.fetchone()
        if not row or not row["kickoff_utc"]:
            return None
        
        return datetime.fromisoformat(row["kickoff_utc"])
    
    def _mark_order_cancelled_in_db(self, order_id: str) -> None:
        """Mark an order as cancelled in our local database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            UPDATE live_orders SET status = 'cancelled', cancelled_at = ?
            WHERE order_id = ?
            """,
            (datetime.now(UTC).isoformat(), order_id),
        )
        self.db.connection.commit()

    # --- Database operations ---

    def _create_agent_run(self, summary: AgentRunSummary) -> int:
        """Create agent run record in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO agent_runs (
                run_type, status, started_at, config_hash, config_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                summary.run_type,
                summary.status,
                summary.started_at.isoformat(),
                summary.config_hash,
                json.dumps(summary.config_summary),
            ),
        )
        self.db.connection.commit()
        run_id = cursor.lastrowid
        assert run_id is not None
        return run_id

    def _complete_agent_run(self, summary: AgentRunSummary) -> None:
        """Update completed agent run in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            UPDATE agent_runs SET
                status = ?,
                completed_at = ?,
                fixtures_evaluated = ?,
                markets_evaluated = ?,
                decisions_made = ?,
                orders_placed = ?,
                orders_filled = ?,
                orders_rejected = ?,
                total_realized_pnl = ?,
                total_unrealized_pnl = ?,
                total_exposure = ?,
                position_count = ?,
                error_count = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                summary.status,
                summary.completed_at.isoformat() if summary.completed_at else None,
                summary.fixtures_evaluated,
                summary.markets_evaluated,
                summary.decisions_made,
                summary.orders_placed,
                summary.orders_filled,
                summary.orders_rejected,
                summary.total_realized_pnl,
                summary.total_unrealized_pnl,
                summary.total_exposure,
                summary.position_count,
                summary.error_count,
                summary.error_message,
                summary.run_id,
            ),
        )
        self.db.connection.commit()

    def _store_decision_record(self, decision: DecisionRecord) -> None:
        """Store decision record in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO decision_records (
                decision_id, run_id, fixture_id, market_ticker, outcome, timestamp,
                market_snapshot_json, model_prediction_json, current_position_json,
                edge_calculation_json, kelly_sizing_json,
                action, exit_reason, order_params_json,
                rationale, filters_passed_json, rejection_reason,
                order_placed, order_id, execution_error, strategy_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.run_id,
                decision.fixture_id,
                decision.market_ticker,
                decision.outcome,
                decision.timestamp.isoformat(),
                json.dumps(decision.market_snapshot.to_dict())
                if decision.market_snapshot
                else None,
                json.dumps(decision.model_prediction.to_dict())
                if decision.model_prediction
                else None,
                json.dumps(decision.current_position.to_dict())
                if decision.current_position
                else None,
                json.dumps(json_serialize(decision.edge_calculation.to_dict()))
                if decision.edge_calculation
                else None,
                json.dumps(decision.kelly_sizing.to_dict())
                if decision.kelly_sizing
                else None,
                decision.action.value,
                decision.exit_reason.value if decision.exit_reason else None,
                json.dumps(decision.order_params.to_dict())
                if decision.order_params
                else None,
                decision.rationale,
                json.dumps(json_serialize(decision.filters_passed)) if decision.filters_passed else None,
                decision.rejection_reason,
                1 if decision.order_placed else 0,
                decision.order_id,
                decision.execution_error,
                decision.strategy_name,
            ),
        )
        self.db.connection.commit()

    def _update_decision_with_order(self, decision: DecisionRecord) -> None:
        """Update decision record with order info after execution."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            UPDATE decision_records 
            SET order_placed = ?, order_id = ?
            WHERE decision_id = ?
            """,
            (
                1 if decision.order_placed else 0,
                decision.order_id,
                decision.decision_id,
            ),
        )
        self.db.connection.commit()

    def _store_paper_order(self, order: PaperOrder) -> None:
        """Store paper order in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO paper_orders (
                order_id, run_id, decision_id, ticker, side, action,
                order_type, price, quantity, filled_quantity, status,
                fill_price, fees, rejection_reason, created_at, filled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.run_id,
                order.decision_id,
                order.ticker,
                order.side,
                order.action,
                order.order_type,
                order.price,
                order.quantity,
                order.filled_quantity,
                order.status,
                order.fill_price,
                order.fees,
                order.rejection_reason,
                order.created_at.isoformat(),
                order.filled_at.isoformat() if order.filled_at else None,
            ),
        )
        self.db.connection.commit()

    def _store_paper_fill(self, fill: PaperFill) -> None:
        """Store paper fill in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO paper_fills (
                fill_id, order_id, ticker, side, action, price, quantity, fees, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.ticker,
                fill.side,
                fill.action,
                fill.price,
                fill.quantity,
                fill.fees,
                fill.timestamp.isoformat(),
            ),
        )
        self.db.connection.commit()

    def _update_bandit_outcomes(self) -> None:
        """Update strategy bandit with outcomes from settled positions."""
        if not self.use_bandit or not self.strategy_bandit:
            return

        # Get recently closed positions with strategy names
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            SELECT
                pp.fixture_id,
                pp.ticker,
                pp.realized_pnl,
                pp.average_entry_price,
                pp.quantity,
                dr.strategy_name
            FROM paper_positions pp
            JOIN decision_records dr ON pp.ticker = dr.market_ticker
            WHERE pp.quantity = 0
              AND pp.realized_pnl IS NOT NULL
              AND dr.strategy_name IS NOT NULL
              AND pp.updated_at > datetime('now', '-1 day')
            """
        )

        settled_positions = cursor.fetchall()

        for row in settled_positions:
            fixture_id, ticker, realized_pnl, entry_price, quantity, strategy_name = row
            if fixture_id and strategy_name and realized_pnl is not None:
                exposure = abs(entry_price * quantity)
                self.strategy_bandit.update_outcome(
                    fixture_id=fixture_id,
                    strategy_name=strategy_name,
                    pnl=realized_pnl,
                    exposure=exposure,
                )
                logger.debug(
                    "bandit_outcome_updated",
                    fixture_id=fixture_id,
                    strategy=strategy_name,
                    pnl=realized_pnl,
                )

    def _store_pnl_snapshot(self, snapshot: PnlSnapshot) -> None:
        """Store P&L snapshot in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO pnl_snapshots (
                snapshot_id, run_id, timestamp,
                total_realized_pnl, total_unrealized_pnl, total_pnl,
                total_exposure, position_count, bankroll, positions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.run_id,
                snapshot.timestamp.isoformat(),
                snapshot.total_realized_pnl,
                snapshot.total_unrealized_pnl,
                snapshot.total_pnl,
                snapshot.total_exposure,
                snapshot.position_count,
                snapshot.bankroll,
                json.dumps(snapshot.positions_snapshot),
            ),
        )
        self.db.connection.commit()

    async def _execute_live_order(
        self,
        decision: DecisionRecord,
        run_summary: AgentRunSummary,
    ) -> OrderData | None:
        """Execute a live order via Kalshi API.
        
        Args:
            decision: Decision record with order_params.
            run_summary: Current run summary for tracking.
            
        Returns:
            OrderData if order was placed, None otherwise.
        """
        if decision.order_params is None:
            return None
            
        if self.kalshi_client is None:
            logger.error("kalshi_client_missing", msg="Cannot place live order without Kalshi client")
            return None

        order_params = decision.order_params
        
        try:
            # Get expiration timestamp from order params metadata
            expiration_ts = None
            if hasattr(order_params, 'metadata') and order_params.metadata:
                expiration_ts = order_params.metadata.get('expiration_ts')

            logger.info(
                "placing_live_order",
                ticker=order_params.ticker,
                side=order_params.side,
                action=order_params.action,
                price=order_params.price,
                quantity=order_params.quantity,
                expiration_ts=expiration_ts,
            )

            order_data = await self.kalshi_client.place_limit_order(
                ticker=order_params.ticker,
                side=order_params.side,
                action=order_params.action,
                price=order_params.price,
                quantity=order_params.quantity,
                client_order_id=decision.decision_id,
                expiration_ts=expiration_ts,
            )
            
            logger.info(
                "live_order_placed",
                order_id=order_data.order_id,
                ticker=order_params.ticker,
                status=order_data.status,
                filled_quantity=order_data.filled_quantity,
            )
            
            run_summary.orders_placed += 1
            if order_data.filled_quantity > 0:
                run_summary.orders_filled += 1
            
            # Store in live_orders table
            self._store_live_order(order_data, decision)
            
            return order_data
            
        except Exception as e:
            logger.error(
                "live_order_failed",
                ticker=order_params.ticker,
                error=str(e),
            )
            run_summary.error_count += 1
            return None

    def _store_live_order(self, order_data: OrderData, decision: DecisionRecord) -> None:
        """Store live order in database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO live_orders (
                order_id, decision_id, ticker, side, action,
                order_type, price, quantity, filled_quantity, status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_data.order_id,
                decision.decision_id,
                order_data.ticker,
                order_data.side,
                order_data.action,
                order_data.order_type,
                order_data.price,
                order_data.quantity,
                order_data.filled_quantity,
                order_data.status,
                order_data.created_time.isoformat() if order_data.created_time else utc_now().isoformat(),
            ),
        )
        self.db.connection.commit()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Footbe trading agent"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["paper", "live"],
        default="live",
        help="Trading mode: paper (simulated) or live (default: live)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Minutes between trading loop iterations",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once instead of continuous loop",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute trades (even in live mode)",
    )

    parser.add_argument(
        "--strategy-config",
        type=str,
        default="configs/strategy_config.yaml",
        help="Path to strategy configuration file",
    )

    parser.add_argument(
        "--bankroll",
        type=float,
        default=None,
        help="Override initial bankroll",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for paper trading (for reproducibility)",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configurations
    config = load_config(args.config)
    strategy_config = StrategyConfig.from_yaml(args.strategy_config)

    # Override bankroll if specified
    if args.bankroll is not None:
        strategy_config.initial_bankroll = args.bankroll

    # Initialize database
    db = Database(config.database.path)
    db.connect()
    db.migrate()

    # Initialize Kalshi client (if credentials available)
    kalshi_client: KalshiClient | None = None
    if config.kalshi.api_key_id and config.kalshi.private_key_path:
        kalshi_client = KalshiClient(config.kalshi)

    # Initialize Telegram notifier (if configured)
    telegram_notifier: TelegramNotifier | None = None
    if config.telegram.is_configured and config.telegram.enabled:
        telegram_notifier = TelegramNotifier(config.telegram)
        logger.info("telegram_notifications_enabled")
    elif config.telegram.enabled and not config.telegram.is_configured:
        logger.warning("telegram_enabled_but_not_configured", 
                      hint="Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    
    # Initialize narrative generator
    narrative_generator = NarrativeGenerator()

    # Initialize Football and NBA API clients for live game data
    from footbe_trader.football.client import FootballApiClient
    from footbe_trader.nba.client import NBAApiClient
    
    football_client: FootballApiClient | None = None
    nba_client: NBAApiClient | None = None
    
    if hasattr(config, 'football_api') and config.football_api.api_key:
        football_client = FootballApiClient(config.football_api)
        nba_client = NBAApiClient(config.football_api)  # Same API key
    
    # Create live game provider
    live_game_provider: LiveGameStateProvider | None = None
    if football_client or nba_client:
        live_game_provider = LiveGameStateProvider(
            football_client=football_client,
            nba_client=nba_client,
        )

    # Load Poisson model
    poisson_model: PoissonModel | None = None
    model_path = Path("models/poisson_v1.pkl")
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                poisson_model = pickle.load(f)
            logger.info(
                "poisson_model_loaded",
                path=str(model_path),
                teams=len(poisson_model.team_ratings),
                home_advantage=poisson_model.home_advantage,
            )
        except Exception as e:
            logger.warning("poisson_model_load_failed", error=str(e))
    else:
        logger.warning("poisson_model_not_found", path=str(model_path))

    # Initialize strategy and simulator
    strategy = EdgeStrategy(strategy_config)
    simulator = PaperTradingSimulator(
        config=strategy_config,
        initial_bankroll=strategy_config.initial_bankroll,
    )

    if args.seed is not None:
        simulator.set_seed(args.seed)

    # Create trading agent
    agent = TradingAgent(
        db=db,
        kalshi_client=kalshi_client,
        strategy=strategy,
        simulator=simulator,
        mode=args.mode,
        dry_run=args.dry_run,
        live_game_provider=live_game_provider,
        telegram_notifier=telegram_notifier,
        narrative_generator=narrative_generator,
        use_bandit=False,  # Disable bandit - use config file strategy
        starting_bankroll=strategy_config.initial_bankroll,
        config=config,  # For model lifecycle manager
        poisson_model=poisson_model,  # Trained prediction model
    )

    try:
        # Enter async context managers for all clients
        async def run_with_clients():
            """Run agent with proper client context management."""
            # Stack of context managers to enter
            contexts_to_enter = []
            if kalshi_client:
                contexts_to_enter.append(kalshi_client)
            if football_client:
                contexts_to_enter.append(football_client)
            if nba_client:
                contexts_to_enter.append(nba_client)
            
            # Simple context management without contextlib stack
            try:
                for ctx in contexts_to_enter:
                    await ctx.__aenter__()
                
                if args.once:
                    summary = await agent.run_once()
                    # Send Telegram notification for single run (live mode only)
                    if telegram_notifier and args.mode == "live":
                        narrative = narrative_generator.generate_run_narrative(
                            fixtures_evaluated=summary.fixtures_evaluated,
                            markets_evaluated=summary.markets_evaluated,
                            decisions_made=summary.decisions_made,
                            orders_placed=summary.orders_placed,
                            orders_filled=summary.orders_filled,
                            skipped_reasons={},
                            trades_by_outcome={},
                            live_games=[],
                            cancelled_stale=0,
                            edge_summary={},
                        )
                        positions_data = [
                            {
                                "ticker": p.ticker,
                                "unrealized_pnl": p.unrealized_pnl,
                                "quantity": p.quantity,
                            }
                            for p in simulator.positions.values()
                        ]
                        position_narrative = narrative_generator.generate_position_narrative(
                            positions=positions_data,
                            total_pnl=summary.total_unrealized_pnl,
                        )
                        full_narrative = f"{narrative}\n\n{position_narrative}"
                        await telegram_notifier.send_run_summary(
                            run_id=summary.run_id or 0,
                            mode=args.mode,
                            fixtures_evaluated=summary.fixtures_evaluated,
                            markets_evaluated=summary.markets_evaluated,
                            decisions_made=summary.decisions_made,
                            orders_placed=summary.orders_placed,
                            orders_filled=summary.orders_filled,
                            realized_pnl=summary.total_realized_pnl,
                            unrealized_pnl=summary.total_unrealized_pnl,
                            total_exposure=summary.total_exposure,
                            position_count=summary.position_count,
                            narrative=full_narrative,
                        )
                else:
                    await agent.run_loop(args.interval)
            finally:
                # Exit contexts in reverse order
                for ctx in reversed(contexts_to_enter):
                    try:
                        await ctx.__aexit__(None, None, None)
                    except Exception as e:
                        logger.warning(f"Error closing context: {e}")
        
        if kalshi_client or football_client or nba_client:
            await run_with_clients()
        else:
            logger.warning("running_without_api_clients")
            if args.once:
                summary = await agent.run_once()
                # Send Telegram notification for single run (live mode only)
                if telegram_notifier and args.mode == "live":
                    narrative = narrative_generator.generate_run_narrative(
                        fixtures_evaluated=summary.fixtures_evaluated,
                        markets_evaluated=summary.markets_evaluated,
                        decisions_made=summary.decisions_made,
                        orders_placed=summary.orders_placed,
                        orders_filled=summary.orders_filled,
                        skipped_reasons={},
                        trades_by_outcome={},
                        live_games=[],
                        cancelled_stale=0,
                        edge_summary={},
                    )
                    positions_data = [
                        {
                            "ticker": p.ticker,
                            "unrealized_pnl": p.unrealized_pnl,
                            "quantity": p.quantity,
                        }
                        for p in simulator.positions.values()
                    ]
                    position_narrative = narrative_generator.generate_position_narrative(
                        positions=positions_data,
                        total_pnl=summary.total_unrealized_pnl,
                    )
                    full_narrative = f"{narrative}\n\n{position_narrative}"
                    await telegram_notifier.send_run_summary(
                        run_id=summary.run_id or 0,
                        mode=args.mode,
                        fixtures_evaluated=summary.fixtures_evaluated,
                        markets_evaluated=summary.markets_evaluated,
                        decisions_made=summary.decisions_made,
                        orders_placed=summary.orders_placed,
                        orders_filled=summary.orders_filled,
                        realized_pnl=summary.total_realized_pnl,
                        unrealized_pnl=summary.total_unrealized_pnl,
                        total_exposure=summary.total_exposure,
                        position_count=summary.position_count,
                        narrative=full_narrative,
                    )
            else:
                await agent.run_loop(args.interval)

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
