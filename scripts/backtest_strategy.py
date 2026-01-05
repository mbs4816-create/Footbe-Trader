#!/usr/bin/env python3
"""Strategy Backtest CLI.

Run a strategy backtest over collected historical snapshots and generate
performance reports.

Usage:
    python scripts/backtest_strategy.py --session-id abc123
    python scripts/backtest_strategy.py --since 2024-01-01 --until 2024-02-01
    python scripts/backtest_strategy.py --fixture 12345
    python scripts/backtest_strategy.py --output-dir reports/backtest_001
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.storage.database import Database
from footbe_trader.storage.models import HistoricalSnapshot
from footbe_trader.strategy.backtest_metrics import (
    BacktestMetricsCalculator,
    generate_backtest_report,
)
from footbe_trader.strategy.strategy_backtest import (
    BacktestConfig,
    StrategyBacktester,
)
from footbe_trader.strategy.trading_strategy import StrategyConfig

logger = get_logger(__name__)


def run_backtest(
    db: Database,
    strategy_config: StrategyConfig,
    backtest_config: BacktestConfig,
    session_id: str | None = None,
    fixture_ids: list[int] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    output_dir: Path | None = None,
) -> int:
    """Run strategy backtest.

    Args:
        db: Database connection.
        strategy_config: Trading strategy configuration.
        backtest_config: Backtest-specific configuration.
        session_id: Filter snapshots by session ID.
        fixture_ids: Filter snapshots by fixture IDs.
        since: Filter snapshots after this time.
        until: Filter snapshots before this time.
        output_dir: Directory to write reports.

    Returns:
        Exit code (0 = success).
    """
    # Load snapshots
    logger.info(
        "loading_snapshots",
        session_id=session_id,
        fixture_ids=fixture_ids,
        since=since.isoformat() if since else None,
        until=until.isoformat() if until else None,
    )

    snapshots = []
    if fixture_ids:
        for fid in fixture_ids:
            fid_snapshots = db.get_historical_snapshots(
                fixture_id=fid,
                session_id=session_id,
                since=since,
                until=until,
            )
            snapshots.extend(fid_snapshots)
    else:
        snapshots = db.get_historical_snapshots(
            session_id=session_id,
            since=since,
            until=until,
        )

    if not snapshots:
        logger.error("no_snapshots_found")
        print("Error: No snapshots found matching the criteria.")
        return 1

    logger.info("snapshots_loaded", count=len(snapshots))

    # Get fixture outcomes and kickoffs
    fixture_ids_in_data = list(set(s.fixture_id for s in snapshots))
    fixture_outcomes = _get_fixture_outcomes(db, fixture_ids_in_data)
    fixture_kickoffs = _get_fixture_kickoffs(db, fixture_ids_in_data)

    logger.info(
        "fixtures_loaded",
        fixtures=len(fixture_ids_in_data),
        with_outcomes=len(fixture_outcomes),
        with_kickoffs=len(fixture_kickoffs),
    )

    # Run backtest
    logger.info("running_backtest")
    backtester = StrategyBacktester(
        strategy_config=strategy_config,
        backtest_config=backtest_config,
    )

    backtest_result = backtester.run(
        snapshots=snapshots,
        fixture_outcomes=fixture_outcomes,
        fixture_kickoffs=fixture_kickoffs,
    )

    # Store backtest in DB
    db.create_strategy_backtest(backtest_result)

    # Store trades and equity curve
    for trade in backtester.get_trades():
        db.create_backtest_trade(trade)
    for equity in backtester.get_equity_curve():
        db.create_backtest_equity(equity)

    # Update backtest record with final results
    db.update_strategy_backtest(backtest_result)

    logger.info(
        "backtest_complete",
        backtest_id=backtest_result.backtest_id,
        total_return=backtest_result.total_return,
        max_drawdown=backtest_result.max_drawdown,
        total_trades=backtest_result.total_trades,
    )

    # Calculate detailed metrics
    metrics_calc = BacktestMetricsCalculator()
    metrics = metrics_calc.calculate(
        backtest=backtest_result,
        trades=backtester.get_trades(),
        equity_curve=backtester.get_equity_curve(),
    )

    # Print summary
    _print_summary(metrics)

    # Generate report files
    if output_dir:
        generate_backtest_report(metrics, output_dir)
        print(f"\nReport written to: {output_dir}")

    return 0


def _get_fixture_outcomes(
    db: Database,
    fixture_ids: list[int],
) -> dict[int, str]:
    """Get actual outcomes for fixtures.

    Args:
        db: Database connection.
        fixture_ids: List of fixture IDs.

    Returns:
        Dict mapping fixture_id to outcome ('home_win', 'draw', 'away_win').
    """
    outcomes = {}
    for fid in fixture_ids:
        fixture = db.get_fixture_v2(fid)
        if fixture and fixture.status == "FT":  # Full Time
            if fixture.home_goals is not None and fixture.away_goals is not None:
                if fixture.home_goals > fixture.away_goals:
                    outcomes[fid] = "home_win"
                elif fixture.away_goals > fixture.home_goals:
                    outcomes[fid] = "away_win"
                else:
                    outcomes[fid] = "draw"
    return outcomes


def _get_fixture_kickoffs(
    db: Database,
    fixture_ids: list[int],
) -> dict[int, datetime]:
    """Get kickoff times for fixtures.

    Args:
        db: Database connection.
        fixture_ids: List of fixture IDs.

    Returns:
        Dict mapping fixture_id to kickoff time.
    """
    kickoffs = {}
    for fid in fixture_ids:
        fixture = db.get_fixture_v2(fid)
        if fixture and fixture.kickoff_utc:
            kickoffs[fid] = fixture.kickoff_utc
    return kickoffs


def _print_summary(metrics: Any) -> None:
    """Print backtest summary to console."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nBacktest ID: {metrics.backtest_id}")
    print(f"Period: {metrics.backtest_period_start} to {metrics.backtest_period_end}")
    print(f"Duration: {metrics.backtest_duration_hours:.1f} hours")

    print("\n" + "-" * 60)
    print("RETURNS")
    print("-" * 60)
    print(f"Initial Bankroll:  ${metrics.initial_bankroll:,.2f}")
    print(f"Final Bankroll:    ${metrics.final_bankroll:,.2f}")
    print(f"Total Return:      ${metrics.returns.total_return:,.2f} ({metrics.returns.total_return_pct:.2%})")
    if metrics.returns.annualized_return:
        print(f"Annualized Return: {metrics.returns.annualized_return:.2%}")

    print("\n" + "-" * 60)
    print("RISK")
    print("-" * 60)
    print(f"Max Drawdown:      {metrics.risk.max_drawdown:.2%}")
    print(f"Avg Drawdown:      {metrics.risk.avg_drawdown:.2%}")
    if metrics.risk.volatility:
        print(f"Volatility:        {metrics.risk.volatility:.4f}")

    print("\n" + "-" * 60)
    print("RISK-ADJUSTED")
    print("-" * 60)
    if metrics.ratios.sharpe_ratio:
        print(f"Sharpe Ratio:      {metrics.ratios.sharpe_ratio:.3f}")
    if metrics.ratios.sortino_ratio:
        print(f"Sortino Ratio:     {metrics.ratios.sortino_ratio:.3f}")
    if metrics.ratios.profit_factor:
        print(f"Profit Factor:     {metrics.ratios.profit_factor:.3f}")

    print("\n" + "-" * 60)
    print("TRADES")
    print("-" * 60)
    print(f"Total Trades:      {metrics.trades.total_trades}")
    print(f"Winning Trades:    {metrics.trades.winning_trades}")
    print(f"Losing Trades:     {metrics.trades.losing_trades}")
    print(f"Win Rate:          {metrics.trades.win_rate:.2%}")
    print(f"Avg Trade:         ${metrics.trades.avg_trade:,.2f}")
    if metrics.trades.avg_hold_time_minutes:
        if metrics.trades.avg_hold_time_minutes < 60:
            print(f"Avg Hold Time:     {metrics.trades.avg_hold_time_minutes:.1f} min")
        else:
            print(f"Avg Hold Time:     {metrics.trades.avg_hold_time_minutes / 60:.1f} hours")

    print("\n" + "-" * 60)
    print("EDGE CALIBRATION")
    print("-" * 60)
    for bucket in metrics.edge_calibration:
        print(
            f"  {bucket.edge_bucket}: n={bucket.trade_count:3d}, "
            f"expected={bucket.expected_return:+.2%}, "
            f"actual={bucket.actual_return:+.2%}, "
            f"error={bucket.calibration_error:+.2%}"
        )

    print("\n" + "-" * 60)
    print("OUTCOME BREAKDOWN")
    print("-" * 60)
    for outcome in metrics.outcome_breakdown:
        print(
            f"  {outcome.outcome:10s}: trades={outcome.trades:3d}, "
            f"win_rate={outcome.win_rate:.0%}, "
            f"pnl=${outcome.total_pnl:+,.2f}"
        )

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run strategy backtest over historical snapshots"
    )

    # Snapshot filtering
    parser.add_argument(
        "--session-id",
        help="Filter snapshots by session ID",
    )
    parser.add_argument(
        "--fixture",
        type=int,
        action="append",
        dest="fixtures",
        help="Filter by fixture ID (can specify multiple)",
    )
    parser.add_argument(
        "--since",
        type=lambda s: datetime.fromisoformat(s),
        help="Filter snapshots after this datetime (ISO format)",
    )
    parser.add_argument(
        "--until",
        type=lambda s: datetime.fromisoformat(s),
        help="Filter snapshots before this datetime (ISO format)",
    )

    # Strategy configuration
    parser.add_argument(
        "--strategy-config",
        type=Path,
        help="Path to strategy config YAML",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        help="Override min_edge_to_enter",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        help="Override kelly_fraction",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        help="Override take_profit threshold",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        help="Override stop_loss threshold",
    )

    # Backtest configuration
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Initial bankroll (default: 10000)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.01,
        help="Slippage in cents (default: 0.01)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write report files",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/footbe.db",
        help="Database path (default: data/footbe.db)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Initialize database
    db = Database(args.db_path)
    db.connect()
    db.ensure_schema()

    try:
        # Load or create strategy config
        if args.strategy_config:
            strategy_config = StrategyConfig.from_yaml(args.strategy_config)
        else:
            strategy_config = StrategyConfig()

        # Apply overrides
        if args.min_edge is not None:
            strategy_config.min_edge_to_enter = args.min_edge
        if args.kelly_fraction is not None:
            strategy_config.kelly_fraction = args.kelly_fraction
        if args.take_profit is not None:
            strategy_config.take_profit = args.take_profit
        if args.stop_loss is not None:
            strategy_config.stop_loss = args.stop_loss

        # Create backtest config
        backtest_config = BacktestConfig(
            initial_bankroll=args.initial_bankroll,
            slippage_cents=args.slippage,
        )

        # Run backtest
        return run_backtest(
            db=db,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            session_id=args.session_id,
            fixture_ids=args.fixtures,
            since=args.since,
            until=args.until,
            output_dir=args.output_dir,
        )

    except Exception as e:
        logger.error("backtest_failed", error=str(e))
        print(f"Error: {e}")
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
