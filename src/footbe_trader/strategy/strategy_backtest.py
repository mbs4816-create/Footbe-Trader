"""Strategy Backtester.

Replays historical snapshots chronologically to simulate trading with
edge-based strategy logic, tracking positions and P&L over time.
"""

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.kalshi.interfaces import MarketData, OrderbookData, OrderbookLevel
from footbe_trader.storage.models import (
    BacktestEquity,
    BacktestTrade,
    HistoricalSnapshot,
    StrategyBacktest,
)
from footbe_trader.strategy.trading_strategy import (
    EdgeStrategy,
    StrategyConfig,
)

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for strategy backtest."""

    initial_bankroll: float = 10000.0

    # Fill simulation
    slippage_cents: float = 0.01
    fill_probability: float = 1.0  # For backtest, assume all fills succeed

    # Time settings
    equity_snapshot_interval_minutes: int = 5
    assume_settlement_at_kickoff: bool = True

    # Filtering
    min_snapshots_per_fixture: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "slippage_cents": self.slippage_cents,
            "fill_probability": self.fill_probability,
            "equity_snapshot_interval_minutes": self.equity_snapshot_interval_minutes,
            "assume_settlement_at_kickoff": self.assume_settlement_at_kickoff,
            "min_snapshots_per_fixture": self.min_snapshots_per_fixture,
        }


@dataclass
class BacktestPosition:
    """Position state during backtest."""

    ticker: str
    fixture_id: int
    outcome: str
    quantity: int = 0
    average_entry_price: float = 0.0
    total_cost: float = 0.0
    entry_timestamp: datetime | None = None
    entry_edge: float | None = None
    entry_model_prob: float | None = None
    mark_price: float = 0.0
    mtm_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L at current mark."""
        if self.quantity == 0:
            return 0.0
        # For yes position: value = quantity * mark_price
        # P&L = value - cost
        return (self.quantity * self.mark_price) - self.total_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "fixture_id": self.fixture_id,
            "outcome": self.outcome,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "total_cost": self.total_cost,
            "mark_price": self.mark_price,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass
class BacktestState:
    """Current state of the backtest."""

    bankroll: float = 10000.0
    positions: dict[str, BacktestPosition] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[BacktestEquity] = field(default_factory=list)
    realized_pnl: float = 0.0
    peak_equity: float = 10000.0
    current_timestamp: datetime | None = None

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_exposure(self) -> float:
        """Total exposure (cost basis of open positions)."""
        return sum(p.total_cost for p in self.positions.values() if p.quantity > 0)

    @property
    def current_equity(self) -> float:
        """Current portfolio value."""
        return self.bankroll + self.total_pnl

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity


class StrategyBacktester:
    """Replays historical snapshots to backtest trading strategy.

    This backtester:
    1. Loads snapshots chronologically by fixture and timestamp
    2. At each tick, uses only data available as-of that time
    3. Computes model probabilities (if available in snapshots)
    4. Applies strategy logic to generate trade decisions
    5. Simulates fills with configurable slippage
    6. Tracks positions, P&L, and equity curve
    7. Handles settlement at fixture kickoff
    """

    def __init__(
        self,
        strategy_config: StrategyConfig | None = None,
        backtest_config: BacktestConfig | None = None,
    ):
        """Initialize backtester.

        Args:
            strategy_config: Trading strategy configuration.
            backtest_config: Backtest-specific configuration.
        """
        self.strategy_config = strategy_config or StrategyConfig()
        self.backtest_config = backtest_config or BacktestConfig()

        # Initialize strategy
        self.strategy = EdgeStrategy(config=self.strategy_config)

        # State
        self.state = BacktestState(bankroll=self.backtest_config.initial_bankroll)
        self.backtest_id = str(uuid.uuid4())
        self._trade_counter = 0
        self._last_equity_snapshot: datetime | None = None

    def run(
        self,
        snapshots: list[HistoricalSnapshot],
        fixture_outcomes: dict[int, str] | None = None,
        fixture_kickoffs: dict[int, datetime] | None = None,
    ) -> StrategyBacktest:
        """Run backtest over historical snapshots.

        Args:
            snapshots: List of historical snapshots, will be sorted by timestamp.
            fixture_outcomes: Dict mapping fixture_id to actual outcome
                              ('home_win', 'draw', 'away_win') for settlement.
            fixture_kickoffs: Dict mapping fixture_id to kickoff time.

        Returns:
            BacktestResult with metrics and trade history.
        """
        fixture_outcomes = fixture_outcomes or {}
        fixture_kickoffs = fixture_kickoffs or {}

        # Sort snapshots by timestamp
        sorted_snapshots = sorted(snapshots, key=lambda s: s.timestamp)

        if not sorted_snapshots:
            return self._create_empty_result()

        # Group snapshots by fixture and timestamp
        snapshots_by_fixture = self._group_snapshots_by_fixture(sorted_snapshots)

        # Filter fixtures with too few snapshots
        snapshots_by_fixture = {
            fid: snaps
            for fid, snaps in snapshots_by_fixture.items()
            if len(snaps) >= self.backtest_config.min_snapshots_per_fixture
        }

        logger.info(
            "backtest_starting",
            backtest_id=self.backtest_id,
            total_snapshots=len(sorted_snapshots),
            fixtures=len(snapshots_by_fixture),
            start_time=sorted_snapshots[0].timestamp.isoformat(),
            end_time=sorted_snapshots[-1].timestamp.isoformat(),
        )

        # Process all snapshots chronologically
        all_timestamps = sorted(set(s.timestamp for s in sorted_snapshots))

        for timestamp in all_timestamps:
            self.state.current_timestamp = timestamp

            # Get all snapshots at this timestamp
            tick_snapshots = [s for s in sorted_snapshots if s.timestamp == timestamp]

            # Group by fixture
            tick_by_fixture: dict[int, list[HistoricalSnapshot]] = defaultdict(list)
            for snap in tick_snapshots:
                tick_by_fixture[snap.fixture_id].append(snap)

            # Process each fixture
            for fixture_id, fixture_snapshots in tick_by_fixture.items():
                kickoff = fixture_kickoffs.get(fixture_id)
                actual_outcome = fixture_outcomes.get(fixture_id)

                # Check if fixture has settled (past kickoff)
                if kickoff and timestamp >= kickoff:
                    self._settle_fixture(fixture_id, actual_outcome, timestamp)
                    continue

                # Process trading decisions
                self._process_fixture_tick(
                    fixture_id=fixture_id,
                    snapshots=fixture_snapshots,
                    timestamp=timestamp,
                    kickoff=kickoff,
                )

            # Update marks and record equity
            self._mark_positions(tick_snapshots)
            self._maybe_record_equity(timestamp)

        # Final settlement of any remaining positions
        self._settle_all_remaining(fixture_outcomes, fixture_kickoffs)

        # Record final equity point
        if self.state.current_timestamp:
            self._record_equity(self.state.current_timestamp)

        return self._create_result()

    def _group_snapshots_by_fixture(
        self, snapshots: list[HistoricalSnapshot]
    ) -> dict[int, list[HistoricalSnapshot]]:
        """Group snapshots by fixture ID."""
        by_fixture: dict[int, list[HistoricalSnapshot]] = defaultdict(list)
        for snap in snapshots:
            by_fixture[snap.fixture_id].append(snap)
        return dict(by_fixture)

    def _process_fixture_tick(
        self,
        fixture_id: int,
        snapshots: list[HistoricalSnapshot],
        timestamp: datetime,
        kickoff: datetime | None,
    ) -> None:
        """Process a single tick for a fixture.

        Uses a simplified edge-based decision logic directly rather than
        invoking the full EdgeStrategy to avoid needing full fixture context.

        Args:
            fixture_id: Fixture ID.
            snapshots: Snapshots for this fixture at this timestamp.
            timestamp: Current timestamp.
            kickoff: Fixture kickoff time.
        """
        # Process each outcome snapshot
        for snap in snapshots:
            ticker = snap.ticker
            position = self.state.positions.get(ticker)
            model_prob = snap.model_prob or 0.0
            market_price = snap.best_ask or 0.0

            # Skip if no market price
            if market_price <= 0 or market_price >= 1:
                continue

            # Compute edge
            edge = model_prob - market_price

            # Decision logic based on strategy config
            min_edge = self.strategy_config.min_edge_to_enter
            min_liquidity = self.strategy_config.min_ask_volume

            # Check liquidity (use ask volume as proxy)
            liquidity = snap.ask_volume or 0

            # Entry logic
            if position is None or position.quantity == 0:
                # Consider entering if edge exceeds threshold
                if edge >= min_edge and liquidity >= min_liquidity and model_prob > 0:
                    # Size using fractional Kelly
                    kelly = self._calculate_kelly(edge, market_price, model_prob)
                    bet_size = kelly * self.state.bankroll * self.strategy_config.kelly_fraction

                    # Apply position limits
                    max_bet = self.strategy_config.max_kelly_fraction * self.state.bankroll
                    bet_size = min(bet_size, max_bet)

                    # Convert to quantity
                    price = market_price + self.backtest_config.slippage_cents
                    quantity = int(bet_size / price) if price > 0 else 0

                    if quantity > 0 and bet_size <= self.state.bankroll:
                        self._enter_position(
                            fixture_id=fixture_id,
                            ticker=ticker,
                            outcome=snap.outcome,
                            quantity=quantity,
                            price=price,
                            edge=edge,
                            model_prob=model_prob,
                            timestamp=timestamp,
                        )

            else:
                # Consider exiting existing position
                # Exit if edge reverses or crosses TP/SL
                entry_price = position.average_entry_price
                current_mid = snap.mid or market_price

                # Calculate P&L percentage
                if entry_price > 0:
                    pnl_pct = (current_mid - entry_price) / entry_price
                else:
                    pnl_pct = 0.0

                should_exit = False
                exit_reason = ""

                # Take profit
                if pnl_pct >= self.strategy_config.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"

                # Stop loss
                elif pnl_pct <= -self.strategy_config.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"

                # Edge reversal
                elif edge < 0:
                    should_exit = True
                    exit_reason = "edge_reversal"

                if should_exit:
                    exit_price = snap.best_bid or current_mid
                    exit_price -= self.backtest_config.slippage_cents
                    self._exit_position(
                        position=position,
                        price=exit_price,
                        reason=exit_reason,
                        timestamp=timestamp,
                    )

    def _calculate_kelly(self, edge: float, market_price: float, model_prob: float) -> float:
        """Calculate Kelly criterion bet fraction.

        Args:
            edge: Edge (model_prob - market_price).
            market_price: Current market ask price.
            model_prob: Model probability.

        Returns:
            Kelly fraction (capped at 1.0).
        """
        if model_prob <= 0 or model_prob >= 1:
            return 0.0

        # Win odds (decimal)
        win_payout = 1.0 / market_price - 1 if market_price > 0 else 0

        if win_payout <= 0:
            return 0.0

        # Kelly formula: f* = (b*p - q) / b
        # where b = odds, p = win prob, q = lose prob
        q = 1 - model_prob
        kelly = (win_payout * model_prob - q) / win_payout

        return max(0.0, min(kelly, 1.0))

    def _enter_position(
        self,
        fixture_id: int,
        ticker: str,
        outcome: str,
        quantity: int,
        price: float,
        edge: float,
        model_prob: float,
        timestamp: datetime,
    ) -> None:
        """Enter a new position.

        Args:
            fixture_id: Fixture ID.
            ticker: Market ticker.
            outcome: Outcome type.
            quantity: Number of contracts.
            price: Fill price.
            edge: Entry edge.
            model_prob: Model probability.
            timestamp: Entry timestamp.
        """
        cost = price * quantity

        position = BacktestPosition(
            ticker=ticker,
            fixture_id=fixture_id,
            outcome=outcome,
            quantity=quantity,
            average_entry_price=price,
            total_cost=cost,
            entry_timestamp=timestamp,
            entry_edge=edge,
            entry_model_prob=model_prob,
            mark_price=price,
        )
        self.state.positions[ticker] = position
        self.state.bankroll -= cost

        # Record trade
        self._trade_counter += 1
        trade = BacktestTrade(
            backtest_id=self.backtest_id,
            trade_id=f"T{self._trade_counter:06d}",
            fixture_id=fixture_id,
            ticker=ticker,
            outcome=outcome,
            entry_timestamp=timestamp,
            entry_price=price,
            entry_quantity=quantity,
            entry_edge=edge,
            entry_model_prob=model_prob,
            entry_reason="edge_threshold",
        )
        self.state.trades.append(trade)

        logger.debug(
            "backtest_position_entered",
            ticker=ticker,
            quantity=quantity,
            price=price,
            edge=edge,
        )

    def _exit_position(
        self,
        position: BacktestPosition,
        price: float,
        reason: str,
        timestamp: datetime,
    ) -> None:
        """Exit an existing position.

        Args:
            position: Position to exit.
            price: Exit price.
            reason: Exit reason.
            timestamp: Exit timestamp.
        """
        # Calculate realized P&L
        exit_value = position.quantity * price
        realized = exit_value - position.total_cost

        # Update state
        self.state.bankroll += exit_value
        self.state.realized_pnl += realized

        # Update trade record
        for trade in self.state.trades:
            if trade.ticker == position.ticker and trade.exit_timestamp is None:
                trade.exit_timestamp = timestamp
                trade.exit_price = price
                trade.exit_quantity = position.quantity
                trade.exit_reason = reason
                trade.realized_pnl = realized
                trade.hold_time_minutes = int(
                    (timestamp - trade.entry_timestamp).total_seconds() / 60
                ) if trade.entry_timestamp else 0
                break

        # Remove position
        del self.state.positions[position.ticker]

        logger.debug(
            "backtest_position_exited",
            ticker=position.ticker,
            quantity=position.quantity,
            price=price,
            reason=reason,
            realized_pnl=realized,
        )

    def _settle_fixture(
        self,
        fixture_id: int,
        actual_outcome: str | None,
        timestamp: datetime,
    ) -> None:
        """Settle all positions for a fixture.

        Args:
            fixture_id: Fixture ID.
            actual_outcome: Actual outcome ('home_win', 'draw', 'away_win').
            timestamp: Settlement timestamp.
        """
        positions_to_settle = [
            (ticker, pos)
            for ticker, pos in self.state.positions.items()
            if pos.fixture_id == fixture_id and pos.quantity > 0
        ]

        for ticker, position in positions_to_settle:
            # Determine settlement price
            if actual_outcome is None:
                # Unknown outcome, settle at mid
                settlement_price = position.mark_price
            elif position.outcome == actual_outcome:
                # Winning outcome settles at 1.0
                settlement_price = 1.0
            else:
                # Losing outcome settles at 0.0
                settlement_price = 0.0

            # Calculate P&L
            settlement_value = settlement_price * position.quantity
            pnl = settlement_value - position.total_cost

            # Update state
            self.state.realized_pnl += pnl
            self.state.bankroll += settlement_value

            # Update trade record
            for trade in reversed(self.state.trades):
                if trade.ticker == ticker and trade.exit_timestamp is None:
                    trade.exit_timestamp = timestamp
                    trade.exit_price = settlement_price
                    trade.exit_quantity = position.quantity
                    trade.exit_reason = "settlement"
                    trade.realized_pnl = pnl
                    trade.return_pct = (
                        pnl / position.total_cost if position.total_cost > 0 else 0
                    )
                    if trade.entry_timestamp:
                        trade.hold_time_minutes = (
                            timestamp - trade.entry_timestamp
                        ).total_seconds() / 60
                    trade.mtm_history_json = position.mtm_history.copy()
                    break

            logger.debug(
                "backtest_position_settled",
                ticker=ticker,
                outcome=position.outcome,
                actual=actual_outcome,
                settlement_price=settlement_price,
                pnl=pnl,
            )

            # Clear position
            position.quantity = 0
            position.total_cost = 0

    def _settle_all_remaining(
        self,
        fixture_outcomes: dict[int, str],
        fixture_kickoffs: dict[int, datetime],
    ) -> None:
        """Settle all remaining open positions."""
        remaining_fixtures = set(
            pos.fixture_id
            for pos in self.state.positions.values()
            if pos.quantity > 0
        )

        for fixture_id in remaining_fixtures:
            kickoff = fixture_kickoffs.get(fixture_id)
            outcome = fixture_outcomes.get(fixture_id)
            settlement_time = kickoff or self.state.current_timestamp or utc_now()
            self._settle_fixture(fixture_id, outcome, settlement_time)

    def _mark_positions(self, snapshots: list[HistoricalSnapshot]) -> None:
        """Update mark prices for all positions.

        Args:
            snapshots: Current tick's snapshots.
        """
        # Build ticker -> mid price map
        marks = {snap.ticker: snap.mid or 0.5 for snap in snapshots}

        for ticker, position in self.state.positions.items():
            if position.quantity > 0:
                if ticker in marks:
                    position.mark_price = marks[ticker]

                    # Record MTM history
                    if self.state.current_timestamp:
                        position.mtm_history.append({
                            "timestamp": self.state.current_timestamp.isoformat(),
                            "mid": position.mark_price,
                            "unrealized_pnl": position.unrealized_pnl,
                        })

        # Update peak equity
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity

    def _maybe_record_equity(self, timestamp: datetime) -> None:
        """Record equity snapshot if interval has passed."""
        if self._last_equity_snapshot is None:
            self._record_equity(timestamp)
            return

        delta = (timestamp - self._last_equity_snapshot).total_seconds() / 60
        if delta >= self.backtest_config.equity_snapshot_interval_minutes:
            self._record_equity(timestamp)

    def _record_equity(self, timestamp: datetime) -> None:
        """Record a point on the equity curve."""
        equity = BacktestEquity(
            backtest_id=self.backtest_id,
            timestamp=timestamp,
            bankroll=self.state.bankroll,
            total_exposure=self.state.total_exposure,
            position_count=len(
                [p for p in self.state.positions.values() if p.quantity > 0]
            ),
            realized_pnl=self.state.realized_pnl,
            unrealized_pnl=self.state.unrealized_pnl,
            total_pnl=self.state.total_pnl,
            drawdown=self.state.drawdown,
        )
        self.state.equity_curve.append(equity)
        self._last_equity_snapshot = timestamp

    def _get_fixture_exposure(self, fixture_id: int) -> float:
        """Get total exposure for a fixture."""
        return sum(
            p.total_cost
            for p in self.state.positions.values()
            if p.fixture_id == fixture_id and p.quantity > 0
        )

    def _snapshot_to_orderbook(self, snap: HistoricalSnapshot) -> OrderbookData:
        """Convert historical snapshot to OrderbookData."""
        # Build from raw JSON if available
        raw = snap.raw_orderbook_json or {}

        yes_bids = [
            OrderbookLevel(price=l["price"], quantity=l["quantity"])
            for l in raw.get("yes_bids", [])
        ]
        yes_asks = [
            OrderbookLevel(price=l["price"], quantity=l["quantity"])
            for l in raw.get("yes_asks", [])
        ]

        # If no raw data, build from summary
        if not yes_bids and snap.best_bid is not None:
            yes_bids = [OrderbookLevel(price=snap.best_bid, quantity=snap.bid_volume or 0)]
        if not yes_asks and snap.best_ask is not None:
            yes_asks = [OrderbookLevel(price=snap.best_ask, quantity=snap.ask_volume or 0)]

        return OrderbookData(
            ticker=snap.ticker,
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            timestamp=snap.timestamp,
        )

    def _snapshot_to_market(self, snap: HistoricalSnapshot) -> MarketData | None:
        """Convert historical snapshot to MarketData."""
        raw = snap.raw_market_json or {}

        if not raw and snap.yes_price is None:
            return None

        return MarketData(
            ticker=snap.ticker,
            title="",
            status="open",
            yes_bid=snap.yes_price or snap.best_bid or 0,
            yes_ask=snap.best_ask or 0,
            no_bid=snap.no_price or 0,
            no_ask=0,
            volume_24h=snap.volume_24h or 0,
            open_interest=snap.open_interest or 0,
            raw_data=raw,
        )

    def _create_empty_result(self) -> StrategyBacktest:
        """Create an empty backtest result."""
        return StrategyBacktest(
            backtest_id=self.backtest_id,
            started_at=utc_now(),
            completed_at=utc_now(),
            status="completed",
            strategy_config_hash=self.strategy_config.config_hash(),
            strategy_config_json=self.strategy_config.to_dict(),
            initial_bankroll=self.backtest_config.initial_bankroll,
            final_bankroll=self.backtest_config.initial_bankroll,
            total_return=0.0,
            max_drawdown=0.0,
            total_trades=0,
        )

    def _create_result(self) -> StrategyBacktest:
        """Create the backtest result."""
        # Calculate metrics
        initial = self.backtest_config.initial_bankroll
        final = self.state.bankroll + self.state.realized_pnl
        total_return = (final - initial) / initial if initial > 0 else 0

        # Max drawdown from equity curve
        max_dd = max((e.drawdown for e in self.state.equity_curve), default=0.0)

        # Trade stats
        winning = sum(1 for t in self.state.trades if t.realized_pnl > 0)
        losing = sum(1 for t in self.state.trades if t.realized_pnl < 0)
        hold_times = [
            t.hold_time_minutes
            for t in self.state.trades
            if t.hold_time_minutes is not None
        ]
        avg_hold = sum(hold_times) / len(hold_times) if hold_times else None

        # Per-outcome stats
        per_outcome = self._calculate_per_outcome_stats()

        # Per-fixture stats
        per_fixture = self._calculate_per_fixture_stats()

        # Edge calibration
        edge_calibration = self._calculate_edge_calibration()

        return StrategyBacktest(
            backtest_id=self.backtest_id,
            started_at=self.state.equity_curve[0].timestamp if self.state.equity_curve else utc_now(),
            completed_at=utc_now(),
            status="completed",
            strategy_config_hash=self.strategy_config.config_hash(),
            strategy_config_json=self.strategy_config.to_dict(),
            snapshot_start=self.state.equity_curve[0].timestamp if self.state.equity_curve else None,
            snapshot_end=self.state.equity_curve[-1].timestamp if self.state.equity_curve else None,
            fixtures_included=list(set(t.fixture_id for t in self.state.trades)),
            initial_bankroll=initial,
            final_bankroll=final,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=self._calculate_sharpe(),
            total_trades=len(self.state.trades),
            winning_trades=winning,
            losing_trades=losing,
            avg_hold_time_minutes=avg_hold,
            per_outcome_stats_json=per_outcome,
            per_fixture_stats_json=per_fixture,
            edge_calibration_json=edge_calibration,
            results_json={
                "equity_curve": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "bankroll": e.bankroll,
                        "pnl": e.total_pnl,
                        "drawdown": e.drawdown,
                    }
                    for e in self.state.equity_curve
                ],
                "trades": [
                    {
                        "trade_id": t.trade_id,
                        "ticker": t.ticker,
                        "outcome": t.outcome,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "pnl": t.realized_pnl,
                        "exit_reason": t.exit_reason,
                    }
                    for t in self.state.trades
                ],
            },
        )

    def _calculate_per_outcome_stats(self) -> dict[str, Any]:
        """Calculate statistics per outcome type."""
        stats: dict[str, Any] = {}

        for outcome in ("home_win", "draw", "away_win"):
            trades = [t for t in self.state.trades if t.outcome == outcome]
            if not trades:
                continue

            total_pnl = sum(t.realized_pnl for t in trades)
            winning = sum(1 for t in trades if t.realized_pnl > 0)

            stats[outcome] = {
                "trades": len(trades),
                "winning": winning,
                "losing": len(trades) - winning,
                "win_rate": winning / len(trades) if trades else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(trades) if trades else 0,
            }

        return stats

    def _calculate_per_fixture_stats(self) -> dict[str, Any]:
        """Calculate statistics per fixture."""
        stats: dict[int, Any] = {}

        fixture_ids = set(t.fixture_id for t in self.state.trades)
        for fid in fixture_ids:
            trades = [t for t in self.state.trades if t.fixture_id == fid]
            total_pnl = sum(t.realized_pnl for t in trades)

            stats[fid] = {
                "trades": len(trades),
                "total_pnl": total_pnl,
            }

        return {str(k): v for k, v in stats.items()}

    def _calculate_edge_calibration(self) -> dict[str, Any]:
        """Calculate edge vs realized returns calibration.

        Groups trades by entry edge bucket and compares to realized returns.
        """
        buckets: dict[str, list[float]] = {
            "0.00-0.05": [],
            "0.05-0.10": [],
            "0.10-0.15": [],
            "0.15-0.20": [],
            "0.20+": [],
        }

        for trade in self.state.trades:
            if trade.entry_edge is None or trade.return_pct is None:
                continue

            edge = trade.entry_edge
            if edge < 0.05:
                buckets["0.00-0.05"].append(trade.return_pct)
            elif edge < 0.10:
                buckets["0.05-0.10"].append(trade.return_pct)
            elif edge < 0.15:
                buckets["0.10-0.15"].append(trade.return_pct)
            elif edge < 0.20:
                buckets["0.15-0.20"].append(trade.return_pct)
            else:
                buckets["0.20+"].append(trade.return_pct)

        calibration = {}
        for bucket, returns in buckets.items():
            if returns:
                calibration[bucket] = {
                    "count": len(returns),
                    "avg_return": sum(returns) / len(returns),
                    "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                }

        return calibration

    def _calculate_sharpe(self) -> float | None:
        """Calculate Sharpe-like ratio from equity curve returns.

        Uses marked-to-mid returns between equity snapshots.
        """
        if len(self.state.equity_curve) < 2:
            return None

        # Calculate returns between equity snapshots
        returns = []
        for i in range(1, len(self.state.equity_curve)):
            prev = self.state.equity_curve[i - 1]
            curr = self.state.equity_curve[i]
            if prev.bankroll + prev.total_pnl > 0:
                ret = (
                    (curr.bankroll + curr.total_pnl)
                    / (prev.bankroll + prev.total_pnl)
                ) - 1
                returns.append(ret)

        if not returns:
            return None

        # Calculate Sharpe (assuming risk-free rate = 0)
        mean_return = sum(returns) / len(returns)
        if len(returns) < 2:
            return None

        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_return = variance ** 0.5

        if std_return == 0:
            return None

        # Annualize assuming ~5 min intervals
        # 365 days * 24 hours * 12 intervals = ~105,120 intervals/year
        # Sharpe = mean / std * sqrt(intervals_per_year)
        intervals_per_year = 365 * 24 * (60 / self.backtest_config.equity_snapshot_interval_minutes)
        sharpe = mean_return / std_return * (intervals_per_year ** 0.5)

        return sharpe

    def get_trades(self) -> list[BacktestTrade]:
        """Get all trades from the backtest."""
        return self.state.trades

    def get_equity_curve(self) -> list[BacktestEquity]:
        """Get equity curve from the backtest."""
        return self.state.equity_curve
