"""Paper Trading Simulator.

Simulates order execution and tracks positions and P&L for paper trading mode.
Uses orderbook snapshots to simulate fills with configurable slippage and
fill probability.
"""

import random
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.kalshi.interfaces import OrderbookData
from footbe_trader.strategy.decision_record import (
    AgentRunSummary,
    DecisionAction,
    DecisionRecord,
    OrderParams,
    PositionState,
)
from footbe_trader.strategy.trading_strategy import StrategyConfig

logger = get_logger(__name__)


@dataclass
class PaperOrder:
    """Paper trading order."""

    order_id: str
    run_id: int | None = None
    decision_id: str = ""
    ticker: str = ""
    side: str = ""  # "yes" or "no"
    action: str = ""  # "buy" or "sell"
    order_type: str = "limit"
    price: float = 0.0
    quantity: int = 0
    filled_quantity: int = 0
    status: str = "pending"  # "pending", "filled", "partial", "rejected"
    fill_price: float = 0.0
    fees: float = 0.0
    created_at: datetime = field(default_factory=utc_now)
    filled_at: datetime | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "run_id": self.run_id,
            "decision_id": self.decision_id,
            "ticker": self.ticker,
            "side": self.side,
            "action": self.action,
            "order_type": self.order_type,
            "price": self.price,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "status": self.status,
            "fill_price": self.fill_price,
            "fees": self.fees,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class PaperFill:
    """Paper trading fill record."""

    fill_id: str
    order_id: str
    ticker: str
    side: str
    action: str
    price: float
    quantity: int
    fees: float = 0.0
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "ticker": self.ticker,
            "side": self.side,
            "action": self.action,
            "price": self.price,
            "quantity": self.quantity,
            "fees": self.fees,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PaperPosition:
    """Paper trading position."""

    ticker: str
    outcome: str = ""  # "home_win", "draw", "away_win"
    fixture_id: int | None = None
    quantity: int = 0
    average_entry_price: float = 0.0
    total_cost: float = 0.0  # Total spent acquiring position
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    mark_price: float = 0.0  # Last mark price
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    opened_at: datetime | None = None  # When position was first opened

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def is_open(self) -> bool:
        """Check if position is open (has quantity)."""
        return self.quantity > 0

    def to_state(self) -> PositionState:
        """Convert to PositionState for decision records."""
        return PositionState(
            ticker=self.ticker,
            quantity=self.quantity,
            average_entry_price=self.average_entry_price,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            total_cost=self.total_cost,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "outcome": self.outcome,
            "fixture_id": self.fixture_id,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "total_cost": self.total_cost,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "mark_price": self.mark_price,
            "total_pnl": self.total_pnl,
            "is_open": self.is_open,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class PnlSnapshot:
    """Point-in-time P&L snapshot."""

    snapshot_id: str = ""
    run_id: int | None = None
    timestamp: datetime = field(default_factory=utc_now)
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_exposure: float = 0.0
    position_count: int = 0
    bankroll: float = 0.0
    positions_snapshot: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_exposure": self.total_exposure,
            "position_count": self.position_count,
            "bankroll": self.bankroll,
            "positions_snapshot": self.positions_snapshot,
        }


class PaperTradingSimulator:
    """Simulates order execution and tracks positions for paper trading.

    This simulator:
    1. Receives order requests from the strategy
    2. Simulates fills based on orderbook data with configurable slippage
    3. Tracks positions with average entry prices
    4. Calculates realized and unrealized P&L
    5. Supports position marking to market
    """

    def __init__(
        self,
        config: StrategyConfig | None = None,
        initial_bankroll: float | None = None,
        run_id: int | None = None,
    ):
        """Initialize simulator.

        Args:
            config: Strategy configuration with paper trading settings.
            initial_bankroll: Starting bankroll. If None, uses config value.
            run_id: Current run ID for tracking.
        """
        self.config = config or StrategyConfig.from_yaml()
        self.initial_bankroll = initial_bankroll or self.config.initial_bankroll
        self.run_id = run_id

        # State
        self._positions: dict[str, PaperPosition] = {}  # ticker -> position
        self._orders: list[PaperOrder] = []
        self._fills: list[PaperFill] = []
        self._pnl_snapshots: list[PnlSnapshot] = []
        self._bankroll = self.initial_bankroll
        self._total_realized_pnl: float = 0.0

        # Random seed for reproducibility in tests
        self._rng = random.Random()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible fills."""
        self._rng = random.Random(seed)

    def set_run_id(self, run_id: int | None) -> None:
        """Set current run ID."""
        self.run_id = run_id

    @property
    def positions(self) -> dict[str, PaperPosition]:
        """Get all positions (open and closed)."""
        return self._positions.copy()

    @property
    def open_positions(self) -> dict[str, PaperPosition]:
        """Get only open positions."""
        return {k: v for k, v in self._positions.items() if v.is_open}

    @property
    def orders(self) -> list[PaperOrder]:
        """Get all orders."""
        return self._orders.copy()

    @property
    def fills(self) -> list[PaperFill]:
        """Get all fills."""
        return self._fills.copy()

    @property
    def bankroll(self) -> float:
        """Get current bankroll."""
        return self._bankroll

    @property
    def total_realized_pnl(self) -> float:
        """Get total realized P&L across all positions."""
        return self._total_realized_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all open positions."""
        return sum(p.unrealized_pnl for p in self._positions.values() if p.is_open)

    @property
    def total_exposure(self) -> float:
        """Get total exposure (cost basis of open positions)."""
        return sum(p.total_cost for p in self._positions.values() if p.is_open)

    def get_position(self, ticker: str) -> PaperPosition | None:
        """Get position for a ticker."""
        return self._positions.get(ticker)

    def get_position_quantity(self, ticker: str) -> int:
        """Get quantity for a ticker (0 if no position)."""
        pos = self._positions.get(ticker)
        return pos.quantity if pos else 0

    def get_position_entry_price(self, ticker: str) -> float:
        """Get average entry price for a ticker (0 if no position)."""
        pos = self._positions.get(ticker)
        return pos.average_entry_price if pos else 0.0

    def execute_decision(
        self,
        decision: DecisionRecord,
        orderbook: OrderbookData | None = None,
    ) -> PaperOrder | None:
        """Execute a trading decision.

        Args:
            decision: Decision record with order parameters.
            orderbook: Current orderbook for fill simulation.

        Returns:
            Paper order if order was created, None otherwise.
        """
        # Only execute BUY and EXIT actions
        if decision.action not in (DecisionAction.BUY, DecisionAction.EXIT):
            return None

        if decision.order_params is None:
            return None

        return self.submit_order(
            order_params=decision.order_params,
            decision_id=decision.decision_id,
            orderbook=orderbook,
            outcome=decision.outcome,
            fixture_id=decision.fixture_id,
        )

    def submit_order(
        self,
        order_params: OrderParams,
        decision_id: str = "",
        orderbook: OrderbookData | None = None,
        outcome: str = "",
        fixture_id: int | None = None,
    ) -> PaperOrder:
        """Submit an order and simulate fill.

        Args:
            order_params: Order parameters.
            decision_id: Associated decision ID.
            orderbook: Current orderbook for fill simulation.
            outcome: Outcome type for position tracking.
            fixture_id: Associated fixture ID.

        Returns:
            Paper order with fill status.
        """
        order = PaperOrder(
            order_id=str(uuid.uuid4()),
            run_id=self.run_id,
            decision_id=decision_id,
            ticker=order_params.ticker,
            side=order_params.side,
            action=order_params.action,
            order_type=order_params.order_type,
            price=order_params.price,
            quantity=order_params.quantity,
        )

        self._orders.append(order)

        # Simulate fill
        self._simulate_fill(order, orderbook, outcome, fixture_id)

        return order

    def _simulate_fill(
        self,
        order: PaperOrder,
        orderbook: OrderbookData | None,
        outcome: str,
        fixture_id: int | None,
    ) -> None:
        """Simulate order fill based on orderbook."""
        # Check fill probability
        if self._rng.random() > self.config.fill_probability:
            order.status = "rejected"
            order.rejection_reason = "Simulated fill rejection (random)"
            logger.info(
                "paper_order_rejected",
                order_id=order.order_id,
                ticker=order.ticker,
                reason="random_rejection",
            )
            return

        # Determine fill price with slippage
        if order.action == "buy":
            # Buying: use ask price + slippage
            if orderbook and orderbook.best_yes_ask is not None:
                fill_price = orderbook.best_yes_ask + self.config.slippage_cents
            else:
                fill_price = order.price + self.config.slippage_cents
        else:
            # Selling: use bid price - slippage
            if orderbook and orderbook.best_yes_bid is not None:
                fill_price = orderbook.best_yes_bid - self.config.slippage_cents
            else:
                fill_price = order.price - self.config.slippage_cents

        # Ensure fill price is valid
        fill_price = max(0.01, min(0.99, fill_price))

        # Determine fill quantity (partial fills)
        min_fill_frac, max_fill_frac = self.config.partial_fill_range
        fill_fraction = self._rng.uniform(min_fill_frac, max_fill_frac)
        fill_quantity = max(1, int(order.quantity * fill_fraction))

        # Check available volume if we have orderbook
        if orderbook:
            if order.action == "buy":
                available = orderbook.total_ask_volume
            else:
                available = orderbook.total_bid_volume
            fill_quantity = min(fill_quantity, available)

        if fill_quantity <= 0:
            order.status = "rejected"
            order.rejection_reason = "Insufficient liquidity"
            return

        # Simulate fees (Kalshi charges ~$0.03-0.05 per contract)
        fees = fill_quantity * 0.04

        # Update order
        order.filled_quantity = fill_quantity
        order.fill_price = fill_price
        order.fees = fees
        order.filled_at = datetime.now(UTC)
        order.status = "filled" if fill_quantity >= order.quantity else "partial"

        # Create fill record
        fill = PaperFill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            action=order.action,
            price=fill_price,
            quantity=fill_quantity,
            fees=fees,
        )
        self._fills.append(fill)

        # Update position
        self._update_position(
            ticker=order.ticker,
            action=order.action,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fees=fees,
            outcome=outcome,
            fixture_id=fixture_id,
        )

        logger.info(
            "paper_order_filled",
            order_id=order.order_id,
            ticker=order.ticker,
            action=order.action,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fees=fees,
        )

    def _update_position(
        self,
        ticker: str,
        action: str,
        fill_price: float,
        fill_quantity: int,
        fees: float,
        outcome: str,
        fixture_id: int | None,
    ) -> None:
        """Update position after a fill."""
        position = self._positions.get(ticker)

        if position is None:
            position = PaperPosition(
                ticker=ticker,
                outcome=outcome,
                fixture_id=fixture_id,
                opened_at=utc_now(),  # Set when position is first created
            )
            self._positions[ticker] = position

        if action == "buy":
            # Add to position
            old_cost = position.total_cost
            new_cost = fill_price * fill_quantity + fees
            total_cost = old_cost + new_cost
            total_quantity = position.quantity + fill_quantity

            # Set opened_at if this is the first fill
            if position.quantity == 0 and total_quantity > 0 and position.opened_at is None:
                position.opened_at = utc_now()

            if total_quantity > 0:
                position.average_entry_price = total_cost / total_quantity
            position.quantity = total_quantity
            position.total_cost = total_cost

            # Deduct from bankroll
            self._bankroll -= new_cost

        else:  # sell
            # Close (part of) position
            if position.quantity <= 0:
                logger.warning(
                    "paper_sell_no_position",
                    ticker=ticker,
                    action=action,
                )
                return

            # Calculate realized P&L
            sell_quantity = min(fill_quantity, position.quantity)
            cost_basis = position.average_entry_price * sell_quantity
            proceeds = fill_price * sell_quantity - fees
            realized_pnl = proceeds - cost_basis

            # Update position
            position.quantity -= sell_quantity
            position.total_cost -= cost_basis
            position.realized_pnl += realized_pnl

            # Update totals
            self._total_realized_pnl += realized_pnl
            self._bankroll += proceeds

            if position.quantity <= 0:
                position.quantity = 0
                position.total_cost = 0
                position.average_entry_price = 0

        position.updated_at = datetime.now(UTC)

    def mark_to_market(
        self,
        ticker: str,
        mark_price: float,
    ) -> None:
        """Mark a position to market price.

        Args:
            ticker: Market ticker.
            mark_price: Current market price (mid price).
        """
        position = self._positions.get(ticker)
        if position is None or position.quantity <= 0:
            return

        position.mark_price = mark_price

        # Unrealized P&L = (mark_price - entry_price) * quantity
        # Subtract estimated exit fees
        estimated_exit_fees = position.quantity * 0.04
        position.unrealized_pnl = (
            (mark_price - position.average_entry_price) * position.quantity
            - estimated_exit_fees
        )
        position.updated_at = datetime.now(UTC)

    def mark_all_positions(
        self,
        prices: dict[str, float],
    ) -> None:
        """Mark all positions to market.

        Args:
            prices: Dictionary of ticker -> mark price.
        """
        for ticker, price in prices.items():
            self.mark_to_market(ticker, price)

    def take_pnl_snapshot(self) -> PnlSnapshot:
        """Take a P&L snapshot of current state."""
        snapshot = PnlSnapshot(
            snapshot_id=str(uuid.uuid4()),
            run_id=self.run_id,
            timestamp=datetime.now(UTC),
            total_realized_pnl=self._total_realized_pnl,
            total_unrealized_pnl=self.total_unrealized_pnl,
            total_pnl=self._total_realized_pnl + self.total_unrealized_pnl,
            total_exposure=self.total_exposure,
            position_count=len(self.open_positions),
            bankroll=self._bankroll,
            positions_snapshot=[p.to_dict() for p in self._positions.values()],
        )
        self._pnl_snapshots.append(snapshot)
        return snapshot

    def get_pnl_snapshots(self) -> list[PnlSnapshot]:
        """Get all P&L snapshots."""
        return self._pnl_snapshots.copy()

    def get_fixture_exposure(self, fixture_id: int) -> float:
        """Get total exposure for a fixture."""
        return sum(
            p.total_cost
            for p in self._positions.values()
            if p.fixture_id == fixture_id and p.is_open
        )

    def settle_position(
        self,
        ticker: str,
        settlement_value: float,
    ) -> float:
        """Settle a position at expiry.

        Args:
            ticker: Market ticker.
            settlement_value: 1.0 if "yes" wins, 0.0 if "no" wins.

        Returns:
            Realized P&L from settlement.
        """
        position = self._positions.get(ticker)
        if position is None or position.quantity <= 0:
            return 0.0

        # Settlement proceeds
        proceeds = position.quantity * settlement_value

        # Cost basis
        cost_basis = position.total_cost

        # Realized P&L
        realized_pnl = proceeds - cost_basis

        # Update position
        position.realized_pnl += realized_pnl
        position.unrealized_pnl = 0
        position.quantity = 0
        position.total_cost = 0
        position.average_entry_price = 0
        position.updated_at = datetime.now(UTC)

        # Update totals
        self._total_realized_pnl += realized_pnl
        self._bankroll += proceeds

        logger.info(
            "paper_position_settled",
            ticker=ticker,
            settlement_value=settlement_value,
            proceeds=proceeds,
            realized_pnl=realized_pnl,
        )

        return realized_pnl

    def update_run_summary(self, summary: AgentRunSummary) -> AgentRunSummary:
        """Update agent run summary with current state.

        Args:
            summary: Run summary to update.

        Returns:
            Updated summary.
        """
        summary.orders_placed = len(self._orders)
        summary.orders_filled = len([o for o in self._orders if o.status in ("filled", "partial")])
        summary.orders_rejected = len([o for o in self._orders if o.status == "rejected"])
        summary.total_realized_pnl = self._total_realized_pnl
        summary.total_unrealized_pnl = self.total_unrealized_pnl
        summary.total_exposure = self.total_exposure
        summary.position_count = len(self.open_positions)

        return summary

    def reset(self) -> None:
        """Reset simulator state."""
        self._positions.clear()
        self._orders.clear()
        self._fills.clear()
        self._pnl_snapshots.clear()
        self._bankroll = self.initial_bankroll
        self._total_realized_pnl = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export state to dictionary."""
        return {
            "bankroll": self._bankroll,
            "initial_bankroll": self.initial_bankroll,
            "total_realized_pnl": self._total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_exposure": self.total_exposure,
            "positions": {k: v.to_dict() for k, v in self._positions.items()},
            "orders": [o.to_dict() for o in self._orders],
            "fills": [f.to_dict() for f in self._fills],
        }
