"""Decision record types for trading agent.

A DecisionRecord captures the full state and rationale for each trading
decision made by the agent, whether a trade was placed or not.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from footbe_trader.common.time_utils import utc_now


class DecisionAction(str, Enum):
    """Action taken by the strategy."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    SKIP = "skip"  # Did not trade due to filters


class ExitReason(str, Enum):
    """Reason for exiting a position."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    EDGE_FLIP = "edge_flip"  # Edge went negative
    MARKET_CLOSE = "market_close"  # Approaching expiry
    MANUAL = "manual"  # Manual exit request


@dataclass
class MarketSnapshot:
    """Snapshot of market state at decision time."""

    ticker: str
    best_bid: float | None = None
    best_ask: float | None = None
    mid_price: float | None = None
    spread: float | None = None
    bid_volume: int = 0
    ask_volume: int = 0
    total_volume: int = 0
    last_price: float | None = None
    status: str = "open"
    close_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "total_volume": self.total_volume,
            "last_price": self.last_price,
            "status": self.status,
            "close_time": self.close_time.isoformat() if self.close_time else None,
        }


@dataclass
class ModelPrediction:
    """Model prediction snapshot for decision record."""

    model_name: str
    model_version: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    confidence: float = 1.0
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prob_home_win": self.prob_home_win,
            "prob_draw": self.prob_draw,
            "prob_away_win": self.prob_away_win,
            "confidence": self.confidence,
            "features": self.features,
        }


@dataclass
class EdgeCalculation:
    """Edge calculation for a specific outcome."""

    outcome: str  # "home_win", "draw", "away_win"
    model_prob: float
    market_price: float  # Ask price for buying
    edge: float  # model_prob - market_price
    is_tradeable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.outcome,
            "model_prob": self.model_prob,
            "market_price": self.market_price,
            "edge": self.edge,
            "is_tradeable": self.is_tradeable,
        }


@dataclass
class KellySizing:
    """Kelly criterion position sizing calculation."""

    edge: float
    win_prob: float
    odds: float  # 1/market_price - 1 (profit ratio if win)
    kelly_fraction: float  # Full Kelly: edge / (odds * (1 - win_prob))
    adjusted_fraction: float  # After applying fraction multiplier
    position_size: int  # Final position in contracts
    capped_reason: str | None = None  # Why position was capped

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge": self.edge,
            "win_prob": self.win_prob,
            "odds": self.odds,
            "kelly_fraction": self.kelly_fraction,
            "adjusted_fraction": self.adjusted_fraction,
            "position_size": self.position_size,
            "capped_reason": self.capped_reason,
        }


@dataclass
class OrderParams:
    """Parameters for order to be placed."""

    ticker: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    order_type: str  # "limit" or "market"
    price: float
    quantity: int
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra params like expiration_ts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "side": self.side,
            "action": self.action,
            "order_type": self.order_type,
            "price": self.price,
            "quantity": self.quantity,
            "metadata": self.metadata,
        }


@dataclass
class PositionState:
    """Current position state in a market."""

    ticker: str
    quantity: int = 0
    average_entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_cost": self.total_cost,
        }


@dataclass
class DecisionRecord:
    """Complete record of a trading decision.

    This captures all inputs, calculations, and outputs for a single
    decision on a single market outcome. Even "no trade" decisions are
    recorded for analysis and debugging.
    """

    # Identifiers
    decision_id: str = ""
    run_id: int | None = None
    fixture_id: int | None = None
    market_ticker: str = ""
    outcome: str = ""  # "home_win", "draw", "away_win"
    timestamp: datetime = field(default_factory=utc_now)

    # Inputs
    market_snapshot: MarketSnapshot | None = None
    model_prediction: ModelPrediction | None = None
    current_position: PositionState | None = None

    # Calculations
    edge_calculation: EdgeCalculation | None = None
    kelly_sizing: KellySizing | None = None

    # Decision
    action: DecisionAction = DecisionAction.SKIP
    exit_reason: ExitReason | None = None
    order_params: OrderParams | None = None

    # Rationale
    rationale: str = ""  # Human-readable explanation
    filters_passed: dict[str, bool] = field(default_factory=dict)
    rejection_reason: str | None = None

    # Execution (filled in after order attempt)
    order_placed: bool = False
    order_id: str | None = None
    execution_error: str | None = None

    # Strategy tracking (for multi-armed bandit)
    strategy_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "fixture_id": self.fixture_id,
            "market_ticker": self.market_ticker,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "market_snapshot": self.market_snapshot.to_dict()
            if self.market_snapshot
            else None,
            "model_prediction": self.model_prediction.to_dict()
            if self.model_prediction
            else None,
            "current_position": self.current_position.to_dict()
            if self.current_position
            else None,
            "edge_calculation": self.edge_calculation.to_dict()
            if self.edge_calculation
            else None,
            "kelly_sizing": self.kelly_sizing.to_dict()
            if self.kelly_sizing
            else None,
            "action": self.action.value,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "order_params": self.order_params.to_dict()
            if self.order_params
            else None,
            "rationale": self.rationale,
            "filters_passed": self.filters_passed,
            "rejection_reason": self.rejection_reason,
            "order_placed": self.order_placed,
            "order_id": self.order_id,
            "execution_error": self.execution_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        """Create from dictionary."""
        record = cls(
            decision_id=data.get("decision_id", ""),
            run_id=data.get("run_id"),
            fixture_id=data.get("fixture_id"),
            market_ticker=data.get("market_ticker", ""),
            outcome=data.get("outcome", ""),
            rationale=data.get("rationale", ""),
            filters_passed=data.get("filters_passed", {}),
            rejection_reason=data.get("rejection_reason"),
            order_placed=data.get("order_placed", False),
            order_id=data.get("order_id"),
            execution_error=data.get("execution_error"),
        )

        if data.get("action"):
            record.action = DecisionAction(data["action"])

        if data.get("exit_reason"):
            record.exit_reason = ExitReason(data["exit_reason"])

        return record


@dataclass
class AgentRunSummary:
    """Summary of an agent trading loop run."""

    run_id: int | None = None
    run_type: str = "paper"  # "paper", "live"
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    status: str = "running"  # "running", "completed", "failed"

    # Counters
    fixtures_evaluated: int = 0
    markets_evaluated: int = 0
    decisions_made: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0

    # P&L
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_exposure: float = 0.0
    position_count: int = 0

    # Errors
    error_count: int = 0
    error_message: str | None = None

    # Config
    config_hash: str = ""
    config_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "status": self.status,
            "fixtures_evaluated": self.fixtures_evaluated,
            "markets_evaluated": self.markets_evaluated,
            "decisions_made": self.decisions_made,
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_exposure": self.total_exposure,
            "position_count": self.position_count,
            "error_count": self.error_count,
            "error_message": self.error_message,
            "config_hash": self.config_hash,
            "config_summary": self.config_summary,
        }
