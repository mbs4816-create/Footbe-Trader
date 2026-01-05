"""Strategy interfaces and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from footbe_trader.kalshi.interfaces import MarketData
from footbe_trader.modeling.interfaces import PredictionResult


@dataclass
class Signal:
    """Trading signal from strategy."""

    market_id: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    target_price: float
    quantity: int
    edge: float  # expected edge (model prob - market prob)
    confidence: float
    reason: str = ""
    direction: str = "long"  # "long" or "short"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_price(self) -> float:
        """Alias for target_price for executor compatibility."""
        return self.target_price

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "market_id": self.market_id,
            "side": self.side,
            "action": self.action,
            "target_price": self.target_price,
            "quantity": self.quantity,
            "edge": self.edge,
            "confidence": self.confidence,
            "reason": self.reason,
            "direction": self.direction,
            "metadata": self.metadata,
        }


@dataclass
class StrategyContext:
    """Context for strategy decision-making."""

    predictions: list[PredictionResult]
    market: MarketData
    current_position: int = 0
    max_position: int = 100
    risk_budget: float = 1000.0


class IStrategy(ABC):
    """Interface for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...

    @abstractmethod
    def generate_signals(self, context: StrategyContext) -> list[Signal]:
        """Generate trading signals.

        Args:
            context: Strategy context with predictions and market data.

        Returns:
            List of trading signals.
        """
        ...

    @abstractmethod
    def should_trade(self, context: StrategyContext) -> bool:
        """Determine if conditions are suitable for trading.

        Args:
            context: Strategy context.

        Returns:
            True if trading conditions are met.
        """
        ...
