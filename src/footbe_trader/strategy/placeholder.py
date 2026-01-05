"""Placeholder strategy implementation."""

from footbe_trader.common.logging import get_logger
from footbe_trader.strategy.interfaces import IStrategy, Signal, StrategyContext

logger = get_logger(__name__)


class PlaceholderStrategy(IStrategy):
    """Placeholder strategy that generates no signals.

    This is a stub implementation for testing the pipeline.
    Real implementation would:
    - Compare model probabilities to market prices
    - Calculate expected value and edge
    - Apply position limits and risk management
    - Generate buy/sell signals
    """

    def __init__(self, min_edge: float = 0.05) -> None:
        """Initialize placeholder strategy.

        Args:
            min_edge: Minimum edge required to trade.
        """
        self.min_edge = min_edge

    @property
    def name(self) -> str:
        """Strategy name."""
        return "placeholder"

    def generate_signals(self, context: StrategyContext) -> list[Signal]:
        """Generate signals (placeholder returns empty list).

        Args:
            context: Strategy context.

        Returns:
            Empty list (no trading in placeholder).
        """
        logger.debug(
            "placeholder_generate_signals",
            market_id=context.market.external_id,
            num_predictions=len(context.predictions),
        )

        # Placeholder: No signals
        # Real implementation would compare model probs to market prices
        # and generate signals when edge exceeds threshold

        return []

    def should_trade(self, context: StrategyContext) -> bool:
        """Check trading conditions (placeholder returns False).

        Args:
            context: Strategy context.

        Returns:
            False (no trading in placeholder).
        """
        logger.debug(
            "placeholder_should_trade",
            market_id=context.market.external_id,
        )

        # Placeholder: Never trade
        return False
