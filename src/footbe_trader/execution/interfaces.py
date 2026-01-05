"""Execution interfaces and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from footbe_trader.kalshi.interfaces import OrderData
from footbe_trader.strategy.interfaces import Signal


@dataclass
class ExecutionResult:
    """Result of order execution attempt."""

    signal: Signal
    success: bool
    order: OrderData | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class IExecutor(ABC):
    """Interface for order execution."""

    @abstractmethod
    async def execute(self, signal: Signal) -> ExecutionResult:
        """Execute a trading signal.

        Args:
            signal: Trading signal to execute.

        Returns:
            Execution result.
        """
        ...

    @abstractmethod
    async def cancel_all_orders(self, market_id: str | None = None) -> int:
        """Cancel all open orders.

        Args:
            market_id: Optional market ID to filter by.

        Returns:
            Number of orders cancelled.
        """
        ...

    @property
    @abstractmethod
    def dry_run(self) -> bool:
        """Check if executor is in dry-run mode."""
        ...
