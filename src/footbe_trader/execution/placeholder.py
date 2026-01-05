"""Placeholder executor implementation."""

from footbe_trader.common.logging import get_logger
from footbe_trader.execution.interfaces import ExecutionResult, IExecutor
from footbe_trader.kalshi.interfaces import IKalshiClient
from footbe_trader.strategy.interfaces import Signal

logger = get_logger(__name__)


class PlaceholderExecutor(IExecutor):
    """Placeholder executor that logs but doesn't trade.

    This is a stub implementation for testing the pipeline.
    Real implementation would:
    - Validate signals
    - Apply position limits
    - Place orders via Kalshi client
    - Track order status
    """

    def __init__(
        self,
        kalshi_client: IKalshiClient,
        dry_run_mode: bool = True,
    ) -> None:
        """Initialize placeholder executor.

        Args:
            kalshi_client: Kalshi API client.
            dry_run_mode: If True, log but don't execute.
        """
        self._kalshi_client = kalshi_client
        self._dry_run = dry_run_mode

    @property
    def dry_run(self) -> bool:
        """Check if executor is in dry-run mode."""
        return self._dry_run

    async def execute(self, signal: Signal) -> ExecutionResult:
        """Execute a trading signal (placeholder logs only).

        Args:
            signal: Trading signal to execute.

        Returns:
            Execution result (always unsuccessful in placeholder).
        """
        logger.info(
            "placeholder_execute",
            market_id=signal.market_id,
            side=signal.side,
            action=signal.action,
            price=signal.target_price,
            quantity=signal.quantity,
            edge=signal.edge,
            dry_run=self._dry_run,
        )

        if self._dry_run:
            return ExecutionResult(
                signal=signal,
                success=False,
                order=None,
                error_message="Dry run mode - no execution",
                metadata={"dry_run": True},
            )

        # Placeholder: Would place real order here
        # order = await self._kalshi_client.place_order(...)

        return ExecutionResult(
            signal=signal,
            success=False,
            order=None,
            error_message="Placeholder executor - no real trading",
        )

    async def cancel_all_orders(self, market_id: str | None = None) -> int:
        """Cancel all orders (placeholder does nothing).

        Args:
            market_id: Optional market ID to filter by.

        Returns:
            0 (no orders cancelled in placeholder).
        """
        logger.info(
            "placeholder_cancel_all_orders",
            market_id=market_id,
            dry_run=self._dry_run,
        )

        # Placeholder: No orders to cancel
        return 0
