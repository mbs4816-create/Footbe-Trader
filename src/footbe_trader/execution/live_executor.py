"""Live Trading Executor with Safety Mechanisms.

Provides safe execution of trades with:
- Dry-run mode (prints orders without sending)
- Kill-switch integration
- Alert system integration
- Configuration verification
"""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.execution.alerts import AlertConfig, AlertManager
from footbe_trader.execution.interfaces import ExecutionResult, IExecutor
from footbe_trader.execution.kill_switch import (
    KillSwitchConfig,
    KillSwitchManager,
    KillSwitchReason,
)
from footbe_trader.kalshi.interfaces import (
    IKalshiTradingClient,
    OrderData,
)
from footbe_trader.strategy.interfaces import Signal

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Trading execution mode."""

    DRY_RUN = "dry_run"  # Print orders but don't send
    PAPER = "paper"  # Paper trading simulation
    LIVE = "live"  # Real trading


@dataclass
class LiveExecutorConfig:
    """Configuration for live executor."""

    # Execution mode
    mode: ExecutionMode = ExecutionMode.DRY_RUN

    # Safety flags
    enable_live_trading: bool = False
    require_environment_confirmation: bool = True
    environment_name: str = "production"

    # Order defaults
    default_side: str = "yes"
    max_order_quantity: int = 100
    max_order_value: float = 1000.0

    # Logging
    log_all_orders: bool = True
    log_dry_run_orders: bool = True

    # Kill-switch config
    kill_switch_config: KillSwitchConfig = field(default_factory=KillSwitchConfig)

    # Alert config
    alert_config: AlertConfig = field(default_factory=AlertConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "enable_live_trading": self.enable_live_trading,
            "require_environment_confirmation": self.require_environment_confirmation,
            "environment_name": self.environment_name,
            "default_side": self.default_side,
            "max_order_quantity": self.max_order_quantity,
            "max_order_value": self.max_order_value,
            "kill_switch_config": self.kill_switch_config.to_dict(),
            "alert_config": self.alert_config.to_dict(),
        }


class LiveTradingNotEnabledError(Exception):
    """Raised when live trading is attempted but not enabled."""

    pass


class KillSwitchTrippedError(Exception):
    """Raised when trading is blocked by a kill-switch."""

    def __init__(self, reason: KillSwitchReason, message: str = ""):
        self.reason = reason
        super().__init__(message or f"Kill-switch tripped: {reason.value}")


class LiveExecutor(IExecutor):
    """Live trading executor with safety mechanisms.

    This executor provides safe trade execution with multiple layers of protection:

    1. Mode Selection:
       - DRY_RUN: Logs orders but doesn't send them
       - PAPER: Uses paper trading simulation
       - LIVE: Actually places orders on Kalshi

    2. Safety Checks:
       - Requires explicit enable_live_trading=true in config
       - Requires environment confirmation (FOOTBE_ENABLE_LIVE_TRADING=true)
       - Integrates with kill-switch system
       - Validates order parameters

    3. Alerting:
       - Alerts on every trade execution
       - Alerts on kill-switch trips
       - Alerts on errors

    Usage:
        # Dry-run mode (safe for testing)
        config = LiveExecutorConfig(mode=ExecutionMode.DRY_RUN)
        executor = LiveExecutor(kalshi_client, config)

        # Execute signals
        result = await executor.execute(signal)

        # Live mode (requires explicit flags)
        config = LiveExecutorConfig(
            mode=ExecutionMode.LIVE,
            enable_live_trading=True,
        )
        # Also requires: FOOTBE_ENABLE_LIVE_TRADING=true environment variable
    """

    def __init__(
        self,
        kalshi_client: IKalshiTradingClient,
        config: LiveExecutorConfig | None = None,
    ):
        """Initialize live executor.

        Args:
            kalshi_client: Kalshi trading client.
            config: Executor configuration.
        """
        self.kalshi_client = kalshi_client
        self.config = config or LiveExecutorConfig()

        # Initialize kill-switch manager
        self.kill_switch = KillSwitchManager(self.config.kill_switch_config)

        # Initialize alert manager
        self.alerts = AlertManager(self.config.alert_config)

        # State
        self._started = False
        self._execution_count = 0
        self._session_id = str(uuid.uuid4())

        # Validate configuration
        self._validate_config()

        logger.info(
            "live_executor_initialized",
            mode=self.config.mode.value,
            session_id=self._session_id,
        )

    def _validate_config(self) -> None:
        """Validate configuration for live trading."""
        if self.config.mode == ExecutionMode.LIVE:
            if not self.config.enable_live_trading:
                raise LiveTradingNotEnabledError(
                    "Live trading requires enable_live_trading=true in config"
                )

            if self.config.require_environment_confirmation:
                env_flag = os.environ.get("FOOTBE_ENABLE_LIVE_TRADING", "").lower()
                if env_flag != "true":
                    raise LiveTradingNotEnabledError(
                        "Live trading requires FOOTBE_ENABLE_LIVE_TRADING=true "
                        "environment variable"
                    )

            logger.warning(
                "live_trading_enabled",
                environment=self.config.environment_name,
                session_id=self._session_id,
            )

    @property
    def dry_run(self) -> bool:
        """Check if executor is in dry-run mode."""
        return self.config.mode == ExecutionMode.DRY_RUN

    @property
    def is_live(self) -> bool:
        """Check if executor is in live mode."""
        return self.config.mode == ExecutionMode.LIVE

    def start(self) -> None:
        """Start the executor."""
        if self._started:
            return

        self._started = True
        self.alerts.alert_trading_started(
            mode=self.config.mode.value,
            config=self.config.to_dict(),
            session_id=self._session_id,
        )

        logger.info(
            "executor_started",
            mode=self.config.mode.value,
            session_id=self._session_id,
        )

    def stop(self, reason: str = "manual") -> None:
        """Stop the executor.

        Args:
            reason: Reason for stopping.
        """
        if not self._started:
            return

        self._started = False
        self.alerts.alert_trading_stopped(
            reason=reason,
            session_id=self._session_id,
            execution_count=self._execution_count,
        )

        logger.info(
            "executor_stopped",
            reason=reason,
            session_id=self._session_id,
            execution_count=self._execution_count,
        )

    async def execute(self, signal: Signal) -> ExecutionResult:
        """Execute a trading signal.

        Args:
            signal: Trading signal to execute.

        Returns:
            Execution result.
        """
        # Check kill-switches
        if not self.kill_switch.can_trade():
            reason = self.kill_switch.get_trip_reason()
            self.alerts.alert_kill_switch_tripped(
                reason=reason.value if reason else "unknown",
                message="Trade blocked by kill-switch",
                signal=signal.to_dict() if hasattr(signal, 'to_dict') else str(signal),
            )
            return ExecutionResult(
                signal=signal,
                success=False,
                error_message=f"Kill-switch tripped: {reason.value if reason else 'unknown'}",
            )

        # Validate signal
        validation_error = self._validate_signal(signal)
        if validation_error:
            return ExecutionResult(
                signal=signal,
                success=False,
                error_message=validation_error,
            )

        # Execute based on mode
        if self.config.mode == ExecutionMode.DRY_RUN:
            return await self._execute_dry_run(signal)
        elif self.config.mode == ExecutionMode.PAPER:
            return await self._execute_paper(signal)
        else:
            return await self._execute_live(signal)

    async def _execute_dry_run(self, signal: Signal) -> ExecutionResult:
        """Execute signal in dry-run mode (log only).

        Args:
            signal: Trading signal.

        Returns:
            Simulated execution result.
        """
        # Generate fake order ID
        order_id = f"DRY-{uuid.uuid4().hex[:8]}"

        # Log the order
        if self.config.log_dry_run_orders:
            logger.info(
                "dry_run_order",
                order_id=order_id,
                ticker=signal.market_id,
                side=self.config.default_side,
                action="buy" if signal.direction == "long" else "sell",
                price=signal.entry_price,
                quantity=signal.quantity,
            )

        # Alert
        self.alerts.alert_order_placed(
            ticker=signal.market_id,
            side=self.config.default_side,
            action="buy" if signal.direction == "long" else "sell",
            price=signal.entry_price,
            quantity=signal.quantity,
            order_id=order_id,
            dry_run=True,
        )

        # Create fake order data
        order = OrderData(
            order_id=order_id,
            ticker=signal.market_id,
            side=self.config.default_side,
            action="buy" if signal.direction == "long" else "sell",
            order_type="limit",
            price=signal.entry_price,
            quantity=signal.quantity,
            filled_quantity=signal.quantity,
            remaining_quantity=0,
            status="executed",
            created_time=utc_now(),
        )

        self._execution_count += 1

        return ExecutionResult(
            signal=signal,
            success=True,
            order=order,
            metadata={"dry_run": True},
        )

    async def _execute_paper(self, signal: Signal) -> ExecutionResult:
        """Execute signal in paper trading mode.

        Args:
            signal: Trading signal.

        Returns:
            Paper execution result.
        """
        # Paper trading would use the PaperTradingSimulator
        # For now, treat same as dry-run
        return await self._execute_dry_run(signal)

    async def _execute_live(self, signal: Signal) -> ExecutionResult:
        """Execute signal in live mode.

        Args:
            signal: Trading signal.

        Returns:
            Live execution result.
        """
        try:
            # Record API call attempt
            self.kill_switch.record_api_call(success=True)

            # Determine order parameters
            side = self.config.default_side
            action = "buy" if signal.direction == "long" else "sell"
            price = signal.entry_price
            quantity = min(signal.quantity, self.config.max_order_quantity)

            # Place order
            order = await self.kalshi_client.place_limit_order(
                ticker=signal.market_id,
                side=side,
                action=action,
                price=price,
                quantity=quantity,
            )

            # Alert on success
            self.alerts.alert_trade_executed(
                ticker=signal.market_id,
                side=side,
                action=action,
                price=order.price,
                quantity=order.filled_quantity,
                order_id=order.order_id,
            )

            # Update kill-switch state
            exposure_delta = order.price * order.filled_quantity
            self.kill_switch.record_trade(exposure_delta=exposure_delta)

            self._execution_count += 1

            logger.info(
                "live_order_executed",
                order_id=order.order_id,
                ticker=signal.market_id,
                status=order.status,
                filled=order.filled_quantity,
            )

            return ExecutionResult(
                signal=signal,
                success=True,
                order=order,
            )

        except Exception as e:
            # Record API failure
            self.kill_switch.record_api_call(success=False, error=str(e))

            # Alert on failure
            self.alerts.alert_order_failed(
                ticker=signal.market_id,
                error=str(e),
            )

            logger.error(
                "live_order_failed",
                ticker=signal.market_id,
                error=str(e),
            )

            return ExecutionResult(
                signal=signal,
                success=False,
                error_message=str(e),
            )

    async def cancel_all_orders(self, market_id: str | None = None) -> int:
        """Cancel all open orders.

        Args:
            market_id: Optional market ID to filter by.

        Returns:
            Number of orders cancelled.
        """
        if self.config.mode == ExecutionMode.DRY_RUN:
            logger.info("dry_run_cancel_all_orders", market_id=market_id)
            return 0

        try:
            # Get open orders
            orders, _ = await self.kalshi_client.list_orders(
                ticker=market_id,
                status="resting",
            )

            # Cancel each order
            cancelled = 0
            for order in orders:
                if await self.kalshi_client.cancel_order(order.order_id):
                    cancelled += 1

            logger.info(
                "orders_cancelled",
                count=cancelled,
                market_id=market_id,
            )

            return cancelled

        except Exception as e:
            logger.error("cancel_orders_failed", error=str(e))
            return 0

    def _validate_signal(self, signal: Signal) -> str | None:
        """Validate a trading signal.

        Args:
            signal: Signal to validate.

        Returns:
            Error message if invalid, None if valid.
        """
        if not signal.market_id:
            return "Signal missing market_id"

        if signal.quantity <= 0:
            return f"Invalid quantity: {signal.quantity}"

        if signal.quantity > self.config.max_order_quantity:
            return f"Quantity {signal.quantity} exceeds max {self.config.max_order_quantity}"

        order_value = signal.entry_price * signal.quantity
        if order_value > self.config.max_order_value:
            return f"Order value ${order_value:.2f} exceeds max ${self.config.max_order_value:.2f}"

        if not 0.01 <= signal.entry_price <= 0.99:
            return f"Invalid price: {signal.entry_price}"

        return None

    # --- Model Health Integration ---

    def check_model_health(self, probabilities: dict[str, float]) -> bool:
        """Check model output health before trading.

        Args:
            probabilities: Model output probabilities.

        Returns:
            True if healthy, False if unhealthy.
        """
        is_healthy = self.kill_switch.validate_model_health(probabilities)

        if not is_healthy:
            tripped_switches = self.kill_switch.get_tripped_switches()
            for switch in tripped_switches:
                if switch.reason == KillSwitchReason.MODEL_HEALTH:
                    self.alerts.alert_model_error(
                        model_name="prediction_model",
                        error=switch.message,
                        probabilities=probabilities,
                    )

        return is_healthy

    # --- Status Methods ---

    def get_status(self) -> dict[str, Any]:
        """Get executor status.

        Returns:
            Status dictionary.
        """
        return {
            "session_id": self._session_id,
            "mode": self.config.mode.value,
            "started": self._started,
            "execution_count": self._execution_count,
            "kill_switch": self.kill_switch.get_status_report(),
            "config": self.config.to_dict(),
        }

    def get_kill_switch_status(self) -> dict[str, Any]:
        """Get kill-switch status.

        Returns:
            Kill-switch status report.
        """
        return self.kill_switch.get_status_report()

    def reset_kill_switches(self) -> None:
        """Reset all kill-switches (use with caution)."""
        self.kill_switch.reset_all_switches()
        self.alerts.alert_kill_switch_reset(reason="all")
        logger.warning("all_kill_switches_reset", session_id=self._session_id)

    def halt_trading(self, reason: str = "manual") -> None:
        """Manually halt trading.

        Args:
            reason: Reason for halt.
        """
        self.kill_switch.halt_trading(reason)
        self.alerts.alert_kill_switch_tripped(
            reason="manual",
            message=f"Manual halt: {reason}",
        )

    def resume_trading(self) -> bool:
        """Attempt to resume trading.

        Returns:
            True if trading resumed, False if still blocked.
        """
        return self.kill_switch.resume_trading()
