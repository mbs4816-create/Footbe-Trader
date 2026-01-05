"""Kill-Switch System for Trading Safety.

Provides circuit breakers and safety limits for live trading:
- Daily loss limit
- Maximum open exposure
- API error rate monitoring
- Model health validation
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now

logger = get_logger(__name__)


class KillSwitchReason(Enum):
    """Reason for kill-switch activation."""

    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_EXPOSURE = "max_exposure"
    API_ERROR_RATE = "api_error_rate"
    MODEL_HEALTH = "model_health"
    MANUAL = "manual"
    CONFIGURATION = "configuration"


class KillSwitchStatus(Enum):
    """Status of kill-switch system."""

    ACTIVE = "active"  # Trading allowed
    TRIPPED = "tripped"  # Trading halted
    MANUAL_HALT = "manual_halt"  # Manually halted
    NOT_CONFIGURED = "not_configured"  # Not properly configured


@dataclass
class KillSwitch:
    """Individual kill-switch state."""

    name: str
    reason: KillSwitchReason
    enabled: bool = True
    tripped: bool = False
    trip_time: datetime | None = None
    trip_value: float | None = None
    threshold: float = 0.0
    message: str = ""

    def trip(self, value: float, message: str = "") -> None:
        """Trip this kill-switch.

        Args:
            value: Current value that caused the trip.
            message: Additional message.
        """
        self.tripped = True
        self.trip_time = utc_now()
        self.trip_value = value
        self.message = message

    def reset(self) -> None:
        """Reset this kill-switch."""
        self.tripped = False
        self.trip_time = None
        self.trip_value = None
        self.message = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "reason": self.reason.value,
            "enabled": self.enabled,
            "tripped": self.tripped,
            "trip_time": self.trip_time.isoformat() if self.trip_time else None,
            "trip_value": self.trip_value,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class KillSwitchConfig:
    """Configuration for kill-switch system."""

    # Daily loss limit
    daily_loss_limit_enabled: bool = True
    daily_loss_limit_dollars: float = 500.0

    # Maximum exposure
    max_exposure_enabled: bool = True
    max_exposure_dollars: float = 2000.0

    # API error rate circuit breaker
    api_error_rate_enabled: bool = True
    api_error_window_seconds: int = 300  # 5 minutes
    api_error_threshold: int = 10  # Errors in window
    api_error_rate_threshold: float = 0.5  # 50% error rate

    # Model health checks
    model_health_enabled: bool = True
    model_prob_sum_tolerance: float = 0.01  # Allow 1% deviation from 1.0

    # Reset settings
    auto_reset_after_hours: float | None = None  # Auto-reset after N hours
    require_manual_reset: bool = True  # Require manual intervention

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "daily_loss_limit": {
                "enabled": self.daily_loss_limit_enabled,
                "limit_dollars": self.daily_loss_limit_dollars,
            },
            "max_exposure": {
                "enabled": self.max_exposure_enabled,
                "limit_dollars": self.max_exposure_dollars,
            },
            "api_error_rate": {
                "enabled": self.api_error_rate_enabled,
                "window_seconds": self.api_error_window_seconds,
                "error_threshold": self.api_error_threshold,
                "rate_threshold": self.api_error_rate_threshold,
            },
            "model_health": {
                "enabled": self.model_health_enabled,
                "prob_sum_tolerance": self.model_prob_sum_tolerance,
            },
            "auto_reset_after_hours": self.auto_reset_after_hours,
            "require_manual_reset": self.require_manual_reset,
        }


@dataclass
class TradingState:
    """Current trading state for kill-switch evaluation."""

    daily_pnl: float = 0.0
    daily_pnl_reset_time: datetime = field(default_factory=utc_now)
    current_exposure: float = 0.0
    open_positions_count: int = 0
    last_model_probs: dict[str, float] = field(default_factory=dict)

    def reset_daily(self) -> None:
        """Reset daily tracking."""
        self.daily_pnl = 0.0
        self.daily_pnl_reset_time = utc_now()


class KillSwitchManager:
    """Manages trading kill-switches for safety.

    The kill-switch manager monitors trading activity and can halt
    trading when safety thresholds are exceeded:

    1. Daily Loss Limit: Halts if daily losses exceed threshold
    2. Max Exposure: Halts if open exposure exceeds limit
    3. API Error Rate: Halts if too many API errors occur
    4. Model Health: Halts if model outputs are invalid

    Usage:
        manager = KillSwitchManager(config)

        # Before each trade
        if manager.can_trade():
            # Execute trade
            ...
        else:
            # Trading halted
            reason = manager.get_trip_reason()

        # Update state after trade
        manager.record_trade(pnl=-50.0)

        # Record API calls
        manager.record_api_call(success=True)
        manager.record_api_call(success=False, error="timeout")

        # Validate model
        if not manager.validate_model_health(probs):
            # Model unhealthy
    """

    def __init__(self, config: KillSwitchConfig | None = None):
        """Initialize kill-switch manager.

        Args:
            config: Kill-switch configuration.
        """
        self.config = config or KillSwitchConfig()
        self.state = TradingState()
        self._manual_halt = False

        # Initialize individual kill-switches
        self.switches: dict[KillSwitchReason, KillSwitch] = {
            KillSwitchReason.DAILY_LOSS_LIMIT: KillSwitch(
                name="Daily Loss Limit",
                reason=KillSwitchReason.DAILY_LOSS_LIMIT,
                enabled=self.config.daily_loss_limit_enabled,
                threshold=self.config.daily_loss_limit_dollars,
            ),
            KillSwitchReason.MAX_EXPOSURE: KillSwitch(
                name="Maximum Exposure",
                reason=KillSwitchReason.MAX_EXPOSURE,
                enabled=self.config.max_exposure_enabled,
                threshold=self.config.max_exposure_dollars,
            ),
            KillSwitchReason.API_ERROR_RATE: KillSwitch(
                name="API Error Rate",
                reason=KillSwitchReason.API_ERROR_RATE,
                enabled=self.config.api_error_rate_enabled,
                threshold=self.config.api_error_threshold,
            ),
            KillSwitchReason.MODEL_HEALTH: KillSwitch(
                name="Model Health",
                reason=KillSwitchReason.MODEL_HEALTH,
                enabled=self.config.model_health_enabled,
                threshold=self.config.model_prob_sum_tolerance,
            ),
        }

        # API call tracking
        self._api_calls: deque[tuple[float, bool]] = deque()  # (timestamp, success)

        logger.info(
            "kill_switch_manager_initialized",
            daily_loss_limit=self.config.daily_loss_limit_dollars,
            max_exposure=self.config.max_exposure_dollars,
        )

    @property
    def status(self) -> KillSwitchStatus:
        """Get current kill-switch status."""
        if self._manual_halt:
            return KillSwitchStatus.MANUAL_HALT

        for switch in self.switches.values():
            if switch.enabled and switch.tripped:
                return KillSwitchStatus.TRIPPED

        return KillSwitchStatus.ACTIVE

    def can_trade(self) -> bool:
        """Check if trading is currently allowed.

        Returns:
            True if trading is allowed, False if halted.
        """
        # Check manual halt
        if self._manual_halt:
            return False

        # Check for auto-reset
        self._check_auto_reset()

        # Check if any switch is tripped
        for switch in self.switches.values():
            if switch.enabled and switch.tripped:
                return False

        return True

    def get_trip_reason(self) -> KillSwitchReason | None:
        """Get the reason trading was halted.

        Returns:
            KillSwitchReason if halted, None if active.
        """
        if self._manual_halt:
            return KillSwitchReason.MANUAL

        for switch in self.switches.values():
            if switch.enabled and switch.tripped:
                return switch.reason

        return None

    def get_tripped_switches(self) -> list[KillSwitch]:
        """Get all tripped kill-switches.

        Returns:
            List of tripped switches.
        """
        return [
            switch
            for switch in self.switches.values()
            if switch.enabled and switch.tripped
        ]

    # --- State Updates ---

    def record_trade(
        self,
        pnl: float = 0.0,
        exposure_delta: float = 0.0,
    ) -> None:
        """Record a trade for kill-switch tracking.

        Args:
            pnl: Realized P&L from the trade.
            exposure_delta: Change in exposure (positive = more exposure).
        """
        # Check if daily reset needed
        self._check_daily_reset()

        # Update state
        self.state.daily_pnl += pnl
        self.state.current_exposure += exposure_delta

        # Check daily loss limit
        self._check_daily_loss_limit()

        # Check max exposure
        self._check_max_exposure()

    def update_exposure(self, total_exposure: float) -> None:
        """Update current exposure directly.

        Args:
            total_exposure: Total current exposure in dollars.
        """
        self.state.current_exposure = total_exposure
        self._check_max_exposure()

    def record_api_call(
        self,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record an API call for error rate tracking.

        Args:
            success: Whether the call succeeded.
            error: Error message if failed.
        """
        now = time.time()
        self._api_calls.append((now, success))

        # Prune old calls
        cutoff = now - self.config.api_error_window_seconds
        while self._api_calls and self._api_calls[0][0] < cutoff:
            self._api_calls.popleft()

        # Check error rate
        self._check_api_error_rate()

        if not success:
            logger.warning("api_call_failed", error=error)

    def validate_model_health(
        self,
        probabilities: dict[str, float],
    ) -> bool:
        """Validate model output health.

        Checks:
        - No NaN values
        - Probabilities sum to approximately 1.0
        - All values are between 0 and 1

        Args:
            probabilities: Dict of outcome probabilities.

        Returns:
            True if model is healthy, False if unhealthy.
        """
        switch = self.switches[KillSwitchReason.MODEL_HEALTH]
        if not switch.enabled:
            return True

        self.state.last_model_probs = probabilities

        # Check for empty
        if not probabilities:
            switch.trip(0.0, "Empty probabilities")
            logger.error("model_health_check_failed", reason="empty_probabilities")
            return False

        # Check for NaN
        import math
        for key, value in probabilities.items():
            if math.isnan(value) or math.isinf(value):
                switch.trip(value, f"NaN/Inf in {key}")
                logger.error(
                    "model_health_check_failed",
                    reason="nan_or_inf",
                    key=key,
                    value=value,
                )
                return False

        # Check range
        for key, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                switch.trip(value, f"Out of range: {key}={value}")
                logger.error(
                    "model_health_check_failed",
                    reason="out_of_range",
                    key=key,
                    value=value,
                )
                return False

        # Check sum (should be close to 1.0)
        total = sum(probabilities.values())
        if abs(total - 1.0) > self.config.model_prob_sum_tolerance:
            switch.trip(total, f"Probabilities sum to {total}, expected ~1.0")
            logger.error(
                "model_health_check_failed",
                reason="sum_not_one",
                sum=total,
                tolerance=self.config.model_prob_sum_tolerance,
            )
            return False

        return True

    # --- Manual Control ---

    def halt_trading(self, reason: str = "") -> None:
        """Manually halt trading.

        Args:
            reason: Reason for halt.
        """
        self._manual_halt = True
        logger.warning("manual_trading_halt", reason=reason)

    def resume_trading(self) -> bool:
        """Resume trading if safe to do so.

        Returns:
            True if trading resumed, False if still blocked.
        """
        if self.config.require_manual_reset:
            # Must explicitly reset tripped switches
            for switch in self.switches.values():
                if switch.tripped:
                    logger.warning(
                        "cannot_resume_switch_tripped",
                        switch=switch.name,
                        reason=switch.reason.value,
                    )
                    return False

        self._manual_halt = False
        logger.info("trading_resumed")
        return True

    def reset_switch(self, reason: KillSwitchReason) -> bool:
        """Reset a specific kill-switch.

        Args:
            reason: Which switch to reset.

        Returns:
            True if reset, False if not found.
        """
        switch = self.switches.get(reason)
        if switch:
            switch.reset()
            logger.info("kill_switch_reset", switch=switch.name)
            return True
        return False

    def reset_all_switches(self) -> None:
        """Reset all kill-switches."""
        for switch in self.switches.values():
            switch.reset()
        self._manual_halt = False
        logger.info("all_kill_switches_reset")

    # --- Private Methods ---

    def _check_daily_reset(self) -> None:
        """Check if daily tracking should reset."""
        now = utc_now()
        reset_time = self.state.daily_pnl_reset_time

        # Reset at midnight UTC
        if now.date() > reset_time.date():
            self.state.reset_daily()
            # Also reset daily loss switch
            self.switches[KillSwitchReason.DAILY_LOSS_LIMIT].reset()
            logger.info("daily_tracking_reset")

    def _check_daily_loss_limit(self) -> None:
        """Check daily loss limit kill-switch."""
        switch = self.switches[KillSwitchReason.DAILY_LOSS_LIMIT]
        if not switch.enabled or switch.tripped:
            return

        if self.state.daily_pnl <= -self.config.daily_loss_limit_dollars:
            switch.trip(
                self.state.daily_pnl,
                f"Daily loss ${abs(self.state.daily_pnl):.2f} exceeds limit ${self.config.daily_loss_limit_dollars:.2f}",
            )
            logger.error(
                "kill_switch_tripped",
                reason="daily_loss_limit",
                daily_pnl=self.state.daily_pnl,
                limit=self.config.daily_loss_limit_dollars,
            )

    def _check_max_exposure(self) -> None:
        """Check max exposure kill-switch."""
        switch = self.switches[KillSwitchReason.MAX_EXPOSURE]
        if not switch.enabled or switch.tripped:
            return

        if self.state.current_exposure >= self.config.max_exposure_dollars:
            switch.trip(
                self.state.current_exposure,
                f"Exposure ${self.state.current_exposure:.2f} exceeds limit ${self.config.max_exposure_dollars:.2f}",
            )
            logger.error(
                "kill_switch_tripped",
                reason="max_exposure",
                exposure=self.state.current_exposure,
                limit=self.config.max_exposure_dollars,
            )

    def _check_api_error_rate(self) -> None:
        """Check API error rate kill-switch."""
        switch = self.switches[KillSwitchReason.API_ERROR_RATE]
        if not switch.enabled or switch.tripped:
            return

        if not self._api_calls:
            return

        # Count errors in window
        total_calls = len(self._api_calls)
        error_count = sum(1 for _, success in self._api_calls if not success)

        # Check absolute error count
        if error_count >= self.config.api_error_threshold:
            switch.trip(
                error_count,
                f"{error_count} API errors in {self.config.api_error_window_seconds}s window",
            )
            logger.error(
                "kill_switch_tripped",
                reason="api_error_count",
                errors=error_count,
                threshold=self.config.api_error_threshold,
            )
            return

        # Check error rate
        if total_calls >= 10:  # Only check rate with enough samples
            error_rate = error_count / total_calls
            if error_rate >= self.config.api_error_rate_threshold:
                switch.trip(
                    error_rate,
                    f"API error rate {error_rate:.1%} exceeds threshold {self.config.api_error_rate_threshold:.1%}",
                )
                logger.error(
                    "kill_switch_tripped",
                    reason="api_error_rate",
                    error_rate=error_rate,
                    threshold=self.config.api_error_rate_threshold,
                )

    def _check_auto_reset(self) -> None:
        """Check if any switches should auto-reset."""
        if self.config.auto_reset_after_hours is None:
            return

        reset_threshold = timedelta(hours=self.config.auto_reset_after_hours)

        for switch in self.switches.values():
            if switch.tripped and switch.trip_time:
                if utc_now() - switch.trip_time >= reset_threshold:
                    switch.reset()
                    logger.info(
                        "kill_switch_auto_reset",
                        switch=switch.name,
                        after_hours=self.config.auto_reset_after_hours,
                    )

    # --- Reporting ---

    def get_status_report(self) -> dict[str, Any]:
        """Get detailed status report.

        Returns:
            Status report dictionary.
        """
        return {
            "status": self.status.value,
            "can_trade": self.can_trade(),
            "manual_halt": self._manual_halt,
            "state": {
                "daily_pnl": self.state.daily_pnl,
                "daily_pnl_reset_time": self.state.daily_pnl_reset_time.isoformat(),
                "current_exposure": self.state.current_exposure,
            },
            "switches": {
                reason.value: switch.to_dict()
                for reason, switch in self.switches.items()
            },
            "api_stats": {
                "calls_in_window": len(self._api_calls),
                "errors_in_window": sum(
                    1 for _, success in self._api_calls if not success
                ),
                "window_seconds": self.config.api_error_window_seconds,
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"KillSwitchManager(status={self.status.value}, can_trade={self.can_trade()})"
