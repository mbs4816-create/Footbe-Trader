"""Alert System for Trading Events.

Provides alerting for critical trading events:
- Trade executions
- Kill-switch trips
- Mapping failures
- System errors
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity level."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Type of alert."""

    TRADE_EXECUTED = "trade_executed"
    ORDER_PLACED = "order_placed"
    ORDER_FAILED = "order_failed"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    KILL_SWITCH_TRIPPED = "kill_switch_tripped"
    KILL_SWITCH_RESET = "kill_switch_reset"
    MAPPING_FAILURE = "mapping_failure"
    API_ERROR = "api_error"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"
    DAILY_SUMMARY = "daily_summary"
    TRADING_STARTED = "trading_started"
    TRADING_STOPPED = "trading_stopped"


@dataclass
class Alert:
    """Alert data."""

    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=utc_now)
    data: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "acknowledged": self.acknowledged,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
        }

    def acknowledge(self) -> None:
        """Acknowledge this alert."""
        self.acknowledged = True
        self.acknowledged_at = utc_now()


class AlertHandler(Protocol):
    """Protocol for alert handlers."""

    def handle(self, alert: Alert) -> None:
        """Handle an alert.

        Args:
            alert: Alert to handle.
        """
        ...


class LogAlertHandler:
    """Handler that logs alerts."""

    def __init__(self, min_level: AlertLevel = AlertLevel.INFO):
        """Initialize handler.

        Args:
            min_level: Minimum level to log.
        """
        self.min_level = min_level
        self._level_order = list(AlertLevel)

    def handle(self, alert: Alert) -> None:
        """Log the alert."""
        if self._level_order.index(alert.level) < self._level_order.index(
            self.min_level
        ):
            return

        log_func = {
            AlertLevel.DEBUG: logger.debug,
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)

        log_func(
            "alert",
            alert_type=alert.alert_type.value,
            title=alert.title,
            message=alert.message,
            **alert.data,
        )


class FileAlertHandler:
    """Handler that writes alerts to a file."""

    def __init__(
        self,
        file_path: Path | str,
        min_level: AlertLevel = AlertLevel.WARNING,
    ):
        """Initialize handler.

        Args:
            file_path: Path to alert log file.
            min_level: Minimum level to write.
        """
        self.file_path = Path(file_path)
        self.min_level = min_level
        self._level_order = list(AlertLevel)

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, alert: Alert) -> None:
        """Write alert to file."""
        if self._level_order.index(alert.level) < self._level_order.index(
            self.min_level
        ):
            return

        with open(self.file_path, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")


class CallbackAlertHandler:
    """Handler that calls a callback function."""

    def __init__(
        self,
        callback: Callable[[Alert], None],
        min_level: AlertLevel = AlertLevel.INFO,
    ):
        """Initialize handler.

        Args:
            callback: Function to call with alert.
            min_level: Minimum level to trigger callback.
        """
        self.callback = callback
        self.min_level = min_level
        self._level_order = list(AlertLevel)

    def handle(self, alert: Alert) -> None:
        """Call the callback."""
        if self._level_order.index(alert.level) < self._level_order.index(
            self.min_level
        ):
            return

        self.callback(alert)


@dataclass
class AlertConfig:
    """Configuration for alert system."""

    # Enable/disable alert types
    enabled: bool = True
    log_alerts: bool = True
    file_alerts: bool = False
    file_path: str = "logs/alerts.jsonl"

    # Minimum levels
    min_log_level: AlertLevel = AlertLevel.INFO
    min_file_level: AlertLevel = AlertLevel.WARNING

    # Alert-specific settings
    alert_on_every_trade: bool = True
    alert_on_kill_switch: bool = True
    alert_on_mapping_failure: bool = True
    alert_on_api_error: bool = True

    # Rate limiting
    max_alerts_per_minute: int = 60
    suppress_duplicates_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "log_alerts": self.log_alerts,
            "file_alerts": self.file_alerts,
            "file_path": self.file_path,
            "min_log_level": self.min_log_level.value,
            "min_file_level": self.min_file_level.value,
            "alert_on_every_trade": self.alert_on_every_trade,
            "alert_on_kill_switch": self.alert_on_kill_switch,
            "alert_on_mapping_failure": self.alert_on_mapping_failure,
            "alert_on_api_error": self.alert_on_api_error,
        }


class AlertManager:
    """Manages alerts for trading system.

    Handles sending alerts through multiple channels and maintains
    an alert history for debugging and analysis.

    Usage:
        manager = AlertManager()

        # Add custom handlers
        manager.add_handler(CallbackAlertHandler(my_slack_notify))

        # Send alerts
        manager.alert_trade_executed(ticker="MARKET-1", price=0.45, quantity=10)
        manager.alert_kill_switch_tripped("daily_loss_limit", daily_pnl=-500)
    """

    def __init__(self, config: AlertConfig | None = None):
        """Initialize alert manager.

        Args:
            config: Alert configuration.
        """
        self.config = config or AlertConfig()
        self.handlers: list[AlertHandler] = []
        self.history: list[Alert] = []
        self._last_alerts: dict[str, datetime] = {}  # For duplicate suppression

        # Add default handlers
        if self.config.log_alerts:
            self.handlers.append(LogAlertHandler(self.config.min_log_level))

        if self.config.file_alerts:
            self.handlers.append(
                FileAlertHandler(self.config.file_path, self.config.min_file_level)
            )

    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler.

        Args:
            handler: Handler to add.
        """
        self.handlers.append(handler)

    def send(self, alert: Alert) -> None:
        """Send an alert through all handlers.

        Args:
            alert: Alert to send.
        """
        if not self.config.enabled:
            return

        # Check for duplicate suppression
        alert_key = f"{alert.alert_type.value}:{alert.title}"
        if alert_key in self._last_alerts:
            elapsed = (utc_now() - self._last_alerts[alert_key]).total_seconds()
            if elapsed < self.config.suppress_duplicates_seconds:
                return

        self._last_alerts[alert_key] = utc_now()

        # Send to all handlers
        for handler in self.handlers:
            try:
                handler.handle(alert)
            except Exception as e:
                logger.error("alert_handler_error", handler=type(handler).__name__, error=str(e))

        # Add to history
        self.history.append(alert)

        # Prune old history (keep last 1000)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    # --- Convenience Methods ---

    def alert_trade_executed(
        self,
        ticker: str,
        side: str,
        action: str,
        price: float,
        quantity: int,
        order_id: str | None = None,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Alert on trade execution.

        Args:
            ticker: Market ticker.
            side: Side (yes/no).
            action: Action (buy/sell).
            price: Execution price.
            quantity: Quantity executed.
            order_id: Order ID.
            pnl: Realized P&L if closing.
            **kwargs: Additional data.
        """
        if not self.config.alert_on_every_trade:
            return

        data = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "price": price,
            "quantity": quantity,
            "order_id": order_id,
            "pnl": pnl,
            **kwargs,
        }

        message = f"{action.upper()} {quantity} {side} @ ${price:.2f}"
        if pnl is not None:
            message += f" (P&L: ${pnl:+.2f})"

        self.send(
            Alert(
                alert_type=AlertType.TRADE_EXECUTED,
                level=AlertLevel.INFO,
                title=f"Trade: {ticker}",
                message=message,
                data=data,
            )
        )

    def alert_order_placed(
        self,
        ticker: str,
        side: str,
        action: str,
        price: float,
        quantity: int,
        order_id: str,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> None:
        """Alert on order placement.

        Args:
            ticker: Market ticker.
            side: Side (yes/no).
            action: Action (buy/sell).
            price: Limit price.
            quantity: Order quantity.
            order_id: Order ID.
            dry_run: Whether this was a dry-run.
            **kwargs: Additional data.
        """
        data = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "price": price,
            "quantity": quantity,
            "order_id": order_id,
            "dry_run": dry_run,
            **kwargs,
        }

        prefix = "[DRY-RUN] " if dry_run else ""
        self.send(
            Alert(
                alert_type=AlertType.ORDER_PLACED,
                level=AlertLevel.INFO,
                title=f"{prefix}Order: {ticker}",
                message=f"{prefix}{action.upper()} {quantity} {side} @ ${price:.2f}",
                data=data,
            )
        )

    def alert_order_failed(
        self,
        ticker: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        """Alert on order failure.

        Args:
            ticker: Market ticker.
            error: Error message.
            **kwargs: Additional data.
        """
        self.send(
            Alert(
                alert_type=AlertType.ORDER_FAILED,
                level=AlertLevel.ERROR,
                title=f"Order Failed: {ticker}",
                message=error,
                data={"ticker": ticker, "error": error, **kwargs},
            )
        )

    def alert_kill_switch_tripped(
        self,
        reason: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Alert on kill-switch trip.

        Args:
            reason: Kill-switch reason.
            message: Description message.
            **kwargs: Additional data.
        """
        if not self.config.alert_on_kill_switch:
            return

        self.send(
            Alert(
                alert_type=AlertType.KILL_SWITCH_TRIPPED,
                level=AlertLevel.CRITICAL,
                title=f"⚠️ KILL-SWITCH: {reason}",
                message=message,
                data={"reason": reason, **kwargs},
            )
        )

    def alert_kill_switch_reset(
        self,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Alert on kill-switch reset.

        Args:
            reason: Kill-switch that was reset.
            **kwargs: Additional data.
        """
        self.send(
            Alert(
                alert_type=AlertType.KILL_SWITCH_RESET,
                level=AlertLevel.INFO,
                title=f"Kill-switch Reset: {reason}",
                message=f"Kill-switch {reason} has been reset",
                data={"reason": reason, **kwargs},
            )
        )

    def alert_mapping_failure(
        self,
        fixture_id: int,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Alert on fixture mapping failure.

        Args:
            fixture_id: Fixture ID that failed to map.
            reason: Failure reason.
            **kwargs: Additional data.
        """
        if not self.config.alert_on_mapping_failure:
            return

        self.send(
            Alert(
                alert_type=AlertType.MAPPING_FAILURE,
                level=AlertLevel.WARNING,
                title=f"Mapping Failed: Fixture {fixture_id}",
                message=reason,
                data={"fixture_id": fixture_id, "reason": reason, **kwargs},
            )
        )

    def alert_api_error(
        self,
        endpoint: str,
        error: str,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Alert on API error.

        Args:
            endpoint: API endpoint.
            error: Error message.
            status_code: HTTP status code.
            **kwargs: Additional data.
        """
        if not self.config.alert_on_api_error:
            return

        self.send(
            Alert(
                alert_type=AlertType.API_ERROR,
                level=AlertLevel.ERROR,
                title=f"API Error: {endpoint}",
                message=error,
                data={
                    "endpoint": endpoint,
                    "error": error,
                    "status_code": status_code,
                    **kwargs,
                },
            )
        )

    def alert_model_error(
        self,
        model_name: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        """Alert on model error.

        Args:
            model_name: Model name.
            error: Error description.
            **kwargs: Additional data.
        """
        self.send(
            Alert(
                alert_type=AlertType.MODEL_ERROR,
                level=AlertLevel.ERROR,
                title=f"Model Error: {model_name}",
                message=error,
                data={"model_name": model_name, "error": error, **kwargs},
            )
        )

    def alert_trading_started(
        self,
        mode: str,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Alert on trading session start.

        Args:
            mode: Trading mode (live/paper/dry-run).
            config: Configuration summary.
            **kwargs: Additional data.
        """
        self.send(
            Alert(
                alert_type=AlertType.TRADING_STARTED,
                level=AlertLevel.INFO,
                title=f"Trading Started ({mode})",
                message=f"Trading session started in {mode} mode",
                data={"mode": mode, "config": config or {}, **kwargs},
            )
        )

    def alert_trading_stopped(
        self,
        reason: str,
        **kwargs: Any,
    ) -> None:
        """Alert on trading session stop.

        Args:
            reason: Reason for stopping.
            **kwargs: Additional data.
        """
        self.send(
            Alert(
                alert_type=AlertType.TRADING_STOPPED,
                level=AlertLevel.WARNING,
                title="Trading Stopped",
                message=reason,
                data={"reason": reason, **kwargs},
            )
        )

    # --- History Methods ---

    def get_recent_alerts(
        self,
        count: int = 50,
        alert_type: AlertType | None = None,
        min_level: AlertLevel | None = None,
    ) -> list[Alert]:
        """Get recent alerts from history.

        Args:
            count: Maximum alerts to return.
            alert_type: Filter by type.
            min_level: Filter by minimum level.

        Returns:
            List of recent alerts.
        """
        alerts = self.history.copy()

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if min_level:
            level_order = list(AlertLevel)
            min_idx = level_order.index(min_level)
            alerts = [a for a in alerts if level_order.index(a.level) >= min_idx]

        return alerts[-count:]

    def get_unacknowledged_alerts(self) -> list[Alert]:
        """Get unacknowledged alerts.

        Returns:
            List of unacknowledged alerts.
        """
        return [a for a in self.history if not a.acknowledged]

    def acknowledge_all(self) -> int:
        """Acknowledge all alerts.

        Returns:
            Number of alerts acknowledged.
        """
        count = 0
        for alert in self.history:
            if not alert.acknowledged:
                alert.acknowledge()
                count += 1
        return count
