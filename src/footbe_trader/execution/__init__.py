"""Execution module for order placement and trading safety."""

from footbe_trader.execution.alerts import (
    Alert,
    AlertConfig,
    AlertHandler,
    AlertLevel,
    AlertManager,
    AlertType,
)
from footbe_trader.execution.interfaces import ExecutionResult, IExecutor
from footbe_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchManager,
    KillSwitchReason,
    KillSwitchStatus,
    TradingState,
)
from footbe_trader.execution.live_executor import (
    ExecutionMode,
    KillSwitchTrippedError,
    LiveExecutor,
    LiveExecutorConfig,
    LiveTradingNotEnabledError,
)
from footbe_trader.execution.placeholder import PlaceholderExecutor

__all__ = [
    # Interfaces
    "IExecutor",
    "ExecutionResult",
    # Alerts
    "Alert",
    "AlertConfig",
    "AlertHandler",
    "AlertLevel",
    "AlertManager",
    "AlertType",
    # Kill-switch
    "KillSwitch",
    "KillSwitchConfig",
    "KillSwitchManager",
    "KillSwitchReason",
    "KillSwitchStatus",
    "TradingState",
    # Live executor
    "ExecutionMode",
    "KillSwitchTrippedError",
    "LiveExecutor",
    "LiveExecutorConfig",
    "LiveTradingNotEnabledError",
    # Placeholder
    "PlaceholderExecutor",
]
