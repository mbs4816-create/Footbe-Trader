"""Strategy module for signal generation."""

from footbe_trader.strategy.interfaces import IStrategy, Signal
from footbe_trader.strategy.placeholder import PlaceholderStrategy

__all__ = [
    "IStrategy",
    "Signal",
    "PlaceholderStrategy",
]
