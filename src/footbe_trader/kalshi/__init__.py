"""Kalshi API client module."""

from footbe_trader.kalshi.auth import KalshiAuth
from footbe_trader.kalshi.client import KalshiApiError, KalshiClient
from footbe_trader.kalshi.interfaces import (
    BalanceData,
    EventData,
    FillData,
    IKalshiClient,
    MarketData,
    OrderbookData,
    OrderbookLevel,
    OrderData,
    PositionData,
)
from footbe_trader.kalshi.rate_limit import RateLimiter, RequestLogger, RetryConfig

__all__ = [
    # Client
    "KalshiClient",
    "KalshiApiError",
    "KalshiAuth",
    # Interfaces
    "IKalshiClient",
    # Data types
    "BalanceData",
    "EventData",
    "FillData",
    "MarketData",
    "OrderbookData",
    "OrderbookLevel",
    "OrderData",
    "PositionData",
    # Rate limiting
    "RateLimiter",
    "RequestLogger",
    "RetryConfig",
]
