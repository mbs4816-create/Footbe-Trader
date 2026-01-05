"""Tests for Kalshi API client with mocked responses.

Uses pytest-httpx for mocking HTTP requests in a VCR-style pattern.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from footbe_trader.common.config import KalshiConfig
from footbe_trader.kalshi.interfaces import (
    BalanceData,
    EventData,
    FillData,
    MarketData,
    OrderbookData,
    OrderbookLevel,
    PositionData,
)


# Sample API responses for VCR-style mocking
SAMPLE_BALANCE_RESPONSE = {
    "balance": 50000,  # $500.00 in cents
}

SAMPLE_EVENTS_RESPONSE = {
    "events": [
        {
            "event_ticker": "FOOTBALL-EPL-2024",
            "title": "English Premier League 2024",
            "category": "Sports",
            "mutually_exclusive": True,
            "series_ticker": "EPL-2024",
        },
        {
            "event_ticker": "FOOTBALL-UCL-2024",
            "title": "UEFA Champions League 2024",
            "category": "Sports",
            "mutually_exclusive": True,
            "series_ticker": "UCL-2024",
        },
    ],
    "cursor": None,
}

SAMPLE_MARKETS_RESPONSE = {
    "markets": [
        {
            "ticker": "ARSENAL-WIN-24",
            "title": "Will Arsenal win the 2024 EPL?",
            "event_ticker": "FOOTBALL-EPL-2024",
            "status": "open",
            "yes_bid": 4500,  # $0.45
            "yes_ask": 4700,  # $0.47
            "no_bid": 5300,
            "no_ask": 5500,
            "last_price": 4600,
            "volume": 100000,
            "volume_24h": 5000,
            "open_interest": 25000,
            "close_time": "2024-05-30T23:59:59Z",
            "result": None,
            "subtitle": "EPL Championship",
            "open_time": "2023-08-01T00:00:00Z",
            "expiration_time": "2024-05-30T23:59:59Z",
            "custom_strike": None,
            "can_close_early": True,
            "response_price_units": "cents",
        },
        {
            "ticker": "MANCITY-WIN-24",
            "title": "Will Manchester City win the 2024 EPL?",
            "event_ticker": "FOOTBALL-EPL-2024",
            "status": "open",
            "yes_bid": 3500,
            "yes_ask": 3700,
            "no_bid": 6300,
            "no_ask": 6500,
            "last_price": 3600,
            "volume": 150000,
            "volume_24h": 8000,
            "open_interest": 40000,
            "close_time": "2024-05-30T23:59:59Z",
            "result": None,
            "subtitle": "EPL Championship",
            "open_time": "2023-08-01T00:00:00Z",
            "expiration_time": "2024-05-30T23:59:59Z",
            "custom_strike": None,
            "can_close_early": True,
            "response_price_units": "cents",
        },
    ],
    "cursor": None,
}

SAMPLE_SINGLE_MARKET_RESPONSE = {
    "market": SAMPLE_MARKETS_RESPONSE["markets"][0]
}

SAMPLE_ORDERBOOK_RESPONSE = {
    "orderbook": {
        "yes": [
            [45, 100],  # price 45 cents, quantity 100
            [44, 200],
            [43, 150],
        ],
        "no": [
            [53, 80],
            [54, 120],
            [55, 90],
        ],
    }
}

SAMPLE_POSITIONS_RESPONSE = {
    "market_positions": [
        {
            "ticker": "ARSENAL-WIN-24",
            "position": 50,
            "market_exposure": 2500,
            "realized_pnl": 100,
            "total_traded": 3000,
            "resting_orders_count": 2,
        }
    ],
    "cursor": None,
}

SAMPLE_FILLS_RESPONSE = {
    "fills": [
        {
            "trade_id": "trade-123",
            "ticker": "ARSENAL-WIN-24",
            "order_id": "order-456",
            "side": "yes",
            "action": "buy",
            "count": 10,
            "yes_price": 4500,
            "no_price": 5500,
            "created_time": "2024-01-15T10:30:00Z",
            "is_taker": True,
        }
    ],
    "cursor": None,
}


@pytest.fixture
def kalshi_config() -> KalshiConfig:
    """Create test Kalshi config."""
    return KalshiConfig(
        api_key_id="test-key-id",
        private_key="",  # Will be mocked
        use_demo=True,
        rate_limit_per_second=100,  # High limit for tests
        max_retries=0,  # No retries in tests
    )


class TestRateLimiting:
    """Tests for rate limiting behavior."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_token_bucket(self):
        """Test that rate limiter uses token bucket algorithm."""
        from footbe_trader.kalshi.rate_limit import RateLimiter
        
        limiter = RateLimiter(requests_per_second=10, burst_size=5)
        
        # Should be able to make 5 requests immediately (burst_size)
        for i in range(5):
            wait_time = await limiter.acquire()
            # First requests should have minimal wait
            assert wait_time < 0.5, f"Request {i} waited too long: {wait_time}"
    
    def test_request_logger_tracks_timing(self):
        """Test that request logger tracks timing stats."""
        from footbe_trader.kalshi.rate_limit import RequestLogger
        
        logger = RequestLogger()
        start1 = logger.log_request("GET", "/test")
        logger.log_response("GET", "/test", 200, start1)
        
        start2 = logger.log_request("GET", "/test")
        logger.log_response("GET", "/test", 200, start2)
        
        stats = logger.get_stats()
        assert stats["count"] == 2


class TestRetryBehavior:
    """Tests for retry and backoff behavior."""
    
    def test_retry_config_calculates_backoff(self):
        """Test exponential backoff calculation."""
        from footbe_trader.kalshi.rate_limit import RetryConfig
        
        config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            retryable_status_codes=(429, 500),
        )
        
        # First attempt: 1.0 * 2^0 = 1.0 (with jitter)
        delay0 = config.get_delay(0)
        assert 0.5 < delay0 <= 1.5  # Base + jitter
        
        # Second attempt: 1.0 * 2^1 = 2.0 (with jitter)
        delay1 = config.get_delay(1)
        assert 1.0 < delay1 <= 3.0
        
        # Should respect max delay
        delay_many = config.get_delay(10)
        assert delay_many <= 15.0  # max (10) + jitter


class TestOrderbookDataParsing:
    """Tests for orderbook data parsing and calculations."""
    
    def test_orderbook_mid_calculation(self):
        """Test mid-price calculation."""
        orderbook = OrderbookData(
            ticker="TEST-MARKET",
            yes_bids=[OrderbookLevel(price=0.45, quantity=100)],
            yes_asks=[OrderbookLevel(price=0.55, quantity=100)],
        )
        
        assert orderbook.mid_price == pytest.approx(0.50)
        assert orderbook.spread == pytest.approx(0.10)
    
    def test_empty_orderbook(self):
        """Test handling of empty orderbook."""
        orderbook = OrderbookData(
            ticker="TEST-MARKET",
            yes_bids=[],
            yes_asks=[],
        )
        
        assert orderbook.best_yes_bid == 0.0
        assert orderbook.best_yes_ask == 1.0  # Default when no asks


class TestDataModels:
    """Tests for data model creation and validation."""
    
    def test_event_data_creation(self):
        """Test EventData dataclass creation."""
        event = EventData(
            event_ticker="FOOTBALL-EPL-2024",
            title="English Premier League 2024",
            category="Sports",
        )
        assert event.event_ticker == "FOOTBALL-EPL-2024"
        assert event.title == "English Premier League 2024"
    
    def test_market_data_creation(self):
        """Test MarketData dataclass creation."""
        market = MarketData(
            ticker="ARSENAL-WIN-24",
            event_ticker="FOOTBALL-EPL-2024",
            title="Will Arsenal win the 2024 EPL?",
            status="open",
            yes_bid=0.45,
            yes_ask=0.47,
        )
        assert market.ticker == "ARSENAL-WIN-24"
        assert market.status == "open"
        assert market.yes_bid == 0.45
    
    def test_position_data_creation(self):
        """Test PositionData dataclass creation."""
        position = PositionData(
            ticker="ARSENAL-WIN-24",
            position=50,
            market_exposure=25.00,
            realized_pnl=1.00,
        )
        assert position.ticker == "ARSENAL-WIN-24"
        assert position.position == 50
    
    def test_fill_data_creation(self):
        """Test FillData dataclass creation."""
        fill = FillData(
            trade_id="trade-123",
            ticker="ARSENAL-WIN-24",
            order_id="order-456",
            side="yes",
            action="buy",
            price=0.45,
            count=10,
            is_taker=True,
        )
        assert fill.trade_id == "trade-123"
        assert fill.side == "yes"
        assert fill.action == "buy"
    
    def test_balance_data_creation(self):
        """Test BalanceData dataclass creation."""
        balance = BalanceData(
            balance=500.00,
            portfolio_value=1000.00,
        )
        assert balance.balance == 500.00
        assert balance.portfolio_value == 1000.00
