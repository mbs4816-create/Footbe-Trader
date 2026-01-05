"""Tests for snapshot collector module."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from footbe_trader.kalshi.interfaces import MarketData, OrderbookData, OrderbookLevel
from footbe_trader.storage.models import HistoricalSnapshot
from footbe_trader.strategy.mapping import FixtureMarketMapping
from footbe_trader.strategy.snapshot_collector import (
    CollectionResult,
    CollectorConfig,
    SnapshotCollector,
    create_historical_snapshot_from_orderbook,
)


class TestCollectorConfig:
    """Tests for CollectorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CollectorConfig()
        assert config.interval_minutes == 5
        assert config.max_fixtures == 50
        assert config.min_hours_to_kickoff == 0.5
        assert config.max_hours_to_kickoff == 168.0
        assert config.include_model_predictions is True
        assert config.save_raw_json is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CollectorConfig(
            interval_minutes=10,
            max_fixtures=20,
            min_hours_to_kickoff=1.0,
            max_hours_to_kickoff=72.0,
        )
        assert config.interval_minutes == 10
        assert config.max_fixtures == 20
        assert config.min_hours_to_kickoff == 1.0
        assert config.max_hours_to_kickoff == 72.0

    def test_to_dict(self):
        """Test configuration serialization."""
        config = CollectorConfig(interval_minutes=5)
        data = config.to_dict()
        assert data["interval_minutes"] == 5
        assert "max_fixtures" in data
        assert "min_hours_to_kickoff" in data


class TestCollectionResult:
    """Tests for CollectionResult."""

    def test_default_values(self):
        """Test default result values."""
        result = CollectionResult()
        assert result.fixtures_checked == 0
        assert result.snapshots_collected == 0
        assert result.errors == []
        assert result.snapshot_ids == []

    def test_to_dict(self):
        """Test result serialization."""
        result = CollectionResult(
            fixtures_checked=5,
            snapshots_collected=15,
            errors=["Error 1"],
        )
        data = result.to_dict()
        assert data["fixtures_checked"] == 5
        assert data["snapshots_collected"] == 15
        assert data["errors"] == ["Error 1"]


class TestSnapshotCollector:
    """Tests for SnapshotCollector."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        collector = SnapshotCollector()
        assert collector.config.interval_minutes == 5
        assert collector.session_id is not None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = CollectorConfig(interval_minutes=10)
        collector = SnapshotCollector(config=config, session_id="test-session")
        assert collector.config.interval_minutes == 10
        assert collector.session_id == "test-session"

    def test_create_session(self):
        """Test session creation."""
        collector = SnapshotCollector()
        session = collector.create_session()
        
        assert session.session_id == collector.session_id
        assert session.status == "running"
        assert session.interval_minutes == collector.config.interval_minutes

    def test_complete_session_success(self):
        """Test successful session completion."""
        collector = SnapshotCollector()
        collector.create_session()
        session = collector.complete_session()
        
        assert session is not None
        assert session.status == "completed"
        assert session.ended_at is not None
        assert session.error_message is None

    def test_complete_session_with_error(self):
        """Test session completion with error."""
        collector = SnapshotCollector()
        collector.create_session()
        session = collector.complete_session(error="Test error")
        
        assert session is not None
        assert session.status == "failed"
        assert session.error_message == "Test error"

    def test_filter_mappings_by_time(self):
        """Test filtering mappings by time window."""
        collector = SnapshotCollector(
            config=CollectorConfig(
                min_hours_to_kickoff=1.0,
                max_hours_to_kickoff=24.0,
            )
        )

        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        mappings = [
            FixtureMarketMapping(fixture_id=1),  # Valid
            FixtureMarketMapping(fixture_id=2),  # Valid
            FixtureMarketMapping(fixture_id=3),  # Too soon
            FixtureMarketMapping(fixture_id=4),  # Too far
            FixtureMarketMapping(fixture_id=5),  # No kickoff
        ]

        kickoffs = {
            1: now + timedelta(hours=2),   # Valid
            2: now + timedelta(hours=12),  # Valid
            3: now + timedelta(hours=0.5), # Too soon
            4: now + timedelta(hours=48),  # Too far
        }

        filtered = collector._filter_mappings_by_time(mappings, kickoffs, now)

        assert len(filtered) == 2
        fixture_ids = [m.fixture_id for m in filtered]
        assert 1 in fixture_ids
        assert 2 in fixture_ids

    def test_filter_mappings_respects_max(self):
        """Test that max_fixtures limit is respected."""
        collector = SnapshotCollector(
            config=CollectorConfig(
                max_fixtures=2,
                min_hours_to_kickoff=0,
                max_hours_to_kickoff=100,
            )
        )

        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        mappings = [
            FixtureMarketMapping(fixture_id=i) for i in range(10)
        ]

        kickoffs = {
            i: now + timedelta(hours=5) for i in range(10)
        }

        filtered = collector._filter_mappings_by_time(mappings, kickoffs, now)
        assert len(filtered) == 2

    @pytest.mark.asyncio
    async def test_collect_snapshots_empty_mappings(self):
        """Test collection with no mappings."""
        collector = SnapshotCollector()
        collector.create_session()

        kalshi_mock = AsyncMock()
        result = await collector.collect_snapshots(
            mappings=[],
            kalshi_client=kalshi_mock,
        )

        assert result.fixtures_checked == 0
        assert result.snapshots_collected == 0

    @pytest.mark.asyncio
    async def test_collect_snapshots_with_mapping(self):
        """Test collection with valid mapping."""
        collector = SnapshotCollector(
            config=CollectorConfig(
                min_hours_to_kickoff=0,
                max_hours_to_kickoff=100,
            )
        )
        collector.create_session()

        # Create mock mapping
        mapping = FixtureMarketMapping(
            fixture_id=12345,
            ticker_home_win="TICKER-HOME",
            ticker_draw="TICKER-DRAW",
            ticker_away_win="TICKER-AWAY",
        )

        # Create mock Kalshi client
        kalshi_mock = AsyncMock()
        kalshi_mock.get_orderbook.return_value = OrderbookData(
            ticker="TICKER-HOME",
            yes_bids=[OrderbookLevel(price=0.45, quantity=100)],
            yes_asks=[OrderbookLevel(price=0.50, quantity=100)],
        )
        kalshi_mock.get_market.return_value = MarketData(
            ticker="TICKER-HOME",
            event_ticker="EVENT-TICKER",
            title="Test Market",
            status="open",
            yes_bid=0.45,
            yes_ask=0.50,
            no_bid=0.50,
            no_ask=0.55,
        )

        # Use future kickoff relative to actual current time
        from footbe_trader.common.time_utils import utc_now
        now = utc_now()
        kickoffs = {12345: now + timedelta(hours=24)}

        result = await collector.collect_snapshots(
            mappings=[mapping],
            kalshi_client=kalshi_mock,
            fixture_kickoffs=kickoffs,
        )

        assert result.fixtures_checked == 1
        assert result.snapshots_collected == 3  # 3 outcomes

    def test_orderbook_to_dict(self):
        """Test orderbook serialization."""
        collector = SnapshotCollector()
        orderbook = OrderbookData(
            ticker="TEST",
            yes_bids=[OrderbookLevel(price=0.45, quantity=100)],
            yes_asks=[OrderbookLevel(price=0.50, quantity=50)],
        )

        data = collector._orderbook_to_dict(orderbook)

        assert data["ticker"] == "TEST"
        assert len(data["yes_bids"]) == 1
        assert data["yes_bids"][0]["price"] == 0.45
        assert data["yes_bids"][0]["quantity"] == 100


class TestCreateHistoricalSnapshotFromOrderbook:
    """Tests for factory function."""

    def test_creates_snapshot(self):
        """Test snapshot creation from orderbook."""
        orderbook = OrderbookData(
            ticker="TEST-TICKER",
            yes_bids=[OrderbookLevel(price=0.40, quantity=100)],
            yes_asks=[OrderbookLevel(price=0.45, quantity=50)],
        )
        market = MarketData(
            ticker="TEST-TICKER",
            event_ticker="EVENT-TICKER",
            title="Test",
            status="open",
            yes_bid=0.40,
            yes_ask=0.45,
            no_bid=0.55,
            no_ask=0.60,
            volume_24h=1000,
            open_interest=500,
        )

        snapshot = create_historical_snapshot_from_orderbook(
            fixture_id=12345,
            ticker="TEST-TICKER",
            outcome="home_win",
            orderbook=orderbook,
            market=market,
            session_id="test-session",
            model_prob=0.55,
            model_version="v1.0",
        )

        assert snapshot.fixture_id == 12345
        assert snapshot.ticker == "TEST-TICKER"
        assert snapshot.outcome == "home_win"
        assert snapshot.best_bid == 0.40
        assert snapshot.best_ask == 0.45
        assert snapshot.mid == pytest.approx(0.425)
        assert snapshot.spread == pytest.approx(0.05)
        assert snapshot.session_id == "test-session"
        assert snapshot.model_prob == 0.55
        assert snapshot.model_version == "v1.0"
        assert snapshot.volume_24h == 1000
        assert snapshot.open_interest == 500

    def test_creates_snapshot_without_market(self):
        """Test snapshot creation without market data."""
        orderbook = OrderbookData(
            ticker="TEST",
            yes_bids=[OrderbookLevel(price=0.40, quantity=100)],
            yes_asks=[OrderbookLevel(price=0.45, quantity=50)],
        )

        snapshot = create_historical_snapshot_from_orderbook(
            fixture_id=12345,
            ticker="TEST",
            outcome="draw",
            orderbook=orderbook,
        )

        assert snapshot.fixture_id == 12345
        assert snapshot.ticker == "TEST"
        assert snapshot.best_bid == 0.40
        assert snapshot.best_ask == 0.45
        assert snapshot.yes_price is None
        assert snapshot.volume_24h is None
