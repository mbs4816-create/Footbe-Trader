"""Historical Snapshot Collector.

Collects orderbook and market snapshots at regular intervals for mapped
fixtures, storing them in the database for later strategy backtesting.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.kalshi.interfaces import IKalshiClient, MarketData, OrderbookData
from footbe_trader.modeling.interfaces import IModel, PredictionResult
from footbe_trader.storage.models import HistoricalSnapshot, SnapshotSession
from footbe_trader.strategy.mapping import FixtureMarketMapping

logger = get_logger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for snapshot collector."""

    interval_minutes: int = 5
    max_fixtures: int = 50
    min_hours_to_kickoff: float = 0.5
    max_hours_to_kickoff: float = 168.0
    include_model_predictions: bool = True
    save_raw_json: bool = True
    output_dir: Path | None = None  # Optional: save raw JSON files

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interval_minutes": self.interval_minutes,
            "max_fixtures": self.max_fixtures,
            "min_hours_to_kickoff": self.min_hours_to_kickoff,
            "max_hours_to_kickoff": self.max_hours_to_kickoff,
            "include_model_predictions": self.include_model_predictions,
            "save_raw_json": self.save_raw_json,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


@dataclass
class CollectionResult:
    """Result of a single collection cycle."""

    timestamp: datetime = field(default_factory=utc_now)
    fixtures_checked: int = 0
    snapshots_collected: int = 0
    errors: list[str] = field(default_factory=list)
    snapshot_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "fixtures_checked": self.fixtures_checked,
            "snapshots_collected": self.snapshots_collected,
            "errors": self.errors,
            "snapshot_ids": self.snapshot_ids,
        }


class SnapshotCollector:
    """Collects historical snapshots for strategy backtesting.

    This collector:
    1. Finds all mapped fixtures within the configured time window
    2. For each fixture, fetches orderbook data for all mapped markets
    3. Optionally generates model predictions
    4. Stores snapshots in the database with fixture linkage
    5. Optionally saves raw JSON files for offline analysis
    """

    def __init__(
        self,
        config: CollectorConfig | None = None,
        session_id: str | None = None,
    ):
        """Initialize collector.

        Args:
            config: Collector configuration.
            session_id: Existing session ID to continue, or None for new session.
        """
        self.config = config or CollectorConfig()
        self.session_id = session_id or str(uuid.uuid4())
        self._session: SnapshotSession | None = None
        self._total_snapshots = 0

    def create_session(self) -> SnapshotSession:
        """Create a new collection session.

        Returns:
            New session object (not yet persisted).
        """
        self._session = SnapshotSession(
            session_id=self.session_id,
            started_at=utc_now(),
            status="running",
            interval_minutes=self.config.interval_minutes,
            config_json=self.config.to_dict(),
        )
        return self._session

    def get_session(self) -> SnapshotSession | None:
        """Get current session."""
        return self._session

    async def collect_snapshots(
        self,
        mappings: list[FixtureMarketMapping],
        kalshi_client: IKalshiClient,
        prediction_model: IModel | None = None,
        fixture_kickoffs: dict[int, datetime] | None = None,
    ) -> CollectionResult:
        """Collect snapshots for all mapped fixtures.

        Args:
            mappings: List of fixture-to-market mappings.
            kalshi_client: Kalshi API client for fetching orderbooks.
            prediction_model: Optional model for generating predictions.
            fixture_kickoffs: Dict mapping fixture_id to kickoff time.

        Returns:
            Collection result with statistics and snapshot IDs.
        """
        result = CollectionResult(timestamp=utc_now())
        now = utc_now()

        # Filter mappings by time window
        active_mappings = self._filter_mappings_by_time(
            mappings, fixture_kickoffs or {}, now
        )

        result.fixtures_checked = len(active_mappings)
        logger.info(
            "collecting_snapshots",
            session_id=self.session_id,
            active_fixtures=len(active_mappings),
            total_mappings=len(mappings),
        )

        for mapping in active_mappings:
            try:
                fixture_snapshots = await self._collect_fixture_snapshots(
                    mapping=mapping,
                    kalshi_client=kalshi_client,
                    prediction_model=prediction_model,
                    timestamp=now,
                )
                result.snapshots_collected += len(fixture_snapshots)
                self._total_snapshots += len(fixture_snapshots)

            except Exception as e:
                error_msg = f"Error collecting fixture {mapping.fixture_id}: {e}"
                result.errors.append(error_msg)
                logger.error(
                    "snapshot_collection_error",
                    fixture_id=mapping.fixture_id,
                    error=str(e),
                )

        # Update session stats
        if self._session:
            self._session.fixtures_tracked = result.fixtures_checked
            self._session.snapshots_collected = self._total_snapshots

        logger.info(
            "collection_complete",
            session_id=self.session_id,
            snapshots=result.snapshots_collected,
            errors=len(result.errors),
        )

        return result

    def _filter_mappings_by_time(
        self,
        mappings: list[FixtureMarketMapping],
        fixture_kickoffs: dict[int, datetime],
        now: datetime,
    ) -> list[FixtureMarketMapping]:
        """Filter mappings to those within the collection time window.

        Args:
            mappings: All available mappings.
            fixture_kickoffs: Dict mapping fixture_id to kickoff time.
            now: Current timestamp.

        Returns:
            Filtered list of mappings.
        """
        filtered = []
        min_kickoff = now + timedelta(hours=self.config.min_hours_to_kickoff)
        max_kickoff = now + timedelta(hours=self.config.max_hours_to_kickoff)

        for mapping in mappings:
            kickoff = fixture_kickoffs.get(mapping.fixture_id)
            if kickoff is None:
                # No kickoff time, skip
                continue

            if min_kickoff <= kickoff <= max_kickoff:
                filtered.append(mapping)

        # Apply max fixtures limit
        return filtered[: self.config.max_fixtures]

    async def _collect_fixture_snapshots(
        self,
        mapping: FixtureMarketMapping,
        kalshi_client: IKalshiClient,
        prediction_model: IModel | None,
        timestamp: datetime,
    ) -> list[HistoricalSnapshot]:
        """Collect snapshots for a single fixture.

        Args:
            mapping: Fixture-to-market mapping.
            kalshi_client: Kalshi API client.
            prediction_model: Optional prediction model.
            timestamp: Snapshot timestamp.

        Returns:
            List of collected snapshots.
        """
        snapshots = []

        # Collect each outcome's market
        outcome_tickers = [
            ("home_win", mapping.ticker_home_win),
            ("draw", mapping.ticker_draw),
            ("away_win", mapping.ticker_away_win),
        ]

        for outcome, ticker in outcome_tickers:
            if not ticker:
                continue

            try:
                snapshot = await self._collect_market_snapshot(
                    fixture_id=mapping.fixture_id,
                    ticker=ticker,
                    outcome=outcome,
                    kalshi_client=kalshi_client,
                    prediction_model=prediction_model,
                    timestamp=timestamp,
                )
                if snapshot:
                    snapshots.append(snapshot)

            except Exception as e:
                logger.warning(
                    "market_snapshot_error",
                    fixture_id=mapping.fixture_id,
                    ticker=ticker,
                    outcome=outcome,
                    error=str(e),
                )

        return snapshots

    async def _collect_market_snapshot(
        self,
        fixture_id: int,
        ticker: str,
        outcome: str,
        kalshi_client: IKalshiClient,
        prediction_model: IModel | None,
        timestamp: datetime,
    ) -> HistoricalSnapshot | None:
        """Collect a single market snapshot.

        Args:
            fixture_id: API-Football fixture ID.
            ticker: Kalshi market ticker.
            outcome: Outcome type ('home_win', 'draw', 'away_win').
            kalshi_client: Kalshi API client.
            prediction_model: Optional prediction model.
            timestamp: Snapshot timestamp.

        Returns:
            Historical snapshot or None on error.
        """
        # Fetch orderbook
        orderbook = await kalshi_client.get_orderbook(ticker)

        # Fetch market data
        market = await kalshi_client.get_market(ticker)

        # Build snapshot
        snapshot = HistoricalSnapshot(
            session_id=self.session_id,
            fixture_id=fixture_id,
            ticker=ticker,
            outcome=outcome,
            timestamp=timestamp,
            best_bid=orderbook.best_yes_bid if orderbook.yes_bids else None,
            best_ask=orderbook.best_yes_ask if orderbook.yes_asks else None,
            mid=orderbook.mid_price,
            spread=orderbook.spread,
            bid_volume=orderbook.total_bid_volume,
            ask_volume=orderbook.total_ask_volume,
            yes_price=market.yes_bid if market else None,
            no_price=market.no_bid if market else None,
            volume_24h=market.volume_24h if market else None,
            open_interest=market.open_interest if market else None,
        )

        # Add raw JSON if configured
        if self.config.save_raw_json:
            snapshot.raw_orderbook_json = self._orderbook_to_dict(orderbook)
            if market:
                snapshot.raw_market_json = market.raw_data

        # Add model prediction if available
        if prediction_model:
            try:
                prediction = await self._get_model_prediction(
                    prediction_model, fixture_id, outcome
                )
                if prediction:
                    snapshot.model_prob = prediction.probability
                    snapshot.model_version = prediction.model_version
            except Exception as e:
                logger.debug(
                    "prediction_unavailable",
                    fixture_id=fixture_id,
                    outcome=outcome,
                    error=str(e),
                )

        # Save to file if output directory configured
        if self.config.output_dir:
            self._save_snapshot_file(snapshot)

        return snapshot

    def _orderbook_to_dict(self, orderbook: OrderbookData) -> dict[str, Any]:
        """Convert orderbook to dictionary for storage."""
        return {
            "ticker": orderbook.ticker,
            "yes_bids": [
                {"price": l.price, "quantity": l.quantity} for l in orderbook.yes_bids
            ],
            "yes_asks": [
                {"price": l.price, "quantity": l.quantity} for l in orderbook.yes_asks
            ],
            "no_bids": [
                {"price": l.price, "quantity": l.quantity} for l in orderbook.no_bids
            ],
            "no_asks": [
                {"price": l.price, "quantity": l.quantity} for l in orderbook.no_asks
            ],
            "best_yes_bid": orderbook.best_yes_bid,
            "best_yes_ask": orderbook.best_yes_ask,
            "mid": orderbook.mid_price,
            "spread": orderbook.spread,
        }

    async def _get_model_prediction(
        self,
        model: IModel,
        fixture_id: int,
        outcome: str,
    ) -> PredictionResult | None:
        """Get model prediction for a fixture outcome.

        Args:
            model: Prediction model.
            fixture_id: Fixture ID.
            outcome: Outcome type.

        Returns:
            Prediction result or None.
        """
        # This would integrate with the actual prediction model
        # For now, return None as predictions require feature data
        return None

    def _save_snapshot_file(self, snapshot: HistoricalSnapshot) -> None:
        """Save snapshot to JSON file.

        Args:
            snapshot: Snapshot to save.
        """
        if not self.config.output_dir:
            return

        output_dir = self.config.output_dir / self.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{snapshot.fixture_id}_{snapshot.outcome}_"
            f"{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        filepath = output_dir / filename

        data = {
            "session_id": snapshot.session_id,
            "fixture_id": snapshot.fixture_id,
            "ticker": snapshot.ticker,
            "outcome": snapshot.outcome,
            "timestamp": snapshot.timestamp.isoformat(),
            "orderbook": {
                "best_bid": snapshot.best_bid,
                "best_ask": snapshot.best_ask,
                "mid": snapshot.mid,
                "spread": snapshot.spread,
                "bid_volume": snapshot.bid_volume,
                "ask_volume": snapshot.ask_volume,
            },
            "market": {
                "yes_price": snapshot.yes_price,
                "no_price": snapshot.no_price,
                "volume_24h": snapshot.volume_24h,
                "open_interest": snapshot.open_interest,
            },
            "model": {
                "prob": snapshot.model_prob,
                "version": snapshot.model_version,
            },
            "raw_orderbook": snapshot.raw_orderbook_json,
            "raw_market": snapshot.raw_market_json,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug("snapshot_saved_to_file", filepath=str(filepath))

    def complete_session(self, error: str | None = None) -> SnapshotSession | None:
        """Mark session as complete.

        Args:
            error: Error message if session failed.

        Returns:
            Updated session object.
        """
        if not self._session:
            return None

        self._session.ended_at = utc_now()
        self._session.status = "failed" if error else "completed"
        self._session.error_message = error

        return self._session


def create_historical_snapshot_from_orderbook(
    fixture_id: int,
    ticker: str,
    outcome: str,
    orderbook: OrderbookData,
    market: MarketData | None = None,
    session_id: str | None = None,
    model_prob: float | None = None,
    model_version: str | None = None,
) -> HistoricalSnapshot:
    """Factory function to create a HistoricalSnapshot from orderbook data.

    Args:
        fixture_id: API-Football fixture ID.
        ticker: Market ticker.
        outcome: Outcome type.
        orderbook: Orderbook data.
        market: Optional market data.
        session_id: Optional session ID.
        model_prob: Optional model probability.
        model_version: Optional model version.

    Returns:
        Populated HistoricalSnapshot.
    """
    return HistoricalSnapshot(
        session_id=session_id,
        fixture_id=fixture_id,
        ticker=ticker,
        outcome=outcome,
        timestamp=utc_now(),
        best_bid=orderbook.best_yes_bid if orderbook.yes_bids else None,
        best_ask=orderbook.best_yes_ask if orderbook.yes_asks else None,
        mid=orderbook.mid_price,
        spread=orderbook.spread,
        bid_volume=orderbook.total_bid_volume,
        ask_volume=orderbook.total_ask_volume,
        yes_price=market.yes_bid if market else None,
        no_price=market.no_bid if market else None,
        volume_24h=market.volume_24h if market else None,
        open_interest=market.open_interest if market else None,
        model_prob=model_prob,
        model_version=model_version,
        raw_orderbook_json={
            "ticker": orderbook.ticker,
            "yes_bids": [
                {"price": l.price, "quantity": l.quantity} for l in orderbook.yes_bids
            ],
            "yes_asks": [
                {"price": l.price, "quantity": l.quantity} for l in orderbook.yes_asks
            ],
        },
        raw_market_json=market.raw_data if market else {},
    )
