#!/usr/bin/env python3
"""Snapshot Collection CLI.

Collect historical orderbook and market snapshots for mapped fixtures
at regular intervals for later strategy backtesting.

Usage:
    python scripts/collect_snapshots.py --interval 5 --duration 24
    python scripts/collect_snapshots.py --once
    python scripts/collect_snapshots.py --session-id abc123 --resume
"""

import argparse
import asyncio
import signal
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.storage.database import Database
from footbe_trader.strategy.mapping import MappingConfig, MappingEngine
from footbe_trader.strategy.snapshot_collector import (
    CollectionResult,
    CollectorConfig,
    SnapshotCollector,
)

logger = get_logger(__name__)

# Global shutdown flag
_shutdown_requested = False


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals."""
    global _shutdown_requested
    logger.info("shutdown_requested", signal=signum)
    _shutdown_requested = True


async def run_collection(
    collector: SnapshotCollector,
    db: Database,
    kalshi_client: KalshiClient,
    mapping_engine: MappingEngine,
    once: bool = False,
    duration_hours: float | None = None,
) -> None:
    """Run the collection loop.

    Args:
        collector: Snapshot collector instance.
        db: Database connection.
        kalshi_client: Kalshi API client.
        mapping_engine: Mapping engine for fixture lookup.
        once: If True, run once and exit.
        duration_hours: Maximum duration to run (None = indefinite).
    """
    global _shutdown_requested

    # Create and store session
    session = collector.create_session()
    db.create_snapshot_session(session)

    start_time = utc_now()
    end_time = start_time + timedelta(hours=duration_hours) if duration_hours else None

    logger.info(
        "collection_started",
        session_id=session.session_id,
        interval_minutes=collector.config.interval_minutes,
        duration_hours=duration_hours,
    )

    try:
        while not _shutdown_requested:
            # Check duration limit
            if end_time and utc_now() >= end_time:
                logger.info("duration_limit_reached")
                break

            # Refresh mappings
            mappings = await mapping_engine.get_active_mappings(db, kalshi_client)

            # Get fixture kickoffs
            fixture_kickoffs = await _get_fixture_kickoffs(db, mappings)

            # Collect snapshots
            result = await collector.collect_snapshots(
                mappings=mappings,
                kalshi_client=kalshi_client,
                fixture_kickoffs=fixture_kickoffs,
            )

            # Store snapshots
            for snap in result.snapshot_ids:
                # Snapshots are stored by collector, this is for DB storage
                pass

            # Log result
            logger.info(
                "collection_cycle_complete",
                fixtures=result.fixtures_checked,
                snapshots=result.snapshots_collected,
                errors=len(result.errors),
            )

            # Update session in DB
            updated_session = collector.get_session()
            if updated_session:
                db.update_snapshot_session(updated_session)

            if once:
                break

            # Wait for next interval
            await asyncio.sleep(collector.config.interval_minutes * 60)

    except Exception as e:
        logger.error("collection_error", error=str(e))
        session = collector.complete_session(error=str(e))
        if session:
            db.update_snapshot_session(session)
        raise

    # Complete session
    session = collector.complete_session()
    if session:
        db.update_snapshot_session(session)

    logger.info(
        "collection_completed",
        session_id=collector.session_id,
        total_snapshots=session.snapshots_collected if session else 0,
    )


async def _get_fixture_kickoffs(
    db: Database,
    mappings: list,
) -> dict[int, datetime]:
    """Get kickoff times for mapped fixtures.

    Args:
        db: Database connection.
        mappings: List of fixture-market mappings.

    Returns:
        Dict mapping fixture_id to kickoff time.
    """
    kickoffs = {}
    for mapping in mappings:
        fixture = db.get_fixture_v2(mapping.fixture_id)
        if fixture and fixture.kickoff_utc:
            kickoffs[mapping.fixture_id] = fixture.kickoff_utc
    return kickoffs


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect historical snapshots for strategy backtesting"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Collection interval in minutes (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Maximum duration to run in hours (default: indefinite)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    parser.add_argument(
        "--session-id",
        help="Session ID (for resuming)",
    )
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=50,
        help="Maximum fixtures to track (default: 50)",
    )
    parser.add_argument(
        "--min-hours",
        type=float,
        default=0.5,
        help="Minimum hours to kickoff (default: 0.5)",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=168,
        help="Maximum hours to kickoff (default: 168 = 1 week)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save raw JSON files",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/footbe.db",
        help="Database path (default: data/footbe.db)",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error("config_load_failed", error=str(e))
        return 1

    # Initialize database
    db = Database(args.db_path)
    db.connect()
    db.ensure_schema()

    # Initialize Kalshi client
    kalshi_client = KalshiClient(
        api_key=config.get("kalshi", {}).get("api_key", ""),
        api_secret=config.get("kalshi", {}).get("api_secret", ""),
    )

    # Initialize mapping engine
    mapping_config = MappingConfig()
    mapping_engine = MappingEngine(config=mapping_config)

    # Initialize collector
    collector_config = CollectorConfig(
        interval_minutes=args.interval,
        max_fixtures=args.max_fixtures,
        min_hours_to_kickoff=args.min_hours,
        max_hours_to_kickoff=args.max_hours,
        output_dir=args.output_dir,
    )
    collector = SnapshotCollector(
        config=collector_config,
        session_id=args.session_id,
    )

    try:
        await kalshi_client.connect()

        await run_collection(
            collector=collector,
            db=db,
            kalshi_client=kalshi_client,
            mapping_engine=mapping_engine,
            once=args.once,
            duration_hours=args.duration,
        )

        return 0

    except KeyboardInterrupt:
        logger.info("collection_interrupted")
        return 0

    except Exception as e:
        logger.error("collection_failed", error=str(e))
        return 1

    finally:
        await kalshi_client.close()
        db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
