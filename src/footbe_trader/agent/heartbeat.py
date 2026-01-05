"""Heartbeat agent orchestrator."""

import asyncio
import hashlib
import json
import sys
from typing import Any

import click

from footbe_trader.common.config import AppConfig, load_config
from footbe_trader.common.logging import get_logger, setup_logging
from footbe_trader.common.time_utils import utc_now
from footbe_trader.football.client import FootballApiClient
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.storage.database import Database
from footbe_trader.storage.models import Run, Snapshot

logger = get_logger(__name__)


class Heartbeat:
    """Heartbeat agent that performs health checks and state snapshots.

    This is the main orchestrator for the trading agent. Currently
    implements a simple heartbeat that:
    - Loads configuration
    - Opens database connection
    - Creates a run record
    - Calls stub API clients
    - Creates a snapshot record
    - Exits cleanly
    """

    def __init__(self, config: AppConfig, db: Database):
        """Initialize heartbeat agent.

        Args:
            config: Application configuration.
            db: Database connection.
        """
        self.config = config
        self.db = db
        self._run_id: int | None = None

    @property
    def run_id(self) -> int | None:
        """Get current run ID."""
        return self._run_id

    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for tracking."""
        config_json = json.dumps(self.config.model_dump(), sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    async def run(self) -> int:
        """Execute heartbeat.

        Returns:
            Exit code (0 for success).
        """
        logger.info(
            "heartbeat_starting",
            environment=self.config.environment,
            agent_name=self.config.agent.name,
        )

        # Create run record
        run = Run(
            run_type="heartbeat",
            status="running",
            config_hash=self._compute_config_hash(),
            started_at=utc_now(),
            metadata={
                "environment": self.config.environment,
                "agent_name": self.config.agent.name,
                "dry_run": self.config.agent.dry_run,
            },
        )
        self._run_id = self.db.create_run(run)

        try:
            # Call API clients (stubs)
            snapshot_data = await self._collect_snapshot()

            # Create snapshot record
            snapshot = Snapshot(
                run_id=self._run_id,
                snapshot_type="heartbeat",
                data=snapshot_data,
                created_at=utc_now(),
            )
            snapshot_id = self.db.create_snapshot(snapshot)

            # Mark run as completed
            self.db.complete_run(self._run_id, status="completed")

            logger.info(
                "heartbeat_completed",
                run_id=self._run_id,
                snapshot_id=snapshot_id,
            )
            return 0

        except Exception as e:
            logger.exception("heartbeat_failed", run_id=self._run_id, error=str(e))
            if self._run_id:
                self.db.complete_run(
                    self._run_id, status="failed", error_message=str(e)
                )
            return 1

    async def _collect_snapshot(self) -> dict[str, Any]:
        """Collect snapshot data from API clients.

        Returns:
            Snapshot data dictionary.
        """
        snapshot_data: dict[str, Any] = {
            "timestamp": utc_now().isoformat(),
            "football_api": {},
            "kalshi_api": {},
        }

        # Call Football API client
        async with FootballApiClient(self.config.football_api) as football_client:
            football_healthy = await football_client.health_check()
            snapshot_data["football_api"] = {
                "healthy": football_healthy,
                "fixtures_count": 0,  # Stub returns empty
            }
            logger.info("football_api_checked", healthy=football_healthy)

        # Call Kalshi API client
        async with KalshiClient(self.config.kalshi) as kalshi_client:
            kalshi_healthy = await kalshi_client.health_check()
            positions = await kalshi_client.get_positions()
            snapshot_data["kalshi_api"] = {
                "healthy": kalshi_healthy,
                "positions_count": len(positions),
            }
            logger.info("kalshi_api_checked", healthy=kalshi_healthy)

        return snapshot_data


def run_heartbeat(config_path: str) -> int:
    """Run heartbeat with given config.

    Args:
        config_path: Path to configuration file.

    Returns:
        Exit code.
    """
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Setup logging
    setup_logging(config.logging)

    logger.info("config_loaded", config_path=config_path, environment=config.environment)

    # Open database and run heartbeat
    db = Database(config.database.path)
    try:
        db.connect()
        db.migrate()

        heartbeat = Heartbeat(config, db)
        exit_code = asyncio.run(heartbeat.run())

        return exit_code

    except Exception as e:
        logger.exception("startup_failed", error=str(e))
        return 1

    finally:
        db.close()


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def main(config: str) -> None:
    """Run the heartbeat agent."""
    exit_code = run_heartbeat(config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
