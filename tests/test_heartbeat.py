"""Tests for heartbeat agent end-to-end."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from footbe_trader.agent.heartbeat import Heartbeat, run_heartbeat
from footbe_trader.common.config import AppConfig, load_config
from footbe_trader.common.logging import setup_logging
from footbe_trader.storage.database import Database


class TestHeartbeat:
    """Test heartbeat agent."""

    @pytest.fixture
    def test_config(self, temp_dir: Path) -> AppConfig:
        """Create test configuration."""
        config_data = {
            "environment": "test",
            "database": {
                "path": str(temp_dir / "heartbeat_test.db"),
            },
            "football_api": {
                "api_key": "test",
            },
            "kalshi": {
                "api_key": "test",
                "api_secret": "test",
            },
            "logging": {
                "level": "DEBUG",
                "format": "console",
            },
            "agent": {
                "name": "test-heartbeat",
                "dry_run": True,
            },
        }
        config_path = temp_dir / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return load_config(config_path)

    @pytest.fixture
    def test_db(self, test_config: AppConfig) -> Database:
        """Create test database."""
        db = Database(test_config.database.path)
        db.connect()
        db.migrate()
        yield db
        db.close()

    @pytest.mark.asyncio
    async def test_heartbeat_run(self, test_config: AppConfig, test_db: Database):
        """Test heartbeat creates run and snapshot records."""
        setup_logging(test_config.logging)

        heartbeat = Heartbeat(test_config, test_db)

        with patch(
            "footbe_trader.football.client.FootballApiClient.health_check",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "footbe_trader.kalshi.client.KalshiClient.health_check",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "footbe_trader.kalshi.client.KalshiClient.get_positions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            exit_code = await heartbeat.run()

        assert exit_code == 0
        assert heartbeat.run_id is not None

        # Verify run record
        run = test_db.get_run(heartbeat.run_id)
        assert run is not None
        assert run.run_type == "heartbeat"
        assert run.status == "completed"
        assert run.completed_at is not None

        # Verify snapshot record
        snapshots = test_db.get_snapshots_for_run(heartbeat.run_id)
        assert len(snapshots) == 1
        assert snapshots[0].snapshot_type == "heartbeat"
        assert "football_api" in snapshots[0].data
        assert "kalshi_api" in snapshots[0].data

    @pytest.mark.asyncio
    async def test_heartbeat_handles_errors(
        self, test_config: AppConfig, test_db: Database
    ):
        """Test heartbeat handles errors gracefully."""
        setup_logging(test_config.logging)

        heartbeat = Heartbeat(test_config, test_db)

        with patch(
            "footbe_trader.football.client.FootballApiClient.health_check",
            new_callable=AsyncMock,
            side_effect=Exception("API connection failed"),
        ):
            exit_code = await heartbeat.run()

        assert exit_code == 1
        assert heartbeat.run_id is not None

        # Verify run marked as failed
        run = test_db.get_run(heartbeat.run_id)
        assert run is not None
        assert run.status == "failed"
        assert "API connection failed" in (run.error_message or "")


class TestRunHeartbeat:
    """Test run_heartbeat function (integration)."""

    def test_run_heartbeat_end_to_end(self, temp_dir: Path):
        """Test complete heartbeat run from config to DB."""
        # Create config
        config_data = {
            "environment": "e2e_test",
            "database": {
                "path": str(temp_dir / "e2e_test.db"),
            },
            "logging": {
                "level": "WARNING",
                "format": "console",
            },
            "agent": {
                "name": "e2e-test-agent",
                "dry_run": True,
            },
        }
        config_path = temp_dir / "e2e_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Run heartbeat
        with patch(
            "footbe_trader.football.client.FootballApiClient.health_check",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "footbe_trader.kalshi.client.KalshiClient.health_check",
            new_callable=AsyncMock,
            return_value=True,
        ), patch(
            "footbe_trader.kalshi.client.KalshiClient.get_positions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            exit_code = run_heartbeat(str(config_path))

        assert exit_code == 0

        # Verify database state
        db = Database(temp_dir / "e2e_test.db")
        db.connect()

        # Check run exists
        cursor = db.execute("SELECT COUNT(*) FROM runs WHERE status = 'completed'")
        count = cursor.fetchone()[0]
        assert count == 1

        # Check snapshot exists
        cursor = db.execute("SELECT COUNT(*) FROM snapshots WHERE snapshot_type = 'heartbeat'")
        count = cursor.fetchone()[0]
        assert count == 1

        db.close()

    def test_run_heartbeat_missing_config(self, temp_dir: Path):
        """Test run_heartbeat with missing config file."""
        missing_path = temp_dir / "missing.yaml"
        exit_code = run_heartbeat(str(missing_path))
        assert exit_code == 1
