"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
import yaml

from footbe_trader.common.config import AppConfig, load_config
from footbe_trader.storage.database import Database


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dev_config_path(temp_dir: Path) -> Path:
    """Create a temporary dev config file."""
    config_data = {
        "environment": "test",
        "database": {
            "path": str(temp_dir / "test.db"),
            "echo": False,
        },
        "football_api": {
            "base_url": "https://api-football-v1.p.rapidapi.com/v3",
            "api_key": "test_key",
            "timeout_seconds": 10,
            "rate_limit_per_minute": 30,
        },
        "kalshi": {
            "base_url": "https://demo-api.kalshi.co/trade-api/v2",
            "api_key": "test_key",
            "api_secret": "test_secret",
            "timeout_seconds": 10,
        },
        "logging": {
            "level": "DEBUG",
            "format": "console",
            "log_file": None,
        },
        "agent": {
            "name": "test-agent",
            "heartbeat_interval_seconds": 60,
            "dry_run": True,
        },
    }
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def config(dev_config_path: Path) -> AppConfig:
    """Load test configuration."""
    return load_config(dev_config_path)


@pytest.fixture
def db(temp_dir: Path) -> Database:
    """Create a test database."""
    db_path = temp_dir / "test.db"
    database = Database(db_path)
    database.connect()
    database.migrate()
    yield database
    database.close()
