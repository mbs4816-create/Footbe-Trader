"""Tests for configuration loading."""

from pathlib import Path

import pytest
import yaml

from footbe_trader.common.config import AppConfig, load_config


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_valid_config(self, dev_config_path: Path):
        """Test loading a valid configuration file."""
        config = load_config(dev_config_path)

        assert isinstance(config, AppConfig)
        assert config.environment == "test"
        assert config.agent.name == "test-agent"
        assert config.agent.dry_run is True

    def test_load_minimal_config(self, temp_dir: Path):
        """Test loading a minimal configuration with defaults."""
        config_data = {"environment": "minimal"}
        config_path = temp_dir / "minimal.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)

        assert config.environment == "minimal"
        # Check defaults are applied
        assert config.database.path == "data/footbe.db"
        assert config.logging.level == "INFO"
        assert config.agent.dry_run is True

    def test_load_empty_config(self, temp_dir: Path):
        """Test loading an empty configuration file."""
        config_path = temp_dir / "empty.yaml"
        config_path.write_text("")

        config = load_config(config_path)

        # All defaults should be used
        assert config.environment == "dev"
        assert config.database.path == "data/footbe.db"

    def test_load_missing_config(self, temp_dir: Path):
        """Test loading a missing configuration file."""
        config_path = temp_dir / "missing.yaml"

        with pytest.raises(FileNotFoundError):
            load_config(config_path)

    def test_config_database_section(self, dev_config_path: Path):
        """Test database configuration section."""
        config = load_config(dev_config_path)

        assert "test.db" in config.database.path
        assert config.database.echo is False

    def test_config_api_sections(self, dev_config_path: Path):
        """Test API configuration sections."""
        config = load_config(dev_config_path)

        assert config.football_api.api_key == "test_key"
        assert config.football_api.timeout_seconds == 10

        # Kalshi config now uses api_key_id instead of api_key
        assert config.kalshi is not None

    def test_config_logging_section(self, dev_config_path: Path):
        """Test logging configuration section."""
        config = load_config(dev_config_path)

        assert config.logging.level == "DEBUG"
        assert config.logging.format == "console"
        assert config.logging.log_file is None
