"""Tests for run modes configuration."""

from pathlib import Path
import tempfile

import pytest

from footbe_trader.agent.objective import AgentObjective
from footbe_trader.agent.run_modes import (
    RunMode,
    RunModeConfig,
    get_mode_config,
    list_available_modes,
    load_config_from_yaml,
    save_config_to_yaml,
    PAPER_CONSERVATIVE_CONFIG,
    PAPER_AGGRESSIVE_CONFIG,
    LIVE_SMALL_CONFIG,
    LIVE_SCALED_CONFIG,
)


class TestRunMode:
    """Tests for RunMode enum."""

    def test_mode_values(self):
        """Test mode values."""
        assert RunMode.PAPER_CONSERVATIVE.value == "paper_conservative"
        assert RunMode.PAPER_AGGRESSIVE.value == "paper_aggressive"
        assert RunMode.LIVE_SMALL.value == "live_small"
        assert RunMode.LIVE_SCALED.value == "live_scaled"

    def test_from_string(self):
        """Test parsing from string."""
        assert RunMode.from_string("paper_conservative") == RunMode.PAPER_CONSERVATIVE
        assert RunMode.from_string("paper_aggressive") == RunMode.PAPER_AGGRESSIVE
        assert RunMode.from_string("live_small") == RunMode.LIVE_SMALL
        assert RunMode.from_string("live_scaled") == RunMode.LIVE_SCALED

    def test_from_string_with_hyphens(self):
        """Test parsing with hyphens."""
        assert RunMode.from_string("paper-conservative") == RunMode.PAPER_CONSERVATIVE
        assert RunMode.from_string("live-small") == RunMode.LIVE_SMALL

    def test_from_string_invalid(self):
        """Test invalid mode string."""
        with pytest.raises(ValueError):
            RunMode.from_string("invalid_mode")


class TestRunModeConfig:
    """Tests for RunModeConfig."""

    def test_paper_conservative_defaults(self):
        """Test paper conservative configuration."""
        config = PAPER_CONSERVATIVE_CONFIG

        assert config.mode == RunMode.PAPER_CONSERVATIVE
        assert config.is_live is False
        assert config.target_weekly_return == 0.03  # 3%
        assert config.max_drawdown == 0.10  # 10%
        assert config.max_gross_exposure == 0.50  # 50%

    def test_paper_aggressive_defaults(self):
        """Test paper aggressive configuration."""
        config = PAPER_AGGRESSIVE_CONFIG

        assert config.mode == RunMode.PAPER_AGGRESSIVE
        assert config.is_live is False
        assert config.target_weekly_return == 0.10  # 10%
        assert config.max_drawdown == 0.15  # 15%
        assert config.max_gross_exposure == 0.80  # 80%

    def test_live_small_defaults(self):
        """Test live small configuration."""
        config = LIVE_SMALL_CONFIG

        assert config.mode == RunMode.LIVE_SMALL
        assert config.is_live is True
        assert config.target_weekly_return == 0.05  # 5%
        assert config.max_drawdown == 0.12  # 12% (tighter)
        assert config.max_daily_loss_pct == 0.03  # 3% daily limit

    def test_live_scaled_defaults(self):
        """Test live scaled configuration."""
        config = LIVE_SCALED_CONFIG

        assert config.mode == RunMode.LIVE_SCALED
        assert config.is_live is True
        assert config.target_weekly_return == 0.08  # 8%
        assert config.max_drawdown == 0.15  # 15%

    def test_to_objective(self):
        """Test converting to AgentObjective."""
        config = PAPER_AGGRESSIVE_CONFIG
        objective = config.to_objective()

        assert isinstance(objective, AgentObjective)
        assert objective.target_weekly_return == config.target_weekly_return
        assert objective.max_drawdown == config.max_drawdown

    def test_to_dict(self):
        """Test serialization."""
        config = PAPER_CONSERVATIVE_CONFIG
        data = config.to_dict()

        assert data["mode"] == "paper_conservative"
        assert data["is_live"] is False
        assert "pacing" in data
        assert "circuit_breakers" in data

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "mode": "paper_aggressive",
            "name": "Test Config",
            "description": "Test description",
            "is_live": False,
            "target_weekly_return": 0.10,
            "max_drawdown": 0.15,
            "pacing": {
                "behind_threshold": 0.7,
                "ahead_threshold": 1.5,
            },
        }

        config = RunModeConfig.from_dict(data)

        assert config.mode == RunMode.PAPER_AGGRESSIVE
        assert config.target_weekly_return == 0.10
        assert config.pacing_behind_threshold == 0.7

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict roundtrip."""
        original = PAPER_AGGRESSIVE_CONFIG
        data = original.to_dict()
        restored = RunModeConfig.from_dict(data)

        assert restored.mode == original.mode
        assert restored.target_weekly_return == original.target_weekly_return
        assert restored.max_drawdown == original.max_drawdown
        assert restored.is_live == original.is_live


class TestConfigFunctions:
    """Tests for configuration helper functions."""

    def test_get_mode_config_by_enum(self):
        """Test getting config by enum."""
        config = get_mode_config(RunMode.PAPER_AGGRESSIVE)
        assert config.mode == RunMode.PAPER_AGGRESSIVE

    def test_get_mode_config_by_string(self):
        """Test getting config by string."""
        config = get_mode_config("paper_conservative")
        assert config.mode == RunMode.PAPER_CONSERVATIVE

    def test_list_available_modes(self):
        """Test listing available modes."""
        modes = list_available_modes()

        assert len(modes) == 4
        mode_names = [m["mode"] for m in modes]
        assert "paper_conservative" in mode_names
        assert "paper_aggressive" in mode_names
        assert "live_small" in mode_names
        assert "live_scaled" in mode_names


class TestYamlPersistence:
    """Tests for YAML file persistence."""

    def test_save_and_load_config(self):
        """Test saving and loading config to YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"

            # Save
            save_config_to_yaml(PAPER_AGGRESSIVE_CONFIG, path)
            assert path.exists()

            # Load
            loaded = load_config_from_yaml(path)

            assert loaded.mode == PAPER_AGGRESSIVE_CONFIG.mode
            assert loaded.target_weekly_return == PAPER_AGGRESSIVE_CONFIG.target_weekly_return
            assert loaded.is_live == PAPER_AGGRESSIVE_CONFIG.is_live

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "config.yaml"

            save_config_to_yaml(LIVE_SMALL_CONFIG, path)
            assert path.exists()

    def test_load_preserves_drawdown_bands(self):
        """Test that drawdown bands are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"

            original = PAPER_AGGRESSIVE_CONFIG
            save_config_to_yaml(original, path)
            loaded = load_config_from_yaml(path)

            # Drawdown bands should match
            assert len(loaded.drawdown_bands) == len(original.drawdown_bands)
            for (t1, m1), (t2, m2) in zip(original.drawdown_bands, loaded.drawdown_bands):
                assert t1 == pytest.approx(t2)
                assert m1 == pytest.approx(m2)
