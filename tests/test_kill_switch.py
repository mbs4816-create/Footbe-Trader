"""Tests for kill-switch system."""

import math
import time
from datetime import UTC, datetime, timedelta

import pytest

from footbe_trader.execution.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    KillSwitchManager,
    KillSwitchReason,
    KillSwitchStatus,
    TradingState,
)


class TestKillSwitchConfig:
    """Tests for KillSwitchConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = KillSwitchConfig()

        assert config.daily_loss_limit_enabled is True
        assert config.daily_loss_limit_dollars == 500.0
        assert config.max_exposure_enabled is True
        assert config.max_exposure_dollars == 2000.0
        assert config.api_error_rate_enabled is True
        assert config.model_health_enabled is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = KillSwitchConfig(
            daily_loss_limit_dollars=200.0,
            max_exposure_dollars=1000.0,
            api_error_threshold=5,
        )

        assert config.daily_loss_limit_dollars == 200.0
        assert config.max_exposure_dollars == 1000.0
        assert config.api_error_threshold == 5

    def test_to_dict(self):
        """Test configuration serialization."""
        config = KillSwitchConfig()
        data = config.to_dict()

        assert "daily_loss_limit" in data
        assert data["daily_loss_limit"]["enabled"] is True
        assert "max_exposure" in data
        assert "api_error_rate" in data
        assert "model_health" in data


class TestKillSwitch:
    """Tests for individual KillSwitch."""

    def test_creation(self):
        """Test kill-switch creation."""
        switch = KillSwitch(
            name="Test Switch",
            reason=KillSwitchReason.DAILY_LOSS_LIMIT,
            threshold=500.0,
        )

        assert switch.name == "Test Switch"
        assert switch.reason == KillSwitchReason.DAILY_LOSS_LIMIT
        assert switch.enabled is True
        assert switch.tripped is False
        assert switch.threshold == 500.0

    def test_trip(self):
        """Test tripping a kill-switch."""
        switch = KillSwitch(
            name="Test",
            reason=KillSwitchReason.DAILY_LOSS_LIMIT,
        )

        switch.trip(value=-600.0, message="Loss limit exceeded")

        assert switch.tripped is True
        assert switch.trip_value == -600.0
        assert switch.trip_time is not None
        assert "Loss limit" in switch.message

    def test_reset(self):
        """Test resetting a kill-switch."""
        switch = KillSwitch(
            name="Test",
            reason=KillSwitchReason.DAILY_LOSS_LIMIT,
        )
        switch.trip(value=-600.0)

        switch.reset()

        assert switch.tripped is False
        assert switch.trip_value is None
        assert switch.trip_time is None

    def test_to_dict(self):
        """Test serialization."""
        switch = KillSwitch(
            name="Test",
            reason=KillSwitchReason.MAX_EXPOSURE,
            threshold=1000.0,
        )

        data = switch.to_dict()

        assert data["name"] == "Test"
        assert data["reason"] == "max_exposure"
        assert data["threshold"] == 1000.0


class TestTradingState:
    """Tests for TradingState."""

    def test_default_values(self):
        """Test default state."""
        state = TradingState()

        assert state.daily_pnl == 0.0
        assert state.current_exposure == 0.0
        assert state.open_positions_count == 0

    def test_reset_daily(self):
        """Test daily reset."""
        state = TradingState()
        state.daily_pnl = -200.0

        state.reset_daily()

        assert state.daily_pnl == 0.0


class TestKillSwitchManager:
    """Tests for KillSwitchManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = KillSwitchManager()

        assert manager.status == KillSwitchStatus.ACTIVE
        assert manager.can_trade() is True

    def test_custom_config(self):
        """Test with custom configuration."""
        config = KillSwitchConfig(
            daily_loss_limit_dollars=100.0,
            max_exposure_dollars=500.0,
        )
        manager = KillSwitchManager(config)

        assert manager.config.daily_loss_limit_dollars == 100.0
        assert manager.config.max_exposure_dollars == 500.0

    def test_daily_loss_limit_trip(self):
        """Test daily loss limit kill-switch."""
        config = KillSwitchConfig(daily_loss_limit_dollars=100.0)
        manager = KillSwitchManager(config)

        # Record losses
        manager.record_trade(pnl=-50.0)
        assert manager.can_trade() is True

        manager.record_trade(pnl=-60.0)  # Total: -110
        assert manager.can_trade() is False
        assert manager.status == KillSwitchStatus.TRIPPED
        assert manager.get_trip_reason() == KillSwitchReason.DAILY_LOSS_LIMIT

    def test_max_exposure_trip(self):
        """Test max exposure kill-switch."""
        config = KillSwitchConfig(max_exposure_dollars=500.0)
        manager = KillSwitchManager(config)

        # Build exposure
        manager.update_exposure(400.0)
        assert manager.can_trade() is True

        manager.update_exposure(600.0)
        assert manager.can_trade() is False
        assert manager.get_trip_reason() == KillSwitchReason.MAX_EXPOSURE

    def test_api_error_count_trip(self):
        """Test API error rate kill-switch by count."""
        config = KillSwitchConfig(
            api_error_threshold=5,
            api_error_window_seconds=60,
        )
        manager = KillSwitchManager(config)

        # Record errors
        for _ in range(4):
            manager.record_api_call(success=False, error="timeout")
        assert manager.can_trade() is True

        manager.record_api_call(success=False, error="timeout")
        assert manager.can_trade() is False
        assert manager.get_trip_reason() == KillSwitchReason.API_ERROR_RATE

    def test_api_error_rate_trip(self):
        """Test API error rate kill-switch by percentage."""
        config = KillSwitchConfig(
            api_error_threshold=100,  # High count threshold
            api_error_rate_threshold=0.5,  # 50% rate
        )
        manager = KillSwitchManager(config)

        # Record mixed calls - need enough for rate calculation
        for _ in range(6):
            manager.record_api_call(success=True)
        for _ in range(6):
            manager.record_api_call(success=False)

        # 50% error rate should trip
        assert manager.can_trade() is False

    def test_model_health_valid(self):
        """Test model health validation with valid probabilities."""
        manager = KillSwitchManager()

        probs = {
            "home_win": 0.4,
            "draw": 0.3,
            "away_win": 0.3,
        }

        assert manager.validate_model_health(probs) is True
        assert manager.can_trade() is True

    def test_model_health_nan(self):
        """Test model health validation with NaN."""
        manager = KillSwitchManager()

        probs = {
            "home_win": float("nan"),
            "draw": 0.3,
            "away_win": 0.3,
        }

        assert manager.validate_model_health(probs) is False
        assert manager.can_trade() is False
        assert manager.get_trip_reason() == KillSwitchReason.MODEL_HEALTH

    def test_model_health_sum_not_one(self):
        """Test model health validation with incorrect sum."""
        manager = KillSwitchManager()

        probs = {
            "home_win": 0.5,
            "draw": 0.5,
            "away_win": 0.5,
        }  # Sum = 1.5

        assert manager.validate_model_health(probs) is False
        assert manager.get_trip_reason() == KillSwitchReason.MODEL_HEALTH

    def test_model_health_out_of_range(self):
        """Test model health with out-of-range probability."""
        manager = KillSwitchManager()

        probs = {
            "home_win": 1.5,  # Invalid
            "draw": -0.2,  # Invalid
            "away_win": -0.3,  # Invalid
        }

        assert manager.validate_model_health(probs) is False

    def test_manual_halt(self):
        """Test manual trading halt."""
        manager = KillSwitchManager()

        manager.halt_trading(reason="maintenance")

        assert manager.can_trade() is False
        assert manager.status == KillSwitchStatus.MANUAL_HALT
        assert manager.get_trip_reason() == KillSwitchReason.MANUAL

    def test_manual_resume(self):
        """Test manual trading resume."""
        config = KillSwitchConfig(require_manual_reset=False)
        manager = KillSwitchManager(config)

        manager.halt_trading()
        assert manager.can_trade() is False

        manager.resume_trading()
        assert manager.can_trade() is True

    def test_reset_switch(self):
        """Test resetting individual switch."""
        config = KillSwitchConfig(daily_loss_limit_dollars=100.0)
        manager = KillSwitchManager(config)

        # Trip the switch
        manager.record_trade(pnl=-150.0)
        assert manager.can_trade() is False

        # Reset it
        manager.reset_switch(KillSwitchReason.DAILY_LOSS_LIMIT)
        assert manager.can_trade() is True

    def test_reset_all_switches(self):
        """Test resetting all switches."""
        config = KillSwitchConfig(
            daily_loss_limit_dollars=100.0,
            max_exposure_dollars=100.0,
        )
        manager = KillSwitchManager(config)

        # Trip multiple switches
        manager.record_trade(pnl=-150.0)
        manager.update_exposure(200.0)

        tripped = manager.get_tripped_switches()
        assert len(tripped) == 2

        # Reset all
        manager.reset_all_switches()
        assert manager.can_trade() is True
        assert len(manager.get_tripped_switches()) == 0

    def test_get_status_report(self):
        """Test status report generation."""
        manager = KillSwitchManager()
        report = manager.get_status_report()

        assert "status" in report
        assert "can_trade" in report
        assert "state" in report
        assert "switches" in report
        assert "api_stats" in report

    def test_disabled_switch_does_not_trip(self):
        """Test that disabled switches don't trip."""
        config = KillSwitchConfig(
            daily_loss_limit_enabled=False,
            daily_loss_limit_dollars=100.0,
        )
        manager = KillSwitchManager(config)

        manager.record_trade(pnl=-200.0)
        assert manager.can_trade() is True  # Switch disabled, should still trade


class TestKillSwitchManagerDailyReset:
    """Tests for daily reset behavior."""

    def test_daily_reset_clears_pnl(self):
        """Test that daily reset clears P&L tracking."""
        manager = KillSwitchManager()

        # Record a loss
        manager.record_trade(pnl=-100.0)
        assert manager.state.daily_pnl == -100.0

        # Simulate day change by modifying reset time
        manager.state.daily_pnl_reset_time = datetime.now(UTC) - timedelta(days=2)

        # Next trade should trigger reset
        manager.record_trade(pnl=-10.0)
        assert manager.state.daily_pnl == -10.0  # Reset to just this trade


class TestKillSwitchManagerApiTracking:
    """Tests for API call tracking."""

    def test_api_calls_expire_from_window(self):
        """Test that old API calls are pruned."""
        config = KillSwitchConfig(api_error_window_seconds=1)
        manager = KillSwitchManager(config)

        # Record some calls
        manager.record_api_call(success=False)
        assert len(manager._api_calls) == 1

        # Wait for window to expire
        time.sleep(1.1)

        # New call should prune old ones
        manager.record_api_call(success=True)
        assert len(manager._api_calls) == 1  # Only the new call


class TestKillSwitchManagerRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test string representation."""
        manager = KillSwitchManager()
        repr_str = repr(manager)

        assert "KillSwitchManager" in repr_str
        assert "active" in repr_str
        assert "can_trade=True" in repr_str
