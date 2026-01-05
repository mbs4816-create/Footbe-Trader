"""Tests for agent objective module."""

from datetime import timedelta

import pytest

from footbe_trader.agent.objective import (
    AgentObjective,
    DrawdownBand,
    DrawdownThrottle,
    EquitySnapshot,
    PacingAdjustment,
    PacingState,
    PacingTracker,
    TimeToKickoffCategory,
    classify_time_to_kickoff,
    get_time_to_kickoff_adjustment,
)
from footbe_trader.common.time_utils import utc_now


class TestAgentObjective:
    """Tests for AgentObjective dataclass."""

    def test_default_values(self):
        """Test default objective values."""
        obj = AgentObjective()
        assert obj.target_weekly_return == 0.10  # 10%
        assert obj.max_drawdown == 0.15  # 15%
        assert obj.max_exposure_per_fixture == 0.015  # 1.5%
        assert obj.max_exposure_per_league == 0.20  # 20%
        assert obj.max_gross_exposure == 0.80  # 80%

    def test_custom_values(self):
        """Test custom objective values."""
        obj = AgentObjective(
            target_weekly_return=0.05,
            max_drawdown=0.10,
            max_exposure_per_fixture=0.02,
        )
        assert obj.target_weekly_return == 0.05
        assert obj.max_drawdown == 0.10
        assert obj.max_exposure_per_fixture == 0.02

    def test_to_dict(self):
        """Test serialization."""
        obj = AgentObjective()
        data = obj.to_dict()
        assert "target_weekly_return" in data
        assert "max_drawdown" in data
        assert data["target_weekly_return"] == 0.10


class TestDrawdownBand:
    """Tests for DrawdownBand enum."""

    def test_band_values(self):
        """Test band string values."""
        assert DrawdownBand.NONE.value == "none"
        assert DrawdownBand.LIGHT.value == "light"
        assert DrawdownBand.MODERATE.value == "moderate"
        assert DrawdownBand.SEVERE.value == "severe"


class TestDrawdownThrottle:
    """Tests for DrawdownThrottle."""

    def test_no_drawdown(self):
        """Test with no drawdown."""
        throttle = DrawdownThrottle()
        band = throttle.get_band(0.0)
        mult = throttle.get_multiplier(0.0)

        assert band == DrawdownBand.NONE
        assert mult == 1.0

    def test_light_drawdown(self):
        """Test light drawdown (0-5%)."""
        throttle = DrawdownThrottle()
        band = throttle.get_band(0.03)
        mult = throttle.get_multiplier(0.03)

        assert band == DrawdownBand.NONE  # Under 5% threshold
        assert mult == 1.0

    def test_moderate_drawdown(self):
        """Test moderate drawdown (5-10%)."""
        throttle = DrawdownThrottle()
        band = throttle.get_band(0.07)
        mult = throttle.get_multiplier(0.07)

        assert band == DrawdownBand.LIGHT
        assert mult == 0.7

    def test_heavy_drawdown(self):
        """Test heavy drawdown (10-15%)."""
        throttle = DrawdownThrottle()
        band = throttle.get_band(0.12)
        mult = throttle.get_multiplier(0.12)

        assert band == DrawdownBand.MODERATE
        assert mult == 0.4

    def test_severe_drawdown(self):
        """Test severe drawdown (>15%)."""
        throttle = DrawdownThrottle()
        band = throttle.get_band(0.20)
        mult = throttle.get_multiplier(0.20)

        assert band == DrawdownBand.SEVERE
        assert mult == 0.0

    def test_can_enter(self):
        """Test can_enter method."""
        throttle = DrawdownThrottle()
        assert throttle.can_enter(0.03) is True  # NONE
        assert throttle.can_enter(0.07) is True  # LIGHT
        assert throttle.can_enter(0.12) is True  # MODERATE
        assert throttle.can_enter(0.20) is False  # SEVERE


class TestEquitySnapshot:
    """Tests for EquitySnapshot."""

    def test_create_snapshot(self):
        """Test creating an equity snapshot."""
        snap = EquitySnapshot(
            timestamp=utc_now(),
            equity=105000.0,
            peak_equity=110000.0,
            drawdown=0.05,
        )
        assert snap.equity == 105000.0
        assert snap.peak_equity == 110000.0
        assert snap.drawdown == 0.05

    def test_to_dict(self):
        """Test serialization."""
        snap = EquitySnapshot(
            equity=100000.0,
            peak_equity=100000.0,
            drawdown=0.0,
        )
        data = snap.to_dict()
        assert data["equity"] == 100000.0
        assert "timestamp" in data


class TestPacingState:
    """Tests for PacingState enum."""

    def test_state_values(self):
        """Test state values."""
        assert PacingState.BEHIND_PACE.value == "behind_pace"
        assert PacingState.ON_PACE.value == "on_pace"
        assert PacingState.AHEAD_OF_PACE.value == "ahead_of_pace"


class TestPacingTracker:
    """Tests for PacingTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        obj = AgentObjective(target_weekly_return=0.10)
        tracker = PacingTracker(objective=obj, initial_equity=100000.0)

        assert tracker.initial_equity == 100000.0
        assert tracker.peak_equity == 100000.0
        assert tracker.current_equity == 100000.0

    def test_record_equity(self):
        """Test equity recording."""
        obj = AgentObjective(target_weekly_return=0.10)
        tracker = PacingTracker(objective=obj, initial_equity=100000.0)

        snap = tracker.record_equity(105000.0)

        assert tracker.current_equity == 105000.0
        assert tracker.peak_equity == 105000.0
        assert snap.equity == 105000.0

    def test_peak_tracking(self):
        """Test peak equity tracking."""
        obj = AgentObjective(target_weekly_return=0.10)
        tracker = PacingTracker(objective=obj, initial_equity=100000.0)

        tracker.record_equity(110000.0)
        tracker.record_equity(105000.0)

        assert tracker.peak_equity == 110000.0
        assert tracker.current_equity == 105000.0

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        obj = AgentObjective(target_weekly_return=0.10)
        tracker = PacingTracker(objective=obj, initial_equity=100000.0)

        tracker.record_equity(110000.0)
        tracker.record_equity(99000.0)

        drawdown = tracker.current_drawdown
        assert drawdown == pytest.approx(0.10)  # 10% down from 110k

    def test_pacing_state(self):
        """Test pacing state classification."""
        obj = AgentObjective(target_weekly_return=0.10)
        tracker = PacingTracker(objective=obj, initial_equity=100000.0)

        state = tracker.get_pacing_state()
        # At start, rolling return is 0, target is 10%, so behind pace
        assert state == PacingState.BEHIND_PACE

    def test_pacing_adjustment(self):
        """Test pacing adjustment generation."""
        obj = AgentObjective(target_weekly_return=0.10)
        tracker = PacingTracker(objective=obj, initial_equity=100000.0)

        tracker.record_equity(105000.0)

        adjustment = tracker.get_pacing_adjustment()
        assert isinstance(adjustment, PacingAdjustment)
        assert adjustment.sizing_multiplier >= 0.5
        assert adjustment.sizing_multiplier <= 2.0

    def test_rolling_7d_return(self):
        """Test rolling 7 day return calculation."""
        obj = AgentObjective(target_weekly_return=0.10)
        tracker = PacingTracker(objective=obj, initial_equity=100000.0)

        # Start at 100k, go to 110k
        tracker.record_equity(110000.0)

        # Return should be positive
        ret = tracker.get_rolling_7d_return()
        assert ret >= 0


class TestTimeToKickoff:
    """Tests for time to kickoff functions."""

    def test_classify_very_early(self):
        """Test very early classification (>48h)."""
        cat = classify_time_to_kickoff(60.0)
        assert cat == TimeToKickoffCategory.VERY_EARLY

    def test_classify_early(self):
        """Test early classification (24-48h)."""
        cat = classify_time_to_kickoff(36.0)
        assert cat == TimeToKickoffCategory.EARLY

    def test_classify_standard(self):
        """Test standard classification (6-24h)."""
        cat = classify_time_to_kickoff(12.0)
        assert cat == TimeToKickoffCategory.STANDARD

    def test_classify_optimal(self):
        """Test optimal classification (2-6h)."""
        cat = classify_time_to_kickoff(4.0)
        assert cat == TimeToKickoffCategory.OPTIMAL

    def test_classify_late(self):
        """Test late classification (<2h)."""
        cat = classify_time_to_kickoff(1.0)
        assert cat == TimeToKickoffCategory.LATE

    def test_adjustment_optimal_baseline(self):
        """Test optimal window has baseline multipliers."""
        edge_mult, size_mult, reason = get_time_to_kickoff_adjustment(4.0)
        assert edge_mult == 1.0
        assert size_mult == 1.0
        assert "optimal" in reason.lower()

    def test_adjustment_very_early_premium(self):
        """Test very early has edge premium."""
        edge_mult, size_mult, reason = get_time_to_kickoff_adjustment(60.0)
        assert edge_mult > 1.0  # Higher edge required
        assert size_mult < 1.0  # Smaller size
        assert "very early" in reason.lower() or "uncertainty" in reason.lower()

    def test_adjustment_late_premium(self):
        """Test late has edge premium."""
        edge_mult, size_mult, reason = get_time_to_kickoff_adjustment(0.5)
        assert edge_mult > 1.0  # Higher edge required
        assert size_mult < 1.0  # Smaller size
