"""Tests for trading policy module."""

import pytest

from footbe_trader.agent.objective import (
    AgentObjective,
    DrawdownBand,
    PacingState,
    TimeToKickoffCategory,
)
from footbe_trader.agent.policy import (
    ExposureState,
    PolicyDecision,
    TradingPolicy,
)
from footbe_trader.common.time_utils import utc_now


class TestExposureState:
    """Tests for ExposureState dataclass."""

    def test_create_empty_state(self):
        """Test creating empty exposure state."""
        state = ExposureState()
        assert state.gross_exposure == 0.0
        assert len(state.exposure_by_fixture) == 0
        assert len(state.exposure_by_league) == 0

    def test_exposure_tracking(self):
        """Test tracking exposures by fixture and league."""
        state = ExposureState(
            gross_exposure=5000.0,
            exposure_by_fixture={12345: 1000.0},
            exposure_by_league={"EPL": 3000.0},
        )
        assert state.gross_exposure == 5000.0
        assert state.exposure_by_fixture[12345] == 1000.0
        assert state.exposure_by_league["EPL"] == 3000.0

    def test_get_fixture_exposure(self):
        """Test get_fixture_exposure method."""
        state = ExposureState(
            exposure_by_fixture={12345: 1000.0},
        )
        assert state.get_fixture_exposure(12345) == 1000.0
        assert state.get_fixture_exposure(99999) == 0.0  # Missing

    def test_get_league_exposure(self):
        """Test get_league_exposure method."""
        state = ExposureState(
            exposure_by_league={"EPL": 3000.0},
        )
        assert state.get_league_exposure("EPL") == 3000.0
        assert state.get_league_exposure("LaLiga") == 0.0  # Missing


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_create_decision(self):
        """Test creating a policy decision."""
        decision = PolicyDecision(
            fixture_id=12345,
            league="EPL",
            outcome="home_win",
            hours_to_kickoff=4.0,
            raw_edge=0.05,
            raw_position_size=100.0,
            equity=100000.0,
            pacing_state=PacingState.ON_PACE,
            drawdown=0.02,
            drawdown_band=DrawdownBand.NONE,
            time_category="optimal",
            current_fixture_exposure=0.0,
            current_league_exposure=0.0,
            current_gross_exposure=0.0,
            max_fixture_exposure=1500.0,
            max_league_exposure=20000.0,
            max_gross_exposure=80000.0,
        )
        assert decision.fixture_id == 12345
        assert decision.raw_edge == 0.05
        assert decision.pacing_state == PacingState.ON_PACE

    def test_to_dict(self):
        """Test serialization."""
        decision = PolicyDecision(
            fixture_id=12345,
            league="EPL",
            outcome="home_win",
            hours_to_kickoff=4.0,
            raw_edge=0.03,
            raw_position_size=100.0,
            equity=100000.0,
            pacing_state=PacingState.ON_PACE,
            drawdown=0.02,
            drawdown_band=DrawdownBand.NONE,
            time_category=TimeToKickoffCategory.OPTIMAL,
            current_fixture_exposure=0.0,
            current_league_exposure=0.0,
            current_gross_exposure=0.0,
            max_fixture_exposure=1500.0,
            max_league_exposure=20000.0,
            max_gross_exposure=80000.0,
        )
        data = decision.to_dict()
        assert "fixture_id" in data
        assert "raw_edge" in data


class TestTradingPolicy:
    """Tests for TradingPolicy."""

    def test_initialization(self):
        """Test policy initialization."""
        obj = AgentObjective(target_weekly_return=0.10)
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        assert policy.objective == obj

    def test_update_equity(self):
        """Test updating equity."""
        obj = AgentObjective(target_weekly_return=0.10)
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        policy.update_equity(105000.0)

        assert policy.pacing_tracker.current_equity == 105000.0

    def test_update_exposure(self):
        """Test updating exposure state."""
        obj = AgentObjective(target_weekly_return=0.10)
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        policy.update_exposure(
            gross_exposure=10000.0,
            exposure_by_fixture={12345: 1000.0},
            exposure_by_league={"EPL": 5000.0},
            position_count=5,
        )

        assert policy.exposure_state.gross_exposure == 10000.0

    def test_evaluate_good_trade(self):
        """Test evaluating a good trade."""
        obj = AgentObjective(
            target_weekly_return=0.10,
            max_exposure_per_fixture=0.02,  # 2%
        )
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        decision = policy.evaluate_trade(
            fixture_id=12345,
            league="EPL",
            outcome="home_win",
            model_prob=0.55,
            market_price=0.50,
            base_edge_threshold=0.02,
            kelly_position_size=100,
            hours_to_kickoff=4.0,  # Optimal window
            price_per_contract=0.50,
        )

        assert decision.raw_edge == pytest.approx(0.05)  # 55% - 50%
        # Should be approved unless pacing or other factors reject it

    def test_evaluate_low_edge_trade(self):
        """Test rejecting low edge trade."""
        obj = AgentObjective(target_weekly_return=0.10)
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        decision = policy.evaluate_trade(
            fixture_id=12345,
            league="EPL",
            outcome="home_win",
            model_prob=0.51,
            market_price=0.50,  # Only 1% edge
            base_edge_threshold=0.02,  # Requires 2% edge
            kelly_position_size=100,
            hours_to_kickoff=4.0,
            price_per_contract=0.50,
        )

        assert decision.raw_edge == pytest.approx(0.01)
        # Should be rejected due to low edge
        assert decision.can_trade is False or len(decision.rejection_reasons) > 0

    def test_evaluate_high_drawdown_blocks(self):
        """Test that severe drawdown blocks trades."""
        obj = AgentObjective(target_weekly_return=0.10, max_drawdown=0.15)
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        # Simulate severe drawdown
        policy.update_equity(120000.0)  # Peak
        policy.update_equity(95000.0)   # >20% drawdown

        decision = policy.evaluate_trade(
            fixture_id=12345,
            league="EPL",
            outcome="home_win",
            model_prob=0.60,
            market_price=0.50,  # 10% edge
            base_edge_threshold=0.02,
            kelly_position_size=100,
            hours_to_kickoff=4.0,
            price_per_contract=0.50,
        )

        # Severe drawdown should block
        assert decision.can_trade is False
        assert any("drawdown" in r.lower() for r in decision.rejection_reasons)

    def test_evaluate_exposure_limit(self):
        """Test exposure limit enforcement."""
        obj = AgentObjective(
            target_weekly_return=0.10,
            max_exposure_per_fixture=0.015,  # 1.5%
        )
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        # Set existing exposure near limit
        policy.update_exposure(
            gross_exposure=5000.0,
            exposure_by_fixture={12345: 1400.0},  # Near 1.5% limit (1500)
            exposure_by_league={"EPL": 5000.0},
            position_count=5,
        )

        decision = policy.evaluate_trade(
            fixture_id=12345,
            league="EPL",
            outcome="home_win",
            model_prob=0.60,
            market_price=0.50,  # Good edge
            base_edge_threshold=0.02,
            kelly_position_size=500,  # Would push over limit
            hours_to_kickoff=4.0,
            price_per_contract=0.50,
        )

        # Should either reject or reduce size significantly
        if decision.can_trade:
            # Final size should be capped
            assert decision.final_position_size < 500

    def test_get_status(self):
        """Test status retrieval."""
        obj = AgentObjective(target_weekly_return=0.10)
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        policy.update_equity(105000.0)

        status = policy.get_status()

        assert "equity" in status
        assert "drawdown" in status
        assert "pacing_state" in status
        assert status["equity"] == 105000.0

    def test_generate_status_report(self):
        """Test status report generation."""
        obj = AgentObjective(target_weekly_return=0.10)
        policy = TradingPolicy(objective=obj, initial_equity=100000.0)

        policy.update_equity(105000.0)

        report = policy.generate_status_report()

        assert isinstance(report, str)
        assert "TRADING POLICY STATUS" in report
        assert "105" in report  # Should show equity
