"""Tests for decision record types."""

from datetime import UTC, datetime
import pytest

from footbe_trader.strategy.decision_record import (
    AgentRunSummary,
    DecisionAction,
    DecisionRecord,
    EdgeCalculation,
    ExitReason,
    KellySizing,
    MarketSnapshot,
    ModelPrediction,
    OrderParams,
    PositionState,
)


class TestDecisionAction:
    """Tests for DecisionAction enum."""

    def test_values(self):
        """Should have correct values."""
        assert DecisionAction.BUY.value == "buy"
        assert DecisionAction.SELL.value == "sell"
        assert DecisionAction.HOLD.value == "hold"
        assert DecisionAction.EXIT.value == "exit"
        assert DecisionAction.SKIP.value == "skip"


class TestExitReason:
    """Tests for ExitReason enum."""

    def test_values(self):
        """Should have correct values."""
        assert ExitReason.TAKE_PROFIT.value == "take_profit"
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.EDGE_FLIP.value == "edge_flip"
        assert ExitReason.MARKET_CLOSE.value == "market_close"


class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass."""

    def test_creation(self):
        """Should create snapshot with all fields."""
        snapshot = MarketSnapshot(
            ticker="TEST-TICKER",
            best_bid=0.40,
            best_ask=0.42,
            mid_price=0.41,
            spread=0.02,
            bid_volume=100,
            ask_volume=150,
            status="open",
        )
        
        assert snapshot.ticker == "TEST-TICKER"
        assert snapshot.mid_price == 0.41
        assert snapshot.spread == 0.02

    def test_to_dict(self):
        """Should convert to dictionary."""
        snapshot = MarketSnapshot(
            ticker="TEST",
            best_bid=0.40,
            best_ask=0.42,
        )
        
        d = snapshot.to_dict()
        
        assert d["ticker"] == "TEST"
        assert d["best_bid"] == 0.40
        assert d["best_ask"] == 0.42


class TestModelPrediction:
    """Tests for ModelPrediction dataclass."""

    def test_creation(self):
        """Should create prediction with probabilities."""
        pred = ModelPrediction(
            model_name="test_model",
            model_version="1.0.0",
            prob_home_win=0.45,
            prob_draw=0.30,
            prob_away_win=0.25,
            confidence=0.8,
        )
        
        assert pred.model_name == "test_model"
        assert pred.prob_home_win == 0.45
        assert pred.confidence == 0.8

    def test_to_dict(self):
        """Should convert to dictionary."""
        pred = ModelPrediction(
            model_name="test",
            model_version="1.0",
            prob_home_win=0.5,
            prob_draw=0.25,
            prob_away_win=0.25,
        )
        
        d = pred.to_dict()
        
        assert d["model_name"] == "test"
        assert d["prob_home_win"] == 0.5
        assert d["prob_draw"] == 0.25

    def test_probabilities_sum(self):
        """Probabilities should sum to 1."""
        pred = ModelPrediction(
            model_name="test",
            model_version="1.0",
            prob_home_win=0.45,
            prob_draw=0.30,
            prob_away_win=0.25,
        )
        
        total = pred.prob_home_win + pred.prob_draw + pred.prob_away_win
        assert total == pytest.approx(1.0)


class TestEdgeCalculation:
    """Tests for EdgeCalculation dataclass."""

    def test_positive_edge(self):
        """Should calculate positive edge."""
        calc = EdgeCalculation(
            outcome="home_win",
            model_prob=0.50,
            market_price=0.40,
            edge=0.10,
            is_tradeable=True,
        )
        
        assert calc.edge == 0.10
        assert calc.is_tradeable

    def test_negative_edge(self):
        """Should calculate negative edge."""
        calc = EdgeCalculation(
            outcome="draw",
            model_prob=0.30,
            market_price=0.40,
            edge=-0.10,
            is_tradeable=False,
        )
        
        assert calc.edge == -0.10
        assert not calc.is_tradeable

    def test_to_dict(self):
        """Should convert to dictionary."""
        calc = EdgeCalculation(
            outcome="away_win",
            model_prob=0.25,
            market_price=0.20,
            edge=0.05,
        )
        
        d = calc.to_dict()
        
        assert d["outcome"] == "away_win"
        assert d["edge"] == 0.05


class TestKellySizing:
    """Tests for KellySizing dataclass."""

    def test_creation(self):
        """Should create sizing calculation."""
        sizing = KellySizing(
            edge=0.10,
            win_prob=0.50,
            odds=1.5,
            kelly_fraction=0.15,
            adjusted_fraction=0.0375,
            position_size=50,
        )
        
        assert sizing.edge == 0.10
        assert sizing.kelly_fraction == 0.15
        assert sizing.position_size == 50

    def test_capped_reason(self):
        """Should record capping reason."""
        sizing = KellySizing(
            edge=0.10,
            win_prob=0.50,
            odds=1.5,
            kelly_fraction=0.25,
            adjusted_fraction=0.0625,
            position_size=100,
            capped_reason="Capped at max position per market",
        )
        
        assert sizing.capped_reason is not None
        assert "max position" in sizing.capped_reason.lower()

    def test_to_dict(self):
        """Should convert to dictionary."""
        sizing = KellySizing(
            edge=0.10,
            win_prob=0.50,
            odds=1.5,
            kelly_fraction=0.15,
            adjusted_fraction=0.0375,
            position_size=50,
        )
        
        d = sizing.to_dict()
        
        assert d["edge"] == 0.10
        assert d["position_size"] == 50


class TestOrderParams:
    """Tests for OrderParams dataclass."""

    def test_buy_order(self):
        """Should create buy order params."""
        params = OrderParams(
            ticker="TEST-TICKER",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.40,
            quantity=50,
        )
        
        assert params.ticker == "TEST-TICKER"
        assert params.action == "buy"
        assert params.quantity == 50

    def test_sell_order(self):
        """Should create sell order params."""
        params = OrderParams(
            ticker="TEST-TICKER",
            side="yes",
            action="sell",
            order_type="limit",
            price=0.55,
            quantity=50,
        )
        
        assert params.action == "sell"

    def test_to_dict(self):
        """Should convert to dictionary."""
        params = OrderParams(
            ticker="TEST",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.40,
            quantity=50,
        )
        
        d = params.to_dict()
        
        assert d["ticker"] == "TEST"
        assert d["price"] == 0.40


class TestPositionState:
    """Tests for PositionState dataclass."""

    def test_creation(self):
        """Should create position state."""
        state = PositionState(
            ticker="TEST",
            quantity=50,
            average_entry_price=0.40,
            realized_pnl=5.0,
            unrealized_pnl=3.0,
        )
        
        assert state.quantity == 50
        assert state.average_entry_price == 0.40

    def test_to_dict(self):
        """Should convert to dictionary."""
        state = PositionState(
            ticker="TEST",
            quantity=50,
        )
        
        d = state.to_dict()
        
        assert d["ticker"] == "TEST"
        assert d["quantity"] == 50


class TestDecisionRecord:
    """Tests for DecisionRecord dataclass."""

    def test_creation_minimal(self):
        """Should create with minimal fields."""
        record = DecisionRecord(
            decision_id="decision-1",
            market_ticker="TEST-TICKER",
            outcome="home_win",
        )
        
        assert record.decision_id == "decision-1"
        assert record.action == DecisionAction.SKIP  # Default

    def test_creation_full(self):
        """Should create with all fields."""
        record = DecisionRecord(
            decision_id="decision-1",
            run_id=123,
            fixture_id=456,
            market_ticker="TEST-TICKER",
            outcome="home_win",
            action=DecisionAction.BUY,
            rationale="Positive edge detected",
            market_snapshot=MarketSnapshot(ticker="TEST"),
            model_prediction=ModelPrediction(
                model_name="test",
                model_version="1.0",
                prob_home_win=0.5,
                prob_draw=0.25,
                prob_away_win=0.25,
            ),
            edge_calculation=EdgeCalculation(
                outcome="home_win",
                model_prob=0.5,
                market_price=0.4,
                edge=0.1,
            ),
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.4,
                quantity=50,
            ),
        )
        
        assert record.action == DecisionAction.BUY
        assert record.order_params is not None
        assert record.order_params.quantity == 50

    def test_to_dict(self):
        """Should convert to dictionary."""
        record = DecisionRecord(
            decision_id="decision-1",
            run_id=123,
            fixture_id=456,
            market_ticker="TEST",
            outcome="home_win",
            action=DecisionAction.BUY,
            rationale="Test rationale",
        )
        
        d = record.to_dict()
        
        assert d["decision_id"] == "decision-1"
        assert d["run_id"] == 123
        assert d["action"] == "buy"
        assert d["rationale"] == "Test rationale"

    def test_to_dict_with_nested_objects(self):
        """Should serialize nested objects."""
        record = DecisionRecord(
            decision_id="decision-1",
            market_ticker="TEST",
            outcome="draw",
            action=DecisionAction.EXIT,
            exit_reason=ExitReason.TAKE_PROFIT,
            market_snapshot=MarketSnapshot(
                ticker="TEST",
                best_bid=0.50,
                best_ask=0.52,
            ),
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="sell",
                order_type="limit",
                price=0.50,
                quantity=50,
            ),
        )
        
        d = record.to_dict()
        
        assert d["action"] == "exit"
        assert d["exit_reason"] == "take_profit"
        assert d["market_snapshot"]["best_bid"] == 0.50
        assert d["order_params"]["action"] == "sell"

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "decision_id": "decision-1",
            "run_id": 123,
            "fixture_id": 456,
            "market_ticker": "TEST",
            "outcome": "home_win",
            "action": "buy",
            "rationale": "Test",
        }
        
        record = DecisionRecord.from_dict(data)
        
        assert record.decision_id == "decision-1"
        assert record.action == DecisionAction.BUY

    def test_filters_passed(self):
        """Should track filter results."""
        record = DecisionRecord(
            decision_id="decision-1",
            market_ticker="TEST",
            outcome="home_win",
            filters_passed={
                "market_status": True,
                "min_liquidity": True,
                "max_spread": False,
                "min_edge": True,
            },
        )
        
        d = record.to_dict()
        
        assert d["filters_passed"]["market_status"] is True
        assert d["filters_passed"]["max_spread"] is False


class TestAgentRunSummary:
    """Tests for AgentRunSummary dataclass."""

    def test_creation(self):
        """Should create run summary."""
        summary = AgentRunSummary(
            run_id=1,
            run_type="paper",
            status="running",
        )
        
        assert summary.run_id == 1
        assert summary.run_type == "paper"
        assert summary.status == "running"

    def test_counters(self):
        """Should track counters."""
        summary = AgentRunSummary()
        
        summary.fixtures_evaluated = 10
        summary.markets_evaluated = 30
        summary.decisions_made = 30
        summary.orders_placed = 5
        summary.orders_filled = 4
        
        assert summary.fixtures_evaluated == 10
        assert summary.orders_filled == 4

    def test_pnl_tracking(self):
        """Should track P&L."""
        summary = AgentRunSummary(
            total_realized_pnl=100.0,
            total_unrealized_pnl=50.0,
            total_exposure=500.0,
            position_count=5,
        )
        
        assert summary.total_realized_pnl == 100.0
        assert summary.total_unrealized_pnl == 50.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        summary = AgentRunSummary(
            run_id=1,
            run_type="paper",
            status="completed",
            fixtures_evaluated=10,
            total_realized_pnl=100.0,
        )
        
        d = summary.to_dict()
        
        assert d["run_id"] == 1
        assert d["run_type"] == "paper"
        assert d["fixtures_evaluated"] == 10
        assert d["total_realized_pnl"] == 100.0

    def test_config_tracking(self):
        """Should track config."""
        summary = AgentRunSummary(
            config_hash="abc123",
            config_summary={
                "min_edge": 0.05,
                "kelly_fraction": 0.25,
            },
        )
        
        assert summary.config_hash == "abc123"
        assert summary.config_summary["min_edge"] == 0.05
