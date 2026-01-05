"""Tests for edge-based trading strategy."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch
import pytest

from footbe_trader.kalshi.interfaces import MarketData, OrderbookData, OrderbookLevel
from footbe_trader.strategy.decision_record import (
    DecisionAction,
    ExitReason,
    ModelPrediction,
)
from footbe_trader.strategy.mapping import FixtureMarketMapping
from footbe_trader.strategy.trading_strategy import (
    EdgeStrategy,
    FixtureContext,
    OutcomeContext,
    StrategyConfig,
    create_agent_run_summary,
)


class TestStrategyConfig:
    """Tests for StrategyConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = StrategyConfig()
        
        assert config.min_edge_to_enter == 0.05
        assert config.exit_edge_buffer == -0.01
        assert config.min_model_confidence == 0.6
        assert config.kelly_fraction == 0.25
        assert config.max_position_per_market == 100
        assert config.take_profit == 0.15
        assert config.stop_loss == 0.20

    def test_config_hash(self):
        """Should generate consistent hash."""
        config1 = StrategyConfig()
        config2 = StrategyConfig()
        
        # Same config should have same hash
        assert config1.config_hash() == config2.config_hash()
        
        # Different config should have different hash
        config3 = StrategyConfig(min_edge_to_enter=0.10)
        assert config1.config_hash() != config3.config_hash()

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = StrategyConfig(
            min_edge_to_enter=0.06,
            kelly_fraction=0.30,
        )
        
        d = config.to_dict()
        
        assert d["min_edge_to_enter"] == 0.06
        assert d["kelly_fraction"] == 0.30
        assert "max_position_per_market" in d


class TestEdgeStrategy:
    """Tests for EdgeStrategy."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return StrategyConfig(
            min_edge_to_enter=0.05,
            exit_edge_buffer=-0.01,
            min_model_confidence=0.5,
            kelly_fraction=0.25,
            max_kelly_fraction=0.10,
            max_position_per_market=100,
            max_exposure_per_fixture=500.0,
            max_global_exposure=2000.0,
            initial_bankroll=10000.0,
            take_profit=0.15,
            stop_loss=0.20,
            min_ask_volume=5,
            max_spread=0.15,
            min_hours_to_close=2.0,
            max_hours_to_close=168.0,
        )

    @pytest.fixture
    def strategy(self, config):
        """Create strategy instance."""
        return EdgeStrategy(config)

    @pytest.fixture
    def fixture_context(self):
        """Create test fixture context."""
        mapping = FixtureMarketMapping(
            fixture_id=12345,
            ticker_home_win="TICKER-HOME",
            ticker_draw="TICKER-DRAW",
            ticker_away_win="TICKER-AWAY",
            structure_type="1X2",
            confidence_score=0.9,
        )
        return FixtureContext(
            fixture_id=12345,
            home_team="Manchester United",
            away_team="Liverpool",
            kickoff_time=datetime.now(UTC) + timedelta(hours=24),
            league="EPL",
            mapping=mapping,
        )

    @pytest.fixture
    def model_prediction(self):
        """Create test model prediction."""
        return ModelPrediction(
            model_name="test_model",
            model_version="1.0.0",
            prob_home_win=0.50,
            prob_draw=0.25,
            prob_away_win=0.25,
            confidence=0.8,
        )

    def _create_orderbook(
        self,
        best_bid: float = 0.35,
        best_ask: float = 0.40,
        bid_volume: int = 50,
        ask_volume: int = 50,
    ) -> OrderbookData:
        """Create test orderbook."""
        return OrderbookData(
            ticker="TEST",
            yes_bids=[OrderbookLevel(price=best_bid, quantity=bid_volume)],
            yes_asks=[OrderbookLevel(price=best_ask, quantity=ask_volume)],
            no_bids=[OrderbookLevel(price=1 - best_ask, quantity=ask_volume)],
            no_asks=[OrderbookLevel(price=1 - best_bid, quantity=bid_volume)],
            timestamp=datetime.now(UTC),
        )

    def _create_market_data(
        self,
        ticker: str = "TEST",
        status: str = "open",
        close_time: datetime | None = None,
    ) -> MarketData:
        """Create test market data."""
        if close_time is None:
            close_time = datetime.now(UTC) + timedelta(days=1)
        return MarketData(
            ticker=ticker,
            event_ticker="EVENT",
            title="Test Market",
            status=status,
            close_time=close_time,
            yes_bid=0.35,
            yes_ask=0.40,
        )

    def test_name_and_version(self, strategy):
        """Should have name and version."""
        assert strategy.name == "edge_strategy_v1"
        assert strategy.version == "1.0.0"

    def test_set_run_id(self, strategy):
        """Should set run ID."""
        strategy.set_run_id(123)
        assert strategy._current_run_id == 123

    def test_evaluate_fixture_with_positive_edge(
        self, strategy, fixture_context, model_prediction
    ):
        """Should generate buy decision when edge is positive."""
        # Model: 50% home win
        # Market: 40 cents ask (40% implied)
        # Edge: 50% - 40% = 10% (above 5% threshold)
        
        orderbook = self._create_orderbook(best_bid=0.35, best_ask=0.40)
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
            current_position=0,
        )
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=0.0,
            bankroll=10000.0,
        )
        
        assert len(decisions) == 1
        decision = decisions[0]
        
        assert decision.action == DecisionAction.BUY
        assert decision.order_params is not None
        assert decision.order_params.ticker == "TICKER-HOME"
        assert decision.order_params.price == 0.40
        assert decision.order_params.quantity > 0
        assert decision.edge_calculation.edge == pytest.approx(0.10, rel=0.01)

    def test_evaluate_fixture_with_negative_edge(
        self, strategy, fixture_context, model_prediction
    ):
        """Should skip when edge is negative."""
        # Model: 50% home win
        # Market: 55 cents ask (55% implied)
        # Edge: 50% - 55% = -5% (below threshold)
        
        orderbook = self._create_orderbook(best_bid=0.50, best_ask=0.55)
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=0.0,
        )
        
        assert len(decisions) == 1
        decision = decisions[0]
        
        assert decision.action == DecisionAction.SKIP
        assert decision.order_params is None
        assert "below threshold" in decision.rejection_reason.lower()

    def test_evaluate_fixture_insufficient_liquidity(
        self, strategy, fixture_context, model_prediction
    ):
        """Should skip when liquidity is too low."""
        # Good edge but low liquidity
        orderbook = self._create_orderbook(
            best_bid=0.35, best_ask=0.40, ask_volume=2  # Below min 5
        )
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=0.0,
        )
        
        decision = decisions[0]
        assert decision.action == DecisionAction.SKIP
        assert decision.filters_passed.get("min_ask_volume") is False

    def test_evaluate_fixture_market_closed(
        self, strategy, fixture_context, model_prediction
    ):
        """Should skip when market is closed."""
        orderbook = self._create_orderbook()
        market = self._create_market_data(status="closed")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=0.0,
        )
        
        decision = decisions[0]
        assert decision.action == DecisionAction.SKIP
        assert decision.filters_passed.get("market_status") is False

    def test_evaluate_fixture_wide_spread(
        self, strategy, fixture_context, model_prediction
    ):
        """Should skip when spread is too wide."""
        # 20 cent spread (above 15 cent max)
        orderbook = self._create_orderbook(best_bid=0.30, best_ask=0.50)
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=0.0,
        )
        
        decision = decisions[0]
        assert decision.action == DecisionAction.SKIP
        assert decision.filters_passed.get("max_spread") is False

    def test_kelly_sizing_basic(self, strategy, fixture_context, model_prediction):
        """Should apply fractional Kelly sizing."""
        # Edge = 10%, ask = 0.40
        # Full Kelly should give a reasonable size
        orderbook = self._create_orderbook(best_bid=0.35, best_ask=0.40)
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=0.0,
            bankroll=10000.0,
        )
        
        decision = decisions[0]
        assert decision.kelly_sizing is not None
        assert decision.kelly_sizing.kelly_fraction > 0
        assert decision.kelly_sizing.adjusted_fraction > 0
        assert decision.kelly_sizing.position_size > 0
        assert decision.kelly_sizing.position_size <= 100  # Per-market cap

    def test_position_cap_per_market(self, strategy, fixture_context):
        """Should cap position at max per market."""
        # Very high edge to trigger large Kelly
        model_pred = ModelPrediction(
            model_name="test",
            model_version="1.0.0",
            prob_home_win=0.90,  # Very confident
            prob_draw=0.05,
            prob_away_win=0.05,
            confidence=0.9,
        )
        
        # Low ask price = high potential profit = large Kelly
        orderbook = self._create_orderbook(best_bid=0.15, best_ask=0.20)
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_pred,
            global_exposure=0.0,
            bankroll=100000.0,  # Large bankroll
        )
        
        decision = decisions[0]
        assert decision.kelly_sizing.position_size <= 100

    def test_global_exposure_cap(self, strategy, fixture_context, model_prediction):
        """Should respect global exposure limit."""
        orderbook = self._create_orderbook(best_bid=0.35, best_ask=0.40)
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        # Already at 90% of global exposure limit
        global_exposure = 1800.0  # 90% of 2000
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=global_exposure,
            bankroll=10000.0,
        )
        
        decision = decisions[0]
        
        if decision.action == DecisionAction.BUY:
            # Position should be capped
            max_new_exposure = 200.0  # Remaining
            max_position = int(max_new_exposure / 0.40)
            assert decision.order_params.quantity <= max_position

    def test_evaluate_exit_take_profit(self, strategy):
        """Should trigger take profit exit."""
        # Entry at 0.40, current mid at 0.56 (0.16 profit, above 0.15 threshold)
        orderbook = self._create_orderbook(best_bid=0.54, best_ask=0.58)
        market = self._create_market_data(status="open")
        
        model_pred = ModelPrediction(
            model_name="test",
            model_version="1.0.0",
            prob_home_win=0.50,
            prob_draw=0.25,
            prob_away_win=0.25,
            confidence=0.8,
        )
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
            current_position=50,
            average_entry_price=0.40,
        )
        
        decision = strategy.evaluate_exit(
            outcome=outcome,
            model_prediction=model_pred,
            entry_price=0.40,
            fixture_id=12345,
        )
        
        assert decision.action == DecisionAction.EXIT
        assert decision.exit_reason == ExitReason.TAKE_PROFIT
        assert decision.order_params is not None
        assert decision.order_params.action == "sell"
        assert decision.order_params.quantity == 50

    def test_evaluate_exit_stop_loss(self, strategy):
        """Should trigger stop loss exit."""
        # Entry at 0.50, current mid at 0.29 (0.21 loss, above 0.20 threshold)
        orderbook = self._create_orderbook(best_bid=0.27, best_ask=0.31)
        market = self._create_market_data(status="open")
        
        model_pred = ModelPrediction(
            model_name="test",
            model_version="1.0.0",
            prob_home_win=0.30,
            prob_draw=0.35,
            prob_away_win=0.35,
            confidence=0.8,
        )
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
            current_position=50,
            average_entry_price=0.50,
        )
        
        decision = strategy.evaluate_exit(
            outcome=outcome,
            model_prediction=model_pred,
            entry_price=0.50,
            fixture_id=12345,
        )
        
        assert decision.action == DecisionAction.EXIT
        assert decision.exit_reason == ExitReason.STOP_LOSS

    def test_evaluate_exit_edge_flip(self, strategy):
        """Should exit when edge flips negative beyond buffer."""
        # Model now says 35% but mid is 40% = -5% edge (below -1% buffer)
        orderbook = self._create_orderbook(best_bid=0.38, best_ask=0.42)
        market = self._create_market_data(status="open")
        
        model_pred = ModelPrediction(
            model_name="test",
            model_version="1.0.0",
            prob_home_win=0.35,  # Model dropped
            prob_draw=0.35,
            prob_away_win=0.30,
            confidence=0.8,
        )
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
            current_position=50,
            average_entry_price=0.38,  # Entry price
        )
        
        decision = strategy.evaluate_exit(
            outcome=outcome,
            model_prediction=model_pred,
            entry_price=0.38,
            fixture_id=12345,
        )
        
        assert decision.action == DecisionAction.EXIT
        assert decision.exit_reason == ExitReason.EDGE_FLIP

    def test_evaluate_exit_hold(self, strategy):
        """Should hold when no exit conditions met."""
        # Entry at 0.40, current mid at 0.45 (not at TP/SL)
        # Model still positive edge
        orderbook = self._create_orderbook(best_bid=0.43, best_ask=0.47)
        market = self._create_market_data(status="open")
        
        model_pred = ModelPrediction(
            model_name="test",
            model_version="1.0.0",
            prob_home_win=0.50,  # Still positive edge vs 0.45 mid
            prob_draw=0.25,
            prob_away_win=0.25,
            confidence=0.8,
        )
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
            current_position=50,
            average_entry_price=0.40,
        )
        
        decision = strategy.evaluate_exit(
            outcome=outcome,
            model_prediction=model_pred,
            entry_price=0.40,
            fixture_id=12345,
        )
        
        assert decision.action == DecisionAction.HOLD
        assert decision.order_params is None

    def test_evaluate_exit_market_close_approaching(self, strategy):
        """Should exit when market close is approaching."""
        orderbook = self._create_orderbook(best_bid=0.43, best_ask=0.47)
        # Close in 30 minutes (below 1 hour threshold)
        market = self._create_market_data(
            status="open",
            close_time=datetime.now(UTC) + timedelta(minutes=30),
        )
        
        model_pred = ModelPrediction(
            model_name="test",
            model_version="1.0.0",
            prob_home_win=0.50,
            prob_draw=0.25,
            prob_away_win=0.25,
            confidence=0.8,
        )
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
            current_position=50,
            average_entry_price=0.40,
        )
        
        decision = strategy.evaluate_exit(
            outcome=outcome,
            model_prediction=model_pred,
            entry_price=0.40,
            fixture_id=12345,
        )
        
        assert decision.action == DecisionAction.EXIT
        assert decision.exit_reason == ExitReason.MARKET_CLOSE

    def test_multiple_outcomes(self, strategy, fixture_context):
        """Should evaluate all outcomes for a fixture."""
        model_pred = ModelPrediction(
            model_name="test",
            model_version="1.0.0",
            prob_home_win=0.50,
            prob_draw=0.25,
            prob_away_win=0.25,
            confidence=0.8,
        )
        
        market = self._create_market_data(status="open")
        
        outcomes = [
            OutcomeContext(
                outcome="home_win",
                ticker="TICKER-HOME",
                market_data=market,
                orderbook=self._create_orderbook(best_ask=0.40),  # Edge = 10%
            ),
            OutcomeContext(
                outcome="draw",
                ticker="TICKER-DRAW",
                market_data=market,
                orderbook=self._create_orderbook(best_ask=0.30),  # Edge = -5%
            ),
            OutcomeContext(
                outcome="away_win",
                ticker="TICKER-AWAY",
                market_data=market,
                orderbook=self._create_orderbook(best_ask=0.15),  # Edge = 10% (0.25 - 0.15)
            ),
        ]
        
        decisions = strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=outcomes,
            model_prediction=model_pred,
            global_exposure=0.0,
        )
        
        assert len(decisions) == 3
        
        # Home win should be buy (10% edge)
        assert decisions[0].action == DecisionAction.BUY
        
        # Draw should be skip (-5% edge)
        assert decisions[1].action == DecisionAction.SKIP
        
        # Away win should be buy (10% edge: 0.25 - 0.15)
        assert decisions[2].action == DecisionAction.BUY

    def test_decision_records_accumulated(self, strategy, fixture_context, model_prediction):
        """Should accumulate decision records."""
        orderbook = self._create_orderbook()
        market = self._create_market_data(status="open")
        
        outcome = OutcomeContext(
            outcome="home_win",
            ticker="TICKER-HOME",
            market_data=market,
            orderbook=orderbook,
        )
        
        # Clear any existing records
        strategy.clear_decision_records()
        
        strategy.evaluate_fixture(
            fixture=fixture_context,
            outcomes=[outcome],
            model_prediction=model_prediction,
            global_exposure=0.0,
        )
        
        records = strategy.get_decision_records()
        assert len(records) == 1
        assert records[0].fixture_id == 12345
        
        # Clear and verify
        strategy.clear_decision_records()
        assert len(strategy.get_decision_records()) == 0


class TestCreateAgentRunSummary:
    """Tests for create_agent_run_summary function."""

    def test_creates_summary(self):
        """Should create run summary with config."""
        summary = create_agent_run_summary(run_type="paper")
        
        assert summary.run_type == "paper"
        assert summary.status == "running"
        assert summary.config_hash != ""
        assert summary.config_summary != {}
