"""Tests for paper trading simulator."""

from datetime import UTC, datetime, timedelta
import pytest

from footbe_trader.kalshi.interfaces import OrderbookData, OrderbookLevel
from footbe_trader.strategy.decision_record import (
    DecisionAction,
    DecisionRecord,
    OrderParams,
)
from footbe_trader.strategy.paper_trading import (
    PaperFill,
    PaperOrder,
    PaperPosition,
    PaperTradingSimulator,
    PnlSnapshot,
)
from footbe_trader.strategy.trading_strategy import StrategyConfig


class TestPaperPosition:
    """Tests for PaperPosition dataclass."""

    def test_is_open(self):
        """Should track if position is open."""
        pos = PaperPosition(ticker="TEST", quantity=0)
        assert not pos.is_open
        
        pos.quantity = 10
        assert pos.is_open

    def test_total_pnl(self):
        """Should calculate total P&L."""
        pos = PaperPosition(
            ticker="TEST",
            realized_pnl=5.0,
            unrealized_pnl=3.0,
        )
        assert pos.total_pnl == 8.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        pos = PaperPosition(
            ticker="TEST",
            outcome="home_win",
            fixture_id=123,
            quantity=50,
            average_entry_price=0.40,
        )
        
        d = pos.to_dict()
        
        assert d["ticker"] == "TEST"
        assert d["outcome"] == "home_win"
        assert d["fixture_id"] == 123
        assert d["quantity"] == 50
        assert d["average_entry_price"] == 0.40

    def test_to_state(self):
        """Should convert to PositionState."""
        pos = PaperPosition(
            ticker="TEST",
            quantity=50,
            average_entry_price=0.40,
            realized_pnl=5.0,
            unrealized_pnl=3.0,
        )
        
        state = pos.to_state()
        
        assert state.ticker == "TEST"
        assert state.quantity == 50
        assert state.average_entry_price == 0.40


class TestPaperOrder:
    """Tests for PaperOrder dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        order = PaperOrder(
            order_id="order-1",
            ticker="TEST",
            side="yes",
            action="buy",
            price=0.40,
            quantity=50,
            status="filled",
        )
        
        d = order.to_dict()
        
        assert d["order_id"] == "order-1"
        assert d["ticker"] == "TEST"
        assert d["action"] == "buy"
        assert d["price"] == 0.40


class TestPaperTradingSimulator:
    """Tests for PaperTradingSimulator."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return StrategyConfig(
            slippage_cents=0.01,
            fill_probability=1.0,  # Always fill for deterministic tests
            partial_fill_range=(1.0, 1.0),  # Full fills
            initial_bankroll=10000.0,
        )

    @pytest.fixture
    def simulator(self, config):
        """Create simulator instance."""
        sim = PaperTradingSimulator(config=config)
        sim.set_seed(42)  # Reproducible
        return sim

    def _create_orderbook(
        self,
        best_bid: float = 0.40,
        best_ask: float = 0.42,
        volume: int = 100,
    ) -> OrderbookData:
        """Create test orderbook."""
        return OrderbookData(
            ticker="TEST",
            yes_bids=[OrderbookLevel(price=best_bid, quantity=volume)],
            yes_asks=[OrderbookLevel(price=best_ask, quantity=volume)],
            no_bids=[OrderbookLevel(price=1 - best_ask, quantity=volume)],
            no_asks=[OrderbookLevel(price=1 - best_bid, quantity=volume)],
            timestamp=datetime.now(UTC),
        )

    def test_initial_state(self, simulator):
        """Should have correct initial state."""
        assert simulator.bankroll == 10000.0
        assert simulator.total_realized_pnl == 0.0
        assert simulator.total_unrealized_pnl == 0.0
        assert simulator.total_exposure == 0.0
        assert len(simulator.positions) == 0
        assert len(simulator.orders) == 0
        assert len(simulator.fills) == 0

    def test_submit_buy_order(self, simulator):
        """Should execute buy order and create position."""
        orderbook = self._create_orderbook(best_ask=0.40)
        
        order_params = OrderParams(
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.40,
            quantity=50,
        )
        
        order = simulator.submit_order(
            order_params=order_params,
            orderbook=orderbook,
            outcome="home_win",
            fixture_id=123,
        )
        
        assert order.status in ("filled", "partial")
        assert order.filled_quantity > 0
        assert order.fill_price == pytest.approx(0.41, abs=0.02)  # Ask + slippage
        
        # Check position created
        position = simulator.get_position("TEST-MARKET")
        assert position is not None
        assert position.quantity > 0
        assert position.outcome == "home_win"
        assert position.fixture_id == 123

    def test_submit_sell_order(self, simulator):
        """Should execute sell order and update position."""
        # First buy at lower price
        buy_orderbook = self._create_orderbook(best_bid=0.38, best_ask=0.40)
        
        buy_params = OrderParams(
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.40,
            quantity=50,
        )
        
        simulator.submit_order(
            order_params=buy_params,
            orderbook=buy_orderbook,
            outcome="home_win",
        )
        
        # Now sell at higher price (price increased significantly)
        sell_orderbook = self._create_orderbook(best_bid=0.60, best_ask=0.62)
        
        sell_params = OrderParams(
            ticker="TEST-MARKET",
            side="yes",
            action="sell",
            order_type="limit",
            price=0.60,
            quantity=50,
        )
        
        order = simulator.submit_order(
            order_params=sell_params,
            orderbook=sell_orderbook,
        )
        
        assert order.status in ("filled", "partial")
        
        # Position should be closed
        position = simulator.get_position("TEST-MARKET")
        assert position.quantity == 0
        # Sold at ~0.59 (bid - slippage), bought at ~0.41 (ask + slippage)
        # Profit = (0.59 - 0.41) * 50 - fees ≈ $9 - fees
        assert position.realized_pnl > 0  # Made profit

    def test_partial_fills(self, config):
        """Should handle partial fills."""
        config.partial_fill_range = (0.5, 0.5)  # Always 50% fill
        config.fill_probability = 1.0
        sim = PaperTradingSimulator(config=config)
        sim.set_seed(42)
        
        orderbook = self._create_orderbook()
        
        order_params = OrderParams(
            ticker="TEST",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.42,
            quantity=100,
        )
        
        order = sim.submit_order(order_params=order_params, orderbook=orderbook)
        
        assert order.status == "partial"
        assert order.filled_quantity == 50  # 50% of 100

    def test_order_rejection(self, config):
        """Should handle order rejections."""
        config.fill_probability = 0.0  # Never fill
        sim = PaperTradingSimulator(config=config)
        sim.set_seed(42)
        
        order_params = OrderParams(
            ticker="TEST",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.42,
            quantity=50,
        )
        
        order = sim.submit_order(order_params=order_params)
        
        assert order.status == "rejected"
        assert order.filled_quantity == 0

    def test_bankroll_tracking(self, simulator):
        """Should track bankroll correctly."""
        initial_bankroll = simulator.bankroll
        
        orderbook = self._create_orderbook(best_ask=0.40)
        order_params = OrderParams(
            ticker="TEST",
            side="yes",
            action="buy",
            order_type="limit",
            price=0.40,
            quantity=50,
        )
        
        simulator.submit_order(order_params=order_params, orderbook=orderbook)
        
        # Bankroll should decrease by (fill_price * quantity + fees)
        position = simulator.get_position("TEST")
        assert simulator.bankroll < initial_bankroll
        assert simulator.bankroll == pytest.approx(
            initial_bankroll - position.total_cost, abs=0.1
        )

    def test_pnl_calculation(self, simulator):
        """Should calculate P&L correctly."""
        # Buy at 0.40
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        # Sell at 0.55 (profit)
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="sell",
                order_type="limit",
                price=0.55,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_bid=0.55),
        )
        
        # Should have realized profit
        assert simulator.total_realized_pnl > 0

    def test_mark_to_market(self, simulator):
        """Should mark positions to market."""
        orderbook = self._create_orderbook(best_ask=0.40)
        
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=orderbook,
        )
        
        # Mark to higher price
        simulator.mark_to_market("TEST", 0.50)
        
        position = simulator.get_position("TEST")
        assert position.mark_price == 0.50
        assert position.unrealized_pnl > 0

    def test_mark_all_positions(self, simulator):
        """Should mark all positions to market."""
        # Create multiple positions
        for ticker in ["TEST-1", "TEST-2", "TEST-3"]:
            simulator.submit_order(
                order_params=OrderParams(
                    ticker=ticker,
                    side="yes",
                    action="buy",
                    order_type="limit",
                    price=0.40,
                    quantity=50,
                ),
                orderbook=self._create_orderbook(best_ask=0.40),
            )
        
        prices = {
            "TEST-1": 0.50,
            "TEST-2": 0.35,
            "TEST-3": 0.45,
        }
        
        simulator.mark_all_positions(prices)
        
        assert simulator.get_position("TEST-1").mark_price == 0.50
        assert simulator.get_position("TEST-2").mark_price == 0.35
        assert simulator.get_position("TEST-3").mark_price == 0.45

    def test_pnl_snapshot(self, simulator):
        """Should take P&L snapshots."""
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        simulator.mark_to_market("TEST", 0.50)
        
        snapshot = simulator.take_pnl_snapshot()
        
        assert snapshot.snapshot_id != ""
        assert snapshot.position_count == 1
        assert snapshot.total_exposure > 0
        assert len(snapshot.positions_snapshot) == 1

    def test_fixture_exposure(self, simulator):
        """Should track exposure per fixture."""
        # Two positions for same fixture
        simulator.submit_order(
            order_params=OrderParams(
                ticker="FIXTURE-1-HOME",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
            fixture_id=1,
        )
        
        simulator.submit_order(
            order_params=OrderParams(
                ticker="FIXTURE-1-DRAW",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.25,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.25),
            fixture_id=1,
        )
        
        # Different fixture
        simulator.submit_order(
            order_params=OrderParams(
                ticker="FIXTURE-2-HOME",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.50,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.50),
            fixture_id=2,
        )
        
        exposure_1 = simulator.get_fixture_exposure(1)
        exposure_2 = simulator.get_fixture_exposure(2)
        
        assert exposure_1 > 0
        assert exposure_2 > 0
        assert exposure_1 > exposure_2  # Fixture 1 has two positions

    def test_settle_position_win(self, simulator):
        """Should settle position on win."""
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        bankroll_before = simulator.bankroll
        
        # Settle at $1 (yes wins)
        pnl = simulator.settle_position("TEST", settlement_value=1.0)
        
        # Should make profit: 50 * $1 - cost
        assert pnl > 0
        assert simulator.bankroll > bankroll_before

    def test_settle_position_loss(self, simulator):
        """Should settle position on loss."""
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        # Settle at $0 (no wins)
        pnl = simulator.settle_position("TEST", settlement_value=0.0)
        
        # Should lose entire cost
        assert pnl < 0

    def test_execute_decision(self, simulator):
        """Should execute decision record."""
        decision = DecisionRecord(
            decision_id="decision-1",
            fixture_id=123,
            market_ticker="TEST-MARKET",
            outcome="home_win",
            action=DecisionAction.BUY,
            order_params=OrderParams(
                ticker="TEST-MARKET",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
        )
        
        orderbook = self._create_orderbook(best_ask=0.40)
        
        order = simulator.execute_decision(decision, orderbook)
        
        assert order is not None
        assert order.decision_id == "decision-1"
        assert order.ticker == "TEST-MARKET"

    def test_execute_decision_skip(self, simulator):
        """Should not execute skip decisions."""
        decision = DecisionRecord(
            decision_id="decision-1",
            action=DecisionAction.SKIP,
        )
        
        order = simulator.execute_decision(decision, None)
        
        assert order is None

    def test_open_positions_filter(self, simulator):
        """Should filter for open positions only."""
        # Create and close a position
        simulator.submit_order(
            order_params=OrderParams(
                ticker="CLOSED",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        simulator.submit_order(
            order_params=OrderParams(
                ticker="CLOSED",
                side="yes",
                action="sell",
                order_type="limit",
                price=0.50,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_bid=0.50),
        )
        
        # Create open position
        simulator.submit_order(
            order_params=OrderParams(
                ticker="OPEN",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        all_positions = simulator.positions
        open_positions = simulator.open_positions
        
        assert len(all_positions) == 2
        assert len(open_positions) == 1
        assert "OPEN" in open_positions

    def test_reset(self, simulator):
        """Should reset all state."""
        # Create some state
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        simulator.reset()
        
        assert simulator.bankroll == 10000.0
        assert len(simulator.positions) == 0
        assert len(simulator.orders) == 0
        assert len(simulator.fills) == 0
        assert simulator.total_realized_pnl == 0.0

    def test_to_dict(self, simulator):
        """Should export state to dictionary."""
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        d = simulator.to_dict()
        
        assert "bankroll" in d
        assert "positions" in d
        assert "orders" in d
        assert "fills" in d
        assert d["initial_bankroll"] == 10000.0

    def test_average_entry_price_multiple_buys(self, simulator):
        """Should calculate average entry price for multiple buys."""
        # Buy 50 at 0.40
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.40,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.40),
        )
        
        # Buy another 50 at 0.50
        simulator.submit_order(
            order_params=OrderParams(
                ticker="TEST",
                side="yes",
                action="buy",
                order_type="limit",
                price=0.50,
                quantity=50,
            ),
            orderbook=self._create_orderbook(best_ask=0.50),
        )
        
        position = simulator.get_position("TEST")
        
        # Average should be between 0.40 and 0.50
        assert 0.40 < position.average_entry_price < 0.55
        assert position.quantity == 100


class TestPnlSnapshot:
    """Tests for PnlSnapshot dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        snapshot = PnlSnapshot(
            snapshot_id="snap-1",
            run_id=123,
            total_realized_pnl=10.0,
            total_unrealized_pnl=5.0,
            total_pnl=15.0,
            bankroll=9985.0,
        )
        
        d = snapshot.to_dict()
        
        assert d["snapshot_id"] == "snap-1"
        assert d["run_id"] == 123
        assert d["total_pnl"] == 15.0
