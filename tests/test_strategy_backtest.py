"""Tests for strategy backtest module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from footbe_trader.kalshi.interfaces import OrderbookData, OrderbookLevel
from footbe_trader.storage.models import HistoricalSnapshot
from footbe_trader.strategy.strategy_backtest import (
    BacktestConfig,
    BacktestPosition,
    BacktestState,
    StrategyBacktester,
)
from footbe_trader.strategy.trading_strategy import StrategyConfig


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = BacktestConfig()
        assert config.initial_bankroll == 10000.0
        assert config.slippage_cents == 0.01
        assert config.fill_probability == 1.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_bankroll=5000.0,
            slippage_cents=0.02,
            fill_probability=0.9,
        )
        assert config.initial_bankroll == 5000.0
        assert config.slippage_cents == 0.02
        assert config.fill_probability == 0.9

    def test_to_dict(self):
        """Test serialization."""
        config = BacktestConfig(initial_bankroll=5000.0)
        data = config.to_dict()
        assert data["initial_bankroll"] == 5000.0
        assert "slippage_cents" in data


class TestBacktestPosition:
    """Tests for BacktestPosition."""

    def test_default_values(self):
        """Test default position state."""
        pos = BacktestPosition(ticker="TEST", fixture_id=1, outcome="home_win")
        assert pos.quantity == 0
        assert pos.average_entry_price == 0.0
        assert pos.total_cost == 0.0
        assert pos.unrealized_pnl == 0.0

    def test_unrealized_pnl_calculation(self):
        """Test unrealized P&L calculation."""
        pos = BacktestPosition(
            ticker="TEST",
            fixture_id=1,
            outcome="home_win",
            quantity=100,
            total_cost=50.0,  # $0.50 per contract
            mark_price=0.60,
        )
        # Value = 100 * 0.60 = 60
        # PnL = 60 - 50 = 10
        assert pos.unrealized_pnl == 10.0

    def test_unrealized_pnl_loss(self):
        """Test unrealized P&L with loss."""
        pos = BacktestPosition(
            ticker="TEST",
            fixture_id=1,
            outcome="home_win",
            quantity=100,
            total_cost=50.0,
            mark_price=0.40,
        )
        # Value = 100 * 0.40 = 40
        # PnL = 40 - 50 = -10
        assert pos.unrealized_pnl == -10.0

    def test_unrealized_pnl_zero_quantity(self):
        """Test unrealized P&L with no position."""
        pos = BacktestPosition(
            ticker="TEST",
            fixture_id=1,
            outcome="home_win",
            quantity=0,
            total_cost=0,
            mark_price=0.60,
        )
        assert pos.unrealized_pnl == 0.0

    def test_to_dict(self):
        """Test position serialization."""
        pos = BacktestPosition(
            ticker="TEST",
            fixture_id=1,
            outcome="home_win",
            quantity=50,
        )
        data = pos.to_dict()
        assert data["ticker"] == "TEST"
        assert data["quantity"] == 50


class TestBacktestState:
    """Tests for BacktestState."""

    def test_default_values(self):
        """Test default state."""
        state = BacktestState()
        assert state.bankroll == 10000.0
        assert state.realized_pnl == 0.0
        assert state.positions == {}
        assert state.trades == []

    def test_unrealized_pnl_total(self):
        """Test total unrealized P&L calculation."""
        state = BacktestState()
        state.positions["A"] = BacktestPosition(
            ticker="A",
            fixture_id=1,
            outcome="home_win",
            quantity=100,
            total_cost=50,
            mark_price=0.60,
        )
        state.positions["B"] = BacktestPosition(
            ticker="B",
            fixture_id=1,
            outcome="draw",
            quantity=50,
            total_cost=15,
            mark_price=0.40,
        )
        # A: 60 - 50 = 10
        # B: 20 - 15 = 5
        assert state.unrealized_pnl == 15.0

    def test_total_exposure(self):
        """Test total exposure calculation."""
        state = BacktestState()
        state.positions["A"] = BacktestPosition(
            ticker="A",
            fixture_id=1,
            outcome="home_win",
            quantity=100,
            total_cost=50,
        )
        state.positions["B"] = BacktestPosition(
            ticker="B",
            fixture_id=1,
            outcome="draw",
            quantity=50,
            total_cost=25,
        )
        assert state.total_exposure == 75.0

    def test_current_equity(self):
        """Test current equity calculation."""
        state = BacktestState(bankroll=10000.0)
        state.realized_pnl = 100.0
        state.positions["A"] = BacktestPosition(
            ticker="A",
            fixture_id=1,
            outcome="home_win",
            quantity=100,
            total_cost=50,
            mark_price=0.60,
        )
        # Equity = bankroll + realized + unrealized
        # = 10000 + 100 + 10 = 10110
        assert state.current_equity == 10110.0

    def test_drawdown(self):
        """Test drawdown calculation."""
        state = BacktestState(bankroll=9000.0, peak_equity=10000.0)
        # Drawdown = (10000 - 9000) / 10000 = 0.10
        assert state.drawdown == 0.10


class TestStrategyBacktester:
    """Tests for StrategyBacktester."""

    def test_init_default_config(self):
        """Test initialization with defaults."""
        backtester = StrategyBacktester()
        assert backtester.backtest_config.initial_bankroll == 10000.0
        assert backtester.state.bankroll == 10000.0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        strategy_config = StrategyConfig(min_edge_to_enter=0.10)
        backtest_config = BacktestConfig(initial_bankroll=5000.0)

        backtester = StrategyBacktester(
            strategy_config=strategy_config,
            backtest_config=backtest_config,
        )

        assert backtester.strategy_config.min_edge_to_enter == 0.10
        assert backtester.backtest_config.initial_bankroll == 5000.0
        assert backtester.state.bankroll == 5000.0

    def test_run_empty_snapshots(self):
        """Test running with no snapshots."""
        backtester = StrategyBacktester()
        result = backtester.run(snapshots=[])

        assert result.status == "completed"
        assert result.total_trades == 0
        assert result.total_return == 0.0

    def test_run_with_snapshots(self):
        """Test running with snapshots."""
        # Create test snapshots
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        snapshots = []

        for i in range(5):
            timestamp = now + timedelta(minutes=i * 5)
            snapshots.append(
                HistoricalSnapshot(
                    fixture_id=12345,
                    ticker="TEST-HOME",
                    outcome="home_win",
                    timestamp=timestamp,
                    best_bid=0.45,
                    best_ask=0.50,
                    mid=0.475,
                    spread=0.05,
                    bid_volume=100,
                    ask_volume=100,
                    model_prob=0.65,  # High prob triggers buy
                )
            )

        # Set kickoff in the future
        kickoff = now + timedelta(hours=24)
        fixture_kickoffs = {12345: kickoff}
        fixture_outcomes = {12345: "home_win"}

        # Use strategy that will trade
        strategy_config = StrategyConfig(
            min_edge_to_enter=0.05,  # Low threshold
            min_model_confidence=0.5,
            max_spread=0.20,
        )
        backtester = StrategyBacktester(strategy_config=strategy_config)

        result = backtester.run(
            snapshots=snapshots,
            fixture_outcomes=fixture_outcomes,
            fixture_kickoffs=fixture_kickoffs,
        )

        assert result.status == "completed"
        assert result.backtest_id == backtester.backtest_id

    def test_snapshot_to_orderbook(self):
        """Test converting snapshot to orderbook."""
        backtester = StrategyBacktester()
        snapshot = HistoricalSnapshot(
            ticker="TEST",
            best_bid=0.45,
            best_ask=0.50,
            bid_volume=100,
            ask_volume=50,
            timestamp=datetime.now(UTC),
        )

        orderbook = backtester._snapshot_to_orderbook(snapshot)

        assert orderbook.ticker == "TEST"
        assert orderbook.best_yes_bid == 0.45
        assert orderbook.best_yes_ask == 0.50

    def test_snapshot_to_orderbook_with_raw_json(self):
        """Test converting snapshot with raw orderbook JSON."""
        backtester = StrategyBacktester()
        snapshot = HistoricalSnapshot(
            ticker="TEST",
            best_bid=0.45,
            best_ask=0.50,
            timestamp=datetime.now(UTC),
            raw_orderbook_json={
                "yes_bids": [
                    {"price": 0.45, "quantity": 100},
                    {"price": 0.44, "quantity": 200},
                ],
                "yes_asks": [
                    {"price": 0.50, "quantity": 50},
                ],
            },
        )

        orderbook = backtester._snapshot_to_orderbook(snapshot)

        assert len(orderbook.yes_bids) == 2
        assert orderbook.yes_bids[0].price == 0.45
        assert orderbook.yes_bids[1].quantity == 200

    def test_get_fixture_exposure(self):
        """Test fixture exposure calculation."""
        backtester = StrategyBacktester()
        backtester.state.positions["A"] = BacktestPosition(
            ticker="A",
            fixture_id=100,
            outcome="home_win",
            quantity=50,
            total_cost=25,
        )
        backtester.state.positions["B"] = BacktestPosition(
            ticker="B",
            fixture_id=100,
            outcome="draw",
            quantity=30,
            total_cost=10,
        )
        backtester.state.positions["C"] = BacktestPosition(
            ticker="C",
            fixture_id=200,
            outcome="home_win",
            quantity=100,
            total_cost=50,
        )

        exposure_100 = backtester._get_fixture_exposure(100)
        exposure_200 = backtester._get_fixture_exposure(200)

        assert exposure_100 == 35.0
        assert exposure_200 == 50.0

    def test_settlement_winning_position(self):
        """Test position settlement for winning outcome."""
        backtester = StrategyBacktester()
        backtester.state.positions["TEST"] = BacktestPosition(
            ticker="TEST",
            fixture_id=100,
            outcome="home_win",
            quantity=100,
            total_cost=50,  # $0.50 per contract
            average_entry_price=0.50,
        )

        now = datetime.now(UTC)
        backtester._settle_fixture(100, "home_win", now)

        # Winning position settles at 1.0
        # Value = 100 * 1.0 = 100
        # PnL = 100 - 50 = 50
        assert backtester.state.realized_pnl == 50.0
        assert backtester.state.positions["TEST"].quantity == 0

    def test_settlement_losing_position(self):
        """Test position settlement for losing outcome."""
        backtester = StrategyBacktester()
        backtester.state.positions["TEST"] = BacktestPosition(
            ticker="TEST",
            fixture_id=100,
            outcome="home_win",
            quantity=100,
            total_cost=50,
            average_entry_price=0.50,
        )

        now = datetime.now(UTC)
        backtester._settle_fixture(100, "away_win", now)

        # Losing position settles at 0.0
        # Value = 100 * 0.0 = 0
        # PnL = 0 - 50 = -50
        assert backtester.state.realized_pnl == -50.0
        assert backtester.state.positions["TEST"].quantity == 0

    def test_mark_positions(self):
        """Test position marking."""
        backtester = StrategyBacktester()
        backtester.state.current_timestamp = datetime.now(UTC)
        backtester.state.positions["A"] = BacktestPosition(
            ticker="A",
            fixture_id=100,
            outcome="home_win",
            quantity=100,
            total_cost=50,
            mark_price=0.50,
        )

        snapshots = [
            HistoricalSnapshot(
                ticker="A",
                mid=0.60,
                timestamp=datetime.now(UTC),
            )
        ]

        backtester._mark_positions(snapshots)

        assert backtester.state.positions["A"].mark_price == 0.60

    def test_create_result(self):
        """Test result creation."""
        backtester = StrategyBacktester(
            backtest_config=BacktestConfig(initial_bankroll=10000.0)
        )
        backtester.state.bankroll = 9500.0
        backtester.state.realized_pnl = 500.0

        result = backtester._create_result()

        assert result.backtest_id == backtester.backtest_id
        assert result.status == "completed"
        assert result.initial_bankroll == 10000.0
        # final = bankroll + realized = 9500 + 500 = 10000
        assert result.final_bankroll == 10000.0

    def test_get_trades_and_equity(self):
        """Test getting trades and equity curve."""
        backtester = StrategyBacktester()
        
        # Should be empty initially
        assert backtester.get_trades() == []
        assert backtester.get_equity_curve() == []


class TestBacktesterEdgeCases:
    """Edge case tests for StrategyBacktester."""

    def test_multiple_fixtures(self):
        """Test handling multiple fixtures."""
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        snapshots = []
        for fixture_id in [100, 200, 300]:
            for i in range(3):
                snapshots.append(
                    HistoricalSnapshot(
                        fixture_id=fixture_id,
                        ticker=f"FIX{fixture_id}-HOME",
                        outcome="home_win",
                        timestamp=now + timedelta(minutes=i * 5),
                        best_bid=0.45,
                        best_ask=0.50,
                        mid=0.475,
                        spread=0.05,
                    )
                )

        kickoffs = {
            100: now + timedelta(hours=24),
            200: now + timedelta(hours=48),
            300: now + timedelta(hours=72),
        }

        backtester = StrategyBacktester()
        result = backtester.run(
            snapshots=snapshots,
            fixture_kickoffs=kickoffs,
        )

        assert result.status == "completed"

    def test_snapshots_out_of_order(self):
        """Test handling snapshots that arrive out of order."""
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

        # Create snapshots in random order
        snapshots = [
            HistoricalSnapshot(
                fixture_id=100,
                ticker="TEST",
                outcome="home_win",
                timestamp=now + timedelta(minutes=10),
                mid=0.50,
            ),
            HistoricalSnapshot(
                fixture_id=100,
                ticker="TEST",
                outcome="home_win",
                timestamp=now,  # Earlier
                mid=0.45,
            ),
            HistoricalSnapshot(
                fixture_id=100,
                ticker="TEST",
                outcome="home_win",
                timestamp=now + timedelta(minutes=5),
                mid=0.48,
            ),
        ]

        backtester = StrategyBacktester()
        result = backtester.run(
            snapshots=snapshots,
            fixture_kickoffs={100: now + timedelta(hours=24)},
        )

        # Should process in chronological order
        assert result.status == "completed"

    def test_fixture_with_too_few_snapshots(self):
        """Test that fixtures with too few snapshots are skipped."""
        now = datetime.now(UTC)

        snapshots = [
            HistoricalSnapshot(
                fixture_id=100,
                ticker="TEST",
                outcome="home_win",
                timestamp=now,
                mid=0.50,
            ),
            # Only 1 snapshot, below minimum
        ]

        backtest_config = BacktestConfig(min_snapshots_per_fixture=3)
        backtester = StrategyBacktester(backtest_config=backtest_config)

        result = backtester.run(
            snapshots=snapshots,
            fixture_kickoffs={100: now + timedelta(hours=24)},
        )

        # Fixture should be filtered out
        assert result.status == "completed"
        assert result.total_trades == 0
