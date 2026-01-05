"""Tests for backtest metrics module."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from footbe_trader.storage.models import (
    BacktestEquity,
    BacktestTrade,
    StrategyBacktest,
)
from footbe_trader.strategy.backtest_metrics import (
    BacktestMetricsCalculator,
    BacktestMetricsResult,
    EdgeCalibrationBucket,
    ExposureMetrics,
    OutcomeBreakdown,
    RatioMetrics,
    ReturnMetrics,
    RiskMetrics,
    TradeMetrics,
    generate_backtest_report,
)


class TestReturnMetrics:
    """Tests for ReturnMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = ReturnMetrics()
        assert metrics.total_return == 0.0
        assert metrics.total_return_pct == 0.0
        assert metrics.annualized_return is None

    def test_to_dict(self):
        """Test serialization."""
        metrics = ReturnMetrics(
            total_return=1000.0,
            total_return_pct=0.10,
            best_day=0.05,
            worst_day=-0.03,
        )
        data = metrics.to_dict()
        assert data["total_return"] == 1000.0
        assert data["total_return_pct"] == 0.10
        assert data["best_day"] == 0.05


class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = RiskMetrics()
        assert metrics.max_drawdown == 0.0
        assert metrics.volatility is None

    def test_to_dict(self):
        """Test serialization."""
        metrics = RiskMetrics(
            max_drawdown=0.15,
            avg_drawdown=0.05,
            volatility=0.02,
        )
        data = metrics.to_dict()
        assert data["max_drawdown"] == 0.15
        assert data["avg_drawdown"] == 0.05


class TestTradeMetrics:
    """Tests for TradeMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = TradeMetrics()
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_to_dict(self):
        """Test serialization."""
        metrics = TradeMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.60,
        )
        data = metrics.to_dict()
        assert data["total_trades"] == 100
        assert data["win_rate"] == 0.60


class TestBacktestMetricsCalculator:
    """Tests for BacktestMetricsCalculator."""

    def _create_sample_backtest(self) -> StrategyBacktest:
        """Create a sample backtest result."""
        return StrategyBacktest(
            backtest_id="test-001",
            strategy_config_hash="abc123",
            initial_bankroll=10000.0,
            final_bankroll=11000.0,
            total_return=0.10,
            total_trades=20,
        )

    def _create_sample_trades(self) -> list[BacktestTrade]:
        """Create sample trades."""
        now = datetime.now(UTC)
        trades = []

        # Winning trades
        for i in range(12):
            trades.append(
                BacktestTrade(
                    backtest_id="test-001",
                    trade_id=f"T{i:04d}",
                    fixture_id=100 + i,
                    ticker=f"TICKER-{i}",
                    outcome="home_win" if i % 3 == 0 else "draw" if i % 3 == 1 else "away_win",
                    entry_timestamp=now + timedelta(hours=i),
                    entry_price=0.45,
                    entry_quantity=10,
                    entry_edge=0.05 + (i * 0.01),
                    entry_model_prob=0.50 + (i * 0.02),
                    exit_timestamp=now + timedelta(hours=i, minutes=30),
                    exit_price=0.55,
                    exit_quantity=10,
                    exit_reason="take_profit",
                    realized_pnl=1.0,  # $1 profit
                    return_pct=0.22,  # 22% return
                    hold_time_minutes=30,
                )
            )

        # Losing trades
        for i in range(8):
            trades.append(
                BacktestTrade(
                    backtest_id="test-001",
                    trade_id=f"T{12 + i:04d}",
                    fixture_id=200 + i,
                    ticker=f"TICKER-LOSS-{i}",
                    outcome="home_win",
                    entry_timestamp=now + timedelta(hours=12 + i),
                    entry_price=0.50,
                    entry_quantity=10,
                    entry_edge=0.05,
                    entry_model_prob=0.55,
                    exit_timestamp=now + timedelta(hours=12 + i, minutes=45),
                    exit_price=0.40,
                    exit_quantity=10,
                    exit_reason="stop_loss",
                    realized_pnl=-1.0,  # $1 loss
                    return_pct=-0.20,  # -20% return
                    hold_time_minutes=45,
                )
            )

        return trades

    def _create_sample_equity_curve(self) -> list[BacktestEquity]:
        """Create sample equity curve."""
        now = datetime.now(UTC)
        curve = []

        bankroll = 10000.0
        for i in range(24):  # 24 hours of data
            timestamp = now + timedelta(hours=i)
            pnl = 10 * i  # Linear growth
            drawdown = 0.01 * (i % 5)  # Periodic drawdowns

            curve.append(
                BacktestEquity(
                    backtest_id="test-001",
                    timestamp=timestamp,
                    bankroll=bankroll - 50,  # Some exposure taken
                    total_exposure=50.0,
                    position_count=2,
                    realized_pnl=pnl,
                    unrealized_pnl=10.0,
                    total_pnl=pnl + 10,
                    drawdown=drawdown,
                )
            )

        return curve

    def test_calculate_returns(self):
        """Test return metrics calculation."""
        calculator = BacktestMetricsCalculator()
        backtest = self._create_sample_backtest()
        equity = self._create_sample_equity_curve()

        metrics = calculator._calculate_return_metrics(backtest, equity)

        assert metrics.total_return == 1000.0  # 11000 - 10000
        assert metrics.total_return_pct == 0.10  # 10%

    def test_calculate_trade_metrics(self):
        """Test trade metrics calculation."""
        calculator = BacktestMetricsCalculator()
        trades = self._create_sample_trades()

        metrics = calculator._calculate_trade_metrics(trades)

        assert metrics.total_trades == 20
        assert metrics.winning_trades == 12
        assert metrics.losing_trades == 8
        assert metrics.win_rate == 0.60

    def test_calculate_trade_metrics_empty(self):
        """Test trade metrics with no trades."""
        calculator = BacktestMetricsCalculator()
        metrics = calculator._calculate_trade_metrics([])

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        calculator = BacktestMetricsCalculator()
        equity = self._create_sample_equity_curve()

        metrics = calculator._calculate_risk_metrics(equity)

        assert metrics.max_drawdown >= 0
        assert metrics.avg_drawdown >= 0

    def test_calculate_risk_metrics_empty(self):
        """Test risk metrics with no data."""
        calculator = BacktestMetricsCalculator()
        metrics = calculator._calculate_risk_metrics([])

        assert metrics.max_drawdown == 0.0

    def test_calculate_exposure_metrics(self):
        """Test exposure metrics calculation."""
        calculator = BacktestMetricsCalculator()
        equity = self._create_sample_equity_curve()

        metrics = calculator._calculate_exposure_metrics(equity, 10000.0)

        assert metrics.avg_exposure == 50.0
        assert metrics.max_exposure == 50.0
        assert metrics.avg_position_count == 2.0
        assert metrics.exposure_utilization == 0.005  # 50/10000

    def test_calculate_edge_calibration(self):
        """Test edge calibration calculation."""
        calculator = BacktestMetricsCalculator()
        trades = self._create_sample_trades()

        calibration = calculator._calculate_edge_calibration(trades)

        assert len(calibration) > 0
        # Check buckets have data
        buckets = {b.edge_bucket: b for b in calibration}
        assert "0.05-0.10" in buckets or "0.10-0.15" in buckets

    def test_calculate_outcome_breakdown(self):
        """Test outcome breakdown calculation."""
        calculator = BacktestMetricsCalculator()
        trades = self._create_sample_trades()

        breakdown = calculator._calculate_outcome_breakdown(trades)

        assert len(breakdown) > 0
        outcomes = {b.outcome for b in breakdown}
        assert "home_win" in outcomes

    def test_full_calculation(self):
        """Test full metrics calculation."""
        calculator = BacktestMetricsCalculator()
        backtest = self._create_sample_backtest()
        trades = self._create_sample_trades()
        equity = self._create_sample_equity_curve()

        result = calculator.calculate(backtest, trades, equity)

        assert result.backtest_id == "test-001"
        assert result.initial_bankroll == 10000.0
        assert result.returns.total_return == 1000.0
        assert result.trades.total_trades == 20
        assert result.fixture_count == 20  # All unique fixtures

    def test_aggregate_daily_returns(self):
        """Test daily return aggregation."""
        calculator = BacktestMetricsCalculator()
        now = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

        equity = []
        for day in range(3):
            for hour in range(4):
                equity.append(
                    BacktestEquity(
                        backtest_id="test",
                        timestamp=now + timedelta(days=day, hours=hour * 6),
                        bankroll=10000 + (day * 100) + (hour * 10),
                        total_pnl=day * 100 + hour * 10,
                    )
                )

        daily_returns = calculator._aggregate_daily_returns(equity)

        # Should have 2 returns (day 1 vs day 0, day 2 vs day 1)
        assert len(daily_returns) == 2


class TestReportGeneration:
    """Tests for report generation."""

    def _create_sample_metrics(self) -> BacktestMetricsResult:
        """Create sample metrics result."""
        now = datetime.now(UTC)
        return BacktestMetricsResult(
            backtest_id="test-001",
            strategy_config_hash="abc123",
            backtest_period_start=now,
            backtest_period_end=now + timedelta(hours=24),
            backtest_duration_hours=24.0,
            initial_bankroll=10000.0,
            final_bankroll=11000.0,
            returns=ReturnMetrics(
                total_return=1000.0,
                total_return_pct=0.10,
                positive_days=5,
                negative_days=2,
            ),
            risk=RiskMetrics(
                max_drawdown=0.05,
                avg_drawdown=0.02,
            ),
            ratios=RatioMetrics(
                sharpe_ratio=1.5,
                profit_factor=2.0,
            ),
            trades=TradeMetrics(
                total_trades=20,
                winning_trades=12,
                losing_trades=8,
                win_rate=0.60,
                avg_win=10.0,
                avg_loss=-8.0,
                avg_trade=2.0,
                avg_hold_time_minutes=30.0,
            ),
            exposure=ExposureMetrics(
                avg_exposure=500.0,
                max_exposure=1000.0,
                avg_position_count=3.0,
                max_position_count=5,
                exposure_utilization=0.05,
            ),
            edge_calibration=[
                EdgeCalibrationBucket(
                    edge_bucket="0.05-0.10",
                    trade_count=10,
                    expected_return=0.07,
                    actual_return=0.08,
                    win_rate=0.70,
                    calibration_error=-0.01,
                ),
                EdgeCalibrationBucket(
                    edge_bucket="0.10-0.15",
                    trade_count=5,
                    expected_return=0.12,
                    actual_return=0.10,
                    win_rate=0.60,
                    calibration_error=0.02,
                ),
            ],
            outcome_breakdown=[
                OutcomeBreakdown(
                    outcome="home_win",
                    trades=8,
                    winning=5,
                    losing=3,
                    win_rate=0.625,
                    total_pnl=50.0,
                    avg_pnl=6.25,
                ),
                OutcomeBreakdown(
                    outcome="draw",
                    trades=6,
                    winning=3,
                    losing=3,
                    win_rate=0.50,
                    total_pnl=10.0,
                    avg_pnl=1.67,
                ),
            ],
            fixture_count=15,
        )

    def test_generate_report_creates_files(self):
        """Test that report generation creates expected files."""
        metrics = self._create_sample_metrics()

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_backtest_report(metrics, output_dir)

            # Check files exist
            assert (output_dir / "backtest_summary.txt").exists()
            assert (output_dir / "backtest_metrics.json").exists()
            assert (output_dir / "edge_calibration.csv").exists()
            assert (output_dir / "outcome_breakdown.csv").exists()

    def test_generate_report_summary_content(self):
        """Test summary file content."""
        metrics = self._create_sample_metrics()

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_backtest_report(metrics, output_dir)

            summary = (output_dir / "backtest_summary.txt").read_text()

            assert "BACKTEST RESULTS" in summary or "BACKTEST REPORT" in summary
            assert "test-001" in summary
            assert "10,000" in summary or "10000" in summary

    def test_generate_report_json_valid(self):
        """Test JSON file is valid."""
        import json

        metrics = self._create_sample_metrics()

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_backtest_report(metrics, output_dir)

            json_path = output_dir / "backtest_metrics.json"
            with open(json_path) as f:
                data = json.load(f)

            assert data["backtest_id"] == "test-001"
            assert data["initial_bankroll"] == 10000.0

    def test_generate_report_csv_valid(self):
        """Test CSV files are valid."""
        metrics = self._create_sample_metrics()

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            generate_backtest_report(metrics, output_dir)

            # Check edge calibration CSV
            edge_csv = (output_dir / "edge_calibration.csv").read_text()
            lines = edge_csv.strip().split("\n")
            assert len(lines) >= 2  # Header + data
            assert "edge_bucket" in lines[0]

            # Check outcome breakdown CSV
            outcome_csv = (output_dir / "outcome_breakdown.csv").read_text()
            lines = outcome_csv.strip().split("\n")
            assert len(lines) >= 2
            assert "outcome" in lines[0]


class TestMetricsToDict:
    """Tests for metrics serialization."""

    def test_backtest_metrics_result_to_dict(self):
        """Test full result serialization."""
        result = BacktestMetricsResult(
            backtest_id="test",
            strategy_config_hash="hash",
            initial_bankroll=10000.0,
            final_bankroll=11000.0,
            fixture_count=10,
        )

        data = result.to_dict()

        assert data["backtest_id"] == "test"
        assert data["initial_bankroll"] == 10000.0
        assert data["fixture_count"] == 10
        assert "returns" in data
        assert "risk" in data
        assert "trades" in data

    def test_edge_calibration_bucket_to_dict(self):
        """Test edge calibration serialization."""
        bucket = EdgeCalibrationBucket(
            edge_bucket="0.05-0.10",
            trade_count=20,
            expected_return=0.07,
            actual_return=0.08,
            win_rate=0.65,
            calibration_error=-0.01,
        )

        data = bucket.to_dict()

        assert data["edge_bucket"] == "0.05-0.10"
        assert data["trade_count"] == 20
        assert data["calibration_error"] == -0.01

    def test_outcome_breakdown_to_dict(self):
        """Test outcome breakdown serialization."""
        breakdown = OutcomeBreakdown(
            outcome="home_win",
            trades=15,
            winning=10,
            losing=5,
            win_rate=0.667,
            total_pnl=100.0,
            avg_pnl=6.67,
            avg_edge=0.08,
        )

        data = breakdown.to_dict()

        assert data["outcome"] == "home_win"
        assert data["trades"] == 15
        assert data["avg_edge"] == 0.08
