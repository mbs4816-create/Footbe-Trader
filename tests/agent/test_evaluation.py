"""Tests for edge bucket evaluation module."""

from datetime import timedelta

import pytest

from footbe_trader.agent.evaluation import (
    EdgeBucket,
    EdgeBucketEvaluator,
    EvaluationReport,
    TradeRecord,
)
from footbe_trader.common.time_utils import utc_now


class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_create_open_trade(self):
        """Test creating an open trade."""
        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=utc_now(),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.05,
            model_prob=0.55,
            market_prob=0.50,
        )

        assert trade.is_closed is False
        assert trade.is_winner is False  # Not closed yet
        assert trade.return_pct is None

    def test_create_closed_trade(self):
        """Test creating a closed trade."""
        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=utc_now() - timedelta(hours=2),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.05,
            model_prob=0.55,
            market_prob=0.50,
            exit_time=utc_now(),
            settlement_value=1.0,  # Won
            realized_pnl=50.0,
        )

        assert trade.is_closed is True
        assert trade.is_winner is True

    def test_return_pct_calculation(self):
        """Test return percentage calculation."""
        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=utc_now(),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.05,
            model_prob=0.55,
            market_prob=0.50,
            realized_pnl=25.0,  # 50% return on 50 cost
        )

        assert trade.return_pct == pytest.approx(0.50)

    def test_to_dict(self):
        """Test serialization."""
        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=utc_now(),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.05,
            model_prob=0.55,
            market_prob=0.50,
        )

        data = trade.to_dict()
        assert data["trade_id"] == "test-001"
        assert data["predicted_edge"] == 0.05


class TestEdgeBucket:
    """Tests for EdgeBucket dataclass."""

    def test_create_bucket(self):
        """Test creating a bucket."""
        bucket = EdgeBucket(
            edge_min=0.02,
            edge_max=0.04,
            bucket_name="2-4%",
            total_trades=20,
            winners=12,
            losers=8,
            total_pnl=500.0,
        )

        assert bucket.closed_trades == 20
        assert bucket.win_rate == pytest.approx(0.60)

    def test_empty_bucket(self):
        """Test empty bucket."""
        bucket = EdgeBucket(
            edge_min=0.02,
            edge_max=0.04,
            bucket_name="2-4%",
        )

        assert bucket.closed_trades == 0
        assert bucket.win_rate is None


class TestEdgeBucketEvaluator:
    """Tests for EdgeBucketEvaluator."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = EdgeBucketEvaluator()
        assert len(evaluator.bucket_definitions) > 0

    def test_record_trade(self):
        """Test recording a trade."""
        evaluator = EdgeBucketEvaluator()

        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=utc_now(),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.03,
            model_prob=0.55,
            market_prob=0.50,
        )

        evaluator.record_trade(trade)

        assert len(evaluator.get_trades()) == 1

    def test_update_trade(self):
        """Test updating a trade."""
        evaluator = EdgeBucketEvaluator()

        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=utc_now(),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.03,
            model_prob=0.55,
            market_prob=0.50,
        )

        evaluator.record_trade(trade)

        success = evaluator.update_trade(
            trade_id="test-001",
            exit_time=utc_now(),
            settlement_value=1.0,
            realized_pnl=50.0,
        )

        assert success is True
        trades = evaluator.get_trades()
        assert trades[0].is_closed is True
        assert trades[0].realized_pnl == 50.0

    def test_get_bucket_for_edge(self):
        """Test bucket assignment by edge."""
        evaluator = EdgeBucketEvaluator()

        assert evaluator.get_bucket_for_edge(0.005) == "0-1%"
        assert evaluator.get_bucket_for_edge(0.015) == "1-2%"
        assert evaluator.get_bucket_for_edge(0.03) == "2-4%"
        assert evaluator.get_bucket_for_edge(0.05) == "4-6%"
        assert evaluator.get_bucket_for_edge(0.08) == "6-10%"
        assert evaluator.get_bucket_for_edge(0.15) == "10%+"

    def test_generate_empty_report(self):
        """Test generating report with no trades."""
        evaluator = EdgeBucketEvaluator()
        report = evaluator.generate_report()

        assert report.total_trades == 0
        assert "No trades" in report.calibration_summary

    def test_generate_report_with_trades(self):
        """Test generating report with trades."""
        evaluator = EdgeBucketEvaluator()

        now = utc_now()

        # Add some trades
        for i in range(5):
            trade = TradeRecord(
                trade_id=f"test-{i:03d}",
                fixture_id=12345 + i,
                outcome="home_win",
                ticker=f"HW_{12345 + i}",
                entry_time=now - timedelta(hours=i),
                entry_price=0.50,
                quantity=100,
                predicted_edge=0.03,
                model_prob=0.55,
                market_prob=0.50,
                settlement_value=1.0 if i % 2 == 0 else 0.0,
                realized_pnl=50.0 if i % 2 == 0 else -50.0,
                hours_to_kickoff=4.0,
            )
            evaluator.record_trade(trade)

        report = evaluator.generate_report()

        assert report.total_trades == 5
        assert report.total_closed == 5
        assert report.overall_win_rate == pytest.approx(0.60)  # 3/5

    def test_generate_text_report(self):
        """Test text report generation."""
        evaluator = EdgeBucketEvaluator()

        now = utc_now()

        # Add a trade
        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=now - timedelta(hours=1),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.03,
            model_prob=0.55,
            market_prob=0.50,
            settlement_value=1.0,
            realized_pnl=50.0,
        )
        evaluator.record_trade(trade)

        text = evaluator.generate_text_report()

        assert "EDGE BUCKET EVALUATION REPORT" in text
        assert "BY EDGE BUCKET" in text
        assert "BY OUTCOME" in text

    def test_clear_trades(self):
        """Test clearing trades."""
        evaluator = EdgeBucketEvaluator()

        trade = TradeRecord(
            trade_id="test-001",
            fixture_id=12345,
            outcome="home_win",
            ticker="HW_12345",
            entry_time=utc_now(),
            entry_price=0.50,
            quantity=100,
            predicted_edge=0.03,
            model_prob=0.55,
            market_prob=0.50,
        )

        evaluator.record_trade(trade)
        assert len(evaluator.get_trades()) == 1

        evaluator.clear_trades()
        assert len(evaluator.get_trades()) == 0

    def test_stats_by_outcome(self):
        """Test statistics grouped by outcome."""
        evaluator = EdgeBucketEvaluator()

        now = utc_now()

        for outcome in ["home_win", "draw", "away_win"]:
            for i in range(3):
                trade = TradeRecord(
                    trade_id=f"{outcome}-{i}",
                    fixture_id=12345 + i,
                    outcome=outcome,
                    ticker=f"{outcome}_{12345 + i}",
                    entry_time=now - timedelta(hours=i),
                    entry_price=0.50,
                    quantity=100,
                    predicted_edge=0.03,
                    model_prob=0.55,
                    market_prob=0.50,
                    settlement_value=1.0 if i == 0 else 0.0,
                    realized_pnl=50.0 if i == 0 else -50.0,
                )
                evaluator.record_trade(trade)

        report = evaluator.generate_report()

        assert "home_win" in report.stats_by_outcome
        assert "draw" in report.stats_by_outcome
        assert "away_win" in report.stats_by_outcome
        assert report.stats_by_outcome["home_win"]["total_trades"] == 3

    def test_stats_by_time_category(self):
        """Test statistics grouped by time to kickoff."""
        evaluator = EdgeBucketEvaluator()

        now = utc_now()

        # Trades at different times to kickoff
        times = [
            ("very_early", 60.0),
            ("early", 36.0),
            ("standard", 12.0),
            ("optimal", 4.0),
            ("late", 1.0),
        ]

        for i, (_, hours) in enumerate(times):
            trade = TradeRecord(
                trade_id=f"test-{i}",
                fixture_id=12345 + i,
                outcome="home_win",
                ticker=f"HW_{12345 + i}",
                entry_time=now - timedelta(hours=i),
                entry_price=0.50,
                quantity=100,
                predicted_edge=0.03,
                model_prob=0.55,
                market_prob=0.50,
                hours_to_kickoff=hours,
                settlement_value=1.0,
                realized_pnl=50.0,
            )
            evaluator.record_trade(trade)

        report = evaluator.generate_report()

        # Should have stats for each time category
        assert "optimal" in report.stats_by_time_category
        assert "late" in report.stats_by_time_category


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def test_create_report(self):
        """Test creating a report."""
        report = EvaluationReport(
            report_id="test-report",
            total_trades=100,
            total_closed=80,
            overall_win_rate=0.55,
            overall_pnl=500.0,
        )

        assert report.report_id == "test-report"
        assert report.total_trades == 100
        assert report.overall_win_rate == 0.55

    def test_to_dict(self):
        """Test serialization."""
        report = EvaluationReport(
            report_id="test-report",
            total_trades=100,
            total_closed=80,
        )

        data = report.to_dict()
        assert data["report_id"] == "test-report"
        assert data["total_trades"] == 100
