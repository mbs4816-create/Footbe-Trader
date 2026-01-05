"""Tests for the reporting charts module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from footbe_trader.reporting.charts import ChartGenerator, ChartResult
from footbe_trader.reporting.config import ReportingConfig


@pytest.fixture
def chart_generator():
    """Create a ChartGenerator instance."""
    config = ReportingConfig(embed_charts_base64=True)
    return ChartGenerator(config)


class TestChartResult:
    """Tests for ChartResult dataclass."""
    
    def test_get_html_img_base64(self):
        """Test HTML img tag with base64 data."""
        result = ChartResult(
            chart_id="test",
            title="Test Chart",
            base64_data="abc123",
        )
        
        html = result.get_html_img()
        
        assert "data:image/png;base64,abc123" in html
        assert "alt=\"Test Chart\"" in html
    
    def test_get_html_img_file_path(self):
        """Test HTML img tag with file path."""
        result = ChartResult(
            chart_id="test",
            title="Test Chart",
            png_path=Path("/path/to/chart.png"),
        )
        
        html = result.get_html_img()
        
        assert "/path/to/chart.png" in html
    
    def test_get_html_img_fallback(self):
        """Test HTML fallback when no image data."""
        result = ChartResult(
            chart_id="test",
            title="Test Chart",
        )
        
        html = result.get_html_img()
        
        assert "[Chart: Test Chart]" in html
    
    def test_get_markdown_img_file_path(self):
        """Test Markdown image syntax with file path."""
        result = ChartResult(
            chart_id="test",
            title="Test Chart",
            png_path=Path("/path/to/chart.png"),
        )
        
        md = result.get_markdown_img()
        
        assert "![Test Chart]" in md
        assert "/path/to/chart.png" in md


class TestChartGenerator:
    """Tests for ChartGenerator class."""
    
    def test_equity_curve_empty_data(self, chart_generator):
        """Test equity curve with empty data."""
        result = chart_generator.equity_curve(
            timestamps=[],
            values=[],
            title="Empty Equity Curve",
        )
        
        assert result.chart_id == "equity_curve"
        assert result.title == "Empty Equity Curve"
        assert result.base64_data is not None  # Still generates image
    
    def test_equity_curve_with_data(self, chart_generator):
        """Test equity curve with sample data."""
        now = datetime.now()
        timestamps = [now + timedelta(hours=i) for i in range(10)]
        values = [10000 + i * 50 for i in range(10)]
        
        result = chart_generator.equity_curve(
            timestamps=timestamps,
            values=values,
            title="Test Equity Curve",
            chart_id="test_equity",
        )
        
        assert result.chart_id == "test_equity"
        assert result.base64_data is not None
        assert len(result.base64_data) > 100  # Non-trivial content
    
    def test_equity_curve_with_target_line(self, chart_generator):
        """Test equity curve with target reference line."""
        now = datetime.now()
        timestamps = [now + timedelta(hours=i) for i in range(5)]
        values = [10000, 10100, 10050, 10200, 10150]
        
        result = chart_generator.equity_curve(
            timestamps=timestamps,
            values=values,
            title="Equity with Target",
            target_line=10500.0,
        )
        
        assert result.base64_data is not None
    
    def test_equity_curve_save_to_file(self, chart_generator, tmp_path):
        """Test saving equity curve to file."""
        now = datetime.now()
        timestamps = [now + timedelta(hours=i) for i in range(5)]
        values = [10000 + i * 25 for i in range(5)]
        
        save_path = tmp_path / "test_chart.png"
        
        result = chart_generator.equity_curve(
            timestamps=timestamps,
            values=values,
            title="Saved Chart",
            save_path=save_path,
        )
        
        assert result.png_path == save_path
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    
    def test_edge_bucket_bar_empty(self, chart_generator):
        """Test edge bucket bar chart with empty data."""
        result = chart_generator.edge_bucket_bar(
            buckets=[],
            counts=[],
            title="Empty Buckets",
        )
        
        assert result.chart_id == "edge_buckets"
        assert result.base64_data is not None
    
    def test_edge_bucket_bar_with_data(self, chart_generator):
        """Test edge bucket bar chart with sample data."""
        buckets = ["<5%", "5-10%", "10-15%", "15%+"]
        counts = [10, 25, 15, 5]
        
        result = chart_generator.edge_bucket_bar(
            buckets=buckets,
            counts=counts,
            title="Edge Distribution",
            chart_id="test_edge",
        )
        
        assert result.chart_id == "test_edge"
        assert result.base64_data is not None
    
    def test_decision_pie_empty(self, chart_generator):
        """Test decision pie chart with empty data."""
        result = chart_generator.decision_pie(
            actions={},
            title="Empty Decisions",
        )
        
        assert result.chart_id == "decision_pie"
        assert result.base64_data is not None
    
    def test_decision_pie_with_data(self, chart_generator):
        """Test decision pie chart with sample data."""
        actions = {
            "buy": 30,
            "skip": 50,
            "hold": 10,
            "exit": 5,
        }
        
        result = chart_generator.decision_pie(
            actions=actions,
            title="Decision Distribution",
            chart_id="test_pie",
        )
        
        assert result.chart_id == "test_pie"
        assert result.base64_data is not None
    
    def test_pace_vs_target_empty(self, chart_generator):
        """Test pace vs target chart with empty data."""
        result = chart_generator.pace_vs_target(
            dates=[],
            actual_returns=[],
            target_return=0.10,
            tolerance=0.02,
            title="Empty Pace",
        )
        
        assert result.chart_id == "pace_vs_target"
        assert result.base64_data is not None
    
    def test_pace_vs_target_with_data(self, chart_generator):
        """Test pace vs target chart with sample data."""
        now = datetime.now()
        dates = [now + timedelta(days=i) for i in range(7)]
        returns = [0.01, 0.025, 0.04, 0.055, 0.07, 0.085, 0.10]
        
        result = chart_generator.pace_vs_target(
            dates=dates,
            actual_returns=returns,
            target_return=0.10,
            tolerance=0.02,
            title="Pace vs Target",
        )
        
        assert result.base64_data is not None
    
    def test_drawdown_chart_empty(self, chart_generator):
        """Test drawdown chart with empty data."""
        result = chart_generator.drawdown_chart(
            timestamps=[],
            drawdowns=[],
            title="Empty Drawdown",
        )
        
        assert result.chart_id == "drawdown"
        assert result.base64_data is not None
    
    def test_drawdown_chart_with_data(self, chart_generator):
        """Test drawdown chart with sample data."""
        now = datetime.now()
        timestamps = [now + timedelta(hours=i) for i in range(10)]
        drawdowns = [0.0, 0.01, 0.02, 0.05, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01]
        
        result = chart_generator.drawdown_chart(
            timestamps=timestamps,
            drawdowns=drawdowns,
            title="Drawdown Chart",
            threshold_lines=[(0.07, "Soft"), (0.15, "Hard")],
        )
        
        assert result.base64_data is not None
    
    def test_rejection_reasons_bar_empty(self, chart_generator):
        """Test rejection reasons chart with empty data."""
        result = chart_generator.rejection_reasons_bar(
            reasons={},
            title="No Rejections",
        )
        
        assert result.chart_id == "rejection_reasons"
        assert result.base64_data is not None
    
    def test_rejection_reasons_bar_with_data(self, chart_generator):
        """Test rejection reasons chart with sample data."""
        reasons = {
            "edge_too_low": 25,
            "exposure_limit": 15,
            "drawdown_severe": 5,
            "market_closed": 3,
        }
        
        result = chart_generator.rejection_reasons_bar(
            reasons=reasons,
            title="Skip Reasons",
            max_reasons=10,
        )
        
        assert result.base64_data is not None
    
    def test_rejection_reasons_truncates_long_labels(self, chart_generator):
        """Test that long rejection reason labels are truncated."""
        reasons = {
            "this_is_a_very_long_rejection_reason_that_exceeds_forty_characters": 10,
            "short": 5,
        }
        
        result = chart_generator.rejection_reasons_bar(
            reasons=reasons,
            title="Long Labels",
        )
        
        # Should generate without error
        assert result.base64_data is not None
    
    def test_trades_timeline_empty(self, chart_generator):
        """Test trades timeline with empty data."""
        result = chart_generator.trades_timeline(
            trade_times=[],
            trade_pnls=[],
            title="No Trades",
        )
        
        assert result.chart_id == "trades_timeline"
        assert result.base64_data is not None
    
    def test_trades_timeline_with_data(self, chart_generator):
        """Test trades timeline with sample data."""
        now = datetime.now()
        trade_times = [now + timedelta(hours=i * 2) for i in range(10)]
        trade_pnls = [10, -5, 25, -10, 15, 30, -20, 5, 40, -15]
        
        result = chart_generator.trades_timeline(
            trade_times=trade_times,
            trade_pnls=trade_pnls,
            title="Trade Timeline",
        )
        
        assert result.base64_data is not None
    
    def test_chart_generator_custom_config(self, tmp_path):
        """Test chart generator with custom configuration."""
        config = ReportingConfig(
            reports_dir=tmp_path,
            chart_dpi=150,
            chart_width=12.0,
            chart_height=8.0,
            embed_charts_base64=False,
        )
        
        generator = ChartGenerator(config)
        
        now = datetime.now()
        result = generator.equity_curve(
            timestamps=[now],
            values=[10000],
            title="Custom Config Chart",
        )
        
        # With embed_charts_base64=False, should be None
        assert result.base64_data is None
