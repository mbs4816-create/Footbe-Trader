"""Tests for the reporting render module."""

import pytest
from datetime import datetime
from pathlib import Path

from footbe_trader.reporting.charts import ChartResult
from footbe_trader.reporting.config import ReportingConfig
from footbe_trader.reporting.queries import (
    DayStats,
    DecisionSummary,
    EdgeBucketStats,
    RunSummary,
    WeekStats,
)
from footbe_trader.reporting.render import (
    HealthCheck,
    HealthCheckItem,
    ReportRenderer,
)


@pytest.fixture
def renderer():
    """Create a ReportRenderer instance."""
    config = ReportingConfig(
        generate_html=True,
        generate_markdown=True,
    )
    return ReportRenderer(config)


@pytest.fixture
def sample_run():
    """Create a sample RunSummary."""
    return RunSummary(
        run_id=1,
        run_type="paper_conservative",
        status="completed",
        started_at=datetime(2024, 1, 15, 10, 0, 0),
        completed_at=datetime(2024, 1, 15, 10, 5, 0),
        fixtures_evaluated=5,
        markets_evaluated=15,
        decisions_made=10,
        orders_placed=3,
        orders_filled=2,
        total_realized_pnl=25.50,
        total_unrealized_pnl=10.00,
        total_pnl=35.50,
        total_exposure=500.0,
        position_count=2,
    )


@pytest.fixture
def sample_decisions():
    """Create sample DecisionSummary objects."""
    return [
        DecisionSummary(
            decision_id="d1",
            run_id=1,
            fixture_id=123,
            market_ticker="TICKER-A",
            outcome="home_win",
            timestamp=datetime(2024, 1, 15, 10, 1, 0),
            action="buy",
            edge=0.08,
            size_executed=10,
            rationale="Good edge opportunity",
        ),
        DecisionSummary(
            decision_id="d2",
            run_id=1,
            fixture_id=123,
            market_ticker="TICKER-A",
            outcome="draw",
            timestamp=datetime(2024, 1, 15, 10, 2, 0),
            action="skip",
            edge=0.02,
            rejection_reason="Edge too low",
        ),
    ]


@pytest.fixture
def sample_charts():
    """Create sample ChartResult objects."""
    return {
        "decision_pie": ChartResult(
            chart_id="decision_pie",
            title="Decisions",
            base64_data="abc123",
        ),
        "edge_buckets": ChartResult(
            chart_id="edge_buckets",
            title="Edge Distribution",
            base64_data="def456",
        ),
    }


class TestHealthCheck:
    """Tests for health check classes."""
    
    def test_health_check_item_healthy(self):
        """Test healthy status icon."""
        item = HealthCheckItem(
            name="Test",
            is_healthy=True,
            message="All good",
        )
        
        assert item.status_icon == "✅"
    
    def test_health_check_item_unhealthy(self):
        """Test unhealthy status icon."""
        item = HealthCheckItem(
            name="Test",
            is_healthy=False,
            message="Problem detected",
        )
        
        assert item.status_icon == "⚠️"
    
    def test_health_check_all_healthy(self):
        """Test overall healthy when all items healthy."""
        health = HealthCheck(items=[
            HealthCheckItem("A", True, "OK"),
            HealthCheckItem("B", True, "OK"),
        ])
        
        assert health.is_healthy is True
    
    def test_health_check_any_unhealthy(self):
        """Test overall unhealthy when any item unhealthy."""
        health = HealthCheck(items=[
            HealthCheckItem("A", True, "OK"),
            HealthCheckItem("B", False, "Problem"),
        ])
        
        assert health.is_healthy is False


class TestReportRenderer:
    """Tests for ReportRenderer class."""
    
    def test_format_currency_positive(self, renderer):
        """Test currency formatting for positive values."""
        result = renderer._format_currency(1234.56)
        assert result == "$1,234.56"
    
    def test_format_currency_negative(self, renderer):
        """Test currency formatting for negative values."""
        result = renderer._format_currency(-1234.56)
        assert result == "-$1,234.56"
    
    def test_format_percent(self, renderer):
        """Test percentage formatting."""
        result = renderer._format_percent(0.1234)
        assert result == "12.34%"
    
    def test_render_run_report_markdown(
        self, renderer, sample_run, sample_decisions, sample_charts, tmp_path
    ):
        """Test rendering a run report to Markdown."""
        # Disable HTML for this test
        renderer.config.generate_html = False
        
        md_path, html_path = renderer.render_run_report(
            run=sample_run,
            decisions=sample_decisions,
            charts=sample_charts,
            output_dir=tmp_path,
        )
        
        assert md_path is not None
        assert md_path.exists()
        assert html_path is None
        
        content = md_path.read_text()
        assert "Run Report: 1" in content
        assert "paper_conservative" in content
        assert "35.50" in content  # Total P&L
    
    def test_render_run_report_html(
        self, renderer, sample_run, sample_decisions, sample_charts, tmp_path
    ):
        """Test rendering a run report to HTML."""
        # Disable Markdown for this test
        renderer.config.generate_markdown = False
        
        md_path, html_path = renderer.render_run_report(
            run=sample_run,
            decisions=sample_decisions,
            charts=sample_charts,
            output_dir=tmp_path,
        )
        
        assert md_path is None
        assert html_path is not None
        assert html_path.exists()
        
        content = html_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Run Report" in content
    
    def test_render_run_report_both_formats(
        self, renderer, sample_run, sample_decisions, sample_charts, tmp_path
    ):
        """Test rendering a run report to both formats."""
        md_path, html_path = renderer.render_run_report(
            run=sample_run,
            decisions=sample_decisions,
            charts=sample_charts,
            output_dir=tmp_path,
        )
        
        assert md_path is not None
        assert html_path is not None
        assert md_path.exists()
        assert html_path.exists()
    
    def test_render_daily_report(self, renderer, tmp_path):
        """Test rendering a daily report."""
        stats = DayStats(
            date="2024-01-15",
            run_count=5,
            decisions_count=50,
            trades_count=20,
            skips_count=30,
            total_pnl=100.0,
            ending_equity=10100.0,
            max_drawdown=0.02,
        )
        
        runs = [
            RunSummary(
                run_id=1,
                run_type="paper_conservative",
                status="completed",
                started_at=datetime(2024, 1, 15, 10, 0),
                decisions_made=10,
                total_pnl=20.0,
            ),
        ]
        
        md_path, html_path = renderer.render_daily_report(
            date="2024-01-15",
            stats=stats,
            runs=runs,
            charts={},
            output_dir=tmp_path,
        )
        
        assert md_path is not None
        assert md_path.name == "2024-01-15.md"
        
        content = md_path.read_text()
        assert "Daily Report: 2024-01-15" in content
        assert "Total Runs | 5" in content
    
    def test_render_weekly_report(self, renderer, tmp_path):
        """Test rendering a weekly report."""
        stats = WeekStats(
            week_start="2024-01-15",
            week_end="2024-01-21",
            run_count=20,
            decisions_count=200,
            trades_count=80,
            skips_count=120,
            total_pnl=500.0,
            return_pct=0.05,
            target_return=0.10,
            pace_status="behind_pace",
            ending_equity=10500.0,
            max_drawdown=0.03,
        )
        
        daily_stats = [
            DayStats(date="2024-01-15", run_count=3, trades_count=10),
            DayStats(date="2024-01-16", run_count=4, trades_count=12),
        ]
        
        edge_bucket_stats = [
            EdgeBucketStats(bucket="<5%", count=50, trades_count=10, skips_count=40),
            EdgeBucketStats(bucket="5-10%", count=30, trades_count=20, skips_count=10),
        ]
        
        md_path, html_path = renderer.render_weekly_report(
            week_start="2024-01-15",
            week_end="2024-01-21",
            stats=stats,
            daily_stats=daily_stats,
            edge_bucket_stats=edge_bucket_stats,
            charts={},
            output_dir=tmp_path,
        )
        
        assert md_path is not None
        assert md_path.name == "2024-01-15.md"
        
        content = md_path.read_text()
        assert "Weekly Report" in content
        assert "behind_pace" in content.upper() or "BEHIND_PACE" in content
    
    def test_render_index(self, renderer, tmp_path):
        """Test rendering the navigation index."""
        health = HealthCheck(items=[
            HealthCheckItem("Agent Activity", True, "OK"),
        ])
        
        md_path, html_path = renderer.render_index(
            total_runs=100,
            latest_run=None,
            current_equity=10000.0,
            weekly_return=0.05,
            target_weekly=0.10,
            weekly_reports=[],
            daily_reports=[],
            recent_runs=[],
            health=health,
            output_dir=tmp_path,
        )
        
        assert md_path is not None
        assert md_path.name == "index.md"
        
        content = md_path.read_text()
        assert "Trading Agent Reports" in content
        assert "Total Runs | 100" in content
    
    def test_markdown_to_html_headers(self, renderer):
        """Test markdown header conversion."""
        md = "# Header 1\n## Header 2\n### Header 3"
        html = renderer._markdown_to_html(md, {})
        
        assert "<h1>Header 1</h1>" in html
        assert "<h2>Header 2</h2>" in html
        assert "<h3>Header 3</h3>" in html
    
    def test_markdown_to_html_table(self, renderer):
        """Test markdown table conversion."""
        md = """| Col1 | Col2 |
|------|------|
| A    | B    |
| C    | D    |"""
        
        html = renderer._markdown_to_html(md, {})
        
        assert "<table>" in html
        assert "<th>Col1</th>" in html
        assert "<td>A</td>" in html
    
    def test_markdown_to_html_bold(self, renderer):
        """Test markdown bold conversion."""
        md = "This is **bold** text"
        html = renderer._markdown_to_html(md, {})
        
        assert "<strong>bold</strong>" in html
    
    def test_markdown_to_html_links(self, renderer):
        """Test markdown link conversion."""
        md = "Click [here](http://example.com)"
        html = renderer._markdown_to_html(md, {})
        
        assert '<a href="http://example.com">here</a>' in html
    
    def test_markdown_to_html_horizontal_rule(self, renderer):
        """Test markdown horizontal rule conversion."""
        md = "Above\n---\nBelow"
        html = renderer._markdown_to_html(md, {})
        
        assert "<hr>" in html
    
    def test_markdown_to_html_code_block(self, renderer):
        """Test markdown code block conversion."""
        md = "```\ncode here\n```"
        html = renderer._markdown_to_html(md, {})
        
        assert "<pre><code>" in html
        assert "code here" in html
        assert "</code></pre>" in html
