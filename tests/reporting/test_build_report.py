"""Tests for the reporting build_report module."""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from footbe_trader.reporting.build_report import (
    DailyReport,
    ReportBuilder,
    RunReport,
    WeeklyReport,
)
from footbe_trader.reporting.config import ReportingConfig


@pytest.fixture
def db_connection(tmp_path):
    """Create an in-memory database with test schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    # Create all necessary tables
    conn.execute("""
        CREATE TABLE agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            fixtures_evaluated INTEGER DEFAULT 0,
            markets_evaluated INTEGER DEFAULT 0,
            decisions_made INTEGER DEFAULT 0,
            orders_placed INTEGER DEFAULT 0,
            orders_filled INTEGER DEFAULT 0,
            orders_rejected INTEGER DEFAULT 0,
            total_realized_pnl REAL DEFAULT 0.0,
            total_unrealized_pnl REAL DEFAULT 0.0,
            total_exposure REAL DEFAULT 0.0,
            position_count INTEGER DEFAULT 0,
            error_count INTEGER DEFAULT 0,
            error_message TEXT
        )
    """)
    
    conn.execute("""
        CREATE TABLE decision_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL UNIQUE,
            run_id INTEGER,
            fixture_id INTEGER,
            market_ticker TEXT NOT NULL,
            outcome TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            league_key TEXT,
            hours_to_kickoff REAL,
            time_category TEXT,
            pace_state TEXT,
            drawdown REAL DEFAULT 0.0,
            drawdown_band TEXT,
            gross_exposure REAL DEFAULT 0.0,
            equity REAL DEFAULT 0.0,
            throttle_multiplier REAL DEFAULT 1.0,
            size_intended INTEGER,
            size_executed INTEGER,
            edge_bucket TEXT,
            edge_calculation_json TEXT,
            market_snapshot_json TEXT,
            model_prediction_json TEXT,
            rationale TEXT,
            rejection_reason TEXT,
            order_placed INTEGER DEFAULT 0
        )
    """)
    
    conn.execute("""
        CREATE TABLE pnl_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL UNIQUE,
            run_id INTEGER,
            timestamp TEXT NOT NULL,
            total_realized_pnl REAL DEFAULT 0.0,
            total_unrealized_pnl REAL DEFAULT 0.0,
            total_pnl REAL DEFAULT 0.0,
            total_exposure REAL DEFAULT 0.0,
            position_count INTEGER DEFAULT 0,
            bankroll REAL DEFAULT 0.0
        )
    """)
    
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def report_builder(db_connection, tmp_path):
    """Create a ReportBuilder instance."""
    config = ReportingConfig(reports_dir=tmp_path / "reports")
    return ReportBuilder(db_connection, config)


@pytest.fixture
def sample_data(db_connection):
    """Insert sample data for testing."""
    cursor = db_connection.cursor()
    
    # Insert a run
    cursor.execute("""
        INSERT INTO agent_runs (
            run_type, status, started_at, completed_at,
            fixtures_evaluated, markets_evaluated, decisions_made,
            orders_placed, orders_filled, total_realized_pnl, total_unrealized_pnl
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "paper_conservative",
        "completed",
        "2024-01-15T10:00:00",
        "2024-01-15T10:05:00",
        5, 15, 10, 3, 2, 25.50, 10.00,
    ))
    run_id = cursor.lastrowid
    
    # Insert decisions
    for i in range(5):
        action = "buy" if i % 2 == 0 else "skip"
        cursor.execute("""
            INSERT INTO decision_records (
                decision_id, run_id, fixture_id, market_ticker, outcome,
                timestamp, action, edge_bucket
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"d{i}",
            run_id,
            100 + i,
            f"TICKER-{i}",
            "home_win",
            f"2024-01-15T10:0{i}:00",
            action,
            "<5%" if i < 2 else "5-10%",
        ))
    
    # Insert P&L snapshots
    for i in range(5):
        cursor.execute("""
            INSERT INTO pnl_snapshots (
                snapshot_id, run_id, timestamp, bankroll, total_pnl
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            f"snap_{i}",
            run_id,
            f"2024-01-15T10:0{i}:00",
            10000 + i * 10,
            i * 10,
        ))
    
    db_connection.commit()
    return {"run_id": run_id}


class TestRunReport:
    """Tests for RunReport dataclass."""
    
    def test_to_dict(self, report_builder, sample_data):
        """Test conversion to dictionary."""
        report = report_builder.build_run_report(sample_data["run_id"])
        
        assert report is not None
        data = report.to_dict()
        
        assert "run_id" in data
        assert "run" in data
        assert "decisions" in data
        assert data["run_id"] == sample_data["run_id"]


class TestReportBuilder:
    """Tests for ReportBuilder class."""
    
    def test_build_run_report(self, report_builder, sample_data):
        """Test building a run report."""
        report = report_builder.build_run_report(sample_data["run_id"])
        
        assert report is not None
        assert isinstance(report, RunReport)
        assert report.run_id == sample_data["run_id"]
        assert len(report.decisions) == 5
        assert report.run.status == "completed"
    
    def test_build_run_report_not_found(self, report_builder):
        """Test building report for non-existent run."""
        report = report_builder.build_run_report(999)
        assert report is None
    
    def test_build_run_report_generates_charts(self, report_builder, sample_data):
        """Test that run report generates charts."""
        report = report_builder.build_run_report(sample_data["run_id"])
        
        assert report is not None
        assert "decision_pie" in report.charts
        assert "edge_buckets" in report.charts
    
    def test_build_run_report_saves_files(self, report_builder, sample_data):
        """Test that run report saves MD and HTML files."""
        report = report_builder.build_run_report(sample_data["run_id"])
        
        assert report is not None
        assert report.markdown_path is not None
        assert report.html_path is not None
        assert report.markdown_path.exists()
        assert report.html_path.exists()
    
    def test_build_run_report_saves_artifact(self, report_builder, sample_data):
        """Test that run report saves JSON artifact."""
        report = report_builder.build_run_report(sample_data["run_id"], save_artifact=True)
        
        assert report is not None
        assert report.artifact_path is not None
        assert report.artifact_path.exists()
        
        # Verify artifact content
        artifact_data = json.loads(report.artifact_path.read_text())
        assert artifact_data["run_id"] == sample_data["run_id"]
        assert "decisions" in artifact_data
    
    def test_build_daily_report(self, report_builder, sample_data):
        """Test building a daily report."""
        report = report_builder.build_daily_report("2024-01-15")
        
        assert isinstance(report, DailyReport)
        assert report.date == "2024-01-15"
        assert report.stats.decisions_count == 5
    
    def test_build_daily_report_default_today(self, report_builder, sample_data, db_connection):
        """Test building daily report defaults to today."""
        # Add data for today
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO agent_runs (run_type, status, started_at)
            VALUES (?, ?, ?)
        """, ("paper_conservative", "completed", f"{today}T12:00:00"))
        db_connection.commit()
        
        report = report_builder.build_daily_report()
        
        assert isinstance(report, DailyReport)
        assert report.date == today
    
    def test_build_daily_report_generates_charts(self, report_builder, sample_data):
        """Test that daily report generates charts when data available."""
        report = report_builder.build_daily_report("2024-01-15")
        
        # With P&L snapshots, should have equity curve
        assert "equity_curve" in report.charts or len(report.charts) > 0
    
    def test_build_weekly_report(self, report_builder, sample_data):
        """Test building a weekly report."""
        # 2024-01-15 is a Monday
        report = report_builder.build_weekly_report("2024-01-15")
        
        assert isinstance(report, WeeklyReport)
        assert report.week_start == "2024-01-15"
        assert report.week_end == "2024-01-21"
    
    def test_build_weekly_report_calculates_pace(self, report_builder, sample_data):
        """Test that weekly report calculates pace status."""
        report = report_builder.build_weekly_report("2024-01-15")
        
        assert report.stats.pace_status in ["ahead_of_pace", "on_pace", "behind_pace"]
    
    def test_build_index(self, report_builder, sample_data):
        """Test building the navigation index."""
        md_path, html_path = report_builder.build_index()
        
        assert md_path is not None
        assert html_path is not None
        assert md_path.exists()
        assert html_path.exists()
    
    def test_build_index_includes_health_check(self, report_builder, sample_data):
        """Test that index includes health check."""
        md_path, _ = report_builder.build_index()
        
        content = md_path.read_text()
        assert "Health Check" in content
    
    def test_build_all_reports(self, report_builder, sample_data):
        """Test building all reports."""
        # Should not raise
        report_builder.build_all_reports(days_back=1)
        
        # Check that index was created
        assert (report_builder.config.reports_dir / "index.md").exists()
    
    def test_directories_created(self, report_builder):
        """Test that report directories are created."""
        assert report_builder.config.reports_dir.exists()
        assert report_builder.config.artifacts_dir.exists()
        assert report_builder.config.runs_dir.exists()
        assert report_builder.config.daily_dir.exists()
        assert report_builder.config.weekly_dir.exists()


class TestReportBuilderIntegration:
    """Integration tests for the full reporting pipeline."""
    
    def test_full_pipeline(self, db_connection, tmp_path):
        """Test the full reporting pipeline end-to-end."""
        # Setup
        config = ReportingConfig(reports_dir=tmp_path / "reports")
        builder = ReportBuilder(db_connection, config)
        
        # Insert more comprehensive test data
        cursor = db_connection.cursor()
        
        # Multiple runs across multiple days
        for day in range(3):
            date = f"2024-01-{15 + day:02d}"
            for run_num in range(2):
                cursor.execute("""
                    INSERT INTO agent_runs (
                        run_type, status, started_at, completed_at,
                        decisions_made, orders_placed, total_realized_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    "paper_conservative",
                    "completed",
                    f"{date}T{10 + run_num * 4}:00:00",
                    f"{date}T{10 + run_num * 4}:05:00",
                    10,
                    3,
                    50.0 + day * 20 + run_num * 10,
                ))
                run_id = cursor.lastrowid
                
                # Add decisions
                for i in range(10):
                    cursor.execute("""
                        INSERT INTO decision_records (
                            decision_id, run_id, fixture_id, market_ticker,
                            outcome, timestamp, action, edge_bucket
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"d_{date}_{run_num}_{i}",
                        run_id,
                        100 + i,
                        f"TICKER-{i}",
                        "home_win",
                        f"{date}T{10 + run_num * 4}:0{i}:00",
                        "buy" if i % 3 == 0 else "skip",
                        "<5%" if i < 3 else "5-10%" if i < 7 else "10-15%",
                    ))
                
                # Add P&L snapshots
                for i in range(5):
                    cursor.execute("""
                        INSERT INTO pnl_snapshots (
                            snapshot_id, run_id, timestamp, bankroll
                        ) VALUES (?, ?, ?, ?)
                    """, (
                        f"snap_{date}_{run_num}_{i}",
                        run_id,
                        f"{date}T{10 + run_num * 4}:0{i}:00",
                        10000 + day * 100 + run_num * 20 + i * 5,
                    ))
        
        db_connection.commit()
        
        # Build all reports
        builder.build_all_reports(days_back=7)
        
        # Verify outputs
        assert (config.reports_dir / "index.md").exists()
        assert (config.reports_dir / "index.html").exists()
        
        # Check daily reports
        daily_files = list(config.daily_dir.glob("*.md"))
        assert len(daily_files) >= 1
        
        # Check weekly reports
        weekly_files = list(config.weekly_dir.glob("*.md"))
        assert len(weekly_files) >= 1
        
        # Note: build_all_reports builds daily/weekly/index reports,
        # NOT individual run reports with artifacts. 
        # Artifacts are created by build_run_report() which is tested separately.
