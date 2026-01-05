"""Tests for the reporting queries module."""

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from footbe_trader.reporting.queries import (
    DecisionSummary,
    DayStats,
    EdgeBucketStats,
    PnLSnapshot,
    ReportingQueries,
    RunSummary,
    WeekStats,
)


@pytest.fixture
def db_connection(tmp_path):
    """Create an in-memory database with test schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    # Create agent_runs table
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
    
    # Create decision_records table
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
    
    # Create pnl_snapshots table
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
def queries(db_connection):
    """Create ReportingQueries instance."""
    return ReportingQueries(db_connection)


@pytest.fixture
def sample_run(db_connection):
    """Insert a sample run."""
    cursor = db_connection.cursor()
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
        5,
        15,
        10,
        3,
        2,
        25.50,
        10.00,
    ))
    db_connection.commit()
    return cursor.lastrowid


@pytest.fixture
def sample_decisions(db_connection, sample_run):
    """Insert sample decisions."""
    cursor = db_connection.cursor()
    
    decisions = [
        ("d1", sample_run, 123, "TICKER-A", "home_win", "2024-01-15T10:01:00", "buy", "<5%"),
        ("d2", sample_run, 123, "TICKER-A", "draw", "2024-01-15T10:02:00", "skip", "5-10%"),
        ("d3", sample_run, 124, "TICKER-B", "home_win", "2024-01-15T10:03:00", "buy", "10-15%"),
        ("d4", sample_run, 124, "TICKER-B", "away_win", "2024-01-15T10:04:00", "skip", "<5%"),
    ]
    
    for d in decisions:
        cursor.execute("""
            INSERT INTO decision_records (
                decision_id, run_id, fixture_id, market_ticker, outcome,
                timestamp, action, edge_bucket
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, d)
    
    db_connection.commit()
    return [d[0] for d in decisions]


class TestRunSummary:
    """Tests for RunSummary dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        run = RunSummary(
            run_id=1,
            run_type="paper_conservative",
            status="completed",
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            completed_at=datetime(2024, 1, 15, 10, 5, 0),
            decisions_made=10,
            total_pnl=35.50,
        )
        
        data = run.to_dict()
        
        assert data["run_id"] == 1
        assert data["run_type"] == "paper_conservative"
        assert data["total_pnl"] == 35.50
        assert "started_at" in data


class TestDecisionSummary:
    """Tests for DecisionSummary dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        decision = DecisionSummary(
            decision_id="d1",
            run_id=1,
            fixture_id=123,
            market_ticker="TICKER-A",
            outcome="home_win",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            action="buy",
            edge=0.08,
            edge_bucket="5-10%",
        )
        
        data = decision.to_dict()
        
        assert data["decision_id"] == "d1"
        assert data["action"] == "buy"
        assert data["edge"] == 0.08


class TestReportingQueries:
    """Tests for ReportingQueries class."""
    
    def test_get_run(self, queries, sample_run):
        """Test fetching a specific run."""
        run = queries.get_run(sample_run)
        
        assert run is not None
        assert run.run_id == sample_run
        assert run.run_type == "paper_conservative"
        assert run.status == "completed"
        assert run.decisions_made == 10
        assert run.total_pnl == 35.50  # realized + unrealized
    
    def test_get_run_not_found(self, queries):
        """Test fetching non-existent run."""
        run = queries.get_run(999)
        assert run is None
    
    def test_get_latest_run(self, queries, sample_run):
        """Test getting the most recent run."""
        run = queries.get_latest_run()
        
        assert run is not None
        assert run.run_id == sample_run
    
    def test_get_latest_run_with_type_filter(self, queries, db_connection, sample_run):
        """Test getting latest run filtered by type."""
        # Add another run of different type
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO agent_runs (run_type, status, started_at)
            VALUES (?, ?, ?)
        """, ("live_small", "completed", "2024-01-15T11:00:00"))
        db_connection.commit()
        
        # Get latest paper_conservative
        run = queries.get_latest_run(run_type="paper_conservative")
        assert run is not None
        assert run.run_type == "paper_conservative"
    
    def test_get_runs_in_range(self, queries, sample_run):
        """Test fetching runs in a date range."""
        start = datetime(2024, 1, 15, 0, 0, 0)
        end = datetime(2024, 1, 16, 0, 0, 0)
        
        runs = queries.get_runs_in_range(start, end)
        
        assert len(runs) == 1
        assert runs[0].run_id == sample_run
    
    def test_get_runs_in_range_empty(self, queries, sample_run):
        """Test fetching runs when none exist in range."""
        start = datetime(2024, 2, 1, 0, 0, 0)
        end = datetime(2024, 2, 2, 0, 0, 0)
        
        runs = queries.get_runs_in_range(start, end)
        
        assert len(runs) == 0
    
    def test_get_decisions_for_run(self, queries, sample_run, sample_decisions):
        """Test fetching decisions for a run."""
        decisions = queries.get_decisions_for_run(sample_run)
        
        assert len(decisions) == 4
        assert all(d.run_id == sample_run for d in decisions)
    
    def test_get_decisions_in_range(self, queries, sample_run, sample_decisions):
        """Test fetching decisions in a date range."""
        start = datetime(2024, 1, 15, 10, 0, 0)
        end = datetime(2024, 1, 15, 10, 5, 0)
        
        decisions = queries.get_decisions_in_range(start, end)
        
        assert len(decisions) == 4
    
    def test_get_decisions_with_action_filter(self, queries, sample_run, sample_decisions):
        """Test fetching decisions filtered by action."""
        start = datetime(2024, 1, 15, 0, 0, 0)
        end = datetime(2024, 1, 16, 0, 0, 0)
        
        decisions = queries.get_decisions_in_range(start, end, action_filter="buy")
        
        assert len(decisions) == 2
        assert all(d.action == "buy" for d in decisions)
    
    def test_get_action_counts_for_run(self, queries, sample_run, sample_decisions):
        """Test counting actions for a run."""
        counts = queries.get_action_counts_for_run(sample_run)
        
        assert counts["buy"] == 2
        assert counts["skip"] == 2
    
    def test_get_rejection_reasons_for_run(self, queries, db_connection, sample_run):
        """Test counting rejection reasons for a run."""
        # Add decisions with rejection reasons
        cursor = db_connection.cursor()
        cursor.execute("""
            INSERT INTO decision_records (
                decision_id, run_id, fixture_id, market_ticker, outcome,
                timestamp, action, rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("d_rej1", sample_run, 125, "TICKER-C", "home_win",
              "2024-01-15T10:05:00", "skip", "edge_too_low"))
        cursor.execute("""
            INSERT INTO decision_records (
                decision_id, run_id, fixture_id, market_ticker, outcome,
                timestamp, action, rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("d_rej2", sample_run, 126, "TICKER-D", "home_win",
              "2024-01-15T10:06:00", "skip", "edge_too_low"))
        db_connection.commit()
        
        reasons = queries.get_rejection_reasons_for_run(sample_run)
        
        assert "edge_too_low" in reasons
        assert reasons["edge_too_low"] == 2
    
    def test_get_edge_bucket_stats_for_run(self, queries, sample_run, sample_decisions):
        """Test getting edge bucket statistics for a run."""
        stats = queries.get_edge_bucket_stats_for_run(sample_run)
        
        assert len(stats) == 3  # <5%, 5-10%, 10-15%
        
        # Find the <5% bucket
        bucket_lt5 = next(s for s in stats if s.bucket == "<5%")
        assert bucket_lt5.count == 2
        assert bucket_lt5.trades_count == 1  # One buy
        assert bucket_lt5.skips_count == 1  # One skip
    
    def test_get_pnl_snapshots_for_run(self, queries, db_connection, sample_run):
        """Test fetching P&L snapshots for a run."""
        # Add some snapshots
        cursor = db_connection.cursor()
        for i in range(3):
            cursor.execute("""
                INSERT INTO pnl_snapshots (
                    snapshot_id, run_id, timestamp, bankroll, total_pnl
                ) VALUES (?, ?, ?, ?, ?)
            """, (f"snap_{i}", sample_run, f"2024-01-15T10:0{i}:00",
                  10000 + i * 10, i * 10))
        db_connection.commit()
        
        snapshots = queries.get_pnl_snapshots_for_run(sample_run)
        
        assert len(snapshots) == 3
        assert snapshots[0].bankroll == 10000
        assert snapshots[2].bankroll == 10020
    
    def test_get_equity_curve(self, queries, db_connection, sample_run):
        """Test getting equity curve data."""
        # Add some snapshots
        cursor = db_connection.cursor()
        for i in range(5):
            cursor.execute("""
                INSERT INTO pnl_snapshots (
                    snapshot_id, run_id, timestamp, bankroll
                ) VALUES (?, ?, ?, ?)
            """, (f"eq_{i}", sample_run, f"2024-01-15T10:0{i}:00", 10000 + i * 25))
        db_connection.commit()
        
        start = datetime(2024, 1, 15, 0, 0, 0)
        end = datetime(2024, 1, 16, 0, 0, 0)
        
        curve = queries.get_equity_curve(start, end)
        
        assert len(curve) == 5
        assert curve[0][1] == 10000  # First bankroll
        assert curve[4][1] == 10100  # Last bankroll
    
    def test_get_day_stats(self, queries, db_connection, sample_run, sample_decisions):
        """Test getting daily statistics."""
        start = datetime(2024, 1, 14, 0, 0, 0)
        end = datetime(2024, 1, 17, 0, 0, 0)
        
        stats = queries.get_day_stats(start, end)
        
        # Should have stats for 2024-01-15
        assert len(stats) >= 1
        day_stat = next((s for s in stats if s.date == "2024-01-15"), None)
        assert day_stat is not None
        assert day_stat.decisions_count == 4
        assert day_stat.trades_count == 2
        assert day_stat.skips_count == 2


class TestEdgeBucketStats:
    """Tests for EdgeBucketStats dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = EdgeBucketStats(
            bucket="5-10%",
            count=10,
            trades_count=7,
            skips_count=3,
            total_pnl=50.0,
            avg_edge=0.075,
            win_rate=0.6,
        )
        
        data = stats.to_dict()
        
        assert data["bucket"] == "5-10%"
        assert data["count"] == 10
        assert data["win_rate"] == 0.6


class TestDayStats:
    """Tests for DayStats dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
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
        
        data = stats.to_dict()
        
        assert data["date"] == "2024-01-15"
        assert data["trades_count"] == 20
        assert data["max_drawdown"] == 0.02


class TestWeekStats:
    """Tests for WeekStats dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
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
        
        data = stats.to_dict()
        
        assert data["week_start"] == "2024-01-15"
        assert data["return_pct"] == 0.05
        assert data["pace_status"] == "behind_pace"
