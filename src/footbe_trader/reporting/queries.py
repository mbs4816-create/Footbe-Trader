"""Database queries for reporting.

Provides structured queries for fetching decisions, runs, P&L snapshots,
and aggregations needed by the report generator.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now

logger = get_logger(__name__)


@dataclass
class DecisionSummary:
    """Summary of a single decision for reporting."""
    
    decision_id: str
    run_id: int
    fixture_id: int | None
    market_ticker: str
    outcome: str
    timestamp: datetime
    action: str
    
    # Context
    league_key: str | None = None
    hours_to_kickoff: float | None = None
    time_category: str | None = None
    
    # Pacing/drawdown state
    pace_state: str | None = None
    drawdown: float = 0.0
    drawdown_band: str | None = None
    gross_exposure: float = 0.0
    equity: float = 0.0
    
    # Edge and sizing
    edge: float | None = None
    edge_bucket: str | None = None
    throttle_multiplier: float = 1.0
    size_intended: int | None = None
    size_executed: int | None = None
    
    # Rationale
    rationale: str | None = None
    rejection_reason: str | None = None
    order_placed: bool = False
    
    # Market snapshot info
    best_bid: float | None = None
    best_ask: float | None = None
    model_prob: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "fixture_id": self.fixture_id,
            "market_ticker": self.market_ticker,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "league_key": self.league_key,
            "hours_to_kickoff": self.hours_to_kickoff,
            "time_category": self.time_category,
            "pace_state": self.pace_state,
            "drawdown": self.drawdown,
            "drawdown_band": self.drawdown_band,
            "gross_exposure": self.gross_exposure,
            "equity": self.equity,
            "edge": self.edge,
            "edge_bucket": self.edge_bucket,
            "throttle_multiplier": self.throttle_multiplier,
            "size_intended": self.size_intended,
            "size_executed": self.size_executed,
            "rationale": self.rationale,
            "rejection_reason": self.rejection_reason,
            "order_placed": self.order_placed,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "model_prob": self.model_prob,
        }


@dataclass
class RunSummary:
    """Summary of an agent run."""
    
    run_id: int
    run_type: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    
    # Counts
    fixtures_evaluated: int = 0
    markets_evaluated: int = 0
    decisions_made: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    
    # P&L
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_exposure: float = 0.0
    position_count: int = 0
    
    # Errors
    error_count: int = 0
    error_message: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "fixtures_evaluated": self.fixtures_evaluated,
            "markets_evaluated": self.markets_evaluated,
            "decisions_made": self.decisions_made,
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_rejected": self.orders_rejected,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_exposure": self.total_exposure,
            "position_count": self.position_count,
            "error_count": self.error_count,
            "error_message": self.error_message,
        }


@dataclass
class PnLSnapshot:
    """Point-in-time P&L record."""
    
    snapshot_id: str
    run_id: int
    timestamp: datetime
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    total_exposure: float = 0.0
    position_count: int = 0
    bankroll: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_exposure": self.total_exposure,
            "position_count": self.position_count,
            "bankroll": self.bankroll,
        }


@dataclass
class EdgeBucketStats:
    """Statistics for an edge bucket."""
    
    bucket: str
    count: int = 0
    trades_count: int = 0  # Actual trades (not skips)
    skips_count: int = 0
    total_pnl: float = 0.0
    avg_edge: float = 0.0
    win_rate: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bucket": self.bucket,
            "count": self.count,
            "trades_count": self.trades_count,
            "skips_count": self.skips_count,
            "total_pnl": self.total_pnl,
            "avg_edge": self.avg_edge,
            "win_rate": self.win_rate,
        }


@dataclass
class DayStats:
    """Statistics for a single day."""
    
    date: str  # YYYY-MM-DD
    run_count: int = 0
    decisions_count: int = 0
    trades_count: int = 0
    skips_count: int = 0
    total_pnl: float = 0.0
    ending_equity: float = 0.0
    max_drawdown: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "run_count": self.run_count,
            "decisions_count": self.decisions_count,
            "trades_count": self.trades_count,
            "skips_count": self.skips_count,
            "total_pnl": self.total_pnl,
            "ending_equity": self.ending_equity,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class WeekStats:
    """Statistics for a single week."""
    
    week_start: str  # YYYY-MM-DD (Monday)
    week_end: str  # YYYY-MM-DD (Sunday)
    run_count: int = 0
    decisions_count: int = 0
    trades_count: int = 0
    skips_count: int = 0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    target_return: float = 0.0
    pace_status: str = "on_pace"  # ahead, on_pace, behind
    ending_equity: float = 0.0
    max_drawdown: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "week_start": self.week_start,
            "week_end": self.week_end,
            "run_count": self.run_count,
            "decisions_count": self.decisions_count,
            "trades_count": self.trades_count,
            "skips_count": self.skips_count,
            "total_pnl": self.total_pnl,
            "return_pct": self.return_pct,
            "target_return": self.target_return,
            "pace_status": self.pace_status,
            "ending_equity": self.ending_equity,
            "max_drawdown": self.max_drawdown,
        }


class ReportingQueries:
    """Database queries for the reporting subsystem."""
    
    def __init__(self, connection: sqlite3.Connection):
        """Initialize with database connection.
        
        Args:
            connection: SQLite database connection.
        """
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
    
    def get_run(self, run_id: int) -> RunSummary | None:
        """Get summary for a specific run.
        
        Args:
            run_id: The agent run ID.
            
        Returns:
            RunSummary or None if not found.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM agent_runs WHERE id = ?
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        
        return self._row_to_run_summary(row)
    
    def get_runs_in_range(
        self,
        start: datetime,
        end: datetime,
        run_type: str | None = None,
    ) -> list[RunSummary]:
        """Get all runs in a date range.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            run_type: Optional filter by run type.
            
        Returns:
            List of RunSummary objects.
        """
        cursor = self.connection.cursor()
        
        if run_type:
            cursor.execute(
                """
                SELECT * FROM agent_runs
                WHERE started_at >= ? AND started_at < ?
                AND run_type = ?
                ORDER BY started_at
                """,
                (start.isoformat(), end.isoformat(), run_type),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM agent_runs
                WHERE started_at >= ? AND started_at < ?
                ORDER BY started_at
                """,
                (start.isoformat(), end.isoformat()),
            )
        
        return [self._row_to_run_summary(row) for row in cursor.fetchall()]
    
    def get_latest_run(self, run_type: str | None = None) -> RunSummary | None:
        """Get the most recent run.
        
        Args:
            run_type: Optional filter by run type.
            
        Returns:
            RunSummary or None if no runs exist.
        """
        cursor = self.connection.cursor()
        
        if run_type:
            cursor.execute(
                """
                SELECT * FROM agent_runs
                WHERE run_type = ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (run_type,),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM agent_runs
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
        
        row = cursor.fetchone()
        return self._row_to_run_summary(row) if row else None
    
    def get_decisions_for_run(self, run_id: int) -> list[DecisionSummary]:
        """Get all decisions for a specific run.
        
        Args:
            run_id: The agent run ID.
            
        Returns:
            List of DecisionSummary objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM decision_records
            WHERE run_id = ?
            ORDER BY timestamp
            """,
            (run_id,),
        )
        
        return [self._row_to_decision_summary(row) for row in cursor.fetchall()]
    
    def get_decisions_in_range(
        self,
        start: datetime,
        end: datetime,
        action_filter: str | None = None,
    ) -> list[DecisionSummary]:
        """Get all decisions in a date range.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            action_filter: Optional filter by action (buy, skip, etc.).
            
        Returns:
            List of DecisionSummary objects.
        """
        cursor = self.connection.cursor()
        
        if action_filter:
            cursor.execute(
                """
                SELECT * FROM decision_records
                WHERE timestamp >= ? AND timestamp < ?
                AND action = ?
                ORDER BY timestamp
                """,
                (start.isoformat(), end.isoformat(), action_filter),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM decision_records
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
                """,
                (start.isoformat(), end.isoformat()),
            )
        
        return [self._row_to_decision_summary(row) for row in cursor.fetchall()]
    
    def get_pnl_snapshots_for_run(self, run_id: int) -> list[PnLSnapshot]:
        """Get P&L snapshots for a specific run.
        
        Args:
            run_id: The agent run ID.
            
        Returns:
            List of PnLSnapshot objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM pnl_snapshots
            WHERE run_id = ?
            ORDER BY timestamp
            """,
            (run_id,),
        )
        
        return [self._row_to_pnl_snapshot(row) for row in cursor.fetchall()]
    
    def get_pnl_snapshots_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[PnLSnapshot]:
        """Get P&L snapshots in a date range.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            
        Returns:
            List of PnLSnapshot objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM pnl_snapshots
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
            """,
            (start.isoformat(), end.isoformat()),
        )
        
        return [self._row_to_pnl_snapshot(row) for row in cursor.fetchall()]
    
    def get_edge_bucket_stats_for_run(self, run_id: int) -> list[EdgeBucketStats]:
        """Get edge bucket statistics for a specific run.
        
        Args:
            run_id: The agent run ID.
            
        Returns:
            List of EdgeBucketStats objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT 
                edge_bucket,
                COUNT(*) as count,
                SUM(CASE WHEN action != 'skip' THEN 1 ELSE 0 END) as trades_count,
                SUM(CASE WHEN action = 'skip' THEN 1 ELSE 0 END) as skips_count
            FROM decision_records
            WHERE run_id = ? AND edge_bucket IS NOT NULL
            GROUP BY edge_bucket
            ORDER BY edge_bucket
            """,
            (run_id,),
        )
        
        results = []
        for row in cursor.fetchall():
            results.append(EdgeBucketStats(
                bucket=row["edge_bucket"] or "unknown",
                count=row["count"],
                trades_count=row["trades_count"],
                skips_count=row["skips_count"],
            ))
        
        return results
    
    def get_edge_bucket_stats_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[EdgeBucketStats]:
        """Get edge bucket statistics for a date range.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            
        Returns:
            List of EdgeBucketStats objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT 
                edge_bucket,
                COUNT(*) as count,
                SUM(CASE WHEN action != 'skip' THEN 1 ELSE 0 END) as trades_count,
                SUM(CASE WHEN action = 'skip' THEN 1 ELSE 0 END) as skips_count
            FROM decision_records
            WHERE timestamp >= ? AND timestamp < ?
            AND edge_bucket IS NOT NULL
            GROUP BY edge_bucket
            ORDER BY edge_bucket
            """,
            (start.isoformat(), end.isoformat()),
        )
        
        results = []
        for row in cursor.fetchall():
            results.append(EdgeBucketStats(
                bucket=row["edge_bucket"] or "unknown",
                count=row["count"],
                trades_count=row["trades_count"],
                skips_count=row["skips_count"],
            ))
        
        return results
    
    def get_action_counts_for_run(self, run_id: int) -> dict[str, int]:
        """Get counts of each action type for a run.
        
        Args:
            run_id: The agent run ID.
            
        Returns:
            Dict mapping action name to count.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT action, COUNT(*) as count
            FROM decision_records
            WHERE run_id = ?
            GROUP BY action
            """,
            (run_id,),
        )
        
        return {row["action"]: row["count"] for row in cursor.fetchall()}
    
    def get_action_counts_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, int]:
        """Get counts of each action type in a date range.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            
        Returns:
            Dict mapping action name to count.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT action, COUNT(*) as count
            FROM decision_records
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY action
            """,
            (start.isoformat(), end.isoformat()),
        )
        
        return {row["action"]: row["count"] for row in cursor.fetchall()}
    
    def get_rejection_reasons_for_run(self, run_id: int) -> dict[str, int]:
        """Get counts of each rejection reason for a run.
        
        Args:
            run_id: The agent run ID.
            
        Returns:
            Dict mapping rejection reason to count.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT rejection_reason, COUNT(*) as count
            FROM decision_records
            WHERE run_id = ? AND rejection_reason IS NOT NULL
            GROUP BY rejection_reason
            ORDER BY count DESC
            """,
            (run_id,),
        )
        
        return {row["rejection_reason"]: row["count"] for row in cursor.fetchall()}
    
    def get_day_stats(
        self,
        start: datetime,
        end: datetime,
    ) -> list[DayStats]:
        """Get daily statistics for a date range.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            
        Returns:
            List of DayStats objects for each day.
        """
        cursor = self.connection.cursor()
        
        # Get decision stats by day
        cursor.execute(
            """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as decisions_count,
                SUM(CASE WHEN action != 'skip' THEN 1 ELSE 0 END) as trades_count,
                SUM(CASE WHEN action = 'skip' THEN 1 ELSE 0 END) as skips_count
            FROM decision_records
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY DATE(timestamp)
            ORDER BY date
            """,
            (start.isoformat(), end.isoformat()),
        )
        
        decision_rows = {row["date"]: row for row in cursor.fetchall()}
        
        # Get run counts by day
        cursor.execute(
            """
            SELECT 
                DATE(started_at) as date,
                COUNT(*) as run_count
            FROM agent_runs
            WHERE started_at >= ? AND started_at < ?
            GROUP BY DATE(started_at)
            """,
            (start.isoformat(), end.isoformat()),
        )
        
        run_rows = {row["date"]: row for row in cursor.fetchall()}
        
        # Combine into DayStats
        all_dates = set(decision_rows.keys()) | set(run_rows.keys())
        results = []
        
        for date in sorted(all_dates):
            decision_row = decision_rows.get(date)
            run_row = run_rows.get(date)
            
            results.append(DayStats(
                date=date,
                run_count=run_row["run_count"] if run_row else 0,
                decisions_count=decision_row["decisions_count"] if decision_row else 0,
                trades_count=decision_row["trades_count"] if decision_row else 0,
                skips_count=decision_row["skips_count"] if decision_row else 0,
            ))
        
        return results
    
    def get_week_stats(
        self,
        start: datetime,
        end: datetime,
        target_weekly_return: float = 0.10,
        target_tolerance: float = 0.02,
    ) -> list[WeekStats]:
        """Get weekly statistics for a date range.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            target_weekly_return: Target weekly return percentage.
            target_tolerance: Tolerance for "on pace" status.
            
        Returns:
            List of WeekStats objects for each week.
        """
        # Calculate weeks in range
        weeks = []
        current = start
        while current < end:
            # Get Monday of current week
            monday = current - timedelta(days=current.weekday())
            sunday = monday + timedelta(days=6)
            
            if monday.date() not in [w["monday"] for w in weeks]:
                weeks.append({
                    "monday": monday.date(),
                    "sunday": sunday.date(),
                })
            
            current += timedelta(days=7)
        
        results = []
        for week in weeks:
            week_start = datetime.combine(week["monday"], datetime.min.time())
            week_end = datetime.combine(
                week["sunday"], datetime.max.time()
            ) + timedelta(microseconds=1)
            
            # Get decision stats for week
            cursor = self.connection.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as decisions_count,
                    SUM(CASE WHEN action != 'skip' THEN 1 ELSE 0 END) as trades_count,
                    SUM(CASE WHEN action = 'skip' THEN 1 ELSE 0 END) as skips_count
                FROM decision_records
                WHERE timestamp >= ? AND timestamp < ?
                """,
                (week_start.isoformat(), week_end.isoformat()),
            )
            decision_row = cursor.fetchone()
            
            # Get run count for week
            cursor.execute(
                """
                SELECT COUNT(*) as run_count
                FROM agent_runs
                WHERE started_at >= ? AND started_at < ?
                """,
                (week_start.isoformat(), week_end.isoformat()),
            )
            run_row = cursor.fetchone()
            
            # Calculate pace status
            # This is simplified - real implementation would check actual returns
            pace_status = "on_pace"
            
            results.append(WeekStats(
                week_start=str(week["monday"]),
                week_end=str(week["sunday"]),
                run_count=run_row["run_count"] if run_row else 0,
                decisions_count=decision_row["decisions_count"] if decision_row else 0,
                trades_count=decision_row["trades_count"] if decision_row else 0,
                skips_count=decision_row["skips_count"] if decision_row else 0,
                target_return=target_weekly_return,
                pace_status=pace_status,
            ))
        
        return results
    
    def get_equity_curve(
        self,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, float]]:
        """Get equity curve data points for charting.
        
        Args:
            start: Start datetime (inclusive).
            end: End datetime (exclusive).
            
        Returns:
            List of (timestamp, bankroll) tuples.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT timestamp, bankroll
            FROM pnl_snapshots
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
            """,
            (start.isoformat(), end.isoformat()),
        )
        
        return [
            (datetime.fromisoformat(row["timestamp"]), row["bankroll"])
            for row in cursor.fetchall()
        ]
    
    def _row_to_run_summary(self, row: sqlite3.Row) -> RunSummary:
        """Convert database row to RunSummary."""
        return RunSummary(
            run_id=row["id"],
            run_type=row["run_type"],
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"] else None
            ),
            fixtures_evaluated=row["fixtures_evaluated"] or 0,
            markets_evaluated=row["markets_evaluated"] or 0,
            decisions_made=row["decisions_made"] or 0,
            orders_placed=row["orders_placed"] or 0,
            orders_filled=row["orders_filled"] or 0,
            orders_rejected=row["orders_rejected"] or 0,
            total_realized_pnl=row["total_realized_pnl"] or 0.0,
            total_unrealized_pnl=row["total_unrealized_pnl"] or 0.0,
            total_pnl=(row["total_realized_pnl"] or 0.0) + (row["total_unrealized_pnl"] or 0.0),
            total_exposure=row["total_exposure"] or 0.0,
            position_count=row["position_count"] or 0,
            error_count=row["error_count"] or 0,
            error_message=row["error_message"],
        )
    
    def _row_to_decision_summary(self, row: sqlite3.Row) -> DecisionSummary:
        """Convert database row to DecisionSummary."""
        # Parse edge from edge_calculation_json if available
        edge = None
        if row["edge_calculation_json"]:
            try:
                edge_data = json.loads(row["edge_calculation_json"])
                edge = edge_data.get("edge")
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Parse market snapshot for bid/ask/prob
        best_bid = None
        best_ask = None
        if row["market_snapshot_json"]:
            try:
                snapshot_data = json.loads(row["market_snapshot_json"])
                best_bid = snapshot_data.get("best_bid")
                best_ask = snapshot_data.get("best_ask")
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Parse model prediction
        model_prob = None
        if row["model_prediction_json"]:
            try:
                pred_data = json.loads(row["model_prediction_json"])
                model_prob = pred_data.get("probability")
            except (json.JSONDecodeError, TypeError):
                pass
        
        return DecisionSummary(
            decision_id=row["decision_id"],
            run_id=row["run_id"],
            fixture_id=row["fixture_id"],
            market_ticker=row["market_ticker"],
            outcome=row["outcome"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            action=row["action"],
            league_key=row["league_key"] if "league_key" in row.keys() else None,
            hours_to_kickoff=(
                row["hours_to_kickoff"]
                if "hours_to_kickoff" in row.keys()
                else None
            ),
            time_category=(
                row["time_category"]
                if "time_category" in row.keys()
                else None
            ),
            pace_state=(
                row["pace_state"]
                if "pace_state" in row.keys()
                else None
            ),
            drawdown=(
                row["drawdown"]
                if "drawdown" in row.keys()
                else 0.0
            ),
            drawdown_band=(
                row["drawdown_band"]
                if "drawdown_band" in row.keys()
                else None
            ),
            gross_exposure=(
                row["gross_exposure"]
                if "gross_exposure" in row.keys()
                else 0.0
            ),
            equity=(
                row["equity"]
                if "equity" in row.keys()
                else 0.0
            ),
            edge=edge,
            edge_bucket=(
                row["edge_bucket"]
                if "edge_bucket" in row.keys()
                else None
            ),
            throttle_multiplier=(
                row["throttle_multiplier"]
                if "throttle_multiplier" in row.keys()
                else 1.0
            ),
            size_intended=(
                row["size_intended"]
                if "size_intended" in row.keys()
                else None
            ),
            size_executed=(
                row["size_executed"]
                if "size_executed" in row.keys()
                else None
            ),
            rationale=row["rationale"],
            rejection_reason=row["rejection_reason"],
            order_placed=bool(row["order_placed"]),
            best_bid=best_bid,
            best_ask=best_ask,
            model_prob=model_prob,
        )
    
    def _row_to_pnl_snapshot(self, row: sqlite3.Row) -> PnLSnapshot:
        """Convert database row to PnLSnapshot."""
        return PnLSnapshot(
            snapshot_id=row["snapshot_id"],
            run_id=row["run_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            total_realized_pnl=row["total_realized_pnl"] or 0.0,
            total_unrealized_pnl=row["total_unrealized_pnl"] or 0.0,
            total_pnl=row["total_pnl"] or 0.0,
            total_exposure=row["total_exposure"] or 0.0,
            position_count=row["position_count"] or 0,
            bankroll=row["bankroll"] or 0.0,
        )
    
    # =========================================================================
    # Bulk export methods for Google Sheets integration
    # =========================================================================
    
    def get_all_runs(self) -> list[RunSummary]:
        """Get all runs from the database.
        
        Returns:
            List of all RunSummary objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM agent_runs
            ORDER BY started_at DESC
            """
        )
        return [self._row_to_run_summary(row) for row in cursor.fetchall()]
    
    def get_all_decisions(self) -> list[DecisionSummary]:
        """Get all decisions from the database.
        
        Returns:
            List of all DecisionSummary objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM decision_records
            ORDER BY timestamp DESC
            """
        )
        return [self._row_to_decision_summary(row) for row in cursor.fetchall()]
    
    def get_all_pnl_snapshots(self) -> list[PnLSnapshot]:
        """Get all P&L snapshots from the database.
        
        Returns:
            List of all PnLSnapshot objects.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM pnl_snapshots
            ORDER BY timestamp DESC
            """
        )
        return [self._row_to_pnl_snapshot(row) for row in cursor.fetchall()]
    
    def get_all_daily_stats(self) -> list[DayStats]:
        """Get aggregated daily statistics.
        
        Returns:
            List of DayStats objects for each day.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT 
                date(started_at) as day,
                COUNT(*) as run_count,
                SUM(decisions_made) as decisions_count,
                SUM(orders_placed) as trades_count,
                SUM(decisions_made - orders_placed) as skips_count,
                SUM(total_realized_pnl + total_unrealized_pnl) as total_pnl
            FROM agent_runs
            GROUP BY date(started_at)
            ORDER BY day DESC
            """
        )
        
        results = []
        for row in cursor.fetchall():
            results.append(DayStats(
                date=row["day"],
                run_count=row["run_count"] or 0,
                decisions_count=row["decisions_count"] or 0,
                trades_count=row["trades_count"] or 0,
                skips_count=row["skips_count"] or 0,
                total_pnl=row["total_pnl"] or 0.0,
            ))
        return results
    
    def get_all_weekly_stats(self) -> list[WeekStats]:
        """Get aggregated weekly statistics.
        
        Returns:
            List of WeekStats objects for each week.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT 
                date(started_at, 'weekday 0', '-6 days') as week_start,
                date(started_at, 'weekday 0') as week_end,
                COUNT(*) as run_count,
                SUM(decisions_made) as decisions_count,
                SUM(orders_placed) as trades_count,
                SUM(decisions_made - orders_placed) as skips_count,
                SUM(total_realized_pnl + total_unrealized_pnl) as total_pnl
            FROM agent_runs
            GROUP BY date(started_at, 'weekday 0', '-6 days')
            ORDER BY week_start DESC
            """
        )
        
        results = []
        for row in cursor.fetchall():
            results.append(WeekStats(
                week_start=row["week_start"],
                week_end=row["week_end"],
                run_count=row["run_count"] or 0,
                decisions_count=row["decisions_count"] or 0,
                trades_count=row["trades_count"] or 0,
                skips_count=row["skips_count"] or 0,
                total_pnl=row["total_pnl"] or 0.0,
            ))
        return results
    
    def get_all_positions(self) -> list[dict]:
        """Get all current positions.
        
        Returns:
            List of position dicts.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT * FROM paper_positions
            WHERE quantity > 0
            ORDER BY ticker
            """
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "ticker": row["ticker"],
                "quantity": row["quantity"],
                "average_entry_price": row["average_entry_price"],
                "mark_price": row["mark_price"] or 0,
                "unrealized_pnl": row["unrealized_pnl"] or 0,
                "realized_pnl": row["realized_pnl"] or 0,
                "updated_at": row["updated_at"] if "updated_at" in row.keys() else "",
            })
        return results

