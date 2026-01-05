"""Report building orchestration.

Coordinates queries, chart generation, and rendering to produce
complete Run, Daily, and Weekly reports.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.reporting.charts import ChartGenerator, ChartResult
from footbe_trader.reporting.config import ReportingConfig
from footbe_trader.reporting.queries import (
    DayStats,
    DecisionSummary,
    EdgeBucketStats,
    PnLSnapshot,
    ReportingQueries,
    RunSummary,
    WeekStats,
)
from footbe_trader.reporting.render import HealthCheck, HealthCheckItem, ReportRenderer

logger = get_logger(__name__)


@dataclass
class RunReport:
    """A generated run report."""
    
    run_id: int
    run: RunSummary
    decisions: list[DecisionSummary]
    charts: dict[str, ChartResult]
    markdown_path: Path | None = None
    html_path: Path | None = None
    artifact_path: Path | None = None
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for artifact."""
        return {
            "run_id": self.run_id,
            "run": self.run.to_dict(),
            "decisions": [d.to_dict() for d in self.decisions],
            "markdown_path": str(self.markdown_path) if self.markdown_path else None,
            "html_path": str(self.html_path) if self.html_path else None,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class DailyReport:
    """A generated daily report."""
    
    date: str
    stats: DayStats
    runs: list[RunSummary]
    charts: dict[str, ChartResult]
    markdown_path: Path | None = None
    html_path: Path | None = None
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class WeeklyReport:
    """A generated weekly report."""
    
    week_start: str
    week_end: str
    stats: WeekStats
    daily_stats: list[DayStats]
    edge_bucket_stats: list[EdgeBucketStats]
    charts: dict[str, ChartResult]
    markdown_path: Path | None = None
    html_path: Path | None = None
    generated_at: datetime = field(default_factory=datetime.now)


class ReportBuilder:
    """Orchestrates report generation.
    
    Usage:
        builder = ReportBuilder(db_connection)
        
        # Generate a run report
        report = builder.build_run_report(run_id=123)
        
        # Generate today's daily report
        report = builder.build_daily_report()
        
        # Generate this week's report
        report = builder.build_weekly_report()
        
        # Rebuild the navigation index
        builder.build_index()
    """
    
    def __init__(
        self,
        connection: sqlite3.Connection,
        config: ReportingConfig | None = None,
    ):
        """Initialize the report builder.
        
        Args:
            connection: SQLite database connection.
            config: Reporting configuration.
        """
        self.connection = connection
        self.config = config or ReportingConfig()
        self.queries = ReportingQueries(connection)
        self.charts = ChartGenerator(self.config)
        self.renderer = ReportRenderer(self.config)
        
        # Ensure directories exist
        self.config.ensure_directories()
    
    def build_run_report(
        self,
        run_id: int,
        save_artifact: bool = True,
    ) -> RunReport | None:
        """Build a report for a specific run.
        
        Args:
            run_id: The agent run ID.
            save_artifact: Whether to save JSON artifact.
            
        Returns:
            RunReport or None if run not found.
        """
        logger.info("building_run_report", run_id=run_id)
        
        # Fetch data
        run = self.queries.get_run(run_id)
        if not run:
            logger.warning("run_not_found", run_id=run_id)
            return None
        
        decisions = self.queries.get_decisions_for_run(run_id)
        action_counts = self.queries.get_action_counts_for_run(run_id)
        rejection_reasons = self.queries.get_rejection_reasons_for_run(run_id)
        edge_bucket_stats = self.queries.get_edge_bucket_stats_for_run(run_id)
        
        # Generate charts
        charts: dict[str, ChartResult] = {}
        
        # Decision pie chart
        if action_counts:
            charts["decision_pie"] = self.charts.decision_pie(
                actions=action_counts,
                title=f"Decisions for Run {run_id}",
                chart_id=f"run_{run_id}_decisions",
            )
        
        # Edge bucket chart
        if edge_bucket_stats:
            charts["edge_buckets"] = self.charts.edge_bucket_bar(
                buckets=[eb.bucket for eb in edge_bucket_stats],
                counts=[eb.count for eb in edge_bucket_stats],
                title=f"Edge Distribution for Run {run_id}",
                chart_id=f"run_{run_id}_edge",
            )
        
        # Rejection reasons chart
        if rejection_reasons:
            charts["rejection_reasons"] = self.charts.rejection_reasons_bar(
                reasons=rejection_reasons,
                title=f"Skip Reasons for Run {run_id}",
                chart_id=f"run_{run_id}_rejections",
            )
        
        # Render report
        md_path, html_path = self.renderer.render_run_report(
            run=run,
            decisions=decisions,
            charts=charts,
            output_dir=self.config.runs_dir,
        )
        
        # Create report object
        report = RunReport(
            run_id=run_id,
            run=run,
            decisions=decisions,
            charts=charts,
            markdown_path=md_path,
            html_path=html_path,
        )
        
        # Save artifact
        if save_artifact:
            report.artifact_path = self._save_run_artifact(report)
        
        logger.info(
            "run_report_built",
            run_id=run_id,
            decisions=len(decisions),
            md_path=str(md_path) if md_path else None,
            html_path=str(html_path) if html_path else None,
        )
        
        return report
    
    def build_daily_report(
        self,
        date: datetime | str | None = None,
    ) -> DailyReport:
        """Build a report for a specific day.
        
        Args:
            date: Date to report on (default: today).
            
        Returns:
            DailyReport object.
        """
        # Parse date
        if date is None:
            report_date = datetime.now().date()
        elif isinstance(date, str):
            report_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            report_date = date.date()
        
        date_str = str(report_date)
        logger.info("building_daily_report", date=date_str)
        
        # Date range for the day
        start = datetime.combine(report_date, datetime.min.time())
        end = start + timedelta(days=1)
        
        # Fetch data
        runs = self.queries.get_runs_in_range(start, end)
        decisions = self.queries.get_decisions_in_range(start, end)
        action_counts = self.queries.get_action_counts_in_range(start, end)
        pnl_snapshots = self.queries.get_pnl_snapshots_in_range(start, end)
        equity_curve = self.queries.get_equity_curve(start, end)
        edge_bucket_stats = self.queries.get_edge_bucket_stats_in_range(start, end)
        
        # Calculate stats
        total_pnl = sum(r.total_pnl for r in runs)
        ending_equity = pnl_snapshots[-1].bankroll if pnl_snapshots else 0.0
        
        # Calculate max drawdown
        max_drawdown = 0.0
        if equity_curve:
            peak = equity_curve[0][1]
            for _, equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        stats = DayStats(
            date=date_str,
            run_count=len(runs),
            decisions_count=len(decisions),
            trades_count=sum(1 for d in decisions if d.action != "skip"),
            skips_count=sum(1 for d in decisions if d.action == "skip"),
            total_pnl=total_pnl,
            ending_equity=ending_equity,
            max_drawdown=max_drawdown,
        )
        
        # Generate charts
        charts: dict[str, ChartResult] = {}
        
        # Equity curve
        if equity_curve:
            timestamps, values = zip(*equity_curve)
            charts["equity_curve"] = self.charts.equity_curve(
                timestamps=list(timestamps),
                values=list(values),
                title=f"Equity Curve - {date_str}",
                chart_id=f"daily_{date_str}_equity",
            )
        
        # Decision pie
        if action_counts:
            charts["decision_pie"] = self.charts.decision_pie(
                actions=action_counts,
                title=f"Decisions - {date_str}",
                chart_id=f"daily_{date_str}_decisions",
            )
        
        # Edge buckets
        if edge_bucket_stats:
            charts["edge_buckets"] = self.charts.edge_bucket_bar(
                buckets=[eb.bucket for eb in edge_bucket_stats],
                counts=[eb.count for eb in edge_bucket_stats],
                title=f"Edge Distribution - {date_str}",
                chart_id=f"daily_{date_str}_edge",
            )
        
        # Drawdown
        if equity_curve:
            timestamps, values = zip(*equity_curve)
            # Calculate drawdown series
            peak = values[0]
            drawdowns = []
            for v in values:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                drawdowns.append(dd)
            
            charts["drawdown"] = self.charts.drawdown_chart(
                timestamps=list(timestamps),
                drawdowns=drawdowns,
                title=f"Drawdown - {date_str}",
                chart_id=f"daily_{date_str}_drawdown",
                threshold_lines=[(0.07, "Soft"), (0.15, "Hard")],
            )
        
        # Render report
        md_path, html_path = self.renderer.render_daily_report(
            date=date_str,
            stats=stats,
            runs=runs,
            charts=charts,
            output_dir=self.config.daily_dir,
        )
        
        report = DailyReport(
            date=date_str,
            stats=stats,
            runs=runs,
            charts=charts,
            markdown_path=md_path,
            html_path=html_path,
        )
        
        logger.info(
            "daily_report_built",
            date=date_str,
            runs=len(runs),
            decisions=len(decisions),
        )
        
        return report
    
    def build_weekly_report(
        self,
        week_start: datetime | str | None = None,
    ) -> WeeklyReport:
        """Build a report for a specific week.
        
        Args:
            week_start: Monday of the week (default: current week).
            
        Returns:
            WeeklyReport object.
        """
        # Parse week start
        if week_start is None:
            today = datetime.now()
            monday = today - timedelta(days=today.weekday())
            week_start_date = monday.date()
        elif isinstance(week_start, str):
            week_start_date = datetime.strptime(week_start, "%Y-%m-%d").date()
        else:
            week_start_date = week_start.date()
        
        week_end_date = week_start_date + timedelta(days=6)
        week_start_str = str(week_start_date)
        week_end_str = str(week_end_date)
        
        logger.info("building_weekly_report", week_start=week_start_str, week_end=week_end_str)
        
        # Date range for the week
        start = datetime.combine(week_start_date, datetime.min.time())
        end = datetime.combine(week_end_date, datetime.max.time()) + timedelta(microseconds=1)
        
        # Fetch data
        runs = self.queries.get_runs_in_range(start, end)
        decisions = self.queries.get_decisions_in_range(start, end)
        pnl_snapshots = self.queries.get_pnl_snapshots_in_range(start, end)
        equity_curve = self.queries.get_equity_curve(start, end)
        edge_bucket_stats = self.queries.get_edge_bucket_stats_in_range(start, end)
        daily_stats = self.queries.get_day_stats(start, end)
        
        # Calculate weekly stats
        total_pnl = sum(r.total_pnl for r in runs)
        starting_equity = pnl_snapshots[0].bankroll if pnl_snapshots else 0.0
        ending_equity = pnl_snapshots[-1].bankroll if pnl_snapshots else 0.0
        return_pct = (ending_equity - starting_equity) / starting_equity if starting_equity > 0 else 0
        
        # Calculate max drawdown
        max_drawdown = 0.0
        if equity_curve:
            peak = equity_curve[0][1]
            for _, equity in equity_curve:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # Determine pace status
        target = self.config.target_weekly_return
        tolerance = self.config.target_tolerance
        if return_pct >= target + tolerance:
            pace_status = "ahead_of_pace"
        elif return_pct <= target - tolerance:
            pace_status = "behind_pace"
        else:
            pace_status = "on_pace"
        
        stats = WeekStats(
            week_start=week_start_str,
            week_end=week_end_str,
            run_count=len(runs),
            decisions_count=len(decisions),
            trades_count=sum(1 for d in decisions if d.action != "skip"),
            skips_count=sum(1 for d in decisions if d.action == "skip"),
            total_pnl=total_pnl,
            return_pct=return_pct,
            target_return=target,
            pace_status=pace_status,
            ending_equity=ending_equity,
            max_drawdown=max_drawdown,
        )
        
        # Generate charts
        charts: dict[str, ChartResult] = {}
        
        # Equity curve
        if equity_curve:
            timestamps, values = zip(*equity_curve)
            charts["equity_curve"] = self.charts.equity_curve(
                timestamps=list(timestamps),
                values=list(values),
                title=f"Equity Curve - Week of {week_start_str}",
                chart_id=f"weekly_{week_start_str}_equity",
            )
        
        # Pace vs target
        if daily_stats:
            # Calculate cumulative returns by day
            dates = []
            cumulative_returns = []
            cumulative = 0.0
            for ds in daily_stats:
                dates.append(datetime.strptime(ds.date, "%Y-%m-%d"))
                cumulative += ds.total_pnl / starting_equity if starting_equity > 0 else 0
                cumulative_returns.append(cumulative)
            
            if dates:
                charts["pace_vs_target"] = self.charts.pace_vs_target(
                    dates=dates,
                    actual_returns=cumulative_returns,
                    target_return=target,
                    tolerance=tolerance,
                    title=f"Pace vs Target - Week of {week_start_str}",
                    chart_id=f"weekly_{week_start_str}_pace",
                )
        
        # Edge buckets
        if edge_bucket_stats:
            charts["edge_buckets"] = self.charts.edge_bucket_bar(
                buckets=[eb.bucket for eb in edge_bucket_stats],
                counts=[eb.count for eb in edge_bucket_stats],
                title=f"Edge Distribution - Week of {week_start_str}",
                chart_id=f"weekly_{week_start_str}_edge",
            )
        
        # Drawdown
        if equity_curve:
            timestamps, values = zip(*equity_curve)
            peak = values[0]
            drawdowns = []
            for v in values:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                drawdowns.append(dd)
            
            charts["drawdown"] = self.charts.drawdown_chart(
                timestamps=list(timestamps),
                drawdowns=drawdowns,
                title=f"Drawdown - Week of {week_start_str}",
                chart_id=f"weekly_{week_start_str}_drawdown",
                threshold_lines=[(0.07, "Soft"), (0.15, "Hard")],
            )
        
        # Render report
        md_path, html_path = self.renderer.render_weekly_report(
            week_start=week_start_str,
            week_end=week_end_str,
            stats=stats,
            daily_stats=daily_stats,
            edge_bucket_stats=edge_bucket_stats,
            charts=charts,
            output_dir=self.config.weekly_dir,
        )
        
        report = WeeklyReport(
            week_start=week_start_str,
            week_end=week_end_str,
            stats=stats,
            daily_stats=daily_stats,
            edge_bucket_stats=edge_bucket_stats,
            charts=charts,
            markdown_path=md_path,
            html_path=html_path,
        )
        
        logger.info(
            "weekly_report_built",
            week_start=week_start_str,
            runs=len(runs),
            return_pct=return_pct,
            pace_status=pace_status,
        )
        
        return report
    
    def build_index(self) -> tuple[Path | None, Path | None]:
        """Build the navigation index.
        
        Returns:
            Tuple of (markdown_path, html_path).
        """
        logger.info("building_index")
        
        # Get recent data
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        # Recent runs
        recent_runs = self.queries.get_runs_in_range(week_ago, now)
        recent_runs = sorted(recent_runs, key=lambda r: r.started_at, reverse=True)[:10]
        
        # Latest run
        latest_run = self.queries.get_latest_run()
        
        # Get latest P&L snapshot for current equity
        pnl_snapshots = self.queries.get_pnl_snapshots_in_range(week_ago, now)
        current_equity = pnl_snapshots[-1].bankroll if pnl_snapshots else 0.0
        
        # Calculate weekly return
        week_start = now - timedelta(days=now.weekday())
        week_snapshots = self.queries.get_pnl_snapshots_in_range(
            datetime.combine(week_start.date(), datetime.min.time()),
            now,
        )
        if week_snapshots and len(week_snapshots) >= 2:
            start_equity = week_snapshots[0].bankroll
            weekly_return = (current_equity - start_equity) / start_equity if start_equity > 0 else 0
        else:
            weekly_return = 0.0
        
        # Get recent weekly stats
        weekly_reports = self.queries.get_week_stats(
            month_ago,
            now,
            self.config.target_weekly_return,
            self.config.target_tolerance,
        )
        
        # Get recent daily stats
        daily_reports = self.queries.get_day_stats(week_ago, now)
        
        # Build health check
        health_items = []
        
        # Check if agent ran recently
        if latest_run:
            hours_since = (now - latest_run.started_at).total_seconds() / 3600
            if hours_since < 4:
                health_items.append(HealthCheckItem(
                    name="Agent Activity",
                    is_healthy=True,
                    message=f"Last run {hours_since:.1f} hours ago",
                ))
            else:
                health_items.append(HealthCheckItem(
                    name="Agent Activity",
                    is_healthy=False,
                    message=f"No runs in {hours_since:.1f} hours",
                ))
        else:
            health_items.append(HealthCheckItem(
                name="Agent Activity",
                is_healthy=False,
                message="No runs recorded",
            ))
        
        # Check drawdown
        if pnl_snapshots:
            peak = max(s.bankroll for s in pnl_snapshots)
            current_drawdown = (peak - current_equity) / peak if peak > 0 else 0
            if current_drawdown < 0.07:
                health_items.append(HealthCheckItem(
                    name="Drawdown",
                    is_healthy=True,
                    message=f"Current: {current_drawdown*100:.1f}%",
                ))
            elif current_drawdown < 0.15:
                health_items.append(HealthCheckItem(
                    name="Drawdown",
                    is_healthy=False,
                    message=f"Elevated: {current_drawdown*100:.1f}%",
                ))
            else:
                health_items.append(HealthCheckItem(
                    name="Drawdown",
                    is_healthy=False,
                    message=f"SEVERE: {current_drawdown*100:.1f}%",
                ))
        
        # Check pace
        target = self.config.target_weekly_return
        if weekly_return >= target - self.config.target_tolerance:
            health_items.append(HealthCheckItem(
                name="Weekly Pace",
                is_healthy=True,
                message=f"Return: {weekly_return*100:.1f}% (target: {target*100:.0f}%)",
            ))
        else:
            health_items.append(HealthCheckItem(
                name="Weekly Pace",
                is_healthy=False,
                message=f"Behind: {weekly_return*100:.1f}% (target: {target*100:.0f}%)",
            ))
        
        health = HealthCheck(items=health_items)
        
        # Render index
        md_path, html_path = self.renderer.render_index(
            total_runs=len(recent_runs),
            latest_run=latest_run,
            current_equity=current_equity,
            weekly_return=weekly_return,
            target_weekly=self.config.target_weekly_return,
            weekly_reports=weekly_reports,
            daily_reports=daily_reports,
            recent_runs=recent_runs,
            health=health,
            output_dir=self.config.reports_dir,
        )
        
        logger.info("index_built", md_path=str(md_path), html_path=str(html_path))
        
        return md_path, html_path
    
    def _save_run_artifact(self, report: RunReport) -> Path:
        """Save JSON artifact for a run report.
        
        Args:
            report: The RunReport to save.
            
        Returns:
            Path to the saved artifact.
        """
        artifact_path = self.config.artifacts_dir / f"run_{report.run_id}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        artifact_data = report.to_dict()
        artifact_path.write_text(json.dumps(artifact_data, indent=2, default=str))
        
        logger.info("run_artifact_saved", path=str(artifact_path))
        
        return artifact_path
    
    def build_all_reports(
        self,
        days_back: int = 7,
    ) -> None:
        """Build all reports for recent period.
        
        Args:
            days_back: Number of days to generate reports for.
        """
        logger.info("building_all_reports", days_back=days_back)
        
        now = datetime.now()
        
        # Build daily reports
        for i in range(days_back):
            date = now - timedelta(days=i)
            try:
                self.build_daily_report(date)
            except Exception as e:
                logger.error("daily_report_failed", date=str(date.date()), error=str(e))
        
        # Build weekly report for current week
        try:
            self.build_weekly_report()
        except Exception as e:
            logger.error("weekly_report_failed", error=str(e))
        
        # Build index
        try:
            self.build_index()
        except Exception as e:
            logger.error("index_failed", error=str(e))
        
        logger.info("all_reports_built")
