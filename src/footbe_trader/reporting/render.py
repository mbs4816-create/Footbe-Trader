"""Report rendering with Jinja2 templates.

Renders reports in Markdown and HTML formats using Jinja2 templating.
Handles chart embedding, table formatting, and navigation generation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from footbe_trader.common.logging import get_logger
from footbe_trader.reporting.charts import ChartResult
from footbe_trader.reporting.config import ReportingConfig

logger = get_logger(__name__)


# Built-in templates (when no external template directory)
RUN_REPORT_MD_TEMPLATE = """# Run Report: {{ run.run_id }}

**Type:** {{ run.run_type }}  
**Status:** {{ run.status }}  
**Started:** {{ run.started_at }}  
**Completed:** {{ run.completed_at or "In Progress" }}

---

## Summary

| Metric | Value |
|--------|-------|
| Fixtures Evaluated | {{ run.fixtures_evaluated }} |
| Markets Evaluated | {{ run.markets_evaluated }} |
| Decisions Made | {{ run.decisions_made }} |
| Orders Placed | {{ run.orders_placed }} |
| Orders Filled | {{ run.orders_filled }} |
| Orders Rejected | {{ run.orders_rejected }} |

## P&L

| Metric | Value |
|--------|-------|
| Realized P&L | ${{ "%.2f"|format(run.total_realized_pnl) }} |
| Unrealized P&L | ${{ "%.2f"|format(run.total_unrealized_pnl) }} |
| Total P&L | ${{ "%.2f"|format(run.total_pnl) }} |
| Total Exposure | ${{ "%.2f"|format(run.total_exposure) }} |
| Position Count | {{ run.position_count }} |

{% if charts.decision_pie %}
## Decision Distribution

{{ charts.decision_pie.get_markdown_img() }}
{% endif %}

{% if charts.edge_buckets %}
## Edge Distribution

{{ charts.edge_buckets.get_markdown_img() }}
{% endif %}

{% if charts.rejection_reasons %}
## Skip Reasons

{{ charts.rejection_reasons.get_markdown_img() }}
{% endif %}

## Decisions

{% if decisions %}
| Time | Market | Action | Edge | Size | Rationale |
|------|--------|--------|------|------|-----------|
{% for d in decisions %}
| {{ d.timestamp.strftime('%H:%M:%S') }} | {{ d.market_ticker }} | {{ d.action }} | {{ "%.1f%%"|format((d.edge or 0) * 100) }} | {{ d.size_executed or "-" }} | {{ (d.rationale or d.rejection_reason or "-")[:50] }} |
{% endfor %}
{% else %}
*No decisions recorded.*
{% endif %}

{% if run.error_message %}
## Errors

```
{{ run.error_message }}
```
{% endif %}

---

*Generated: {{ generated_at }}*
"""

DAILY_REPORT_MD_TEMPLATE = """# Daily Report: {{ date }}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Runs | {{ stats.run_count }} |
| Total Decisions | {{ stats.decisions_count }} |
| Trades Executed | {{ stats.trades_count }} |
| Trades Skipped | {{ stats.skips_count }} |
| Total P&L | ${{ "%.2f"|format(stats.total_pnl) }} |
| Ending Equity | ${{ "%.2f"|format(stats.ending_equity) }} |
| Max Drawdown | {{ "%.1f%%"|format(stats.max_drawdown * 100) }} |

{% if charts.equity_curve %}
## Equity Curve

{{ charts.equity_curve.get_markdown_img() }}
{% endif %}

{% if charts.decision_pie %}
## Decision Distribution

{{ charts.decision_pie.get_markdown_img() }}
{% endif %}

{% if charts.edge_buckets %}
## Edge Bucket Performance

{{ charts.edge_buckets.get_markdown_img() }}
{% endif %}

{% if charts.drawdown %}
## Drawdown

{{ charts.drawdown.get_markdown_img() }}
{% endif %}

## Runs

{% if runs %}
| Run ID | Type | Started | Status | Decisions | P&L |
|--------|------|---------|--------|-----------|-----|
{% for r in runs %}
| [{{ r.run_id }}](../runs/{{ r.run_id }}.md) | {{ r.run_type }} | {{ r.started_at.strftime('%H:%M') }} | {{ r.status }} | {{ r.decisions_made }} | ${{ "%.2f"|format(r.total_pnl) }} |
{% endfor %}
{% else %}
*No runs recorded.*
{% endif %}

---

*Generated: {{ generated_at }}*
"""

WEEKLY_REPORT_MD_TEMPLATE = """# Weekly Report: {{ week_start }} to {{ week_end }}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Runs | {{ stats.run_count }} |
| Total Decisions | {{ stats.decisions_count }} |
| Trades Executed | {{ stats.trades_count }} |
| Trades Skipped | {{ stats.skips_count }} |
| Weekly Return | {{ "%.2f%%"|format(stats.return_pct * 100) }} |
| Target Return | {{ "%.2f%%"|format(stats.target_return * 100) }} |
| Pace Status | **{{ stats.pace_status | upper }}** |
| Ending Equity | ${{ "%.2f"|format(stats.ending_equity) }} |
| Max Drawdown | {{ "%.1f%%"|format(stats.max_drawdown * 100) }} |

{% if charts.pace_vs_target %}
## Pace vs Target

{{ charts.pace_vs_target.get_markdown_img() }}
{% endif %}

{% if charts.equity_curve %}
## Equity Curve

{{ charts.equity_curve.get_markdown_img() }}
{% endif %}

{% if charts.edge_buckets %}
## Edge Bucket Performance

Are edge estimates translating into actual returns?

{{ charts.edge_buckets.get_markdown_img() }}

{% if edge_bucket_stats %}
| Bucket | Decisions | Trades | Skips | Avg Edge | Win Rate | P&L |
|--------|-----------|--------|-------|----------|----------|-----|
{% for eb in edge_bucket_stats %}
| {{ eb.bucket }} | {{ eb.count }} | {{ eb.trades_count }} | {{ eb.skips_count }} | {{ "%.1f%%"|format(eb.avg_edge * 100) }} | {{ "%.0f%%"|format(eb.win_rate * 100) }} | ${{ "%.2f"|format(eb.total_pnl) }} |
{% endfor %}
{% endif %}
{% endif %}

{% if charts.drawdown %}
## Drawdown

{{ charts.drawdown.get_markdown_img() }}
{% endif %}

## Daily Breakdown

{% if daily_stats %}
| Date | Runs | Decisions | Trades | Skips | P&L |
|------|------|-----------|--------|-------|-----|
{% for d in daily_stats %}
| [{{ d.date }}](../daily/{{ d.date }}.md) | {{ d.run_count }} | {{ d.decisions_count }} | {{ d.trades_count }} | {{ d.skips_count }} | ${{ "%.2f"|format(d.total_pnl) }} |
{% endfor %}
{% else %}
*No daily data available.*
{% endif %}

## Recommendations

{% if stats.pace_status == "behind_pace" %}
⚠️ **Behind Pace**: Consider reviewing edge thresholds or increasing position sizes within risk limits.
{% elif stats.pace_status == "ahead_of_pace" %}
✅ **Ahead of Pace**: Maintain current strategy. Consider banking profits by tightening risk controls.
{% else %}
✅ **On Pace**: Strategy is performing as expected.
{% endif %}

{% if stats.max_drawdown > 0.10 %}
⚠️ **Elevated Drawdown**: Max drawdown of {{ "%.1f%%"|format(stats.max_drawdown * 100) }} approaches soft threshold. Monitor closely.
{% endif %}

---

*Generated: {{ generated_at }}*
"""

INDEX_MD_TEMPLATE = """# Trading Agent Reports

Welcome to the trading agent reporting dashboard.

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total Runs | {{ total_runs }} |
| Latest Run | {{ latest_run.run_id if latest_run else "N/A" }} |
| Latest Status | {{ latest_run.status if latest_run else "N/A" }} |
| Current Equity | ${{ "%.2f"|format(current_equity) }} |
| Weekly Return | {{ "%.2f%%"|format(weekly_return * 100) }} |
| Target Weekly | {{ "%.2f%%"|format(target_weekly * 100) }} |

## Recent Weekly Reports

{% if weekly_reports %}
| Week | Return | Pace | Link |
|------|--------|------|------|
{% for w in weekly_reports %}
| {{ w.week_start }} - {{ w.week_end }} | {{ "%.2f%%"|format(w.return_pct * 100) }} | {{ w.pace_status }} | [View](weekly/{{ w.week_start }}.md) |
{% endfor %}
{% else %}
*No weekly reports available.*
{% endif %}

## Recent Daily Reports

{% if daily_reports %}
| Date | Runs | P&L | Link |
|------|------|-----|------|
{% for d in daily_reports %}
| {{ d.date }} | {{ d.run_count }} | ${{ "%.2f"|format(d.total_pnl) }} | [View](daily/{{ d.date }}.md) |
{% endfor %}
{% else %}
*No daily reports available.*
{% endif %}

## Recent Runs

{% if recent_runs %}
| Run ID | Type | Started | Status | P&L | Link |
|--------|------|---------|--------|-----|------|
{% for r in recent_runs %}
| {{ r.run_id }} | {{ r.run_type }} | {{ r.started_at.strftime('%Y-%m-%d %H:%M') }} | {{ r.status }} | ${{ "%.2f"|format(r.total_pnl) }} | [View](runs/{{ r.run_id }}.md) |
{% endfor %}
{% else %}
*No runs available.*
{% endif %}

---

## Health Check

{% if health.is_healthy %}
✅ **System Healthy**
{% else %}
⚠️ **Attention Required**
{% endif %}

{% for item in health.items %}
- {{ item.status_icon }} {{ item.name }}: {{ item.message }}
{% endfor %}

---

*Last updated: {{ updated_at }}*
"""


# HTML wrapper with styling
HTML_WRAPPER = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary: #2196F3;
            --success: #4CAF50;
            --warning: #FF9800;
            --danger: #f44336;
            --bg: #f5f5f5;
            --card-bg: #ffffff;
            --text: #333333;
            --border: #e0e0e0;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background-color: var(--bg);
            margin: 0;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: var(--primary);
            border-bottom: 2px solid var(--primary);
            padding-bottom: 10px;
        }
        
        h2 {
            color: #555;
            margin-top: 30px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        tr:hover {
            background-color: #f8f9fa;
        }
        
        img {
            max-width: 100%;
            height: auto;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        code, pre {
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }
        
        pre {
            padding: 15px;
            overflow-x: auto;
        }
        
        a {
            color: var(--primary);
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .status-success { color: var(--success); }
        .status-warning { color: var(--warning); }
        .status-danger { color: var(--danger); }
        
        .metric-card {
            display: inline-block;
            padding: 15px 25px;
            margin: 5px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary);
        }
        
        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        
        hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 30px 0;
        }
        
        .nav {
            background: #f8f9fa;
            padding: 10px 20px;
            margin: -30px -30px 30px -30px;
            border-radius: 8px 8px 0 0;
            border-bottom: 1px solid var(--border);
        }
        
        .nav a {
            margin-right: 20px;
        }
        
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: #666;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <nav class="nav">
            <a href="../index.html">📊 Dashboard</a>
            <a href="../weekly/">📅 Weekly</a>
            <a href="../daily/">📆 Daily</a>
            <a href="../runs/">🔄 Runs</a>
        </nav>
        {{ content }}
        <div class="footer">
            Generated by Footbe Trader Reporting System
        </div>
    </div>
</body>
</html>
"""


@dataclass
class HealthCheckItem:
    """Single health check item for the dashboard."""
    
    name: str
    is_healthy: bool
    message: str
    
    @property
    def status_icon(self) -> str:
        return "✅" if self.is_healthy else "⚠️"


@dataclass
class HealthCheck:
    """Overall health check status."""
    
    items: list[HealthCheckItem] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return all(item.is_healthy for item in self.items)


class ReportRenderer:
    """Renders reports using Jinja2 templates."""
    
    def __init__(
        self,
        config: ReportingConfig | None = None,
        template_dir: Path | None = None,
    ):
        """Initialize the report renderer.
        
        Args:
            config: Reporting configuration.
            template_dir: Optional directory containing custom templates.
        """
        self.config = config or ReportingConfig()
        
        # Set up Jinja2 environment
        if template_dir and template_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
            )
            self.use_custom_templates = True
        else:
            # Use built-in string templates
            self.env = Environment(
                autoescape=select_autoescape(["html", "xml"]),
            )
            self.use_custom_templates = False
        
        # Add custom filters
        self.env.filters["format_currency"] = self._format_currency
        self.env.filters["format_percent"] = self._format_percent
    
    @staticmethod
    def _format_currency(value: float) -> str:
        """Format value as currency."""
        if value >= 0:
            return f"${value:,.2f}"
        else:
            return f"-${abs(value):,.2f}"
    
    @staticmethod
    def _format_percent(value: float) -> str:
        """Format value as percentage."""
        return f"{value * 100:.2f}%"
    
    def render_run_report(
        self,
        run: Any,
        decisions: list[Any],
        charts: dict[str, ChartResult],
        output_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        """Render a run report.
        
        Args:
            run: RunSummary object.
            decisions: List of DecisionSummary objects.
            charts: Dict of chart_id to ChartResult.
            output_dir: Directory to write reports.
            
        Returns:
            Tuple of (markdown_path, html_path).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        context = {
            "run": run,
            "decisions": decisions,
            "charts": charts,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        md_path = None
        html_path = None
        
        # Render Markdown
        if self.config.generate_markdown:
            if self.use_custom_templates:
                template = self.env.get_template("run_report.md")
            else:
                template = self.env.from_string(RUN_REPORT_MD_TEMPLATE)
            
            md_content = template.render(**context)
            md_path = output_dir / f"{run.run_id}.md"
            md_path.write_text(md_content)
            logger.info("run_report_markdown_generated", path=str(md_path))
        
        # Render HTML
        if self.config.generate_html:
            # First render markdown content, then wrap in HTML
            if self.use_custom_templates:
                template = self.env.get_template("run_report.md")
            else:
                template = self.env.from_string(RUN_REPORT_MD_TEMPLATE)
            
            md_content = template.render(**context)
            
            # Convert markdown tables and formatting to HTML
            html_content = self._markdown_to_html(md_content, charts)
            
            # Wrap in HTML template
            wrapper = self.env.from_string(HTML_WRAPPER)
            full_html = wrapper.render(
                title=f"Run Report: {run.run_id}",
                content=html_content,
            )
            
            html_path = output_dir / f"{run.run_id}.html"
            html_path.write_text(full_html)
            logger.info("run_report_html_generated", path=str(html_path))
        
        return md_path, html_path
    
    def render_daily_report(
        self,
        date: str,
        stats: Any,
        runs: list[Any],
        charts: dict[str, ChartResult],
        output_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        """Render a daily report.
        
        Args:
            date: Date string (YYYY-MM-DD).
            stats: DayStats object.
            runs: List of RunSummary objects for the day.
            charts: Dict of chart_id to ChartResult.
            output_dir: Directory to write reports.
            
        Returns:
            Tuple of (markdown_path, html_path).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        context = {
            "date": date,
            "stats": stats,
            "runs": runs,
            "charts": charts,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        md_path = None
        html_path = None
        
        # Render Markdown
        if self.config.generate_markdown:
            if self.use_custom_templates:
                template = self.env.get_template("daily_report.md")
            else:
                template = self.env.from_string(DAILY_REPORT_MD_TEMPLATE)
            
            md_content = template.render(**context)
            md_path = output_dir / f"{date}.md"
            md_path.write_text(md_content)
            logger.info("daily_report_markdown_generated", path=str(md_path))
        
        # Render HTML
        if self.config.generate_html:
            if self.use_custom_templates:
                template = self.env.get_template("daily_report.md")
            else:
                template = self.env.from_string(DAILY_REPORT_MD_TEMPLATE)
            
            md_content = template.render(**context)
            html_content = self._markdown_to_html(md_content, charts)
            
            wrapper = self.env.from_string(HTML_WRAPPER)
            full_html = wrapper.render(
                title=f"Daily Report: {date}",
                content=html_content,
            )
            
            html_path = output_dir / f"{date}.html"
            html_path.write_text(full_html)
            logger.info("daily_report_html_generated", path=str(html_path))
        
        return md_path, html_path
    
    def render_weekly_report(
        self,
        week_start: str,
        week_end: str,
        stats: Any,
        daily_stats: list[Any],
        edge_bucket_stats: list[Any],
        charts: dict[str, ChartResult],
        output_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        """Render a weekly report.
        
        Args:
            week_start: Week start date (YYYY-MM-DD).
            week_end: Week end date (YYYY-MM-DD).
            stats: WeekStats object.
            daily_stats: List of DayStats for each day.
            edge_bucket_stats: List of EdgeBucketStats.
            charts: Dict of chart_id to ChartResult.
            output_dir: Directory to write reports.
            
        Returns:
            Tuple of (markdown_path, html_path).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        context = {
            "week_start": week_start,
            "week_end": week_end,
            "stats": stats,
            "daily_stats": daily_stats,
            "edge_bucket_stats": edge_bucket_stats,
            "charts": charts,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        md_path = None
        html_path = None
        
        # Render Markdown
        if self.config.generate_markdown:
            if self.use_custom_templates:
                template = self.env.get_template("weekly_report.md")
            else:
                template = self.env.from_string(WEEKLY_REPORT_MD_TEMPLATE)
            
            md_content = template.render(**context)
            md_path = output_dir / f"{week_start}.md"
            md_path.write_text(md_content)
            logger.info("weekly_report_markdown_generated", path=str(md_path))
        
        # Render HTML
        if self.config.generate_html:
            if self.use_custom_templates:
                template = self.env.get_template("weekly_report.md")
            else:
                template = self.env.from_string(WEEKLY_REPORT_MD_TEMPLATE)
            
            md_content = template.render(**context)
            html_content = self._markdown_to_html(md_content, charts)
            
            wrapper = self.env.from_string(HTML_WRAPPER)
            full_html = wrapper.render(
                title=f"Weekly Report: {week_start} to {week_end}",
                content=html_content,
            )
            
            html_path = output_dir / f"{week_start}.html"
            html_path.write_text(full_html)
            logger.info("weekly_report_html_generated", path=str(html_path))
        
        return md_path, html_path
    
    def render_index(
        self,
        total_runs: int,
        latest_run: Any | None,
        current_equity: float,
        weekly_return: float,
        target_weekly: float,
        weekly_reports: list[Any],
        daily_reports: list[Any],
        recent_runs: list[Any],
        health: HealthCheck,
        output_dir: Path,
    ) -> tuple[Path | None, Path | None]:
        """Render the navigation index.
        
        Args:
            total_runs: Total number of runs.
            latest_run: Most recent RunSummary.
            current_equity: Current equity value.
            weekly_return: Current week's return.
            target_weekly: Target weekly return.
            weekly_reports: List of recent WeekStats.
            daily_reports: List of recent DayStats.
            recent_runs: List of recent RunSummary objects.
            health: HealthCheck status.
            output_dir: Directory to write index.
            
        Returns:
            Tuple of (markdown_path, html_path).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        context = {
            "total_runs": total_runs,
            "latest_run": latest_run,
            "current_equity": current_equity,
            "weekly_return": weekly_return,
            "target_weekly": target_weekly,
            "weekly_reports": weekly_reports,
            "daily_reports": daily_reports,
            "recent_runs": recent_runs,
            "health": health,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        md_path = None
        html_path = None
        
        # Render Markdown
        if self.config.generate_markdown:
            if self.use_custom_templates:
                template = self.env.get_template("index.md")
            else:
                template = self.env.from_string(INDEX_MD_TEMPLATE)
            
            md_content = template.render(**context)
            md_path = output_dir / "index.md"
            md_path.write_text(md_content)
            logger.info("index_markdown_generated", path=str(md_path))
        
        # Render HTML
        if self.config.generate_html:
            if self.use_custom_templates:
                template = self.env.get_template("index.md")
            else:
                template = self.env.from_string(INDEX_MD_TEMPLATE)
            
            md_content = template.render(**context)
            html_content = self._markdown_to_html(md_content, {})
            
            # Index uses a simpler wrapper without nav
            index_wrapper = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary: #2196F3;
            --success: #4CAF50;
            --warning: #FF9800;
            --danger: #f44336;
            --bg: #f5f5f5;
            --card-bg: #ffffff;
            --text: #333333;
            --border: #e0e0e0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            background-color: var(--bg);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { color: var(--primary); border-bottom: 2px solid var(--primary); padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background-color: #f8f9fa; font-weight: 600; }
        tr:hover { background-color: #f8f9fa; }
        a { color: var(--primary); text-decoration: none; }
        a:hover { text-decoration: underline; }
        hr { border: none; border-top: 1px solid var(--border); margin: 30px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid var(--border); font-size: 12px; color: #666; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        {{ content }}
        <div class="footer">Generated by Footbe Trader Reporting System</div>
    </div>
</body>
</html>"""
            
            wrapper = self.env.from_string(index_wrapper)
            full_html = wrapper.render(
                title="Trading Agent Dashboard",
                content=html_content,
            )
            
            html_path = output_dir / "index.html"
            html_path.write_text(full_html)
            logger.info("index_html_generated", path=str(html_path))
        
        return md_path, html_path
    
    def _markdown_to_html(
        self,
        md_content: str,
        charts: dict[str, ChartResult],
    ) -> str:
        """Convert markdown content to HTML.
        
        This is a simple converter for our specific markdown format.
        For production, consider using a proper markdown library.
        
        Args:
            md_content: Markdown content.
            charts: Dict of charts for image embedding.
            
        Returns:
            HTML content.
        """
        html_lines = []
        in_table = False
        in_code = False
        table_lines = []
        
        for line in md_content.split("\n"):
            # Code blocks
            if line.startswith("```"):
                if in_code:
                    html_lines.append("</code></pre>")
                    in_code = False
                else:
                    html_lines.append("<pre><code>")
                    in_code = True
                continue
            
            if in_code:
                html_lines.append(line)
                continue
            
            # Tables
            if line.startswith("|"):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
                continue
            elif in_table:
                html_lines.append(self._convert_table(table_lines))
                in_table = False
                table_lines = []
            
            # Headers
            if line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            # Horizontal rule
            elif line.strip() == "---":
                html_lines.append("<hr>")
            # Bold text
            elif "**" in line:
                line = self._convert_bold(line)
                html_lines.append(f"<p>{line}</p>")
            # Images (base64 or markdown)
            elif "![" in line and "](" in line:
                html_lines.append(self._convert_image(line, charts))
            # Links
            elif "[" in line and "](" in line:
                line = self._convert_links(line)
                html_lines.append(f"<p>{line}</p>")
            # Lists
            elif line.strip().startswith("- "):
                html_lines.append(f"<li>{line.strip()[2:]}</li>")
            elif line.strip().startswith("* "):
                html_lines.append(f"<li>{line.strip()[2:]}</li>")
            # Emphasis
            elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
                html_lines.append(f"<p><em>{line[1:-1]}</em></p>")
            # Plain text
            elif line.strip():
                html_lines.append(f"<p>{line}</p>")
            else:
                html_lines.append("")
        
        # Handle any remaining table
        if in_table:
            html_lines.append(self._convert_table(table_lines))
        
        return "\n".join(html_lines)
    
    def _convert_table(self, lines: list[str]) -> str:
        """Convert markdown table to HTML."""
        if len(lines) < 2:
            return ""
        
        html = ["<table>"]
        
        for i, line in enumerate(lines):
            # Skip separator line
            if set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            
            cells = [c.strip() for c in line.split("|")[1:-1]]
            
            if i == 0:
                html.append("<thead><tr>")
                for cell in cells:
                    html.append(f"<th>{self._convert_links(cell)}</th>")
                html.append("</tr></thead>")
                html.append("<tbody>")
            else:
                html.append("<tr>")
                for cell in cells:
                    html.append(f"<td>{self._convert_links(cell)}</td>")
                html.append("</tr>")
        
        html.append("</tbody></table>")
        return "\n".join(html)
    
    def _convert_bold(self, text: str) -> str:
        """Convert **bold** to <strong>."""
        import re
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    
    def _convert_links(self, text: str) -> str:
        """Convert [text](url) to <a href>."""
        import re
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    
    def _convert_image(self, text: str, charts: dict[str, ChartResult]) -> str:
        """Convert markdown image to HTML img tag."""
        import re
        
        # Check for base64 data
        match = re.search(r"!\[([^\]]*)\]\((data:image/png;base64,[^)]+)\)", text)
        if match:
            alt = match.group(1)
            src = match.group(2)
            return f'<img src="{src}" alt="{alt}" />'
        
        # Check for file path
        match = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", text)
        if match:
            alt = match.group(1)
            src = match.group(2)
            return f'<img src="{src}" alt="{alt}" />'
        
        return text
