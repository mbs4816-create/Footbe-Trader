"""Reporting subsystem for trading agent.

This module provides comprehensive decision capture, reporting, and auditing
capabilities for the trading agent. It answers key questions:

- What did the agent do?
- Why did it make each decision?
- How is it performing relative to targets?
- Are edge estimates translating into actual returns?

The reporting system consists of:

1. Decision Capture: Every market evaluation is recorded, including SKIP decisions
2. Report Generator: Creates human-readable reports at Run, Daily, and Weekly levels
3. Charts: Matplotlib-based visualizations for equity curves, edge buckets, etc.
4. Artifact Saving: JSON artifacts for full audit trails
5. Navigation Index: Central dashboard linking all reports

Reports are generated in both Markdown and HTML formats with embedded charts.
"""

from footbe_trader.reporting.config import ReportingConfig
from footbe_trader.reporting.queries import ReportingQueries
from footbe_trader.reporting.charts import ChartGenerator
from footbe_trader.reporting.render import ReportRenderer
from footbe_trader.reporting.build_report import (
    ReportBuilder,
    RunReport,
    DailyReport,
    WeeklyReport,
)

__all__ = [
    "ReportingConfig",
    "ReportingQueries",
    "ChartGenerator",
    "ReportRenderer",
    "ReportBuilder",
    "RunReport",
    "DailyReport",
    "WeeklyReport",
]
