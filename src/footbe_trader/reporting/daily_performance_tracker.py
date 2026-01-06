"""Daily Performance Tracker with Aggressive Targets.

Tracks progress towards 10-12% daily return targets with 20% max drawdown limit.

Key Insight: Sports betting P&L is realized at game settlement, NOT when positions
are opened. This means:

1. A position opened Monday may not settle until Saturday
2. Daily tracking must account for UNREALIZED gains that will materialize
3. Pace calculation must project when gains will be realized
4. Risk management must prevent over-concentration in single day's games

Example:
- Start: $1000 bankroll
- Target: 10% daily = $100/day
- Open 5 positions on Monday totaling $200 exposure
- Unrealized P&L: +$50 (will realize Saturday when games finish)
- Tuesday "pace": Need to open more positions to hit target
- But ALSO track Saturday's projected realization: already on track for +$50

This prevents:
- Over-trading early in week (not realizing gains fast enough)
- Under-trading (missing opportunities to compound)
- Concentration risk (too much on one day's games)
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np

from footbe_trader.common.logging import get_logger
from footbe_trader.storage.database import Database

logger = get_logger(__name__)


@dataclass
class DailyTarget:
    """Daily return targets."""

    date: date
    starting_bankroll: float

    # Target range (10-12% daily)
    target_low: float  # 10%
    target_high: float  # 12%
    target_mid: float  # 11% (primary target)

    # Actual performance
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0

    # Exposure
    current_exposure: float = 0.0
    max_exposure_reached: float = 0.0

    # Positions
    positions_opened: int = 0
    positions_closed: int = 0
    positions_pending_settlement: int = 0

    # Game timing
    games_today: int = 0
    games_tomorrow: int = 0
    games_next_7_days: int = 0

    # Projected P&L by settlement date
    projected_pnl_by_date: dict[str, float] = field(default_factory=dict)

    # Status
    status: str = "on_track"  # on_track, ahead, behind, at_risk

    def __post_init__(self):
        """Calculate total P&L."""
        self.total_pnl = self.realized_pnl + self.unrealized_pnl

    @property
    def target_met(self) -> bool:
        """Check if target met."""
        return self.total_pnl >= self.target_low

    @property
    def stretch_goal_met(self) -> bool:
        """Check if stretch goal met."""
        return self.total_pnl >= self.target_high

    @property
    def completion_pct(self) -> float:
        """Progress towards target."""
        if self.target_mid == 0:
            return 0.0
        return self.total_pnl / self.target_mid

    @property
    def return_pct(self) -> float:
        """Actual return percentage."""
        if self.starting_bankroll == 0:
            return 0.0
        return self.total_pnl / self.starting_bankroll


@dataclass
class WeeklyProjection:
    """Projection for the week ahead."""

    week_start: date
    week_end: date

    # Targets
    weekly_target_low: float   # 70% (compounded 10% daily)
    weekly_target_high: float  # 113% (compounded 12% daily)

    # Current progress
    week_to_date_pnl: float = 0.0
    days_elapsed: int = 0
    days_remaining: int = 7

    # Projected settlements
    expected_settlements_by_day: dict[str, float] = field(default_factory=dict)
    total_projected_week_pnl: float = 0.0

    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown_this_week: float = 0.0
    drawdown_limit: float = 0.20  # 20%

    @property
    def on_pace_for_target(self) -> bool:
        """Check if on pace to hit weekly target."""
        if self.days_remaining == 0:
            return self.week_to_date_pnl >= self.weekly_target_low

        # Project final week P&L
        projected_final = self.week_to_date_pnl + self.total_projected_week_pnl
        return projected_final >= self.weekly_target_low

    @property
    def days_ahead_behind(self) -> float:
        """How many days ahead/behind schedule (negative = behind)."""
        if self.days_elapsed == 0:
            return 0.0

        # Expected progress at this point in week
        expected_rate = 0.10  # 10% per day minimum
        expected_so_far = (1 + expected_rate) ** self.days_elapsed - 1

        # Actual progress
        actual_rate = self.week_to_date_pnl
        actual_equivalent_days = np.log(1 + actual_rate) / np.log(1 + expected_rate)

        return actual_equivalent_days - self.days_elapsed


class DailyPerformanceTracker:
    """Tracks daily performance against aggressive targets.

    Targets:
    - 10-12% daily returns
    - Max 20% drawdown
    - Compound to 70-113% weekly

    Key features:
    - Tracks unrealized P&L by settlement date
    - Projects when gains will materialize
    - Prevents over-concentration on single days
    - Warns when falling behind pace
    """

    def __init__(self, db: Database, starting_bankroll: float = 1000.0):
        """Initialize tracker.

        Args:
            db: Database connection.
            starting_bankroll: Starting bankroll amount.
        """
        self.db = db
        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll

        # Targets
        self.daily_target_low = 0.10   # 10%
        self.daily_target_high = 0.12  # 12%
        self.daily_target_mid = 0.11   # 11%
        self.max_drawdown_limit = 0.20 # 20%

        # Performance history
        self.daily_targets: dict[date, DailyTarget] = {}
        self.weekly_projections: dict[date, WeeklyProjection] = {}

        # Peak tracking for drawdown
        self.peak_bankroll = starting_bankroll
        self.peak_date = date.today()

    def get_today_target(self) -> DailyTarget:
        """Get or create today's target."""
        today = date.today()

        if today not in self.daily_targets:
            self.daily_targets[today] = DailyTarget(
                date=today,
                starting_bankroll=self.current_bankroll,
                target_low=self.current_bankroll * self.daily_target_low,
                target_high=self.current_bankroll * self.daily_target_high,
                target_mid=self.current_bankroll * self.daily_target_mid,
            )

        return self.daily_targets[today]

    def update_daily_progress(
        self,
        realized_pnl: float,
        unrealized_pnl: float,
        current_exposure: float,
        positions_opened: int,
        positions_closed: int,
        positions_pending: int,
    ):
        """Update today's progress.

        Args:
            realized_pnl: Realized P&L today.
            unrealized_pnl: Unrealized P&L (all open positions).
            current_exposure: Current dollar exposure.
            positions_opened: Positions opened today.
            positions_closed: Positions closed today.
            positions_pending: Positions awaiting settlement.
        """
        target = self.get_today_target()

        target.realized_pnl = realized_pnl
        target.unrealized_pnl = unrealized_pnl
        target.total_pnl = realized_pnl + unrealized_pnl
        target.current_exposure = current_exposure
        target.max_exposure_reached = max(target.max_exposure_reached, current_exposure)
        target.positions_opened = positions_opened
        target.positions_closed = positions_closed
        target.positions_pending_settlement = positions_pending

        # Update status
        target.status = self._calculate_status(target)

        # Update bankroll
        self.current_bankroll = self.starting_bankroll + target.total_pnl

        # Track peak for drawdown
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll
            self.peak_date = date.today()

    def project_settlement_dates(self, positions: list[dict[str, Any]]):
        """Project when unrealized P&L will be realized.

        Args:
            positions: List of open positions with settlement dates.
        """
        target = self.get_today_target()
        target.projected_pnl_by_date = {}

        for pos in positions:
            settlement_date = pos.get("settlement_date")
            unrealized = pos.get("unrealized_pnl", 0.0)

            if settlement_date and unrealized != 0:
                date_str = settlement_date.strftime("%Y-%m-%d")
                if date_str not in target.projected_pnl_by_date:
                    target.projected_pnl_by_date[date_str] = 0.0
                target.projected_pnl_by_date[date_str] += unrealized

    def get_weekly_projection(self) -> WeeklyProjection:
        """Get projection for current week."""
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)

        if week_start not in self.weekly_projections:
            # Calculate weekly targets (compound daily)
            weekly_low = (1 + self.daily_target_low) ** 7 - 1
            weekly_high = (1 + self.daily_target_high) ** 7 - 1

            self.weekly_projections[week_start] = WeeklyProjection(
                week_start=week_start,
                week_end=week_end,
                weekly_target_low=self.starting_bankroll * weekly_low,
                weekly_target_high=self.starting_bankroll * weekly_high,
                days_elapsed=(today - week_start).days + 1,
                days_remaining=max(0, (week_end - today).days),
            )

        projection = self.weekly_projections[week_start]

        # Update week-to-date P&L
        projection.week_to_date_pnl = sum(
            t.total_pnl
            for d, t in self.daily_targets.items()
            if week_start <= d <= today
        )

        # Update projected settlements
        projection.expected_settlements_by_day = {}
        for target in self.daily_targets.values():
            for date_str, pnl in target.projected_pnl_by_date.items():
                if date_str not in projection.expected_settlements_by_day:
                    projection.expected_settlements_by_day[date_str] = 0.0
                projection.expected_settlements_by_day[date_str] += pnl

        projection.total_projected_week_pnl = sum(projection.expected_settlements_by_day.values())

        # Update drawdown
        projection.current_drawdown = self.get_current_drawdown()
        projection.max_drawdown_this_week = max(
            projection.max_drawdown_this_week,
            projection.current_drawdown,
        )

        return projection

    def _calculate_status(self, target: DailyTarget) -> str:
        """Calculate status for a daily target.

        Args:
            target: Daily target.

        Returns:
            Status string.
        """
        completion = target.completion_pct

        if completion >= 1.2:
            return "crushing_it"
        elif completion >= 1.0:
            return "ahead"
        elif completion >= 0.75:
            return "on_track"
        elif completion >= 0.50:
            return "behind"
        else:
            return "at_risk"

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self.peak_bankroll == 0:
            return 0.0

        drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
        return max(0.0, drawdown)

    def is_drawdown_limit_breached(self) -> bool:
        """Check if drawdown limit breached."""
        return self.get_current_drawdown() >= self.max_drawdown_limit

    def get_remaining_trade_capacity(self) -> float:
        """Get remaining capacity before hitting drawdown limit.

        Returns:
            Dollar amount of loss we can sustain before hitting limit.
        """
        current_drawdown = self.get_current_drawdown()
        remaining_drawdown = self.max_drawdown_limit - current_drawdown

        if remaining_drawdown <= 0:
            return 0.0

        return self.peak_bankroll * remaining_drawdown

    def generate_daily_report(self) -> dict[str, Any]:
        """Generate comprehensive daily performance report."""
        target = self.get_today_target()
        weekly = self.get_weekly_projection()

        return {
            "date": date.today().isoformat(),
            "timestamp": datetime.now(UTC).isoformat(),

            # Current status
            "current_bankroll": self.current_bankroll,
            "starting_bankroll": self.starting_bankroll,
            "total_return": self.current_bankroll - self.starting_bankroll,
            "total_return_pct": (self.current_bankroll - self.starting_bankroll) / self.starting_bankroll,

            # Today's performance
            "today": {
                "target_low": target.target_low,
                "target_mid": target.target_mid,
                "target_high": target.target_high,
                "realized_pnl": target.realized_pnl,
                "unrealized_pnl": target.unrealized_pnl,
                "total_pnl": target.total_pnl,
                "return_pct": target.return_pct,
                "completion_pct": target.completion_pct,
                "status": target.status,
                "target_met": target.target_met,
                "stretch_goal_met": target.stretch_goal_met,
            },

            # Position metrics
            "positions": {
                "opened_today": target.positions_opened,
                "closed_today": target.positions_closed,
                "pending_settlement": target.positions_pending_settlement,
                "current_exposure": target.current_exposure,
                "max_exposure_today": target.max_exposure_reached,
            },

            # Weekly projection
            "weekly": {
                "target_low": weekly.weekly_target_low,
                "target_high": weekly.weekly_target_high,
                "week_to_date_pnl": weekly.week_to_date_pnl,
                "projected_week_pnl": weekly.total_projected_week_pnl,
                "on_pace": weekly.on_pace_for_target,
                "days_ahead_behind": weekly.days_ahead_behind,
                "days_elapsed": weekly.days_elapsed,
                "days_remaining": weekly.days_remaining,
            },

            # Risk metrics
            "risk": {
                "current_drawdown": self.get_current_drawdown(),
                "drawdown_pct": f"{self.get_current_drawdown():.1%}",
                "max_drawdown_limit": self.max_drawdown_limit,
                "drawdown_limit_pct": f"{self.max_drawdown_limit:.1%}",
                "limit_breached": self.is_drawdown_limit_breached(),
                "remaining_capacity": self.get_remaining_trade_capacity(),
                "peak_bankroll": self.peak_bankroll,
                "peak_date": self.peak_date.isoformat(),
            },

            # Settlement projections
            "projected_settlements": target.projected_pnl_by_date,
        }

    def generate_pace_alert(self) -> str | None:
        """Generate alert if pace is off.

        Returns:
            Alert message or None if on pace.
        """
        target = self.get_today_target()
        weekly = self.get_weekly_projection()

        alerts = []

        # Check if behind daily target
        if target.status == "at_risk":
            alerts.append(
                f"🚨 CRITICAL: Only {target.completion_pct:.0%} of daily target reached. "
                f"Need ${target.target_mid - target.total_pnl:.2f} more to hit target."
            )
        elif target.status == "behind":
            alerts.append(
                f"⚠️ WARNING: {target.completion_pct:.0%} of daily target. "
                f"Need ${target.target_mid - target.total_pnl:.2f} to hit 11% target."
            )

        # Check if behind weekly pace
        if not weekly.on_pace_for_target:
            days_behind = abs(weekly.days_ahead_behind)
            alerts.append(
                f"📉 Week is {days_behind:.1f} days behind pace. "
                f"Projected week: ${weekly.week_to_date_pnl + weekly.total_projected_week_pnl:.2f} "
                f"vs target ${weekly.weekly_target_low:.2f}"
            )

        # Check drawdown
        if self.is_drawdown_limit_breached():
            alerts.append(
                f"🛑 STOP: Drawdown limit breached! "
                f"Current: {self.get_current_drawdown():.1%}, Limit: {self.max_drawdown_limit:.1%}. "
                f"PAUSE TRADING IMMEDIATELY."
            )
        elif self.get_current_drawdown() > self.max_drawdown_limit * 0.75:
            alerts.append(
                f"⚠️ Drawdown warning: {self.get_current_drawdown():.1%} "
                f"(limit: {self.max_drawdown_limit:.1%}). "
                f"Only ${self.get_remaining_trade_capacity():.2f} capacity remaining."
            )

        # Check if crushing it
        if target.stretch_goal_met:
            alerts.append(
                f"🚀 Stretch goal achieved! {target.return_pct:.1%} return today "
                f"(target: {self.daily_target_high:.1%})"
            )

        return "\n".join(alerts) if alerts else None

    def persist_daily_snapshot(self):
        """Save daily snapshot to database."""
        report = self.generate_daily_report()

        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT INTO daily_performance_snapshots (
                date, timestamp, current_bankroll, total_return_pct,
                daily_target_met, weekly_on_pace, current_drawdown,
                report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["date"],
                report["timestamp"],
                report["current_bankroll"],
                report["total_return_pct"],
                report["today"]["target_met"],
                report["weekly"]["on_pace"],
                report["risk"]["current_drawdown"],
                str(report),
            ),
        )
        self.db.connection.commit()

        logger.info("daily_snapshot_persisted", date=report["date"])
