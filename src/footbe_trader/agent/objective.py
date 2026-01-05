"""Agent Objective System.

Defines explicit objectives, constraints, and pacing logic for the trading agent.
The agent targets ~10% weekly equity growth while enforcing strict drawdown control.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now

logger = get_logger(__name__)


class PacingState(str, Enum):
    """Agent's progress relative to target."""

    AHEAD_OF_PACE = "ahead_of_pace"  # Above target trajectory
    ON_PACE = "on_pace"  # Within tolerance of target
    BEHIND_PACE = "behind_pace"  # Below target trajectory


class DrawdownBand(str, Enum):
    """Drawdown severity bands for throttling."""

    NONE = "none"  # < 5% drawdown
    LIGHT = "light"  # 5-10% drawdown
    MODERATE = "moderate"  # 10-15% drawdown
    SEVERE = "severe"  # > 15% drawdown (no new entries)


class TimeToKickoffCategory(str, Enum):
    """Categorization of time until fixture kickoff."""

    VERY_EARLY = "very_early"  # > 48 hours
    EARLY = "early"  # 24-48 hours
    STANDARD = "standard"  # 6-24 hours
    OPTIMAL = "optimal"  # 2-6 hours (core trading window)
    LATE = "late"  # < 2 hours


@dataclass
class AgentObjective:
    """Defines the agent's trading objective.

    The objective is expressed as a target equity growth rate with
    hard and soft constraints that must not be violated.
    """

    # Primary target
    target_weekly_return: float = 0.10  # 10% per rolling 7-day window
    target_tolerance: float = 0.02  # ±2% considered "on pace"

    # Hard constraints (never violate)
    max_drawdown: float = 0.15  # 15% max drawdown from peak
    max_exposure_per_fixture: float = 0.015  # 1.5% of equity
    max_exposure_per_league: float = 0.20  # 20% of equity
    max_gross_exposure: float = 0.80  # 80% of equity

    # Soft drawdown threshold (start throttling)
    soft_drawdown_threshold: float = 0.07  # 7% - start reducing risk

    # Behavioral preferences
    prefer_many_small_trades: bool = True
    min_trades_per_week_target: int = 20  # Prefer diversification
    max_single_trade_fraction: float = 0.03  # 3% max per single trade

    # Early market penalties
    early_market_edge_multiplier: float = 1.5  # Require 50% more edge early
    early_market_size_multiplier: float = 0.5  # Half size for early markets
    very_early_market_edge_multiplier: float = 2.0  # Require 100% more edge

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_weekly_return": self.target_weekly_return,
            "target_tolerance": self.target_tolerance,
            "max_drawdown": self.max_drawdown,
            "max_exposure_per_fixture": self.max_exposure_per_fixture,
            "max_exposure_per_league": self.max_exposure_per_league,
            "max_gross_exposure": self.max_gross_exposure,
            "soft_drawdown_threshold": self.soft_drawdown_threshold,
            "prefer_many_small_trades": self.prefer_many_small_trades,
            "min_trades_per_week_target": self.min_trades_per_week_target,
            "max_single_trade_fraction": self.max_single_trade_fraction,
            "early_market_edge_multiplier": self.early_market_edge_multiplier,
            "early_market_size_multiplier": self.early_market_size_multiplier,
        }


@dataclass
class DrawdownThrottle:
    """Drawdown-based position sizing throttle.

    Defines throttle bands that multiplicatively reduce position sizing
    as drawdown increases.
    """

    # Throttle band thresholds
    band_light_threshold: float = 0.05  # 5% drawdown
    band_moderate_threshold: float = 0.10  # 10% drawdown
    band_severe_threshold: float = 0.15  # 15% drawdown

    # Sizing multipliers per band
    multiplier_none: float = 1.00  # 100% sizing
    multiplier_light: float = 0.70  # 70% sizing
    multiplier_moderate: float = 0.40  # 40% sizing
    multiplier_severe: float = 0.00  # 0% - no new entries

    def get_band(self, drawdown: float) -> DrawdownBand:
        """Get the drawdown band for a given drawdown level.

        Args:
            drawdown: Current drawdown as fraction (0.10 = 10%)

        Returns:
            DrawdownBand enum value.
        """
        if drawdown >= self.band_severe_threshold:
            return DrawdownBand.SEVERE
        elif drawdown >= self.band_moderate_threshold:
            return DrawdownBand.MODERATE
        elif drawdown >= self.band_light_threshold:
            return DrawdownBand.LIGHT
        else:
            return DrawdownBand.NONE

    def get_multiplier(self, drawdown: float) -> float:
        """Get the sizing multiplier for a given drawdown level.

        Args:
            drawdown: Current drawdown as fraction.

        Returns:
            Sizing multiplier (0.0 to 1.0).
        """
        band = self.get_band(drawdown)
        if band == DrawdownBand.SEVERE:
            return self.multiplier_severe
        elif band == DrawdownBand.MODERATE:
            return self.multiplier_moderate
        elif band == DrawdownBand.LIGHT:
            return self.multiplier_light
        else:
            return self.multiplier_none

    def can_enter(self, drawdown: float) -> bool:
        """Check if new entries are allowed at this drawdown level.

        Args:
            drawdown: Current drawdown as fraction.

        Returns:
            True if new entries are allowed.
        """
        return self.get_band(drawdown) != DrawdownBand.SEVERE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "band_light_threshold": self.band_light_threshold,
            "band_moderate_threshold": self.band_moderate_threshold,
            "band_severe_threshold": self.band_severe_threshold,
            "multiplier_none": self.multiplier_none,
            "multiplier_light": self.multiplier_light,
            "multiplier_moderate": self.multiplier_moderate,
            "multiplier_severe": self.multiplier_severe,
        }


@dataclass
class EquitySnapshot:
    """Point-in-time equity snapshot for tracking."""

    timestamp: datetime = field(default_factory=utc_now)
    equity: float = 0.0
    cash: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "cash": self.cash,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
        }


@dataclass
class PacingAdjustment:
    """Adjustments to strategy based on pacing state."""

    pacing_state: PacingState
    edge_threshold_multiplier: float  # Multiply base edge threshold
    sizing_multiplier: float  # Multiply position size
    prefer_exits: bool  # Prefer taking profits
    rationale: str  # Human-readable explanation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pacing_state": self.pacing_state.value,
            "edge_threshold_multiplier": self.edge_threshold_multiplier,
            "sizing_multiplier": self.sizing_multiplier,
            "prefer_exits": self.prefer_exits,
            "rationale": self.rationale,
        }


class PacingTracker:
    """Tracks agent's pace toward weekly target.

    Computes rolling 7-day equity change and classifies the agent's
    progress as AHEAD, ON_PACE, or BEHIND relative to target.
    """

    def __init__(
        self,
        objective: AgentObjective | None = None,
        initial_equity: float = 10000.0,
    ):
        """Initialize pacing tracker.

        Args:
            objective: Agent objective with targets.
            initial_equity: Starting equity for tracking.
        """
        self.objective = objective or AgentObjective()
        self.initial_equity = initial_equity
        self._equity_history: list[EquitySnapshot] = []
        self._peak_equity = initial_equity

        # Record initial snapshot
        self._equity_history.append(
            EquitySnapshot(
                equity=initial_equity,
                cash=initial_equity,
                peak_equity=initial_equity,
                drawdown=0.0,
            )
        )

    @property
    def peak_equity(self) -> float:
        """Get peak equity observed."""
        return self._peak_equity

    @property
    def current_equity(self) -> float:
        """Get most recent equity value."""
        if self._equity_history:
            return self._equity_history[-1].equity
        return self.initial_equity

    @property
    def current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self.current_equity) / self._peak_equity

    def record_equity(
        self,
        equity: float,
        cash: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> EquitySnapshot:
        """Record new equity snapshot.

        Args:
            equity: Total equity (cash + positions).
            cash: Cash component.
            unrealized_pnl: Unrealized P&L from open positions.
            realized_pnl: Realized P&L from closed positions.

        Returns:
            The created snapshot.
        """
        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Calculate drawdown
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity

        snapshot = EquitySnapshot(
            timestamp=utc_now(),
            equity=equity,
            cash=cash,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            peak_equity=self._peak_equity,
            drawdown=drawdown,
        )
        self._equity_history.append(snapshot)

        # Prune old history (keep last 30 days)
        cutoff = utc_now() - timedelta(days=30)
        self._equity_history = [
            s for s in self._equity_history if s.timestamp >= cutoff
        ]

        logger.debug(
            "equity_recorded",
            equity=equity,
            peak=self._peak_equity,
            drawdown=f"{drawdown:.2%}",
        )

        return snapshot

    def get_rolling_7d_return(self) -> float:
        """Calculate rolling 7-day return.

        Returns:
            7-day return as fraction (0.05 = 5% return).
        """
        now = utc_now()
        cutoff = now - timedelta(days=7)

        # Find snapshot closest to 7 days ago
        old_snapshots = [s for s in self._equity_history if s.timestamp <= cutoff]

        if not old_snapshots:
            # Use oldest available or initial
            if len(self._equity_history) > 1:
                old_equity = self._equity_history[0].equity
            else:
                old_equity = self.initial_equity
        else:
            old_equity = old_snapshots[-1].equity

        current = self.current_equity

        if old_equity <= 0:
            return 0.0

        return (current - old_equity) / old_equity

    def get_required_return(self) -> float:
        """Calculate required return to hit weekly target.

        Returns:
            Required return for remaining week.
        """
        current_return = self.get_rolling_7d_return()
        required = self.objective.target_weekly_return - current_return
        return max(0.0, required)

    def get_pacing_state(self) -> PacingState:
        """Classify current pacing state.

        Returns:
            PacingState enum value.
        """
        current_return = self.get_rolling_7d_return()
        target = self.objective.target_weekly_return
        tolerance = self.objective.target_tolerance

        if current_return >= target:
            return PacingState.AHEAD_OF_PACE
        elif current_return >= target - tolerance:
            return PacingState.ON_PACE
        else:
            return PacingState.BEHIND_PACE

    def get_pacing_adjustment(self) -> PacingAdjustment:
        """Get strategy adjustments based on pacing.

        Returns:
            PacingAdjustment with multipliers and preferences.
        """
        state = self.get_pacing_state()
        current_return = self.get_rolling_7d_return()
        target = self.objective.target_weekly_return

        if state == PacingState.AHEAD_OF_PACE:
            # Ahead: raise bar, reduce size, prefer exits
            excess = current_return - target
            # More ahead = more conservative
            edge_mult = 1.0 + min(excess / target, 0.5)  # Up to 1.5x edge required
            size_mult = max(0.5, 1.0 - excess / target)  # Down to 0.5x size

            return PacingAdjustment(
                pacing_state=state,
                edge_threshold_multiplier=edge_mult,
                sizing_multiplier=size_mult,
                prefer_exits=True,
                rationale=(
                    f"Ahead of pace ({current_return:.1%} vs {target:.1%} target). "
                    f"Raising edge threshold by {(edge_mult-1)*100:.0f}%, "
                    f"reducing size to {size_mult*100:.0f}%. Prefer locking profits."
                ),
            )

        elif state == PacingState.ON_PACE:
            # On pace: use baseline
            return PacingAdjustment(
                pacing_state=state,
                edge_threshold_multiplier=1.0,
                sizing_multiplier=1.0,
                prefer_exits=False,
                rationale=(
                    f"On pace ({current_return:.1%} vs {target:.1%} target). "
                    f"Using baseline thresholds and sizing."
                ),
            )

        else:
            # Behind: slightly lower bar, allow more size (within limits)
            deficit = target - current_return
            # More behind = slightly more aggressive (but capped)
            edge_mult = max(0.80, 1.0 - deficit / (target * 2))  # Down to 0.8x
            size_mult = min(1.25, 1.0 + deficit / target)  # Up to 1.25x

            return PacingAdjustment(
                pacing_state=state,
                edge_threshold_multiplier=edge_mult,
                sizing_multiplier=size_mult,
                prefer_exits=False,
                rationale=(
                    f"Behind pace ({current_return:.1%} vs {target:.1%} target). "
                    f"Lowering edge threshold by {(1-edge_mult)*100:.0f}%, "
                    f"allowing size up to {size_mult*100:.0f}%. "
                    f"Still respecting all risk limits."
                ),
            )

    def get_equity_history(
        self,
        days: int = 7,
    ) -> list[EquitySnapshot]:
        """Get equity history for period.

        Args:
            days: Number of days of history.

        Returns:
            List of equity snapshots.
        """
        cutoff = utc_now() - timedelta(days=days)
        return [s for s in self._equity_history if s.timestamp >= cutoff]

    def to_dict(self) -> dict[str, Any]:
        """Convert current state to dictionary."""
        return {
            "initial_equity": self.initial_equity,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": self.current_drawdown,
            "rolling_7d_return": self.get_rolling_7d_return(),
            "required_return": self.get_required_return(),
            "pacing_state": self.get_pacing_state().value,
            "pacing_adjustment": self.get_pacing_adjustment().to_dict(),
        }


def classify_time_to_kickoff(hours: float) -> TimeToKickoffCategory:
    """Classify time until kickoff into categories.

    Args:
        hours: Hours until fixture kickoff.

    Returns:
        TimeToKickoffCategory enum value.
    """
    if hours > 48:
        return TimeToKickoffCategory.VERY_EARLY
    elif hours > 24:
        return TimeToKickoffCategory.EARLY
    elif hours > 6:
        return TimeToKickoffCategory.STANDARD
    elif hours >= 2:
        return TimeToKickoffCategory.OPTIMAL
    else:
        return TimeToKickoffCategory.LATE


def get_time_to_kickoff_adjustment(
    hours: float,
    objective: AgentObjective | None = None,
) -> tuple[float, float, str]:
    """Get edge and sizing adjustments for time to kickoff.

    Args:
        hours: Hours until kickoff.
        objective: Agent objective with multipliers.

    Returns:
        Tuple of (edge_multiplier, size_multiplier, rationale).
    """
    obj = objective or AgentObjective()
    category = classify_time_to_kickoff(hours)

    if category == TimeToKickoffCategory.VERY_EARLY:
        return (
            obj.very_early_market_edge_multiplier,
            obj.early_market_size_multiplier * 0.5,  # Extra reduction
            f"Very early market ({hours:.1f}h to kickoff). "
            f"Requiring {obj.very_early_market_edge_multiplier}x edge, "
            f"25% sizing due to uncertainty.",
        )

    elif category == TimeToKickoffCategory.EARLY:
        return (
            obj.early_market_edge_multiplier,
            obj.early_market_size_multiplier,
            f"Early market ({hours:.1f}h to kickoff). "
            f"Requiring {obj.early_market_edge_multiplier}x edge, "
            f"50% sizing.",
        )

    elif category == TimeToKickoffCategory.STANDARD:
        return (
            1.1,  # Slight premium for standard window
            0.85,  # Slight reduction
            f"Standard window ({hours:.1f}h to kickoff). "
            f"1.1x edge threshold, 85% sizing.",
        )

    elif category == TimeToKickoffCategory.OPTIMAL:
        return (
            1.0,  # Baseline
            1.0,  # Full sizing
            f"Optimal trading window ({hours:.1f}h to kickoff). "
            f"Baseline thresholds and full sizing.",
        )

    else:  # LATE
        return (
            1.2,  # Higher bar late
            0.70,  # Reduced sizing
            f"Late market ({hours:.1f}h to kickoff). "
            f"1.2x edge threshold, 70% sizing due to limited time.",
        )
