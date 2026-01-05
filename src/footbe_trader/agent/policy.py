"""Trading Policy Module.

Orchestrates all the factors that influence trading decisions:
- Pacing toward target
- Drawdown throttling
- Time-to-kickoff adjustments
- Exposure limits

The policy produces a consolidated view of "how aggressive should we be right now"
and generates human-readable explanations for all decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from footbe_trader.agent.objective import (
    AgentObjective,
    DrawdownBand,
    DrawdownThrottle,
    EquitySnapshot,
    PacingAdjustment,
    PacingState,
    PacingTracker,
    TimeToKickoffCategory,
    classify_time_to_kickoff,
    get_time_to_kickoff_adjustment,
)
from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now

logger = get_logger(__name__)


@dataclass
class ExposureState:
    """Current exposure state across the portfolio."""

    gross_exposure: float = 0.0  # Total $ at risk
    exposure_by_fixture: dict[int, float] = field(default_factory=dict)
    exposure_by_league: dict[str, float] = field(default_factory=dict)
    position_count: int = 0
    trade_count_7d: int = 0  # Trades in last 7 days

    def get_fixture_exposure(self, fixture_id: int) -> float:
        """Get exposure for a specific fixture."""
        return self.exposure_by_fixture.get(fixture_id, 0.0)

    def get_league_exposure(self, league: str) -> float:
        """Get exposure for a specific league."""
        return self.exposure_by_league.get(league, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gross_exposure": self.gross_exposure,
            "exposure_by_fixture": self.exposure_by_fixture,
            "exposure_by_league": self.exposure_by_league,
            "position_count": self.position_count,
            "trade_count_7d": self.trade_count_7d,
        }


@dataclass
class PolicyDecision:
    """Consolidated policy decision for a potential trade.

    Captures all factors considered and the final multipliers applied.
    """

    # Input context
    fixture_id: int
    league: str
    outcome: str
    hours_to_kickoff: float
    raw_edge: float
    raw_position_size: float
    equity: float

    # State at decision time
    pacing_state: PacingState
    drawdown: float
    drawdown_band: DrawdownBand
    time_category: TimeToKickoffCategory

    # Exposure checks
    current_fixture_exposure: float
    current_league_exposure: float
    current_gross_exposure: float
    max_fixture_exposure: float
    max_league_exposure: float
    max_gross_exposure: float

    # Adjustments applied
    pacing_edge_mult: float = 1.0
    pacing_size_mult: float = 1.0
    drawdown_size_mult: float = 1.0
    time_edge_mult: float = 1.0
    time_size_mult: float = 1.0

    # Final values
    adjusted_edge_threshold: float = 0.0
    adjusted_position_size: float = 0.0
    final_position_size: int = 0  # After all caps

    # Decision
    can_trade: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    # Explanation
    explanation: str = ""
    explanation_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_id": self.fixture_id,
            "league": self.league,
            "outcome": self.outcome,
            "hours_to_kickoff": self.hours_to_kickoff,
            "raw_edge": self.raw_edge,
            "raw_position_size": self.raw_position_size,
            "equity": self.equity,
            "pacing_state": self.pacing_state.value,
            "drawdown": self.drawdown,
            "drawdown_band": self.drawdown_band.value,
            "time_category": self.time_category.value,
            "current_fixture_exposure": self.current_fixture_exposure,
            "current_league_exposure": self.current_league_exposure,
            "current_gross_exposure": self.current_gross_exposure,
            "pacing_edge_mult": self.pacing_edge_mult,
            "pacing_size_mult": self.pacing_size_mult,
            "drawdown_size_mult": self.drawdown_size_mult,
            "time_edge_mult": self.time_edge_mult,
            "time_size_mult": self.time_size_mult,
            "adjusted_edge_threshold": self.adjusted_edge_threshold,
            "adjusted_position_size": self.adjusted_position_size,
            "final_position_size": self.final_position_size,
            "can_trade": self.can_trade,
            "rejection_reasons": self.rejection_reasons,
            "explanation": self.explanation,
            "explanation_factors": self.explanation_factors,
        }


class TradingPolicy:
    """Orchestrates trading policy decisions.

    Combines pacing, drawdown, time-to-kickoff, and exposure factors
    to determine how aggressive the agent should be for each trade.
    """

    def __init__(
        self,
        objective: AgentObjective | None = None,
        drawdown_throttle: DrawdownThrottle | None = None,
        pacing_tracker: PacingTracker | None = None,
        initial_equity: float = 10000.0,
    ):
        """Initialize trading policy.

        Args:
            objective: Agent objective with targets and limits.
            drawdown_throttle: Drawdown-based throttle configuration.
            pacing_tracker: Tracker for pacing toward target.
            initial_equity: Starting equity.
        """
        self.objective = objective or AgentObjective()
        self.drawdown_throttle = drawdown_throttle or DrawdownThrottle()
        self.pacing_tracker = pacing_tracker or PacingTracker(
            objective=self.objective,
            initial_equity=initial_equity,
        )
        self.exposure_state = ExposureState()

        logger.info(
            "trading_policy_initialized",
            target_return=f"{self.objective.target_weekly_return:.0%}",
            max_drawdown=f"{self.objective.max_drawdown:.0%}",
        )

    def update_equity(
        self,
        equity: float,
        cash: float = 0.0,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> EquitySnapshot:
        """Update equity tracking.

        Args:
            equity: Total equity (cash + mark-to-market).
            cash: Cash component.
            unrealized_pnl: Unrealized P&L.
            realized_pnl: Realized P&L.

        Returns:
            Created equity snapshot.
        """
        return self.pacing_tracker.record_equity(
            equity=equity,
            cash=cash,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )

    def update_exposure(
        self,
        gross_exposure: float,
        exposure_by_fixture: dict[int, float],
        exposure_by_league: dict[str, float],
        position_count: int,
        trade_count_7d: int = 0,
    ) -> None:
        """Update exposure state.

        Args:
            gross_exposure: Total $ at risk.
            exposure_by_fixture: Exposure per fixture ID.
            exposure_by_league: Exposure per league.
            position_count: Number of open positions.
            trade_count_7d: Trades in last 7 days.
        """
        self.exposure_state = ExposureState(
            gross_exposure=gross_exposure,
            exposure_by_fixture=exposure_by_fixture,
            exposure_by_league=exposure_by_league,
            position_count=position_count,
            trade_count_7d=trade_count_7d,
        )

    def evaluate_trade(
        self,
        fixture_id: int,
        league: str,
        outcome: str,
        model_prob: float,
        market_price: float,
        base_edge_threshold: float,
        kelly_position_size: float,
        hours_to_kickoff: float,
        price_per_contract: float,
    ) -> PolicyDecision:
        """Evaluate a potential trade against policy.

        Args:
            fixture_id: Fixture identifier.
            league: League identifier.
            outcome: Outcome type (home_win, draw, away_win).
            model_prob: Model's probability estimate.
            market_price: Current market ask price.
            base_edge_threshold: Strategy's base edge threshold.
            kelly_position_size: Kelly-optimal position size.
            hours_to_kickoff: Hours until fixture starts.
            price_per_contract: Price per contract for exposure calc.

        Returns:
            PolicyDecision with all factors and final sizing.
        """
        equity = self.pacing_tracker.current_equity
        drawdown = self.pacing_tracker.current_drawdown
        raw_edge = model_prob - market_price

        # Get adjustments from each source
        pacing_adj = self.pacing_tracker.get_pacing_adjustment()
        drawdown_mult = self.drawdown_throttle.get_multiplier(drawdown)
        drawdown_band = self.drawdown_throttle.get_band(drawdown)
        time_edge_mult, time_size_mult, time_rationale = get_time_to_kickoff_adjustment(
            hours_to_kickoff, self.objective
        )
        time_category = classify_time_to_kickoff(hours_to_kickoff)

        # Exposure limits in dollars
        max_fixture_exp = equity * self.objective.max_exposure_per_fixture
        max_league_exp = equity * self.objective.max_exposure_per_league
        max_gross_exp = equity * self.objective.max_gross_exposure

        # Current exposures
        current_fixture_exp = self.exposure_state.get_fixture_exposure(fixture_id)
        current_league_exp = self.exposure_state.get_league_exposure(league)
        current_gross_exp = self.exposure_state.gross_exposure

        # Create decision record
        decision = PolicyDecision(
            fixture_id=fixture_id,
            league=league,
            outcome=outcome,
            hours_to_kickoff=hours_to_kickoff,
            raw_edge=raw_edge,
            raw_position_size=kelly_position_size,
            equity=equity,
            pacing_state=pacing_adj.pacing_state,
            drawdown=drawdown,
            drawdown_band=drawdown_band,
            time_category=time_category,
            current_fixture_exposure=current_fixture_exp,
            current_league_exposure=current_league_exp,
            current_gross_exposure=current_gross_exp,
            max_fixture_exposure=max_fixture_exp,
            max_league_exposure=max_league_exp,
            max_gross_exposure=max_gross_exp,
            pacing_edge_mult=pacing_adj.edge_threshold_multiplier,
            pacing_size_mult=pacing_adj.sizing_multiplier,
            drawdown_size_mult=drawdown_mult,
            time_edge_mult=time_edge_mult,
            time_size_mult=time_size_mult,
        )

        # Build explanation factors
        factors = []

        # Factor 1: Model edge
        factors.append(
            f"Model prob: {model_prob:.1%}, Market: {market_price:.1%}, "
            f"Edge: {raw_edge:+.1%}"
        )

        # Factor 2: Time to kickoff
        factors.append(
            f"Time to kickoff: {hours_to_kickoff:.1f}h ({time_category.value})"
        )

        # Factor 3: Pacing
        factors.append(
            f"Pacing: {pacing_adj.pacing_state.value} "
            f"(7d return: {self.pacing_tracker.get_rolling_7d_return():.1%})"
        )

        # Factor 4: Drawdown
        factors.append(
            f"Drawdown: {drawdown:.1%} ({drawdown_band.value} band)"
        )

        # Factor 5: Exposure
        factors.append(
            f"Exposure - Fixture: ${current_fixture_exp:.0f}/${max_fixture_exp:.0f}, "
            f"League: ${current_league_exp:.0f}/${max_league_exp:.0f}, "
            f"Gross: ${current_gross_exp:.0f}/${max_gross_exp:.0f}"
        )

        decision.explanation_factors = factors

        # Check rejection conditions
        rejection_reasons = []

        # Check 1: Severe drawdown - no new entries
        if drawdown_band == DrawdownBand.SEVERE:
            rejection_reasons.append(
                f"Drawdown {drawdown:.1%} exceeds severe threshold "
                f"({self.drawdown_throttle.band_severe_threshold:.0%}). "
                f"No new entries allowed."
            )

        # Check 2: Edge threshold (with all multipliers)
        adjusted_threshold = (
            base_edge_threshold
            * pacing_adj.edge_threshold_multiplier
            * time_edge_mult
        )
        decision.adjusted_edge_threshold = adjusted_threshold

        if raw_edge < adjusted_threshold:
            rejection_reasons.append(
                f"Edge {raw_edge:.1%} below adjusted threshold {adjusted_threshold:.1%} "
                f"(base {base_edge_threshold:.1%} × pacing {pacing_adj.edge_threshold_multiplier:.2f} "
                f"× time {time_edge_mult:.2f})"
            )

        # Check 3: Fixture exposure limit
        proposed_exposure = kelly_position_size * price_per_contract
        if current_fixture_exp + proposed_exposure > max_fixture_exp:
            rejection_reasons.append(
                f"Would exceed fixture exposure limit: "
                f"${current_fixture_exp:.0f} + ${proposed_exposure:.0f} > ${max_fixture_exp:.0f}"
            )

        # Check 4: League exposure limit
        if current_league_exp + proposed_exposure > max_league_exp:
            rejection_reasons.append(
                f"Would exceed league exposure limit: "
                f"${current_league_exp:.0f} + ${proposed_exposure:.0f} > ${max_league_exp:.0f}"
            )

        # Check 5: Gross exposure limit
        if current_gross_exp + proposed_exposure > max_gross_exp:
            rejection_reasons.append(
                f"Would exceed gross exposure limit: "
                f"${current_gross_exp:.0f} + ${proposed_exposure:.0f} > ${max_gross_exp:.0f}"
            )

        decision.rejection_reasons = rejection_reasons

        if rejection_reasons:
            decision.can_trade = False
            decision.adjusted_position_size = 0
            decision.final_position_size = 0
            decision.explanation = self._build_rejection_explanation(
                decision, rejection_reasons
            )
        else:
            # Calculate adjusted position size
            combined_size_mult = (
                pacing_adj.sizing_multiplier
                * drawdown_mult
                * time_size_mult
            )
            adjusted_size = kelly_position_size * combined_size_mult
            decision.adjusted_position_size = adjusted_size

            # Apply exposure caps
            remaining_fixture = max_fixture_exp - current_fixture_exp
            remaining_league = max_league_exp - current_league_exp
            remaining_gross = max_gross_exp - current_gross_exp
            min_remaining = min(remaining_fixture, remaining_league, remaining_gross)

            max_contracts = int(min_remaining / price_per_contract) if price_per_contract > 0 else 0
            final_size = min(int(adjusted_size), max_contracts)

            # Also apply max single trade fraction
            max_single = int(equity * self.objective.max_single_trade_fraction / price_per_contract)
            final_size = min(final_size, max_single)

            decision.final_position_size = max(0, final_size)
            decision.can_trade = final_size > 0

            decision.explanation = self._build_trade_explanation(decision)

        logger.debug(
            "policy_evaluated",
            fixture_id=fixture_id,
            outcome=outcome,
            can_trade=decision.can_trade,
            final_size=decision.final_position_size,
        )

        return decision

    def _build_trade_explanation(self, decision: PolicyDecision) -> str:
        """Build human-readable explanation for approved trade."""
        lines = [
            "=== Trade Decision ===",
            "",
            *decision.explanation_factors,
            "",
            "--- Adjustments Applied ---",
            f"Edge threshold: {decision.adjusted_edge_threshold:.1%} "
            f"(pacing: ×{decision.pacing_edge_mult:.2f}, time: ×{decision.time_edge_mult:.2f})",
            f"Position sizing: {decision.final_position_size} contracts",
            f"  Raw Kelly: {decision.raw_position_size:.0f}",
            f"  × Pacing: {decision.pacing_size_mult:.2f}",
            f"  × Drawdown: {decision.drawdown_size_mult:.2f}",
            f"  × Time: {decision.time_size_mult:.2f}",
            f"  = Adjusted: {decision.adjusted_position_size:.0f}",
            f"  → Capped: {decision.final_position_size}",
            "",
            f"Action: BUY {decision.final_position_size} contracts of {decision.outcome}",
            f"Rationale: Edge {decision.raw_edge:.1%} exceeds threshold, risk acceptable",
        ]
        return "\n".join(lines)

    def _build_rejection_explanation(
        self,
        decision: PolicyDecision,
        reasons: list[str],
    ) -> str:
        """Build human-readable explanation for rejected trade."""
        lines = [
            "=== Trade Decision (SKIP) ===",
            "",
            *decision.explanation_factors,
            "",
            "--- Rejection Reasons ---",
            *[f"• {r}" for r in reasons],
            "",
            f"Action: SKIP {decision.outcome}",
        ]
        return "\n".join(lines)

    def get_status(self) -> dict[str, Any]:
        """Get current policy status."""
        pacing_adj = self.pacing_tracker.get_pacing_adjustment()
        drawdown = self.pacing_tracker.current_drawdown

        return {
            "equity": self.pacing_tracker.current_equity,
            "peak_equity": self.pacing_tracker.peak_equity,
            "drawdown": drawdown,
            "drawdown_band": self.drawdown_throttle.get_band(drawdown).value,
            "drawdown_multiplier": self.drawdown_throttle.get_multiplier(drawdown),
            "can_enter": self.drawdown_throttle.can_enter(drawdown),
            "pacing_state": pacing_adj.pacing_state.value,
            "rolling_7d_return": self.pacing_tracker.get_rolling_7d_return(),
            "target_weekly_return": self.objective.target_weekly_return,
            "pacing_edge_multiplier": pacing_adj.edge_threshold_multiplier,
            "pacing_size_multiplier": pacing_adj.sizing_multiplier,
            "exposure": self.exposure_state.to_dict(),
        }

    def generate_status_report(self) -> str:
        """Generate human-readable status report."""
        status = self.get_status()
        pacing_adj = self.pacing_tracker.get_pacing_adjustment()

        lines = [
            "=" * 60,
            "TRADING POLICY STATUS",
            "=" * 60,
            "",
            f"Equity: ${status['equity']:,.2f} (Peak: ${status['peak_equity']:,.2f})",
            f"Drawdown: {status['drawdown']:.1%} ({status['drawdown_band']} band)",
            f"  → Sizing multiplier: {status['drawdown_multiplier']:.0%}",
            f"  → New entries allowed: {'YES' if status['can_enter'] else 'NO'}",
            "",
            f"Pacing: {status['pacing_state']}",
            f"  → 7-day return: {status['rolling_7d_return']:.1%}",
            f"  → Target: {status['target_weekly_return']:.1%}",
            f"  → Edge threshold multiplier: {status['pacing_edge_multiplier']:.2f}",
            f"  → Sizing multiplier: {status['pacing_size_multiplier']:.2f}",
            "",
            f"Exposure:",
            f"  → Gross: ${status['exposure']['gross_exposure']:.2f}",
            f"  → Positions: {status['exposure']['position_count']}",
            "",
            "Current behavior:",
            f"  {pacing_adj.rationale}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)
