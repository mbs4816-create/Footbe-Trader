"""Position Invalidator - Exits positions when pre-game assumptions become invalid.

This module handles the critical problem:
- You enter a position based on PRE-GAME model prediction
- Game starts and situation changes (team falls behind, key player injured, etc.)
- Your original thesis is now INVALID
- Must exit position immediately to avoid adverse selection

Examples:
1. Bought Home Win at 0.50 (fair value 0.60)
   - Game starts, home team goes down 0-1 in first 10 minutes
   - Home Win now trading at 0.30 (team is losing!)
   - Original edge calculation is INVALID, must exit

2. Bought Over 2.5 at 0.45 (expecting high-scoring)
   - Game starts defensive, 0-0 at halftime
   - Over 2.5 now at 0.25 (unlikely to hit)
   - Must exit to avoid holding to zero
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from footbe_trader.agent.live_game import GamePhase, LiveGameState, LiveGameStateProvider
from footbe_trader.common.logging import get_logger
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.kalshi.interfaces import OrderData
from footbe_trader.storage.database import Database
from footbe_trader.strategy.paper_trading import PaperPosition, PaperTradingSimulator

logger = get_logger(__name__)


@dataclass
class InvalidationReason:
    """Why a position should be invalidated."""

    reason_code: str
    reason_text: str
    severity: str  # "warning", "critical"
    recommended_action: str  # "monitor", "reduce", "exit_all"

    # Supporting data
    original_edge: float = 0.0
    current_edge: float = 0.0
    edge_degradation: float = 0.0

    price_at_entry: float = 0.0
    current_price: float = 0.0
    price_move_pct: float = 0.0

    game_state_change: str = ""


@dataclass
class PositionInvalidation:
    """Record of a position that needs action."""

    ticker: str
    position_size: int
    entry_price: float
    current_price: float
    unrealized_pnl: float

    fixture_id: int
    outcome: str  # "home_win", "draw", "away_win"

    reasons: list[InvalidationReason]
    recommended_exit_pct: float  # 0.0 to 1.0 (what % to exit)
    urgency: str  # "low", "medium", "high", "immediate"


class PositionInvalidator:
    """Monitors positions and exits when game state invalidates original thesis.

    Key scenarios:
    1. Game State Change: Team falls behind, changes win probability
    2. Time Decay: Game progressing, less time for comeback
    3. Market Movement: Price moved significantly against us
    4. Model Confidence Loss: New information reduces model confidence
    5. Adverse Score: Specific game events (early goal, red card, etc.)
    """

    def __init__(
        self,
        db: Database,
        kalshi_client: KalshiClient | None,
        simulator: PaperTradingSimulator,
        live_game_provider: LiveGameStateProvider,
        dry_run: bool = False,
    ):
        """Initialize position invalidator.

        Args:
            db: Database connection.
            kalshi_client: Kalshi API client (for live mode).
            simulator: Paper trading simulator.
            live_game_provider: Live game state provider.
            dry_run: If True, don't execute exits.
        """
        self.db = db
        self.kalshi_client = kalshi_client
        self.simulator = simulator
        self.live_game_provider = live_game_provider
        self.dry_run = dry_run

        # Invalidation thresholds
        self.edge_degradation_threshold = 0.15  # Exit if edge drops 15%
        self.price_move_threshold = 0.25        # Exit if price moves 25% against us
        self.in_game_time_threshold_minutes = 15  # Check all positions 15min into game
        self.max_position_age_hours = 48       # Exit stale positions

    async def scan_and_invalidate_positions(self) -> list[PositionInvalidation]:
        """Scan all open positions and exit invalid ones.

        Returns:
            List of positions that were invalidated.
        """
        logger.info("scanning_positions_for_invalidation")

        invalidations = []

        # Get all open positions
        positions = self.simulator.open_positions

        for ticker, position in positions.items():
            if not position.is_open or position.quantity == 0:
                continue

            # Check if position should be invalidated
            invalidation = await self._check_position(position)

            if invalidation and invalidation.recommended_exit_pct > 0:
                invalidations.append(invalidation)

                # Execute exit if urgent
                if invalidation.urgency in ["high", "immediate"]:
                    await self._execute_invalidation_exit(invalidation)

        logger.info(
            "position_scan_complete",
            positions_checked=len(positions),
            invalidations_found=len(invalidations),
        )

        return invalidations

    async def _check_position(self, position: PaperPosition) -> PositionInvalidation | None:
        """Check if a position should be invalidated.

        Args:
            position: Position to check.

        Returns:
            PositionInvalidation if should exit, None otherwise.
        """
        reasons = []

        # Get current game state
        live_state = await self._get_live_state(position.fixture_id)

        if not live_state:
            # Can't get game state, monitor but don't exit
            return None

        # Check 1: Game has started (thesis was pre-game)
        if live_state.is_live:
            reason = await self._check_game_started(position, live_state)
            if reason:
                reasons.append(reason)

        # Check 2: Score is against our position
        if live_state.score:
            reason = self._check_adverse_score(position, live_state)
            if reason:
                reasons.append(reason)

        # Check 3: Market has moved significantly
        current_price = await self._get_current_price(position.ticker)
        if current_price:
            reason = self._check_price_movement(position, current_price)
            if reason:
                reasons.append(reason)

        # Check 4: Position is too old
        reason = self._check_position_staleness(position)
        if reason:
            reasons.append(reason)

        # Check 5: Game phase risk (closing minutes)
        if live_state.phase == GamePhase.CLOSING:
            reason = self._check_closing_minutes_risk(position, live_state)
            if reason:
                reasons.append(reason)

        if not reasons:
            return None

        # Calculate recommended exit percentage and urgency
        exit_pct, urgency = self._calculate_exit_recommendation(reasons, live_state)

        return PositionInvalidation(
            ticker=position.ticker,
            position_size=position.quantity,
            entry_price=position.average_entry_price,
            current_price=current_price or 0.0,
            unrealized_pnl=position.unrealized_pnl,
            fixture_id=position.fixture_id or 0,
            outcome=position.outcome,
            reasons=reasons,
            recommended_exit_pct=exit_pct,
            urgency=urgency,
        )

    async def _check_game_started(
        self,
        position: PaperPosition,
        live_state: LiveGameState,
    ) -> InvalidationReason | None:
        """Check if game starting invalidates pre-game position.

        Args:
            position: Position to check.
            live_state: Current game state.

        Returns:
            InvalidationReason if should exit.
        """
        if not live_state.is_live:
            return None

        # If we entered PRE-GAME and game is now LIVE, original thesis may be invalid
        # Severity depends on how much time has passed

        minutes_elapsed = 0
        if live_state.timing and live_state.timing.elapsed_minutes:
            minutes_elapsed = live_state.timing.elapsed_minutes

        # Early game (0-15 min): Monitor but don't panic exit
        if minutes_elapsed < self.in_game_time_threshold_minutes:
            return InvalidationReason(
                reason_code="GAME_STARTED_EARLY",
                reason_text=f"Game started {minutes_elapsed} minutes ago, monitoring position",
                severity="warning",
                recommended_action="monitor",
                game_state_change=f"Phase: {live_state.phase.value}, Elapsed: {minutes_elapsed}min",
            )

        # Mid-game (15-45 min): If losing money, consider exit
        if position.unrealized_pnl < -position.average_entry_price * position.quantity * 0.10:
            return InvalidationReason(
                reason_code="GAME_PROGRESSING_WITH_LOSS",
                reason_text=f"Game at {minutes_elapsed} min, position down {position.unrealized_pnl:.2f}",
                severity="critical",
                recommended_action="reduce",
                game_state_change=f"Phase: {live_state.phase.value}, PnL: {position.unrealized_pnl:.2f}",
            )

        return None

    def _check_adverse_score(
        self,
        position: PaperPosition,
        live_state: LiveGameState,
    ) -> InvalidationReason | None:
        """Check if current score is against our position.

        Args:
            position: Position to check.
            live_state: Current game state.

        Returns:
            InvalidationReason if score is adverse.
        """
        if not live_state.score:
            return None

        score_diff = live_state.score.score_diff
        is_adverse = False
        reason_text = ""

        # Check if score contradicts our position
        if position.outcome == "home_win":
            if score_diff < 0:  # Home losing
                is_adverse = True
                reason_text = f"Bought HOME WIN but home team is LOSING {live_state.score.home_score}-{live_state.score.away_score}"
            elif score_diff == 0 and live_state.timing and live_state.timing.game_progress and live_state.timing.game_progress > 0.75:
                is_adverse = True
                reason_text = f"Bought HOME WIN but game is TIED with {live_state.timing.minutes_remaining}min left"

        elif position.outcome == "away_win":
            if score_diff > 0:  # Away losing
                is_adverse = True
                reason_text = f"Bought AWAY WIN but away team is LOSING {live_state.score.home_score}-{live_state.score.away_score}"
            elif score_diff == 0 and live_state.timing and live_state.timing.game_progress and live_state.timing.game_progress > 0.75:
                is_adverse = True
                reason_text = f"Bought AWAY WIN but game is TIED with {live_state.timing.minutes_remaining}min left"

        elif position.outcome == "draw":
            if abs(score_diff) >= 2:  # Either team winning by 2+
                is_adverse = True
                reason_text = f"Bought DRAW but score is {live_state.score.home_score}-{live_state.score.away_score}"

        if not is_adverse:
            return None

        return InvalidationReason(
            reason_code="ADVERSE_SCORE",
            reason_text=reason_text,
            severity="critical",
            recommended_action="exit_all",
            game_state_change=f"Score: {live_state.score.home_score}-{live_state.score.away_score}",
        )

    def _check_price_movement(
        self,
        position: PaperPosition,
        current_price: float,
    ) -> InvalidationReason | None:
        """Check if price has moved significantly against us.

        Args:
            position: Position to check.
            current_price: Current market price.

        Returns:
            InvalidationReason if price moved too much.
        """
        entry_price = position.average_entry_price
        price_change = current_price - entry_price
        price_change_pct = price_change / entry_price if entry_price > 0 else 0

        # We bought expecting price to go UP
        # If price drops significantly, market knows something we don't
        if price_change_pct < -self.price_move_threshold:
            return InvalidationReason(
                reason_code="ADVERSE_PRICE_MOVEMENT",
                reason_text=f"Price dropped {price_change_pct:.1%} from entry (${entry_price:.2f} → ${current_price:.2f})",
                severity="critical",
                recommended_action="reduce",
                price_at_entry=entry_price,
                current_price=current_price,
                price_move_pct=price_change_pct,
            )

        return None

    def _check_position_staleness(
        self,
        position: PaperPosition,
    ) -> InvalidationReason | None:
        """Check if position has been open too long.

        Args:
            position: Position to check.

        Returns:
            InvalidationReason if position is stale.
        """
        if not position.opened_at:
            return None

        age_hours = (datetime.now(UTC) - position.opened_at).total_seconds() / 3600

        if age_hours > self.max_position_age_hours:
            return InvalidationReason(
                reason_code="STALE_POSITION",
                reason_text=f"Position has been open for {age_hours:.1f} hours (max: {self.max_position_age_hours}h)",
                severity="warning",
                recommended_action="reduce",
            )

        return None

    def _check_closing_minutes_risk(
        self,
        position: PaperPosition,
        live_state: LiveGameState,
    ) -> InvalidationReason | None:
        """Check if position is risky in closing minutes.

        Args:
            position: Position to check.
            live_state: Current game state.

        Returns:
            InvalidationReason if risky.
        """
        if not live_state.timing or not live_state.timing.minutes_remaining:
            return None

        minutes_left = live_state.timing.minutes_remaining

        # In final 5 minutes, positions become very binary
        # If we're not winning, exit to avoid 100% loss
        if minutes_left <= 5:
            if position.unrealized_pnl < 0:
                return InvalidationReason(
                    reason_code="CLOSING_MINUTES_LOSING",
                    reason_text=f"Only {minutes_left} minutes left and position is losing",
                    severity="critical",
                    recommended_action="exit_all",
                    game_state_change=f"Minutes remaining: {minutes_left}",
                )

        return None

    def _calculate_exit_recommendation(
        self,
        reasons: list[InvalidationReason],
        live_state: LiveGameState,
    ) -> tuple[float, str]:
        """Calculate what % to exit and urgency level.

        Args:
            reasons: List of invalidation reasons.
            live_state: Current game state.

        Returns:
            (exit_percentage, urgency)
        """
        if not reasons:
            return 0.0, "low"

        # Count severity levels
        critical_count = sum(1 for r in reasons if r.severity == "critical")
        warning_count = len(reasons) - critical_count

        # Determine exit percentage
        if critical_count >= 2:
            # Multiple critical issues: exit everything
            exit_pct = 1.0
            urgency = "immediate"
        elif critical_count == 1:
            # One critical issue: exit 50-75%
            if any(r.reason_code == "ADVERSE_SCORE" for r in reasons):
                # Score is wrong, exit all
                exit_pct = 1.0
                urgency = "immediate"
            else:
                exit_pct = 0.75
                urgency = "high"
        elif warning_count >= 3:
            # Multiple warnings: exit 50%
            exit_pct = 0.50
            urgency = "medium"
        else:
            # Minor issues: reduce 25%
            exit_pct = 0.25
            urgency = "low"

        # Adjust based on game phase
        if live_state.phase == GamePhase.CLOSING:
            # In final minutes, be more aggressive
            exit_pct = min(1.0, exit_pct * 1.5)
            if urgency == "low":
                urgency = "medium"
            elif urgency == "medium":
                urgency = "high"

        return exit_pct, urgency

    async def _execute_invalidation_exit(self, invalidation: PositionInvalidation):
        """Execute exit for invalidated position.

        Args:
            invalidation: Position to exit.
        """
        quantity_to_exit = int(invalidation.position_size * invalidation.recommended_exit_pct)

        if quantity_to_exit <= 0:
            return

        logger.warning(
            "executing_invalidation_exit",
            ticker=invalidation.ticker,
            quantity=quantity_to_exit,
            exit_pct=invalidation.recommended_exit_pct,
            urgency=invalidation.urgency,
            reasons=[r.reason_code for r in invalidation.reasons],
        )

        if self.dry_run:
            logger.info("dry_run_mode_no_exit", ticker=invalidation.ticker)
            return

        # Get current market price
        current_price = await self._get_current_price(invalidation.ticker)
        if not current_price:
            logger.error("cannot_exit_no_price", ticker=invalidation.ticker)
            return

        # Execute market sell order (accept current bid)
        if self.kalshi_client:
            await self._execute_live_exit(invalidation, quantity_to_exit, current_price)
        else:
            self._execute_paper_exit(invalidation, quantity_to_exit, current_price)

    async def _execute_live_exit(
        self,
        invalidation: PositionInvalidation,
        quantity: int,
        current_price: float,
    ):
        """Execute live exit via Kalshi API."""
        try:
            # Get current best bid
            orderbook = await self.kalshi_client.get_orderbook(invalidation.ticker)
            bid_price = orderbook.best_yes_bid if orderbook else current_price * 0.95

            # Place immediate sell order at bid
            order = await self.kalshi_client.place_limit_order(
                ticker=invalidation.ticker,
                side="yes",
                action="sell",
                price=bid_price,
                quantity=quantity,
            )

            logger.info(
                "invalidation_exit_executed",
                ticker=invalidation.ticker,
                order_id=order.order_id,
                quantity=quantity,
                price=bid_price,
            )

        except Exception as e:
            logger.error(
                "invalidation_exit_failed",
                ticker=invalidation.ticker,
                error=str(e),
            )

    def _execute_paper_exit(
        self,
        invalidation: PositionInvalidation,
        quantity: int,
        current_price: float,
    ):
        """Execute paper exit (simulation)."""
        # Simulate selling at current bid (apply slippage)
        exit_price = current_price * 0.98  # 2% slippage

        # Update position in simulator
        position = self.simulator.positions.get(invalidation.ticker)
        if position:
            # Reduce position
            position.quantity -= quantity

            # Calculate realized P&L
            entry_cost = invalidation.entry_price * quantity
            exit_proceeds = exit_price * quantity
            realized_pnl = exit_proceeds - entry_cost

            position.realized_pnl += realized_pnl
            self.simulator.total_realized_pnl += realized_pnl

            logger.info(
                "paper_invalidation_exit",
                ticker=invalidation.ticker,
                quantity=quantity,
                entry_price=invalidation.entry_price,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
            )

    async def _get_live_state(self, fixture_id: int) -> LiveGameState | None:
        """Get live state for a fixture."""
        if not self.live_game_provider:
            return None

        try:
            # Try football first
            state = await self.live_game_provider.get_football_game_state(fixture_id)
            if state and state.is_tradeable:
                return state

            # Try NBA (fixture_id > 1 billion for NBA)
            if fixture_id > 1_000_000_000:
                state = await self.live_game_provider.get_nba_game_state(fixture_id)
                return state

        except Exception as e:
            logger.warning("failed_to_get_live_state", fixture_id=fixture_id, error=str(e))

        return None

    async def _get_current_price(self, ticker: str) -> float | None:
        """Get current market price for a ticker."""
        if self.kalshi_client:
            try:
                orderbook = await self.kalshi_client.get_orderbook(ticker)
                return orderbook.mid_price if orderbook else None
            except Exception:
                return None
        else:
            # Get from database for paper trading
            cursor = self.db.connection.cursor()
            cursor.execute(
                "SELECT yes_bid, yes_ask FROM kalshi_markets WHERE ticker = ?",
                (ticker,),
            )
            row = cursor.fetchone()
            if row and row["yes_bid"] and row["yes_ask"]:
                return (row["yes_bid"] + row["yes_ask"]) / 2
            return None
