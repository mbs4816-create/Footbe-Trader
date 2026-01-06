"""Trading Strategy v1: Edge-based with fractional Kelly sizing.

This module implements the core trading strategy that:
1. Computes edge as model probability minus market ask price
2. Filters trades by edge threshold and liquidity requirements
3. Sizes positions using fractional Kelly criterion with hard caps
4. Manages exits based on take profit, stop loss, and edge reversal
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from footbe_trader.common.logging import get_logger
from footbe_trader.kalshi.interfaces import MarketData, OrderbookData, OrderData
from footbe_trader.strategy.decision_record import (
    AgentRunSummary,
    DecisionAction,
    DecisionRecord,
    EdgeCalculation,
    ExitReason,
    KellySizing,
    MarketSnapshot,
    ModelPrediction,
    OrderParams,
    PositionState,
)
from footbe_trader.strategy.interfaces import IStrategy, Signal, StrategyContext
from footbe_trader.strategy.mapping import FixtureMarketMapping

logger = get_logger(__name__)

# Default config path
DEFAULT_STRATEGY_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "strategy_config.yaml"
)


@dataclass
class StrategyConfig:
    """Configuration for EdgeStrategy."""

    # Edge thresholds
    min_edge_to_enter: float = 0.05
    exit_edge_buffer: float = -0.01
    min_model_confidence: float = 0.6

    # Liquidity requirements
    min_ask_volume: int = 10
    min_bid_volume: int = 10
    max_spread: float = 0.10

    # Kelly parameters
    kelly_fraction: float = 0.25
    max_kelly_fraction: float = 0.10

    # Position limits
    max_position_per_market: int = 100
    max_exposure_per_fixture: float = 500.0
    max_global_exposure: float = 2000.0
    initial_bankroll: float = 10000.0

    # Exit rules
    take_profit: float = 0.15
    stop_loss: float = 0.20
    close_before_expiry_hours: float = 1.0

    # Market filtering
    allowed_statuses: list[str] = field(default_factory=lambda: ["open"])
    min_hours_to_close: float = 2.0
    max_hours_to_close: float = 168.0
    
    # In-game trading safeguards
    # Don't place new orders if game has started (kickoff has passed)
    require_pre_game: bool = True
    # Minimum minutes before kickoff to place orders
    min_minutes_before_kickoff: int = 5
    # Maximum allowed deviation from our model price to market price
    # If market ask is > model_prob + max_price_deviation, skip (price too high)
    # If market ask is < model_prob - max_price_deviation, flag as suspicious (in-game swing)
    max_price_deviation_to_enter: float = 0.30  # 30 cents / 30%
    # Cancel resting orders if price has moved this much from our limit
    stale_order_threshold: float = 0.20  # 20 cents

    # Paper trading
    slippage_cents: float = 0.01
    fill_probability: float = 0.8
    partial_fill_range: tuple[float, float] = (0.5, 1.0)

    # Logging
    log_all_decisions: bool = True
    include_orderbook_snapshot: bool = True
    include_features: bool = True

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "StrategyConfig":
        """Load configuration from YAML file."""
        path = Path(path) if path else DEFAULT_STRATEGY_CONFIG_PATH

        if not path.exists():
            logger.warning("strategy_config_not_found", path=str(path))
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            min_edge_to_enter=data.get("edge", {}).get("min_edge_to_enter", 0.05),
            exit_edge_buffer=data.get("edge", {}).get("exit_edge_buffer", -0.01),
            min_model_confidence=data.get("edge", {}).get("min_model_confidence", 0.6),
            min_ask_volume=data.get("liquidity", {}).get("min_ask_volume", 10),
            min_bid_volume=data.get("liquidity", {}).get("min_bid_volume", 10),
            max_spread=data.get("liquidity", {}).get("max_spread", 0.10),
            kelly_fraction=data.get("kelly", {}).get("fraction", 0.25),
            max_kelly_fraction=data.get("kelly", {}).get("max_kelly_fraction", 0.10),
            max_position_per_market=data.get("limits", {}).get(
                "max_position_per_market", 100
            ),
            max_exposure_per_fixture=data.get("limits", {}).get(
                "max_exposure_per_fixture", 500.0
            ),
            max_global_exposure=data.get("limits", {}).get("max_global_exposure", 2000.0),
            initial_bankroll=data.get("limits", {}).get("initial_bankroll", 10000.0),
            take_profit=data.get("exit_rules", {}).get("take_profit", 0.15),
            stop_loss=data.get("exit_rules", {}).get("stop_loss", 0.20),
            close_before_expiry_hours=data.get("exit_rules", {}).get(
                "close_before_expiry_hours", 1.0
            ),
            allowed_statuses=data.get("market_filter", {}).get(
                "allowed_statuses", ["open"]
            ),
            min_hours_to_close=data.get("market_filter", {}).get("min_hours_to_close", 2.0),
            max_hours_to_close=data.get("market_filter", {}).get(
                "max_hours_to_close", 168.0
            ),
            # In-game trading safeguards
            require_pre_game=data.get("in_game", {}).get("require_pre_game", True),
            min_minutes_before_kickoff=data.get("in_game", {}).get(
                "min_minutes_before_kickoff", 5
            ),
            max_price_deviation_to_enter=data.get("in_game", {}).get(
                "max_price_deviation_to_enter", 0.30
            ),
            stale_order_threshold=data.get("in_game", {}).get(
                "stale_order_threshold", 0.20
            ),
            slippage_cents=data.get("paper_trading", {}).get("slippage_cents", 0.01),
            fill_probability=data.get("paper_trading", {}).get("fill_probability", 0.8),
            partial_fill_range=tuple(
                data.get("paper_trading", {}).get("partial_fill_range", [0.5, 1.0])
            ),
            log_all_decisions=data.get("logging", {}).get("log_all_decisions", True),
            include_orderbook_snapshot=data.get("logging", {}).get(
                "include_orderbook_snapshot", True
            ),
            include_features=data.get("logging", {}).get("include_features", True),
        )

    def config_hash(self) -> str:
        """Generate hash of configuration for tracking."""
        config_str = str(self.__dict__)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_edge_to_enter": self.min_edge_to_enter,
            "exit_edge_buffer": self.exit_edge_buffer,
            "kelly_fraction": self.kelly_fraction,
            "max_position_per_market": self.max_position_per_market,
            "max_exposure_per_fixture": self.max_exposure_per_fixture,
            "max_global_exposure": self.max_global_exposure,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
        }


@dataclass
class FixtureContext:
    """Context for evaluating a fixture."""

    fixture_id: int
    home_team: str
    away_team: str
    kickoff_time: datetime
    league: str
    mapping: FixtureMarketMapping
    current_exposure: float = 0.0


@dataclass
class OutcomeContext:
    """Context for evaluating a specific outcome market."""

    outcome: str  # "home_win", "draw", "away_win"
    ticker: str
    market_data: MarketData | None = None
    orderbook: OrderbookData | None = None
    model_prob: float = 0.0
    current_position: int = 0
    average_entry_price: float = 0.0


class EdgeStrategy(IStrategy):
    """Edge-based trading strategy with Kelly sizing.

    This strategy:
    1. Computes edge for each outcome (home_win, draw, away_win)
    2. Filters based on edge threshold and liquidity
    3. Sizes positions using fractional Kelly
    4. Manages exits based on TP/SL and edge reversal
    """

    def __init__(self, config: StrategyConfig | None = None):
        """Initialize strategy.

        Args:
            config: Strategy configuration. If None, loads from default path.
        """
        self.config = config or StrategyConfig.from_yaml()
        self._decision_records: list[DecisionRecord] = []
        self._current_run_id: int | None = None

    @property
    def name(self) -> str:
        """Strategy name."""
        return "edge_strategy_v1"

    @property
    def version(self) -> str:
        """Strategy version."""
        return "1.0.0"

    def set_run_id(self, run_id: int | None) -> None:
        """Set current run ID for decision records."""
        self._current_run_id = run_id

    def get_decision_records(self) -> list[DecisionRecord]:
        """Get accumulated decision records."""
        return self._decision_records.copy()

    def clear_decision_records(self) -> None:
        """Clear accumulated decision records."""
        self._decision_records.clear()

    def generate_signals(self, context: StrategyContext) -> list[Signal]:
        """Generate trading signals from strategy context.

        This is the IStrategy interface method. For more detailed control,
        use evaluate_fixture() and evaluate_exit() directly.
        """
        signals: list[Signal] = []

        for prediction in context.predictions:
            if prediction.prediction_type == "home_win_prob":
                edge = prediction.value - (1 - context.market.yes_ask)
            elif prediction.prediction_type == "away_win_prob":
                edge = prediction.value - context.market.yes_ask
            else:
                continue

            if edge >= self.config.min_edge_to_enter:
                signals.append(
                    Signal(
                        market_id=context.market.ticker,
                        side="yes",
                        action="buy",
                        target_price=context.market.yes_ask,
                        quantity=min(10, context.max_position - context.current_position),
                        edge=edge,
                        confidence=prediction.confidence,
                        reason=f"Edge {edge:.2%} above threshold",
                    )
                )

        return signals

    def should_trade(self, context: StrategyContext) -> bool:
        """Check if conditions are suitable for trading."""
        market = context.market

        # Check market status
        if market.status not in self.config.allowed_statuses:
            return False

        # Check spread
        if market.yes_ask > 0 and market.yes_bid > 0:
            spread = market.yes_ask - market.yes_bid
            if spread > self.config.max_spread:
                return False

        return True

    # --- Core Strategy Methods ---

    def evaluate_fixture(
        self,
        fixture: FixtureContext,
        outcomes: list[OutcomeContext],
        model_prediction: ModelPrediction,
        global_exposure: float,
        bankroll: float | None = None,
    ) -> list[DecisionRecord]:
        """Evaluate a fixture and generate entry decisions for all outcomes.

        Args:
            fixture: Fixture context with mapping information.
            outcomes: List of outcome contexts with market data.
            model_prediction: Model probabilities for the fixture.
            global_exposure: Current total exposure across all positions.
            bankroll: Current bankroll (uses config.initial_bankroll if None).

        Returns:
            List of decision records for each outcome.
        """
        bankroll = bankroll or self.config.initial_bankroll
        decisions: list[DecisionRecord] = []

        for outcome_ctx in outcomes:
            decision = self._evaluate_entry(
                fixture=fixture,
                outcome=outcome_ctx,
                model_prediction=model_prediction,
                global_exposure=global_exposure,
                bankroll=bankroll,
            )
            decisions.append(decision)
            self._decision_records.append(decision)

            # Update exposure tracking if we're entering
            if decision.action == DecisionAction.BUY and decision.order_params:
                entry_cost = decision.order_params.price * decision.order_params.quantity
                global_exposure += entry_cost
                fixture.current_exposure += entry_cost

        return decisions

    def evaluate_exit(
        self,
        outcome: OutcomeContext,
        model_prediction: ModelPrediction,
        entry_price: float,
        fixture_id: int,
    ) -> DecisionRecord:
        """Evaluate whether to exit an existing position.

        Args:
            outcome: Outcome context with current market data.
            model_prediction: Current model probabilities.
            entry_price: Average entry price for the position.
            fixture_id: Fixture ID for the position.

        Returns:
            Decision record with exit decision.
        """
        decision = self._evaluate_position_exit(
            outcome=outcome,
            model_prediction=model_prediction,
            entry_price=entry_price,
            fixture_id=fixture_id,
        )

        self._decision_records.append(decision)
        return decision

    # --- Entry Evaluation ---

    def _evaluate_entry(
        self,
        fixture: FixtureContext,
        outcome: OutcomeContext,
        model_prediction: ModelPrediction,
        global_exposure: float,
        bankroll: float,
    ) -> DecisionRecord:
        """Evaluate entry for a single outcome market."""
        decision = DecisionRecord(
            decision_id=str(uuid.uuid4()),
            run_id=self._current_run_id,
            fixture_id=fixture.fixture_id,
            market_ticker=outcome.ticker,
            outcome=outcome.outcome,
            timestamp=datetime.now(UTC),
            model_prediction=model_prediction,
        )

        # Get model probability for this outcome
        if outcome.outcome == "home_win":
            model_prob = model_prediction.prob_home_win
        elif outcome.outcome == "draw":
            model_prob = model_prediction.prob_draw
        elif outcome.outcome == "away_win":
            model_prob = model_prediction.prob_away_win
        else:
            decision.action = DecisionAction.SKIP
            decision.rejection_reason = f"Unknown outcome: {outcome.outcome}"
            return decision

        outcome.model_prob = model_prob

        # Build market snapshot
        if outcome.orderbook:
            decision.market_snapshot = MarketSnapshot(
                ticker=outcome.ticker,
                best_bid=outcome.orderbook.best_yes_bid,
                best_ask=outcome.orderbook.best_yes_ask,
                mid_price=outcome.orderbook.mid_price,
                spread=outcome.orderbook.spread,
                bid_volume=outcome.orderbook.total_bid_volume,
                ask_volume=outcome.orderbook.total_ask_volume,
                status=outcome.market_data.status if outcome.market_data else "unknown",
                close_time=outcome.market_data.close_time if outcome.market_data else None,
            )

        # Build position state
        if outcome.current_position != 0:
            decision.current_position = PositionState(
                ticker=outcome.ticker,
                quantity=outcome.current_position,
                average_entry_price=outcome.average_entry_price,
            )

        # Run filters
        filters_passed = self._check_entry_filters(
            outcome=outcome,
            fixture=fixture,
            model_prediction=model_prediction,
            global_exposure=global_exposure,
        )
        decision.filters_passed = filters_passed

        if not all(filters_passed.values()):
            failed_filters = [k for k, v in filters_passed.items() if not v]
            decision.action = DecisionAction.SKIP
            decision.rejection_reason = f"Failed filters: {', '.join(failed_filters)}"
            decision.rationale = (
                f"Skipping {outcome.outcome} for {fixture.home_team} vs {fixture.away_team}: "
                f"{decision.rejection_reason}"
            )
            return decision

        # Calculate edge
        ask_price = outcome.orderbook.best_yes_ask if outcome.orderbook else 0.0
        if ask_price <= 0:
            decision.action = DecisionAction.SKIP
            decision.rejection_reason = "No valid ask price"
            return decision

        edge = model_prob - ask_price
        decision.edge_calculation = EdgeCalculation(
            outcome=outcome.outcome,
            model_prob=model_prob,
            market_price=ask_price,
            edge=edge,
            is_tradeable=edge >= self.config.min_edge_to_enter,
        )

        if edge < self.config.min_edge_to_enter:
            decision.action = DecisionAction.SKIP
            decision.rejection_reason = (
                f"Edge {edge:.2%} below threshold {self.config.min_edge_to_enter:.2%}"
            )
            decision.rationale = (
                f"No trade on {outcome.outcome}: model_prob={model_prob:.2%}, "
                f"ask={ask_price:.2f}, edge={edge:.2%}"
            )
            return decision

        # Calculate Kelly sizing
        kelly_sizing = self._calculate_kelly_sizing(
            edge=edge,
            win_prob=model_prob,
            ask_price=ask_price,
            bankroll=bankroll,
            fixture=fixture,
            outcome=outcome,
            global_exposure=global_exposure,
        )
        decision.kelly_sizing = kelly_sizing

        if kelly_sizing.position_size <= 0:
            decision.action = DecisionAction.SKIP
            decision.rejection_reason = "Position size is zero after limits"
            return decision

        # Build order
        decision.action = DecisionAction.BUY
        decision.order_params = OrderParams(
            ticker=outcome.ticker,
            side="yes",
            action="buy",
            order_type="limit",
            price=ask_price,
            quantity=kelly_sizing.position_size,
        )
        decision.rationale = (
            f"Buying {kelly_sizing.position_size} contracts of {outcome.outcome} "
            f"at {ask_price:.2f}. Edge: {edge:.2%}, "
            f"Model: {model_prob:.2%}, Kelly: {kelly_sizing.kelly_fraction:.2%}"
        )

        return decision

    def _check_entry_filters(
        self,
        outcome: OutcomeContext,
        fixture: FixtureContext,
        model_prediction: ModelPrediction,
        global_exposure: float,
    ) -> dict[str, bool]:
        """Check all entry filters and return results."""
        filters: dict[str, bool] = {}

        # Market status
        if outcome.market_data:
            filters["market_status"] = (
                outcome.market_data.status in self.config.allowed_statuses
            )
        else:
            filters["market_status"] = False

        # Market timing
        if outcome.market_data and outcome.market_data.close_time:
            hours_to_close = (
                outcome.market_data.close_time - datetime.now(UTC)
            ).total_seconds() / 3600
            filters["min_hours_to_close"] = hours_to_close >= self.config.min_hours_to_close
            filters["max_hours_to_close"] = hours_to_close <= self.config.max_hours_to_close
        else:
            filters["min_hours_to_close"] = False
            filters["max_hours_to_close"] = False

        # Orderbook exists
        filters["has_orderbook"] = outcome.orderbook is not None

        if outcome.orderbook:
            # Liquidity
            filters["min_ask_volume"] = (
                outcome.orderbook.total_ask_volume >= self.config.min_ask_volume
            )

            # Spread
            if outcome.orderbook.spread is not None:
                filters["max_spread"] = outcome.orderbook.spread <= self.config.max_spread
            else:
                filters["max_spread"] = False

            # Valid prices
            filters["valid_prices"] = (
                outcome.orderbook.best_yes_ask is not None
                and outcome.orderbook.best_yes_ask > 0
            )
        else:
            filters["min_ask_volume"] = False
            filters["max_spread"] = False
            filters["valid_prices"] = False

        # Model confidence
        filters["min_confidence"] = (
            model_prediction.confidence >= self.config.min_model_confidence
        )

        # Global exposure limit
        filters["global_exposure"] = global_exposure < self.config.max_global_exposure

        # Fixture exposure limit
        filters["fixture_exposure"] = (
            fixture.current_exposure < self.config.max_exposure_per_fixture
        )
        
        # --- In-game trading safeguards ---
        
        # Pre-game requirement: game must not have started
        if self.config.require_pre_game:
            minutes_to_kickoff = (
                fixture.kickoff_time - datetime.now(UTC)
            ).total_seconds() / 60
            filters["pre_game"] = minutes_to_kickoff >= self.config.min_minutes_before_kickoff
        else:
            filters["pre_game"] = True
        
        # Price deviation check: detect suspicious prices that suggest in-game movement
        # If market ask is much lower than model probability, the game state may have changed
        if outcome.orderbook and outcome.orderbook.best_yes_ask:
            ask_price = outcome.orderbook.best_yes_ask
            
            # Get model probability for this outcome
            if outcome.outcome == "home_win":
                model_prob = model_prediction.prob_home_win
            elif outcome.outcome == "draw":
                model_prob = model_prediction.prob_draw
            elif outcome.outcome == "away_win":
                model_prob = model_prediction.prob_away_win
            else:
                model_prob = 0.0
            
            # Price deviation = how much cheaper the ask is than model expects
            # Positive deviation = ask is lower than model (potential in-game move)
            price_deviation = model_prob - ask_price
            
            # If ask is much lower than model, this could indicate:
            # 1. The team is losing/behind in the game
            # 2. News/injury that invalidates our model
            # Either way, we should NOT buy at this "cheap" price
            filters["price_deviation"] = price_deviation <= self.config.max_price_deviation_to_enter
        else:
            filters["price_deviation"] = True

        return filters

    # --- Kelly Sizing ---

    def _calculate_kelly_sizing(
        self,
        edge: float,
        win_prob: float,
        ask_price: float,
        bankroll: float,
        fixture: FixtureContext,
        outcome: OutcomeContext,
        global_exposure: float,
    ) -> KellySizing:
        """Calculate Kelly criterion position sizing with caps."""
        # Calculate odds (profit ratio if we win)
        # If we buy at ask_price and win, we get $1 back, profit = 1 - ask_price
        # Odds = profit / stake = (1 - ask_price) / ask_price
        if ask_price <= 0 or ask_price >= 1:
            return KellySizing(
                edge=edge,
                win_prob=win_prob,
                odds=0,
                kelly_fraction=0,
                adjusted_fraction=0,
                position_size=0,
                capped_reason="Invalid ask price",
            )

        odds = (1 - ask_price) / ask_price

        # Kelly formula: f* = (p * b - q) / b
        # where p = win prob, q = 1-p, b = odds
        # Simplified: f* = p - q/b = p - (1-p)/b
        if odds <= 0:
            return KellySizing(
                edge=edge,
                win_prob=win_prob,
                odds=odds,
                kelly_fraction=0,
                adjusted_fraction=0,
                position_size=0,
                capped_reason="Non-positive odds",
            )

        kelly_fraction = win_prob - (1 - win_prob) / odds

        # Apply fractional Kelly
        adjusted_fraction = kelly_fraction * self.config.kelly_fraction

        # Cap at max Kelly fraction
        capped_reason: str | None = None
        if adjusted_fraction > self.config.max_kelly_fraction:
            adjusted_fraction = self.config.max_kelly_fraction
            capped_reason = f"Capped at max Kelly fraction ({self.config.max_kelly_fraction})"

        # Calculate dollar amount to bet
        bet_amount = bankroll * adjusted_fraction

        # Convert to contracts (shares)
        # Each contract costs ask_price dollars
        raw_position = int(bet_amount / ask_price)

        # Apply position limits
        position_size = raw_position

        # Per-market limit
        if position_size > self.config.max_position_per_market:
            position_size = self.config.max_position_per_market
            capped_reason = f"Capped at max position per market ({self.config.max_position_per_market})"

        # Per-fixture exposure limit
        entry_cost = position_size * ask_price
        remaining_fixture_exposure = (
            self.config.max_exposure_per_fixture - fixture.current_exposure
        )
        if entry_cost > remaining_fixture_exposure:
            position_size = int(remaining_fixture_exposure / ask_price)
            capped_reason = f"Capped by fixture exposure limit"

        # Global exposure limit
        remaining_global = self.config.max_global_exposure - global_exposure
        if entry_cost > remaining_global:
            position_size = int(remaining_global / ask_price)
            capped_reason = f"Capped by global exposure limit"

        # Ensure positive
        position_size = max(0, position_size)

        return KellySizing(
            edge=edge,
            win_prob=win_prob,
            odds=odds,
            kelly_fraction=kelly_fraction,
            adjusted_fraction=adjusted_fraction,
            position_size=position_size,
            capped_reason=capped_reason,
        )

    # --- Exit Evaluation ---

    def _evaluate_position_exit(
        self,
        outcome: OutcomeContext,
        model_prediction: ModelPrediction,
        entry_price: float,
        fixture_id: int,
    ) -> DecisionRecord:
        """Evaluate whether to exit a position."""
        decision = DecisionRecord(
            decision_id=str(uuid.uuid4()),
            run_id=self._current_run_id,
            fixture_id=fixture_id,
            market_ticker=outcome.ticker,
            outcome=outcome.outcome,
            timestamp=datetime.now(UTC),
            model_prediction=model_prediction,
        )

        # Build market snapshot
        if outcome.orderbook:
            decision.market_snapshot = MarketSnapshot(
                ticker=outcome.ticker,
                best_bid=outcome.orderbook.best_yes_bid,
                best_ask=outcome.orderbook.best_yes_ask,
                mid_price=outcome.orderbook.mid_price,
                spread=outcome.orderbook.spread,
                bid_volume=outcome.orderbook.total_bid_volume,
                ask_volume=outcome.orderbook.total_ask_volume,
                status=outcome.market_data.status if outcome.market_data else "unknown",
                close_time=outcome.market_data.close_time if outcome.market_data else None,
            )

        decision.current_position = PositionState(
            ticker=outcome.ticker,
            quantity=outcome.current_position,
            average_entry_price=entry_price,
        )

        # No position to exit
        if outcome.current_position <= 0:
            decision.action = DecisionAction.HOLD
            decision.rationale = "No position to exit"
            return decision

        # Get current mid price
        mid_price = outcome.orderbook.mid_price if outcome.orderbook else None
        if mid_price is None:
            decision.action = DecisionAction.HOLD
            decision.rationale = "No mid price available for exit evaluation"
            return decision

        # Check take profit
        if mid_price >= entry_price + self.config.take_profit:
            decision.action = DecisionAction.EXIT
            decision.exit_reason = ExitReason.TAKE_PROFIT
            bid_price = outcome.orderbook.best_yes_bid if outcome.orderbook else mid_price
            decision.order_params = OrderParams(
                ticker=outcome.ticker,
                side="yes",
                action="sell",
                order_type="limit",
                price=bid_price or mid_price,
                quantity=outcome.current_position,
            )
            decision.rationale = (
                f"Take profit triggered: mid={mid_price:.2f}, "
                f"entry={entry_price:.2f}, TP threshold={entry_price + self.config.take_profit:.2f}"
            )
            return decision

        # Check stop loss
        if mid_price <= entry_price - self.config.stop_loss:
            decision.action = DecisionAction.EXIT
            decision.exit_reason = ExitReason.STOP_LOSS
            bid_price = outcome.orderbook.best_yes_bid if outcome.orderbook else mid_price
            decision.order_params = OrderParams(
                ticker=outcome.ticker,
                side="yes",
                action="sell",
                order_type="limit",
                price=bid_price or mid_price,
                quantity=outcome.current_position,
            )
            decision.rationale = (
                f"Stop loss triggered: mid={mid_price:.2f}, "
                f"entry={entry_price:.2f}, SL threshold={entry_price - self.config.stop_loss:.2f}"
            )
            return decision

        # Check edge flip (model probability dropped)
        if outcome.outcome == "home_win":
            model_prob = model_prediction.prob_home_win
        elif outcome.outcome == "draw":
            model_prob = model_prediction.prob_draw
        elif outcome.outcome == "away_win":
            model_prob = model_prediction.prob_away_win
        else:
            model_prob = 0.0

        # Current edge based on mid price
        current_edge = model_prob - mid_price

        decision.edge_calculation = EdgeCalculation(
            outcome=outcome.outcome,
            model_prob=model_prob,
            market_price=mid_price,
            edge=current_edge,
            is_tradeable=False,
        )

        if current_edge < self.config.exit_edge_buffer:
            decision.action = DecisionAction.EXIT
            decision.exit_reason = ExitReason.EDGE_FLIP
            bid_price = outcome.orderbook.best_yes_bid if outcome.orderbook else mid_price
            decision.order_params = OrderParams(
                ticker=outcome.ticker,
                side="yes",
                action="sell",
                order_type="limit",
                price=bid_price or mid_price,
                quantity=outcome.current_position,
            )
            decision.rationale = (
                f"Edge flip exit: current_edge={current_edge:.2%}, "
                f"buffer={self.config.exit_edge_buffer:.2%}, model_prob={model_prob:.2%}"
            )
            return decision

        # Check time-based exit (approaching market close)
        if outcome.market_data and outcome.market_data.close_time:
            hours_to_close = (
                outcome.market_data.close_time - datetime.now(UTC)
            ).total_seconds() / 3600

            if hours_to_close <= self.config.close_before_expiry_hours:
                decision.action = DecisionAction.EXIT
                decision.exit_reason = ExitReason.MARKET_CLOSE
                bid_price = outcome.orderbook.best_yes_bid if outcome.orderbook else mid_price
                decision.order_params = OrderParams(
                    ticker=outcome.ticker,
                    side="yes",
                    action="sell",
                    order_type="limit",
                    price=bid_price or mid_price,
                    quantity=outcome.current_position,
                )
                decision.rationale = (
                    f"Time-based exit: {hours_to_close:.1f}h to close, "
                    f"threshold={self.config.close_before_expiry_hours:.1f}h"
                )
                return decision

        # Hold position
        decision.action = DecisionAction.HOLD
        decision.rationale = (
            f"Holding position: mid={mid_price:.2f}, entry={entry_price:.2f}, "
            f"edge={current_edge:.2%}, TP={entry_price + self.config.take_profit:.2f}, "
            f"SL={entry_price - self.config.stop_loss:.2f}"
        )
        return decision


@dataclass
class StaleOrderInfo:
    """Information about a stale resting order that should be cancelled."""
    
    order_id: str
    ticker: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    limit_price: float
    current_market_price: float
    price_divergence: float  # How far market has moved from our limit
    reason: str  # Why this order is considered stale


class StaleOrderDetector:
    """Detects stale resting orders that should be cancelled.
    
    Orders become stale when the market price has moved significantly
    away from our limit price. This typically indicates:
    1. In-game price movement (team falling behind)
    2. News/injury that invalidates our model
    3. Market conditions have changed
    
    In these cases, holding onto the order means we'd only get filled
    when conditions are bad for us.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize detector.
        
        Args:
            config: Strategy configuration with stale_order_threshold.
        """
        self.config = config
    
    def check_order(
        self,
        order: "OrderData",
        current_ask: float,
        current_bid: float,
        kickoff_time: datetime | None = None,
    ) -> StaleOrderInfo | None:
        """Check if a resting order is stale and should be cancelled.
        
        Args:
            order: The resting order to check.
            current_ask: Current best ask price.
            current_bid: Current best bid price.
            kickoff_time: When the game starts (if known).
            
        Returns:
            StaleOrderInfo if order should be cancelled, None otherwise.
        """
        # Only check resting buy orders (sell orders are exits, different logic)
        if order.action != "buy" or order.status != "resting":
            return None
        
        # For buy YES orders: our limit should be near current ask
        # If current ask is much higher than our limit, market moved against us
        if order.side == "yes":
            current_market_price = current_ask
            price_divergence = current_market_price - order.price
        else:
            # For buy NO orders: our limit should be near current NO ask
            # NO ask = 1 - YES bid, so if YES bid dropped, NO ask increased
            current_market_price = current_bid  # YES bid is relevant for NO trades
            price_divergence = (1 - current_market_price) - order.price
        
        # Check if price has diverged beyond threshold
        if price_divergence > self.config.stale_order_threshold:
            reason = (
                f"Market moved {price_divergence:.0%} away from limit. "
                f"Limit: ${order.price:.2f}, Current market: ${current_market_price:.2f}"
            )
            return StaleOrderInfo(
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                action=order.action,
                limit_price=order.price,
                current_market_price=current_market_price,
                price_divergence=price_divergence,
                reason=reason,
            )
        
        # Check if game has started (if we have kickoff time)
        if kickoff_time and self.config.require_pre_game:
            now = datetime.now(UTC)
            if now >= kickoff_time:
                reason = (
                    f"Game has started (kickoff: {kickoff_time.isoformat()}). "
                    f"Pre-match orders are no longer valid."
                )
                return StaleOrderInfo(
                    order_id=order.order_id,
                    ticker=order.ticker,
                    side=order.side,
                    action=order.action,
                    limit_price=order.price,
                    current_market_price=current_market_price,
                    price_divergence=price_divergence,
                    reason=reason,
                )
        
        return None


def create_agent_run_summary(
    run_type: str = "paper",
    config: StrategyConfig | None = None,
) -> AgentRunSummary:
    """Create a new agent run summary."""
    config = config or StrategyConfig.from_yaml()
    return AgentRunSummary(
        run_type=run_type,
        config_hash=config.config_hash(),
        config_summary=config.to_dict(),
    )
