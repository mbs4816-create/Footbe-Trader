"""Multi-Armed Bandit for Strategy Selection.

Uses Thompson Sampling to dynamically select between multiple trading strategies,
balancing exploration (trying uncertain strategies) with exploitation (using
known good strategies).

This enables the agent to:
1. Test multiple strategies simultaneously
2. Adapt to changing market conditions
3. Avoid commitment to suboptimal strategies
4. Continuously learn which approach works best
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from footbe_trader.common.logging import get_logger
from footbe_trader.strategy.trading_strategy import EdgeStrategy, FixtureContext, StrategyConfig
from footbe_trader.storage.database import Database

logger = get_logger(__name__)


@dataclass
class StrategyArm:
    """One arm of the multi-armed bandit."""

    name: str
    description: str
    config: StrategyConfig
    strategy: EdgeStrategy

    # Beta distribution parameters (Bayesian success/failure tracking)
    alpha: float = 1.0  # Success count (+ prior)
    beta: float = 1.0   # Failure count (+ prior)

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    total_exposure: float = 0.0
    avg_sharpe: float = 0.0

    # Usage stats
    times_selected: int = 0
    last_selected: datetime | None = None

    def sample_reward_probability(self) -> float:
        """Sample from posterior Beta distribution."""
        return np.random.beta(self.alpha, self.beta)

    def expected_reward(self) -> float:
        """Expected reward (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    def update_success(self, pnl: float, exposure: float):
        """Update after a successful trade."""
        self.alpha += 1
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        self.total_pnl += pnl
        self.total_exposure += exposure

    def update_failure(self, pnl: float, exposure: float):
        """Update after a failed trade."""
        self.beta += 1
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_exposure += exposure

    def calculate_sharpe(self, recent_pnls: list[float]) -> float:
        """Calculate rolling Sharpe ratio from recent P&Ls."""
        if len(recent_pnls) < 2:
            return 0.0

        returns = np.array(recent_pnls)
        if returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(252) * (returns.mean() / returns.std())
        return float(sharpe)


@dataclass
class StrategySelection:
    """Record of strategy selection and outcome."""

    selection_id: str
    timestamp: datetime
    fixture_id: int
    strategy_name: str
    sampled_probability: float
    actual_pnl: float = 0.0
    actual_exposure: float = 0.0
    outcome: str = "pending"  # pending, success, failure


class StrategyBandit:
    """Multi-armed bandit for dynamic strategy selection.

    Implements Thompson Sampling:
    1. Each strategy has a Beta(α, β) belief about success probability
    2. For each decision, sample from each strategy's posterior
    3. Select strategy with highest sampled probability
    4. Update beliefs based on observed outcome
    """

    def __init__(
        self,
        db: Database,
        enable_thompson_sampling: bool = True,
        exploration_bonus: float = 0.0,
    ):
        """Initialize strategy bandit.

        Args:
            db: Database connection.
            enable_thompson_sampling: If False, use epsilon-greedy instead.
            exploration_bonus: Bonus for less-tried strategies (UCB-style).
        """
        self.db = db
        self.enable_thompson_sampling = enable_thompson_sampling
        self.exploration_bonus = exploration_bonus

        # Initialize strategy arms
        self.arms: dict[str, StrategyArm] = {}
        self._initialize_strategies()

        # Selection history
        self.selection_history: list[StrategySelection] = []

        # Performance window for Sharpe calculation
        self.performance_window_days = 7

    def _initialize_strategies(self):
        """Initialize multiple strategy configurations."""

        # Strategy 1: Ultra Aggressive (target 10-12% daily)
        ultra_aggressive_config = StrategyConfig(
            min_edge_to_enter=0.02,       # Take 2%+ edges
            exit_edge_buffer=-0.03,       # Hold until -3% edge
            min_model_confidence=0.55,    # Lower confidence threshold
            kelly_fraction=0.50,          # 50% Kelly (high risk)
            max_kelly_fraction=0.25,      # Allow 25% of bankroll per trade
            max_position_per_market=50,   # Larger positions
            max_exposure_per_fixture=250, # More per fixture
            max_global_exposure=1000,     # Much higher total exposure
            take_profit=0.08,             # Take profit at 8 cents
            stop_loss=0.15,               # Tighter stop loss
            require_pre_game=True,
            min_minutes_before_kickoff=3,
            max_price_deviation_to_enter=0.35,
        )

        # Strategy 2: Aggressive (high volume, moderate risk)
        aggressive_config = StrategyConfig(
            min_edge_to_enter=0.03,
            exit_edge_buffer=-0.02,
            min_model_confidence=0.58,
            kelly_fraction=0.35,
            max_kelly_fraction=0.15,
            max_position_per_market=30,
            max_exposure_per_fixture=150,
            max_global_exposure=600,
            take_profit=0.10,
            stop_loss=0.18,
            require_pre_game=True,
        )

        # Strategy 3: Balanced (proven baseline)
        balanced_config = StrategyConfig(
            min_edge_to_enter=0.05,
            exit_edge_buffer=-0.01,
            min_model_confidence=0.60,
            kelly_fraction=0.25,
            max_kelly_fraction=0.10,
            max_position_per_market=20,
            max_exposure_per_fixture=100,
            max_global_exposure=400,
            take_profit=0.15,
            stop_loss=0.20,
            require_pre_game=True,
        )

        # Strategy 4: Opportunistic (only trade best edges)
        opportunistic_config = StrategyConfig(
            min_edge_to_enter=0.08,       # High edge required
            exit_edge_buffer=0.00,        # Exit at neutral edge
            min_model_confidence=0.65,    # High confidence
            kelly_fraction=0.40,          # Aggressive sizing on good edges
            max_kelly_fraction=0.20,
            max_position_per_market=40,
            max_exposure_per_fixture=200,
            max_global_exposure=800,
            take_profit=0.12,
            stop_loss=0.25,
            require_pre_game=True,
        )

        # Strategy 5: In-Game Specialist (trades during games)
        in_game_config = StrategyConfig(
            min_edge_to_enter=0.06,
            exit_edge_buffer=-0.02,
            min_model_confidence=0.70,    # Need high confidence in-game
            kelly_fraction=0.20,          # More conservative in-game
            max_kelly_fraction=0.10,
            max_position_per_market=15,
            max_exposure_per_fixture=75,
            max_global_exposure=300,
            take_profit=0.10,
            stop_loss=0.15,               # Tight stops in-game
            require_pre_game=False,       # Allow in-game trading
            max_price_deviation_to_enter=0.20,  # Stricter in-game
        )

        # Create strategy arms
        self.arms = {
            "ultra_aggressive": StrategyArm(
                name="ultra_aggressive",
                description="Ultra aggressive: 2%+ edge, 50% Kelly, targets 10-12% daily",
                config=ultra_aggressive_config,
                strategy=EdgeStrategy(ultra_aggressive_config),
            ),
            "aggressive": StrategyArm(
                name="aggressive",
                description="Aggressive: 3%+ edge, 35% Kelly, high volume",
                config=aggressive_config,
                strategy=EdgeStrategy(aggressive_config),
            ),
            "balanced": StrategyArm(
                name="balanced",
                description="Balanced: 5%+ edge, 25% Kelly, proven baseline",
                config=balanced_config,
                strategy=EdgeStrategy(balanced_config),
            ),
            "opportunistic": StrategyArm(
                name="opportunistic",
                description="Opportunistic: 8%+ edge, 40% Kelly, quality over quantity",
                config=opportunistic_config,
                strategy=EdgeStrategy(opportunistic_config),
            ),
            "in_game": StrategyArm(
                name="in_game",
                description="In-game specialist: trades during live games",
                config=in_game_config,
                strategy=EdgeStrategy(in_game_config),
            ),
        }

        logger.info("strategy_bandit_initialized", num_strategies=len(self.arms))

    def select_strategy(self, fixture: FixtureContext) -> tuple[EdgeStrategy, str]:
        """Select best strategy for a fixture using Thompson Sampling.

        Args:
            fixture: Fixture context.

        Returns:
            (selected_strategy, strategy_name)
        """
        if self.enable_thompson_sampling:
            selected_name = self._thompson_sampling_selection()
        else:
            selected_name = self._epsilon_greedy_selection(epsilon=0.10)

        arm = self.arms[selected_name]
        arm.times_selected += 1
        arm.last_selected = datetime.now(UTC)

        # Record selection
        selection = StrategySelection(
            selection_id=f"{fixture.fixture_id}_{selected_name}_{datetime.now(UTC).timestamp()}",
            timestamp=datetime.now(UTC),
            fixture_id=fixture.fixture_id,
            strategy_name=selected_name,
            sampled_probability=arm.sample_reward_probability(),
        )
        self.selection_history.append(selection)

        logger.info(
            "strategy_selected",
            strategy=selected_name,
            fixture_id=fixture.fixture_id,
            expected_reward=arm.expected_reward(),
            times_selected=arm.times_selected,
        )

        return arm.strategy, selected_name

    def _thompson_sampling_selection(self) -> str:
        """Select strategy using Thompson Sampling."""
        samples = {}
        for name, arm in self.arms.items():
            # Sample from posterior
            sampled_prob = arm.sample_reward_probability()

            # Add exploration bonus (UCB-style)
            if self.exploration_bonus > 0 and arm.times_selected > 0:
                total_selections = sum(a.times_selected for a in self.arms.values())
                bonus = self.exploration_bonus * np.sqrt(
                    np.log(total_selections) / arm.times_selected
                )
                sampled_prob += bonus

            samples[name] = sampled_prob

        # Select best sample
        return max(samples, key=samples.get)

    def _epsilon_greedy_selection(self, epsilon: float = 0.10) -> str:
        """Select strategy using epsilon-greedy."""
        if np.random.rand() < epsilon:
            # Explore: random strategy
            return np.random.choice(list(self.arms.keys()))
        else:
            # Exploit: best expected reward
            return max(self.arms.items(), key=lambda x: x[1].expected_reward())[0]

    def update_outcome(
        self,
        fixture_id: int,
        strategy_name: str,
        pnl: float,
        exposure: float,
    ):
        """Update strategy arm after observing outcome.

        Args:
            fixture_id: Fixture ID.
            strategy_name: Strategy that was used.
            pnl: Realized P&L.
            exposure: Exposure amount.
        """
        arm = self.arms.get(strategy_name)
        if not arm:
            logger.warning("unknown_strategy_update", strategy=strategy_name)
            return

        # Define success: positive P&L or P&L > -5% of exposure
        is_success = pnl > 0 or (exposure > 0 and pnl / exposure > -0.05)

        if is_success:
            arm.update_success(pnl, exposure)
        else:
            arm.update_failure(pnl, exposure)

        # Update selection history
        for selection in reversed(self.selection_history):
            if selection.fixture_id == fixture_id and selection.strategy_name == strategy_name:
                selection.actual_pnl = pnl
                selection.actual_exposure = exposure
                selection.outcome = "success" if is_success else "failure"
                break

        # Calculate rolling Sharpe
        recent_pnls = self._get_recent_pnls(strategy_name)
        arm.avg_sharpe = arm.calculate_sharpe(recent_pnls)

        logger.info(
            "strategy_updated",
            strategy=strategy_name,
            outcome="success" if is_success else "failure",
            pnl=pnl,
            alpha=arm.alpha,
            beta=arm.beta,
            expected_reward=arm.expected_reward(),
            sharpe=arm.avg_sharpe,
        )

    def _get_recent_pnls(self, strategy_name: str, days: int = 7) -> list[float]:
        """Get recent P&Ls for a strategy."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        recent = [
            s.actual_pnl
            for s in self.selection_history
            if s.strategy_name == strategy_name
            and s.timestamp >= cutoff
            and s.outcome != "pending"
        ]
        return recent

    def get_performance_report(self) -> dict[str, Any]:
        """Generate performance report across all strategies."""
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_selections": len(self.selection_history),
            "strategies": {},
        }

        for name, arm in self.arms.items():
            win_rate = arm.winning_trades / arm.total_trades if arm.total_trades > 0 else 0.0
            avg_pnl = arm.total_pnl / arm.total_trades if arm.total_trades > 0 else 0.0

            report["strategies"][name] = {
                "description": arm.description,
                "times_selected": arm.times_selected,
                "total_trades": arm.total_trades,
                "winning_trades": arm.winning_trades,
                "win_rate": win_rate,
                "total_pnl": arm.total_pnl,
                "avg_pnl": avg_pnl,
                "sharpe_ratio": arm.avg_sharpe,
                "expected_reward": arm.expected_reward(),
                "alpha": arm.alpha,
                "beta": arm.beta,
                "last_selected": arm.last_selected.isoformat() if arm.last_selected else None,
            }

        return report

    def get_best_strategy(self) -> tuple[EdgeStrategy, str]:
        """Get currently best performing strategy (pure exploitation)."""
        best_arm = max(self.arms.items(), key=lambda x: x[1].expected_reward())
        return best_arm[1].strategy, best_arm[0]

    def reset_exploration(self):
        """Reset all arms to uniform prior (for restarting exploration)."""
        for arm in self.arms.values():
            arm.alpha = 1.0
            arm.beta = 1.0
        logger.info("exploration_reset")

    def persist_state(self):
        """Persist bandit state to database."""
        cursor = self.db.connection.cursor()

        for name, arm in self.arms.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO strategy_bandit_state (
                    strategy_name, alpha, beta,
                    total_trades, winning_trades, total_pnl,
                    times_selected, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    arm.alpha,
                    arm.beta,
                    arm.total_trades,
                    arm.winning_trades,
                    arm.total_pnl,
                    arm.times_selected,
                    datetime.now(UTC).isoformat(),
                ),
            )

        self.db.connection.commit()
        logger.info("bandit_state_persisted")

    def load_state(self):
        """Load bandit state from database."""
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT * FROM strategy_bandit_state")

        for row in cursor.fetchall():
            name = row["strategy_name"]
            if name in self.arms:
                arm = self.arms[name]
                arm.alpha = row["alpha"]
                arm.beta = row["beta"]
                arm.total_trades = row["total_trades"]
                arm.winning_trades = row["winning_trades"]
                arm.total_pnl = row["total_pnl"]
                arm.times_selected = row["times_selected"]

        logger.info("bandit_state_loaded", num_strategies=len(self.arms))
