"""Run Modes Configuration.

Defines different trading modes (conservative, aggressive, live_small, live_scaled)
with appropriate objectives and risk parameters for each.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from footbe_trader.agent.objective import AgentObjective
from footbe_trader.common.logging import get_logger

logger = get_logger(__name__)


class RunMode(Enum):
    """Available trading modes."""

    PAPER_CONSERVATIVE = "paper_conservative"
    PAPER_AGGRESSIVE = "paper_aggressive"
    LIVE_SMALL = "live_small"
    LIVE_SCALED = "live_scaled"

    @classmethod
    def from_string(cls, value: str) -> "RunMode":
        """Parse from string."""
        value = value.lower().replace("-", "_")
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Unknown run mode: {value}")


@dataclass
class RunModeConfig:
    """Configuration for a specific run mode."""

    # Mode identification
    mode: RunMode
    name: str
    description: str

    # Trading environment
    is_live: bool = False
    paper_initial_equity: float = 100_000.0

    # Objective parameters
    target_weekly_return: float = 0.05  # 5% default, 10% for aggressive
    max_drawdown: float = 0.15  # Hard stop at 15%
    soft_drawdown_threshold: float = 0.07  # Start throttling at 7%

    # Position limits (as % of equity)
    max_exposure_per_fixture: float = 0.015  # 1.5%
    max_exposure_per_league: float = 0.20  # 20%
    max_gross_exposure: float = 0.80  # 80%

    # Edge thresholds
    min_edge_threshold: float = 0.005  # 0.5% minimum
    target_edge_threshold: float = 0.02  # 2% target
    high_confidence_edge: float = 0.05  # 5%+ is high confidence

    # Drawdown throttle bands
    drawdown_bands: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.05, 1.0),   # 0-5% drawdown: 100% sizing
        (0.10, 0.7),   # 5-10% drawdown: 70% sizing
        (0.15, 0.4),   # 10-15% drawdown: 40% sizing
    ])

    # Pacing parameters
    pacing_behind_threshold: float = 0.7  # Behind if <70% of target
    pacing_ahead_threshold: float = 1.5   # Ahead if >150% of target
    max_pacing_multiplier: float = 1.5    # Max boost when behind
    min_pacing_multiplier: float = 0.7    # Min reduction when ahead

    # Time to kickoff adjustments
    very_early_discount: float = 0.3   # >48h: 30% discount
    early_discount: float = 0.15       # 24-48h: 15% discount
    late_discount: float = 0.25        # <2h: 25% discount

    # Circuit breaker thresholds (for live modes)
    max_daily_loss_pct: float = 0.05      # 5% max daily loss
    max_consecutive_losses: int = 5
    cooldown_minutes_after_circuit: int = 30

    # Execution
    max_orders_per_minute: int = 10
    min_time_between_trades_seconds: float = 5.0

    def to_objective(self) -> AgentObjective:
        """Convert to AgentObjective."""
        return AgentObjective(
            target_weekly_return=self.target_weekly_return,
            max_drawdown=self.max_drawdown,
            max_exposure_per_fixture=self.max_exposure_per_fixture,
            max_exposure_per_league=self.max_exposure_per_league,
            max_gross_exposure=self.max_gross_exposure,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "name": self.name,
            "description": self.description,
            "is_live": self.is_live,
            "paper_initial_equity": self.paper_initial_equity,
            "target_weekly_return": self.target_weekly_return,
            "max_drawdown": self.max_drawdown,
            "soft_drawdown_threshold": self.soft_drawdown_threshold,
            "max_exposure_per_fixture": self.max_exposure_per_fixture,
            "max_exposure_per_league": self.max_exposure_per_league,
            "max_gross_exposure": self.max_gross_exposure,
            "min_edge_threshold": self.min_edge_threshold,
            "target_edge_threshold": self.target_edge_threshold,
            "high_confidence_edge": self.high_confidence_edge,
            "drawdown_bands": [
                {"threshold": t, "multiplier": m}
                for t, m in self.drawdown_bands
            ],
            "pacing": {
                "behind_threshold": self.pacing_behind_threshold,
                "ahead_threshold": self.pacing_ahead_threshold,
                "max_multiplier": self.max_pacing_multiplier,
                "min_multiplier": self.min_pacing_multiplier,
            },
            "time_to_kickoff": {
                "very_early_discount": self.very_early_discount,
                "early_discount": self.early_discount,
                "late_discount": self.late_discount,
            },
            "circuit_breakers": {
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_consecutive_losses": self.max_consecutive_losses,
                "cooldown_minutes": self.cooldown_minutes_after_circuit,
            },
            "execution": {
                "max_orders_per_minute": self.max_orders_per_minute,
                "min_time_between_trades_seconds": self.min_time_between_trades_seconds,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunModeConfig":
        """Create from dictionary."""
        # Parse drawdown bands
        bands_data = data.get("drawdown_bands", [])
        bands = [(b["threshold"], b["multiplier"]) for b in bands_data]

        # Parse nested objects
        pacing = data.get("pacing", {})
        ttk = data.get("time_to_kickoff", {})
        circuit = data.get("circuit_breakers", {})
        execution = data.get("execution", {})

        return cls(
            mode=RunMode.from_string(data["mode"]),
            name=data["name"],
            description=data["description"],
            is_live=data.get("is_live", False),
            paper_initial_equity=data.get("paper_initial_equity", 100_000.0),
            target_weekly_return=data.get("target_weekly_return", 0.05),
            max_drawdown=data.get("max_drawdown", 0.15),
            soft_drawdown_threshold=data.get("soft_drawdown_threshold", 0.07),
            max_exposure_per_fixture=data.get("max_exposure_per_fixture", 0.015),
            max_exposure_per_league=data.get("max_exposure_per_league", 0.20),
            max_gross_exposure=data.get("max_gross_exposure", 0.80),
            min_edge_threshold=data.get("min_edge_threshold", 0.005),
            target_edge_threshold=data.get("target_edge_threshold", 0.02),
            high_confidence_edge=data.get("high_confidence_edge", 0.05),
            drawdown_bands=bands if bands else [
                (0.05, 1.0),
                (0.10, 0.7),
                (0.15, 0.4),
            ],
            pacing_behind_threshold=pacing.get("behind_threshold", 0.7),
            pacing_ahead_threshold=pacing.get("ahead_threshold", 1.5),
            max_pacing_multiplier=pacing.get("max_multiplier", 1.5),
            min_pacing_multiplier=pacing.get("min_multiplier", 0.7),
            very_early_discount=ttk.get("very_early_discount", 0.3),
            early_discount=ttk.get("early_discount", 0.15),
            late_discount=ttk.get("late_discount", 0.25),
            max_daily_loss_pct=circuit.get("max_daily_loss_pct", 0.05),
            max_consecutive_losses=circuit.get("max_consecutive_losses", 5),
            cooldown_minutes_after_circuit=circuit.get("cooldown_minutes", 30),
            max_orders_per_minute=execution.get("max_orders_per_minute", 10),
            min_time_between_trades_seconds=execution.get(
                "min_time_between_trades_seconds", 5.0
            ),
        )


# Pre-defined configurations for each mode
PAPER_CONSERVATIVE_CONFIG = RunModeConfig(
    mode=RunMode.PAPER_CONSERVATIVE,
    name="Paper Conservative",
    description="Conservative paper trading for model validation. Low risk, patient approach.",
    is_live=False,
    paper_initial_equity=100_000.0,
    target_weekly_return=0.03,  # 3% weekly target
    max_drawdown=0.10,  # 10% max drawdown
    soft_drawdown_threshold=0.05,
    max_exposure_per_fixture=0.01,  # 1% per fixture
    max_exposure_per_league=0.15,
    max_gross_exposure=0.50,  # Only 50% deployed max
    min_edge_threshold=0.02,  # Higher edge threshold
    target_edge_threshold=0.03,
    high_confidence_edge=0.06,
)

PAPER_AGGRESSIVE_CONFIG = RunModeConfig(
    mode=RunMode.PAPER_AGGRESSIVE,
    name="Paper Aggressive",
    description="Aggressive paper trading targeting 10% weekly growth. Tests edge utilization.",
    is_live=False,
    paper_initial_equity=100_000.0,
    target_weekly_return=0.10,  # 10% weekly target
    max_drawdown=0.15,  # 15% max drawdown
    soft_drawdown_threshold=0.07,
    max_exposure_per_fixture=0.02,  # 2% per fixture
    max_exposure_per_league=0.25,
    max_gross_exposure=0.80,  # 80% deployed max
    min_edge_threshold=0.005,  # Lower threshold to take more trades
    target_edge_threshold=0.02,
    high_confidence_edge=0.05,
    max_pacing_multiplier=1.8,  # More aggressive when behind
)

LIVE_SMALL_CONFIG = RunModeConfig(
    mode=RunMode.LIVE_SMALL,
    name="Live Small",
    description="Small stake live trading to validate execution. Prove the system works with real money.",
    is_live=True,
    target_weekly_return=0.05,  # 5% weekly target
    max_drawdown=0.12,  # 12% max drawdown (tighter for real money)
    soft_drawdown_threshold=0.05,
    max_exposure_per_fixture=0.01,  # 1% per fixture
    max_exposure_per_league=0.15,
    max_gross_exposure=0.50,
    min_edge_threshold=0.015,  # Slightly higher for live
    target_edge_threshold=0.025,
    high_confidence_edge=0.05,
    max_daily_loss_pct=0.03,  # Tighter daily loss limit
    max_consecutive_losses=4,
    cooldown_minutes_after_circuit=60,  # Longer cooldown
)

LIVE_SCALED_CONFIG = RunModeConfig(
    mode=RunMode.LIVE_SCALED,
    name="Live Scaled",
    description="Full scale live trading after successful validation period.",
    is_live=True,
    target_weekly_return=0.08,  # 8% weekly target (slightly less than paper)
    max_drawdown=0.15,
    soft_drawdown_threshold=0.07,
    max_exposure_per_fixture=0.015,  # 1.5% per fixture
    max_exposure_per_league=0.20,
    max_gross_exposure=0.70,  # Slightly less than paper max
    min_edge_threshold=0.01,
    target_edge_threshold=0.02,
    high_confidence_edge=0.05,
    max_daily_loss_pct=0.04,
    max_consecutive_losses=5,
    cooldown_minutes_after_circuit=45,
)


# Registry of all configurations
_MODE_CONFIGS: dict[RunMode, RunModeConfig] = {
    RunMode.PAPER_CONSERVATIVE: PAPER_CONSERVATIVE_CONFIG,
    RunMode.PAPER_AGGRESSIVE: PAPER_AGGRESSIVE_CONFIG,
    RunMode.LIVE_SMALL: LIVE_SMALL_CONFIG,
    RunMode.LIVE_SCALED: LIVE_SCALED_CONFIG,
}


def get_mode_config(mode: RunMode | str) -> RunModeConfig:
    """Get configuration for a run mode.

    Args:
        mode: Run mode enum or string.

    Returns:
        Configuration for the mode.
    """
    if isinstance(mode, str):
        mode = RunMode.from_string(mode)
    return _MODE_CONFIGS[mode]


def list_available_modes() -> list[dict[str, str]]:
    """List all available modes with descriptions.

    Returns:
        List of mode info dictionaries.
    """
    return [
        {
            "mode": cfg.mode.value,
            "name": cfg.name,
            "description": cfg.description,
            "is_live": cfg.is_live,
            "target_weekly_return": f"{cfg.target_weekly_return:.0%}",
        }
        for cfg in _MODE_CONFIGS.values()
    ]


def save_config_to_yaml(config: RunModeConfig, path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    logger.info(
        "config_saved",
        mode=config.mode.value,
        path=str(path),
    )


def load_config_from_yaml(path: Path) -> RunModeConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Loaded configuration.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    config = RunModeConfig.from_dict(data)
    logger.info(
        "config_loaded",
        mode=config.mode.value,
        path=str(path),
    )
    return config


def generate_all_config_files(output_dir: Path) -> None:
    """Generate YAML config files for all modes.

    Args:
        output_dir: Directory to write config files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in _MODE_CONFIGS.values():
        filename = f"{config.mode.value}.yaml"
        save_config_to_yaml(config, output_dir / filename)

    logger.info(
        "all_configs_generated",
        output_dir=str(output_dir),
        count=len(_MODE_CONFIGS),
    )
