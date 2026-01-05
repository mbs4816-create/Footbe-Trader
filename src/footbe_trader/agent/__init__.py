"""Agent module for goal-driven trading.

This module provides:
- Objective definition with weekly return targets and constraints
- Pacing tracking to know if we're ahead/behind target
- Drawdown-based risk throttling
- Time-to-kickoff aware trading adjustments
- Edge bucket evaluation for model calibration
- Run mode configurations for different trading scenarios
"""

from footbe_trader.agent.heartbeat import Heartbeat, main
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
from footbe_trader.agent.policy import (
    ExposureState,
    PolicyDecision,
    TradingPolicy,
)
from footbe_trader.agent.evaluation import (
    EdgeBucket,
    EdgeBucketEvaluator,
    EvaluationReport,
    TradeRecord,
)
from footbe_trader.agent.run_modes import (
    RunMode,
    RunModeConfig,
    get_mode_config,
    list_available_modes,
    load_config_from_yaml,
    save_config_to_yaml,
    PAPER_CONSERVATIVE_CONFIG,
    PAPER_AGGRESSIVE_CONFIG,
    LIVE_SMALL_CONFIG,
    LIVE_SCALED_CONFIG,
)

__all__ = [
    # Heartbeat (existing)
    "Heartbeat",
    "main",
    # Objective & Constraints
    "AgentObjective",
    "DrawdownBand",
    "DrawdownThrottle",
    "EquitySnapshot",
    # Pacing
    "PacingAdjustment",
    "PacingState",
    "PacingTracker",
    # Time to kickoff
    "TimeToKickoffCategory",
    "classify_time_to_kickoff",
    "get_time_to_kickoff_adjustment",
    # Policy
    "ExposureState",
    "PolicyDecision",
    "TradingPolicy",
    # Evaluation
    "EdgeBucket",
    "EdgeBucketEvaluator",
    "EvaluationReport",
    "TradeRecord",
    # Run Modes
    "RunMode",
    "RunModeConfig",
    "get_mode_config",
    "list_available_modes",
    "load_config_from_yaml",
    "save_config_to_yaml",
    "PAPER_CONSERVATIVE_CONFIG",
    "PAPER_AGGRESSIVE_CONFIG",
    "LIVE_SMALL_CONFIG",
    "LIVE_SCALED_CONFIG",
]
