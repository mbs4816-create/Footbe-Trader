# Agent Trading Policy

This document describes how the trading agent makes decisions about aggressiveness, position sizing, and risk management.

## Overview

The agent's goal-driven policy system orchestrates multiple factors to determine:
- **What to trade** (edge thresholds)
- **How much to trade** (position sizing)
- **When to be aggressive or conservative** (pacing and drawdown)

## Core Components

### 1. Agent Objective (`objective.py`)

Defines explicit targets and constraints:

```python
AgentObjective(
    target_weekly_return=0.10,      # 10% per rolling 7 days
    max_drawdown=0.15,              # 15% hard stop
    max_exposure_per_fixture=0.015, # 1.5% of equity per fixture
    max_exposure_per_league=0.20,   # 20% per league
    max_gross_exposure=0.80,        # 80% max deployed
)
```

### 2. Pacing Tracker

Tracks the agent's progress toward the weekly target and adjusts behavior:

| Pacing State | Condition | Edge Adjustment | Size Adjustment |
|--------------|-----------|-----------------|-----------------|
| AHEAD_OF_PACE | Return ≥ Target | Raise threshold (more selective) | Reduce size |
| ON_PACE | Within ±2% of target | Baseline thresholds | Baseline sizing |
| BEHIND_PACE | Return < Target - 2% | Lower threshold (more aggressive) | Increase size |

### 3. Drawdown Throttle

Position sizing is throttled based on current drawdown from peak:

| Drawdown Range | Band | Sizing Multiplier | New Entries |
|----------------|------|-------------------|-------------|
| 0-5% | NONE | 100% | ✓ Allowed |
| 5-10% | LIGHT | 70% | ✓ Allowed |
| 10-15% | MODERATE | 40% | ✓ Allowed |
| >15% | SEVERE | 0% | ✗ Blocked |

### 4. Time-to-Kickoff Adjustments

Edge requirements and sizing vary based on when we're trading relative to kickoff:

| Category | Hours to Kickoff | Edge Multiplier | Size Multiplier | Rationale |
|----------|------------------|-----------------|-----------------|-----------|
| VERY_EARLY | >48h | 2.0x | 25% | High uncertainty |
| EARLY | 24-48h | 1.5x | 50% | Moderate uncertainty |
| STANDARD | 6-24h | 1.1x | 85% | Minor premium |
| OPTIMAL | 2-6h | 1.0x | 100% | Core trading window |
| LATE | <2h | 1.2x | 70% | Limited time to react |

## Decision Flow

When evaluating a potential trade:

```
1. Calculate raw edge = model_prob - market_price
2. Get pacing adjustment based on 7-day rolling return
3. Get drawdown throttle multiplier
4. Get time-to-kickoff adjustment
5. Apply all multipliers to edge threshold and position size
6. Check exposure limits (fixture, league, gross)
7. Generate human-readable explanation
8. Return PolicyDecision (can_trade, final_size, explanation)
```

## Run Modes

Four pre-configured modes are available:

### Paper Conservative
- **Target**: 3% weekly return
- **Max Drawdown**: 10%
- **Max Gross Exposure**: 50%
- **Edge Threshold**: 2% minimum
- **Use Case**: Model validation, cautious testing

### Paper Aggressive
- **Target**: 10% weekly return
- **Max Drawdown**: 15%
- **Max Gross Exposure**: 80%
- **Edge Threshold**: 0.5% minimum
- **Use Case**: Stress testing, edge utilization

### Live Small
- **Target**: 5% weekly return
- **Max Drawdown**: 12% (tighter)
- **Max Gross Exposure**: 50%
- **Daily Loss Limit**: 3%
- **Use Case**: Real money validation with small stakes

### Live Scaled
- **Target**: 8% weekly return
- **Max Drawdown**: 15%
- **Max Gross Exposure**: 70%
- **Daily Loss Limit**: 4%
- **Use Case**: Full production trading

## Edge Bucket Evaluation

The `EdgeBucketEvaluator` analyzes historical trades to answer: "When the model predicts X% edge, how accurate is that?"

Buckets:
- 0-1%
- 1-2%
- 2-4%
- 4-6%
- 6-10%
- 10%+

For each bucket, it computes:
- Win rate
- Average return
- Calibration error (predicted edge - realized return)
- Statistics by outcome (home_win, draw, away_win)
- Statistics by time category

## Example Policy Decision

```
=== Trade Decision ===

Model prob: 55.0%, Market: 50.0%, Edge: +5.0%
Time to kickoff: 4.0h (optimal)
Pacing: on_pace (7d return: 8.5%)
Drawdown: 3.2% (none band)
Exposure - Fixture: $0/1500, League: $2500/$20000, Gross: $15000/$80000

--- Adjustments Applied ---
Edge threshold: 2.0% (pacing: ×1.00, time: ×1.00)
Position sizing: 85 contracts
  Raw Kelly: 100
  × Pacing: 1.00
  × Drawdown: 1.00
  × Time: 1.00
  = Adjusted: 100
  → Capped: 85

Action: BUY 85 contracts of home_win
Rationale: Edge 5.0% exceeds threshold, risk acceptable
```

## Configuration Files

Configuration files are stored in `config/run_modes/`:

```
config/run_modes/
├── paper_conservative.yaml
├── paper_aggressive.yaml
├── live_small.yaml
└── live_scaled.yaml
```

Each file contains all parameters for a run mode and can be customized.

## Usage

```python
from footbe_trader.agent import (
    TradingPolicy,
    AgentObjective,
    get_mode_config,
    RunMode,
)

# Load a run mode
config = get_mode_config(RunMode.PAPER_AGGRESSIVE)

# Create policy from config
policy = TradingPolicy(
    objective=config.to_objective(),
    initial_equity=config.paper_initial_equity,
)

# Update with current state
policy.update_equity(current_equity)
policy.update_exposure(
    gross_exposure=total_exposure,
    exposure_by_fixture=fixture_exposures,
    exposure_by_league=league_exposures,
    position_count=num_positions,
)

# Evaluate a trade
decision = policy.evaluate_trade(
    fixture_id=12345,
    league="EPL",
    outcome="home_win",
    model_prob=0.55,
    market_price=0.50,
    base_edge_threshold=0.02,
    kelly_position_size=100,
    hours_to_kickoff=4.0,
    price_per_contract=0.50,
)

if decision.can_trade:
    print(f"Trade approved: {decision.final_position_size} contracts")
    print(decision.explanation)
else:
    print(f"Trade rejected: {decision.rejection_reasons}")
```

## Status Reporting

```python
# Get structured status
status = policy.get_status()

# Get human-readable report
print(policy.generate_status_report())
```

Output:
```
============================================================
TRADING POLICY STATUS
============================================================

Equity: $105,000.00 (Peak: $105,000.00)
Drawdown: 0.0% (none band)
  → Sizing multiplier: 100%
  → New entries allowed: YES

Pacing: on_pace
  → 7-day return: 5.0%
  → Target: 10.0%
  → Edge threshold multiplier: 1.00
  → Sizing multiplier: 1.00

Exposure:
  → Gross: $15,000.00
  → Positions: 12

Current behavior:
  On pace toward target. Maintaining baseline thresholds and sizing.

============================================================
```
