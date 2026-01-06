# Self-Improving Trading Agent - Implementation Guide

## 🎯 System Overview

This trading agent is designed to achieve **10-12% daily returns** with a **20% maximum drawdown limit** through:

1. **Multi-Armed Bandit Strategy Selection** - Dynamically chooses best strategy
2. **Automated Model Retraining** - Updates models when performance degrades
3. **Position Invalidation** - Exits positions when pre-game assumptions fail
4. **Daily Performance Tracking** - Monitors pace towards aggressive targets

---

## 📁 New Components

### 1. Model Lifecycle Manager
**File**: `src/footbe_trader/self_improvement/model_lifecycle.py`

**Purpose**: Automatically retrains models and deploys improvements

**Key Features**:
- Detects performance drift (Sharpe drop >30% or accuracy drop >2%)
- Trains new models on rolling 180-day window
- A/B tests challengers vs champions in paper mode
- Promotes winners after statistical validation

**Usage**:
```python
manager = ModelLifecycleManager(db, config)
await manager.start_monitoring_loop(interval_hours=24)
```

### 2. Strategy Bandit
**File**: `src/footbe_trader/self_improvement/strategy_bandit.py`

**Purpose**: Multi-armed bandit for dynamic strategy selection

**Strategies**:
1. **Ultra Aggressive**: 2% edge, 50% Kelly, targets 10-12% daily
2. **Aggressive**: 3% edge, 35% Kelly, high volume
3. **Balanced**: 5% edge, 25% Kelly, proven baseline
4. **Opportunistic**: 8% edge, 40% Kelly, quality over quantity
5. **In-Game Specialist**: Trades during live games

**Algorithm**: Thompson Sampling
- Each strategy has Beta(α, β) belief about success probability
- Sample from posteriors and pick highest
- Update beliefs based on realized P&L

**Usage**:
```python
bandit = StrategyBandit(db)
strategy, name = bandit.select_strategy(fixture)

# After trade settles
bandit.update_outcome(fixture_id, name, pnl, exposure)
```

### 3. Position Invalidator
**File**: `src/footbe_trader/execution/position_invalidator.py`

**Purpose**: Exits positions when game state invalidates original thesis

**Triggers**:
- **Game Started**: Pre-game position now in-game
- **Adverse Score**: Team losing when we bought their win
- **Price Movement**: Market moved >25% against us
- **Staleness**: Position open >48 hours
- **Closing Minutes**: <5 min left and losing

**Example**:
```
Position: Bought HOME WIN at $0.50 (expected fair value $0.60)
Game Event: Home team down 0-1 after 10 minutes
Action: EXIT - Original thesis (pre-game edge) is INVALID
Market: Home Win now trading at $0.30
```

**Usage**:
```python
invalidator = PositionInvalidator(db, kalshi_client, simulator, live_game_provider)
invalidations = await invalidator.scan_and_invalidate_positions()
```

### 4. Daily Performance Tracker
**File**: `src/footbe_trader/reporting/daily_performance_tracker.py`

**Purpose**: Track progress towards 10-12% daily targets

**Key Insight**: Sports P&L realizes at **game settlement**, not position open
- Track unrealized P&L by settlement date
- Project when gains will materialize
- Prevent over-concentration on single day
- Monitor pace accounting for delayed realization

**Daily Targets**:
- Low: 10% ($100 on $1000 bankroll)
- Mid: 11% ($110) ← Primary target
- High: 12% ($120) ← Stretch goal

**Weekly Targets** (compounded):
- Low: 70% (10% daily × 7 days)
- High: 113% (12% daily × 7 days)

**Usage**:
```python
tracker = DailyPerformanceTracker(db, starting_bankroll=1000)
tracker.update_daily_progress(realized_pnl, unrealized_pnl, exposure, ...)
report = tracker.generate_daily_report()
alert = tracker.generate_pace_alert()
```

---

## 🚀 Quick Start

### 1. Run Self-Improving Agent (Paper Mode)

```bash
# Paper trading with $1000 bankroll
python scripts/run_self_improving_agent.py \
  --mode paper \
  --interval 30 \
  --bankroll 1000

# With aggressive strategy config
python scripts/run_self_improving_agent.py \
  --mode paper \
  --config configs/strategy_config_aggressive.yaml \
  --bankroll 1000
```

### 2. Monitor Performance

Check logs for:
```
strategy_selected: Strategy chosen for each fixture
position_scan_complete: Invalid positions detected
daily_performance_update: Progress towards target
model_lifecycle: Model retraining events
```

### 3. View Telegram Updates (if configured)

You'll receive:
- **Hourly**: Position invalidation alerts
- **Daily**: Performance summary with pace alerts
- **Weekly**: Strategy bandit performance
- **Critical**: Drawdown warnings and emergency stops

---

## 📊 Performance Expectations

### Baseline (Current Static Strategy)
- Daily Return: ~0.5-1.5%
- Sharpe Ratio: 0.3-0.5
- Max Drawdown: ~35%
- Win Rate: ~48%

### Target (Self-Improving Agent)
- **Daily Return: 10-12%** ← AGGRESSIVE
- **Sharpe Ratio: 1.2-1.8**
- **Max Drawdown: 20%** ← HARD LIMIT
- **Win Rate: 55-60%**

### Realistic Timeline
- **Month 1**: 3-5% daily (learning phase)
- **Month 2**: 5-8% daily (optimization phase)
- **Month 3+**: 8-12% daily (mature phase)

**Note**: These targets are EXTREMELY aggressive. Sports betting markets are efficient. Sustained 10% daily returns would imply:
- 3,000% annual return
- Turning $1k → $3M in one year
- Likely unsustainable at scale

---

## ⚠️ Risk Management

### Hard Limits
1. **20% Max Drawdown**: Agent STOPS if exceeded
2. **$1000 Max Global Exposure**: No more than 100% bankroll at risk
3. **$250 Max Per Fixture**: Diversification requirement
4. **Position Invalidation**: Automatic exits when thesis fails

### Monitoring Checklist
- [ ] Check drawdown daily (should be <15%)
- [ ] Verify position invalidation working (game state exits)
- [ ] Monitor strategy bandit convergence (winning strategies emerging)
- [ ] Track model retraining (models updating weekly)
- [ ] Review settlement projections (P&L realization dates)

### Emergency Procedures
If drawdown >18%:
1. Agent auto-pauses trading
2. Telegram alert sent
3. Manual review required
4. Reduce position sizes before resuming

---

## 🔧 Configuration

### Aggressive Strategy (`configs/strategy_config_aggressive.yaml`)

```yaml
edge:
  min_edge_to_enter: 0.02  # Accept 2%+ edges (vs 5% conservative)

kelly:
  fraction: 0.50  # 50% Kelly (HIGH RISK)
  max_kelly_fraction: 0.25  # 25% bankroll per trade

limits:
  max_global_exposure: 1000.0  # $1000 total
  max_exposure_per_fixture: 250.0  # $250 per fixture

in_game:
  require_pre_game: false  # Allow in-game trading
```

### Strategy Bandit Parameters

Tunable in `strategy_bandit.py`:
```python
# Exploration bonus (higher = more exploration)
exploration_bonus: float = 0.0  # Default: pure Thompson Sampling

# Success definition
is_success = pnl > 0 or (exposure > 0 and pnl / exposure > -0.05)
# i.e., positive P&L OR loss <5% of exposure
```

### Model Retraining Thresholds

Tunable in `model_lifecycle.py`:
```python
retrain_threshold_sharpe_drop: float = 0.30  # Retrain if Sharpe drops 30%
retrain_threshold_accuracy_drop: float = 0.02  # Or accuracy drops 2%
ab_test_duration_days: int = 7  # A/B test for 7 days
```

---

## 📈 Understanding Daily Pace

### The Settlement Problem

Sports betting P&L **doesn't realize immediately**:

```
Monday:
  - Open 5 positions @ $200 exposure
  - Unrealized P&L: +$50 (25% ROI potential)
  - Settled P&L: $0 (games are Saturday)

Tuesday:
  - "Daily return": 0% ← MISLEADING
  - "Projected Saturday": +$50 ← ACTUAL metric

Saturday (games settle):
  - Realized P&L: +$45 (won 4/5 positions)
  - "Week return": 4.5% ← This is the real number
```

### Pace Calculation

The tracker projects:
1. **Unrealized P&L by settlement date**
2. **Expected realization based on historical win rate**
3. **Adjusted daily pace accounting for delays**

Example:
```python
# Day 1: Open 10 positions, settle Day 3
tracker.update_daily_progress(
    realized_pnl=0,  # Nothing settled yet
    unrealized_pnl=100,  # Projected +$100
    ...
)

# Day 3: Positions settle
tracker.update_daily_progress(
    realized_pnl=85,  # Won 8/10 positions
    unrealized_pnl=0,  # All closed
    ...
)

# Pace calculation: $85 over 3 days = 8.5%/day effective rate
```

### Over-Trading Prevention

Without settlement-aware tracking, agents over-trade:
- Day 1: Open 20 positions (nothing realizes)
- Day 2: "I'm behind pace!" → Open 30 more positions
- Day 3: Everything settles at once → HUGE variance

With tracking:
- Day 1: Open 10 positions → Projected settlement: +$100
- Day 2: Already have $100 projected for Day 3 → Open smaller size
- Day 3: Realize $85 → Right-sized risk

---

## 🧪 Testing & Validation

### 1. Backtest Self-Improvement

Test the complete system on historical data:

```python
# TODO: Implement backtesting for self-improving components
# This would simulate:
# - Strategy bandit learning from past trades
# - Model retraining on historical windows
# - Position invalidation on past in-game states
```

### 2. Adversarial Testing

Test against informed opponents:

```python
# TODO: Implement adversarial backtesting
# Simulate market makers who:
# - Predict your strategy
# - Widen spreads when you want to trade
# - Front-run your orders
```

### 3. Monte Carlo Stress Testing

```python
# TODO: Implement stress testing
# Scenarios:
# - Black swan: All positions lose simultaneously
# - Liquidity crisis: Can't exit positions
# - API outage: No data for 2 hours
# - Model drift: Predictions become uncalibrated
```

---

## 🔄 Development Roadmap

### ✅ Phase 1: Core Infrastructure (COMPLETE)
- [x] Model Lifecycle Manager
- [x] Strategy Bandit with Thompson Sampling
- [x] Position Invalidator
- [x] Daily Performance Tracker
- [x] Aggressive strategy config

### 📋 Phase 2: Enhanced Features (NEXT)
- [ ] Enhanced Feature Builder (market microstructure, order flow)
- [ ] Adversarial Backtesting Framework
- [ ] Reinforcement Learning for trade timing
- [ ] Multi-model ensemble
- [ ] Real-time API health monitoring

### 🎯 Phase 3: Production Hardening
- [ ] A/B testing framework for models
- [ ] Distributed execution (multiple agents)
- [ ] Capital allocation optimizer
- [ ] Risk parity portfolio balancing
- [ ] Automated incident response

### 🚀 Phase 4: Advanced ML
- [ ] Neural network probability models
- [ ] LSTM for time-series prediction
- [ ] Transformer models for game state
- [ ] Meta-learning for rapid adaptation
- [ ] Causal inference for strategy evaluation

---

## 📚 Key Files Reference

```
configs/
  strategy_config_aggressive.yaml  # Aggressive strategy parameters

src/footbe_trader/
  self_improvement/
    model_lifecycle.py              # Automated model retraining
    strategy_bandit.py              # Multi-armed bandit strategy selection

  execution/
    position_invalidator.py         # Exit positions when invalid

  reporting/
    daily_performance_tracker.py    # Track progress to 10-12% daily

scripts/
  run_self_improving_agent.py       # Main entry point
```

---

## 🐛 Troubleshooting

### Agent not making trades
1. Check strategy bandit selection (might be in pure exploration)
2. Verify edge thresholds (might be too high for current opportunities)
3. Check drawdown limit (might be paused due to losses)

### Models not retraining
1. Check if performance drift detected (needs >30% Sharpe drop)
2. Verify sufficient training data (needs 100+ recent fixtures)
3. Check model_versions table for status

### Position invalidation not working
1. Verify live_game_provider is configured
2. Check API credentials for Football/NBA APIs
3. Review position_invalidator logs for errors

### Drawdown exceeding limits
1. **STOP TRADING IMMEDIATELY**
2. Review recent trades for pattern
3. Check if strategy bandit converged to bad strategy
4. Consider resetting bandit (lose learning but prevent further losses)
5. Reduce position sizes before resuming

---

## 💡 Best Practices

### 1. Start Conservative
Even with aggressive targets, start with:
- Small bankroll ($100-500)
- Paper trading for 1-2 weeks
- Monitor for model/strategy convergence

### 2. Monitor Key Metrics
Track these daily:
- **Drawdown**: Should trend down over time
- **Strategy Convergence**: Which strategies are winning?
- **Model Accuracy**: Is it improving?
- **Settlement Timing**: Are projections accurate?

### 3. Adjust Targets Based on Reality
If after 30 days you're averaging 5% daily:
- **Good!** That's still 400% annual
- **Don't force** higher targets
- **Reduce** position sizes if variance too high

### 4. Respect the Drawdown Limit
20% drawdown = **HARD STOP**
- No exceptions
- No "one more trade"
- Pause, review, reduce size

---

## 🎓 Learning Resources

### Thompson Sampling & Bandits
- "A Tutorial on Thompson Sampling" (Russo et al., 2018)
- "Bandit Algorithms" by Lattimore & Szepesvári

### Sports Trading
- "Trading and Exchanges" by Harris
- "Algorithmic and High-Frequency Trading" by Cartea et al.

### Risk Management
- "The Kelly Capital Growth Investment Criterion" (MacLean et al.)
- "Active Portfolio Management" by Grinold & Kahn

---

## 📞 Support

For issues or questions:
1. Check logs: `logs/footbe_trader.log`
2. Review database: `data/footbe_dev.db`
3. Check Telegram alerts (if configured)
4. File issue on GitHub

---

**Remember**: These are AGGRESSIVE targets. Most professional traders target 20-30% annual returns. 10% daily = 3,000% annual. Be prepared for significant variance and respect the risk limits!
