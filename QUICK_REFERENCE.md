# Self-Improving Trading Agent - Quick Reference

## 🚀 Start Trading

```bash
# Paper trading with $1000
python scripts/run_self_improving_agent.py --mode paper --bankroll 1000

# Live trading (CAREFUL!)
python scripts/run_self_improving_agent.py --mode live --bankroll 1000 --dry-run

# Monitor logs
tail -f logs/footbe_trader.log
```

---

## 📊 Key Metrics

| Metric | Target | Alert Level | Stop Level |
|--------|--------|-------------|------------|
| Daily Return | 10-12% | <7% | N/A |
| Max Drawdown | <15% | 15% | **20%** |
| Win Rate | >55% | <50% | N/A |
| Sharpe Ratio | >1.2 | <0.8 | N/A |
| Position Invalidations | <3/day | >5/day | >10/day |

---

## ⚡ Quick Commands

### Check Performance
```bash
# Today's snapshot
sqlite3 data/footbe_dev.db \
  "SELECT * FROM daily_performance_snapshots ORDER BY date DESC LIMIT 1"

# Strategy bandit state
sqlite3 data/footbe_dev.db \
  "SELECT * FROM strategy_bandit_state ORDER BY total_pnl DESC"

# Recent model versions
sqlite3 data/footbe_dev.db \
  "SELECT * FROM model_versions WHERE status='deployed'"
```

### Reset Bandit (if stuck)
```python
from footbe_trader.self_improvement.strategy_bandit import StrategyBandit
bandit = StrategyBandit(db)
bandit.reset_exploration()
bandit.persist_state()
```

### Force Model Retrain
```python
from footbe_trader.self_improvement.model_lifecycle import ModelLifecycleManager
manager = ModelLifecycleManager(db, config)
await manager._retrain_model("football")
```

---

## 🎯 5 Strategies (Multi-Armed Bandit)

| Strategy | Edge Min | Kelly | Exposure | Target Daily |
|----------|----------|-------|----------|--------------|
| **Ultra Aggressive** | 2% | 50% | $1000 | **10-12%** |
| **Aggressive** | 3% | 35% | $600 | 8-10% |
| **Balanced** | 5% | 25% | $400 | 5-7% |
| **Opportunistic** | 8% | 40% | $800 | 7-9% |
| **In-Game** | 6% | 20% | $300 | 4-6% |

*Bandit automatically selects best strategy for each fixture*

---

## ⚠️ Position Invalidation Triggers

| Trigger | Severity | Exit % | Example |
|---------|----------|--------|---------|
| Adverse Score | 🔴 Critical | 100% | Bought HOME WIN, team losing |
| Price Drop >25% | 🔴 Critical | 75% | Bought at $0.40, now $0.30 |
| Game Started (pre-game position) | 🟡 Warning | 50% | Entered pre-game, now in-game |
| Stale (>48hrs) | 🟡 Warning | 25% | Position too old |
| Closing Minutes (<5min) + Losing | 🔴 Critical | 100% | Game ending, position negative |

---

## 🔧 Configuration Tuning

### Increase Aggressiveness
```yaml
# configs/strategy_config_aggressive.yaml
kelly:
  fraction: 0.60  # From 0.50
limits:
  max_global_exposure: 1200.0  # From 1000
edge:
  min_edge_to_enter: 0.015  # From 0.02 (lower = more trades)
```

### Reduce Risk
```yaml
kelly:
  fraction: 0.30  # From 0.50
limits:
  max_global_exposure: 600.0  # From 1000
exit_rules:
  stop_loss: 0.10  # From 0.15 (tighter stops)
```

---

## 📈 Performance Tracking

### Daily Progress
```python
from footbe_trader.reporting.daily_performance_tracker import DailyPerformanceTracker

tracker = DailyPerformanceTracker(db, starting_bankroll=1000)
report = tracker.generate_daily_report()

print(f"Target: ${report['today']['target_mid']:.2f}")
print(f"Actual: ${report['today']['total_pnl']:.2f}")
print(f"Progress: {report['today']['completion_pct']:.0%}")
print(f"Status: {report['today']['status']}")
```

### Generate Alert
```python
alert = tracker.generate_pace_alert()
if alert:
    print(alert)
# Output:
# ⚠️ WARNING: 65% of daily target. Need $38.50 to hit 11% target.
# 📉 Week is 1.2 days behind pace. Projected week: $350 vs target $700
```

---

## 🚨 Emergency Procedures

### Drawdown >18%
1. **Agent auto-pauses** (hardcoded at 20%)
2. Manual actions:
   ```python
   # Reduce all position sizes
   for name, arm in bandit.arms.items():
       arm.config.max_position_per_market = 10  # From 50
       arm.config.max_exposure_per_fixture = 50  # From 250
   ```
3. Resume with 50% sizing

### Too Many Invalidations
```yaml
# Increase pre-game buffer
in_game:
  min_minutes_before_kickoff: 15  # From 2
  require_pre_game: true  # Force pre-game only
```

### Models Not Working
```python
# Check recent accuracy
result = backtester.run_season_backtest(recent_fixtures)
print(f"Accuracy: {result.aggregate_metrics.accuracy:.1%}")

# If <48%, retrain immediately
if result.aggregate_metrics.accuracy < 0.48:
    await manager._retrain_model("football")
```

---

## 📱 Telegram Alerts

### Configure
```yaml
# configs/dev.yaml
telegram:
  enabled: true
  bot_token: "YOUR_TOKEN"
  chat_id: "YOUR_CHAT_ID"
```

### Alert Types
- **Hourly**: Position invalidations
- **Daily**: Performance summary, pace alerts
- **Weekly**: Strategy bandit rankings
- **Critical**: Drawdown warnings, emergency stops

---

## 🎓 Learning Indicators

### System is Learning When:
- ✅ Strategy bandit converging (1-2 strategies dominate)
- ✅ Win rate improving (week-over-week)
- ✅ Sharpe ratio increasing
- ✅ Position invalidations decreasing
- ✅ Model accuracy stable or improving

### System is NOT Learning When:
- ❌ Bandit keeps switching strategies randomly
- ❌ Win rate declining
- ❌ Drawdown increasing
- ❌ Many invalidations per day
- ❌ Model accuracy dropping

---

## 📊 Database Schema (New Tables)

```sql
-- Model versions (lifecycle management)
CREATE TABLE model_versions (
    model_id TEXT PRIMARY KEY,
    model_name TEXT,
    version TEXT,
    validation_sharpe REAL,
    status TEXT  -- training/testing/deployed/retired
);

-- Strategy bandit state
CREATE TABLE strategy_bandit_state (
    strategy_name TEXT PRIMARY KEY,
    alpha REAL,  -- Success count
    beta REAL,   -- Failure count
    total_pnl REAL
);

-- Daily performance snapshots
CREATE TABLE daily_performance_snapshots (
    date TEXT PRIMARY KEY,
    current_bankroll REAL,
    total_return_pct REAL,
    daily_target_met INTEGER,
    current_drawdown REAL
);
```

---

## 🔍 Debugging Checklist

### No Trades Being Placed
- [ ] Check fixtures available: `SELECT COUNT(*) FROM fixtures_v2 WHERE kickoff_utc > datetime('now')`
- [ ] Check market mappings: `SELECT COUNT(*) FROM fixture_market_map`
- [ ] Check strategy selection: Review logs for `strategy_selected`
- [ ] Check edge thresholds: May be too high

### High Drawdown
- [ ] Review last 20 trades for pattern
- [ ] Check which strategy is losing: Query bandit state
- [ ] Verify position invalidation working: Check logs for `invalidation_exit`
- [ ] Consider reset: `bandit.reset_exploration()`

### Models Not Retraining
- [ ] Check drift detection: `await manager._detect_drift("football", active_model)`
- [ ] Verify training data: Need 100+ fixtures
- [ ] Check model_versions table: Look for status='testing'

---

## 📈 Success Milestones

### Week 1
- [ ] Agent running continuously
- [ ] At least 10 trades placed
- [ ] Position invalidation working (observed exits)
- [ ] No crashes or errors
- [ ] Return: Any positive return is good!

### Month 1
- [ ] 50+ trades executed
- [ ] Strategy bandit converged (1-2 dominant strategies)
- [ ] Model retrained at least once
- [ ] Return: 20-40% monthly (good!)
- [ ] Drawdown: <15%

### Month 3
- [ ] 200+ trades executed
- [ ] Multiple model updates
- [ ] Consistent weekly returns
- [ ] Return: 50-80% monthly (excellent!)
- [ ] Drawdown: <10%
- [ ] Sharpe: >1.5

---

## 💡 Pro Tips

1. **Start Small**: $100-500 bankroll for first month
2. **Monitor Daily**: Check performance tracker every day
3. **Trust the Invalidation**: Don't manually hold invalid positions
4. **Let Bandit Learn**: Takes 50-100 trades to converge
5. **Respect Drawdown**: 20% = HARD STOP, no exceptions
6. **Settlement Timing**: Remember P&L realizes at game end, not position open
7. **Be Patient**: System needs 3-4 weeks to show true potential

---

## 📞 Quick Support

- Logs: `logs/footbe_trader.log`
- Database: `data/footbe_dev.db`
- Config: `configs/strategy_config_aggressive.yaml`
- Docs: `SELF_IMPROVEMENT_GUIDE.md`
- Architecture: `SYSTEM_ARCHITECTURE.md`

**Emergency**: If drawdown >20%, agent auto-stops. Review, reduce sizes, resume carefully.

---

**Remember**: 10% daily = 3,000% annual. Even 5% daily (1,300% annual) would be extraordinary. Focus on learning and risk management first, returns will follow! 🚀
