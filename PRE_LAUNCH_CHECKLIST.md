# Pre-Launch Checklist for Live Trading

## ⚠️ CRITICAL: Read This Before Live Trading

You are about to launch an **aggressive trading system** targeting **10-12% daily returns** with real money. This is **EXTREMELY HIGH RISK**.

---

## 🔐 Step 1: Configure Kalshi API Credentials

### Get API Credentials from Kalshi
1. Log into [Kalshi](https://kalshi.com)
2. Go to Settings → API
3. Generate API key and download private key file
4. **IMPORTANT**: Fund your Kalshi account with exactly $100 for this test

### Set Environment Variables

```bash
# Add to ~/.zshrc or ~/.bashrc
export KALSHI_API_KEY_ID="your-api-key-id-here"
export KALSHI_PRIVATE_KEY_PATH="/path/to/your/kalshi_private_key.pem"

# Reload shell
source ~/.zshrc  # or source ~/.bashrc
```

### Verify Credentials

```bash
# Check variables are set
echo $KALSHI_API_KEY_ID
echo $KALSHI_PRIVATE_KEY_PATH

# Test API connection
python3 scripts/kalshi_smoketest.py
```

---

## 📊 Step 2: Verify Data Pipeline

### Check Database Has Recent Data

```bash
sqlite3 data/footbe_dev.db <<EOF
-- Check fixtures
SELECT COUNT(*) as fixture_count FROM fixtures_v2
WHERE kickoff_utc > datetime('now');

-- Check market mappings
SELECT COUNT(*) as mapping_count FROM fixture_market_map;

-- Check Kalshi markets
SELECT COUNT(*) as market_count FROM kalshi_markets;
EOF
```

**Required**:
- At least 10 upcoming fixtures
- Market mappings for those fixtures
- Kalshi market data synced

### Run Data Ingestion If Needed

```bash
# Ingest fixtures
python3 scripts/ingest_fixtures.py --league-id 39 --season 2025

# Ingest NBA games
python3 scripts/ingest_nba_games.py --season 2024

# Map to Kalshi markets
python3 scripts/map_fixtures_to_markets.py

# Sync Kalshi market data
python3 scripts/sync_kalshi_markets.py
```

---

## 🧪 Step 3: Backtest the Aggressive Strategy

**DO NOT skip this step!**

```bash
# Backtest on last 90 days
python3 scripts/backtest_strategy.py \
    --config configs/strategy_config_aggressive.yaml \
    --start-date 2024-10-01 \
    --end-date 2025-01-01 \
    --initial-bankroll 100

# Review results
# - Sharpe ratio should be >0.8
# - Max drawdown should be <25%
# - Win rate should be >45%
```

**If backtest shows poor results**: Adjust parameters before going live!

---

## 📱 Step 4: Configure Telegram Alerts (HIGHLY RECOMMENDED)

### Create Telegram Bot
1. Message @BotFather on Telegram
2. Create new bot: `/newbot`
3. Save the bot token

### Get Your Chat ID
```bash
# Start chat with your bot, then:
curl https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
# Look for "chat":{"id": YOUR_CHAT_ID}
```

### Configure in env vars
```bash
export TELEGRAM_BOT_TOKEN="your-bot-token-here"
export TELEGRAM_CHAT_ID="your-chat-id-here"
```

---

## ⚙️ Step 5: Review Configuration

### Check Strategy Config

```bash
cat configs/strategy_config_aggressive.yaml
```

**Key settings**:
- `min_edge_to_enter: 0.02` (2% edge minimum)
- `kelly_fraction: 0.50` (50% Kelly - VERY AGGRESSIVE)
- `max_global_exposure: 1000.0` ($1000 max exposure)
- `stop_loss: 0.15` (15 cent stop loss)

### Adjust If Too Aggressive

For first week, consider:
```yaml
kelly_fraction: 0.25  # 25% Kelly (less aggressive)
max_global_exposure: 50.0  # $50 max (50% of $100 bankroll)
max_exposure_per_fixture: 20.0  # $20 per fixture
```

---

## 🚦 Step 6: Final Safety Checks

### Confirm These Are True:

- [ ] **I have funded Kalshi with $100 (money I can afford to lose)**
- [ ] **I understand 10-12% daily = 3,000% annual (extremely rare)**
- [ ] **I know the system will auto-stop at 20% drawdown**
- [ ] **I have backtest results showing the strategy works historically**
- [ ] **I have Telegram alerts configured for monitoring**
- [ ] **I will check performance daily and can intervene if needed**
- [ ] **I understand this is HIGH RISK experimental trading**

### Review Risks:

| Risk | Mitigation |
|------|------------|
| **Rapid losses** | 20% drawdown auto-stop |
| **Over-trading** | Position limits, exposure caps |
| **Bad fills** | Position invalidation exits bad trades |
| **Model drift** | Automated retraining |
| **API failures** | Graceful error handling, retries |
| **Market manipulation** | Stale order detection |

---

## 🚀 Step 7: Launch Process

### Initial Setup (First Time Only)

```bash
# Create self-improvement tables
python3 <<EOF
import sys
sys.path.insert(0, 'src')
from footbe_trader.storage.database import Database
from footbe_trader.common.config import load_config

config = load_config('configs/dev.yaml')
db = Database(config.database.path)
db.connect()
db.migrate()
cursor = db.connection.cursor()

# Model versions
cursor.execute("""CREATE TABLE IF NOT EXISTS model_versions (
    model_id TEXT PRIMARY KEY, model_name TEXT, version TEXT,
    created_at TEXT, validation_sharpe REAL, status TEXT
)""")

# Strategy bandit
cursor.execute("""CREATE TABLE IF NOT EXISTS strategy_bandit_state (
    strategy_name TEXT PRIMARY KEY, alpha REAL, beta REAL,
    total_pnl REAL, times_selected INTEGER
)""")

# Daily performance
cursor.execute("""CREATE TABLE IF NOT EXISTS daily_performance_snapshots (
    date TEXT PRIMARY KEY, current_bankroll REAL,
    total_return_pct REAL, current_drawdown REAL
)""")

db.connection.commit()
db.close()
print("✓ Database initialized")
EOF
```

### Launch Live Trading

```bash
# Method 1: Use launch script
./scripts/launch_live_trading.sh

# Method 2: Direct command
python3 scripts/run_agent.py \
    --mode live \
    --interval 15 \
    --strategy-config configs/strategy_config_aggressive.yaml \
    --bankroll 100

# Method 3: With dry-run (simulate fills, no real orders)
python3 scripts/run_agent.py \
    --mode live \
    --interval 15 \
    --strategy-config configs/strategy_config_aggressive.yaml \
    --bankroll 100 \
    --dry-run
```

---

## 📊 Step 8: Monitor Performance

### Watch Logs Live
```bash
tail -f logs/footbe_trader.log
```

### Check Daily Performance
```bash
sqlite3 data/footbe_dev.db \
  "SELECT * FROM daily_performance_snapshots ORDER BY date DESC LIMIT 1"
```

### Strategy Bandit State
```bash
sqlite3 data/footbe_dev.db \
  "SELECT strategy_name, alpha, beta, total_pnl, times_selected
   FROM strategy_bandit_state ORDER BY total_pnl DESC"
```

### Current Positions
```bash
sqlite3 data/footbe_dev.db \
  "SELECT ticker, quantity, average_entry_price, unrealized_pnl
   FROM paper_positions WHERE is_open = 1"
```

---

## 🛑 Step 9: Emergency Procedures

### If Drawdown Approaching 18%
```bash
# Stop the agent immediately (Ctrl+C)
^C

# Review recent trades
sqlite3 data/footbe_dev.db \
  "SELECT * FROM decision_records ORDER BY timestamp DESC LIMIT 20"

# Reduce position sizes before resuming
# Edit configs/strategy_config_aggressive.yaml:
#   max_position_per_market: 5 (from 50)
#   max_exposure_per_fixture: 10 (from 250)
```

### If Too Many Position Invalidations
```bash
# Increase pre-game buffer
# Edit configs/strategy_config_aggressive.yaml:
in_game:
  min_minutes_before_kickoff: 15  # From 2
  require_pre_game: true
```

### If Models Performing Poorly
```bash
# Force model retrain
python3 <<EOF
import sys, asyncio
sys.path.insert(0, 'src')
from footbe_trader.self_improvement.model_lifecycle import ModelLifecycleManager
from footbe_trader.storage.database import Database
from footbe_trader.common.config import load_config

config = load_config('configs/dev.yaml')
db = Database(config.database.path)
db.connect()

manager = ModelLifecycleManager(db, config)
asyncio.run(manager._retrain_model("football"))
print("✓ Model retrained")
EOF
```

---

## 📈 Expected First Week

### Realistic Expectations
- **Day 1-2**: High exploration, may see losses (bandit learning)
- **Day 3-5**: Strategies converging, variance decreasing
- **Day 6-7**: Clear winner emerging, more consistent

### Performance Targets
| Metric | Conservative | Moderate | Aggressive | Reality |
|--------|--------------|----------|------------|---------|
| Daily Return | 1-2% | 3-5% | 8-10% | **Varies wildly** |
| Win Rate | 55% | 52% | 48% | 45-60% |
| Sharpe | 2.0 | 1.5 | 0.8 | 0.5-2.5 |
| Max DD | 5% | 10% | 15% | **10-20%** |

**First week goal**: Don't lose >15%. Learning is the priority!

---

## ✅ Pre-Launch Checklist Summary

Run this final check:

```bash
#!/bin/bash
echo "=== PRE-LAUNCH VERIFICATION ==="
echo ""

# Credentials
if [ -n "$KALSHI_API_KEY_ID" ]; then echo "✓ Kalshi API key configured"; else echo "❌ Missing KALSHI_API_KEY_ID"; fi
if [ -n "$KALSHI_PRIVATE_KEY_PATH" ]; then echo "✓ Private key path set"; else echo "❌ Missing KALSHI_PRIVATE_KEY_PATH"; fi

# Data
fixture_count=$(sqlite3 data/footbe_dev.db "SELECT COUNT(*) FROM fixtures_v2 WHERE kickoff_utc > datetime('now')" 2>/dev/null || echo "0")
echo "Upcoming fixtures: $fixture_count"
if [ "$fixture_count" -gt 10 ]; then echo "✓ Sufficient fixtures"; else echo "❌ Need more fixtures"; fi

# Telegram (optional but recommended)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then echo "✓ Telegram configured"; else echo "⚠ Telegram not configured (optional)"; fi

echo ""
echo "If all checks pass, you're ready to launch!"
echo "Run: ./scripts/launch_live_trading.sh"
```

---

## 🎯 Post-Launch Monitoring

### Daily (REQUIRED)
- Check Telegram alerts
- Review daily performance snapshot
- Verify drawdown <15%
- Check for position invalidations

### Weekly (REQUIRED)
- Analyze strategy bandit convergence
- Review model retraining events
- Calculate actual vs target returns
- Adjust parameters if needed

### Monthly
- Full performance review
- Compare to backtest expectations
- Decide to scale up or adjust

---

## 📞 Support & Resources

- **Logs**: `tail -f logs/footbe_trader.log`
- **Database**: `sqlite3 data/footbe_dev.db`
- **Documentation**: `SELF_IMPROVEMENT_GUIDE.md`
- **Quick Reference**: `QUICK_REFERENCE.md`

---

## ⚠️ Final Warning

You are about to trade with **real money** using an **experimental system** targeting **extremely aggressive returns**.

**10-12% daily = 3,000% annual return**

This is **extraordinarily rare**. Even hedge funds that achieve 20-30% annual are considered exceptional.

**Proceed only if**:
1. You can afford to lose 100% of the $100
2. You understand the risks
3. You will monitor daily
4. You will respect the 20% drawdown limit

**Good luck! Trade responsibly.** 🚀
