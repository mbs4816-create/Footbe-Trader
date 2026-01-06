#!/bin/bash
# Launch Self-Improving Trading Agent in LIVE MODE
#
# ⚠️ WARNING: This trades REAL MONEY on Kalshi
# ⚠️ Targets: 10-12% daily returns (EXTREMELY AGGRESSIVE)
# ⚠️ Max Drawdown: 20% (will auto-stop)
#
# Make sure you understand the risks before proceeding!

set -e

echo "=============================================="
echo "  SELF-IMPROVING TRADING AGENT - LIVE MODE"
echo "=============================================="
echo ""
echo "⚠️  WARNING: LIVE TRADING WITH REAL MONEY"
echo ""
echo "Target: 10-12% daily returns"
echo "Initial Bankroll: \$100"
echo "Max Drawdown: 20% (auto-stop)"
echo "Interval: 15 minutes"
echo ""
echo "Strategy: Multi-Armed Bandit (5 strategies)"
echo "  - Ultra Aggressive (50% Kelly, 2% edge)"
echo "  - Aggressive (35% Kelly, 3% edge)"
echo "  - Balanced (25% Kelly, 5% edge)"
echo "  - Opportunistic (40% Kelly, 8% edge)"
echo "  - In-Game Specialist (20% Kelly, 6% edge)"
echo ""
echo "Safety Features:"
echo "  ✓ Position invalidation (exits when game state changes)"
echo "  ✓ Drawdown protection (stops at 20%)"
echo "  ✓ Automated model retraining"
echo "  ✓ Daily performance tracking"
echo ""
read -p "Are you ABSOLUTELY SURE you want to proceed? (type 'YES' to confirm): " confirm

if [ "$confirm" != "YES" ]; then
    echo "Launch cancelled. Good decision to be cautious!"
    exit 1
fi

echo ""
echo "Proceeding with LIVE trading..."
echo ""

# Check if Kalshi credentials are configured
if [ ! -f "configs/dev.yaml" ]; then
    echo "❌ Error: configs/dev.yaml not found"
    exit 1
fi

# Check database exists
if [ ! -f "data/footbe_dev.db" ]; then
    echo "❌ Error: Database not found. Run data ingestion first."
    exit 1
fi

# Create necessary database tables
echo "Setting up self-improvement tables..."
python3 <<EOF
import sys
sys.path.insert(0, 'src')
from footbe_trader.storage.database import Database
from footbe_trader.common.config import load_config

config = load_config('configs/dev.yaml')
db = Database(config.database.path)
db.connect()

cursor = db.connection.cursor()

# Model versions
cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        model_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        training_window_days INTEGER,
        training_samples INTEGER,
        hyperparameters TEXT,
        validation_accuracy REAL,
        validation_log_loss REAL,
        validation_sharpe REAL,
        status TEXT,
        deployed_at TEXT,
        retired_at TEXT,
        artifact_path TEXT
    )
""")

# Strategy bandit state
cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_bandit_state (
        strategy_name TEXT PRIMARY KEY,
        alpha REAL NOT NULL,
        beta REAL NOT NULL,
        total_trades INTEGER DEFAULT 0,
        winning_trades INTEGER DEFAULT 0,
        total_pnl REAL DEFAULT 0.0,
        times_selected INTEGER DEFAULT 0,
        last_updated TEXT
    )
""")

# Daily performance snapshots
cursor.execute("""
    CREATE TABLE IF NOT EXISTS daily_performance_snapshots (
        date TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        current_bankroll REAL NOT NULL,
        total_return_pct REAL NOT NULL,
        daily_target_met INTEGER,
        weekly_on_pace INTEGER,
        current_drawdown REAL,
        report_json TEXT
    )
""")

db.connection.commit()
db.close()

print("✓ Database tables created")
EOF

echo ""
echo "🚀 Launching live trading agent..."
echo "📊 Monitor: tail -f logs/footbe_trader.log"
echo "📱 Telegram alerts enabled (if configured)"
echo ""
echo "Press Ctrl+C to stop (graceful shutdown)"
echo ""

# Launch with aggressive config
python3 scripts/run_agent.py \
    --mode live \
    --interval 15 \
    --strategy-config configs/strategy_config_aggressive.yaml \
    --bankroll 100
