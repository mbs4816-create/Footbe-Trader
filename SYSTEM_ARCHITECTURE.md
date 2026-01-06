# Self-Improving Trading Agent - System Architecture

## 🏗️ Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVING TRADING AGENT                      │
│                  Target: 10-12% Daily | Max DD: 20%                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│  Football API  │  NBA API  │  Kalshi Markets  │  Live Game State   │
│  (fixtures)    │  (games)  │  (orderbooks)    │  (scores/timing)   │
└────────┬─────────────┬──────────────┬─────────────────┬─────────────┘
         │             │              │                 │
         ▼             ▼              ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                            CORE ENGINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STRATEGY BANDIT (Thompson Sampling)                         │  │
│  │  • 5 strategies (Ultra Aggressive → In-Game)                 │  │
│  │  • Beta(α,β) beliefs per strategy                            │  │
│  │  • Dynamic selection based on performance                    │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  EDGE STRATEGY (Selected Variant)                            │  │
│  │  • Edge calculation: model_prob - market_ask                 │  │
│  │  • Kelly sizing: position = fraction * bankroll / price      │  │
│  │  • Entry filters: edge, liquidity, timing                    │  │
│  │  • Exit rules: TP/SL, edge flip, time decay                  │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ORDER EXECUTION                                              │  │
│  │  • Paper trading simulator (with slippage)                   │  │
│  │  • Live Kalshi API (limit orders)                            │  │
│  │  • Fill tracking and position management                     │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       RISK MANAGEMENT LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  POSITION INVALIDATOR                                         │  │
│  │  ✓ Scans positions every iteration                           │  │
│  │  ✓ Checks: adverse score, price moves, staleness             │  │
│  │  ✓ Exits: 25%/50%/75%/100% based on severity                 │  │
│  │  ✓ CRITICAL: Pre-game positions become invalid in-game       │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  DRAWDOWN MONITOR                                             │  │
│  │  • Track peak bankroll                                        │  │
│  │  • Calculate current drawdown                                 │  │
│  │  • HARD STOP at 20% drawdown                                  │  │
│  │  • Alert at 15% (warning threshold)                           │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PERFORMANCE TRACKING LAYER                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  DAILY PERFORMANCE TRACKER                                    │  │
│  │  • Daily targets: 10% (low), 11% (mid), 12% (high)           │  │
│  │  • Tracks realized + unrealized P&L                           │  │
│  │  • Projects P&L by settlement date                            │  │
│  │  • Calculates pace towards target                             │  │
│  │  • Generates alerts when behind                               │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  WEEKLY PROJECTION                                            │  │
│  │  • Compound daily targets: 70-113% weekly                     │  │
│  │  • Days ahead/behind schedule                                 │  │
│  │  • Settlement timing analysis                                 │  │
│  │  • Over-trading prevention                                    │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SELF-IMPROVEMENT LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  MODEL LIFECYCLE MANAGER                                      │  │
│  │  1. Monitor: Detect performance drift                         │  │
│  │     └─> Sharpe drop >30% OR accuracy drop >2%                │  │
│  │  2. Retrain: On rolling 180-day window                        │  │
│  │     └─> New hyperparameters, fresh features                   │  │
│  │  3. A/B Test: Champion vs Challenger (7 days)                 │  │
│  │     └─> Track trades, calculate Sharpe                        │  │
│  │  4. Deploy: Promote if >10% improvement                       │  │
│  │     └─> Retire old champion, activate new                     │  │
│  └────────────────────────────────┬─────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  STRATEGY EVOLUTION                                           │  │
│  │  • Bandit updates beliefs: α (success), β (failure)           │  │
│  │  • Strategies compete: best ones selected more often          │  │
│  │  • Natural selection: poor strategies fade out                │  │
│  │  • Adaptation: responds to market regime changes              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         MONITORING & ALERTS                          │
├─────────────────────────────────────────────────────────────────────┤
│  • Telegram: Daily summaries, pace alerts, emergency stops          │
│  • Logs: Structured JSON logging for all decisions                  │
│  • Database: SQLite persistence for all state                       │
│  • Dashboard: (TODO) Real-time web UI                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Main Loop Flow

```
START
  │
  ├─> Initialize Components
  │   ├─> Strategy Bandit (load state)
  │   ├─> Model Lifecycle (start monitoring)
  │   ├─> Performance Tracker (load history)
  │   └─> Position Invalidator (ready)
  │
  ├─> ⏰ Every 30 minutes:
  │   │
  │   ├─> 1. INVALIDATE POSITIONS
  │   │   ├─> Scan all open positions
  │   │   ├─> Check game state changes
  │   │   ├─> Exit invalid positions (25%/50%/75%/100%)
  │   │   └─> Update P&L
  │   │
  │   ├─> 2. GET TRADEABLE FIXTURES
  │   │   ├─> Query database for upcoming games
  │   │   ├─> Check market mappings exist
  │   │   ├─> Verify within trading window
  │   │   └─> Filter out untradeable
  │   │
  │   ├─> 3. FOR EACH FIXTURE:
  │   │   │
  │   │   ├─> SELECT STRATEGY (Bandit)
  │   │   │   ├─> Sample from Beta posteriors
  │   │   │   ├─> Pick highest sample
  │   │   │   └─> Log selection
  │   │   │
  │   │   ├─> EVALUATE ENTRY (Strategy)
  │   │   │   ├─> Get model probabilities
  │   │   │   ├─> Calculate edge vs market
  │   │   │   ├─> Check entry filters
  │   │   │   ├─> Size position (Kelly)
  │   │   │   └─> Generate order params
  │   │   │
  │   │   ├─> EXECUTE ORDER
  │   │   │   ├─> Paper: Simulate fill
  │   │   │   └─> Live: Place Kalshi order
  │   │   │
  │   │   └─> UPDATE BANDIT
  │   │       └─> (After settlement) Update α/β
  │   │
  │   ├─> 4. UPDATE PERFORMANCE
  │   │   ├─> Calculate realized P&L
  │   │   ├─> Calculate unrealized P&L
  │   │   ├─> Project settlement dates
  │   │   ├─> Calculate pace vs target
  │   │   └─> Check drawdown limit
  │   │
  │   ├─> 5. PERSIST STATE
  │   │   ├─> Save bandit beliefs (α,β)
  │   │   ├─> Save daily snapshot
  │   │   ├─> Save positions
  │   │   └─> Commit to database
  │   │
  │   └─> 6. SEND ALERTS
  │       ├─> Telegram: Performance summary
  │       ├─> Telegram: Pace alerts
  │       └─> Telegram: Emergency stops
  │
  ├─> 🔄 In Background (24hr):
  │   │
  │   └─> MODEL LIFECYCLE
  │       ├─> Check recent performance
  │       ├─> If drift detected: RETRAIN
  │       ├─> If new model ready: A/B TEST
  │       └─> If challenger wins: DEPLOY
  │
  └─> ⚠️ EMERGENCY STOP if:
      ├─> Drawdown > 20%
      ├─> User signal (Ctrl+C)
      └─> Critical error
```

---

## 📊 Data Flow

### Entry Decision
```
1. FIXTURE DATA
   ├─> Home Team: Arsenal
   ├─> Away Team: Chelsea
   ├─> Kickoff: 2026-01-08 15:00 UTC
   └─> League: Premier League

2. MODEL PREDICTION
   ├─> Home Win: 45%
   ├─> Draw: 25%
   ├─> Away Win: 30%
   └─> Confidence: 65%

3. MARKET DATA
   ├─> HOME_WIN ticker: SOCCER-EPL-20260108-ARS-CHE-H
   ├─> Ask: $0.40
   ├─> Bid: $0.38
   └─> Volume: 1000

4. EDGE CALCULATION
   ├─> Edge = Model Prob - Ask Price
   ├─> Edge = 0.45 - 0.40 = 0.05 (5%)
   └─> Pass: Edge > min_edge (0.02)

5. STRATEGY BANDIT
   ├─> Sample Ultra Aggressive: 0.67
   ├─> Sample Aggressive: 0.54
   ├─> Sample Balanced: 0.49
   ├─> Select: Ultra Aggressive
   └─> Config: 50% Kelly, 2% edge min

6. KELLY SIZING
   ├─> Win Prob: 45%
   ├─> Odds: (1-0.40)/0.40 = 1.5
   ├─> Kelly: 0.45 - 0.55/1.5 = 0.083
   ├─> Adjusted: 0.083 * 0.50 = 0.042 (4.2%)
   ├─> Bet: $1000 * 0.042 = $42
   └─> Position: $42 / $0.40 = 105 contracts
                 (Capped at 50 per market)

7. ORDER EXECUTION
   ├─> Ticker: SOCCER-EPL-20260108-ARS-CHE-H
   ├─> Side: YES
   ├─> Action: BUY
   ├─> Price: $0.40
   ├─> Quantity: 50
   └─> Total Cost: $20

8. RESULT (After game)
   ├─> Outcome: Arsenal wins 2-1 (HOME WIN)
   ├─> Settlement: YES contracts pay $1
   ├─> Profit: (50 * $1) - $20 = $30
   ├─> ROI: $30/$20 = 150%
   └─> Update Bandit: α += 1 (success)
```

### Exit Decision (Invalidation)
```
1. POSITION STATE
   ├─> Entry: Bought HOME WIN @ $0.40
   ├─> Quantity: 50 contracts
   ├─> Cost: $20
   └─> Thesis: Pre-game model says Arsenal 45% fair

2. GAME STARTS
   ├─> Time: 10 minutes elapsed
   ├─> Score: Chelsea 1 - Arsenal 0
   └─> Market: HOME WIN now trading @ $0.25

3. INVALIDATION CHECK
   ├─> Adverse Score: ✓ (bought HOME, team losing)
   ├─> Price Movement: ✓ (dropped 37.5%)
   ├─> Game Started: ✓ (pre-game thesis now invalid)
   └─> Severity: CRITICAL

4. EXIT DECISION
   ├─> Reasons: [ADVERSE_SCORE, PRICE_MOVEMENT]
   ├─> Critical Count: 2
   ├─> Recommendation: EXIT 100%
   └─> Urgency: IMMEDIATE

5. EXECUTE EXIT
   ├─> Action: SELL 50 contracts
   ├─> Exit Price: $0.25 (current market)
   ├─> Proceeds: 50 * $0.25 = $12.50
   ├─> Loss: $12.50 - $20 = -$7.50
   └─> Loss %: -37.5%

6. ALTERNATIVE (Without Invalidation)
   ├─> Hold to settlement
   ├─> Final Score: Chelsea 2 - Arsenal 0
   ├─> Settlement: HOME WIN pays $0
   └─> Loss: $0 - $20 = -$20 (100% loss!)

7. BENEFIT
   ├─> Saved: $20 - $7.50 = $12.50
   └─> Prevented: 62.5% additional loss
```

---

## 🎯 Target Achievement Path

### Daily Progress Tracker
```
DAY 1 (Monday)
  Bankroll: $1,000
  Target: $110 (11%)

  Morning:
    ├─> Open 5 positions @ $200 total
    └─> Unrealized P&L: +$50 (games Saturday)

  Status:
    ├─> Realized: $0 (nothing settled yet)
    ├─> Unrealized: +$50
    ├─> Projected Saturday: +$50
    └─> Pace: ON TRACK (already have $50 projected)

DAY 2 (Tuesday)
  Morning:
    ├─> Open 3 positions @ $120 total
    └─> Unrealized P&L: +$30 (games Saturday)

  Status:
    ├─> Realized: $0
    ├─> Unrealized: +$80 total
    ├─> Projected Saturday: +$80
    └─> Pace: ON TRACK (don't over-trade!)

DAY 3 (Saturday - Games Settle)
  Games finish:
    ├─> Won 6/8 positions
    ├─> Realized: +$65 actual (vs +$80 projected)
    └─> Win rate: 75%

  Status:
    ├─> 3-day return: $65 / $1000 = 6.5%
    ├─> Effective daily: 2.1% per day
    └─> Pace: BEHIND (need to increase volume)

WEEK 1 RESULTS
  Target: $1,700 (70% weekly at 10% daily)
  Actual: $1,200 (20% weekly)
  Status: BEHIND but sustainable

  Action:
    ├─> Increase position sizes
    ├─> Lower min_edge threshold
    └─> Enable in-game trading
```

---

## 🔧 Configuration Hierarchy

```
CONSERVATIVE → MODERATE → AGGRESSIVE → ULTRA AGGRESSIVE

min_edge:         0.08   →   0.05   →   0.03   →   0.02
kelly_fraction:   0.10   →   0.25   →   0.35   →   0.50
max_exposure:     $250   →   $500   →   $750   →  $1000
position_size:     10    →    20    →    30    →     50

Expected Return:   2%    →    5%    →    8%    →    12%
Max Drawdown:      5%    →   10%    →   15%    →    20%
Win Rate:         52%    →   50%    →   48%    →    45%
Sharpe:          2.0    →   1.5    →   1.0    →    0.8
```

---

## 📈 Performance Tracking

### Metrics Dashboard
```
┌────────────────────────────────────────────────────────────┐
│                    DAILY PERFORMANCE                       │
├────────────────────────────────────────────────────────────┤
│  Date: 2026-01-06                                          │
│  Bankroll: $1,085 (+8.5%)                                  │
│                                                            │
│  Today's Target: $110 (11%)                                │
│  Today's Actual: $85 (8.5%)                                │
│  Progress: 77% ⚠️ Behind Pace                              │
│                                                            │
│  Realized P&L: $45                                         │
│  Unrealized P&L: $40                                       │
│  Total: $85                                                │
│                                                            │
│  Positions:                                                │
│    Open: 8                                                 │
│    Exposure: $280 (28% of bankroll)                        │
│    Pending Settlement: 8 (tomorrow)                        │
│                                                            │
│  Risk:                                                     │
│    Current Drawdown: 3.2%                                  │
│    Peak Bankroll: $1,120                                   │
│    Remaining Capacity: $189                                │
│                                                            │
│  Weekly Projection:                                        │
│    Target: $1,700 (70%)                                    │
│    Projected: $1,450 (45%)                                 │
│    Days Behind: -1.2                                       │
│                                                            │
│  Strategy Performance:                                     │
│    Ultra Aggressive: Sharpe 1.2, Win 48%, 15 trades       │
│    Aggressive: Sharpe 0.9, Win 45%, 12 trades             │
│    Balanced: Sharpe 1.5, Win 52%, 8 trades ⭐             │
│                                                            │
│  Alerts:                                                   │
│    ⚠️ Behind daily pace by 23%                            │
│    ✓ Drawdown healthy (<5%)                               │
│    ℹ️ 8 positions settling tomorrow                       │
└────────────────────────────────────────────────────────────┘
```

---

## 🚀 Evolution Over Time

```
WEEK 1: Exploration & Calibration
  ├─> Bandit explores all 5 strategies
  ├─> High variance (exploration penalty)
  ├─> Return: 15-25% weekly
  └─> Drawdown: 10-15%

WEEK 2-4: Convergence
  ├─> Bandit identifies 2-3 winning strategies
  ├─> Focus shifts to exploitation
  ├─> Return: 25-40% weekly
  └─> Drawdown: 8-12%

MONTH 2: Optimization
  ├─> Models retrained with recent data
  ├─> Poor strategies eliminated
  ├─> Parameters fine-tuned
  ├─> Return: 40-60% weekly
  └─> Drawdown: 5-10%

MONTH 3+: Maturity
  ├─> System stable and consistent
  ├─> 1-2 dominant strategies
  ├─> Models stay current
  ├─> Return: 50-80% weekly (if sustainable)
  └─> Drawdown: <10%

REALITY CHECK:
  Most likely outcome: 20-40% weekly (excellent!)
  Aspirational: 70% weekly (10% daily)
  Extraordinary: 113% weekly (12% daily)
```

---

This architecture enables **true self-improvement** through:
1. **Adaptive strategy selection** (bandit learns what works)
2. **Automatic model updates** (prevents staleness)
3. **Dynamic risk management** (exits invalid positions)
4. **Realistic performance tracking** (accounts for settlement delays)

The system **learns, adapts, and evolves** - the hallmark of a self-improving agent! 🚀
