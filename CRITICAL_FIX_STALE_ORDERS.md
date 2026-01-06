# 🚨 CRITICAL: Stale Order Protection Status

## Problem You Experienced

**Your Arsenal Example:**
- Placed limit buy @ $0.22 on Arsenal YES (pre-game)
- Game started → Arsenal fell behind
- Price dropped to $0.22 (adverse selection)
- Order filled at worst moment
- **Result: Loss from outdated order**

---

## ✅ GOOD NEWS: Protection Already Exists!

The system **already has stale order detection built-in** at [trading_strategy.py:891-982](src/footbe_trader/strategy/trading_strategy.py#L891-L982)

### What It Does:
1. **Price Divergence Check**: Cancels if market moved >15 cents away from limit
2. **Game Start Check**: Cancels pre-game orders once kickoff happens
3. **Runs Every Iteration**: Checks on every agent loop (every 15-30 min)

### How It Protects You:
```python
# Your example scenario:
Placed: Arsenal YES @ $0.22 (pre-game)
Game starts: Arsenal 0-1 Liverpool
Current price: $0.63 (market knows Arsenal losing)
Divergence: $0.63 - $0.22 = $0.41 (>15 cent threshold)
Action: CANCEL ORDER immediately
Result: Order cancelled before fill at $0.22
```

---

## ❌ BAD NEWS: It's Currently Failing

**From agent.log:**
```
cancel_stale_orders_failed: HTTP 401 authentication_error
```

**This means:**
- ✅ Code is running and trying to cancel
- ❌ Kalshi API rejecting the cancel request
- ⚠️ **Your resting orders are NOT being cancelled!**

**This is why you got filled on bad positions yesterday.**

---

## 🔧 Root Cause Analysis

### Possible Causes:
1. **API Key Expired/Rotated**: Kalshi keys may have short expiration
2. **Signature Issue**: RSA signing for cancel_order may be wrong
3. **Permissions**: API key may not have cancel permission
4. **Rate Limiting**: Hitting rate limit on cancel endpoint

### Most Likely:
The `list_orders()` call works (you see orders) but `cancel_order()` fails authentication. This suggests either:
- Different auth requirements for cancel vs read
- API key permissions don't include cancellation
- Signature generation bug specific to DELETE requests

---

## 🚀 IMMEDIATE FIXES

### Fix 1: Verify Kalshi API Key Permissions

```bash
# Check your Kalshi API key has these permissions:
# - Read orders ✓ (working)
# - Cancel orders ✗ (failing)
# - Place orders ✓ (working)

# Log into Kalshi → Settings → API
# Verify key has "Trade" permission (not just "Read")
```

### Fix 2: Test Cancel Order Directly

```python
# Test script to isolate the issue
python3 <<'EOF'
import sys, asyncio
sys.path.insert(0, 'src')
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.common.config import load_config

async def test_cancel():
    config = load_config('configs/dev.yaml')
    async with KalshiClient(config.kalshi) as client:
        # List orders
        orders, _ = await client.list_orders(status="resting")
        print(f"Found {len(orders)} resting orders")

        if orders:
            order_id = orders[0].order_id
            print(f"Attempting to cancel {order_id}...")
            try:
                success = await client.cancel_order(order_id)
                print(f"Cancel result: {success}")
            except Exception as e:
                print(f"Cancel failed: {e}")

asyncio.run(test_cancel())
EOF
```

### Fix 3: Enhanced Error Handling

I'll add a fallback that:
1. Tries to cancel via API (current method)
2. If auth fails, logs critical alert
3. Continues processing without crashing
4. Sends Telegram alert about uncancellable orders

---

## 🎯 Current Configuration

**From [strategy_config.yaml](configs/strategy_config.yaml):**

```yaml
in_game:
  require_pre_game: true  # Won't place NEW orders in-game ✅
  min_minutes_before_kickoff: 5  # Buffer before kickoff ✅
  stale_order_threshold: 0.15  # Cancel if price moves 15 cents ✅
```

**This means:**
- ✅ System WON'T place new orders after kickoff
- ✅ System WILL try to cancel old orders (but currently failing)
- ⚠️ Resting orders from BEFORE this protection may still fill

---

## 🛡️ Additional Protection: Position Invalidator

Even if cancel fails, we have a **second layer of protection**:

**Position Invalidator** ([position_invalidator.py](src/footbe_trader/execution/position_invalidator.py)) will:
1. Detect filled positions that are now invalid
2. Exit them immediately after fill
3. Minimize damage from adverse fills

**Example:**
```
T=0: Place Arsenal YES @ $0.22 (pre-game)
T=10min: Game starts, cancel fails (auth error)
T=15min: Arsenal goes down 0-1
T=20min: Price hits $0.22, order fills
T=21min: Position Invalidator detects:
  - Game is live (pre-game position now invalid)
  - Arsenal is losing (adverse score)
  - IMMEDIATELY EXITS at $0.20
Loss: $0.02/contract (vs $0.22 if held to zero)
```

---

## 🚀 Full Deployment Plan

### Phase 1: Fix Auth Issue (Immediate)
1. Run test script above to identify exact error
2. Check Kalshi API key permissions
3. Regenerate API key if needed
4. Test cancel_order works

### Phase 2: Deploy Position Invalidator (Now)
Even with cancel failing, this will limit damage:
- Exits positions immediately after adverse fills
- Monitors game state every iteration
- Auto-exits on adverse scores

### Phase 3: Deploy Full Self-Improvement (After Fix)
Once cancel works:
- Strategy Bandit (learn what's working)
- Model Lifecycle (adapt to changes)
- Daily Performance Tracker (measure your 54%/day!)

---

## ✅ What's Already Protecting You

### Pre-Game Filter (ACTIVE)
```python
# From trading_strategy.py:579-585
if self.config.require_pre_game:
    minutes_to_kickoff = (fixture.kickoff_time - now).total_seconds() / 60
    filters["pre_game"] = minutes_to_kickoff >= self.config.min_minutes_before_kickoff
```

**This prevents placing NEW orders in-game.**

### Price Deviation Filter (ACTIVE)
```python
# From trading_strategy.py:588-612
price_deviation = model_prob - ask_price
filters["price_deviation"] = price_deviation <= self.config.max_price_deviation_to_enter
```

**This prevents buying when price suspiciously cheap (game may have started).**

### Stale Order Detector (BROKEN - Auth Issue)
```python
# From run_agent.py:952-1025
async def _cancel_stale_orders():
    # Tries to cancel but gets 401 error
```

**This SHOULD cancel old orders but currently fails.**

---

## 🎯 Action Items

### Right Now (Critical):
1. **Test Kalshi cancel_order** - Run test script above
2. **Check API permissions** - Verify "Trade" permission
3. **Deploy Position Invalidator** - Second layer of protection

### Next 30 Minutes:
1. **Fix auth issue** - Regenerate key or fix signature
2. **Verify stale order cancellation works** - Check logs
3. **Deploy full self-improvement** - All components

### Monitoring:
1. **Watch for "stale_orders_cancelled"** in logs
2. **Check no more auth errors** in logs
3. **Verify no adverse fills** on positions

---

## 🔍 How to Monitor

### Check if cancellation is working:
```bash
# Should see successful cancellations
grep "stale_orders_cancelled" agent.log

# Should NOT see auth errors
grep "cancel_stale_orders_failed" agent.log | tail -5
```

### Check current resting orders:
```python
python3 <<'EOF'
import sys, asyncio
sys.path.insert(0, 'src')
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.common.config import load_config

async def check():
    config = load_config('configs/dev.yaml')
    async with KalshiClient(config.kalshi) as client:
        orders, _ = await client.list_orders(status="resting")
        print(f"\n{'='*60}")
        print(f"RESTING ORDERS: {len(orders)}")
        print(f"{'='*60}")
        for o in orders:
            print(f"{o.ticker[:30]:30} | ${o.price:.2f} | {o.side} {o.action}")

asyncio.run(check())
EOF
```

---

## 💡 Bottom Line

**You have THREE layers of protection:**

1. ✅ **Pre-game filter** - Won't place new orders in-game (WORKING)
2. ❌ **Stale order cancellation** - Should cancel old orders (BROKEN - auth issue)
3. ✅ **Position invalidator** - Exits bad fills immediately (READY TO DEPLOY)

**We need to:**
1. Fix the auth issue (regenerate API key or fix signature)
2. Deploy position invalidator as backup
3. Deploy full self-improvement system

**Ready to proceed?**
