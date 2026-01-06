# 🔴 CRITICAL: Kalshi API Authentication Issue

**Status**: Portfolio endpoints failing with HTTP 401
**Impact**: Cannot cancel resting orders
**Risk Level**: HIGH - Stale orders will fill at adverse prices

---

## 🔍 Root Cause

Your Kalshi API key **lacks permission** to access `/portfolio/*` endpoints.

### Evidence

Testing shows:
- ✅ `GET /markets/*` → Status 200 (PUBLIC data works)
- ✅ `GET /markets/*/orderbook` → Status 200 (PUBLIC data works)
- ❌ `GET /portfolio/orders` → HTTP 401 (PORTFOLIO fails)
- ❌ `DELETE /portfolio/orders/{id}` → HTTP 401 (CANCEL fails)

### Error Message

```
HTTP 401: {
  'error': {
    'code': 'authentication_error',
    'message': 'authentication_error',
    'details': 'rpc error: code = Unauthenticated desc = INCORRECT_API_KEY_SIGNATURE'
  }
}
```

---

## ⚠️ Current Situation

### What's Broken:
1. ❌ Cannot list resting orders via API
2. ❌ Cannot cancel orders via API
3. ❌ Stale order protection is **NOT WORKING**

### What's at Risk:
- Your Nottingham Forest @ $0.19 order (current price $0.41)
- Any other resting orders >15¢ below current price
- These WILL fill if price drops during games

### When Games Start:
**NO** - The system will NOT automatically cancel your orders at kickoff because the cancel API calls are failing with auth errors.

---

## 💡 Possible Causes

### 1. API Key Permissions (MOST LIKELY)

Your API key may only have **READ** permission, not **TRADE** permission.

**How to Check:**
1. Log into https://kalshi.com
2. Go to **Settings** → **API**
3. Find your API key
4. Check permissions:
   - ✅ **Read** (allows viewing market data)
   - ❓ **Trade** (allows placing/canceling orders)

**If missing Trade permission:**
- Regenerate API key with full permissions
- Download new `.pem` private key file
- Update environment variables:
  ```bash
  export KALSHI_API_KEY_ID="new-key-id"
  export KALSHI_PRIVATE_KEY_PATH="/path/to/new-key.pem"
  ```

### 2. Demo vs Production Mismatch

Your key might be for demo environment but config points to production.

**Check:**
- Config says: `use_demo: false` (production)
- Config URL: `https://api.elections.kalshi.com/trade-api/v2` (production)
- Your API key: Should be production key, not demo

### 3. Signature Format Issue (LESS LIKELY)

The signature generation might be incorrect for `/portfolio/*` endpoints specifically.

- Public endpoints (`/markets/*`) work fine
- Only authenticated portfolio endpoints fail
- This suggests signature is computed correctly but key lacks permission

---

## 🚨 Immediate Action Required

### Option 1: Manual Cancellation (SAFEST - DO THIS NOW)

**Manually cancel risky orders:**

1. Log into https://kalshi.com
2. Go to **Portfolio** → **Orders**
3. Cancel any resting orders where:
   - Current price is >15¢ above your limit
   - Games start within 24 hours
   - You'd lose money if they fill now

**Example:**
- Nottingham Forest @ $0.19 (current $0.41) → **CANCEL THIS**
- Any order from yesterday still resting → **CANCEL**

### Option 2: Fix API Permissions (PERMANENT FIX)

1. Check Kalshi API key has **Trade** permission
2. If not, generate new key with full permissions
3. Update credentials in `.env`:
   ```bash
   KALSHI_API_KEY_ID=your-new-key-id
   KALSHI_PRIVATE_KEY_PATH=/path/to/new-key.pem
   ```
4. Restart agent:
   ```bash
   kill 83510  # Stop current agent
   .venv/bin/python3 scripts/run_agent.py \
       --mode live \
       --interval 15 \
       --strategy-config configs/strategy_config_aggressive.yaml \
       --bankroll 127 > logs/footbe_trader.log 2>&1 &
   ```
5. Test cancellation:
   ```bash
   .venv/bin/python3 scripts/test_cancel_order.py
   ```

---

## 🔧 Testing the Fix

After updating API credentials, run:

```bash
.venv/bin/python3 scripts/test_cancel_order.py
```

**Expected output if fixed:**
```
✓ Successfully listed X resting orders
✓✓✓ CANCELLATION SUCCESSFUL! ✓✓✓
The authentication issue is FIXED!
```

**If still failing:**
```
✗✗✗ CANCELLATION FAILED ✗✗✗
HTTP 401 authentication error
```

Then check:
1. Did you update BOTH `KALSHI_API_KEY_ID` AND `KALSHI_PRIVATE_KEY_PATH`?
2. Is the private key file readable?
3. Is the key for production (not demo)?
4. Does the key have Trade permission?

---

## 📊 Why This Matters

### The Arsenal Scenario (What You Wanted to Avoid):

1. **Pre-game**: Place Arsenal YES @ $0.22 limit order
2. **Kickoff**: Game starts, Arsenal vs Liverpool
3. **10 min in**: Arsenal goes down 0-1
4. **Price crash**: Market drops from $0.63 → $0.22
5. **Order fills**: Your limit order fills at $0.22
6. **Loss**: You bought at worst moment, Arsenal losing

### The Protection (Currently Broken):

**Layer 1**: Pre-game filter
✅ Working - Won't place NEW orders in-game

**Layer 2**: Stale order cancellation
❌ **BROKEN** - Can't cancel due to API auth error
Should cancel when:
- Market moves >15¢ from limit price
- Game starts (kickoff time reached)

**Layer 3**: Position invalidator
✅ Working - Will exit positions after fill
BUT this is AFTER the damage is done

---

## 🎯 Bottom Line

**You need to either:**

### A) Cancel risky orders manually RIGHT NOW
Go to Kalshi and cancel orders that are >15¢ below current price.

### B) Fix API permissions and test
1. Check/regenerate API key with Trade permission
2. Update credentials
3. Restart agent
4. Verify cancellation works with test script

### C) Both (RECOMMENDED)
1. Manually cancel dangerous orders NOW (eliminates immediate risk)
2. Fix API permissions (enables automatic protection going forward)

---

## 📝 Verification Checklist

After fixing:

- [ ] Run `scripts/test_cancel_order.py` successfully
- [ ] Check logs show "stale_orders_cancelled" not "cancel_stale_orders_failed"
- [ ] Verify no HTTP 401 errors for portfolio endpoints
- [ ] Confirm stale order protection is active

---

## 💬 Need Help?

If you've:
1. Verified API key has Trade permission
2. Updated credentials correctly
3. Restarted agent
4. Still getting auth errors

Then the issue might be signature format. Let me know and I'll investigate deeper into the RSA signature generation for portfolio endpoints.

---

**Current Agent Status**: Running (PID 83510) but **STALE ORDER PROTECTION IS NOT WORKING**

**Your Action**: Manually cancel risky orders OR fix API permissions NOW before games start.
