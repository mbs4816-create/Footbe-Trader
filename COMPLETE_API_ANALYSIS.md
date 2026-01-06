# 🔍 Complete API Analysis & Configuration Review

**Date**: January 6, 2026
**Purpose**: Thorough examination of all API integrations and configurations

---

## 🎯 Executive Summary

### Key Findings:

1. **Portfolio API Auth**: Issue is NOT with permissions - yesterday's agent placed 64 orders successfully
2. **Today's Agent**: Has NOT attempted to place any orders yet (orders_placed=0)
3. **Order Expiration**: Currently NOT being set - defaults to "Good til Cancelled" (GTC)
4. **Root Cause**: Unknown - need to test if current credentials work for POST

---

## 1️⃣ Kalshi API Integration

### Configuration
**File**: `configs/dev.yaml`
```yaml
kalshi:
  base_url: https://api.elections.kalshi.com/trade-api/v2
  use_demo: false
  api_key_id: ${KALSHI_API_KEY_ID:-}
  private_key_path: ${KALSHI_PRIVATE_KEY_PATH:-}
  timeout_seconds: 30
```

### Authentication Method
- **Type**: RSA-PSS signature-based
- **Headers Required**:
  - `KALSHI-ACCESS-KEY`: API key ID
  - `KALSHI-ACCESS-SIGNATURE`: RSA-PSS signature
  - `KALSHI-ACCESS-TIMESTAMP`: Unix timestamp (ms)
- **Signature Format**: `{timestamp}{METHOD}{/trade-api/v2/path?params}`

### Endpoints Used

| Endpoint | Method | Purpose | Status |
|----------|--------|---------|--------|
| `/markets/{ticker}` | GET | Get market data | ✅ Working |
| `/markets/{ticker}/orderbook` | GET | Get orderbook | ✅ Working |
| `/portfolio/orders` | GET | List orders | ❌ HTTP 401 |
| `/portfolio/orders` | POST | Place order | ❓ Unknown (not tested today) |
| `/portfolio/orders/{id}` | DELETE | Cancel order | ❌ HTTP 401 |
| `/portfolio/balance` | GET | Get balance | ❓ Not tested |
| `/portfolio/positions` | GET | Get positions | ❓ Not tested |

### Yesterday's Performance
```
Date: 2026-01-05
Orders placed: 64 (51 partial, 13 rejected)
Auth status: WORKED (orders were placed)
```

### Today's Activity
```
Date: 2026-01-06
Orders placed: 0
GET /portfolio/orders: HTTP 401
DELETE /portfolio/orders: HTTP 401
```

### Critical Issue: Order Expiration

**Current Behavior**:
```python
# In place_limit_order() - NO expiration time set
body = {
    "ticker": ticker,
    "side": side,
    "action": action,
    "type": "limit",
    "count": quantity,
    "yes_price": price_cents if side == "yes" else None,
    "no_price": price_cents if side == "no" else None,
    # expiration_ts: NOT SET ❌
}
```

**Result**: Orders default to **"Good til Cancelled" (GTC)**

**Problem**: Pre-game orders remain resting after game starts

**Solution Options**:

1. **Set expiration to game start time**:
   ```python
   body["expiration_ts"] = int(fixture.kickoff_time.timestamp() * 1000)
   ```
   - Orders auto-expire at kickoff
   - No manual cancellation needed
   - Eliminates stale order problem entirely

2. **Use Kalshi's native expiration options**:
   - `GTC` - Good til Cancelled (current default)
   - `IOC` - Immediate or Cancel
   - `FOK` - Fill or Kill
   - Custom timestamp

---

## 2️⃣ Football API Integration

### Configuration
```yaml
football_api:
  base_url: https://v3.football.api-sports.io
  api_key: ${FOOTBALL_API_KEY:-}
  timeout_seconds: 30
  rate_limit_per_minute: 30
```

### Authentication Method
- **Type**: API Key header
- **Header**: `x-apisports-key: YOUR_API_KEY`

### Endpoints Used

| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/fixtures` | GET | Get fixtures | 30/min |
| `/fixtures/statistics` | GET | Match stats | 30/min |
| `/standings` | GET | League standings | 30/min |
| `/teams` | GET | Team info | 30/min |
| `/leagues` | GET | League info | 30/min |
| `/odds` | GET | Betting odds | 30/min |

### Current Usage
```python
# From football/client.py
async def get_fixtures(
    league_id: int,
    season: int,
    from_date: date | None = None,
    to_date: date | None = None,
) -> list[FixtureData]:
    # Fetches fixtures with pagination
    # Stores raw responses to disk
    # Rate limited to 30 requests/minute
```

### Integration Points
1. **Fixture Ingestion**: `scripts/ingest_fixtures.py`
2. **Live Game State**: `src/footbe_trader/agent/live_game.py`
   - Fetches live scores during games
   - Updates every iteration (15 min)
   - Used for position invalidation

---

## 3️⃣ NBA API Integration

### Configuration
```yaml
# Uses same config as football_api
football_api:
  api_key: ${FOOTBALL_API_KEY:-}  # Same key for both!
```

### Authentication Method
- **Type**: API Key header
- **Header**: `x-apisports-key: YOUR_API_KEY`
- **Base URL**: `https://v2.nba.api-sports.io` (different from football)

### Endpoints Used

| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/games` | GET | Get NBA games | 30/min |
| `/games/statistics` | GET | Game stats | 30/min |
| `/teams` | GET | NBA teams | 30/min |
| `/standings` | GET | NBA standings | 30/min |

### Current Usage
```python
# From nba/client.py
async def get_games_for_date(
    game_date: date,
    season: int | None = None
) -> list[NBAGame]:
    # Fetches NBA games for specific date
    # Same rate limiting as football (30/min)
    # Stores raw responses
```

### Integration Points
1. **NBA Game Ingestion**: `scripts/ingest_nba_games.py`
2. **Live Game State**: Also uses `src/footbe_trader/agent/live_game.py`
   - Supports both football and NBA
   - Unified interface for live scores

---

## 🔍 Root Cause Analysis: Portfolio Auth Issue

### Timeline

**Yesterday (2026-01-05)**:
- ✅ 64 orders placed successfully
- ✅ POST `/portfolio/orders` worked
- Portfolio auth was working

**Today (2026-01-06)**:
- ❌ GET `/portfolio/orders` → HTTP 401
- ❌ DELETE `/portfolio/orders/{id}` → HTTP 401
- ✅ GET `/markets/*` → Status 200
- ❓ POST `/portfolio/orders` not tested yet (no trading signals)

### Possible Explanations

#### 1. **Credentials Changed Between Yesterday and Today** ⭐ MOST LIKELY
- API key rotated/regenerated
- Private key file changed
- Environment variables not persisting
- Different credentials in yesterday's session vs today's

**How to Check**:
```bash
# Check if .env file was modified
stat .env

# Check environment variable values
echo $KALSHI_API_KEY_ID | head -c 20
echo $KALSHI_PRIVATE_KEY_PATH
```

#### 2. **Session-Based vs Per-Request Auth**
- Yesterday's agent had a valid session
- Today's agent starting fresh with same credentials
- Possible token expiration or session timeout

**How to Test**:
- Place a single test order today
- If POST works but GET doesn't, issue is with GET endpoints specifically
- If POST also fails, credentials are invalid

#### 3. **Rate Limiting or API Key Quota**
- Hit daily/hourly limit
- Kalshi throttling or blocking requests
- Different rate limits for READ vs WRITE

**How to Check**:
Check response headers for rate limit info

#### 4. **Signature Generation Bug for GET with Params**
- GET requests with query params might have signature issues
- `/portfolio/orders?status=resting&limit=100`
- Signature must include full path with query string

**Current Implementation** (client.py:117-124):
```python
if params:
    sorted_params = sorted((k, v) for k, v in params.items() if v is not None)
    if sorted_params:
        query_string = urlencode(sorted_params)
        full_path = f"{full_path}?{query_string}"  # Used for signing
```

This looks correct, but worth double-checking against Kalshi docs.

---

## 🛠️ Immediate Action Items

### 1. Test Current Credentials
```python
# Create test_kalshi_auth.py
import asyncio
from footbe_trader.common.config import load_config
from footbe_trader.kalshi.client import KalshiClient

async def test():
    config = load_config('configs/dev.yaml')
    async with KalshiClient(config.kalshi) as client:
        # Test 1: GET /markets (public) - should work
        market = await client.get_market("KXEPLGAME-26JAN07BOUTOT-BOU")
        print(f"✓ GET /markets: {market.ticker if market else 'FAIL'}")

        # Test 2: GET /portfolio/balance - portfolio endpoint
        try:
            balance = await client.get_balance()
            print(f"✓ GET /portfolio/balance: ${balance}")
        except Exception as e:
            print(f"✗ GET /portfolio/balance: {e}")

        # Test 3: GET /portfolio/orders - list orders
        try:
            orders, _ = await client.list_orders(limit=1)
            print(f"✓ GET /portfolio/orders: {len(orders)} orders")
        except Exception as e:
            print(f"✗ GET /portfolio/orders: {e}")

asyncio.run(test())
```

### 2. Add Order Expiration Time

**File**: `src/footbe_trader/kalshi/client.py`

```python
async def place_limit_order(
    self,
    ticker: str,
    side: str,
    action: str,
    price: float,
    quantity: int,
    client_order_id: str | None = None,
    expiration_ts: int | None = None,  # ADD THIS
) -> OrderData:
    """Place a limit order.

    Args:
        ...
        expiration_ts: Unix timestamp in seconds when order expires.
                      If None, order is Good Til Cancelled (GTC).
    """
    body: dict[str, Any] = {
        "ticker": ticker,
        "side": side,
        "action": action,
        "type": "limit",
        "count": quantity,
        "yes_price": price_cents if side == "yes" else None,
        "no_price": price_cents if side == "no" else None,
    }

    if client_order_id:
        body["client_order_id"] = client_order_id

    # ADD THIS: Set expiration time
    if expiration_ts:
        body["expiration_ts"] = expiration_ts

    # Remove None values
    body = {k: v for k, v in body.items() if v is not None}

    # ... rest of function
```

**Then in strategy**:
```python
# When creating order, pass expiration time
expiration_ts = int(fixture.kickoff_time.timestamp())
order = await client.place_limit_order(
    ...,
    expiration_ts=expiration_ts
)
```

### 3. Check Yesterday's Credentials

```bash
# Check what credentials were used yesterday
grep "KALSHI" .env | head -2

# Check if env file was modified recently
ls -la .env
```

---

## 📊 API Health Dashboard

Create a script to test all APIs:

```bash
#!/bin/bash
# scripts/test_all_apis.sh

echo "=== API Health Check ==="
echo ""

# 1. Football API
echo "1. Football API..."
curl -s -H "x-apisports-key: $FOOTBALL_API_KEY" \
     "https://v3.football.api-sports.io/status" | \
     jq -r '.response | "Requests: \(.requests.current)/\(.requests.limit_day)"'
echo ""

# 2. NBA API (same key)
echo "2. NBA API..."
curl -s -H "x-apisports-key: $FOOTBALL_API_KEY" \
     "https://v2.nba.api-sports.io/status" | \
     jq -r '.response | "Requests: \(.requests.current)/\(.requests.limit_day)"'
echo ""

# 3. Kalshi API (need python for RSA signature)
echo "3. Kalshi API..."
python3 scripts/test_kalshi_auth.py
```

---

## 🎯 Recommendations

### Short Term (Do Today):

1. ✅ **Test current credentials** with all portfolio endpoints
2. ✅ **Add order expiration** to prevent GTC orders
3. ✅ **Set expiration to game kickoff time** for all new orders
4. ⚠️ **Manually check resting orders** on Kalshi.com and cancel risky ones

### Medium Term (This Week):

1. **Implement proper order lifecycle**:
   - Pre-game orders: Expire at kickoff
   - In-game orders: Expire at game end
   - Add configurable expiration buffer (e.g., 5 min before event)

2. **Add portfolio endpoint monitoring**:
   - Log all portfolio API calls
   - Track auth success/failure rates
   - Alert on repeated auth failures

3. **Enhance API testing**:
   - Create comprehensive test suite for all endpoints
   - Test auth with fresh credentials vs cached
   - Verify signature generation for edge cases

### Long Term (This Month):

1. **API credential rotation**:
   - Support for multiple API keys
   - Automatic failover if primary key fails
   - Credential health monitoring

2. **Order management improvements**:
   - Bulk order cancellation
   - Order modification (instead of cancel + replace)
   - Advanced order types (stop-loss, trailing stop)

3. **Multi-API resilience**:
   - Fallback data sources
   - Cross-validation of live scores
   - API quota management across Football + NBA

---

## 📈 API Usage Statistics

### Kalshi API
- **Rate Limit**: Unknown (likely 100-300 req/min)
- **Yesterday**: 64 POST requests (orders placed)
- **Today**: ~100 GET requests (market data)
- **Failed**: All `/portfolio/*` GET/DELETE requests

### Football API
- **Rate Limit**: 30 requests/minute
- **Daily Quota**: 100 requests/day (free tier)
- **Current Usage**: Unknown - check with status endpoint

### NBA API
- **Rate Limit**: 30 requests/minute (shared with Football)
- **Daily Quota**: Shared 100/day
- **Current Usage**: Minimal (season just started)

---

## 🔑 Key Takeaways

1. **Order expiration is critical** - Currently ALL orders are GTC
2. **Portfolio auth worked yesterday** - Issue is environment/credentials, not permissions
3. **Need to test POST today** - Verify if current creds can place orders
4. **Football/NBA APIs are simple** - Just API key headers, no complex auth
5. **Set expiration=kickoff** - This solves the stale order problem entirely

---

**Next Steps**: Run the test scripts above to identify exactly what's broken with portfolio auth.
