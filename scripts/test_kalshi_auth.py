#!/usr/bin/env python3
"""Test Kalshi API authentication comprehensively."""

import asyncio
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.kalshi.client import KalshiClient


async def test_all_endpoints():
    """Test all Kalshi API endpoint categories."""
    config = load_config("configs/dev.yaml")

    print("=" * 60)
    print("KALSHI API AUTHENTICATION TEST")
    print("=" * 60)
    print()

    async with KalshiClient(config.kalshi) as client:
        results = {}

        # Test 1: Public market data (no auth required?)
        print("1. Testing PUBLIC market endpoints...")
        try:
            market = await client.get_market("KXEPLGAME-26JAN07BOUTOT-BOU")
            if market:
                print(f"   ✓ GET /markets/{{ticker}}: {market.ticker}")
                results["markets_get"] = "✓"
            else:
                print("   ✗ Market not found (404)")
                results["markets_get"] = "✗ (404)"
        except Exception as e:
            print(f"   ✗ GET /markets: {e}")
            results["markets_get"] = f"✗ {type(e).__name__}"

        print()

        # Test 2: Portfolio balance
        print("2. Testing PORTFOLIO endpoints...")
        try:
            balance = await client.get_balance()
            print(f"   ✓ GET /portfolio/balance: ${balance:.2f}")
            results["portfolio_balance"] = "✓"
        except Exception as e:
            error_str = str(e)
            if "401" in error_str:
                print(f"   ✗ GET /portfolio/balance: HTTP 401 (Auth failed)")
                results["portfolio_balance"] = "✗ HTTP 401"
            else:
                print(f"   ✗ GET /portfolio/balance: {e}")
                results["portfolio_balance"] = f"✗ {type(e).__name__}"

        print()

        # Test 3: List orders
        try:
            orders, cursor = await client.list_orders(limit=5)
            print(f"   ✓ GET /portfolio/orders: Found {len(orders)} orders")
            results["portfolio_orders_list"] = "✓"

            if orders:
                print(f"\n   First order:")
                print(f"      ID: {orders[0].order_id}")
                print(f"      Ticker: {orders[0].ticker}")
                print(f"      Status: {orders[0].status}")
                print(f"      Side: {orders[0].side} {orders[0].action}")
                print(f"      Price: ${orders[0].price:.2f}")
        except Exception as e:
            error_str = str(e)
            if "401" in error_str:
                print(f"   ✗ GET /portfolio/orders: HTTP 401 (Auth failed)")
                results["portfolio_orders_list"] = "✗ HTTP 401"
            else:
                print(f"   ✗ GET /portfolio/orders: {e}")
                results["portfolio_orders_list"] = f"✗ {type(e).__name__}"

        print()

        # Test 4: List positions
        try:
            positions = await client.list_positions()
            print(f"   ✓ GET /portfolio/positions: Found {len(positions)} positions")
            results["portfolio_positions"] = "✓"
        except Exception as e:
            error_str = str(e)
            if "401" in error_str:
                print(f"   ✗ GET /portfolio/positions: HTTP 401 (Auth failed)")
                results["portfolio_positions"] = "✗ HTTP 401"
            else:
                print(f"   ✗ GET /portfolio/positions: {e}")
                results["portfolio_positions"] = f"✗ {type(e).__name__}"

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print()

        all_pass = all("✓" in v for v in results.values())

        for endpoint, status in results.items():
            symbol = "✓" if "✓" in status else "✗"
            print(f"{symbol} {endpoint:30} {status}")

        print()

        if all_pass:
            print("🎉 ALL TESTS PASSED!")
            print("Your Kalshi API credentials are working correctly.")
            print()
            print("The stale order cancellation should now work.")
        else:
            failed = [k for k, v in results.items() if "401" in v]
            if failed:
                print("❌ AUTHENTICATION FAILED for:")
                for endpoint in failed:
                    print(f"   - {endpoint}")
                print()
                print("DIAGNOSIS:")
                print("  Your API key does NOT have permission for portfolio endpoints.")
                print()
                print("FIX:")
                print("  1. Log into https://kalshi.com")
                print("  2. Go to Settings → API")
                print("  3. Check your API key permissions")
                print("  4. If missing 'Trade' permission, regenerate key")
                print("  5. Download new private key (.pem file)")
                print("  6. Update .env:")
                print("     KALSHI_API_KEY_ID=your-new-key-id")
                print("     KALSHI_PRIVATE_KEY_PATH=/path/to/new-key.pem")
                print("  7. Restart agent and retest")
            else:
                print("⚠️  SOME TESTS FAILED (not auth related)")
                print("Check the error messages above.")

        print()
        return all_pass


if __name__ == "__main__":
    success = asyncio.run(test_all_endpoints())
    sys.exit(0 if success else 1)
