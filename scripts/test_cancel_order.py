#!/usr/bin/env python3
"""Test order cancellation with detailed debugging."""

import asyncio
import os
import sys
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using existing environment")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.kalshi.client import KalshiClient


async def test_cancel():
    """Test listing and cancelling orders."""
    config = load_config("configs/dev.yaml")

    async with KalshiClient(config.kalshi) as client:
        print("=" * 60)
        print("TESTING ORDER CANCELLATION")
        print("=" * 60)
        print()

        # Step 1: List resting orders
        print("Step 1: Listing resting orders...")
        try:
            orders, _ = await client.list_orders(status="resting")
            print(f"✓ Successfully listed {len(orders)} resting orders")

            if not orders:
                print("\n⚠️  No resting orders found to test cancellation")
                print("   Place a test order first, then run this script")
                return

            print("\nResting orders:")
            for i, order in enumerate(orders[:5], 1):  # Show first 5
                print(f"  {i}. {order.ticker}: {order.side} @ ${order.price:.2f} (qty: {order.quantity})")
                print(f"     Order ID: {order.order_id}")

        except Exception as e:
            print(f"✗ Failed to list orders: {e}")
            return

        print()

        # Step 2: Try to cancel first order
        test_order = orders[0]
        print(f"Step 2: Attempting to cancel order {test_order.order_id}...")
        print(f"  Market: {test_order.ticker}")
        print(f"  Side: {test_order.side}")
        print(f"  Price: ${test_order.price:.2f}")
        print()

        try:
            success = await client.cancel_order(test_order.order_id)
            if success:
                print("✓✓✓ CANCELLATION SUCCESSFUL! ✓✓✓")
                print()
                print("The authentication issue is FIXED!")
                print("Stale order protection will now work correctly.")
            else:
                print("✗ Cancellation returned False (order may not exist)")

        except Exception as e:
            print(f"✗✗✗ CANCELLATION FAILED ✗✗✗")
            print(f"Error: {e}")
            print()
            print("This is the authentication error preventing stale order protection.")
            print()

            if "401" in str(e) or "authentication" in str(e).lower():
                print("DIAGNOSIS:")
                print("  - HTTP 401 authentication error")
                print("  - The API key signature is being rejected")
                print()
                print("POSSIBLE CAUSES:")
                print("  1. API key doesn't have 'cancel' permission")
                print("  2. Signature format issue for DELETE requests")
                print("  3. API key expired/needs regeneration")
                print()
                print("NEXT STEPS:")
                print("  1. Check Kalshi API key permissions at:")
                print("     https://kalshi.com → Settings → API")
                print("  2. Verify key has 'Trade' permission (not just 'Read')")
                print("  3. If needed, regenerate API key and update:")
                print("     - KALSHI_API_KEY_ID environment variable")
                print("     - KALSHI_PRIVATE_KEY_PATH to new key file")


if __name__ == "__main__":
    asyncio.run(test_cancel())
