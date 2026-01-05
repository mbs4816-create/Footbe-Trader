#!/usr/bin/env python3
"""Kalshi API smoketest script.

This script tests the Kalshi API integration by:
1. Fetching account balance
2. Listing a small set of markets
3. Fetching orderbook for one ticker
4. Storing the orderbook snapshot in the database

Usage:
    python scripts/kalshi_smoketest.py

Before running, ensure your .env file has:
    KALSHI_API_KEY_ID=your-key-id
    KALSHI_PRIVATE_KEY_PATH=/path/to/your/private-key.pem
    KALSHI_ENVIRONMENT=demo  # or 'production'
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime

import structlog

from footbe_trader.common.config import load_config
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.kalshi.interfaces import OrderbookData
from footbe_trader.storage.database import Database
from footbe_trader.storage.models import OrderbookSnapshot

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)

logger = structlog.get_logger()


def create_orderbook_snapshot(ticker: str, orderbook: OrderbookData) -> OrderbookSnapshot:
    """Convert orderbook data to a snapshot for storage."""
    now = datetime.now()
    
    # Calculate volumes using the actual OrderbookData structure
    bid_volume = orderbook.total_bid_volume
    ask_volume = orderbook.total_ask_volume
    
    return OrderbookSnapshot(
        id=None,
        timestamp=now,
        ticker=ticker,
        best_bid=orderbook.best_yes_bid,
        best_ask=orderbook.best_yes_ask,
        mid=orderbook.mid_price,
        spread=orderbook.spread,
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        volume=bid_volume + ask_volume,
        raw_json={
            "yes_bids": [{"price": l.price, "quantity": l.quantity} for l in orderbook.yes_bids],
            "yes_asks": [{"price": l.price, "quantity": l.quantity} for l in orderbook.yes_asks],
        },
        created_at=now,
    )


async def run_smoketest() -> int:
    """Run the Kalshi API smoketest."""
    print("=" * 60)
    print("Kalshi API Smoketest")
    print("=" * 60)
    print()

    # Load configuration
    try:
        config = load_config()
        logger.info(
            "config_loaded",
            use_demo=config.kalshi.use_demo,
            has_key_id=bool(config.kalshi.api_key_id),
        )
    except Exception as e:
        logger.error("config_load_failed", error=str(e))
        print(f"\n❌ Failed to load config: {e}")
        print("Make sure your .env file is configured correctly.")
        return 1

    # Initialize client and run tests
    try:
        async with KalshiClient(config.kalshi) as client:
            env_type = "DEMO" if config.kalshi.use_demo else "PRODUCTION"
            print(f"🔌 Connected to Kalshi API ({env_type})")

            # 1. Fetch account balance
            print("\n📊 Step 1: Fetching account balance...")
            try:
                balance = await client.get_balance()
                print(f"   ✅ Balance: ${balance.balance:.2f}")
                print(f"   💰 Portfolio Value: ${balance.portfolio_value:.2f}")
            except Exception as e:
                logger.error("balance_fetch_failed", error=str(e))
                print(f"   ❌ Failed to fetch balance: {e}")
                # Continue anyway - balance might require elevated permissions

            # 2. List markets
            print("\n📋 Step 2: Listing markets...")
            try:
                # Try to find active markets
                markets, _ = await client.list_markets(status="open", limit=10)
                print(f"   ✅ Found {len(markets)} open markets")
                
                for i, market in enumerate(markets[:5], 1):
                    title = market.title[:50] if len(market.title) > 50 else market.title
                    print(f"   {i}. {market.ticker}: {title}...")
                    
                if not markets:
                    print("   ⚠️  No open markets found")
                    return 0
                    
            except Exception as e:
                logger.error("markets_fetch_failed", error=str(e))
                print(f"   ❌ Failed to fetch markets: {e}")
                return 1

            # 3. Fetch orderbook for first market
            print("\n📈 Step 3: Fetching orderbook...")
            selected_market = markets[0]
            try:
                orderbook = await client.get_orderbook(selected_market.ticker)
                print(f"   ✅ Orderbook for {selected_market.ticker}")
                print(f"   📊 Best Bid: ${orderbook.best_yes_bid:.2f} | Best Ask: ${orderbook.best_yes_ask:.2f}")
                print(f"   📏 Spread: ${orderbook.spread:.2f} | Mid: ${orderbook.mid_price:.2f}")
                print(f"   📚 Depth: {len(orderbook.yes_bids)} bids, {len(orderbook.yes_asks)} asks")
            except Exception as e:
                logger.error("orderbook_fetch_failed", error=str(e), ticker=selected_market.ticker)
                print(f"   ❌ Failed to fetch orderbook: {e}")
                return 1

            # 4. Store orderbook snapshot in database
            print("\n💾 Step 4: Storing orderbook snapshot...")
            try:
                # Initialize database with migrations
                db_path = config.storage.database_path
                db = Database(db_path)
                db.initialize()
                
                # Create and store snapshot
                snapshot = create_orderbook_snapshot(selected_market.ticker, orderbook)
                snapshot_id = db.create_orderbook_snapshot(snapshot)
                
                print(f"   ✅ Stored snapshot with ID: {snapshot_id}")
                print(f"   📍 Database: {db_path}")
                
                # Verify by reading back
                snapshots = db.get_orderbook_snapshots(ticker=selected_market.ticker, limit=1)
                if snapshots:
                    print(f"   ✅ Verified: Retrieved snapshot for {snapshots[0].ticker}")
                
                db.close()
            except Exception as e:
                logger.error("snapshot_store_failed", error=str(e))
                print(f"   ❌ Failed to store snapshot: {e}")
                return 1

    except Exception as e:
        logger.error("client_init_failed", error=str(e))
        print(f"\n❌ Failed to initialize client: {e}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("✅ Smoketest completed successfully!")
    print("=" * 60)
    
    return 0


def main() -> int:
    """Entry point."""
    return asyncio.run(run_smoketest())


if __name__ == "__main__":
    sys.exit(main())
