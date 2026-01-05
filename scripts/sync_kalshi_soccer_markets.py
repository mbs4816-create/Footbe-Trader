#!/usr/bin/env python3
"""Sync Kalshi soccer markets to database.

Discovers and classifies soccer-related markets from Kalshi,
storing them with parsed team names and league information.

Usage:
    python scripts/sync_kalshi_soccer_markets.py [options]

Examples:
    # Sync all soccer markets
    python scripts/sync_kalshi_soccer_markets.py

    # Sync with verbose output
    python scripts/sync_kalshi_soccer_markets.py -v

    # List synced soccer markets
    python scripts/sync_kalshi_soccer_markets.py --list
"""

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger, setup_logging
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.storage.database import Database
from footbe_trader.strategy.kalshi_discovery import (
    KalshiMarketRepository,
    sync_kalshi_soccer_markets,
)

logger = get_logger(__name__)


def list_soccer_markets(db_path: str) -> None:
    """List soccer markets in the database."""
    conn = sqlite3.connect(db_path)
    repo = KalshiMarketRepository(conn)
    
    events = repo.get_soccer_events()
    
    if not events:
        print("No soccer markets found in database. Run sync first.")
        return
    
    print(f"\n{'='*100}")
    print(f"{'Event Ticker':<35} {'League':<20} {'Home':<20} {'Away':<20}")
    print(f"{'='*100}")
    
    for event in events:
        league = event.league_key or "Unknown"
        home = (event.parsed_canonical_home or event.parsed_home_team or "?")[:18]
        away = (event.parsed_canonical_away or event.parsed_away_team or "?")[:18]
        
        print(f"{event.event_ticker:<35} {league:<20} {home:<20} {away:<20}")
        
        # Get markets for this event
        markets = repo.get_markets_for_event(event.event_ticker)
        for market in markets:
            mtype = market.market_type or "?"
            print(f"    └─ {market.ticker:<30} [{mtype}] {market.title[:40]}")
    
    print(f"\nTotal: {len(events)} soccer events")
    conn.close()


async def run_sync(config_path: str, db_path: str, days_ahead: int) -> None:
    """Run Kalshi market sync."""
    # Load config
    config = load_config(config_path)
    
    # Initialize database
    db = Database(db_path)
    db.connect()
    db.migrate()
    
    print(f"\n{'='*60}")
    print("Syncing Kalshi Soccer Markets")
    print(f"{'='*60}\n")
    
    async with KalshiClient(config.kalshi) as client:
        result = await sync_kalshi_soccer_markets(
            client=client,
            db_connection=db.connection,
            days_ahead=days_ahead,
        )
    
    # Print results
    print(f"\n{'='*60}")
    print("Sync Results")
    print(f"{'='*60}")
    print(f"Total events found:    {result.events_found}")
    print(f"Soccer events:         {result.events_soccer}")
    print(f"Total markets fetched: {result.markets_found}")
    print(f"Soccer markets:        {result.markets_soccer}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync Kalshi soccer markets to database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--config",
        default="configs/dev.yaml",
        help="Path to configuration file (default: configs/dev.yaml)",
    )
    parser.add_argument(
        "--db",
        default="data/footbe.db",
        help="Path to SQLite database (default: data/footbe.db)",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=30,
        help="Days ahead to fetch markets for (default: 30)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List synced soccer markets instead of syncing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging (verbose mode uses structlog directly)
    if args.verbose:
        import structlog
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )
    
    if args.list:
        list_soccer_markets(args.db)
    else:
        asyncio.run(run_sync(
            config_path=args.config,
            db_path=args.db,
            days_ahead=args.days_ahead,
        ))


if __name__ == "__main__":
    main()
