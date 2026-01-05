#!/usr/bin/env python3
"""Sync leagues from API-Football to database.

Discovers available leagues from API-Football and stores them in the
local database with canonical league keys for cross-platform matching.

Usage:
    python scripts/sync_leagues.py [options]

Examples:
    # Sync all leagues
    python scripts/sync_leagues.py

    # Sync only English leagues
    python scripts/sync_leagues.py --country England

    # Sync leagues containing "Premier" in name
    python scripts/sync_leagues.py --name-contains Premier

    # List synced leagues
    python scripts/sync_leagues.py --list
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
from footbe_trader.football.client import FootballApiClient
from footbe_trader.storage.database import Database
from footbe_trader.strategy.league_discovery import (
    LeagueRepository,
    sync_leagues,
)

logger = get_logger(__name__)


def list_leagues(db_path: str) -> None:
    """List all leagues in the database."""
    conn = sqlite3.connect(db_path)
    repo = LeagueRepository(conn)
    
    leagues = repo.get_all_leagues()
    
    if not leagues:
        print("No leagues found in database. Run sync first.")
        return
    
    print(f"\n{'='*80}")
    print(f"{'League ID':<12} {'League Key':<25} {'Country':<15} {'Name':<30}")
    print(f"{'='*80}")
    
    for league in leagues:
        active = "✓" if league.is_active else " "
        seasons = f"[{len(league.seasons_available)} seasons]"
        print(
            f"{league.league_id:<12} "
            f"{(league.league_key or 'N/A'):<25} "
            f"{league.country:<15} "
            f"{league.league_name[:28]:<30} "
            f"{active} {seasons}"
        )
    
    print(f"\nTotal: {len(leagues)} leagues")
    conn.close()


async def run_sync(
    config_path: str,
    db_path: str,
    country: str | None,
    name_contains: str | None,
) -> None:
    """Run league sync."""
    # Load config
    config = load_config(config_path)
    
    # Initialize database
    db = Database(db_path)
    db.connect()
    db.migrate()
    
    # Sync leagues
    async with FootballApiClient(config.football_api) as client:
        result = await sync_leagues(
            client=client,
            db_connection=db.connection,
            country_filter=country,
            name_contains=name_contains,
        )
    
    # Print results
    print(f"\n{'='*60}")
    print("League Sync Results")
    print(f"{'='*60}")
    print(f"Total leagues found: {result.total_found}")
    print(f"New leagues added:   {result.new_leagues}")
    print(f"Leagues updated:     {result.updated_leagues}")
    
    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
    
    # Show sample of synced leagues
    if result.leagues:
        print(f"\nSample leagues:")
        for league in result.leagues[:10]:
            seasons = len(league.seasons_available)
            print(f"  {league.league_id}: {league.league_name} ({league.country}) - {seasons} seasons")
        
        if len(result.leagues) > 10:
            print(f"  ... and {len(result.leagues) - 10} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync leagues from API-Football to database",
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
        "--country",
        help="Filter by country name (e.g., 'England', 'Spain')",
    )
    parser.add_argument(
        "--name-contains",
        help="Filter by league name containing string",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List synced leagues instead of syncing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.list:
        list_leagues(args.db)
    else:
        asyncio.run(run_sync(
            config_path=args.config,
            db_path=args.db,
            country=args.country,
            name_contains=args.name_contains,
        ))


if __name__ == "__main__":
    main()
