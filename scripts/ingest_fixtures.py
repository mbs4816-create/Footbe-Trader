#!/usr/bin/env python3
"""Ingest fixtures from API-Football for any league.

Pulls fixtures for a specific league and season from API-Football
and stores them in the database.

Usage:
    python scripts/ingest_fixtures.py --league_id <id> --season <year> [options]

Examples:
    # Ingest current EPL season
    python scripts/ingest_fixtures.py --league_id 39 --season 2025

    # Ingest La Liga fixtures
    python scripts/ingest_fixtures.py --league_id 140 --season 2025

    # Ingest only upcoming fixtures (next 14 days)
    python scripts/ingest_fixtures.py --league_id 39 --season 2025 --days-ahead 14

    # Ingest MLS fixtures
    python scripts/ingest_fixtures.py --league_id 253 --season 2025
"""

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.football.client import FootballApiClient
from footbe_trader.storage.database import Database
from footbe_trader.storage.models import FixtureV2, Team

logger = get_logger(__name__)


# Common league IDs for reference
LEAGUE_IDS = {
    "epl": 39,
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    "mls": 253,
    "liga_mx": 262,
    "champions_league": 2,
    "europa_league": 3,
}


async def ingest_fixtures(
    config_path: str,
    db_path: str,
    league_id: int,
    season: int,
    days_ahead: int | None = None,
    include_historical: bool = True,
) -> None:
    """Ingest fixtures for a league/season."""
    # Load config
    config = load_config(config_path)
    
    # Initialize database
    db = Database(db_path)
    db.connect()
    db.migrate()
    
    # Calculate date range
    from_date = None
    to_date = None
    
    if days_ahead is not None:
        from_date = datetime.now(UTC)
        to_date = from_date + timedelta(days=days_ahead)
    elif not include_historical:
        from_date = datetime.now(UTC)
    
    print(f"\n{'='*60}")
    print(f"Ingesting fixtures for League {league_id}, Season {season}")
    if from_date:
        print(f"Date range: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d') if to_date else 'end of season'}")
    print(f"{'='*60}\n")
    
    async with FootballApiClient(config.football_api) as client:
        # Get league info first
        try:
            league_info, seasons = await client.get_league_info(league_id)
            print(f"League: {league_info.name} ({league_info.country})")
        except Exception as e:
            print(f"Warning: Could not fetch league info: {e}")
        
        # Fetch fixtures
        fixtures = await client.get_fixtures(
            league_id=league_id,
            season=season,
            from_date=from_date,
            to_date=to_date,
        )
        
        print(f"Found {len(fixtures)} fixtures")
        
        # Store fixtures and teams
        teams_seen = set()
        fixtures_inserted = 0
        fixtures_updated = 0
        
        for fixture in fixtures:
            # Upsert teams if not seen
            if fixture.home_team_id not in teams_seen:
                team = Team(
                    team_id=fixture.home_team_id,
                    name=fixture.home_team_name,
                    country="",
                )
                db.upsert_team(team)
                teams_seen.add(fixture.home_team_id)
            
            if fixture.away_team_id not in teams_seen:
                team = Team(
                    team_id=fixture.away_team_id,
                    name=fixture.away_team_name,
                    country="",
                )
                db.upsert_team(team)
                teams_seen.add(fixture.away_team_id)
            
            # Check if fixture exists
            existing = db.get_fixture_by_id(fixture.fixture_id)
            
            # Create FixtureV2 object
            db_fixture = FixtureV2(
                fixture_id=fixture.fixture_id,
                league_id=fixture.league_id,
                season=fixture.season,
                round=fixture.round,
                home_team_id=fixture.home_team_id,
                away_team_id=fixture.away_team_id,
                kickoff_utc=fixture.kickoff_utc,
                status=fixture.status.value,
                home_goals=fixture.home_goals,
                away_goals=fixture.away_goals,
                venue=fixture.venue,
                referee=fixture.referee,
                raw_json=fixture.raw_data or {},
            )
            db.upsert_fixture(db_fixture)
            
            if existing:
                fixtures_updated += 1
            else:
                fixtures_inserted += 1
        
        # Print results
        print(f"\nResults:")
        print(f"  Teams processed:    {len(teams_seen)}")
        print(f"  Fixtures inserted:  {fixtures_inserted}")
        print(f"  Fixtures updated:   {fixtures_updated}")
        
        # Show sample fixtures
        if fixtures:
            print(f"\nSample fixtures:")
            for f in fixtures[:5]:
                status = f.status.value
                score = ""
                if f.home_goals is not None and f.away_goals is not None:
                    score = f" ({f.home_goals}-{f.away_goals})"
                kickoff = f.kickoff_utc.strftime("%Y-%m-%d %H:%M") if f.kickoff_utc else "TBD"
                print(f"  {kickoff}: {f.home_team_name} vs {f.away_team_name}{score} [{status}]")
            
            if len(fixtures) > 5:
                print(f"  ... and {len(fixtures) - 5} more")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest fixtures from API-Football",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--league_id", "--league-id",
        type=int,
        required=True,
        help="API-Football league ID (e.g., 39 for EPL, 140 for La Liga)",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2025)",
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
        help="Only fetch fixtures for next N days",
    )
    parser.add_argument(
        "--upcoming-only",
        action="store_true",
        help="Only fetch upcoming fixtures (no historical)",
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
    
    asyncio.run(ingest_fixtures(
        config_path=args.config,
        db_path=args.db,
        league_id=args.league_id,
        season=args.season,
        days_ahead=args.days_ahead,
        include_historical=not args.upcoming_only,
    ))


if __name__ == "__main__":
    main()
