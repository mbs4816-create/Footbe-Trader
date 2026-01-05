#!/usr/bin/env python3
"""Map fixtures to Kalshi markets.

Runs the mapping engine to link API-Football fixtures with Kalshi
market tickers, using normalization and configurable scoring.

Usage:
    python scripts/map_fixtures_to_markets.py [options]

Examples:
    # Map upcoming fixtures (next 7 days)
    python scripts/map_fixtures_to_markets.py --days-ahead 7

    # Map fixtures for specific league
    python scripts/map_fixtures_to_markets.py --league-id 39 --days-ahead 14

    # Map with lower confidence threshold
    python scripts/map_fixtures_to_markets.py --min-confidence 0.5

    # Show review queue
    python scripts/map_fixtures_to_markets.py --show-reviews

    # Export mappings as JSON
    python scripts/map_fixtures_to_markets.py --export mappings.json
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger, setup_logging
from footbe_trader.storage.database import Database
from footbe_trader.strategy.kalshi_discovery import KalshiMarketRepository
from footbe_trader.strategy.mapping import (
    FixtureMarketMapper,
    MappingRepository,
    MappingConfig,
    ManualOverrides,
)

logger = get_logger(__name__)


def get_upcoming_fixtures(
    conn: sqlite3.Connection,
    days_ahead: int,
    league_id: int | None = None,
) -> list[dict]:
    """Get upcoming fixtures from database."""
    cursor = conn.cursor()
    
    now = datetime.now(UTC)
    end_date = now + timedelta(days=days_ahead)
    
    query = """
        SELECT f.fixture_id, f.league_id, f.season, f.round,
               f.home_team_id, f.away_team_id, f.kickoff_utc, f.status,
               h.name as home_team_name, a.name as away_team_name
        FROM fixtures_v2 f
        JOIN teams h ON f.home_team_id = h.team_id
        JOIN teams a ON f.away_team_id = a.team_id
        WHERE f.kickoff_utc >= ? AND f.kickoff_utc <= ?
          AND f.status NOT IN ('FT', 'AET', 'PEN')
    """
    params = [now.isoformat(), end_date.isoformat()]
    
    if league_id:
        query += " AND f.league_id = ?"
        params.append(league_id)
    
    query += " ORDER BY f.kickoff_utc"
    
    cursor.execute(query, params)
    
    fixtures = []
    for row in cursor.fetchall():
        fixtures.append({
            "fixture_id": row[0],
            "league_id": row[1],
            "season": row[2],
            "round": row[3],
            "home_team_id": row[4],
            "away_team_id": row[5],
            "kickoff_utc": row[6],
            "status": row[7],
            "home_team_name": row[8],
            "away_team_name": row[9],
        })
    
    return fixtures


def show_reviews(db_path: str) -> None:
    """Show pending reviews."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT fixture_id, fixture_info, candidate_count, top_candidates,
               review_status, created_at
        FROM mapping_reviews
        WHERE review_status = 'PENDING'
        ORDER BY created_at DESC
    """)
    
    rows = cursor.fetchall()
    
    if not rows:
        print("No pending reviews.")
        return
    
    print(f"\n{'='*80}")
    print(f"Pending Mapping Reviews ({len(rows)} items)")
    print(f"{'='*80}\n")
    
    for row in rows:
        fixture_id = row[0]
        info = json.loads(row[1]) if row[1] else {}
        candidate_count = row[2]
        candidates = json.loads(row[3]) if row[3] else []
        created = row[5]
        
        print(f"Fixture {fixture_id}: {info.get('home_team', '?')} vs {info.get('away_team', '?')}")
        print(f"  League: {info.get('league_key', 'Unknown')}")
        print(f"  Kickoff: {info.get('kickoff_utc', 'TBD')}")
        print(f"  Candidates: {candidate_count}")
        
        for i, c in enumerate(candidates[:3]):
            score = c.get("total_score", 0)
            ticker = c.get("event_ticker", "?")
            print(f"    {i+1}. {ticker} (score: {score:.2f})")
        
        print()
    
    conn.close()


def export_mappings(db_path: str, output_path: str) -> None:
    """Export mappings to JSON file."""
    conn = sqlite3.connect(db_path)
    repo = MappingRepository(conn)
    
    mappings = repo.get_mappings_for_league("", min_confidence=0.0)
    
    export_data = {
        "exported_at": datetime.now(UTC).isoformat(),
        "total_mappings": len(mappings),
        "mappings": [m.to_dict() for m in mappings],
    }
    
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"Exported {len(mappings)} mappings to {output_path}")
    conn.close()


def run_mapping(
    db_path: str,
    days_ahead: int,
    league_id: int | None,
    min_confidence: float,
) -> None:
    """Run fixture-to-market mapping."""
    conn = sqlite3.connect(db_path)
    
    # Get upcoming fixtures
    fixtures = get_upcoming_fixtures(conn, days_ahead, league_id)
    
    if not fixtures:
        print("No upcoming fixtures found. Run ingest_fixtures.py first.")
        conn.close()
        return
    
    print(f"\n{'='*70}")
    print(f"Mapping {len(fixtures)} upcoming fixtures")
    print(f"{'='*70}\n")
    
    # Initialize mapper
    kalshi_repo = KalshiMarketRepository(conn)
    config = MappingConfig()
    overrides = ManualOverrides()
    mapper = FixtureMarketMapper(kalshi_repo, config, overrides)
    mapping_repo = MappingRepository(conn)
    
    # Create fixture objects for mapping
    from footbe_trader.football.interfaces import FixtureData, FixtureStatus
    
    results = {
        "mapped": 0,
        "needs_review": 0,
        "no_candidates": 0,
        "total": len(fixtures),
    }
    
    for f in fixtures:
        # Convert to FixtureData
        kickoff = None
        if f["kickoff_utc"]:
            try:
                kickoff = datetime.fromisoformat(f["kickoff_utc"])
            except ValueError:
                pass
        
        fixture_data = FixtureData(
            fixture_id=f["fixture_id"],
            league_id=f["league_id"],
            season=f["season"],
            round=f["round"],
            home_team_id=f["home_team_id"],
            away_team_id=f["away_team_id"],
            home_team_name=f["home_team_name"],
            away_team_name=f["away_team_name"],
            kickoff_utc=kickoff,
            status=FixtureStatus.from_short(f["status"]),
        )
        
        # Map fixture
        result = mapper.map_fixture(
            fixture_data,
            f["home_team_name"],
            f["away_team_name"],
        )
        
        # Process result
        if result.success and result.mapping:
            if result.mapping.confidence_score >= min_confidence:
                mapping_repo.save_mapping(result.mapping)
                results["mapped"] += 1
                
                print(f"✓ {f['home_team_name']} vs {f['away_team_name']}")
                print(f"  → {result.mapping.event_ticker} (confidence: {result.mapping.confidence_score:.2f})")
                print(f"  → H: {result.mapping.ticker_home_win}, D: {result.mapping.ticker_draw}, A: {result.mapping.ticker_away_win}")
                print()
            else:
                results["needs_review"] += 1
                mapping_repo.save_review(
                    result.fixture_id,
                    result.fixture_info,
                    result.candidates,
                )
        elif result.candidates:
            results["needs_review"] += 1
            mapping_repo.save_review(
                result.fixture_id,
                result.fixture_info,
                result.candidates,
            )
            
            print(f"? {f['home_team_name']} vs {f['away_team_name']} - needs review")
            print(f"  Best candidate: {result.candidates[0].event_ticker} (score: {result.candidates[0].total_score:.2f})")
            print()
        else:
            results["no_candidates"] += 1
            
            print(f"✗ {f['home_team_name']} vs {f['away_team_name']} - no candidates found")
            print()
    
    # Summary
    print(f"\n{'='*70}")
    print("Mapping Summary")
    print(f"{'='*70}")
    print(f"Total fixtures:       {results['total']}")
    print(f"Successfully mapped:  {results['mapped']}")
    print(f"Needs review:         {results['needs_review']}")
    print(f"No candidates:        {results['no_candidates']}")
    
    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Map fixtures to Kalshi markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--db",
        default="data/footbe.db",
        help="Path to SQLite database (default: data/footbe.db)",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=7,
        help="Days ahead to map fixtures for (default: 7)",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        help="Filter by API-Football league ID",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.75,
        help="Minimum confidence to auto-accept mapping (default: 0.75)",
    )
    parser.add_argument(
        "--show-reviews",
        action="store_true",
        help="Show pending review queue",
    )
    parser.add_argument(
        "--export",
        help="Export mappings to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging (basic Python logging since we don't have a config)
    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.show_reviews:
        show_reviews(args.db)
    elif args.export:
        export_mappings(args.db, args.export)
    else:
        run_mapping(
            db_path=args.db,
            days_ahead=args.days_ahead,
            league_id=args.league_id,
            min_confidence=args.min_confidence,
        )


if __name__ == "__main__":
    main()
