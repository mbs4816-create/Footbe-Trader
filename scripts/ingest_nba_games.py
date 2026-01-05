#!/usr/bin/env python3
"""Ingest NBA games from API-NBA (api-sports.io).

Pulls NBA games and teams from the API and stores them in the database.

Usage:
    python scripts/ingest_nba_games.py [options]

Examples:
    # Ingest upcoming games (next 7 days)
    python scripts/ingest_nba_games.py --days-ahead 7

    # Ingest games for a specific date
    python scripts/ingest_nba_games.py --date 2026-01-10

    # Ingest all teams
    python scripts/ingest_nba_games.py --teams-only

    # Ingest games for a season
    python scripts/ingest_nba_games.py --season 2025
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger, setup_logging
from footbe_trader.nba.client import NBAApiClient
from footbe_trader.nba.interfaces import NBAGame, NBATeam
from footbe_trader.storage.database import Database

logger = get_logger(__name__)


def upsert_nba_team(conn: sqlite3.Connection, team: NBATeam) -> None:
    """Insert or update an NBA team."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO nba_teams (
            team_id, name, nickname, code, city, conference, division, 
            logo_url, raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(team_id) DO UPDATE SET
            name = excluded.name,
            nickname = excluded.nickname,
            code = excluded.code,
            city = excluded.city,
            conference = excluded.conference,
            division = excluded.division,
            logo_url = excluded.logo_url,
            raw_json = excluded.raw_json,
            updated_at = datetime('now')
        """,
        (
            team.team_id,
            team.name,
            team.nickname,
            team.code,
            team.city,
            team.conference,
            team.division,
            team.logo,
            json.dumps(team.raw_data, default=str),
        ),
    )
    conn.commit()


def upsert_nba_game(conn: sqlite3.Connection, game: NBAGame) -> None:
    """Insert or update an NBA game."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO nba_games (
            game_id, season, league, stage, date_utc, timestamp,
            status, home_team_id, away_team_id, home_score, away_score,
            arena, city, raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(game_id) DO UPDATE SET
            season = excluded.season,
            league = excluded.league,
            stage = excluded.stage,
            date_utc = excluded.date_utc,
            timestamp = excluded.timestamp,
            status = excluded.status,
            home_team_id = excluded.home_team_id,
            away_team_id = excluded.away_team_id,
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            arena = excluded.arena,
            city = excluded.city,
            raw_json = excluded.raw_json,
            updated_at = datetime('now')
        """,
        (
            game.game_id,
            game.season,
            game.league,
            game.stage,
            game.date.isoformat(),
            game.timestamp,
            game.status.value,
            game.home_team.team_id,
            game.away_team.team_id,
            game.home_score,
            game.away_score,
            game.arena,
            game.city,
            json.dumps(game.raw_data, default=str),
        ),
    )
    conn.commit()


def get_team_code(team_id: int, conn: sqlite3.Connection) -> str | None:
    """Get team code by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT code FROM nba_teams WHERE team_id = ?", (team_id,))
    row = cursor.fetchone()
    return row[0] if row else None


async def ingest_teams(
    config_path: str,
    db_path: str,
) -> None:
    """Ingest all NBA teams."""
    config = load_config(config_path)
    
    db = Database(db_path)
    db.connect()
    db.migrate()
    
    print(f"\n{'='*60}")
    print("Ingesting NBA Teams")
    print(f"{'='*60}\n")
    
    async with NBAApiClient(config.football_api) as client:
        teams = await client.get_teams()
        
        print(f"Found {len(teams)} teams")
        
        for team in teams:
            upsert_nba_team(db.connection, team)
            print(f"  {team.code}: {team.name} ({team.city})")
        
        print(f"\nInserted/updated {len(teams)} teams")


async def ingest_games_by_date(
    config_path: str,
    db_path: str,
    game_date: date,
    season: int | None = None,
) -> None:
    """Ingest games for a specific date."""
    config = load_config(config_path)
    
    db = Database(db_path)
    db.connect()
    db.migrate()
    
    print(f"\n{'='*60}")
    print(f"Ingesting NBA Games for {game_date.isoformat()}")
    print(f"{'='*60}\n")
    
    async with NBAApiClient(config.football_api) as client:
        games = await client.get_games_by_date(game_date, season)
        
        print(f"Found {len(games)} games")
        
        for game in games:
            # Ensure teams exist first
            if game.home_team.team_id:
                team = NBATeam(
                    team_id=game.home_team.team_id,
                    name=game.home_team.name,
                    nickname=game.home_team.nickname,
                    code=game.home_team.code,
                    city="",
                )
                upsert_nba_team(db.connection, team)
            
            if game.away_team.team_id:
                team = NBATeam(
                    team_id=game.away_team.team_id,
                    name=game.away_team.name,
                    nickname=game.away_team.nickname,
                    code=game.away_team.code,
                    city="",
                )
                upsert_nba_team(db.connection, team)
            
            upsert_nba_game(db.connection, game)
            status_str = game.status.name
            print(f"  [{status_str}] {game.away_team.nickname} @ {game.home_team.nickname}")
        
        print(f"\nInserted/updated {len(games)} games")


async def ingest_upcoming_games(
    config_path: str,
    db_path: str,
    days_ahead: int,
    season: int | None = None,
) -> None:
    """Ingest upcoming games for the next N days."""
    config = load_config(config_path)
    
    db = Database(db_path)
    db.connect()
    db.migrate()
    
    print(f"\n{'='*60}")
    print(f"Ingesting NBA Games for next {days_ahead} days")
    print(f"{'='*60}\n")
    
    today = date.today()
    total_games = 0
    
    async with NBAApiClient(config.football_api) as client:
        for i in range(days_ahead):
            game_date = today + timedelta(days=i)
            games = await client.get_games_by_date(game_date, season)
            
            if games:
                print(f"\n{game_date.isoformat()}: {len(games)} games")
                
                for game in games:
                    # Ensure teams exist
                    if game.home_team.team_id:
                        team = NBATeam(
                            team_id=game.home_team.team_id,
                            name=game.home_team.name,
                            nickname=game.home_team.nickname,
                            code=game.home_team.code,
                            city="",
                        )
                        upsert_nba_team(db.connection, team)
                    
                    if game.away_team.team_id:
                        team = NBATeam(
                            team_id=game.away_team.team_id,
                            name=game.away_team.name,
                            nickname=game.away_team.nickname,
                            code=game.away_team.code,
                            city="",
                        )
                        upsert_nba_team(db.connection, team)
                    
                    upsert_nba_game(db.connection, game)
                    status_str = game.status.name
                    print(f"    [{status_str}] {game.away_team.nickname} @ {game.home_team.nickname}")
                
                total_games += len(games)
    
    print(f"\n{'='*60}")
    print(f"Total: {total_games} games ingested")
    print(f"{'='*60}")


def list_games(db_path: str, days_ahead: int = 7) -> None:
    """List upcoming NBA games from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    today = date.today()
    end_date = today + timedelta(days=days_ahead)
    
    cursor.execute(
        """
        SELECT 
            g.game_id, g.date_utc, g.status,
            h.code as home_code, h.nickname as home_name,
            a.code as away_code, a.nickname as away_name,
            g.home_score, g.away_score
        FROM nba_games g
        LEFT JOIN nba_teams h ON g.home_team_id = h.team_id
        LEFT JOIN nba_teams a ON g.away_team_id = a.team_id
        WHERE date(g.date_utc) >= date(?)
          AND date(g.date_utc) <= date(?)
        ORDER BY g.date_utc
        """,
        (today.isoformat(), end_date.isoformat()),
    )
    
    rows = cursor.fetchall()
    
    if not rows:
        print("No upcoming games found. Run ingestion first.")
        return
    
    print(f"\n{'='*80}")
    print(f"{'Game ID':<12} {'Date':<12} {'Status':<12} {'Away':<15} {'Home':<15} {'Score'}")
    print(f"{'='*80}")
    
    for row in rows:
        game_id, date_utc, status, home_code, home_name, away_code, away_name, home_score, away_score = row
        game_date = date_utc[:10] if date_utc else "?"
        status_map = {1: "NS", 2: "LIVE", 3: "FIN", 4: "PPD", 5: "DLY", 6: "CAN"}
        status_str = status_map.get(status, str(status))
        score = ""
        if home_score is not None and away_score is not None:
            score = f"{away_score}-{home_score}"
        
        print(f"{game_id:<12} {game_date:<12} {status_str:<12} {away_code or '?':<15} {home_code or '?':<15} {score}")
    
    print(f"\nTotal: {len(rows)} games")
    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest NBA games from API-NBA",
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
        default=7,
        help="Days ahead to fetch games for (default: 7)",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Specific date to fetch games for (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="NBA season year (e.g., 2024 for 2024-25 season)",
    )
    parser.add_argument(
        "--teams-only",
        action="store_true",
        help="Only ingest teams, not games",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List games from database instead of ingesting",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_games(args.db, args.days_ahead)
        return
    
    if args.teams_only:
        asyncio.run(ingest_teams(args.config, args.db))
    elif args.date:
        game_date = date.fromisoformat(args.date)
        asyncio.run(ingest_games_by_date(args.config, args.db, game_date, args.season))
    else:
        asyncio.run(ingest_upcoming_games(args.config, args.db, args.days_ahead, args.season))


if __name__ == "__main__":
    main()
