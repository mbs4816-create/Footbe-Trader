#!/usr/bin/env python3
"""Map NBA games to Kalshi basketball markets.

Creates mappings between NBA games from API-NBA and Kalshi KXNBAGAME markets
based on date and team matching.

Usage:
    python scripts/map_nba_games.py [options]

Examples:
    # Map all unmapped games
    python scripts/map_nba_games.py

    # Show current mappings
    python scripts/map_nba_games.py --list

    # Map with verbose output
    python scripts/map_nba_games.py -v
"""

import argparse
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.logging import get_logger

logger = get_logger(__name__)


def get_unmapped_nba_games(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Get NBA games that haven't been mapped to Kalshi markets yet."""
    cursor = conn.cursor()
    
    # Get games that are upcoming (status = 1 = NOT_STARTED) and not yet mapped
    cursor.execute(
        """
        SELECT 
            g.game_id, g.date_utc, g.status,
            g.home_team_id, g.away_team_id,
            h.code as home_code, h.name as home_name, h.nickname as home_nickname,
            a.code as away_code, a.name as away_name, a.nickname as away_nickname
        FROM nba_games g
        LEFT JOIN nba_teams h ON g.home_team_id = h.team_id
        LEFT JOIN nba_teams a ON g.away_team_id = a.team_id
        WHERE g.status = 1  -- NOT_STARTED
          AND NOT EXISTS (
              SELECT 1 FROM nba_game_market_map m
              WHERE m.game_id = g.game_id
          )
        ORDER BY g.date_utc
        """
    )
    
    games = []
    for row in cursor.fetchall():
        games.append({
            "game_id": row[0],
            "date_utc": row[1],
            "status": row[2],
            "home_team_id": row[3],
            "away_team_id": row[4],
            "home_code": row[5],
            "home_name": row[6],
            "home_nickname": row[7],
            "away_code": row[8],
            "away_name": row[9],
            "away_nickname": row[10],
        })
    
    return games


def get_nba_kalshi_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Get NBA events from Kalshi with their markets."""
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT 
            e.event_ticker, e.title, e.strike_date,
            e.parsed_canonical_away, e.parsed_canonical_home
        FROM kalshi_events e
        WHERE e.is_basketball = 1
        """
    )
    
    events = []
    for row in cursor.fetchall():
        event_ticker = row[0]
        
        # Get markets for this event
        cursor.execute(
            """
            SELECT ticker, market_type, parsed_team, status
            FROM kalshi_markets
            WHERE event_ticker = ?
            """,
            (event_ticker,),
        )
        
        markets = []
        for mrow in cursor.fetchall():
            markets.append({
                "ticker": mrow[0],
                "market_type": mrow[1],
                "parsed_team": mrow[2],
                "status": mrow[3],
            })
        
        events.append({
            "event_ticker": row[0],
            "title": row[1],
            "strike_date": row[2],
            "away_code": row[3],
            "home_code": row[4],
            "markets": markets,
        })
    
    return events


def calculate_confidence(
    game: dict[str, Any],
    event: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Calculate mapping confidence score.
    
    Returns:
        Tuple of (total_score, component_scores).
    """
    components = {}
    
    # Date match (most important)
    game_date = game["date_utc"][:10] if game["date_utc"] else None
    event_date = event["strike_date"][:10] if event["strike_date"] else None
    
    if game_date and event_date:
        if game_date == event_date:
            components["date_match"] = 0.5
        else:
            # Check if within 1 day (timezone issues)
            try:
                gd = datetime.fromisoformat(game_date)
                ed = datetime.fromisoformat(event_date)
                if abs((gd - ed).days) <= 1:
                    components["date_match"] = 0.3
                else:
                    components["date_match"] = 0.0
            except ValueError:
                components["date_match"] = 0.0
    else:
        components["date_match"] = 0.0
    
    # Team code match
    game_home = game.get("home_code", "").upper()
    game_away = game.get("away_code", "").upper()
    event_home = (event.get("home_code") or "").upper()
    event_away = (event.get("away_code") or "").upper()
    
    if game_home and event_home and game_home == event_home:
        components["home_code_match"] = 0.25
    else:
        components["home_code_match"] = 0.0
    
    if game_away and event_away and game_away == event_away:
        components["away_code_match"] = 0.25
    else:
        components["away_code_match"] = 0.0
    
    total = sum(components.values())
    return total, components


def find_best_match(
    game: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float, dict[str, float]]:
    """Find the best matching Kalshi event for an NBA game.
    
    Returns:
        Tuple of (best_event, confidence, components).
    """
    best_event = None
    best_score = 0.0
    best_components: dict[str, float] = {}
    
    for event in events:
        score, components = calculate_confidence(game, event)
        
        if score > best_score:
            best_score = score
            best_event = event
            best_components = components
    
    return best_event, best_score, best_components


def insert_mapping(
    conn: sqlite3.Connection,
    game_id: int,
    event: dict[str, Any],
    confidence: float,
    components: dict[str, float],
) -> int:
    """Insert a game-to-market mapping."""
    cursor = conn.cursor()
    
    # Find home and away market tickers
    ticker_home = None
    ticker_away = None
    
    for market in event.get("markets", []):
        if market["market_type"] == "HOME_WIN":
            ticker_home = market["ticker"]
        elif market["market_type"] == "AWAY_WIN":
            ticker_away = market["ticker"]
    
    cursor.execute(
        """
        INSERT INTO nba_game_market_map (
            game_id, mapping_version,
            ticker_home_win, ticker_away_win,
            event_ticker, confidence_score, confidence_components,
            status, metadata_json, updated_at
        ) VALUES (?, 1, ?, ?, ?, ?, ?, 'AUTO', '{}', datetime('now'))
        """,
        (
            game_id,
            ticker_home,
            ticker_away,
            event["event_ticker"],
            confidence,
            json.dumps(components),
        ),
    )
    conn.commit()
    
    return cursor.lastrowid or 0


def map_nba_games(db_path: str, min_confidence: float = 0.5) -> dict[str, int]:
    """Map NBA games to Kalshi markets."""
    conn = sqlite3.connect(db_path)
    
    print(f"\n{'='*60}")
    print("Mapping NBA Games to Kalshi Markets")
    print(f"{'='*60}\n")
    
    stats = {
        "games_found": 0,
        "events_found": 0,
        "mapped": 0,
        "low_confidence": 0,
        "no_match": 0,
    }
    
    # Get unmapped games and Kalshi events
    games = get_unmapped_nba_games(conn)
    events = get_nba_kalshi_events(conn)
    
    stats["games_found"] = len(games)
    stats["events_found"] = len(events)
    
    print(f"Unmapped NBA games: {len(games)}")
    print(f"Kalshi NBA events:  {len(events)}")
    print()
    
    for game in games:
        game_date = game["date_utc"][:10] if game["date_utc"] else "?"
        game_desc = f"{game['away_code']} @ {game['home_code']} ({game_date})"
        
        best_event, confidence, components = find_best_match(game, events)
        
        if best_event is None:
            print(f"  ❌ {game_desc}: No matching Kalshi event")
            stats["no_match"] += 1
            continue
        
        if confidence < min_confidence:
            print(f"  ⚠️  {game_desc}: Low confidence {confidence:.2f} -> {best_event['event_ticker']}")
            stats["low_confidence"] += 1
            # Still insert but mark as low confidence
            insert_mapping(conn, game["game_id"], best_event, confidence, components)
            stats["mapped"] += 1
            continue
        
        # Good match
        print(f"  ✅ {game_desc}: {confidence:.2f} -> {best_event['event_ticker']}")
        insert_mapping(conn, game["game_id"], best_event, confidence, components)
        stats["mapped"] += 1
        
        # Remove this event from candidates (one-to-one mapping)
        events = [e for e in events if e["event_ticker"] != best_event["event_ticker"]]
    
    print(f"\n{'='*60}")
    print("Mapping Results")
    print(f"{'='*60}")
    print(f"Games found:        {stats['games_found']}")
    print(f"Kalshi events:      {stats['events_found']}")
    print(f"Mapped:             {stats['mapped']}")
    print(f"Low confidence:     {stats['low_confidence']}")
    print(f"No match:           {stats['no_match']}")
    
    conn.close()
    return stats


def list_mappings(db_path: str) -> None:
    """List current NBA game-to-market mappings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT 
            m.game_id, m.event_ticker, m.ticker_home_win, m.ticker_away_win,
            m.confidence_score, m.status,
            g.date_utc,
            h.code as home_code, h.nickname as home_name,
            a.code as away_code, a.nickname as away_name
        FROM nba_game_market_map m
        JOIN nba_games g ON m.game_id = g.game_id
        LEFT JOIN nba_teams h ON g.home_team_id = h.team_id
        LEFT JOIN nba_teams a ON g.away_team_id = a.team_id
        ORDER BY g.date_utc
        """
    )
    
    rows = cursor.fetchall()
    
    if not rows:
        print("No mappings found. Run mapping first.")
        return
    
    print(f"\n{'='*100}")
    print(f"{'Date':<12} {'Game':<25} {'Event':<35} {'Conf':<6} {'Status'}")
    print(f"{'='*100}")
    
    for row in rows:
        (game_id, event_ticker, ticker_home, ticker_away, confidence, status,
         date_utc, home_code, home_name, away_code, away_name) = row
        
        game_date = date_utc[:10] if date_utc else "?"
        game_desc = f"{away_code} @ {home_code}"
        conf_str = f"{confidence:.2f}" if confidence else "?"
        
        print(f"{game_date:<12} {game_desc:<25} {event_ticker:<35} {conf_str:<6} {status}")
        
        if ticker_home or ticker_away:
            print(f"             └─ Home: {ticker_home or '?':<30} Away: {ticker_away or '?'}")
    
    print(f"\nTotal: {len(rows)} mappings")
    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Map NBA games to Kalshi markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--db",
        default="data/footbe.db",
        help="Path to SQLite database (default: data/footbe.db)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for automatic mapping (default: 0.5)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List current mappings instead of creating new ones",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_mappings(args.db)
    else:
        map_nba_games(args.db, args.min_confidence)


if __name__ == "__main__":
    main()
