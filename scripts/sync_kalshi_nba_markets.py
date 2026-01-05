#!/usr/bin/env python3
"""Sync Kalshi NBA (basketball) markets to database.

Discovers and classifies basketball markets from Kalshi (KXNBAGAME series),
storing them with parsed team names.

Usage:
    python scripts/sync_kalshi_nba_markets.py [options]

Examples:
    # Sync all NBA markets
    python scripts/sync_kalshi_nba_markets.py

    # Sync with verbose output
    python scripts/sync_kalshi_nba_markets.py -v

    # List synced NBA markets
    python scripts/sync_kalshi_nba_markets.py --list
"""

import argparse
import asyncio
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger, setup_logging
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.kalshi.interfaces import EventData, MarketData
from footbe_trader.storage.database import Database

logger = get_logger(__name__)

# NBA Series ticker prefix
NBA_SERIES_PREFIX = "KXNBAGAME"

# Pattern to parse date and teams from event ticker
# Example: KXNBAGAME-26JAN07MILGSW -> Jan 7, 2026, MIL vs GSW
TICKER_PATTERN = re.compile(
    r'KXNBAGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})'
)

MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}


@dataclass
class ParsedNBAEvent:
    """Parsed NBA event information."""
    game_date: datetime | None
    away_team_code: str | None  # First team in ticker (visiting)
    home_team_code: str | None  # Second team in ticker (home)


def parse_nba_ticker(ticker: str) -> ParsedNBAEvent:
    """Parse NBA event ticker to extract date and teams.
    
    Format: KXNBAGAME-YYMONDDAWYHOM
    Example: KXNBAGAME-26JAN07MILGSW
        - 26 = year 2026
        - JAN = January
        - 07 = day 7
        - MIL = Milwaukee (away team, listed first)
        - GSW = Golden State (home team, listed second)
    """
    match = TICKER_PATTERN.match(ticker)
    if not match:
        return ParsedNBAEvent(None, None, None)
    
    year_short, month_str, day, away_code, home_code = match.groups()
    
    month = MONTH_MAP.get(month_str.upper())
    if not month:
        return ParsedNBAEvent(None, away_code, home_code)
    
    try:
        year = 2000 + int(year_short)
        day_int = int(day)
        game_date = datetime(year, month, day_int, tzinfo=UTC)
        return ParsedNBAEvent(game_date, away_code, home_code)
    except ValueError:
        return ParsedNBAEvent(None, away_code, home_code)


def parse_team_from_market(market_ticker: str, event_ticker: str) -> str | None:
    """Parse which team a market is for.
    
    Market ticker format: KXNBAGAME-26JAN07MILGSW-MIL
    Returns: MIL
    """
    if not market_ticker.startswith(event_ticker):
        return None
    
    suffix = market_ticker[len(event_ticker):]
    if suffix.startswith("-"):
        return suffix[1:]
    return None


def upsert_kalshi_event(
    conn: sqlite3.Connection,
    event: EventData,
    parsed: ParsedNBAEvent,
) -> None:
    """Insert or update a Kalshi event for NBA."""
    cursor = conn.cursor()
    
    # Parse teams from title if available
    # Title format: "Milwaukee at Golden State" or similar
    home_team = None
    away_team = None
    
    title = event.title
    if " at " in title:
        parts = title.split(" at ")
        away_team = parts[0].strip()
        home_team = parts[1].strip()
    elif " vs " in title.lower():
        parts = re.split(r"\s+vs\.?\s+", title, flags=re.IGNORECASE)
        away_team = parts[0].strip()
        home_team = parts[1].strip() if len(parts) > 1 else None
    
    cursor.execute(
        """
        INSERT INTO kalshi_events (
            event_ticker, series_ticker, title, subtitle, category, sub_category,
            strike_date, is_soccer, is_basketball, sport_type, league_key,
            parsed_home_team, parsed_away_team,
            parsed_canonical_home, parsed_canonical_away,
            market_structure, raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(event_ticker) DO UPDATE SET
            series_ticker = excluded.series_ticker,
            title = excluded.title,
            subtitle = excluded.subtitle,
            category = excluded.category,
            sub_category = excluded.sub_category,
            strike_date = excluded.strike_date,
            is_basketball = excluded.is_basketball,
            sport_type = excluded.sport_type,
            league_key = excluded.league_key,
            parsed_home_team = excluded.parsed_home_team,
            parsed_away_team = excluded.parsed_away_team,
            parsed_canonical_home = excluded.parsed_canonical_home,
            parsed_canonical_away = excluded.parsed_canonical_away,
            market_structure = excluded.market_structure,
            raw_json = excluded.raw_json,
            updated_at = datetime('now')
        """,
        (
            event.event_ticker,
            event.series_ticker,
            event.title,
            event.subtitle,
            event.category,
            event.sub_category,
            parsed.game_date.isoformat() if parsed.game_date else None,
            0,  # is_soccer = false
            1,  # is_basketball = true
            "basketball",  # sport_type
            "NBA",  # league_key
            home_team,  # parsed_home_team
            away_team,  # parsed_away_team
            parsed.home_team_code,  # parsed_canonical_home (team code)
            parsed.away_team_code,  # parsed_canonical_away (team code)
            "MONEYLINE",  # market_structure (2-way)
            json.dumps(event.raw_data, default=str),
        ),
    )
    conn.commit()


def upsert_kalshi_market(
    conn: sqlite3.Connection,
    market: MarketData,
    team_code: str | None,
    is_home: bool | None,
) -> None:
    """Insert or update a Kalshi market for NBA."""
    cursor = conn.cursor()
    
    # Determine market type
    market_type = None
    if team_code:
        market_type = "HOME_WIN" if is_home else "AWAY_WIN"
    
    cursor.execute(
        """
        INSERT INTO kalshi_markets (
            ticker, event_ticker, title, subtitle, status,
            open_time, close_time, expiration_time,
            yes_bid, yes_ask, no_bid, no_ask, last_price, volume,
            is_soccer, is_basketball, sport_type, market_type,
            parsed_team, parsed_canonical_team,
            raw_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(ticker) DO UPDATE SET
            event_ticker = excluded.event_ticker,
            title = excluded.title,
            subtitle = excluded.subtitle,
            status = excluded.status,
            open_time = excluded.open_time,
            close_time = excluded.close_time,
            expiration_time = excluded.expiration_time,
            yes_bid = excluded.yes_bid,
            yes_ask = excluded.yes_ask,
            no_bid = excluded.no_bid,
            no_ask = excluded.no_ask,
            last_price = excluded.last_price,
            volume = excluded.volume,
            is_basketball = excluded.is_basketball,
            sport_type = excluded.sport_type,
            market_type = excluded.market_type,
            parsed_team = excluded.parsed_team,
            parsed_canonical_team = excluded.parsed_canonical_team,
            raw_json = excluded.raw_json,
            updated_at = datetime('now')
        """,
        (
            market.ticker,
            market.event_ticker,
            market.title,
            market.subtitle,
            market.status,
            market.open_time.isoformat() if market.open_time else None,
            market.close_time.isoformat() if market.close_time else None,
            market.expiration_time.isoformat() if market.expiration_time else None,
            market.yes_bid,
            market.yes_ask,
            market.no_bid,
            market.no_ask,
            market.last_price,
            market.volume,
            0,  # is_soccer = false
            1,  # is_basketball = true
            "basketball",  # sport_type
            market_type,
            team_code,  # parsed_team
            team_code,  # parsed_canonical_team (same for NBA - already a code)
            json.dumps(market.raw_data, default=str),
        ),
    )
    conn.commit()


async def sync_nba_markets(
    config_path: str,
    db_path: str,
    days_ahead: int = 30,
) -> dict[str, int]:
    """Sync Kalshi NBA markets to database."""
    config = load_config(config_path)
    
    db = Database(db_path)
    db.connect()
    db.migrate()
    
    print(f"\n{'='*60}")
    print("Syncing Kalshi NBA Markets")
    print(f"{'='*60}\n")
    
    stats = {
        "events_found": 0,
        "markets_found": 0,
        "events_synced": 0,
        "markets_synced": 0,
    }
    
    async with KalshiClient(config.kalshi) as client:
        # Get NBA events from the KXNBAGAME series
        print(f"Fetching events from {NBA_SERIES_PREFIX} series...")
        
        events = await client.get_events_for_series(NBA_SERIES_PREFIX)
        stats["events_found"] = len(events)
        print(f"Found {len(events)} NBA events")
        
        for event in events:
            # Parse event ticker
            parsed = parse_nba_ticker(event.event_ticker)
            
            # Store event
            upsert_kalshi_event(db.connection, event, parsed)
            stats["events_synced"] += 1
            
            # Get markets for this event
            markets = await client.get_markets_for_event(event.event_ticker)
            stats["markets_found"] += len(markets)
            
            for market in markets:
                # Parse which team this market is for
                team_code = parse_team_from_market(market.ticker, event.event_ticker)
                
                # Determine if home or away
                is_home = None
                if team_code and parsed.home_team_code:
                    is_home = team_code == parsed.home_team_code
                
                upsert_kalshi_market(db.connection, market, team_code, is_home)
                stats["markets_synced"] += 1
            
            print(f"  {event.event_ticker}: {event.title} ({len(markets)} markets)")
    
    print(f"\n{'='*60}")
    print("Sync Results")
    print(f"{'='*60}")
    print(f"Events found:   {stats['events_found']}")
    print(f"Events synced:  {stats['events_synced']}")
    print(f"Markets found:  {stats['markets_found']}")
    print(f"Markets synced: {stats['markets_synced']}")
    
    return stats


def list_nba_markets(db_path: str) -> None:
    """List NBA markets in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT 
            e.event_ticker, e.title, e.strike_date,
            e.parsed_canonical_away, e.parsed_canonical_home
        FROM kalshi_events e
        WHERE e.is_basketball = 1
        ORDER BY e.strike_date
        """
    )
    
    events = cursor.fetchall()
    
    if not events:
        print("No NBA markets found in database. Run sync first.")
        return
    
    print(f"\n{'='*100}")
    print(f"{'Event Ticker':<35} {'Date':<12} {'Away':<8} {'Home':<8} {'Title'}")
    print(f"{'='*100}")
    
    for event_ticker, title, strike_date, away_code, home_code in events:
        date_str = strike_date[:10] if strike_date else "?"
        
        print(f"{event_ticker:<35} {date_str:<12} {away_code or '?':<8} {home_code or '?':<8} {title[:40]}")
        
        # Get markets for this event
        cursor.execute(
            """
            SELECT ticker, market_type, title, yes_bid, yes_ask, status
            FROM kalshi_markets
            WHERE event_ticker = ?
            ORDER BY ticker
            """,
            (event_ticker,),
        )
        
        markets = cursor.fetchall()
        for ticker, market_type, mtitle, yes_bid, yes_ask, status in markets:
            bid_str = f"{yes_bid:.0f}" if yes_bid else "?"
            ask_str = f"{yes_ask:.0f}" if yes_ask else "?"
            mtype = market_type or "?"
            print(f"    └─ {ticker:<30} [{mtype}] {bid_str}/{ask_str} ({status})")
    
    print(f"\nTotal: {len(events)} NBA events")
    conn.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync Kalshi NBA markets to database",
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
        help="List synced NBA markets instead of syncing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_nba_markets(args.db)
    else:
        asyncio.run(sync_nba_markets(args.config, args.db, args.days_ahead))


if __name__ == "__main__":
    main()
