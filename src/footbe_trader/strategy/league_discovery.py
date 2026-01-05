"""League discovery and database operations.

Provides utilities for discovering and storing API-Football leagues,
and managing the leagues table.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.football.client import FootballApiClient
from footbe_trader.strategy.normalization import get_league_normalizer

logger = get_logger(__name__)


@dataclass
class LeagueInfo:
    """Information about a soccer league."""
    
    league_id: int
    league_name: str
    country: str
    type: str  # "League" or "Cup"
    logo_url: str | None = None
    seasons_available: list[int] = field(default_factory=list)
    league_key: str | None = None  # Canonical key for cross-platform matching
    is_active: bool = True
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "league_id": self.league_id,
            "league_name": self.league_name,
            "country": self.country,
            "type": self.type,
            "logo_url": self.logo_url,
            "seasons_available": self.seasons_available,
            "league_key": self.league_key,
            "is_active": self.is_active,
        }


@dataclass
class LeagueDiscoveryResult:
    """Result of league discovery operation."""
    
    total_found: int = 0
    new_leagues: int = 0
    updated_leagues: int = 0
    leagues: list[LeagueInfo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class LeagueDiscovery:
    """Discovers and syncs leagues from API-Football."""
    
    def __init__(self, client: FootballApiClient):
        """Initialize with API client.
        
        Args:
            client: Configured FootballApiClient.
        """
        self.client = client
        self.league_normalizer = get_league_normalizer()
    
    async def list_all_leagues(
        self,
        country_filter: str | None = None,
        name_contains: str | None = None,
        current_season_only: bool = False,
    ) -> list[LeagueInfo]:
        """Discover all available leagues from API-Football.
        
        Args:
            country_filter: Filter by country name (case-insensitive).
            name_contains: Filter by league name containing string.
            current_season_only: Only include leagues with current season.
            
        Returns:
            List of discovered leagues.
        """
        # Build request parameters
        params: dict[str, Any] = {}
        if country_filter:
            params["country"] = country_filter
        if current_season_only:
            params["current"] = "true"
        
        # Fetch leagues
        data = await self.client._request(
            "/leagues",
            params=params if params else None,
            cache_key=f"leagues_{'all' if not country_filter else country_filter}",
        )
        
        leagues = []
        for item in data.get("response", []):
            league_data = item.get("league", {})
            country_data = item.get("country", {})
            seasons_data = item.get("seasons", [])
            
            league_name = league_data.get("name", "")
            
            # Apply name filter
            if name_contains and name_contains.lower() not in league_name.lower():
                continue
            
            # Extract seasons
            seasons = [s.get("year") for s in seasons_data if s.get("year")]
            
            # Get canonical league key
            league_id = league_data.get("id", 0)
            league_key = self.league_normalizer.get_key_for_league_id(league_id)
            if not league_key:
                # Fall back to normalizing the name
                league_key = self.league_normalizer.normalize(league_name).canonical
            
            leagues.append(LeagueInfo(
                league_id=league_id,
                league_name=league_name,
                country=country_data.get("name", ""),
                type=league_data.get("type", "League"),
                logo_url=league_data.get("logo"),
                seasons_available=sorted(seasons),
                league_key=league_key,
                is_active=any(s.get("current", False) for s in seasons_data),
                raw_data=item,
            ))
        
        logger.info(
            "leagues_discovered",
            total=len(leagues),
            country_filter=country_filter,
            name_filter=name_contains,
        )
        return leagues
    
    async def list_seasons_for_league(self, league_id: int) -> list[int]:
        """Get available seasons for a specific league.
        
        Args:
            league_id: API-Football league ID.
            
        Returns:
            List of available season years.
        """
        league_info, seasons = await self.client.get_league_info(league_id)
        return sorted([s.year for s in seasons])


class LeagueRepository:
    """Database operations for leagues."""
    
    def __init__(self, db_connection: Any):
        """Initialize with database connection.
        
        Args:
            db_connection: SQLite database connection.
        """
        self.conn = db_connection
    
    def upsert_league(self, league: LeagueInfo) -> int:
        """Insert or update a league.
        
        Args:
            league: League information to store.
            
        Returns:
            Database row ID.
        """
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute(
            "SELECT id FROM leagues WHERE league_id = ?",
            (league.league_id,)
        )
        existing = cursor.fetchone()
        
        now = datetime.now(UTC).isoformat()
        seasons_json = json.dumps(league.seasons_available)
        raw_json = json.dumps(league.raw_data, default=str)
        
        if existing:
            # Update
            cursor.execute("""
                UPDATE leagues SET
                    league_name = ?,
                    country = ?,
                    type = ?,
                    logo_url = ?,
                    seasons_available = ?,
                    league_key = ?,
                    is_active = ?,
                    last_synced_at = ?,
                    raw_json = ?,
                    updated_at = ?
                WHERE league_id = ?
            """, (
                league.league_name,
                league.country,
                league.type,
                league.logo_url,
                seasons_json,
                league.league_key,
                1 if league.is_active else 0,
                now,
                raw_json,
                now,
                league.league_id,
            ))
            row_id = existing[0]
        else:
            # Insert
            cursor.execute("""
                INSERT INTO leagues (
                    league_id, league_name, country, type, logo_url,
                    seasons_available, league_key, is_active, last_synced_at,
                    raw_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                league.league_id,
                league.league_name,
                league.country,
                league.type,
                league.logo_url,
                seasons_json,
                league.league_key,
                1 if league.is_active else 0,
                now,
                raw_json,
                now,
                now,
            ))
            row_id = cursor.lastrowid
        
        self.conn.commit()
        return row_id
    
    def get_league(self, league_id: int) -> LeagueInfo | None:
        """Get a league by API-Football ID.
        
        Args:
            league_id: API-Football league ID.
            
        Returns:
            LeagueInfo if found, None otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT league_id, league_name, country, type, logo_url,
                   seasons_available, league_key, is_active, raw_json
            FROM leagues WHERE league_id = ?
        """, (league_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return LeagueInfo(
            league_id=row[0],
            league_name=row[1],
            country=row[2],
            type=row[3],
            logo_url=row[4],
            seasons_available=json.loads(row[5]) if row[5] else [],
            league_key=row[6],
            is_active=bool(row[7]),
            raw_data=json.loads(row[8]) if row[8] else {},
        )
    
    def get_all_leagues(
        self,
        active_only: bool = False,
        country_filter: str | None = None,
    ) -> list[LeagueInfo]:
        """Get all leagues from database.
        
        Args:
            active_only: Only return active leagues.
            country_filter: Filter by country.
            
        Returns:
            List of leagues.
        """
        cursor = self.conn.cursor()
        
        query = """
            SELECT league_id, league_name, country, type, logo_url,
                   seasons_available, league_key, is_active, raw_json
            FROM leagues WHERE 1=1
        """
        params: list[Any] = []
        
        if active_only:
            query += " AND is_active = 1"
        
        if country_filter:
            query += " AND country = ?"
            params.append(country_filter)
        
        query += " ORDER BY country, league_name"
        
        cursor.execute(query, params)
        
        leagues = []
        for row in cursor.fetchall():
            leagues.append(LeagueInfo(
                league_id=row[0],
                league_name=row[1],
                country=row[2],
                type=row[3],
                logo_url=row[4],
                seasons_available=json.loads(row[5]) if row[5] else [],
                league_key=row[6],
                is_active=bool(row[7]),
                raw_data=json.loads(row[8]) if row[8] else {},
            ))
        
        return leagues
    
    def get_leagues_by_key(self, league_key: str) -> list[LeagueInfo]:
        """Get leagues with a specific canonical key.
        
        Args:
            league_key: Canonical league key.
            
        Returns:
            List of matching leagues.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT league_id, league_name, country, type, logo_url,
                   seasons_available, league_key, is_active, raw_json
            FROM leagues WHERE league_key = ?
        """, (league_key,))
        
        leagues = []
        for row in cursor.fetchall():
            leagues.append(LeagueInfo(
                league_id=row[0],
                league_name=row[1],
                country=row[2],
                type=row[3],
                logo_url=row[4],
                seasons_available=json.loads(row[5]) if row[5] else [],
                league_key=row[6],
                is_active=bool(row[7]),
                raw_data=json.loads(row[8]) if row[8] else {},
            ))
        
        return leagues
    
    def update_league_key(self, league_id: int, league_key: str) -> None:
        """Update the canonical key for a league.
        
        Args:
            league_id: API-Football league ID.
            league_key: New canonical key.
        """
        cursor = self.conn.cursor()
        now = datetime.now(UTC).isoformat()
        cursor.execute("""
            UPDATE leagues SET league_key = ?, updated_at = ?
            WHERE league_id = ?
        """, (league_key, now, league_id))
        self.conn.commit()


async def sync_leagues(
    client: FootballApiClient,
    db_connection: Any,
    country_filter: str | None = None,
    name_contains: str | None = None,
) -> LeagueDiscoveryResult:
    """Sync leagues from API-Football to database.
    
    Args:
        client: API-Football client.
        db_connection: Database connection.
        country_filter: Optional country filter.
        name_contains: Optional name filter.
        
    Returns:
        Discovery result with statistics.
    """
    discovery = LeagueDiscovery(client)
    repo = LeagueRepository(db_connection)
    
    result = LeagueDiscoveryResult()
    
    try:
        leagues = await discovery.list_all_leagues(
            country_filter=country_filter,
            name_contains=name_contains,
        )
        
        result.total_found = len(leagues)
        result.leagues = leagues
        
        for league in leagues:
            existing = repo.get_league(league.league_id)
            repo.upsert_league(league)
            
            if existing:
                result.updated_leagues += 1
            else:
                result.new_leagues += 1
        
        logger.info(
            "leagues_synced",
            total=result.total_found,
            new=result.new_leagues,
            updated=result.updated_leagues,
        )
        
    except Exception as e:
        error_msg = f"League sync failed: {e}"
        result.errors.append(error_msg)
        logger.error("league_sync_failed", error=str(e))
    
    return result
