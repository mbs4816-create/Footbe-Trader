"""Kalshi soccer market discovery and classification.

Discovers and classifies Kalshi markets that are related to soccer/football
matches, parsing team names and market structures.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from footbe_trader.common.logging import get_logger
from footbe_trader.kalshi.client import KalshiClient
from footbe_trader.kalshi.interfaces import EventData, MarketData
from footbe_trader.strategy.normalization import (
    get_league_normalizer,
    get_team_normalizer,
    normalize_team_name,
)

logger = get_logger(__name__)

# Default config path (strategy -> footbe_trader -> src -> project_root)
DEFAULT_MAPPING_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "mapping_config.yaml"

# Pattern to parse date from ticker like "26JAN17" -> Jan 17, 2026
TICKER_DATE_PATTERN = re.compile(r'(\d{2})([A-Z]{3})(\d{2})')
MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}


def parse_date_from_ticker(ticker: str) -> datetime | None:
    """Parse date from Kalshi ticker format.
    
    Examples:
        KXEPLGAME-26JAN17WHUNFO -> Jan 17, 2026
        KXUCLGAME-25DEC28REALBAY -> Dec 28, 2025
    
    Returns:
        datetime in UTC, or None if no date found.
    """
    match = TICKER_DATE_PATTERN.search(ticker)
    if not match:
        return None
    
    year_short, month_str, day = match.groups()
    month = MONTH_MAP.get(month_str.upper())
    if not month:
        return None
    
    try:
        # Assume 20xx century
        year = 2000 + int(year_short)
        day_int = int(day)
        return datetime(year, month, day_int, tzinfo=UTC)
    except ValueError:
        return None


@dataclass
class SoccerMarketClassification:
    """Classification of a Kalshi market as soccer-related."""
    
    is_soccer: bool = False
    confidence: float = 0.0
    league_key: str | None = None  # Canonical league if detected
    
    # Parsed teams
    home_team: str | None = None
    away_team: str | None = None
    canonical_home: str | None = None
    canonical_away: str | None = None
    
    # Market structure
    market_type: str | None = None  # "HOME_WIN", "AWAY_WIN", "DRAW", etc.
    structure_type: str | None = None  # "1X2", "MONEYLINE", "BINARY"
    
    # Which team this specific market references (for individual markets)
    target_team: str | None = None
    canonical_target: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_soccer": self.is_soccer,
            "confidence": self.confidence,
            "league_key": self.league_key,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "canonical_home": self.canonical_home,
            "canonical_away": self.canonical_away,
            "market_type": self.market_type,
            "structure_type": self.structure_type,
            "target_team": self.target_team,
            "canonical_target": self.canonical_target,
        }


@dataclass
class KalshiEventRecord:
    """Record for a Kalshi event in database."""
    
    event_ticker: str
    series_ticker: str | None
    title: str
    subtitle: str | None
    category: str | None
    sub_category: str | None
    strike_date: datetime | None
    is_soccer: bool
    league_key: str | None
    parsed_home_team: str | None
    parsed_away_team: str | None
    parsed_canonical_home: str | None
    parsed_canonical_away: str | None
    market_structure: str | None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class KalshiMarketRecord:
    """Record for a Kalshi market in database."""
    
    ticker: str
    event_ticker: str
    title: str
    subtitle: str | None
    status: str
    open_time: datetime | None
    close_time: datetime | None
    expiration_time: datetime | None
    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    last_price: float | None
    volume: int
    is_soccer: bool
    market_type: str | None
    parsed_team: str | None
    parsed_canonical_team: str | None
    raw_data: dict[str, Any] = field(default_factory=dict)


class SoccerMarketClassifier:
    """Classifies Kalshi markets as soccer-related."""
    
    def __init__(self, config_path: Path | str | None = None):
        """Initialize classifier with configuration.
        
        Args:
            config_path: Path to mapping configuration YAML.
        """
        self.config = self._load_config(config_path or DEFAULT_MAPPING_CONFIG_PATH)
        self.team_normalizer = get_team_normalizer()
        self.league_normalizer = get_league_normalizer()
        
        # Compile patterns
        self._compile_patterns()
    
    def _load_config(self, path: Path | str) -> dict[str, Any]:
        """Load configuration from YAML."""
        path = Path(path)
        if not path.exists():
            logger.warning("mapping_config_not_found", path=str(path))
            return self._default_config()
        
        with open(path) as f:
            return yaml.safe_load(f) or self._default_config()
    
    def _default_config(self) -> dict[str, Any]:
        """Return default configuration."""
        return {
            "soccer_detection": {
                "keywords": ["soccer", "football", "premier league", "la liga", "mls"],
                "categories": ["sports", "soccer", "football"],
                "series_patterns": ["^SOCCER", "^FUTBOL", "-EPL-", "-MLS-"],
            },
            "title_parsing": {
                "vs_patterns": [r"(?P<home>.+?)\s+(?:vs?\.?|versus|@)\s+(?P<away>.+)"],
                "team_win_patterns": [r"will\s+(?P<team>.+?)\s+win"],
            },
        }
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns from config."""
        detection = self.config.get("soccer_detection", {})
        parsing = self.config.get("title_parsing", {})
        
        # Series patterns
        self._series_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in detection.get("series_patterns", [])
        ]
        
        # VS patterns for team extraction
        self._vs_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in parsing.get("vs_patterns", [])
        ]
        
        # Team win patterns
        self._team_win_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in parsing.get("team_win_patterns", [])
        ]
    
    def classify_event(self, event: EventData) -> SoccerMarketClassification:
        """Classify a Kalshi event.
        
        Args:
            event: Event data from Kalshi.
            
        Returns:
            Classification result.
        """
        result = SoccerMarketClassification()
        
        # Check if soccer-related
        is_soccer, confidence = self._detect_soccer(event)
        result.is_soccer = is_soccer
        result.confidence = confidence
        
        if not is_soccer:
            return result
        
        # Try to detect league
        result.league_key = self._detect_league(event)
        
        # Parse teams from title
        home, away = self._parse_teams_from_title(event.title)
        if home and away:
            result.home_team = home
            result.away_team = away
            result.canonical_home = normalize_team_name(home, result.league_key)
            result.canonical_away = normalize_team_name(away, result.league_key)
            result.structure_type = "1X2"  # Likely a match event
        
        return result
    
    def classify_market(
        self,
        market: MarketData,
        event_classification: SoccerMarketClassification | None = None,
    ) -> SoccerMarketClassification:
        """Classify a Kalshi market.
        
        Args:
            market: Market data from Kalshi.
            event_classification: Optional parent event classification.
            
        Returns:
            Classification result.
        """
        result = SoccerMarketClassification()
        
        # Inherit from event if available
        if event_classification:
            result.is_soccer = event_classification.is_soccer
            result.league_key = event_classification.league_key
            result.home_team = event_classification.home_team
            result.away_team = event_classification.away_team
            result.canonical_home = event_classification.canonical_home
            result.canonical_away = event_classification.canonical_away
            result.structure_type = event_classification.structure_type
        else:
            # Classify independently
            is_soccer, confidence = self._detect_soccer_from_market(market)
            result.is_soccer = is_soccer
            result.confidence = confidence
        
        if not result.is_soccer:
            return result
        
        # Determine market type from title
        result.market_type = self._detect_market_type(market.title, market.subtitle)
        
        # Parse target team if applicable
        target = self._parse_target_team(market.title, market.subtitle)
        if target:
            result.target_team = target
            result.canonical_target = normalize_team_name(target, result.league_key)
        
        return result
    
    def _detect_soccer(self, event: EventData) -> tuple[bool, float]:
        """Detect if event is soccer-related.
        
        Returns:
            Tuple of (is_soccer, confidence).
        """
        score = 0.0
        max_score = 3.0
        
        detection = self.config.get("soccer_detection", {})
        keywords = detection.get("keywords", [])
        categories = detection.get("categories", [])
        
        # Check series ticker patterns
        for pattern in self._series_patterns:
            if pattern.search(event.series_ticker or ""):
                score += 1.0
                break
        
        # Check category
        if event.category and event.category.lower() in [c.lower() for c in categories]:
            score += 1.0
        
        # Check keywords in title
        title_lower = event.title.lower()
        for keyword in keywords:
            if keyword.lower() in title_lower:
                score += 0.5
                break
        
        # Check subtitle
        subtitle_lower = (event.subtitle or "").lower()
        for keyword in keywords:
            if keyword.lower() in subtitle_lower:
                score += 0.5
                break
        
        confidence = min(score / max_score, 1.0)
        is_soccer = confidence >= 0.3  # 30% threshold
        
        return is_soccer, confidence
    
    def _detect_soccer_from_market(self, market: MarketData) -> tuple[bool, float]:
        """Detect if market is soccer-related."""
        score = 0.0
        max_score = 3.0
        
        detection = self.config.get("soccer_detection", {})
        keywords = detection.get("keywords", [])
        
        # Check ticker patterns
        for pattern in self._series_patterns:
            if pattern.search(market.ticker or ""):
                score += 1.0
                break
        
        # Check keywords in title
        title_lower = market.title.lower()
        for keyword in keywords:
            if keyword.lower() in title_lower:
                score += 1.0
                break
        
        # Check subtitle
        subtitle_lower = (market.subtitle or "").lower()
        for keyword in keywords:
            if keyword.lower() in subtitle_lower:
                score += 0.5
                break
        
        confidence = min(score / max_score, 1.0)
        is_soccer = confidence >= 0.3
        
        return is_soccer, confidence
    
    def _detect_league(self, event: EventData) -> str | None:
        """Detect league from event."""
        # Check series ticker for common patterns
        ticker = event.series_ticker or ""
        title = event.title
        subtitle = event.subtitle or ""
        
        combined = f"{ticker} {title} {subtitle}".lower()
        
        # Try to normalize league from text
        result = self.league_normalizer.normalize(combined)
        if result.match_source == "alias":
            return result.canonical
        
        # Try common patterns
        patterns = {
            "premier_league": ["epl", "premier league", "english premier"],
            "la_liga": ["la liga", "laliga", "spanish"],
            "serie_a": ["serie a", "italian"],
            "bundesliga": ["bundesliga", "german"],
            "ligue_1": ["ligue 1", "french"],
            "mls": ["mls", "major league soccer"],
            "champions_league": ["champions league", "ucl"],
        }
        
        for key, terms in patterns.items():
            for term in terms:
                if term in combined:
                    return key
        
        return None
    
    def _parse_teams_from_title(self, title: str) -> tuple[str | None, str | None]:
        """Parse home and away teams from title.
        
        Returns:
            Tuple of (home_team, away_team) or (None, None).
        """
        for pattern in self._vs_patterns:
            match = pattern.search(title)
            if match:
                home = match.group("home").strip()
                away = match.group("away").strip()
                
                # Clean up extracted names
                home = self._clean_team_name(home)
                away = self._clean_team_name(away)
                
                if home and away:
                    return home, away
        
        return None, None
    
    def _clean_team_name(self, name: str) -> str:
        """Clean up extracted team name."""
        # Remove common prefixes/suffixes
        name = re.sub(r"^\s*(?:will|to|vs\.?|@)\s+", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+(?:win|draw|lose|beat).*$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\?$", "", name)
        return name.strip()
    
    def _detect_market_type(self, title: str, subtitle: str | None) -> str | None:
        """Detect market type from title/subtitle."""
        combined = f"{title} {subtitle or ""}".lower()
        
        # Check for specific patterns
        if "draw" in combined:
            return "DRAW"
        if "home" in combined and "win" in combined:
            return "HOME_WIN"
        if "away" in combined and "win" in combined:
            return "AWAY_WIN"
        if "to win" in combined or "will win" in combined:
            return "TEAM_WIN"
        if "moneyline" in combined:
            return "MONEYLINE"
        
        return None
    
    def _parse_target_team(self, title: str, subtitle: str | None) -> str | None:
        """Parse target team from market title."""
        for pattern in self._team_win_patterns:
            match = pattern.search(title)
            if match:
                team = match.group("team").strip()
                return self._clean_team_name(team)
        
        return None


class KalshiMarketRepository:
    """Database operations for Kalshi events and markets."""
    
    def __init__(self, db_connection: Any):
        """Initialize with database connection."""
        self.conn = db_connection
    
    def upsert_event(self, event: KalshiEventRecord) -> int:
        """Insert or update a Kalshi event."""
        cursor = self.conn.cursor()
        
        now = datetime.now(UTC).isoformat()
        raw_json = json.dumps(event.raw_data, default=str)
        strike_str = event.strike_date.isoformat() if event.strike_date else None
        
        cursor.execute("""
            INSERT INTO kalshi_events (
                event_ticker, series_ticker, title, subtitle, category,
                sub_category, strike_date, is_soccer, league_key,
                parsed_home_team, parsed_away_team, parsed_canonical_home,
                parsed_canonical_away, market_structure, raw_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_ticker) DO UPDATE SET
                series_ticker = excluded.series_ticker,
                title = excluded.title,
                subtitle = excluded.subtitle,
                category = excluded.category,
                sub_category = excluded.sub_category,
                strike_date = excluded.strike_date,
                is_soccer = excluded.is_soccer,
                league_key = excluded.league_key,
                parsed_home_team = excluded.parsed_home_team,
                parsed_away_team = excluded.parsed_away_team,
                parsed_canonical_home = excluded.parsed_canonical_home,
                parsed_canonical_away = excluded.parsed_canonical_away,
                market_structure = excluded.market_structure,
                raw_json = excluded.raw_json,
                updated_at = excluded.updated_at
        """, (
            event.event_ticker,
            event.series_ticker,
            event.title,
            event.subtitle,
            event.category,
            event.sub_category,
            strike_str,
            1 if event.is_soccer else 0,
            event.league_key,
            event.parsed_home_team,
            event.parsed_away_team,
            event.parsed_canonical_home,
            event.parsed_canonical_away,
            event.market_structure,
            raw_json,
            now,
            now,
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def upsert_market(self, market: KalshiMarketRecord) -> int:
        """Insert or update a Kalshi market."""
        cursor = self.conn.cursor()
        
        now = datetime.now(UTC).isoformat()
        raw_json = json.dumps(market.raw_data, default=str)
        
        open_str = market.open_time.isoformat() if market.open_time else None
        close_str = market.close_time.isoformat() if market.close_time else None
        exp_str = market.expiration_time.isoformat() if market.expiration_time else None
        
        cursor.execute("""
            INSERT INTO kalshi_markets (
                ticker, event_ticker, title, subtitle, status,
                open_time, close_time, expiration_time,
                yes_bid, yes_ask, no_bid, no_ask, last_price, volume,
                is_soccer, market_type, parsed_team, parsed_canonical_team,
                raw_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                is_soccer = excluded.is_soccer,
                market_type = excluded.market_type,
                parsed_team = excluded.parsed_team,
                parsed_canonical_team = excluded.parsed_canonical_team,
                raw_json = excluded.raw_json,
                updated_at = excluded.updated_at
        """, (
            market.ticker,
            market.event_ticker,
            market.title,
            market.subtitle,
            market.status,
            open_str,
            close_str,
            exp_str,
            market.yes_bid,
            market.yes_ask,
            market.no_bid,
            market.no_ask,
            market.last_price,
            market.volume,
            1 if market.is_soccer else 0,
            market.market_type,
            market.parsed_team,
            market.parsed_canonical_team,
            raw_json,
            now,
            now,
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_soccer_events(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[KalshiEventRecord]:
        """Get soccer events from database."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT event_ticker, series_ticker, title, subtitle, category,
                   sub_category, strike_date, is_soccer, league_key,
                   parsed_home_team, parsed_away_team, parsed_canonical_home,
                   parsed_canonical_away, market_structure, raw_json
            FROM kalshi_events
            WHERE is_soccer = 1
        """
        params: list[Any] = []
        
        if from_date:
            query += " AND strike_date >= ?"
            params.append(from_date.isoformat())
        
        if to_date:
            query += " AND strike_date <= ?"
            params.append(to_date.isoformat())
        
        query += " ORDER BY strike_date"
        
        cursor.execute(query, params)
        
        events = []
        for row in cursor.fetchall():
            strike_date = None
            if row[6]:
                try:
                    strike_date = datetime.fromisoformat(row[6])
                except ValueError:
                    pass
            
            events.append(KalshiEventRecord(
                event_ticker=row[0],
                series_ticker=row[1],
                title=row[2],
                subtitle=row[3],
                category=row[4],
                sub_category=row[5],
                strike_date=strike_date,
                is_soccer=bool(row[7]),
                league_key=row[8],
                parsed_home_team=row[9],
                parsed_away_team=row[10],
                parsed_canonical_home=row[11],
                parsed_canonical_away=row[12],
                market_structure=row[13],
                raw_data=json.loads(row[14]) if row[14] else {},
            ))
        
        return events
    
    def get_markets_for_event(self, event_ticker: str) -> list[KalshiMarketRecord]:
        """Get all markets for an event."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ticker, event_ticker, title, subtitle, status,
                   open_time, close_time, expiration_time,
                   yes_bid, yes_ask, no_bid, no_ask, last_price, volume,
                   is_soccer, market_type, parsed_team, parsed_canonical_team, raw_json
            FROM kalshi_markets
            WHERE event_ticker = ?
        """, (event_ticker,))
        
        markets = []
        for row in cursor.fetchall():
            markets.append(self._row_to_market_record(row))
        
        return markets
    
    def get_soccer_markets_by_date(
        self,
        from_date: datetime,
        to_date: datetime,
    ) -> list[KalshiMarketRecord]:
        """Get soccer markets within a date range."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ticker, event_ticker, title, subtitle, status,
                   open_time, close_time, expiration_time,
                   yes_bid, yes_ask, no_bid, no_ask, last_price, volume,
                   is_soccer, market_type, parsed_team, parsed_canonical_team, raw_json
            FROM kalshi_markets
            WHERE is_soccer = 1
              AND close_time >= ?
              AND close_time <= ?
            ORDER BY close_time
        """, (from_date.isoformat(), to_date.isoformat()))
        
        markets = []
        for row in cursor.fetchall():
            markets.append(self._row_to_market_record(row))
        
        return markets
    
    def _row_to_market_record(self, row: tuple) -> KalshiMarketRecord:
        """Convert database row to KalshiMarketRecord."""
        def parse_dt(s: str | None) -> datetime | None:
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None
        
        return KalshiMarketRecord(
            ticker=row[0],
            event_ticker=row[1],
            title=row[2],
            subtitle=row[3],
            status=row[4],
            open_time=parse_dt(row[5]),
            close_time=parse_dt(row[6]),
            expiration_time=parse_dt(row[7]),
            yes_bid=row[8],
            yes_ask=row[9],
            no_bid=row[10],
            no_ask=row[11],
            last_price=row[12],
            volume=row[13],
            is_soccer=bool(row[14]),
            market_type=row[15],
            parsed_team=row[16],
            parsed_canonical_team=row[17],
            raw_data=json.loads(row[18]) if row[18] else {},
        )


@dataclass
class SyncResult:
    """Result of syncing Kalshi markets."""
    
    events_found: int = 0
    events_soccer: int = 0
    markets_found: int = 0
    markets_soccer: int = 0
    errors: list[str] = field(default_factory=list)


async def sync_kalshi_soccer_markets(
    client: KalshiClient,
    db_connection: Any,
    days_ahead: int = 30,
) -> SyncResult:
    """Sync soccer markets from Kalshi to database.
    
    Args:
        client: Kalshi API client.
        db_connection: Database connection.
        days_ahead: Number of days ahead to fetch markets.
        
    Returns:
        Sync result statistics.
    """
    classifier = SoccerMarketClassifier()
    repo = KalshiMarketRepository(db_connection)
    result = SyncResult()
    
    try:
        # Fetch all events
        cursor = None
        all_events: list[EventData] = []
        
        while True:
            events, next_cursor = await client.list_events(
                status="open",
                limit=100,
                cursor=cursor,
            )
            all_events.extend(events)
            
            if not next_cursor:
                break
            cursor = next_cursor
        
        result.events_found = len(all_events)
        logger.info("kalshi_events_fetched", count=result.events_found)
        
        # Classify and store events
        for event in all_events:
            classification = classifier.classify_event(event)
            
            # Parse date from ticker if API doesn't provide strike_date
            strike_date = event.strike_date
            if not strike_date:
                strike_date = parse_date_from_ticker(event.event_ticker)
            
            event_record = KalshiEventRecord(
                event_ticker=event.event_ticker,
                series_ticker=event.series_ticker,
                title=event.title,
                subtitle=event.subtitle,
                category=event.category,
                sub_category=event.sub_category,
                strike_date=strike_date,
                is_soccer=classification.is_soccer,
                league_key=classification.league_key,
                parsed_home_team=classification.home_team,
                parsed_away_team=classification.away_team,
                parsed_canonical_home=classification.canonical_home,
                parsed_canonical_away=classification.canonical_away,
                market_structure=classification.structure_type,
                raw_data=event.raw_data,
            )
            
            repo.upsert_event(event_record)
            
            if classification.is_soccer:
                result.events_soccer += 1
                
                # Fetch markets for soccer events
                markets, _ = await client.list_markets(
                    event_ticker=event.event_ticker,
                    limit=100,
                )
                
                result.markets_found += len(markets)
                
                for market in markets:
                    market_classification = classifier.classify_market(
                        market, classification
                    )
                    
                    market_record = KalshiMarketRecord(
                        ticker=market.ticker,
                        event_ticker=market.event_ticker,
                        title=market.title,
                        subtitle=market.subtitle,
                        status=market.status,
                        open_time=market.open_time,
                        close_time=market.close_time,
                        expiration_time=market.expiration_time,
                        yes_bid=market.yes_bid,
                        yes_ask=market.yes_ask,
                        no_bid=market.no_bid,
                        no_ask=market.no_ask,
                        last_price=market.last_price,
                        volume=market.volume,
                        is_soccer=market_classification.is_soccer,
                        market_type=market_classification.market_type,
                        parsed_team=market_classification.target_team,
                        parsed_canonical_team=market_classification.canonical_target,
                        raw_data=market.raw_data,
                    )
                    
                    repo.upsert_market(market_record)
                    
                    if market_classification.is_soccer:
                        result.markets_soccer += 1
        
        logger.info(
            "kalshi_soccer_markets_synced",
            events_total=result.events_found,
            events_soccer=result.events_soccer,
            markets_total=result.markets_found,
            markets_soccer=result.markets_soccer,
        )
        
    except Exception as e:
        error_msg = f"Kalshi sync failed: {e}"
        result.errors.append(error_msg)
        logger.error("kalshi_sync_failed", error=str(e))
    
    return result
