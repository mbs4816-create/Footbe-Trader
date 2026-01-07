"""Fixture to Market Mapping Engine.

Universal mapping system that links API-Football fixtures to Kalshi market
tickers across any soccer league, using normalization, candidate generation,
and configurable scoring.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from footbe_trader.common.logging import get_logger
from footbe_trader.football.interfaces import FixtureData
from footbe_trader.strategy.kalshi_discovery import (
    KalshiEventRecord,
    KalshiMarketRecord,
    KalshiMarketRepository,
)
from footbe_trader.strategy.normalization import (
    fuzzy_match_ratio,
    get_league_normalizer,
    get_team_normalizer,
    normalize_team_name,
)

logger = get_logger(__name__)

# Default config paths (strategy -> footbe_trader -> src -> project_root)
DEFAULT_MAPPING_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "mapping_config.yaml"
DEFAULT_OVERRIDES_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "manual_market_overrides.yaml"


@dataclass
class MappingCandidate:
    """A candidate market mapping for a fixture."""
    
    event_ticker: str
    event_title: str
    
    # Individual market tickers
    ticker_home_win: str | None = None
    ticker_draw: str | None = None
    ticker_away_win: str | None = None
    
    # Market structure
    structure_type: str = "UNKNOWN"  # "1X2", "NO_DRAW", "BINARY", etc.
    
    # Scoring
    total_score: float = 0.0
    team_match_score: float = 0.0
    date_match_score: float = 0.0
    league_match_score: float = 0.0
    market_type_score: float = 0.0
    text_similarity_score: float = 0.0
    
    # Metadata
    matched_home_team: str | None = None
    matched_away_team: str | None = None
    event_strike_date: datetime | None = None
    close_time: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_ticker": self.event_ticker,
            "event_title": self.event_title,
            "ticker_home_win": self.ticker_home_win,
            "ticker_draw": self.ticker_draw,
            "ticker_away_win": self.ticker_away_win,
            "structure_type": self.structure_type,
            "total_score": self.total_score,
            "score_breakdown": {
                "team_match": self.team_match_score,
                "date_match": self.date_match_score,
                "league_match": self.league_match_score,
                "market_type": self.market_type_score,
                "text_similarity": self.text_similarity_score,
            },
            "matched_home_team": self.matched_home_team,
            "matched_away_team": self.matched_away_team,
        }


@dataclass
class FixtureMarketMapping:
    """A confirmed mapping between a fixture and Kalshi markets."""

    fixture_id: int
    mapping_version: int = 1

    # Team IDs (for model predictions)
    home_team_id: int = 0
    away_team_id: int = 0

    # Market structure
    structure_type: str = "UNKNOWN"

    # Tickers
    ticker_home_win: str | None = None
    ticker_draw: str | None = None
    ticker_away_win: str | None = None
    ticker_home_win_yes: str | None = None
    ticker_home_win_no: str | None = None
    ticker_away_win_yes: str | None = None
    ticker_away_win_no: str | None = None
    event_ticker: str | None = None

    # Confidence
    confidence_score: float = 0.0
    confidence_components: dict[str, float] = field(default_factory=dict)

    # Status
    status: str = "AUTO"  # "AUTO", "MANUAL_OVERRIDE", "REJECTED"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_id": self.fixture_id,
            "mapping_version": self.mapping_version,
            "structure_type": self.structure_type,
            "ticker_home_win": self.ticker_home_win,
            "ticker_draw": self.ticker_draw,
            "ticker_away_win": self.ticker_away_win,
            "ticker_home_win_yes": self.ticker_home_win_yes,
            "ticker_home_win_no": self.ticker_home_win_no,
            "ticker_away_win_yes": self.ticker_away_win_yes,
            "ticker_away_win_no": self.ticker_away_win_no,
            "event_ticker": self.event_ticker,
            "confidence_score": self.confidence_score,
            "confidence_components": self.confidence_components,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class MappingResult:
    """Result of mapping attempt for a fixture."""
    
    fixture_id: int
    fixture_info: dict[str, Any]  # League, teams, kickoff
    success: bool = False
    mapping: FixtureMarketMapping | None = None
    candidates: list[MappingCandidate] = field(default_factory=list)
    reason: str | None = None  # Why mapping failed/succeeded
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_id": self.fixture_id,
            "fixture_info": self.fixture_info,
            "success": self.success,
            "mapping": self.mapping.to_dict() if self.mapping else None,
            "candidates_count": len(self.candidates),
            "reason": self.reason,
        }


class MappingConfig:
    """Configuration for mapping engine."""
    
    def __init__(self, config_path: Path | str | None = None):
        """Load configuration."""
        self.config = self._load_config(config_path or DEFAULT_MAPPING_CONFIG_PATH)
    
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
            "time_window": {
                "hours_before": 24,
                "hours_after": 24,
                "preferred_close_window": 6,
            },
            "confidence": {
                "auto_accept_threshold": 0.85,
                "min_candidate_threshold": 0.30,
                "review_threshold": 0.70,
            },
            "scoring_weights": {
                "team_match": 0.40,
                "date_match": 0.20,
                "league_match": 0.15,
                "market_type": 0.10,
                "text_similarity": 0.15,
            },
            "team_matching": {
                "min_fuzzy_ratio": 80,
                "exact_match_bonus": 0.1,
                "order_swap_penalty": 0.15,
            },
            "market_type_preferences": {
                "1X2": 1.0,
                "NO_DRAW": 0.9,
                "MONEYLINE": 0.8,
                "HOME_WIN_BINARY": 0.6,
                "AWAY_WIN_BINARY": 0.6,
                "UNKNOWN": 0.3,
            },
        }
    
    @property
    def hours_before(self) -> int:
        return self.config.get("time_window", {}).get("hours_before", 24)
    
    @property
    def hours_after(self) -> int:
        return self.config.get("time_window", {}).get("hours_after", 24)
    
    @property
    def auto_accept_threshold(self) -> float:
        return self.config.get("confidence", {}).get("auto_accept_threshold", 0.85)
    
    @property
    def min_candidate_threshold(self) -> float:
        return self.config.get("confidence", {}).get("min_candidate_threshold", 0.30)
    
    @property
    def review_threshold(self) -> float:
        return self.config.get("confidence", {}).get("review_threshold", 0.70)
    
    @property
    def scoring_weights(self) -> dict[str, float]:
        return self.config.get("scoring_weights", {})
    
    @property
    def team_matching(self) -> dict[str, Any]:
        return self.config.get("team_matching", {})
    
    @property
    def market_type_preferences(self) -> dict[str, float]:
        return self.config.get("market_type_preferences", {})


class ManualOverrides:
    """Manages manual market mapping overrides."""
    
    def __init__(self, overrides_path: Path | str | None = None):
        """Load overrides from YAML."""
        self.overrides: dict[str, Any] = {}
        self._load(overrides_path or DEFAULT_OVERRIDES_PATH)
    
    def _load(self, path: Path | str) -> None:
        """Load overrides from file."""
        path = Path(path)
        if not path.exists():
            return
        
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        
        self.overrides = data.get("overrides", {})
        
        # Also load fixture-specific overrides
        for key, value in data.items():
            if key.startswith("fixture_") or key.startswith("match_"):
                self.overrides[key] = value
        
        logger.info("manual_overrides_loaded", count=len(self.overrides))
    
    def get_override(
        self,
        fixture_id: int | None = None,
        home_team: str | None = None,
        away_team: str | None = None,
        kickoff_date: str | None = None,
    ) -> FixtureMarketMapping | None:
        """Look up manual override for a fixture.
        
        Args:
            fixture_id: API-Football fixture ID.
            home_team: Canonical home team name.
            away_team: Canonical away team name.
            kickoff_date: Kickoff date string (YYYY-MM-DD).
            
        Returns:
            FixtureMarketMapping if override exists, None otherwise.
        """
        # Check by fixture ID
        if fixture_id:
            key = f"fixture_{fixture_id}"
            if key in self.overrides:
                return self._parse_override(fixture_id, self.overrides[key])
        
        # Check by teams + date
        for key, data in self.overrides.items():
            if not key.startswith("match_"):
                continue
            
            override_home = (data.get("home_team") or "").lower()
            override_away = (data.get("away_team") or "").lower()
            override_date = data.get("kickoff_date")
            
            home_match = home_team and override_home == home_team.lower()
            away_match = away_team and override_away == away_team.lower()
            date_match = kickoff_date and override_date == kickoff_date
            
            if home_match and away_match and date_match:
                return self._parse_override(fixture_id or 0, data)
        
        return None
    
    def _parse_override(
        self,
        fixture_id: int,
        data: dict[str, Any],
    ) -> FixtureMarketMapping:
        """Parse override data into mapping."""
        return FixtureMarketMapping(
            fixture_id=fixture_id,
            structure_type=data.get("structure_type", "UNKNOWN"),
            ticker_home_win=data.get("ticker_home_win"),
            ticker_draw=data.get("ticker_draw"),
            ticker_away_win=data.get("ticker_away_win"),
            ticker_home_win_yes=data.get("ticker_home_win_yes"),
            ticker_home_win_no=data.get("ticker_home_win_no"),
            ticker_away_win_yes=data.get("ticker_away_win_yes"),
            ticker_away_win_no=data.get("ticker_away_win_no"),
            event_ticker=data.get("event_ticker"),
            confidence_score=1.0,
            status="MANUAL_OVERRIDE",
            metadata={"notes": data.get("notes")},
        )


class FixtureMarketMapper:
    """Maps fixtures to Kalshi markets."""
    
    def __init__(
        self,
        kalshi_repo: KalshiMarketRepository,
        config: MappingConfig | None = None,
        overrides: ManualOverrides | None = None,
    ):
        """Initialize mapper.
        
        Args:
            kalshi_repo: Repository for Kalshi market data.
            config: Mapping configuration.
            overrides: Manual override settings.
        """
        self.kalshi_repo = kalshi_repo
        self.config = config or MappingConfig()
        self.overrides = overrides or ManualOverrides()
        self.team_normalizer = get_team_normalizer()
        self.league_normalizer = get_league_normalizer()
    
    def map_fixture(
        self,
        fixture: FixtureData,
        home_team_name: str,
        away_team_name: str,
    ) -> MappingResult:
        """Map a fixture to Kalshi markets.
        
        Args:
            fixture: API-Football fixture data.
            home_team_name: Home team name.
            away_team_name: Away team name.
            
        Returns:
            Mapping result with candidates and optional confirmed mapping.
        """
        # Get canonical names
        league_key = self.league_normalizer.get_key_for_league_id(fixture.league_id)
        canonical_home = normalize_team_name(home_team_name, league_key)
        canonical_away = normalize_team_name(away_team_name, league_key)
        
        fixture_info = {
            "fixture_id": fixture.fixture_id,
            "league_id": fixture.league_id,
            "league_key": league_key,
            "home_team": home_team_name,
            "away_team": away_team_name,
            "canonical_home": canonical_home,
            "canonical_away": canonical_away,
            "kickoff_utc": fixture.kickoff_utc.isoformat() if fixture.kickoff_utc else None,
        }
        
        result = MappingResult(
            fixture_id=fixture.fixture_id,
            fixture_info=fixture_info,
        )
        
        # Check for manual override
        kickoff_date = fixture.kickoff_utc.strftime("%Y-%m-%d") if fixture.kickoff_utc else None
        override = self.overrides.get_override(
            fixture_id=fixture.fixture_id,
            home_team=canonical_home,
            away_team=canonical_away,
            kickoff_date=kickoff_date,
        )
        
        if override:
            result.success = True
            result.mapping = override
            result.reason = "Manual override applied"
            logger.info(
                "fixture_mapped_override",
                fixture_id=fixture.fixture_id,
                event_ticker=override.event_ticker,
            )
            return result
        
        # Generate candidates
        candidates = self._generate_candidates(
            fixture, canonical_home, canonical_away, league_key
        )
        result.candidates = candidates
        
        if not candidates:
            result.reason = "No candidate markets found"
            logger.info(
                "fixture_no_candidates",
                fixture_id=fixture.fixture_id,
                home=canonical_home,
                away=canonical_away,
            )
            return result
        
        # Sort by score and pick best
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        best = candidates[0]
        
        # Check confidence thresholds
        if best.total_score >= self.config.auto_accept_threshold:
            result.success = True
            result.mapping = self._candidate_to_mapping(fixture.fixture_id, best)
            result.reason = f"Auto-accepted with confidence {best.total_score:.2f}"
        elif best.total_score >= self.config.review_threshold:
            result.reason = f"Needs review: confidence {best.total_score:.2f}"
        else:
            result.reason = f"Low confidence: {best.total_score:.2f}"
        
        logger.info(
            "fixture_mapping_result",
            fixture_id=fixture.fixture_id,
            success=result.success,
            best_score=best.total_score,
            candidates=len(candidates),
            reason=result.reason,
        )
        
        return result
    
    def _generate_candidates(
        self,
        fixture: FixtureData,
        canonical_home: str,
        canonical_away: str,
        league_key: str | None,
    ) -> list[MappingCandidate]:
        """Generate candidate market mappings.
        
        Args:
            fixture: Fixture data.
            canonical_home: Canonical home team name.
            canonical_away: Canonical away team name.
            league_key: Canonical league key.
            
        Returns:
            List of scored candidates.
        """
        if not fixture.kickoff_utc:
            return []
        
        # Time window for candidate search
        from_date = fixture.kickoff_utc - timedelta(hours=self.config.hours_before)
        to_date = fixture.kickoff_utc + timedelta(hours=self.config.hours_after)
        
        # Get soccer events within window
        events = self.kalshi_repo.get_soccer_events(from_date, to_date)
        
        candidates = []
        expected_title = f"{canonical_home} vs {canonical_away}"
        
        for event in events:
            # Score this event as a candidate
            candidate = self._score_candidate(
                event=event,
                fixture=fixture,
                canonical_home=canonical_home,
                canonical_away=canonical_away,
                league_key=league_key,
                expected_title=expected_title,
            )
            
            if candidate and candidate.total_score >= self.config.min_candidate_threshold:
                # Get markets for this event
                markets = self.kalshi_repo.get_markets_for_event(event.event_ticker)
                self._assign_market_tickers(candidate, markets)
                candidates.append(candidate)
        
        return candidates
    
    def _score_candidate(
        self,
        event: KalshiEventRecord,
        fixture: FixtureData,
        canonical_home: str,
        canonical_away: str,
        league_key: str | None,
        expected_title: str,
    ) -> MappingCandidate | None:
        """Score a candidate event for matching.
        
        Returns:
            Scored candidate or None if clearly not a match.
        """
        weights = self.config.scoring_weights
        team_config = self.config.team_matching
        
        candidate = MappingCandidate(
            event_ticker=event.event_ticker,
            event_title=event.title,
            event_strike_date=event.strike_date,
        )
        
        # 1. Team matching score
        event_home = event.parsed_canonical_home or ""
        event_away = event.parsed_canonical_away or ""
        
        home_home_ratio = fuzzy_match_ratio(canonical_home, event_home)
        away_away_ratio = fuzzy_match_ratio(canonical_away, event_away)
        
        # Check for swapped order
        home_away_ratio = fuzzy_match_ratio(canonical_home, event_away)
        away_home_ratio = fuzzy_match_ratio(canonical_away, event_home)
        
        min_ratio = team_config.get("min_fuzzy_ratio", 80)
        
        if home_home_ratio >= min_ratio and away_away_ratio >= min_ratio:
            # Correct order
            avg_ratio = (home_home_ratio + away_away_ratio) / 2
            candidate.team_match_score = avg_ratio / 100.0
            candidate.matched_home_team = event_home
            candidate.matched_away_team = event_away
            
            # Bonus for exact match
            if home_home_ratio == 100 and away_away_ratio == 100:
                candidate.team_match_score += team_config.get("exact_match_bonus", 0.1)
                candidate.team_match_score = min(1.0, candidate.team_match_score)
                
        elif home_away_ratio >= min_ratio and away_home_ratio >= min_ratio:
            # Swapped order
            avg_ratio = (home_away_ratio + away_home_ratio) / 2
            candidate.team_match_score = avg_ratio / 100.0
            candidate.team_match_score -= team_config.get("order_swap_penalty", 0.15)
            candidate.team_match_score = max(0.0, candidate.team_match_score)
            candidate.matched_home_team = event_away
            candidate.matched_away_team = event_home
        else:
            # Teams don't match well enough
            return None
        
        # 2. Date matching score
        if fixture.kickoff_utc and event.strike_date:
            time_diff = abs((fixture.kickoff_utc - event.strike_date).total_seconds())
            hours_diff = time_diff / 3600.0
            
            # Perfect match within 6 hours = 1.0, decays from there
            if hours_diff <= 6:
                candidate.date_match_score = 1.0
            elif hours_diff <= 24:
                candidate.date_match_score = 1.0 - (hours_diff - 6) / 36.0
            else:
                candidate.date_match_score = max(0.0, 0.5 - (hours_diff - 24) / 48.0)
        
        # 3. League matching score
        if league_key and event.league_key:
            if league_key == event.league_key:
                candidate.league_match_score = 1.0
            else:
                candidate.league_match_score = 0.0
        else:
            # Unknown league - neutral score
            candidate.league_match_score = 0.5
        
        # 4. Market type score
        structure = event.market_structure or "UNKNOWN"
        candidate.structure_type = structure
        preferences = self.config.market_type_preferences
        candidate.market_type_score = preferences.get(structure, preferences.get("UNKNOWN", 0.3))
        
        # 5. Text similarity score
        title_similarity = fuzzy_match_ratio(expected_title, event.title)
        candidate.text_similarity_score = title_similarity / 100.0
        
        # Calculate weighted total
        candidate.total_score = (
            candidate.team_match_score * weights.get("team_match", 0.4) +
            candidate.date_match_score * weights.get("date_match", 0.2) +
            candidate.league_match_score * weights.get("league_match", 0.15) +
            candidate.market_type_score * weights.get("market_type", 0.1) +
            candidate.text_similarity_score * weights.get("text_similarity", 0.15)
        )
        
        return candidate
    
    def _assign_market_tickers(
        self,
        candidate: MappingCandidate,
        markets: list[KalshiMarketRecord],
    ) -> None:
        """Assign specific market tickers to candidate.
        
        Identifies which markets are for home win, draw, away win, etc.
        Kalshi uses ticker suffixes like -TOT, -TIE, -BOU for different outcomes.
        """
        for market in markets:
            market_type = market.market_type or ""
            title_lower = market.title.lower()
            ticker = market.ticker or ""
            
            # Parse the ticker suffix (e.g., KXEPLGAME-26JAN07BOUTOT-TIE -> TIE)
            ticker_suffix = ticker.split('-')[-1] if '-' in ticker else ""
            
            # Check for draw (TIE suffix or "draw" in title)
            if ticker_suffix == "TIE" or "draw" in title_lower or market_type == "DRAW":
                candidate.ticker_draw = market.ticker
                continue
            
            # Check for home win by market_type
            if market_type == "HOME_WIN":
                candidate.ticker_home_win = market.ticker
                continue
            
            # Check for away win by market_type
            if market_type == "AWAY_WIN":
                candidate.ticker_away_win = market.ticker
                continue
            
            # Match by ticker suffix against team names
            if ticker_suffix and candidate.matched_home_team and candidate.matched_away_team:
                # Get first 3 chars of team names for matching
                home_abbrevs = self._get_team_abbreviations(candidate.matched_home_team)
                away_abbrevs = self._get_team_abbreviations(candidate.matched_away_team)
                
                if ticker_suffix.upper() in home_abbrevs:
                    candidate.ticker_home_win = market.ticker
                    continue
                elif ticker_suffix.upper() in away_abbrevs:
                    candidate.ticker_away_win = market.ticker
                    continue
            
            # Try to match by parsed team name
            if market.parsed_canonical_team:
                if candidate.matched_home_team:
                    ratio = fuzzy_match_ratio(
                        market.parsed_canonical_team,
                        candidate.matched_home_team,
                    )
                    if ratio >= 80:
                        candidate.ticker_home_win = market.ticker
                        continue
                
                if candidate.matched_away_team:
                    ratio = fuzzy_match_ratio(
                        market.parsed_canonical_team,
                        candidate.matched_away_team,
                    )
                    if ratio >= 80:
                        candidate.ticker_away_win = market.ticker
                        continue
        
        # Determine structure type based on available tickers
        if candidate.ticker_home_win and candidate.ticker_draw and candidate.ticker_away_win:
            candidate.structure_type = "1X2"
        elif candidate.ticker_home_win and candidate.ticker_away_win:
            candidate.structure_type = "NO_DRAW"
        elif candidate.ticker_home_win:
            candidate.structure_type = "HOME_WIN_BINARY"
        elif candidate.ticker_away_win:
            candidate.structure_type = "AWAY_WIN_BINARY"
        else:
            candidate.structure_type = "UNKNOWN"
    
    def _get_team_abbreviations(self, team_name: str) -> set[str]:
        """Get possible abbreviations for a team name.
        
        Examples:
            'Bournemouth' -> {'BOU', 'BOURNEMOUTH'}
            'Manchester City' -> {'MCI', 'MAN', 'MANCHESTERCITY'}
            'Tottenham' -> {'TOT', 'TOTTENHAM'}
        """
        abbrevs = set()
        name_upper = team_name.upper()
        
        # Add full name without spaces
        abbrevs.add(name_upper.replace(' ', ''))
        
        # First 3 letters
        abbrevs.add(name_upper[:3])
        
        # Common abbreviations for Premier League teams
        team_abbrev_map = {
            'ARSENAL': ['ARS', 'AFC'],
            'ASTON VILLA': ['AVL', 'AVA', 'ASTONVILLA'],
            'BOURNEMOUTH': ['BOU', 'AFC'],
            'BRENTFORD': ['BRE', 'BFC'],
            'BRIGHTON': ['BRI', 'BHA', 'BRIGHTON'],
            'BURNLEY': ['BUR', 'BFC'],
            'CHELSEA': ['CHE', 'CFC'],
            'CRYSTAL PALACE': ['CRY', 'CPL', 'CRYSTALPALACE'],
            'EVERTON': ['EVE', 'EFC'],
            'FULHAM': ['FUL', 'FFC'],
            'LEEDS': ['LEE', 'LEU', 'LEEDSUNITED'],
            'LEICESTER': ['LEI', 'LFC'],
            'LIVERPOOL': ['LIV', 'LFC'],
            'MANCHESTER CITY': ['MCI', 'MAN', 'MANCHESTERCITY'],
            'MANCHESTER UNITED': ['MUN', 'MAN', 'MANCHESTERUNITED', 'MANUTD'],
            'NEWCASTLE': ['NEW', 'NFC', 'NEWCASTLE'],
            'NOTTINGHAM': ['NFO', 'NOT', 'NOTTINGHAMFOREST', 'FOREST'],
            'NOTTINGHAM FOREST': ['NFO', 'NOT', 'NOTTINGHAMFOREST', 'FOREST'],
            'SHEFFIELD': ['SHU', 'SFC'],
            'SOUTHAMPTON': ['SOU', 'SFC'],
            'SUNDERLAND': ['SUN', 'SAFC'],
            'TOTTENHAM': ['TOT', 'THS', 'SPURS'],
            'WEST HAM': ['WHU', 'WES', 'WESTHAM'],
            'WOLVERHAMPTON': ['WOL', 'WWC', 'WOLVES'],
            'WOLVES': ['WOL', 'WWC', 'WOLVES'],
            # Bundesliga
            'BAYERN': ['BMU', 'FCB', 'BAY'],
            'BAYERN MÜNCHEN': ['BMU', 'FCB', 'BAY'],
            'BAYERN MUNICH': ['BMU', 'FCB', 'BAY'],
            'WOLFSBURG': ['WOB', 'VFL'],
            'VFL WOLFSBURG': ['WOB', 'VFL'],
            'BORUSSIA DORTMUND': ['BVB', 'DOR'],
            'DORTMUND': ['BVB', 'DOR'],
            'LEVERKUSEN': ['B04', 'LEV'],
            'BAYER LEVERKUSEN': ['B04', 'LEV'],
            'LEIPZIG': ['RBL', 'LEI'],
            'RB LEIPZIG': ['RBL', 'LEI'],
            'EINTRACHT FRANKFURT': ['SGE', 'FRA'],
            'FRANKFURT': ['SGE', 'FRA'],
            'STUTTGART': ['VFB', 'STU'],
            'VFB STUTTGART': ['VFB', 'STU'],
            'WERDER BREMEN': ['SVW', 'BRE'],
            'BREMEN': ['SVW', 'BRE'],
            'HOFFENHEIM': ['TSG', 'HOF'],
            '1899 HOFFENHEIM': ['TSG', 'HOF'],
            'KÖLN': ['KOE', 'COL'],
            'FC KÖLN': ['KOE', 'COL'],
            '1. FC KÖLN': ['KOE', 'COL'],
            'HEIDENHEIM': ['HDH', 'HEI'],
            '1. FC HEIDENHEIM': ['HDH', 'HEI'],
            'MAINZ': ['M05', 'MAI'],
            'FSV MAINZ': ['M05', 'MAI'],
            'UNION BERLIN': ['UNI', 'UNB'],
            'BERLIN': ['UNI', 'UNB'],
            # Serie A
            'INTER': ['INT', 'INTER'],
            'AC MILAN': ['ACM', 'MIL'],
            'MILAN': ['ACM', 'MIL'],
            'JUVENTUS': ['JUV', 'JUVE'],
            'NAPOLI': ['NAP', 'SSC'],
            'ROMA': ['ROM', 'ASR'],
            'AS ROMA': ['ROM', 'ASR'],
            'LAZIO': ['LAZ', 'SSL'],
            'ATALANTA': ['ATA', 'ATALANTA'],
            'FIORENTINA': ['FIO', 'ACF'],
            'BOLOGNA': ['BFC', 'BOL'],
            'TORINO': ['TOR', 'TORINO'],
            'UDINESE': ['UDI', 'UDINESE'],
            'VERONA': ['VER', 'HELLAS'],
            'CAGLIARI': ['CAG', 'CAGLIARI'],
            'GENOA': ['GEN', 'GENOA'],
            'LECCE': ['LEC', 'LECCE'],
            'SASSUOLO': ['SAS', 'SASSUOLO'],
            'PARMA': ['PAR', 'PARMA'],
            'COMO': ['COM', 'COMO'],
            'PISA': ['PIS', 'PISA'],
            'CREMONESE': ['CRE', 'CREMONESE'],
            # La Liga
            'REAL MADRID': ['RMA', 'MAD'],
            'BARCELONA': ['BAR', 'FCB'],
            'ATLETICO MADRID': ['ATM', 'ATL'],
            'ATLETICO': ['ATM', 'ATL'],
            'SEVILLA': ['SEV', 'SEVILLA'],
            'VALENCIA': ['VAL', 'VALENCIA'],
            'VILLARREAL': ['VIL', 'VILLARREAL'],
            'REAL SOCIEDAD': ['RSO', 'REALSOCIEDAD'],
            'BETIS': ['RBB', 'BETIS'],
            'REAL BETIS': ['RBB', 'BETIS'],
            'ATHLETIC': ['ATH', 'ATHLETIC'],
            'GETAFE': ['GET', 'GETAFE'],
            'ESPANYOL': ['ESP', 'ESPANYOL'],
            'CELTA': ['CEL', 'CELTA'],
            'CELTA VIGO': ['CEL', 'CELTA'],
            'OSASUNA': ['OSA', 'OSASUNA'],
            'ALAVES': ['ALA', 'ALAVES'],
            'GIRONA': ['GIR', 'GIRONA'],
            'LEVANTE': ['LEV', 'LEVANTE'],
            'OVIEDO': ['OVI', 'OVIEDO'],
            'ELCHE': ['ELC', 'ELCHE'],
        }
        
        for team_key, team_abbrevs in team_abbrev_map.items():
            if team_key in name_upper or name_upper in team_key:
                abbrevs.update(team_abbrevs)
        
        return abbrevs
    
    def _candidate_to_mapping(
        self,
        fixture_id: int,
        candidate: MappingCandidate,
    ) -> FixtureMarketMapping:
        """Convert candidate to confirmed mapping."""
        return FixtureMarketMapping(
            fixture_id=fixture_id,
            structure_type=candidate.structure_type,
            ticker_home_win=candidate.ticker_home_win,
            ticker_draw=candidate.ticker_draw,
            ticker_away_win=candidate.ticker_away_win,
            event_ticker=candidate.event_ticker,
            confidence_score=candidate.total_score,
            confidence_components={
                "team_match": candidate.team_match_score,
                "date_match": candidate.date_match_score,
                "league_match": candidate.league_match_score,
                "market_type": candidate.market_type_score,
                "text_similarity": candidate.text_similarity_score,
            },
            status="AUTO",
            metadata={
                "matched_home_team": candidate.matched_home_team,
                "matched_away_team": candidate.matched_away_team,
                "event_title": candidate.event_title,
            },
        )


class MappingRepository:
    """Database operations for fixture-market mappings."""
    
    def __init__(self, db_connection: Any):
        """Initialize with database connection."""
        self.conn = db_connection
    
    def save_mapping(self, mapping: FixtureMarketMapping) -> int:
        """Save a fixture-market mapping.
        
        Args:
            mapping: Mapping to save.
            
        Returns:
            Database row ID.
        """
        cursor = self.conn.cursor()
        now = datetime.now(UTC).isoformat()
        
        components_json = json.dumps(mapping.confidence_components)
        metadata_json = json.dumps(mapping.metadata, default=str)
        
        cursor.execute("""
            INSERT INTO fixture_market_map (
                fixture_id, mapping_version, structure_type,
                ticker_home_win, ticker_draw, ticker_away_win,
                ticker_home_win_yes, ticker_home_win_no,
                ticker_away_win_yes, ticker_away_win_no,
                event_ticker, confidence_score, confidence_components,
                status, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fixture_id, mapping_version) DO UPDATE SET
                structure_type = excluded.structure_type,
                ticker_home_win = excluded.ticker_home_win,
                ticker_draw = excluded.ticker_draw,
                ticker_away_win = excluded.ticker_away_win,
                ticker_home_win_yes = excluded.ticker_home_win_yes,
                ticker_home_win_no = excluded.ticker_home_win_no,
                ticker_away_win_yes = excluded.ticker_away_win_yes,
                ticker_away_win_no = excluded.ticker_away_win_no,
                event_ticker = excluded.event_ticker,
                confidence_score = excluded.confidence_score,
                confidence_components = excluded.confidence_components,
                status = excluded.status,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
        """, (
            mapping.fixture_id,
            mapping.mapping_version,
            mapping.structure_type,
            mapping.ticker_home_win,
            mapping.ticker_draw,
            mapping.ticker_away_win,
            mapping.ticker_home_win_yes,
            mapping.ticker_home_win_no,
            mapping.ticker_away_win_yes,
            mapping.ticker_away_win_no,
            mapping.event_ticker,
            mapping.confidence_score,
            components_json,
            mapping.status,
            metadata_json,
            now,
            now,
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_mapping(self, fixture_id: int) -> FixtureMarketMapping | None:
        """Get mapping for a fixture.
        
        Args:
            fixture_id: API-Football fixture ID.
            
        Returns:
            Most recent mapping if exists, None otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fixture_id, mapping_version, structure_type,
                   ticker_home_win, ticker_draw, ticker_away_win,
                   ticker_home_win_yes, ticker_home_win_no,
                   ticker_away_win_yes, ticker_away_win_no,
                   event_ticker, confidence_score, confidence_components,
                   status, metadata_json
            FROM fixture_market_map
            WHERE fixture_id = ?
            ORDER BY mapping_version DESC
            LIMIT 1
        """, (fixture_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return FixtureMarketMapping(
            fixture_id=row[0],
            mapping_version=row[1],
            structure_type=row[2],
            ticker_home_win=row[3],
            ticker_draw=row[4],
            ticker_away_win=row[5],
            ticker_home_win_yes=row[6],
            ticker_home_win_no=row[7],
            ticker_away_win_yes=row[8],
            ticker_away_win_no=row[9],
            event_ticker=row[10],
            confidence_score=row[11],
            confidence_components=json.loads(row[12]) if row[12] else {},
            status=row[13],
            metadata=json.loads(row[14]) if row[14] else {},
        )
    
    def get_mappings_for_league(
        self,
        league_key: str,
        min_confidence: float = 0.0,
    ) -> list[FixtureMarketMapping]:
        """Get all mappings for a league."""
        # This requires joining with fixtures table
        # For now, return all mappings above threshold
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT fixture_id, mapping_version, structure_type,
                   ticker_home_win, ticker_draw, ticker_away_win,
                   ticker_home_win_yes, ticker_home_win_no,
                   ticker_away_win_yes, ticker_away_win_no,
                   event_ticker, confidence_score, confidence_components,
                   status, metadata_json
            FROM fixture_market_map
            WHERE confidence_score >= ?
            ORDER BY fixture_id
        """, (min_confidence,))
        
        mappings = []
        for row in cursor.fetchall():
            mappings.append(FixtureMarketMapping(
                fixture_id=row[0],
                mapping_version=row[1],
                structure_type=row[2],
                ticker_home_win=row[3],
                ticker_draw=row[4],
                ticker_away_win=row[5],
                ticker_home_win_yes=row[6],
                ticker_home_win_no=row[7],
                ticker_away_win_yes=row[8],
                ticker_away_win_no=row[9],
                event_ticker=row[10],
                confidence_score=row[11],
                confidence_components=json.loads(row[12]) if row[12] else {},
                status=row[13],
                metadata=json.loads(row[14]) if row[14] else {},
            ))
        
        return mappings
    
    def save_review(
        self,
        fixture_id: int,
        fixture_info: dict[str, Any],
        candidates: list[MappingCandidate],
    ) -> int:
        """Save mapping for review.
        
        Args:
            fixture_id: Fixture ID.
            fixture_info: Fixture details.
            candidates: Top candidate mappings.
            
        Returns:
            Review row ID.
        """
        cursor = self.conn.cursor()
        now = datetime.now(UTC).isoformat()
        
        info_json = json.dumps(fixture_info, default=str)
        candidates_json = json.dumps(
            [c.to_dict() for c in candidates[:5]],
            default=str,
        )
        
        cursor.execute("""
            INSERT INTO mapping_reviews (
                fixture_id, fixture_info, candidate_count, top_candidates,
                review_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'PENDING', ?, ?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                fixture_info = excluded.fixture_info,
                candidate_count = excluded.candidate_count,
                top_candidates = excluded.top_candidates,
                updated_at = excluded.updated_at
        """, (
            fixture_id,
            info_json,
            len(candidates),
            candidates_json,
            now,
            now,
        ))
        
        self.conn.commit()
        return cursor.lastrowid


def get_mapped_markets(
    db_connection: Any,
    fixture_id: int,
) -> FixtureMarketMapping | None:
    """Get mapped markets for a fixture.
    
    This is the main API for other modules to request mappings.
    
    Args:
        db_connection: Database connection.
        fixture_id: API-Football fixture ID.
        
    Returns:
        Mapping if found and valid, None otherwise.
    """
    repo = MappingRepository(db_connection)
    mapping = repo.get_mapping(fixture_id)
    
    if mapping and mapping.status != "REJECTED":
        return mapping
    
    return None
