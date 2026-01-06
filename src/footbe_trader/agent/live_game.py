"""Live game state management for in-game trading.

This module provides unified access to live game state from both
Football and NBA APIs, enabling the agent to:

1. Detect when games are live vs scheduled vs finished
2. Get current scores for live games
3. Adjust trading strategy based on game state
4. Calculate time-adjusted probabilities
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.football.client import FootballApiClient
from footbe_trader.football.interfaces import FixtureData, FixtureStatus
from footbe_trader.nba.client import NBAApiClient
from footbe_trader.nba.interfaces import NBAGame, NBAGameStatus

logger = get_logger(__name__)


class GamePhase(str, Enum):
    """Unified game phase across sports."""
    
    # Pre-game phases
    SCHEDULED = "scheduled"  # Game scheduled but not today
    PREGAME = "pregame"      # Game starts within a few hours
    IMMINENT = "imminent"    # Game starts within 15 minutes
    
    # Live phases
    EARLY = "early"          # First quarter/half (0-25% of game time)
    MIDDLE = "middle"        # Second-third quarter (25-75% of game time)
    LATE = "late"           # Fourth quarter/last 15 min (75-90%)
    CLOSING = "closing"     # Final minutes (90-100%)
    
    # Breaks
    HALFTIME = "halftime"
    BREAK = "break"
    
    # End phases
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


@dataclass
class LiveScore:
    """Current score state."""
    
    home_score: int
    away_score: int
    
    @property
    def score_diff(self) -> int:
        """Home team lead (negative = away leads)."""
        return self.home_score - self.away_score
    
    @property
    def total_score(self) -> int:
        """Total points scored."""
        return self.home_score + self.away_score
    
    @property
    def is_tied(self) -> bool:
        """Game is tied."""
        return self.home_score == self.away_score


@dataclass
class GameTiming:
    """Game timing information."""
    
    kickoff_time: datetime
    current_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    elapsed_minutes: int | None = None  # Minutes played
    total_minutes: int = 90  # Expected total minutes (90 for soccer, 48 for NBA)
    period: str | None = None  # "1H", "2H", "Q1", "Q2", etc.
    
    @property
    def minutes_to_kickoff(self) -> float:
        """Minutes until game starts (negative if already started)."""
        delta = (self.kickoff_time - self.current_time).total_seconds() / 60
        return delta
    
    @property
    def minutes_remaining(self) -> int | None:
        """Estimated minutes remaining in game."""
        if self.elapsed_minutes is None:
            return None
        return max(0, self.total_minutes - self.elapsed_minutes)
    
    @property
    def game_progress(self) -> float | None:
        """Game progress as percentage (0.0 to 1.0)."""
        if self.elapsed_minutes is None:
            return None
        return min(1.0, self.elapsed_minutes / self.total_minutes)


@dataclass 
class LiveGameState:
    """Comprehensive live game state for trading decisions.
    
    This provides everything the agent needs to make informed
    in-game trading decisions.
    """
    
    # Identifiers
    fixture_id: int | None = None  # Football fixture ID
    game_id: int | None = None     # NBA game ID
    sport: str = "football"        # "football" or "nba"
    
    # Current state
    phase: GamePhase = GamePhase.SCHEDULED
    score: LiveScore | None = None
    timing: GameTiming | None = None
    
    # Team info
    home_team: str = ""
    away_team: str = ""
    
    # Trading implications
    is_tradeable: bool = True  # Can we place trades on this game?
    stale_reason: str | None = None  # Why we can't trade (if applicable)
    
    # Model adjustments (calculated based on live state)
    home_win_adjustment: float = 0.0  # Add to pre-match prob
    draw_adjustment: float = 0.0      # Football only
    away_win_adjustment: float = 0.0
    
    # Raw data for debugging
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_live(self) -> bool:
        """Game is currently in progress."""
        return self.phase in {
            GamePhase.EARLY,
            GamePhase.MIDDLE,
            GamePhase.LATE,
            GamePhase.CLOSING,
            GamePhase.HALFTIME,
            GamePhase.BREAK,
        }
    
    @property
    def is_pregame(self) -> bool:
        """Game hasn't started yet."""
        return self.phase in {
            GamePhase.SCHEDULED,
            GamePhase.PREGAME,
            GamePhase.IMMINENT,
        }
    
    @property
    def is_ended(self) -> bool:
        """Game has finished."""
        return self.phase in {
            GamePhase.FINISHED,
            GamePhase.POSTPONED,
            GamePhase.CANCELLED,
        }
    
    @property
    def home_is_winning(self) -> bool:
        """Home team currently winning."""
        if self.score is None:
            return False
        return self.score.home_score > self.score.away_score
    
    @property
    def away_is_winning(self) -> bool:
        """Away team currently winning."""
        if self.score is None:
            return False
        return self.score.away_score > self.score.home_score


class LiveGameStateProvider:
    """Fetches and provides live game state from APIs.
    
    Usage:
        async with LiveGameStateProvider(football_client, nba_client) as provider:
            state = await provider.get_football_game_state(fixture_id)
            state = await provider.get_nba_game_state(game_id)
    """
    
    def __init__(
        self,
        football_client: FootballApiClient | None = None,
        nba_client: NBAApiClient | None = None,
    ):
        """Initialize provider.
        
        Args:
            football_client: Football API client (optional).
            nba_client: NBA API client (optional).
        """
        self.football_client = football_client
        self.nba_client = nba_client
        
        # Cache for live games to reduce API calls
        self._live_football_cache: dict[int, LiveGameState] = {}
        self._live_nba_cache: dict[int, LiveGameState] = {}
        self._cache_ttl = timedelta(seconds=30)  # Cache for 30 seconds
        self._cache_time: datetime | None = None
    
    async def get_football_game_state(
        self,
        fixture_id: int,
        use_cache: bool = True,
    ) -> LiveGameState:
        """Get live state for a football fixture.
        
        Args:
            fixture_id: API-Football fixture ID.
            use_cache: Whether to use cached data.
            
        Returns:
            LiveGameState with current game status.
        """
        if not self.football_client:
            return LiveGameState(
                fixture_id=fixture_id,
                sport="football",
                is_tradeable=False,
                stale_reason="No football API client available",
            )
        
        # Check cache
        if use_cache and self._is_cache_valid():
            if fixture_id in self._live_football_cache:
                return self._live_football_cache[fixture_id]
        
        try:
            fixture = await self.football_client.get_fixture(fixture_id)
            if fixture is None:
                return LiveGameState(
                    fixture_id=fixture_id,
                    sport="football",
                    is_tradeable=False,
                    stale_reason=f"Fixture {fixture_id} not found",
                )
            
            state = self._parse_football_fixture(fixture)
            self._live_football_cache[fixture_id] = state
            self._cache_time = datetime.now(UTC)
            return state
            
        except Exception as e:
            logger.error("football_game_state_error", fixture_id=fixture_id, error=str(e))
            return LiveGameState(
                fixture_id=fixture_id,
                sport="football",
                is_tradeable=False,
                stale_reason=f"API error: {e}",
            )
    
    async def get_nba_game_state(
        self,
        game_id: int,
        use_cache: bool = True,
    ) -> LiveGameState:
        """Get live state for an NBA game.
        
        Args:
            game_id: NBA API game ID.
            use_cache: Whether to use cached data.
            
        Returns:
            LiveGameState with current game status.
        """
        if not self.nba_client:
            return LiveGameState(
                game_id=game_id,
                sport="nba",
                is_tradeable=False,
                stale_reason="No NBA API client available",
            )
        
        # Check cache
        if use_cache and self._is_cache_valid():
            if game_id in self._live_nba_cache:
                return self._live_nba_cache[game_id]
        
        try:
            game = await self.nba_client.get_game_by_id(game_id)
            if game is None:
                return LiveGameState(
                    game_id=game_id,
                    sport="nba",
                    is_tradeable=False,
                    stale_reason=f"Game {game_id} not found",
                )
            
            state = self._parse_nba_game(game)
            self._live_nba_cache[game_id] = state
            self._cache_time = datetime.now(UTC)
            return state
            
        except Exception as e:
            logger.error("nba_game_state_error", game_id=game_id, error=str(e))
            return LiveGameState(
                game_id=game_id,
                sport="nba",
                is_tradeable=False,
                stale_reason=f"API error: {e}",
            )
    
    async def get_all_live_football_games(self) -> list[LiveGameState]:
        """Get state for all currently live football games."""
        if not self.football_client:
            return []
        
        # Note: API-Football requires getting fixtures for today and filtering
        # by status. This could be optimized with a dedicated live endpoint.
        try:
            from datetime import date
            today = date.today()
            from_dt = datetime(today.year, today.month, today.day, tzinfo=UTC)
            to_dt = from_dt + timedelta(days=1)
            
            fixtures = await self.football_client.get_fixtures(
                league_id=39,  # EPL - could be configurable
                season=2025,   # Current season
                from_date=from_dt,
                to_date=to_dt,
            )
            
            live_states = []
            for fixture in fixtures:
                if fixture.status.is_live:
                    state = self._parse_football_fixture(fixture)
                    live_states.append(state)
                    self._live_football_cache[fixture.fixture_id] = state
            
            self._cache_time = datetime.now(UTC)
            return live_states
            
        except Exception as e:
            logger.error("get_live_football_games_error", error=str(e))
            return []
    
    async def get_all_live_nba_games(self) -> list[LiveGameState]:
        """Get state for all currently live NBA games."""
        if not self.nba_client:
            return []
        
        try:
            games = await self.nba_client.get_live_games()
            
            live_states = []
            for game in games:
                state = self._parse_nba_game(game)
                live_states.append(state)
                self._live_nba_cache[game.game_id] = state
            
            self._cache_time = datetime.now(UTC)
            return live_states
            
        except Exception as e:
            logger.error("get_live_nba_games_error", error=str(e))
            return []
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        return datetime.now(UTC) - self._cache_time < self._cache_ttl
    
    def _parse_football_fixture(self, fixture: FixtureData) -> LiveGameState:
        """Parse football fixture into LiveGameState."""
        # Determine phase
        phase = self._get_football_phase(fixture)
        
        # Build timing
        timing = None
        if fixture.kickoff_utc:
            elapsed = self._estimate_football_elapsed(fixture)
            timing = GameTiming(
                kickoff_time=fixture.kickoff_utc,
                elapsed_minutes=elapsed,
                total_minutes=90,
                period=fixture.status.value,
            )
        
        # Build score
        score = None
        if fixture.home_goals is not None and fixture.away_goals is not None:
            score = LiveScore(
                home_score=fixture.home_goals,
                away_score=fixture.away_goals,
            )
        
        # Calculate probability adjustments based on live state
        adjustments = self._calculate_football_adjustments(fixture, score, timing)
        
        # Determine tradeability
        is_tradeable = True
        stale_reason = None
        if phase == GamePhase.FINISHED:
            is_tradeable = False
            stale_reason = "Game has finished"
        elif phase in {GamePhase.POSTPONED, GamePhase.CANCELLED}:
            is_tradeable = False
            stale_reason = f"Game is {phase.value}"
        
        return LiveGameState(
            fixture_id=fixture.fixture_id,
            sport="football",
            phase=phase,
            score=score,
            timing=timing,
            home_team=fixture.home_team_name,
            away_team=fixture.away_team_name,
            is_tradeable=is_tradeable,
            stale_reason=stale_reason,
            home_win_adjustment=adjustments.get("home", 0.0),
            draw_adjustment=adjustments.get("draw", 0.0),
            away_win_adjustment=adjustments.get("away", 0.0),
            raw_data=fixture.raw_data,
        )
    
    def _parse_nba_game(self, game: NBAGame) -> LiveGameState:
        """Parse NBA game into LiveGameState."""
        # Determine phase
        phase = self._get_nba_phase(game)
        
        # Build timing
        timing = GameTiming(
            kickoff_time=game.date,
            elapsed_minutes=self._estimate_nba_elapsed(game),
            total_minutes=48,  # NBA regulation time
        )
        
        # Build score
        score = None
        if game.home_score is not None and game.away_score is not None:
            score = LiveScore(
                home_score=game.home_score,
                away_score=game.away_score,
            )
        
        # Calculate probability adjustments
        adjustments = self._calculate_nba_adjustments(game, score, timing)
        
        # Determine tradeability
        is_tradeable = True
        stale_reason = None
        if phase == GamePhase.FINISHED:
            is_tradeable = False
            stale_reason = "Game has finished"
        elif phase in {GamePhase.POSTPONED, GamePhase.CANCELLED}:
            is_tradeable = False
            stale_reason = f"Game is {phase.value}"
        
        return LiveGameState(
            game_id=game.game_id,
            sport="nba",
            phase=phase,
            score=score,
            timing=timing,
            home_team=game.home_team.name,
            away_team=game.away_team.name,
            is_tradeable=is_tradeable,
            stale_reason=stale_reason,
            home_win_adjustment=adjustments.get("home", 0.0),
            away_win_adjustment=adjustments.get("away", 0.0),
            raw_data=game.raw_data,
        )
    
    def _get_football_phase(self, fixture: FixtureData) -> GamePhase:
        """Determine football game phase from fixture status."""
        status = fixture.status
        
        if status == FixtureStatus.NOT_STARTED:
            if fixture.kickoff_utc:
                mins_to_kick = (fixture.kickoff_utc - datetime.now(UTC)).total_seconds() / 60
                if mins_to_kick <= 15:
                    return GamePhase.IMMINENT
                elif mins_to_kick <= 120:
                    return GamePhase.PREGAME
            return GamePhase.SCHEDULED
        
        if status == FixtureStatus.FIRST_HALF:
            return GamePhase.EARLY
        if status == FixtureStatus.HALFTIME:
            return GamePhase.HALFTIME
        if status == FixtureStatus.SECOND_HALF:
            return GamePhase.MIDDLE  # Could be LATE based on elapsed time
        if status in {FixtureStatus.EXTRA_TIME, FixtureStatus.PENALTY}:
            return GamePhase.CLOSING
        if status.is_finished:
            return GamePhase.FINISHED
        if status == FixtureStatus.POSTPONED:
            return GamePhase.POSTPONED
        if status in {FixtureStatus.CANCELLED, FixtureStatus.ABANDONED}:
            return GamePhase.CANCELLED
        
        return GamePhase.SCHEDULED
    
    def _get_nba_phase(self, game: NBAGame) -> GamePhase:
        """Determine NBA game phase from game status."""
        status = game.status
        
        if status == NBAGameStatus.NOT_STARTED:
            mins_to_start = (game.date - datetime.now(UTC)).total_seconds() / 60
            if mins_to_start <= 15:
                return GamePhase.IMMINENT
            elif mins_to_start <= 120:
                return GamePhase.PREGAME
            return GamePhase.SCHEDULED
        
        if status == NBAGameStatus.LIVE:
            # Would need more data to determine quarter
            # For now, return MIDDLE as default live state
            return GamePhase.MIDDLE
        
        if status == NBAGameStatus.FINISHED:
            return GamePhase.FINISHED
        if status == NBAGameStatus.POSTPONED:
            return GamePhase.POSTPONED
        if status == NBAGameStatus.CANCELED:
            return GamePhase.CANCELLED
        
        return GamePhase.SCHEDULED
    
    def _estimate_football_elapsed(self, fixture: FixtureData) -> int | None:
        """Estimate elapsed minutes for football match."""
        status = fixture.status
        
        if status == FixtureStatus.FIRST_HALF:
            # Could use kickoff time to estimate, but API might provide elapsed
            return 25  # Approximate
        if status == FixtureStatus.HALFTIME:
            return 45
        if status == FixtureStatus.SECOND_HALF:
            return 67  # Approximate
        if status.is_finished:
            return 90
        
        return None
    
    def _estimate_nba_elapsed(self, game: NBAGame) -> int | None:
        """Estimate elapsed minutes for NBA game."""
        if game.status == NBAGameStatus.NOT_STARTED:
            return None
        if game.status == NBAGameStatus.FINISHED:
            return 48
        if game.status == NBAGameStatus.LIVE:
            # Would need quarter/time data for accuracy
            return 24  # Approximate mid-game
        return None
    
    def _calculate_football_adjustments(
        self,
        fixture: FixtureData,
        score: LiveScore | None,
        timing: GameTiming | None,
    ) -> dict[str, float]:
        """Calculate probability adjustments based on live football state.
        
        These adjustments are added to the pre-match model probabilities
        to account for the current game state.
        
        The logic:
        - Leading team's win probability increases as time passes
        - Trailing team's win probability decreases as time passes
        - Draw probability increases if game is tied late
        - Score differential matters more as time decreases
        """
        adjustments = {"home": 0.0, "draw": 0.0, "away": 0.0}
        
        if score is None or timing is None:
            return adjustments
        
        progress = timing.game_progress or 0.0
        goal_diff = score.score_diff
        
        # Base adjustment per goal based on time remaining
        # Early in game: each goal less impactful
        # Late in game: each goal more decisive
        time_factor = 0.5 + (0.5 * progress)  # 0.5 at start, 1.0 at end
        goal_impact = 0.08 * time_factor  # 8-16% per goal
        
        if goal_diff > 0:
            # Home winning
            adjustments["home"] = goal_diff * goal_impact
            adjustments["away"] = -goal_diff * goal_impact
            adjustments["draw"] = -abs(goal_diff) * goal_impact * 0.5
        elif goal_diff < 0:
            # Away winning
            adjustments["away"] = -goal_diff * goal_impact
            adjustments["home"] = goal_diff * goal_impact
            adjustments["draw"] = -abs(goal_diff) * goal_impact * 0.5
        else:
            # Tied - draw more likely as time passes
            if progress > 0.75:
                adjustments["draw"] = 0.10 * (progress - 0.75) * 4
                adjustments["home"] = -adjustments["draw"] / 2
                adjustments["away"] = -adjustments["draw"] / 2
        
        # Cap adjustments
        for key in adjustments:
            adjustments[key] = max(-0.40, min(0.40, adjustments[key]))
        
        return adjustments
    
    def _calculate_nba_adjustments(
        self,
        game: NBAGame,
        score: LiveScore | None,
        timing: GameTiming | None,
    ) -> dict[str, float]:
        """Calculate probability adjustments based on live NBA state.
        
        NBA has no draws, so we only adjust home/away probabilities.
        NBA games have more scoring, so point differential matters
        differently than football goals.
        """
        adjustments = {"home": 0.0, "away": 0.0}
        
        if score is None or timing is None:
            return adjustments
        
        progress = timing.game_progress or 0.0
        point_diff = score.score_diff
        
        # NBA: adjust based on lead and time remaining
        # Historical data: ~7-point home advantage, each point lead = ~2.5% win prob increase
        # But this scales with time remaining
        
        time_factor = 0.3 + (0.7 * progress)  # 0.3 at start, 1.0 at end
        
        # Each 5-point lead = ~10% probability shift, scaled by time
        lead_buckets = point_diff // 5
        base_adjustment = lead_buckets * 0.10 * time_factor
        
        adjustments["home"] = base_adjustment
        adjustments["away"] = -base_adjustment
        
        # Cap adjustments (can't exceed +/- 45%)
        for key in adjustments:
            adjustments[key] = max(-0.45, min(0.45, adjustments[key]))
        
        return adjustments
    
    def clear_cache(self) -> None:
        """Clear all cached game states."""
        self._live_football_cache.clear()
        self._live_nba_cache.clear()
        self._cache_time = None
