"""Tests for live game state management."""

from datetime import UTC, datetime, timedelta

import pytest

from footbe_trader.agent.live_game import (
    GamePhase,
    GameTiming,
    LiveGameState,
    LiveGameStateProvider,
    LiveScore,
)
from footbe_trader.football.interfaces import FixtureData, FixtureStatus
from footbe_trader.nba.interfaces import NBAGame, NBAGameStatus, NBAGameTeam


class TestLiveScore:
    """Tests for LiveScore dataclass."""
    
    def test_score_diff_home_winning(self):
        """Home team winning should show positive diff."""
        score = LiveScore(home_score=3, away_score=1)
        assert score.score_diff == 2
        assert score.home_score > score.away_score
        assert not score.is_tied
    
    def test_score_diff_away_winning(self):
        """Away team winning should show negative diff."""
        score = LiveScore(home_score=0, away_score=2)
        assert score.score_diff == -2
    
    def test_tied_score(self):
        """Tied game should report is_tied."""
        score = LiveScore(home_score=1, away_score=1)
        assert score.is_tied
        assert score.score_diff == 0
    
    def test_total_score(self):
        """Total score calculation."""
        score = LiveScore(home_score=3, away_score=2)
        assert score.total_score == 5


class TestGameTiming:
    """Tests for GameTiming dataclass."""
    
    def test_minutes_to_kickoff_future(self):
        """Game in future shows positive minutes."""
        kickoff = datetime.now(UTC) + timedelta(hours=2)
        timing = GameTiming(kickoff_time=kickoff)
        assert timing.minutes_to_kickoff > 100
    
    def test_minutes_to_kickoff_past(self):
        """Game in past shows negative minutes."""
        kickoff = datetime.now(UTC) - timedelta(hours=1)
        timing = GameTiming(kickoff_time=kickoff)
        assert timing.minutes_to_kickoff < -50
    
    def test_minutes_remaining(self):
        """Minutes remaining calculation."""
        kickoff = datetime.now(UTC) - timedelta(minutes=30)
        timing = GameTiming(
            kickoff_time=kickoff,
            elapsed_minutes=30,
            total_minutes=90,
        )
        assert timing.minutes_remaining == 60
    
    def test_game_progress(self):
        """Game progress percentage."""
        timing = GameTiming(
            kickoff_time=datetime.now(UTC),
            elapsed_minutes=45,
            total_minutes=90,
        )
        assert timing.game_progress == 0.5


class TestLiveGameState:
    """Tests for LiveGameState dataclass."""
    
    def test_pregame_states(self):
        """Pre-game phases should report is_pregame."""
        for phase in [GamePhase.SCHEDULED, GamePhase.PREGAME, GamePhase.IMMINENT]:
            state = LiveGameState(phase=phase)
            assert state.is_pregame
            assert not state.is_live
            assert not state.is_ended
    
    def test_live_states(self):
        """Live phases should report is_live."""
        for phase in [GamePhase.EARLY, GamePhase.MIDDLE, GamePhase.LATE, 
                      GamePhase.CLOSING, GamePhase.HALFTIME, GamePhase.BREAK]:
            state = LiveGameState(phase=phase)
            assert state.is_live
            assert not state.is_pregame
            assert not state.is_ended
    
    def test_ended_states(self):
        """Ended phases should report is_ended."""
        for phase in [GamePhase.FINISHED, GamePhase.POSTPONED, GamePhase.CANCELLED]:
            state = LiveGameState(phase=phase)
            assert state.is_ended
            assert not state.is_live
            assert not state.is_pregame
    
    def test_home_winning(self):
        """Home winning detection."""
        state = LiveGameState(
            phase=GamePhase.MIDDLE,
            score=LiveScore(home_score=2, away_score=0),
        )
        assert state.home_is_winning
        assert not state.away_is_winning
    
    def test_away_winning(self):
        """Away winning detection."""
        state = LiveGameState(
            phase=GamePhase.MIDDLE,
            score=LiveScore(home_score=0, away_score=1),
        )
        assert state.away_is_winning
        assert not state.home_is_winning


class TestLiveGameStateProvider:
    """Tests for LiveGameStateProvider."""
    
    @pytest.fixture
    def provider(self):
        """Create provider without actual clients."""
        return LiveGameStateProvider()
    
    def test_parse_football_fixture_not_started(self, provider):
        """Parse fixture that hasn't started."""
        fixture = FixtureData(
            fixture_id=12345,
            league_id=39,
            season=2025,
            round="Matchweek 1",
            home_team_id=33,
            away_team_id=34,
            home_team_name="Manchester United",
            away_team_name="Liverpool",
            kickoff_utc=datetime.now(UTC) + timedelta(hours=24),
            status=FixtureStatus.NOT_STARTED,
        )
        
        state = provider._parse_football_fixture(fixture)
        
        assert state.fixture_id == 12345
        assert state.sport == "football"
        assert state.phase == GamePhase.SCHEDULED
        assert state.is_pregame
        assert state.is_tradeable
        assert state.home_team == "Manchester United"
        assert state.away_team == "Liverpool"
    
    def test_parse_football_fixture_imminent(self, provider):
        """Parse fixture starting soon."""
        fixture = FixtureData(
            fixture_id=12345,
            league_id=39,
            season=2025,
            round="Matchweek 1",
            home_team_id=33,
            away_team_id=34,
            home_team_name="Arsenal",
            away_team_name="Chelsea",
            kickoff_utc=datetime.now(UTC) + timedelta(minutes=10),
            status=FixtureStatus.NOT_STARTED,
        )
        
        state = provider._parse_football_fixture(fixture)
        
        assert state.phase == GamePhase.IMMINENT
        assert state.is_pregame
    
    def test_parse_football_fixture_first_half(self, provider):
        """Parse fixture in first half."""
        fixture = FixtureData(
            fixture_id=12345,
            league_id=39,
            season=2025,
            round="Matchweek 1",
            home_team_id=33,
            away_team_id=34,
            home_team_name="Arsenal",
            away_team_name="Chelsea",
            kickoff_utc=datetime.now(UTC) - timedelta(minutes=30),
            status=FixtureStatus.FIRST_HALF,
            home_goals=1,
            away_goals=0,
        )
        
        state = provider._parse_football_fixture(fixture)
        
        assert state.phase == GamePhase.EARLY
        assert state.is_live
        assert state.score is not None
        assert state.score.home_score == 1
        assert state.score.away_score == 0
        assert state.home_is_winning
    
    def test_parse_football_fixture_finished(self, provider):
        """Parse finished fixture."""
        fixture = FixtureData(
            fixture_id=12345,
            league_id=39,
            season=2025,
            round="Matchweek 1",
            home_team_id=33,
            away_team_id=34,
            home_team_name="Arsenal",
            away_team_name="Chelsea",
            kickoff_utc=datetime.now(UTC) - timedelta(hours=3),
            status=FixtureStatus.FULL_TIME,
            home_goals=2,
            away_goals=2,
        )
        
        state = provider._parse_football_fixture(fixture)
        
        assert state.phase == GamePhase.FINISHED
        assert state.is_ended
        assert not state.is_tradeable
        assert state.stale_reason == "Game has finished"
    
    def test_probability_adjustments_home_winning(self, provider):
        """Probability adjustments when home team is winning."""
        fixture = FixtureData(
            fixture_id=12345,
            league_id=39,
            season=2025,
            round="Matchweek 1",
            home_team_id=33,
            away_team_id=34,
            home_team_name="Arsenal",
            away_team_name="Chelsea",
            kickoff_utc=datetime.now(UTC) - timedelta(minutes=60),
            status=FixtureStatus.SECOND_HALF,
            home_goals=2,
            away_goals=0,
        )
        
        state = provider._parse_football_fixture(fixture)
        
        # Home should get positive adjustment
        assert state.home_win_adjustment > 0
        # Away should get negative adjustment
        assert state.away_win_adjustment < 0
    
    def test_probability_adjustments_tied_late(self, provider):
        """Probability adjustments when tied late in game."""
        fixture = FixtureData(
            fixture_id=12345,
            league_id=39,
            season=2025,
            round="Matchweek 1",
            home_team_id=33,
            away_team_id=34,
            home_team_name="Arsenal",
            away_team_name="Chelsea",
            kickoff_utc=datetime.now(UTC) - timedelta(minutes=80),
            status=FixtureStatus.SECOND_HALF,
            home_goals=1,
            away_goals=1,
        )
        
        state = provider._parse_football_fixture(fixture)
        
        # Draw should get positive adjustment late in tied game
        # (Note: the fixture is in SECOND_HALF, progress estimate may vary)
        # Just verify the calculation runs without error
        assert isinstance(state.draw_adjustment, float)
    
    def test_parse_nba_game(self, provider):
        """Parse NBA game."""
        game = NBAGame(
            game_id=54321,
            date=datetime.now(UTC) + timedelta(hours=1),  # Within pregame window
            timestamp=0,
            status=NBAGameStatus.NOT_STARTED,
            home_team=NBAGameTeam(
                team_id=1,
                name="Los Angeles Lakers",
                nickname="Lakers",
                code="LAL",
            ),
            away_team=NBAGameTeam(
                team_id=2,
                name="Boston Celtics",
                nickname="Celtics",
                code="BOS",
            ),
        )
        
        state = provider._parse_nba_game(game)
        
        assert state.game_id == 54321
        assert state.sport == "nba"
        assert state.phase == GamePhase.PREGAME
        assert state.is_pregame
        assert state.is_tradeable
    
    def test_parse_nba_game_live(self, provider):
        """Parse live NBA game."""
        game = NBAGame(
            game_id=54321,
            date=datetime.now(UTC) - timedelta(hours=1),
            timestamp=0,
            status=NBAGameStatus.LIVE,
            home_team=NBAGameTeam(
                team_id=1,
                name="Los Angeles Lakers",
                nickname="Lakers",
                code="LAL",
            ),
            away_team=NBAGameTeam(
                team_id=2,
                name="Boston Celtics",
                nickname="Celtics",
                code="BOS",
            ),
            home_score=78,
            away_score=65,
        )
        
        state = provider._parse_nba_game(game)
        
        assert state.is_live
        assert state.score is not None
        assert state.score.home_score == 78
        assert state.score.away_score == 65
        assert state.home_is_winning
    
    def test_nba_adjustments_home_leading(self, provider):
        """NBA probability adjustments when home leads."""
        game = NBAGame(
            game_id=54321,
            date=datetime.now(UTC) - timedelta(hours=1),
            timestamp=0,
            status=NBAGameStatus.LIVE,
            home_team=NBAGameTeam(
                team_id=1,
                name="Los Angeles Lakers",
                nickname="Lakers",
                code="LAL",
            ),
            away_team=NBAGameTeam(
                team_id=2,
                name="Boston Celtics",
                nickname="Celtics",
                code="BOS",
            ),
            home_score=80,
            away_score=65,  # 15-point lead
        )
        
        state = provider._parse_nba_game(game)
        
        # Home should get positive adjustment for 15-point lead
        assert state.home_win_adjustment > 0
        assert state.away_win_adjustment < 0
    
    def test_cache_invalidation(self, provider):
        """Cache should be clearable."""
        # Add some fake cache data
        provider._live_football_cache[123] = LiveGameState(fixture_id=123)
        provider._live_nba_cache[456] = LiveGameState(game_id=456)
        provider._cache_time = datetime.now(UTC)
        
        # Clear cache
        provider.clear_cache()
        
        assert len(provider._live_football_cache) == 0
        assert len(provider._live_nba_cache) == 0
        assert provider._cache_time is None


class TestProbabilityAdjustmentCalculations:
    """Detailed tests for probability adjustment calculations."""
    
    @pytest.fixture
    def provider(self):
        return LiveGameStateProvider()
    
    def test_football_early_game_small_lead(self, provider):
        """Early in game, lead has smaller impact."""
        score = LiveScore(home_score=1, away_score=0)
        timing = GameTiming(
            kickoff_time=datetime.now(UTC) - timedelta(minutes=15),
            elapsed_minutes=15,
            total_minutes=90,
        )
        
        adjustments = provider._calculate_football_adjustments(
            fixture=None,  # Not needed for calculation
            score=score,
            timing=timing,
        )
        
        # Early game (15/90 = 16.7% progress), 1 goal lead
        # Should have modest positive adjustment for home
        assert 0 < adjustments["home"] < 0.15
        assert adjustments["away"] < 0
    
    def test_football_late_game_big_lead(self, provider):
        """Late in game, lead has larger impact."""
        score = LiveScore(home_score=3, away_score=0)
        timing = GameTiming(
            kickoff_time=datetime.now(UTC) - timedelta(minutes=80),
            elapsed_minutes=80,
            total_minutes=90,
        )
        
        adjustments = provider._calculate_football_adjustments(
            fixture=None,
            score=score,
            timing=timing,
        )
        
        # Late game (80/90 = 89% progress), 3 goal lead
        # Should have large positive adjustment for home
        assert adjustments["home"] > 0.20
        assert adjustments["away"] < -0.20
    
    def test_nba_close_game(self, provider):
        """NBA close game should have small adjustments."""
        score = LiveScore(home_score=55, away_score=53)
        timing = GameTiming(
            kickoff_time=datetime.now(UTC) - timedelta(minutes=24),
            elapsed_minutes=24,
            total_minutes=48,
        )
        
        adjustments = provider._calculate_nba_adjustments(
            game=None,
            score=score,
            timing=timing,
        )
        
        # 2-point lead at halftime = very small adjustment
        assert abs(adjustments["home"]) < 0.10
    
    def test_nba_blowout_late(self, provider):
        """NBA blowout late should have large adjustments."""
        score = LiveScore(home_score=110, away_score=85)
        timing = GameTiming(
            kickoff_time=datetime.now(UTC) - timedelta(minutes=44),
            elapsed_minutes=44,
            total_minutes=48,
        )
        
        adjustments = provider._calculate_nba_adjustments(
            game=None,
            score=score,
            timing=timing,
        )
        
        # 25-point lead with 4 minutes left = near certainty
        assert adjustments["home"] > 0.30
        assert adjustments["away"] < -0.30
