"""Tests for leakage-safe feature engineering."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from footbe_trader.modeling.features import (
    FEATURE_FIELDS_USED,
    FeatureBuilder,
    MatchFeatureVector,
)
from footbe_trader.storage.models import FixtureV2


def make_fixture(
    fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    kickoff_utc: datetime,
    home_goals: int | None = None,
    away_goals: int | None = None,
    status: str = "FT",
    season: int = 2023,
    round_name: str = "Regular Season - 1",
) -> FixtureV2:
    """Helper to create test fixtures."""
    return FixtureV2(
        fixture_id=fixture_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        kickoff_utc=kickoff_utc,
        home_goals=home_goals,
        away_goals=away_goals,
        status=status,
        season=season,
        round=round_name,
    )


class TestMatchFeatureVector:
    """Tests for MatchFeatureVector dataclass."""
    
    def test_to_feature_array_length(self):
        """Feature array should have correct length."""
        fv = MatchFeatureVector(
            fixture_id=1,
            kickoff_utc=datetime.now(),
            home_team_id=100,
            away_team_id=200,
            season=2023,
            round_str="Regular Season - 1",
            home_advantage=1.0,
            home_team_home_goals_scored_avg=1.5,
            home_team_home_goals_conceded_avg=1.0,
            home_team_away_goals_scored_avg=1.2,
            home_team_away_goals_conceded_avg=1.3,
            away_team_home_goals_scored_avg=1.8,
            away_team_home_goals_conceded_avg=0.8,
            away_team_away_goals_scored_avg=1.0,
            away_team_away_goals_conceded_avg=1.5,
            rest_days_diff=2.0,
        )
        
        arr = fv.to_feature_array()
        
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 10  # 10 features
    
    def test_to_feature_array_order(self):
        """Feature array should have correct order."""
        fv = MatchFeatureVector(
            fixture_id=1,
            kickoff_utc=datetime.now(),
            home_team_id=100,
            away_team_id=200,
            season=2023,
            round_str="Regular Season - 1",
            home_advantage=1.0,
            home_team_home_goals_scored_avg=2.0,
            home_team_home_goals_conceded_avg=3.0,
            home_team_away_goals_scored_avg=4.0,
            home_team_away_goals_conceded_avg=5.0,
            away_team_home_goals_scored_avg=6.0,
            away_team_home_goals_conceded_avg=7.0,
            away_team_away_goals_scored_avg=8.0,
            away_team_away_goals_conceded_avg=9.0,
            rest_days_diff=10.0,
        )
        
        arr = fv.to_feature_array()
        
        assert arr[0] == 1.0   # home_advantage
        assert arr[1] == 2.0   # home_team_home_goals_scored_avg
        assert arr[2] == 3.0   # home_team_home_goals_conceded_avg
        assert arr[3] == 4.0   # home_team_away_goals_scored_avg
        assert arr[4] == 5.0   # home_team_away_goals_conceded_avg
        assert arr[5] == 6.0   # away_team_home_goals_scored_avg
        assert arr[6] == 7.0   # away_team_home_goals_conceded_avg
        assert arr[7] == 8.0   # away_team_away_goals_scored_avg
        assert arr[8] == 9.0   # away_team_away_goals_conceded_avg
        assert arr[9] == 10.0  # rest_days_diff


class TestFeatureBuilder:
    """Tests for FeatureBuilder with strict time integrity."""
    
    @pytest.fixture
    def base_time(self) -> datetime:
        """Base time for tests."""
        return datetime(2023, 9, 1, 15, 0, 0)
    
    @pytest.fixture
    def historical_fixtures(self, base_time: datetime) -> list[FixtureV2]:
        """Create historical fixtures for home team 100 and away team 200."""
        fixtures = []
        
        # Home team (100) historical matches - 5 home, 5 away
        for i in range(5):
            # Home games for team 100
            fixtures.append(make_fixture(
                fixture_id=100 + i,
                home_team_id=100,
                away_team_id=300 + i,
                kickoff_utc=base_time - timedelta(days=30 - i * 3),
                home_goals=2,  # Consistent scoring
                away_goals=1,
                status="FT",
            ))
            # Away games for team 100
            fixtures.append(make_fixture(
                fixture_id=200 + i,
                home_team_id=400 + i,
                away_team_id=100,
                kickoff_utc=base_time - timedelta(days=28 - i * 3),
                home_goals=1,
                away_goals=1,  # Draws away
                status="FT",
            ))
        
        # Away team (200) historical matches - 5 home, 5 away
        for i in range(5):
            # Home games for team 200
            fixtures.append(make_fixture(
                fixture_id=300 + i,
                home_team_id=200,
                away_team_id=500 + i,
                kickoff_utc=base_time - timedelta(days=29 - i * 3),
                home_goals=1,
                away_goals=2,  # Losing at home
                status="FT",
            ))
            # Away games for team 200
            fixtures.append(make_fixture(
                fixture_id=400 + i,
                home_team_id=600 + i,
                away_team_id=200,
                kickoff_utc=base_time - timedelta(days=27 - i * 3),
                home_goals=0,
                away_goals=2,  # Strong away
                status="FT",
            ))
        
        return fixtures
    
    def test_time_integrity_no_future_data(
        self, base_time: datetime, historical_fixtures: list[FixtureV2]
    ):
        """Features must only use data from before target kickoff."""
        builder = FeatureBuilder(rolling_window=10)
        
        # Target fixture at base_time
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=200,
            kickoff_utc=base_time,
            status="NS",  # Not started
        )
        
        # Add some "future" fixtures that should NOT be used
        future_fixtures = [
            make_fixture(
                fixture_id=998,
                home_team_id=100,
                away_team_id=700,
                kickoff_utc=base_time + timedelta(days=1),  # AFTER target
                home_goals=5,  # Very high score
                away_goals=0,
                status="FT",
            ),
        ]
        
        all_fixtures = historical_fixtures + future_fixtures
        
        fv = builder.build_features(all_fixtures, target)
        
        assert fv is not None
        # If future data was used, home scoring avg would be higher
        # Team 100 home games: 5 games with 2 goals each = 2.0 avg
        assert fv.home_team_home_goals_scored_avg == 2.0  # No future 5-0 game
    
    def test_excludes_target_fixture_itself(
        self, base_time: datetime, historical_fixtures: list[FixtureV2]
    ):
        """Target fixture should never be used in its own features."""
        builder = FeatureBuilder(rolling_window=10)
        
        # Make target fixture completed (edge case)
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=200,
            kickoff_utc=base_time,
            home_goals=10,  # Extreme score
            away_goals=0,
            status="FT",
        )
        
        all_fixtures = historical_fixtures + [target]
        
        fv = builder.build_features(all_fixtures, target)
        
        assert fv is not None
        # The 10-0 from target should not be included
        assert fv.home_team_home_goals_scored_avg == 2.0  # Still original avg
    
    def test_no_history_uses_defaults(self, base_time: datetime):
        """No historical data should use default values."""
        builder = FeatureBuilder(rolling_window=10, default_goals_avg=1.3)
        
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=200,
            kickoff_utc=base_time,
            status="NS",
        )
        
        # No historical fixtures
        fv = builder.build_features([], target)
        
        assert fv is not None
        assert fv.home_team_home_goals_scored_avg == 1.3  # Default
    
    def test_home_advantage_constant(
        self, base_time: datetime, historical_fixtures: list[FixtureV2]
    ):
        """Home advantage should always be 1.0."""
        builder = FeatureBuilder(rolling_window=10)
        
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=200,
            kickoff_utc=base_time,
            status="NS",
        )
        
        fv = builder.build_features(historical_fixtures, target)
        
        assert fv is not None
        assert fv.home_advantage == 1.0
    
    def test_rest_days_computation(self, base_time: datetime):
        """Rest days should be computed correctly."""
        builder = FeatureBuilder(rolling_window=10)
        
        # Home team played 3 days ago, away team 5 days ago
        fixtures = [
            make_fixture(
                fixture_id=1,
                home_team_id=100,
                away_team_id=300,
                kickoff_utc=base_time - timedelta(days=3),
                home_goals=1,
                away_goals=1,
                status="FT",
            ),
            make_fixture(
                fixture_id=2,
                home_team_id=400,
                away_team_id=200,
                kickoff_utc=base_time - timedelta(days=5),
                home_goals=1,
                away_goals=1,
                status="FT",
            ),
        ]
        
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=200,
            kickoff_utc=base_time,
            status="NS",
        )
        
        fv = builder.build_features(fixtures, target)
        
        assert fv is not None
        # Home has 3 rest days, away has 5 rest days
        # Diff = home - away = 3 - 5 = -2
        assert fv.rest_days_diff == -2.0
    
    def test_rolling_window_respected(self, base_time: datetime):
        """Only most recent N games should be used."""
        builder = FeatureBuilder(rolling_window=3)
        
        # Create 5 games, but only last 3 should be used
        fixtures = []
        for i in range(5):
            fixtures.append(make_fixture(
                fixture_id=i,
                home_team_id=100,
                away_team_id=200 + i,
                kickoff_utc=base_time - timedelta(days=20 - i * 3),
                home_goals=i + 1,  # 1, 2, 3, 4, 5 goals
                away_goals=0,
                status="FT",
            ))
        
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=300,
            kickoff_utc=base_time,
            status="NS",
        )
        
        fv = builder.build_features(fixtures, target)
        
        assert fv is not None
        # Last 3 home games: 3, 4, 5 goals -> avg = 4.0
        assert fv.home_team_home_goals_scored_avg == 4.0
    
    def test_build_features_batch(
        self, base_time: datetime, historical_fixtures: list[FixtureV2]
    ):
        """Batch feature building should work."""
        builder = FeatureBuilder(rolling_window=10)
        
        targets = [
            make_fixture(
                fixture_id=999,
                home_team_id=100,
                away_team_id=200,
                kickoff_utc=base_time,
                status="NS",
            ),
            make_fixture(
                fixture_id=998,
                home_team_id=200,
                away_team_id=100,  # Reversed
                kickoff_utc=base_time + timedelta(days=1),
                status="NS",
            ),
        ]
        
        features = builder.build_features_batch(historical_fixtures, targets)
        
        assert len(features) == 2
        assert features[0].home_team_id == 100
        assert features[1].home_team_id == 200
    
    def test_only_completed_fixtures_used(self, base_time: datetime):
        """Only FT fixtures should be used for features."""
        builder = FeatureBuilder(rolling_window=10)
        
        fixtures = [
            make_fixture(
                fixture_id=1,
                home_team_id=100,
                away_team_id=300,
                kickoff_utc=base_time - timedelta(days=7),
                home_goals=2,
                away_goals=1,
                status="FT",  # Completed
            ),
            make_fixture(
                fixture_id=2,
                home_team_id=100,
                away_team_id=400,
                kickoff_utc=base_time - timedelta(days=3),
                home_goals=None,  # Postponed matches have no goals
                away_goals=None,
                status="PST",  # Postponed - should be excluded
            ),
            # Away team history
            make_fixture(
                fixture_id=3,
                home_team_id=200,
                away_team_id=500,
                kickoff_utc=base_time - timedelta(days=5),
                home_goals=1,
                away_goals=1,
                status="FT",
            ),
        ]
        
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=200,
            kickoff_utc=base_time,
            status="NS",
        )
        
        fv = builder.build_features(fixtures, target)
        
        assert fv is not None
        # Only the 2-1 game should be used, not the postponed one
        assert fv.home_team_home_goals_scored_avg == 2.0


class TestFeatureFieldsValidation:
    """Tests for feature field validation against catalog."""
    
    def test_feature_fields_are_pre_match_safe(self):
        """All used fields should be in catalog as safe."""
        # This test verifies the module-level validation ran successfully
        # If FEATURE_FIELDS_USED contained POST_MATCH_ONLY fields,
        # the module would have failed to import
        assert len(FEATURE_FIELDS_USED) > 0
        
        # The fields declared in the feature module
        expected_fields = [
            "fixture.date",
            "teams.home.id",
            "teams.away.id",
        ]
        
        for field in expected_fields:
            assert field in FEATURE_FIELDS_USED


class TestLeakagePreventionIntegration:
    """Integration tests for data leakage prevention."""
    
    def test_chronological_order_respected(self):
        """Features should respect chronological order."""
        base_time = datetime(2023, 9, 1, 15, 0, 0)
        builder = FeatureBuilder(rolling_window=5)
        
        # Create chronological fixtures
        fixtures = []
        for i in range(10):
            fixtures.append(make_fixture(
                fixture_id=i,
                home_team_id=100,
                away_team_id=200 + i,
                kickoff_utc=base_time + timedelta(days=i),
                home_goals=i,  # Increasing goals over time
                away_goals=0,
                status="FT",
            ))
        
        # Target at day 5
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=300,
            kickoff_utc=base_time + timedelta(days=5),
            status="NS",
        )
        
        # Add away team history
        for i in range(3):
            fixtures.append(make_fixture(
                fixture_id=100 + i,
                home_team_id=300,
                away_team_id=400 + i,
                kickoff_utc=base_time + timedelta(days=i),
                home_goals=1,
                away_goals=1,
                status="FT",
            ))
        
        fv = builder.build_features(fixtures, target)
        
        assert fv is not None
        # Should only use days 0-4 (fixtures 0-4), not days 5-9
        # Days 0-4 have goals 0, 1, 2, 3, 4 -> avg = 2.0
        assert fv.home_team_home_goals_scored_avg == 2.0
    
    def test_no_same_day_leakage(self):
        """Fixtures on same day with earlier kickoff should be included."""
        base_time = datetime(2023, 9, 1, 15, 0, 0)
        builder = FeatureBuilder(rolling_window=5)
        
        fixtures = [
            # Earlier game same day (12:00)
            make_fixture(
                fixture_id=1,
                home_team_id=100,
                away_team_id=300,
                kickoff_utc=base_time.replace(hour=12),  # 3 hours before target
                home_goals=5,
                away_goals=0,
                status="FT",
            ),
            # Previous day game
            make_fixture(
                fixture_id=2,
                home_team_id=100,
                away_team_id=400,
                kickoff_utc=base_time - timedelta(days=1),
                home_goals=1,
                away_goals=0,
                status="FT",
            ),
            # Away team history
            make_fixture(
                fixture_id=3,
                home_team_id=200,
                away_team_id=500,
                kickoff_utc=base_time - timedelta(days=2),
                home_goals=1,
                away_goals=1,
                status="FT",
            ),
        ]
        
        target = make_fixture(
            fixture_id=999,
            home_team_id=100,
            away_team_id=200,
            kickoff_utc=base_time,  # 15:00
            status="NS",
        )
        
        fv = builder.build_features(fixtures, target)
        
        assert fv is not None
        # The 5-0 game at 12:00 same day should be included (kickoff 12:00 < 15:00)
        # With the 5-0 game included: (5 + 1) / 2 = 3.0
        assert fv.home_team_home_goals_scored_avg == 3.0
