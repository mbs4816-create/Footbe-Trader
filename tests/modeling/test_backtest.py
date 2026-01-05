"""Tests for backtest framework."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from footbe_trader.modeling.backtest import (
    BacktestResult,
    Backtester,
    FoldResult,
    generate_rolling_folds,
    generate_season_folds,
    get_completed_fixtures,
    get_outcome,
    run_backtest,
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


class TestGetOutcome:
    """Tests for outcome extraction."""
    
    def test_home_win(self):
        """Home win when home_goals > away_goals."""
        f = make_fixture(1, 100, 200, datetime.now(), 2, 1, "FT")
        assert get_outcome(f) == "H"
    
    def test_away_win(self):
        """Away win when away_goals > home_goals."""
        f = make_fixture(1, 100, 200, datetime.now(), 0, 1, "FT")
        assert get_outcome(f) == "A"
    
    def test_draw(self):
        """Draw when home_goals == away_goals."""
        f = make_fixture(1, 100, 200, datetime.now(), 1, 1, "FT")
        assert get_outcome(f) == "D"
    
    def test_not_finished_returns_none(self):
        """Non-FT status returns None."""
        f = make_fixture(1, 100, 200, datetime.now(), None, None, "NS")
        assert get_outcome(f) is None
    
    def test_missing_goals_returns_none(self):
        """Missing goals returns None."""
        f = make_fixture(1, 100, 200, datetime.now(), None, None, "FT")
        assert get_outcome(f) is None


class TestGetCompletedFixtures:
    """Tests for fixture filtering."""
    
    def test_filters_to_completed(self):
        """Should only return completed fixtures."""
        fixtures = [
            make_fixture(1, 100, 200, datetime.now(), 2, 1, "FT"),
            make_fixture(2, 100, 200, datetime.now(), None, None, "NS"),
            make_fixture(3, 100, 200, datetime.now(), 1, 1, "FT"),
            make_fixture(4, 100, 200, datetime.now(), 0, 0, "PST"),
        ]
        
        completed = get_completed_fixtures(fixtures)
        
        assert len(completed) == 2
        assert completed[0].fixture_id == 1
        assert completed[1].fixture_id == 3


class TestGenerateSeasonFolds:
    """Tests for season-by-season fold generation."""
    
    def test_generates_folds(self):
        """Should generate folds for each test season."""
        base = datetime(2020, 8, 1)
        fixtures = []
        
        # 3 seasons, 10 games each
        for season in [2020, 2021, 2022]:
            for i in range(10):
                fixtures.append(make_fixture(
                    fixture_id=season * 100 + i,
                    home_team_id=100,
                    away_team_id=200,
                    kickoff_utc=base + timedelta(days=(season - 2020) * 365 + i * 7),
                    home_goals=1,
                    away_goals=1,
                    status="FT",
                    season=season,
                ))
        
        folds = list(generate_season_folds(fixtures, min_train_seasons=1))
        
        assert len(folds) == 2  # Test on 2021 and 2022
        
        train1, test1, name1 = folds[0]
        assert "2021" in name1
        assert all(f.season == 2020 for f in train1)
        assert all(f.season == 2021 for f in test1)
        
        train2, test2, name2 = folds[1]
        assert "2022" in name2
        assert all(f.season in [2020, 2021] for f in train2)
        assert all(f.season == 2022 for f in test2)
    
    def test_not_enough_seasons(self):
        """Should return empty if not enough seasons."""
        fixtures = [
            make_fixture(1, 100, 200, datetime.now(), 1, 1, "FT", season=2023)
            for _ in range(10)
        ]
        
        folds = list(generate_season_folds(fixtures, min_train_seasons=1))
        
        assert len(folds) == 0


class TestGenerateRollingFolds:
    """Tests for rolling matchweek fold generation."""
    
    def test_generates_folds(self):
        """Should generate rolling window folds."""
        base = datetime(2023, 8, 1)
        fixtures = []
        
        # 200 games over ~50 weeks
        for i in range(200):
            fixtures.append(make_fixture(
                fixture_id=i,
                home_team_id=100 + (i % 10),
                away_team_id=200 + (i % 10),
                kickoff_utc=base + timedelta(days=i * 2),
                home_goals=1,
                away_goals=0,
                status="FT",
            ))
        
        folds = list(generate_rolling_folds(
            fixtures, min_train_size=50, test_window_days=7
        ))
        
        assert len(folds) > 0
        
        # Check first fold
        train, test, name = folds[0]
        assert len(train) >= 50
        assert len(test) > 0
        
        # Train should be before test
        max_train_time = max(f.kickoff_utc for f in train)
        min_test_time = min(f.kickoff_utc for f in test)
        assert max_train_time < min_test_time
    
    def test_not_enough_data(self):
        """Should return empty if not enough data."""
        fixtures = [
            make_fixture(i, 100, 200, datetime.now(), 1, 1, "FT")
            for i in range(10)
        ]
        
        folds = list(generate_rolling_folds(fixtures, min_train_size=100))
        
        assert len(folds) == 0


class TestFoldResult:
    """Tests for FoldResult dataclass."""
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        from footbe_trader.modeling.metrics import MetricsResult
        
        result = FoldResult(
            fold_name="test_fold",
            train_size=100,
            test_size=20,
            model_name="test_model",
            metrics=MetricsResult(
                log_loss=1.0,
                brier_score=0.5,
                accuracy=0.6,
                n_samples=20,
            ),
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["fold_name"] == "test_fold"
        assert d["train_size"] == 100
        assert "metrics" in d


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        result = BacktestResult(
            mode="season",
            model_name="test_model",
            folds=[],
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["mode"] == "season"
        assert d["model_name"] == "test_model"


class TestBacktester:
    """Tests for Backtester class."""
    
    @pytest.fixture
    def sample_fixtures(self) -> list[FixtureV2]:
        """Create sample fixtures for testing."""
        base = datetime(2020, 8, 1)
        fixtures = []
        
        # Create fixtures for 3 seasons with multiple teams
        teams = list(range(100, 120))  # 20 teams
        fixture_id = 0
        
        for season in [2020, 2021, 2022]:
            season_start = base + timedelta(days=(season - 2020) * 365)
            
            for week in range(38):  # 38 matchweeks
                for match in range(10):  # 10 matches per week
                    home_idx = (week * 10 + match * 2) % len(teams)
                    away_idx = (week * 10 + match * 2 + 1) % len(teams)
                    
                    # Simulate scores
                    home_goals = np.random.poisson(1.5)
                    away_goals = np.random.poisson(1.2)
                    
                    fixtures.append(make_fixture(
                        fixture_id=fixture_id,
                        home_team_id=teams[home_idx],
                        away_team_id=teams[away_idx],
                        kickoff_utc=season_start + timedelta(days=week * 7 + match % 3),
                        home_goals=int(home_goals),
                        away_goals=int(away_goals),
                        status="FT",
                        season=season,
                        round_name=f"Regular Season - {week + 1}",
                    ))
                    fixture_id += 1
        
        return fixtures
    
    def test_backtester_runs(self, sample_fixtures: list[FixtureV2]):
        """Backtester should run without errors."""
        backtester = Backtester(
            model_name="home_advantage",  # Simple model for speed
            rolling_window=5,
        )
        
        result = backtester.run_season_backtest(
            sample_fixtures, min_train_seasons=1
        )
        
        assert isinstance(result, BacktestResult)
        assert result.mode == "season"
        assert len(result.folds) > 0
    
    def test_time_integrity_in_folds(self, sample_fixtures: list[FixtureV2]):
        """Training data should always be before test data."""
        backtester = Backtester(rolling_window=5)
        
        for train, test, _ in generate_season_folds(sample_fixtures, min_train_seasons=1):
            max_train_time = max(f.kickoff_utc for f in train)
            min_test_time = min(f.kickoff_utc for f in test)
            
            assert max_train_time < min_test_time, "Training data must be before test"
    
    def test_bootstrap_comparison_computed(self, sample_fixtures: list[FixtureV2]):
        """Should compute bootstrap comparison."""
        backtester = Backtester(
            model_name="home_advantage",
            baseline_model_name="home_advantage",  # Same model
            rolling_window=5,
        )
        
        result = backtester.run_season_backtest(sample_fixtures)
        
        if len(result.folds) > 0:
            assert result.bootstrap_comparison is not None
            # Same model should have ~0 difference
            assert abs(result.bootstrap_comparison.mean_diff) < 0.1


class TestRunBacktest:
    """Tests for run_backtest convenience function."""
    
    def test_run_season_mode(self):
        """Should run season mode backtest."""
        base = datetime(2020, 8, 1)
        fixtures = []
        
        for season in [2020, 2021]:
            for i in range(50):
                fixtures.append(make_fixture(
                    fixture_id=season * 100 + i,
                    home_team_id=100 + (i % 5),
                    away_team_id=105 + (i % 5),
                    kickoff_utc=base + timedelta(days=(season - 2020) * 365 + i * 3),
                    home_goals=1,
                    away_goals=0,
                    status="FT",
                    season=season,
                ))
        
        result = run_backtest(
            fixtures,
            mode="season",
            model_name="home_advantage",
            rolling_window=5,
        )
        
        assert result.mode == "season"
    
    def test_run_rolling_mode(self):
        """Should run rolling mode backtest."""
        base = datetime(2023, 1, 1)
        fixtures = []
        
        for i in range(200):
            fixtures.append(make_fixture(
                fixture_id=i,
                home_team_id=100 + (i % 5),
                away_team_id=105 + (i % 5),
                kickoff_utc=base + timedelta(days=i),
                home_goals=1,
                away_goals=0,
                status="FT",
            ))
        
        result = run_backtest(
            fixtures,
            mode="rolling",
            model_name="home_advantage",
            min_train_size=50,
            test_window_days=7,
        )
        
        assert result.mode == "rolling"
    
    def test_unknown_mode_raises(self):
        """Should raise for unknown mode."""
        with pytest.raises(ValueError, match="Unknown mode"):
            run_backtest([], mode="invalid")
