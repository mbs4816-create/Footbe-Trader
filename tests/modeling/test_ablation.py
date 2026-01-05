"""Tests for ablation study support."""

from datetime import datetime, timedelta
import numpy as np
import pytest

from footbe_trader.modeling.ablation import (
    AblationResult,
    AblationStudy,
    AblationStudyResult,
    DEFAULT_FEATURE_GROUPS,
    FeatureGroup,
    FeatureMasker,
    format_ablation_report,
)
from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.models import MultinomialLogisticModel


def make_feature_vector(
    fixture_id: int = 1,
    home_team_id: int = 1,
    away_team_id: int = 2,
    outcome: str | None = "H",
    kickoff_utc: datetime | None = None,
) -> MatchFeatureVector:
    """Create a MatchFeatureVector for testing."""
    if kickoff_utc is None:
        kickoff_utc = datetime(2024, 1, 1) + timedelta(days=fixture_id)
    
    np.random.seed(fixture_id)
    
    return MatchFeatureVector(
        fixture_id=fixture_id,
        kickoff_utc=kickoff_utc,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        season=2024,
        round_str="Round 1",
        outcome=outcome,
        home_goals=2 if outcome == "H" else (1 if outcome == "D" else 0),
        away_goals=0 if outcome == "H" else (1 if outcome == "D" else 2),
        home_advantage=1.0,
        home_team_home_goals_scored_avg=1.5 + np.random.randn() * 0.3,
        home_team_home_goals_conceded_avg=1.0 + np.random.randn() * 0.3,
        home_team_away_goals_scored_avg=1.2 + np.random.randn() * 0.3,
        home_team_away_goals_conceded_avg=1.3 + np.random.randn() * 0.3,
        away_team_home_goals_scored_avg=1.4 + np.random.randn() * 0.3,
        away_team_home_goals_conceded_avg=1.1 + np.random.randn() * 0.3,
        away_team_away_goals_scored_avg=1.0 + np.random.randn() * 0.3,
        away_team_away_goals_conceded_avg=1.5 + np.random.randn() * 0.3,
        rest_days_diff=np.random.randn() * 2,
    )


def make_training_data(n_samples: int = 100) -> list[MatchFeatureVector]:
    """Create training data."""
    outcomes = ["H", "D", "A"]
    probs = [0.45, 0.27, 0.28]
    
    features = []
    for i in range(n_samples):
        outcome = np.random.choice(outcomes, p=probs)
        features.append(make_feature_vector(
            fixture_id=i + 1,
            home_team_id=(i % 20) + 1,
            away_team_id=((i + 10) % 20) + 1,
            outcome=outcome,
        ))
    
    return features


class TestFeatureGroup:
    """Tests for FeatureGroup dataclass."""
    
    def test_creation(self):
        """Should create feature group correctly."""
        group = FeatureGroup(
            name="test",
            description="Test group",
            feature_indices=[0, 1],
            feature_names=["feature_0", "feature_1"],
        )
        
        assert group.name == "test"
        assert group.feature_indices == [0, 1]
    
    def test_to_dict(self):
        """to_dict should serialize correctly."""
        group = FeatureGroup(
            name="test",
            description="Test group",
            feature_indices=[0, 1],
            feature_names=["feature_0", "feature_1"],
        )
        
        d = group.to_dict()
        
        assert d["name"] == "test"
        assert d["feature_indices"] == [0, 1]


class TestDefaultFeatureGroups:
    """Tests for default feature group definitions."""
    
    def test_groups_defined(self):
        """Should have predefined groups A-F."""
        assert "A" in DEFAULT_FEATURE_GROUPS
        assert "B" in DEFAULT_FEATURE_GROUPS
        assert "C" in DEFAULT_FEATURE_GROUPS
        assert "D" in DEFAULT_FEATURE_GROUPS
        assert "E" in DEFAULT_FEATURE_GROUPS
        assert "F" in DEFAULT_FEATURE_GROUPS
    
    def test_group_a_is_home_advantage(self):
        """Group A should be home advantage constant."""
        group_a = DEFAULT_FEATURE_GROUPS["A"]
        assert group_a.feature_indices == [0]
        assert "home_advantage" in group_a.feature_names[0].lower()
    
    def test_all_features_covered(self):
        """All 10 features should be covered by groups."""
        covered = set()
        for group in DEFAULT_FEATURE_GROUPS.values():
            covered.update(group.feature_indices)
        
        assert covered == set(range(10))


class TestFeatureMasker:
    """Tests for FeatureMasker."""
    
    def test_get_mask_single_group(self):
        """Should create mask for single group."""
        masker = FeatureMasker()
        mask = masker.get_mask(["A"])
        
        assert mask[0] == True  # Home advantage
        assert not any(mask[1:])  # Others should be False
    
    def test_get_mask_multiple_groups(self):
        """Should create mask for multiple groups."""
        masker = FeatureMasker()
        mask = masker.get_mask(["A", "B"])
        
        assert mask[0] == True  # A: home_advantage
        assert mask[1] == True  # B: home_team_home_goals_scored
        assert mask[2] == True  # B: home_team_home_goals_conceded
    
    def test_apply_mask_1d(self):
        """Should apply mask to 1D array."""
        masker = FeatureMasker()
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mask = masker.get_mask(["A"])
        
        masked = masker.apply_mask(features, mask, fill_value=0.0)
        
        assert masked[0] == 1.0  # Kept
        assert masked[1] == 0.0  # Masked
        assert masked[9] == 0.0  # Masked
    
    def test_apply_mask_2d(self):
        """Should apply mask to 2D array."""
        masker = FeatureMasker()
        features = np.arange(30).reshape(3, 10).astype(float)
        mask = masker.get_mask(["A", "F"])
        
        masked = masker.apply_mask(features, mask, fill_value=-1.0)
        
        # Column 0 (A) and 9 (F) should be kept
        assert masked[0, 0] == 0.0
        assert masked[0, 9] == 9.0
        
        # Other columns should be masked
        assert masked[0, 1] == -1.0
        assert masked[0, 5] == -1.0
    
    def test_mask_feature_vectors(self):
        """Should mask MatchFeatureVector objects."""
        masker = FeatureMasker()
        features = [make_feature_vector(i) for i in range(3)]
        
        # Keep only group A (home advantage)
        masked = masker.mask_feature_vectors(features, ["A"])
        
        assert len(masked) == 3
        for mf in masked:
            assert mf.home_advantage == 1.0  # Kept
            assert mf.home_team_home_goals_scored_avg == 0.0  # Masked
            assert mf.rest_days_diff == 0.0  # Masked


class TestAblationStudy:
    """Tests for AblationStudy."""
    
    def test_run_completes(self):
        """Ablation study should complete without errors."""
        np.random.seed(42)
        
        def model_factory():
            return MultinomialLogisticModel(max_iterations=50)
        
        study = AblationStudy(model_factory)
        
        train_features = make_training_data(100)
        test_features = make_training_data(50)
        
        results = study.run(train_features, test_features)
        
        assert isinstance(results, AblationStudyResult)
        assert results.baseline_log_loss > 0
        assert results.full_model_log_loss > 0
    
    def test_add_group_experiments(self):
        """Should run add-group experiments."""
        np.random.seed(42)
        
        def model_factory():
            return MultinomialLogisticModel(max_iterations=50)
        
        study = AblationStudy(model_factory)
        train = make_training_data(100)
        test = make_training_data(50)
        
        results = study.run(train, test)
        
        # Should have results for groups B-F (A is in baseline)
        assert len(results.add_group_results) == 5
        
        for r in results.add_group_results:
            assert r.experiment_type == "add"
            assert r.group_name in ["B", "C", "D", "E", "F"]
            assert "A" in r.included_groups
    
    def test_drop_group_experiments(self):
        """Should run drop-group experiments."""
        np.random.seed(42)
        
        def model_factory():
            return MultinomialLogisticModel(max_iterations=50)
        
        study = AblationStudy(model_factory)
        train = make_training_data(100)
        test = make_training_data(50)
        
        results = study.run(train, test)
        
        # Should have results for all groups
        assert len(results.drop_group_results) == 6
        
        for r in results.drop_group_results:
            assert r.experiment_type == "drop"
            assert r.group_name not in r.included_groups


class TestAblationStudyResult:
    """Tests for AblationStudyResult."""
    
    def test_get_importance_ranking(self):
        """Should rank features by importance."""
        results = AblationStudyResult(
            baseline_log_loss=1.2,
            full_model_log_loss=1.0,
            add_group_results=[],
            drop_group_results=[
                AblationResult("drop", "A", ["B", "C"], 1.1, 0.1, 5),
                AblationResult("drop", "B", ["A", "C"], 1.2, 0.2, 5),
                AblationResult("drop", "C", ["A", "B"], 1.05, 0.05, 5),
            ],
            feature_groups={
                "A": FeatureGroup("A", "desc", [0], ["f0"]),
                "B": FeatureGroup("B", "desc", [1], ["f1"]),
                "C": FeatureGroup("C", "desc", [2], ["f2"]),
            },
        )
        
        rankings = results.get_importance_ranking()
        
        # B should be most important (highest delta when dropped)
        assert rankings[0][0] == "B"
        assert rankings[0][1] == 0.2
        
        # Then A
        assert rankings[1][0] == "A"
        
        # C least important
        assert rankings[2][0] == "C"
    
    def test_to_dict(self):
        """Should serialize to dictionary."""
        results = AblationStudyResult(
            baseline_log_loss=1.2,
            full_model_log_loss=1.0,
            add_group_results=[],
            drop_group_results=[],
            feature_groups=DEFAULT_FEATURE_GROUPS,
        )
        
        d = results.to_dict()
        
        assert d["baseline_log_loss"] == 1.2
        assert d["full_model_log_loss"] == 1.0
        assert "feature_groups" in d


class TestFormatAblationReport:
    """Tests for ablation report formatting."""
    
    def test_format_produces_string(self):
        """Should produce formatted string report."""
        results = AblationStudyResult(
            baseline_log_loss=1.2,
            full_model_log_loss=1.0,
            add_group_results=[
                AblationResult("add", "B", ["A", "B"], 1.15, -0.05, 3),
            ],
            drop_group_results=[
                AblationResult("drop", "A", ["B"], 1.1, 0.1, 1),
                AblationResult("drop", "B", ["A"], 1.25, 0.25, 1),
            ],
            feature_groups={
                "A": FeatureGroup("A", "Home advantage", [0], ["home_advantage"]),
                "B": FeatureGroup("B", "Home scoring", [1, 2], ["f1", "f2"]),
            },
        )
        
        report = format_ablation_report(results)
        
        assert isinstance(report, str)
        assert "ABLATION STUDY" in report
        assert "Baseline" in report
        assert "ADD-GROUP" in report
        assert "DROP-GROUP" in report
        assert "IMPORTANCE RANKING" in report
