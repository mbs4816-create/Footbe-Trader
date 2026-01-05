"""Ablation study support for feature importance analysis.

This module enables systematic feature ablation experiments:
- Define feature groups (A, B, C, ...)
- Run "add-group" experiments (start minimal, add one group)
- Run "drop-group" experiments (start full, drop one group)
- Compute delta log loss for each feature group

Ablation studies help understand:
- Which feature groups contribute most to model performance
- Whether features are complementary or redundant
- Optimal feature set for production
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.metrics import compute_log_loss


@dataclass
class FeatureGroup:
    """Definition of a feature group for ablation."""
    
    name: str
    description: str
    feature_indices: list[int]  # Indices into feature array
    feature_names: list[str]  # Human-readable names
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "feature_indices": self.feature_indices,
            "feature_names": self.feature_names,
        }


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    
    experiment_type: str  # "add" or "drop"
    group_name: str
    included_groups: list[str]
    log_loss: float
    delta_log_loss: float  # Change from baseline
    n_features: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_type": self.experiment_type,
            "group_name": self.group_name,
            "included_groups": self.included_groups,
            "log_loss": log_loss,
            "delta_log_loss": self.delta_log_loss,
            "n_features": self.n_features,
        }


@dataclass
class AblationStudyResult:
    """Complete ablation study results."""
    
    baseline_log_loss: float
    full_model_log_loss: float
    add_group_results: list[AblationResult]
    drop_group_results: list[AblationResult]
    feature_groups: dict[str, FeatureGroup]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_log_loss": self.baseline_log_loss,
            "full_model_log_loss": self.full_model_log_loss,
            "add_group_results": [r.to_dict() for r in self.add_group_results],
            "drop_group_results": [r.to_dict() for r in self.drop_group_results],
            "feature_groups": {k: v.to_dict() for k, v in self.feature_groups.items()},
        }
    
    def get_importance_ranking(self) -> list[tuple[str, float]]:
        """Get feature groups ranked by importance (delta log loss when dropped)."""
        rankings = []
        for result in self.drop_group_results:
            # Positive delta means dropping the group hurts performance
            rankings.append((result.group_name, result.delta_log_loss))
        
        # Sort by delta (higher = more important)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


# Default feature groups based on MatchFeatureVector
DEFAULT_FEATURE_GROUPS = {
    "A": FeatureGroup(
        name="A",
        description="Home advantage constant",
        feature_indices=[0],
        feature_names=["home_advantage"],
    ),
    "B": FeatureGroup(
        name="B",
        description="Home team home performance",
        feature_indices=[1, 2],
        feature_names=[
            "home_team_home_goals_scored_avg",
            "home_team_home_goals_conceded_avg",
        ],
    ),
    "C": FeatureGroup(
        name="C",
        description="Home team away performance",
        feature_indices=[3, 4],
        feature_names=[
            "home_team_away_goals_scored_avg",
            "home_team_away_goals_conceded_avg",
        ],
    ),
    "D": FeatureGroup(
        name="D",
        description="Away team home performance",
        feature_indices=[5, 6],
        feature_names=[
            "away_team_home_goals_scored_avg",
            "away_team_home_goals_conceded_avg",
        ],
    ),
    "E": FeatureGroup(
        name="E",
        description="Away team away performance",
        feature_indices=[7, 8],
        feature_names=[
            "away_team_away_goals_scored_avg",
            "away_team_away_goals_conceded_avg",
        ],
    ),
    "F": FeatureGroup(
        name="F",
        description="Rest days differential",
        feature_indices=[9],
        feature_names=["rest_days_diff"],
    ),
}


class FeatureMasker:
    """Utility to mask features for ablation experiments."""
    
    def __init__(self, feature_groups: dict[str, FeatureGroup] | None = None):
        """Initialize feature masker.
        
        Args:
            feature_groups: Dictionary of feature group definitions.
        """
        self.feature_groups = feature_groups or DEFAULT_FEATURE_GROUPS
        self.n_features = max(
            max(g.feature_indices) for g in self.feature_groups.values()
        ) + 1
    
    def get_mask(self, included_groups: list[str]) -> NDArray[np.bool_]:
        """Create feature mask for specified groups.
        
        Args:
            included_groups: List of group names to include.
            
        Returns:
            Boolean mask array.
        """
        mask = np.zeros(self.n_features, dtype=bool)
        
        for group_name in included_groups:
            if group_name in self.feature_groups:
                for idx in self.feature_groups[group_name].feature_indices:
                    mask[idx] = True
        
        return mask
    
    def apply_mask(
        self,
        features: NDArray[np.float64],
        mask: NDArray[np.bool_],
        fill_value: float = 0.0,
    ) -> NDArray[np.float64]:
        """Apply mask to feature array.
        
        Args:
            features: Feature array (n_samples, n_features) or (n_features,).
            mask: Boolean mask.
            fill_value: Value to use for masked features.
            
        Returns:
            Masked feature array.
        """
        result = features.copy()
        if result.ndim == 1:
            result[~mask] = fill_value
        else:
            result[:, ~mask] = fill_value
        return result
    
    def mask_feature_vectors(
        self,
        features: list[MatchFeatureVector],
        included_groups: list[str],
    ) -> list[MatchFeatureVector]:
        """Create masked copies of feature vectors.
        
        Args:
            features: Original feature vectors.
            included_groups: Groups to include.
            
        Returns:
            New feature vectors with masked features.
        """
        mask = self.get_mask(included_groups)
        masked_features = []
        
        for f in features:
            # Convert to array, mask, and create new vector
            arr = f.to_feature_array()
            masked_arr = self.apply_mask(arr, mask, fill_value=0.0)
            
            # Create new feature vector with masked values
            masked_f = MatchFeatureVector(
                fixture_id=f.fixture_id,
                kickoff_utc=f.kickoff_utc,
                home_team_id=f.home_team_id,
                away_team_id=f.away_team_id,
                season=f.season,
                round_str=f.round_str,
                outcome=f.outcome,
                home_goals=f.home_goals,
                away_goals=f.away_goals,
                home_advantage=masked_arr[0] if mask[0] else 0.0,
                home_team_home_goals_scored_avg=masked_arr[1] if mask[1] else 0.0,
                home_team_home_goals_conceded_avg=masked_arr[2] if mask[2] else 0.0,
                home_team_away_goals_scored_avg=masked_arr[3] if mask[3] else 0.0,
                home_team_away_goals_conceded_avg=masked_arr[4] if mask[4] else 0.0,
                away_team_home_goals_scored_avg=masked_arr[5] if mask[5] else 0.0,
                away_team_home_goals_conceded_avg=masked_arr[6] if mask[6] else 0.0,
                away_team_away_goals_scored_avg=masked_arr[7] if mask[7] else 0.0,
                away_team_away_goals_conceded_avg=masked_arr[8] if mask[8] else 0.0,
                rest_days_diff=masked_arr[9] if mask[9] else 0.0,
            )
            masked_features.append(masked_f)
        
        return masked_features


class AblationStudy:
    """Run systematic feature ablation experiments.
    
    Usage:
        study = AblationStudy(model_factory, feature_groups)
        results = study.run(train_features, test_features, test_labels)
    """
    
    def __init__(
        self,
        model_factory: Callable[[], Any],
        feature_groups: dict[str, FeatureGroup] | None = None,
    ):
        """Initialize ablation study.
        
        Args:
            model_factory: Function that creates a new model instance.
            feature_groups: Feature group definitions.
        """
        self.model_factory = model_factory
        self.feature_groups = feature_groups or DEFAULT_FEATURE_GROUPS
        self.masker = FeatureMasker(self.feature_groups)
    
    def _train_and_evaluate(
        self,
        train_features: list[MatchFeatureVector],
        test_features: list[MatchFeatureVector],
        y_true: NDArray[np.int64],
        included_groups: list[str],
    ) -> float:
        """Train model on subset of features and evaluate.
        
        Args:
            train_features: Training data.
            test_features: Test data.
            y_true: True labels for test data.
            included_groups: Feature groups to include.
            
        Returns:
            Log loss on test data.
        """
        # Mask features
        masked_train = self.masker.mask_feature_vectors(train_features, included_groups)
        masked_test = self.masker.mask_feature_vectors(test_features, included_groups)
        
        # Train model
        model = self.model_factory()
        try:
            model.fit(masked_train)
            proba = model.predict_proba(masked_test)
            return compute_log_loss(y_true, proba)
        except Exception as e:
            # Return high loss if training fails
            return 2.0
    
    def run(
        self,
        train_features: list[MatchFeatureVector],
        test_features: list[MatchFeatureVector],
        test_labels: NDArray[np.int64] | None = None,
    ) -> AblationStudyResult:
        """Run complete ablation study.
        
        Args:
            train_features: Training data.
            test_features: Test data.
            test_labels: True labels (if None, extracted from test_features).
            
        Returns:
            Complete ablation study results.
        """
        # Get test labels
        if test_labels is None:
            label_map = {"H": 0, "D": 1, "A": 2}
            test_labels = np.array([
                label_map.get(f.outcome, 0) for f in test_features
            ])
        
        all_groups = list(self.feature_groups.keys())
        
        # Baseline: minimal model (just home advantage)
        baseline_loss = self._train_and_evaluate(
            train_features, test_features, test_labels, ["A"]
        )
        
        # Full model: all features
        full_loss = self._train_and_evaluate(
            train_features, test_features, test_labels, all_groups
        )
        
        # Add-group experiments: start with A, add one group at a time
        add_results = []
        base_groups = ["A"]
        base_loss = baseline_loss
        
        for group_name in all_groups:
            if group_name == "A":
                continue  # Already in baseline
            
            current_groups = base_groups + [group_name]
            current_loss = self._train_and_evaluate(
                train_features, test_features, test_labels, current_groups
            )
            
            add_results.append(AblationResult(
                experiment_type="add",
                group_name=group_name,
                included_groups=current_groups,
                log_loss=current_loss,
                delta_log_loss=current_loss - base_loss,  # Negative = improvement
                n_features=sum(
                    len(self.feature_groups[g].feature_indices)
                    for g in current_groups
                ),
            ))
        
        # Drop-group experiments: start with all, drop one group at a time
        drop_results = []
        
        for group_name in all_groups:
            remaining_groups = [g for g in all_groups if g != group_name]
            current_loss = self._train_and_evaluate(
                train_features, test_features, test_labels, remaining_groups
            )
            
            drop_results.append(AblationResult(
                experiment_type="drop",
                group_name=group_name,
                included_groups=remaining_groups,
                log_loss=current_loss,
                delta_log_loss=current_loss - full_loss,  # Positive = group was helpful
                n_features=sum(
                    len(self.feature_groups[g].feature_indices)
                    for g in remaining_groups
                ),
            ))
        
        return AblationStudyResult(
            baseline_log_loss=baseline_loss,
            full_model_log_loss=full_loss,
            add_group_results=add_results,
            drop_group_results=drop_results,
            feature_groups=self.feature_groups,
        )


def format_ablation_report(results: AblationStudyResult) -> str:
    """Format ablation study results as a readable report.
    
    Args:
        results: Ablation study results.
        
    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("=" * 60)
    lines.append("")
    
    lines.append(f"Baseline (group A only) log loss: {results.baseline_log_loss:.4f}")
    lines.append(f"Full model log loss: {results.full_model_log_loss:.4f}")
    lines.append(f"Total improvement: {results.baseline_log_loss - results.full_model_log_loss:.4f}")
    lines.append("")
    
    lines.append("-" * 60)
    lines.append("ADD-GROUP EXPERIMENTS (starting from baseline)")
    lines.append("-" * 60)
    lines.append(f"{'Group':<8} {'Description':<35} {'ΔLL':<10} {'LL':<10}")
    lines.append("-" * 60)
    
    for r in results.add_group_results:
        desc = results.feature_groups[r.group_name].description[:35]
        delta_str = f"{r.delta_log_loss:+.4f}"
        lines.append(f"{r.group_name:<8} {desc:<35} {delta_str:<10} {r.log_loss:.4f}")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append("DROP-GROUP EXPERIMENTS (from full model)")
    lines.append("-" * 60)
    lines.append(f"{'Group':<8} {'Description':<35} {'ΔLL':<10} {'LL':<10}")
    lines.append("-" * 60)
    
    for r in results.drop_group_results:
        desc = results.feature_groups[r.group_name].description[:35]
        delta_str = f"{r.delta_log_loss:+.4f}"
        lines.append(f"{r.group_name:<8} {desc:<35} {delta_str:<10} {r.log_loss:.4f}")
    
    lines.append("")
    lines.append("-" * 60)
    lines.append("FEATURE IMPORTANCE RANKING (by drop-group ΔLL)")
    lines.append("-" * 60)
    
    rankings = results.get_importance_ranking()
    for i, (group_name, delta) in enumerate(rankings, 1):
        desc = results.feature_groups[group_name].description
        lines.append(f"{i}. {group_name}: {desc} (ΔLL={delta:+.4f})")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)
