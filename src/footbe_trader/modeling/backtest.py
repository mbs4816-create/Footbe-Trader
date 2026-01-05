"""Rolling-origin backtest framework for 3-way match prediction.

Supports two modes:
1. Season-by-season: Train on seasons ≤ S-1, test on season S
2. Rolling matchweek: Train on all completed fixtures, test on next matchweek

Strict time integrity: Features computed ONLY from fixtures with kickoff < target.
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from footbe_trader.modeling.features import FeatureBuilder, MatchFeatureVector
from footbe_trader.modeling.metrics import (
    BootstrapResult,
    MetricsResult,
    bootstrap_compare,
    evaluate_predictions,
    outcome_to_label,
)
from footbe_trader.modeling.models import BaseModel, create_model
from footbe_trader.storage.models import FixtureV2


@dataclass
class FoldResult:
    """Results from a single backtest fold."""
    
    fold_name: str
    train_size: int
    test_size: int
    model_name: str
    metrics: MetricsResult
    
    # Store predictions for bootstrap comparison
    fixture_ids: list[int] = field(default_factory=list)
    y_true: NDArray[np.int64] = field(default_factory=lambda: np.array([]))
    y_proba: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without arrays)."""
        return {
            "fold_name": self.fold_name,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "model_name": self.model_name,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    
    mode: str  # "season" or "rolling"
    model_name: str
    folds: list[FoldResult] = field(default_factory=list)
    aggregate_metrics: MetricsResult | None = None
    
    # Comparison to baseline (if computed)
    bootstrap_comparison: BootstrapResult | None = None
    baseline_metrics: MetricsResult | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "mode": self.mode,
            "model_name": self.model_name,
            "folds": [f.to_dict() for f in self.folds],
        }
        if self.aggregate_metrics:
            result["aggregate_metrics"] = self.aggregate_metrics.to_dict()
        if self.bootstrap_comparison:
            result["bootstrap_comparison"] = self.bootstrap_comparison.to_dict()
        if self.baseline_metrics:
            result["baseline_metrics"] = self.baseline_metrics.to_dict()
        return result


def get_outcome(fixture: FixtureV2) -> str | None:
    """Get match outcome as H/D/A.
    
    Returns None if fixture not finished or goals missing.
    """
    if fixture.home_goals is None or fixture.away_goals is None:
        return None
    if fixture.status != "FT":
        return None
    
    if fixture.home_goals > fixture.away_goals:
        return "H"
    elif fixture.home_goals < fixture.away_goals:
        return "A"
    else:
        return "D"


def get_completed_fixtures(fixtures: list[FixtureV2]) -> list[FixtureV2]:
    """Filter to completed fixtures with valid outcomes."""
    return [f for f in fixtures if get_outcome(f) is not None]


def generate_season_folds(
    fixtures: list[FixtureV2],
    min_train_seasons: int = 1,
) -> Generator[tuple[list[FixtureV2], list[FixtureV2], str], None, None]:
    """Generate season-by-season train/test splits.
    
    For each test season S, train on all fixtures from seasons < S.
    
    Args:
        fixtures: All fixtures (will filter to completed).
        min_train_seasons: Minimum number of training seasons.
        
    Yields:
        (train_fixtures, test_fixtures, fold_name)
    """
    completed = get_completed_fixtures(fixtures)
    
    # Get unique seasons sorted
    seasons = sorted(set(f.season for f in completed if f.season))
    
    if len(seasons) < min_train_seasons + 1:
        return  # Not enough seasons
    
    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = set(seasons[:i])
        
        train = [f for f in completed if f.season in train_seasons]
        test = [f for f in completed if f.season == test_season]
        
        # Sort by kickoff time
        train.sort(key=lambda f: f.kickoff_utc)
        test.sort(key=lambda f: f.kickoff_utc)
        
        if len(train) > 0 and len(test) > 0:
            yield train, test, f"train_≤{seasons[i-1]}_test_{test_season}"


def generate_rolling_folds(
    fixtures: list[FixtureV2],
    min_train_size: int = 100,
    test_window_days: int = 7,
) -> Generator[tuple[list[FixtureV2], list[FixtureV2], str], None, None]:
    """Generate rolling matchweek train/test splits.
    
    Train on all completed fixtures up to date, test on next window.
    
    Args:
        fixtures: All fixtures (will filter to completed).
        min_train_size: Minimum training set size.
        test_window_days: Days in each test window.
        
    Yields:
        (train_fixtures, test_fixtures, fold_name)
    """
    from datetime import timedelta
    
    completed = get_completed_fixtures(fixtures)
    completed.sort(key=lambda f: f.kickoff_utc)
    
    if len(completed) < min_train_size + 1:
        return
    
    # Get date range
    start_date = completed[min_train_size - 1].kickoff_utc
    end_date = completed[-1].kickoff_utc
    
    current_date = start_date
    fold_num = 0
    
    while current_date < end_date:
        window_end = current_date + timedelta(days=test_window_days)
        
        # Train: all fixtures with kickoff < current_date
        train = [f for f in completed if f.kickoff_utc < current_date]
        
        # Test: fixtures in [current_date, window_end)
        test = [
            f for f in completed
            if current_date <= f.kickoff_utc < window_end
        ]
        
        if len(train) >= min_train_size and len(test) > 0:
            fold_name = f"fold_{fold_num:03d}_{current_date.strftime('%Y%m%d')}"
            yield train, test, fold_name
            fold_num += 1
        
        current_date = window_end


class Backtester:
    """Rolling-origin backtester with strict time integrity."""
    
    def __init__(
        self,
        model_name: str = "multinomial_logistic",
        baseline_model_name: str = "home_advantage",
        rolling_window: int = 10,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.01,
    ):
        """Initialize backtester.
        
        Args:
            model_name: Model type to evaluate.
            baseline_model_name: Baseline model for comparison.
            rolling_window: Window for rolling features.
            learning_rate: Learning rate for gradient descent.
            n_iterations: Training iterations.
            regularization: L2 regularization strength.
        """
        self.model_name = model_name
        self.baseline_model_name = baseline_model_name
        self.rolling_window = rolling_window
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        
        self.feature_builder = FeatureBuilder(rolling_window=rolling_window)
    
    def _prepare_data(
        self,
        fixtures: list[FixtureV2],
        all_fixtures: list[FixtureV2],
    ) -> tuple[list[MatchFeatureVector], list[str], list[int]]:
        """Build features for fixtures using all_fixtures as history.
        
        Args:
            fixtures: Target fixtures to build features for.
            all_fixtures: All fixtures (for historical context).
            
        Returns:
            (features, outcomes, fixture_ids)
        """
        features = []
        outcomes = []
        fixture_ids = []
        
        for fixture in fixtures:
            # Build features (only uses fixtures with kickoff < target)
            fv = self.feature_builder.build_features(all_fixtures, fixture)
            if fv is None:
                continue
            
            outcome = get_outcome(fixture)
            if outcome is None:
                continue
            
            features.append(fv)
            outcomes.append(outcome)
            fixture_ids.append(fixture.fixture_id)
        
        return features, outcomes, fixture_ids
    
    def _train_and_predict(
        self,
        model: BaseModel,
        train_features: list[MatchFeatureVector],
        train_outcomes: list[str],
        test_features: list[MatchFeatureVector],
    ) -> NDArray[np.float64]:
        """Train model and get predictions.
        
        Args:
            model: Model to train.
            train_features: Training features (with outcomes set).
            train_outcomes: Training outcomes (unused, outcomes are in features).
            test_features: Test features.
            
        Returns:
            Predicted probabilities, shape (n_test, 3).
        """
        # Train using the model's native API (features already have outcomes)
        model.fit(train_features)
        
        # Predict using native API 
        y_proba = model.predict_proba(test_features)
        
        return y_proba
    
    def run_fold(
        self,
        train_fixtures: list[FixtureV2],
        test_fixtures: list[FixtureV2],
        all_fixtures: list[FixtureV2],
        fold_name: str,
    ) -> tuple[FoldResult, FoldResult]:
        """Run backtest on a single fold.
        
        Args:
            train_fixtures: Training fixtures.
            test_fixtures: Test fixtures.
            all_fixtures: All fixtures (for feature building).
            fold_name: Name of this fold.
            
        Returns:
            (model_result, baseline_result)
        """
        # Prepare data
        # For training, we can use train_fixtures as history
        train_features, train_outcomes, _ = self._prepare_data(
            train_fixtures, train_fixtures
        )
        
        # For test, use all fixtures up to test time as potential history
        # But FeatureBuilder already filters by kickoff < target
        test_features, test_outcomes, test_ids = self._prepare_data(
            test_fixtures, all_fixtures
        )
        
        if len(train_features) == 0 or len(test_features) == 0:
            raise ValueError(f"Empty train or test set in fold {fold_name}")
        
        y_true = outcome_to_label(test_outcomes)
        
        # Train and evaluate main model
        model = create_model(
            self.model_name,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            regularization=self.regularization,
        )
        y_proba_model = self._train_and_predict(
            model, train_features, train_outcomes, test_features
        )
        model_metrics = evaluate_predictions(y_true, y_proba_model)
        
        model_result = FoldResult(
            fold_name=fold_name,
            train_size=len(train_features),
            test_size=len(test_features),
            model_name=self.model_name,
            metrics=model_metrics,
            fixture_ids=test_ids,
            y_true=y_true,
            y_proba=y_proba_model,
        )
        
        # Train and evaluate baseline
        baseline = create_model(self.baseline_model_name)
        y_proba_baseline = self._train_and_predict(
            baseline, train_features, train_outcomes, test_features
        )
        baseline_metrics = evaluate_predictions(y_true, y_proba_baseline)
        
        baseline_result = FoldResult(
            fold_name=fold_name,
            train_size=len(train_features),
            test_size=len(test_features),
            model_name=self.baseline_model_name,
            metrics=baseline_metrics,
            fixture_ids=test_ids,
            y_true=y_true,
            y_proba=y_proba_baseline,
        )
        
        return model_result, baseline_result
    
    def run_season_backtest(
        self,
        fixtures: list[FixtureV2],
        min_train_seasons: int = 1,
    ) -> BacktestResult:
        """Run season-by-season backtest.
        
        Args:
            fixtures: All fixtures.
            min_train_seasons: Minimum training seasons.
            
        Returns:
            BacktestResult with all folds and comparison.
        """
        model_folds = []
        baseline_folds = []
        
        for train, test, fold_name in generate_season_folds(fixtures, min_train_seasons):
            try:
                model_result, baseline_result = self.run_fold(
                    train, test, fixtures, fold_name
                )
                model_folds.append(model_result)
                baseline_folds.append(baseline_result)
            except ValueError as e:
                print(f"Skipping fold {fold_name}: {e}")
                continue
        
        return self._aggregate_results(model_folds, baseline_folds, "season")
    
    def run_rolling_backtest(
        self,
        fixtures: list[FixtureV2],
        min_train_size: int = 100,
        test_window_days: int = 7,
    ) -> BacktestResult:
        """Run rolling matchweek backtest.
        
        Args:
            fixtures: All fixtures.
            min_train_size: Minimum training samples.
            test_window_days: Days per test window.
            
        Returns:
            BacktestResult with all folds and comparison.
        """
        model_folds = []
        baseline_folds = []
        
        for train, test, fold_name in generate_rolling_folds(
            fixtures, min_train_size, test_window_days
        ):
            try:
                model_result, baseline_result = self.run_fold(
                    train, test, fixtures, fold_name
                )
                model_folds.append(model_result)
                baseline_folds.append(baseline_result)
            except ValueError as e:
                print(f"Skipping fold {fold_name}: {e}")
                continue
        
        return self._aggregate_results(model_folds, baseline_folds, "rolling")
    
    def _aggregate_results(
        self,
        model_folds: list[FoldResult],
        baseline_folds: list[FoldResult],
        mode: str,
    ) -> BacktestResult:
        """Aggregate fold results and compute bootstrap comparison.
        
        Args:
            model_folds: Model results per fold.
            baseline_folds: Baseline results per fold.
            mode: Backtest mode name.
            
        Returns:
            Aggregated BacktestResult.
        """
        if len(model_folds) == 0:
            return BacktestResult(
                mode=mode,
                model_name=self.model_name,
            )
        
        # Concatenate predictions across folds
        all_y_true = np.concatenate([f.y_true for f in model_folds])
        all_y_proba_model = np.concatenate([f.y_proba for f in model_folds])
        all_y_proba_baseline = np.concatenate([f.y_proba for f in baseline_folds])
        
        # Aggregate metrics
        aggregate_model = evaluate_predictions(all_y_true, all_y_proba_model)
        aggregate_baseline = evaluate_predictions(all_y_true, all_y_proba_baseline)
        
        # Bootstrap comparison
        bootstrap = bootstrap_compare(
            all_y_true, all_y_proba_model, all_y_proba_baseline
        )
        
        return BacktestResult(
            mode=mode,
            model_name=self.model_name,
            folds=model_folds,
            aggregate_metrics=aggregate_model,
            bootstrap_comparison=bootstrap,
            baseline_metrics=aggregate_baseline,
        )


def run_backtest(
    fixtures: list[FixtureV2],
    mode: str = "season",
    model_name: str = "multinomial_logistic",
    baseline_model_name: str = "home_advantage",
    rolling_window: int = 10,
    min_train_seasons: int = 1,
    min_train_size: int = 100,
    test_window_days: int = 7,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    regularization: float = 0.01,
) -> BacktestResult:
    """Run backtest with specified configuration.
    
    Args:
        fixtures: All fixtures to backtest on.
        mode: "season" or "rolling".
        model_name: Model to evaluate.
        baseline_model_name: Baseline for comparison.
        rolling_window: Feature rolling window.
        min_train_seasons: Min seasons for season mode.
        min_train_size: Min samples for rolling mode.
        test_window_days: Test window for rolling mode.
        learning_rate: Learning rate for gradient descent.
        n_iterations: Training iterations.
        regularization: L2 regularization.
        
    Returns:
        BacktestResult with all results.
    """
    backtester = Backtester(
        model_name=model_name,
        baseline_model_name=baseline_model_name,
        rolling_window=rolling_window,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        regularization=regularization,
    )
    
    if mode == "season":
        return backtester.run_season_backtest(fixtures, min_train_seasons)
    elif mode == "rolling":
        return backtester.run_rolling_backtest(
            fixtures, min_train_size, test_window_days
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
