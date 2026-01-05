"""Tests for baseline models."""

from datetime import datetime, timedelta
import random

import numpy as np
import pytest

from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.models import (
    BaseModel,
    HomeAdvantageModel,
    MultinomialLogisticModel,
    PredictionOutput,
    create_model,
)


def make_feature_vector(
    fixture_id: int = 1,
    home_team_id: int = 1,
    away_team_id: int = 2,
    outcome: str | None = "H",
    kickoff_utc: datetime | None = None,
    **kwargs,
) -> MatchFeatureVector:
    """Create a MatchFeatureVector for testing."""
    if kickoff_utc is None:
        kickoff_utc = datetime(2024, 1, 1) + timedelta(days=fixture_id)
    
    return MatchFeatureVector(
        fixture_id=fixture_id,
        kickoff_utc=kickoff_utc,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        season=2024,
        round_str="Round 1",
        outcome=outcome,
        home_goals=kwargs.get("home_goals", 2 if outcome == "H" else 1 if outcome == "D" else 0),
        away_goals=kwargs.get("away_goals", 0 if outcome == "H" else 1 if outcome == "D" else 2),
        home_advantage=kwargs.get("home_advantage", 1.0),
        home_team_home_goals_scored_avg=kwargs.get("home_team_home_goals_scored_avg", 1.5),
        home_team_home_goals_conceded_avg=kwargs.get("home_team_home_goals_conceded_avg", 1.0),
        home_team_away_goals_scored_avg=kwargs.get("home_team_away_goals_scored_avg", 1.2),
        home_team_away_goals_conceded_avg=kwargs.get("home_team_away_goals_conceded_avg", 1.3),
        away_team_home_goals_scored_avg=kwargs.get("away_team_home_goals_scored_avg", 1.4),
        away_team_home_goals_conceded_avg=kwargs.get("away_team_home_goals_conceded_avg", 1.1),
        away_team_away_goals_scored_avg=kwargs.get("away_team_away_goals_scored_avg", 1.0),
        away_team_away_goals_conceded_avg=kwargs.get("away_team_away_goals_conceded_avg", 1.5),
        rest_days_diff=kwargs.get("rest_days_diff", 0.0),
    )


def make_training_data(n_samples: int = 100, outcome_dist: dict | None = None) -> list[MatchFeatureVector]:
    """Create training data with specified outcome distribution."""
    if outcome_dist is None:
        outcome_dist = {"H": 0.45, "D": 0.27, "A": 0.28}
    
    features = []
    outcomes = ["H", "D", "A"]
    
    for i in range(n_samples):
        # Pick outcome based on distribution
        r = random.random()
        cumulative = 0.0
        outcome = "H"
        for o in outcomes:
            cumulative += outcome_dist[o]
            if r < cumulative:
                outcome = o
                break
        
        # Add some variation to features
        features.append(make_feature_vector(
            fixture_id=i + 1,
            home_team_id=(i % 20) + 1,
            away_team_id=((i + 10) % 20) + 1,
            outcome=outcome,
            home_team_home_goals_scored_avg=1.5 + random.gauss(0, 0.3),
            home_team_home_goals_conceded_avg=1.0 + random.gauss(0, 0.3),
            away_team_away_goals_scored_avg=1.0 + random.gauss(0, 0.3),
            away_team_away_goals_conceded_avg=1.5 + random.gauss(0, 0.3),
            rest_days_diff=random.gauss(0, 2),
        ))
    
    return features


class TestPredictionOutput:
    """Tests for PredictionOutput dataclass."""
    
    def test_probabilities_sum_to_one(self):
        """Valid predictions should sum to 1."""
        pred = PredictionOutput(
            fixture_id=1,
            prob_home=0.45,
            prob_draw=0.25,
            prob_away=0.30,
            predicted_outcome="H",
        )
        total = pred.prob_home + pred.prob_draw + pred.prob_away
        assert abs(total - 1.0) < 1e-10
    
    def test_probabilities_property(self):
        """probabilities property should return numpy array."""
        pred = PredictionOutput(
            fixture_id=1,
            prob_home=0.5,
            prob_draw=0.3,
            prob_away=0.2,
            predicted_outcome="H",
        )
        arr = pred.probabilities
        
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3
        assert arr[0] == 0.5
        assert arr[1] == 0.3
        assert arr[2] == 0.2
    
    def test_to_dict(self):
        """to_dict should return dictionary representation."""
        pred = PredictionOutput(
            fixture_id=123,
            prob_home=0.5,
            prob_draw=0.3,
            prob_away=0.2,
            predicted_outcome="H",
        )
        d = pred.to_dict()
        
        assert d["fixture_id"] == 123
        assert d["prob_home"] == 0.5
        assert d["prob_draw"] == 0.3
        assert d["prob_away"] == 0.2
        assert d["predicted_outcome"] == "H"


class TestHomeAdvantageModel:
    """Tests for HomeAdvantageModel (Model0)."""
    
    def test_fit_computes_frequencies(self):
        """Fitting should compute outcome frequencies."""
        model = HomeAdvantageModel()
        
        # 4 home wins, 2 draws, 2 away wins
        features = [
            make_feature_vector(fixture_id=i + 1, outcome=o)
            for i, o in enumerate(["H", "H", "H", "H", "D", "D", "A", "A"])
        ]
        
        model.fit(features)
        
        assert model.is_trained
        assert abs(model.prob_home - 0.5) < 0.01
        assert abs(model.prob_draw - 0.25) < 0.01
        assert abs(model.prob_away - 0.25) < 0.01
    
    def test_predict_proba_returns_constant(self):
        """Predictions should be constant regardless of features."""
        model = HomeAdvantageModel()
        
        # 45% H, 27% D, 28% A
        train_features = make_training_data(100, {"H": 0.45, "D": 0.27, "A": 0.28})
        model.fit(train_features)
        
        # Predict on various inputs
        test_features = make_training_data(10)
        proba = model.predict_proba(test_features)
        
        assert proba.shape == (10, 3)
        
        # All predictions should be approximately the same (within sampling noise)
        for i in range(10):
            # Check they're within reasonable bounds of training distribution
            assert 0.3 < proba[i, 0] < 0.6  # home
            assert 0.1 < proba[i, 1] < 0.4  # draw
            assert 0.1 < proba[i, 2] < 0.4  # away
    
    def test_probabilities_sum_to_one(self):
        """Predicted probabilities should sum to 1."""
        model = HomeAdvantageModel()
        
        train_features = make_training_data(50)
        model.fit(train_features)
        
        proba = model.predict_proba(train_features)
        
        for i in range(len(train_features)):
            assert abs(proba[i].sum() - 1.0) < 1e-10
    
    def test_predict_returns_prediction_outputs(self):
        """predict should return list of PredictionOutput."""
        model = HomeAdvantageModel()
        
        train_features = make_training_data(50)
        model.fit(train_features)
        
        predictions = model.predict(train_features[:5])
        
        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, PredictionOutput)
            assert abs(pred.prob_home + pred.prob_draw + pred.prob_away - 1.0) < 1e-10
    
    def test_not_trained_raises(self):
        """Predicting without training should raise."""
        model = HomeAdvantageModel()
        
        test_features = [make_feature_vector()]
        
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(test_features)
    
    def test_fit_with_empty_outcomes(self):
        """Fitting with no outcomes should use defaults."""
        model = HomeAdvantageModel()
        
        # Features without outcomes
        features = [make_feature_vector(outcome=None)]
        model.fit(features)
        
        assert model.is_trained
        # Should use EPL historical defaults
        assert model.prob_home == 0.45
        assert model.prob_draw == 0.25
        assert model.prob_away == 0.30


class TestMultinomialLogisticModel:
    """Tests for MultinomialLogisticModel (Model1)."""
    
    def test_fit_learns_weights(self):
        """Fitting should learn non-zero weights."""
        model = MultinomialLogisticModel(max_iterations=100, learning_rate=0.1)
        
        features = make_training_data(100)
        model.fit(features)
        
        assert model.is_trained
        assert model.weights is not None
        # 10 features + 1 bias, 3 classes
        assert model.weights.shape == (11, 3)
    
    def test_predict_proba_shape(self):
        """Predictions should have correct shape."""
        model = MultinomialLogisticModel(max_iterations=100)
        
        train_features = make_training_data(100)
        model.fit(train_features)
        
        test_features = make_training_data(20)
        proba = model.predict_proba(test_features)
        
        assert proba.shape == (20, 3)
    
    def test_probabilities_sum_to_one(self):
        """Predicted probabilities should sum to 1."""
        model = MultinomialLogisticModel(max_iterations=100)
        
        features = make_training_data(100)
        model.fit(features)
        proba = model.predict_proba(features)
        
        for i in range(len(features)):
            assert abs(proba[i].sum() - 1.0) < 1e-10
    
    def test_probabilities_positive(self):
        """All probabilities should be positive."""
        model = MultinomialLogisticModel(max_iterations=100)
        
        features = make_training_data(100)
        model.fit(features)
        proba = model.predict_proba(features)
        
        assert np.all(proba > 0)
        assert np.all(proba < 1)
    
    def test_regularization_effect(self):
        """Higher regularization should produce smaller weights."""
        features = make_training_data(100)
        
        model_low_reg = MultinomialLogisticModel(
            max_iterations=100, regularization=0.001
        )
        model_high_reg = MultinomialLogisticModel(
            max_iterations=100, regularization=1.0
        )
        
        model_low_reg.fit(features)
        model_high_reg.fit(features)
        
        # High reg should have smaller weight norm
        norm_low = np.linalg.norm(model_low_reg.weights)
        norm_high = np.linalg.norm(model_high_reg.weights)
        
        assert norm_high < norm_low
    
    def test_predict_returns_prediction_outputs(self):
        """predict should return list of PredictionOutput."""
        model = MultinomialLogisticModel(max_iterations=100)
        
        features = make_training_data(50)
        model.fit(features)
        
        predictions = model.predict(features[:5])
        
        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, PredictionOutput)
            assert abs(pred.prob_home + pred.prob_draw + pred.prob_away - 1.0) < 1e-10
    
    def test_not_trained_raises(self):
        """Predicting without training should raise."""
        model = MultinomialLogisticModel()
        
        test_features = [make_feature_vector()]
        
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(test_features)
    
    def test_insufficient_data_raises(self):
        """Fitting with too little data should raise."""
        model = MultinomialLogisticModel()
        
        # Only 5 samples (need at least 10)
        features = make_training_data(5)
        
        with pytest.raises(ValueError, match="Insufficient training data"):
            model.fit(features)


class TestCreateModel:
    """Tests for model factory function."""
    
    def test_create_home_advantage(self):
        """Should create HomeAdvantageModel."""
        model = create_model("home_advantage")
        assert isinstance(model, HomeAdvantageModel)
    
    def test_create_multinomial_logistic(self):
        """Should create MultinomialLogisticModel."""
        model = create_model("multinomial_logistic")
        assert isinstance(model, MultinomialLogisticModel)
    
    def test_create_with_params(self):
        """Should pass parameters to model."""
        model = create_model(
            "multinomial_logistic",
            learning_rate=0.5,
            max_iterations=200,
            regularization=0.1,
        )
        
        assert isinstance(model, MultinomialLogisticModel)
        assert model.learning_rate == 0.5
        assert model.max_iterations == 200
        assert model.regularization == 0.1
    
    def test_unknown_model_raises(self):
        """Unknown model name should raise."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("unknown_model")


class TestBaseModel:
    """Tests for BaseModel interface."""
    
    def test_is_trained_property(self):
        """is_trained property should work."""
        model = HomeAdvantageModel()
        assert not model.is_trained
        
        features = make_training_data(10)
        model.fit(features)
        assert model.is_trained
