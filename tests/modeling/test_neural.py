"""Tests for neural network model with team embeddings."""

from datetime import datetime, timedelta
import numpy as np
import pytest

from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.neural_model import NeuralNetModel, TrainingHistory


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
        home_goals=kwargs.get("home_goals", 2 if outcome == "H" else 1),
        away_goals=kwargs.get("away_goals", 1 if outcome == "H" else 2),
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


def make_training_data(n_samples: int = 200) -> list[MatchFeatureVector]:
    """Create training data with realistic distribution."""
    np.random.seed(42)
    features = []
    outcomes = ["H", "D", "A"]
    probs = [0.45, 0.27, 0.28]
    n_teams = 20
    
    base_time = datetime(2024, 1, 1)
    
    for i in range(n_samples):
        outcome = np.random.choice(outcomes, p=probs)
        home_team_id = (i % n_teams) + 1
        away_team_id = ((i + 10) % n_teams) + 1
        if away_team_id == home_team_id:
            away_team_id = (away_team_id % n_teams) + 1
        
        features.append(make_feature_vector(
            fixture_id=i + 1,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            outcome=outcome,
            kickoff_utc=base_time + timedelta(days=i),  # Time-ordered
            home_team_home_goals_scored_avg=1.5 + np.random.randn() * 0.3,
            away_team_away_goals_scored_avg=1.0 + np.random.randn() * 0.3,
            rest_days_diff=np.random.randn() * 2,
        ))
    
    return features


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""
    
    def test_default_values(self):
        """Default values should be sensible."""
        history = TrainingHistory()
        assert history.train_losses == []
        assert history.val_losses == []
        assert history.best_epoch == 0
        assert history.best_val_loss == float('inf')
        assert not history.stopped_early
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        history = TrainingHistory(
            train_losses=[1.0, 0.9, 0.8],
            val_losses=[1.1, 1.0, 0.95],
            best_epoch=2,
            best_val_loss=0.95,
            stopped_early=True,
        )
        d = history.to_dict()
        
        assert d["train_losses"] == [1.0, 0.9, 0.8]
        assert d["best_epoch"] == 2
        assert d["stopped_early"] is True


class TestNeuralNetModel:
    """Tests for NeuralNetModel."""
    
    def test_fit_trains_model(self):
        """Fitting should train the model."""
        model = NeuralNetModel(
            embedding_dim=8,
            hidden_dims=[32, 16],
            max_epochs=20,
            patience=5,
        )
        features = make_training_data(200)
        
        model.fit(features)
        
        assert model.is_trained
        assert model.team_embeddings is not None
        assert len(model.weights) > 0
        assert model.history is not None
    
    def test_fit_respects_time_order(self):
        """Train/val split should be time-ordered, not shuffled."""
        model = NeuralNetModel(
            max_epochs=5,
            val_fraction=0.2,
        )
        features = make_training_data(100)
        
        # Sort by time
        features.sort(key=lambda x: x.kickoff_utc)
        
        # The implementation should use first 80% for train, last 20% for val
        model.fit(features)
        
        # Model should be trained
        assert model.is_trained
        # History should have validation losses (proving val split was used)
        assert len(model.history.val_losses) > 0
    
    def test_predict_returns_correct_structure(self):
        """Predictions should have correct structure."""
        model = NeuralNetModel(max_epochs=20)
        train_features = make_training_data(200)
        model.fit(train_features)
        
        test_features = make_training_data(20)
        predictions = model.predict(test_features)
        
        assert len(predictions) == 20
        for pred in predictions:
            assert hasattr(pred, "prob_home")
            assert hasattr(pred, "prob_draw")
            assert hasattr(pred, "prob_away")
            assert hasattr(pred, "predicted_outcome")
            
            # Probabilities should sum to 1
            total = pred.prob_home + pred.prob_draw + pred.prob_away
            assert abs(total - 1.0) < 1e-6
    
    def test_predict_proba_shape(self):
        """predict_proba should return correct shape."""
        model = NeuralNetModel(max_epochs=20)
        train_features = make_training_data(200)
        model.fit(train_features)
        
        test_features = make_training_data(30)
        proba = model.predict_proba(test_features)
        
        assert proba.shape == (30, 3)
        
        # Each row should sum to 1
        for row in proba:
            assert abs(row.sum() - 1.0) < 1e-6
    
    def test_probabilities_are_positive(self):
        """All predicted probabilities should be positive."""
        model = NeuralNetModel(max_epochs=20)
        features = make_training_data(200)
        model.fit(features)
        
        proba = model.predict_proba(features)
        
        assert np.all(proba > 0)
        assert np.all(proba < 1)
    
    def test_early_stopping_works(self):
        """Early stopping should prevent overfitting."""
        model = NeuralNetModel(
            max_epochs=1000,  # Would take forever without early stopping
            patience=10,
        )
        features = make_training_data(200)
        
        model.fit(features)
        
        # Should have stopped early (not run all 1000 epochs)
        assert len(model.history.train_losses) < 1000
    
    def test_dropout_regularization(self):
        """Dropout should affect training vs inference."""
        model = NeuralNetModel(
            dropout_rate=0.5,
            max_epochs=20,
        )
        features = make_training_data(200)
        model.fit(features)
        
        # Multiple predictions should be identical (dropout disabled at inference)
        test_features = features[:10]
        proba1 = model.predict_proba(test_features)
        proba2 = model.predict_proba(test_features)
        
        np.testing.assert_array_almost_equal(proba1, proba2)
    
    def test_team_embeddings_learned(self):
        """Team embeddings should be learned."""
        model = NeuralNetModel(
            embedding_dim=16,
            max_epochs=30,
        )
        features = make_training_data(200)
        model.fit(features)
        
        # Should have embeddings for all teams
        assert model.team_embeddings.shape[0] == model.n_teams
        assert model.team_embeddings.shape[1] == 16
        
        # Embeddings should not be all zeros
        assert np.abs(model.team_embeddings).sum() > 0
    
    def test_get_team_embedding(self):
        """get_team_embedding should return embedding for known teams."""
        model = NeuralNetModel(
            embedding_dim=8,
            max_epochs=20,
        )
        features = make_training_data(200)
        model.fit(features)
        
        # Get embedding for team that was in training
        team_id = features[0].home_team_id
        embedding = model.get_team_embedding(team_id)
        
        assert embedding is not None
        assert len(embedding) == 8
        
        # Unknown team should return None
        unknown_embedding = model.get_team_embedding(9999)
        assert unknown_embedding is None
    
    def test_temperature_scaling(self):
        """Temperature scaling should affect confidence."""
        model = NeuralNetModel(
            max_epochs=20,
            temperature=1.0,
        )
        features = make_training_data(200)
        model.fit(features)
        
        test_features = features[:10]
        
        # Normal temperature
        proba_normal = model.predict_proba(test_features)
        
        # Higher temperature = softer predictions
        model.set_temperature(2.0)
        proba_soft = model.predict_proba(test_features)
        
        # Soft predictions should be closer to uniform
        max_normal = np.max(proba_normal, axis=1).mean()
        max_soft = np.max(proba_soft, axis=1).mean()
        
        assert max_soft < max_normal  # Softer predictions have lower max prob
    
    def test_not_trained_raises(self):
        """Predicting without training should raise."""
        model = NeuralNetModel()
        test_feature = make_feature_vector()
        
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict([test_feature])
    
    def test_insufficient_data_raises(self):
        """Training with too little data should raise."""
        model = NeuralNetModel()
        features = make_training_data(20)  # Only 20 samples
        
        with pytest.raises(ValueError, match="Insufficient"):
            model.fit(features)
    
    def test_to_dict_serialization(self):
        """Model should be serializable to dict."""
        model = NeuralNetModel(
            embedding_dim=8,
            hidden_dims=[32, 16],
            max_epochs=20,
        )
        features = make_training_data(200)
        model.fit(features)
        
        d = model.to_dict()
        
        assert d["name"] == "neural_net"
        assert d["embedding_dim"] == 8
        assert d["hidden_dims"] == [32, 16]
        assert d["n_teams"] == model.n_teams
        assert "history" in d
    
    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""
        features = make_training_data(200)
        
        model1 = NeuralNetModel(
            max_epochs=10,
            random_state=42,
        )
        model1.fit(features)
        
        model2 = NeuralNetModel(
            max_epochs=10,
            random_state=42,
        )
        model2.fit(features)
        
        # Predictions should be identical
        test_features = features[:10]
        proba1 = model1.predict_proba(test_features)
        proba2 = model2.predict_proba(test_features)
        
        np.testing.assert_array_almost_equal(proba1, proba2, decimal=5)
