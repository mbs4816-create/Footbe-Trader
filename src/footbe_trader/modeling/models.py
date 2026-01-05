"""Baseline models for 3-way match outcome prediction.

Models:
- Model0: Home advantage only (constant probabilities)
- Model1: Multinomial logistic regression on simple features
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from footbe_trader.modeling.features import MatchFeatureVector


@dataclass
class PredictionOutput:
    """3-way prediction output."""
    
    fixture_id: int
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_outcome: str  # "H", "D", or "A"
    
    @property
    def probabilities(self) -> NDArray[np.float64]:
        """Return probabilities as array [H, D, A]."""
        return np.array([self.prob_home, self.prob_draw, self.prob_away])
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_id": self.fixture_id,
            "prob_home": self.prob_home,
            "prob_draw": self.prob_draw,
            "prob_away": self.prob_away,
            "predicted_outcome": self.predicted_outcome,
        }


class BaseModel:
    """Base class for prediction models."""
    
    name: str = "base"
    version: str = "1.0"
    
    def __init__(self):
        self._is_trained = False
    
    def fit(self, features: list[MatchFeatureVector]) -> "BaseModel":
        """Train the model.
        
        Args:
            features: Training feature vectors with outcomes.
            
        Returns:
            Self for chaining.
        """
        raise NotImplementedError
    
    def predict(self, features: list[MatchFeatureVector]) -> list[PredictionOutput]:
        """Generate predictions.
        
        Args:
            features: Feature vectors to predict.
            
        Returns:
            List of predictions.
        """
        raise NotImplementedError
    
    def predict_proba(self, features: list[MatchFeatureVector]) -> NDArray[np.float64]:
        """Get probability matrix.
        
        Args:
            features: Feature vectors to predict.
            
        Returns:
            Array of shape (n_samples, 3) with [P(H), P(D), P(A)].
        """
        predictions = self.predict(features)
        return np.array([p.probabilities for p in predictions])
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained


class HomeAdvantageModel(BaseModel):
    """Model0: Home advantage only multinomial.
    
    Uses historical home/draw/away frequencies as constant predictions.
    This is the simplest baseline - any useful model must beat this.
    """
    
    name = "home_advantage_only"
    version = "1.0"
    
    def __init__(self):
        super().__init__()
        # EPL historical averages (will be updated during fit)
        self.prob_home = 0.45
        self.prob_draw = 0.25
        self.prob_away = 0.30
    
    def fit(self, features: list[MatchFeatureVector]) -> "HomeAdvantageModel":
        """Fit by computing historical outcome frequencies.
        
        Args:
            features: Training data with outcomes.
            
        Returns:
            Self.
        """
        outcomes = [f.outcome for f in features if f.outcome is not None]
        
        if not outcomes:
            # Use default EPL averages
            self._is_trained = True
            return self
        
        n = len(outcomes)
        self.prob_home = sum(1 for o in outcomes if o == "H") / n
        self.prob_draw = sum(1 for o in outcomes if o == "D") / n
        self.prob_away = sum(1 for o in outcomes if o == "A") / n
        
        self._is_trained = True
        return self
    
    def predict(self, features: list[MatchFeatureVector]) -> list[PredictionOutput]:
        """Predict using constant probabilities.
        
        Args:
            features: Feature vectors (ignored except for fixture_id).
            
        Returns:
            Constant predictions.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Determine most likely outcome
        probs = [self.prob_home, self.prob_draw, self.prob_away]
        outcomes = ["H", "D", "A"]
        predicted = outcomes[np.argmax(probs)]
        
        return [
            PredictionOutput(
                fixture_id=f.fixture_id,
                prob_home=self.prob_home,
                prob_draw=self.prob_draw,
                prob_away=self.prob_away,
                predicted_outcome=predicted,
            )
            for f in features
        ]


class MultinomialLogisticModel(BaseModel):
    """Model1: Multinomial logistic regression on simple features.
    
    Features:
    - Home advantage constant
    - Rolling goals scored/conceded (home/away split)
    - Rest days differential
    """
    
    name = "multinomial_logistic"
    version = "1.0"
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iterations: int = 1000,
        regularization: float = 0.01,
        random_state: int = 42,
    ):
        """Initialize model.
        
        Args:
            learning_rate: Gradient descent learning rate.
            max_iterations: Maximum training iterations.
            regularization: L2 regularization strength.
            random_state: Random seed for reproducibility.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.random_state = random_state
        
        # Model weights: (n_features + 1, n_classes - 1) for multinomial
        # We use H and D as reference, A as baseline
        self.weights: NDArray[np.float64] | None = None
        
        # Feature normalization parameters
        self.feature_mean: NDArray[np.float64] | None = None
        self.feature_std: NDArray[np.float64] | None = None
    
    def _outcome_to_label(self, outcome: str) -> int:
        """Convert outcome to numeric label."""
        return {"H": 0, "D": 1, "A": 2}[outcome]
    
    def _label_to_outcome(self, label: int) -> str:
        """Convert numeric label to outcome."""
        return ["H", "D", "A"][label]
    
    def _softmax(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute softmax probabilities."""
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _normalize_features(
        self, X: NDArray[np.float64], fit: bool = False
    ) -> NDArray[np.float64]:
        """Normalize features to zero mean, unit variance.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            fit: If True, compute mean/std from data.
            
        Returns:
            Normalized features.
        """
        if fit:
            self.feature_mean = np.mean(X, axis=0)
            self.feature_std = np.std(X, axis=0)
            # Avoid division by zero
            self.feature_std[self.feature_std < 1e-8] = 1.0
        
        return (X - self.feature_mean) / self.feature_std
    
    def fit(self, features: list[MatchFeatureVector]) -> "MultinomialLogisticModel":
        """Train using gradient descent.
        
        Args:
            features: Training data with outcomes.
            
        Returns:
            Self.
        """
        # Filter to only finished matches with outcomes
        train_data = [f for f in features if f.outcome is not None]
        
        if len(train_data) < 10:
            raise ValueError(f"Insufficient training data: {len(train_data)} samples")
        
        # Build feature matrix
        X = np.array([f.to_feature_array() for f in train_data])
        y = np.array([self._outcome_to_label(f.outcome) for f in train_data])
        
        # Normalize features
        X = self._normalize_features(X, fit=True)
        
        # Add bias term
        X = np.column_stack([np.ones(len(X)), X])
        
        n_samples, n_features = X.shape
        n_classes = 3
        
        # Initialize weights
        np.random.seed(self.random_state)
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        
        # One-hot encode targets
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        
        # Gradient descent
        for _ in range(self.max_iterations):
            # Forward pass
            logits = X @ self.weights
            probs = self._softmax(logits)
            
            # Compute gradient
            gradient = X.T @ (probs - y_onehot) / n_samples
            
            # Add L2 regularization (don't regularize bias)
            reg_gradient = self.regularization * self.weights
            reg_gradient[0, :] = 0
            gradient += reg_gradient
            
            # Update weights
            self.weights -= self.learning_rate * gradient
        
        self._is_trained = True
        return self
    
    def predict(self, features: list[MatchFeatureVector]) -> list[PredictionOutput]:
        """Generate predictions.
        
        Args:
            features: Feature vectors.
            
        Returns:
            Predictions with probabilities.
        """
        if not self._is_trained or self.weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Build feature matrix
        X = np.array([f.to_feature_array() for f in features])
        X = self._normalize_features(X, fit=False)
        X = np.column_stack([np.ones(len(X)), X])
        
        # Forward pass
        logits = X @ self.weights
        probs = self._softmax(logits)
        
        # Generate predictions
        predictions = []
        for i, f in enumerate(features):
            prob_h, prob_d, prob_a = probs[i]
            predicted = self._label_to_outcome(np.argmax(probs[i]))
            
            predictions.append(
                PredictionOutput(
                    fixture_id=f.fixture_id,
                    prob_home=float(prob_h),
                    prob_draw=float(prob_d),
                    prob_away=float(prob_a),
                    predicted_outcome=predicted,
                )
            )
        
        return predictions


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Factory function to create models.
    
    Args:
        model_type: Model type string:
            - "home_advantage": Constant probability baseline
            - "multinomial_logistic": Logistic regression
            - "poisson": Time-decayed Poisson with Dixon-Coles
            - "neural_net": MLP with team embeddings
        **kwargs: Model-specific parameters.
        
    Returns:
        Model instance.
    """
    if model_type == "home_advantage":
        return HomeAdvantageModel()
    elif model_type == "multinomial_logistic":
        return MultinomialLogisticModel(**kwargs)
    elif model_type == "poisson":
        from footbe_trader.modeling.poisson_model import PoissonModel
        return PoissonModel(**kwargs)
    elif model_type == "neural_net":
        from footbe_trader.modeling.neural_model import NeuralNetModel
        return NeuralNetModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

