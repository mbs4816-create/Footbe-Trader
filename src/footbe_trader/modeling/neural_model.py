"""Neural network model with team embeddings for match outcome prediction.

This module implements an MLP classifier with:
- Team ID embeddings (learned representations for home/away teams)
- Numeric features from the feature pipeline
- Regularization: dropout, weight decay, early stopping
- Time-respecting train/validation splits
- Temperature scaling for calibration

The model is trained using mini-batch gradient descent with Adam-style updates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.models import BaseModel, PredictionOutput


@dataclass
class TrainingHistory:
    """Training history for neural network."""
    
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    stopped_early: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "stopped_early": self.stopped_early,
        }


class NeuralNetModel(BaseModel):
    """MLP with team embeddings for match outcome prediction.
    
    Architecture:
    - Team embedding layer: team_id -> embedding_dim
    - Concatenate: [home_embed, away_embed, numeric_features]
    - Hidden layers with ReLU activation
    - Dropout for regularization
    - Output: softmax over 3 classes (H, D, A)
    
    Training:
    - Mini-batch gradient descent with momentum
    - L2 weight decay
    - Early stopping on validation loss
    - Time-ordered train/val split (no shuffle leakage)
    """
    
    name = "neural_net"
    version = "1.0"
    
    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
        weight_decay: float = 0.01,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 20,
        val_fraction: float = 0.2,
        random_state: int = 42,
        temperature: float = 1.0,
    ):
        """Initialize neural network model.
        
        Args:
            embedding_dim: Dimension of team embeddings.
            hidden_dims: List of hidden layer sizes.
            dropout_rate: Dropout probability.
            weight_decay: L2 regularization coefficient.
            learning_rate: Learning rate for gradient descent.
            momentum: Momentum coefficient.
            batch_size: Mini-batch size.
            max_epochs: Maximum training epochs.
            patience: Early stopping patience.
            val_fraction: Fraction of data for validation (time-ordered split).
            random_state: Random seed for reproducibility.
            temperature: Temperature for calibration (learned or set).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.temperature = temperature
        
        # Model parameters (initialized during fit)
        self.team_embeddings: NDArray[np.float64] | None = None
        self.weights: list[NDArray[np.float64]] = []
        self.biases: list[NDArray[np.float64]] = []
        
        # Team ID mapping
        self.team_to_idx: dict[int, int] = {}
        self.n_teams: int = 0
        
        # Feature normalization
        self.feature_mean: NDArray[np.float64] | None = None
        self.feature_std: NDArray[np.float64] | None = None
        
        # Training history
        self.history: TrainingHistory | None = None
    
    def _init_weights(self, input_dim: int, output_dim: int) -> tuple[NDArray, NDArray]:
        """Initialize weights using He initialization."""
        std = np.sqrt(2.0 / input_dim)
        W = np.random.randn(input_dim, output_dim) * std
        b = np.zeros(output_dim)
        return W, b
    
    def _relu(self, x: NDArray) -> NDArray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def _relu_grad(self, x: NDArray) -> NDArray:
        """ReLU gradient."""
        return (x > 0).astype(np.float64)
    
    def _softmax(self, x: NDArray) -> NDArray:
        """Softmax with temperature scaling."""
        x_scaled = x / self.temperature
        exp_x = np.exp(x_scaled - np.max(x_scaled, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _dropout(self, x: NDArray, training: bool = True) -> tuple[NDArray, NDArray]:
        """Apply dropout during training."""
        if not training or self.dropout_rate == 0:
            return x, np.ones_like(x)
        
        mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float64)
        return x * mask / (1 - self.dropout_rate), mask
    
    def _forward(
        self,
        home_ids: NDArray[np.int64],
        away_ids: NDArray[np.int64],
        features: NDArray[np.float64],
        training: bool = False,
    ) -> tuple[NDArray[np.float64], list[Any]]:
        """Forward pass through network.
        
        Args:
            home_ids: Home team indices
            away_ids: Away team indices
            features: Numeric features (batch_size, n_features)
            training: Whether to apply dropout
            
        Returns:
            Tuple of (output probabilities, cache for backprop)
        """
        cache = []
        
        # Get team embeddings
        home_embed = self.team_embeddings[home_ids]  # (batch, embed_dim)
        away_embed = self.team_embeddings[away_ids]  # (batch, embed_dim)
        
        # Concatenate: [home_embed, away_embed, features]
        x = np.concatenate([home_embed, away_embed, features], axis=1)
        cache.append(('input', home_ids, away_ids, features))
        
        # Hidden layers
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = x @ W + b
            a = self._relu(z)
            a_drop, mask = self._dropout(a, training)
            cache.append(('hidden', x, z, mask))
            x = a_drop
        
        # Output layer (no activation, softmax applied separately)
        z_out = x @ self.weights[-1] + self.biases[-1]
        probs = self._softmax(z_out)
        cache.append(('output', x, z_out))
        
        return probs, cache
    
    def _backward(
        self,
        probs: NDArray[np.float64],
        y_onehot: NDArray[np.float64],
        cache: list[Any],
    ) -> tuple[list[NDArray], list[NDArray], NDArray]:
        """Backward pass to compute gradients.
        
        Args:
            probs: Predicted probabilities
            y_onehot: One-hot encoded targets
            cache: Forward pass cache
            
        Returns:
            Tuple of (weight_grads, bias_grads, embedding_grad)
        """
        batch_size = probs.shape[0]
        
        # Output layer gradient
        dz_out = (probs - y_onehot) / batch_size
        
        _, x_out, _ = cache[-1]
        dW_out = x_out.T @ dz_out
        db_out = np.sum(dz_out, axis=0)
        
        weight_grads = [dW_out]
        bias_grads = [db_out]
        
        # Backprop through output layer
        da = dz_out @ self.weights[-1].T
        
        # Hidden layers (reverse order)
        for i in range(len(self.weights) - 2, -1, -1):
            layer_type, x_in, z, mask = cache[i + 1]
            
            # Dropout gradient
            da = da * mask / (1 - self.dropout_rate) if self.dropout_rate > 0 else da
            
            # ReLU gradient
            dz = da * self._relu_grad(z)
            
            # Weight gradients
            dW = x_in.T @ dz
            db = np.sum(dz, axis=0)
            
            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)
            
            # Gradient for next layer
            da = dz @ self.weights[i].T
        
        # Embedding gradient (from concatenated input)
        _, home_ids, away_ids, _ = cache[0]
        embed_dim = self.embedding_dim
        
        da_home = da[:, :embed_dim]
        da_away = da[:, embed_dim:2*embed_dim]
        
        # Accumulate embedding gradients
        embed_grad = np.zeros_like(self.team_embeddings)
        np.add.at(embed_grad, home_ids, da_home)
        np.add.at(embed_grad, away_ids, da_away)
        
        return weight_grads, bias_grads, embed_grad
    
    def _cross_entropy_loss(
        self, probs: NDArray[np.float64], y_onehot: NDArray[np.float64]
    ) -> float:
        """Compute cross-entropy loss."""
        eps = 1e-10
        return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))
    
    def _prepare_data(
        self, features: list[MatchFeatureVector]
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Prepare data for training/prediction.
        
        Returns:
            Tuple of (home_ids, away_ids, numeric_features, labels)
        """
        home_ids = np.array([self.team_to_idx.get(f.home_team_id, 0) for f in features])
        away_ids = np.array([self.team_to_idx.get(f.away_team_id, 0) for f in features])
        
        numeric = np.array([f.to_feature_array() for f in features])
        
        # Normalize
        if self.feature_mean is not None:
            numeric = (numeric - self.feature_mean) / self.feature_std
        
        # Labels
        label_map = {"H": 0, "D": 1, "A": 2}
        labels = np.array([label_map.get(f.outcome, 0) for f in features])
        
        return home_ids, away_ids, numeric, labels
    
    def fit(self, features: list[MatchFeatureVector]) -> "NeuralNetModel":
        """Train the neural network.
        
        Args:
            features: Training data (time-ordered).
            
        Returns:
            Self.
        """
        np.random.seed(self.random_state)
        
        # Filter to completed matches
        train_data = [f for f in features if f.outcome is not None]
        
        if len(train_data) < 50:
            raise ValueError(f"Insufficient training data: {len(train_data)} samples")
        
        # Sort by time for proper train/val split
        train_data.sort(key=lambda x: x.kickoff_utc)
        
        # Time-ordered train/val split (no shuffle!)
        split_idx = int(len(train_data) * (1 - self.val_fraction))
        train_set = train_data[:split_idx]
        val_set = train_data[split_idx:]
        
        # Build team vocabulary
        all_teams = set()
        for f in train_data:
            all_teams.add(f.home_team_id)
            all_teams.add(f.away_team_id)
        
        self.team_to_idx = {tid: i for i, tid in enumerate(sorted(all_teams))}
        self.n_teams = len(self.team_to_idx)
        
        # Compute feature normalization from training set only
        train_numeric = np.array([f.to_feature_array() for f in train_set])
        self.feature_mean = np.mean(train_numeric, axis=0)
        self.feature_std = np.std(train_numeric, axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        
        # Initialize parameters
        self.team_embeddings = np.random.randn(self.n_teams, self.embedding_dim) * 0.1
        
        n_numeric = train_numeric.shape[1]
        input_dim = 2 * self.embedding_dim + n_numeric
        
        self.weights = []
        self.biases = []
        
        # Hidden layers
        dims = [input_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            W, b = self._init_weights(dims[i], dims[i + 1])
            self.weights.append(W)
            self.biases.append(b)
        
        # Output layer
        W_out, b_out = self._init_weights(dims[-1], 3)
        self.weights.append(W_out)
        self.biases.append(b_out)
        
        # Prepare data
        train_home, train_away, train_X, train_y = self._prepare_data(train_set)
        val_home, val_away, val_X, val_y = self._prepare_data(val_set)
        
        train_y_onehot = np.eye(3)[train_y]
        val_y_onehot = np.eye(3)[val_y]
        
        # Initialize momentum buffers
        embed_v = np.zeros_like(self.team_embeddings)
        weight_v = [np.zeros_like(W) for W in self.weights]
        bias_v = [np.zeros_like(b) for b in self.biases]
        
        # Training loop
        self.history = TrainingHistory()
        best_weights = None
        best_biases = None
        best_embeddings = None
        
        n_train = len(train_set)
        n_batches = max(1, n_train // self.batch_size)
        
        for epoch in range(self.max_epochs):
            # Shuffle training data (but keep val fixed!)
            perm = np.random.permutation(n_train)
            
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_train)
                idx = perm[start:end]
                
                batch_home = train_home[idx]
                batch_away = train_away[idx]
                batch_X = train_X[idx]
                batch_y = train_y_onehot[idx]
                
                # Forward pass
                probs, cache = self._forward(batch_home, batch_away, batch_X, training=True)
                loss = self._cross_entropy_loss(probs, batch_y)
                epoch_loss += loss
                
                # Backward pass
                weight_grads, bias_grads, embed_grad = self._backward(probs, batch_y, cache)
                
                # Add weight decay
                for i, W in enumerate(self.weights):
                    weight_grads[i] += self.weight_decay * W
                embed_grad += self.weight_decay * self.team_embeddings
                
                # Update with momentum
                embed_v = self.momentum * embed_v - self.learning_rate * embed_grad
                self.team_embeddings += embed_v
                
                for i in range(len(self.weights)):
                    weight_v[i] = self.momentum * weight_v[i] - self.learning_rate * weight_grads[i]
                    bias_v[i] = self.momentum * bias_v[i] - self.learning_rate * bias_grads[i]
                    self.weights[i] += weight_v[i]
                    self.biases[i] += bias_v[i]
            
            # Compute validation loss
            val_probs, _ = self._forward(val_home, val_away, val_X, training=False)
            val_loss = self._cross_entropy_loss(val_probs, val_y_onehot)
            
            train_loss = epoch_loss / n_batches
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < self.history.best_val_loss:
                self.history.best_val_loss = val_loss
                self.history.best_epoch = epoch
                best_weights = [W.copy() for W in self.weights]
                best_biases = [b.copy() for b in self.biases]
                best_embeddings = self.team_embeddings.copy()
            
            # Early stopping
            if epoch - self.history.best_epoch >= self.patience:
                self.history.stopped_early = True
                break
        
        # Restore best weights
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            self.team_embeddings = best_embeddings
        
        self._is_trained = True
        return self
    
    def predict(self, features: list[MatchFeatureVector]) -> list[PredictionOutput]:
        """Generate predictions.
        
        Args:
            features: Feature vectors.
            
        Returns:
            Predictions with probabilities.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        home_ids, away_ids, X, _ = self._prepare_data(features)
        probs, _ = self._forward(home_ids, away_ids, X, training=False)
        
        predictions = []
        outcomes = ["H", "D", "A"]
        
        for i, f in enumerate(features):
            pred_idx = np.argmax(probs[i])
            predictions.append(
                PredictionOutput(
                    fixture_id=f.fixture_id,
                    prob_home=float(probs[i, 0]),
                    prob_draw=float(probs[i, 1]),
                    prob_away=float(probs[i, 2]),
                    predicted_outcome=outcomes[pred_idx],
                )
            )
        
        return predictions
    
    def predict_proba(self, features: list[MatchFeatureVector]) -> NDArray[np.float64]:
        """Get probability matrix.
        
        Args:
            features: Feature vectors.
            
        Returns:
            Array of shape (n_samples, 3) with [P(H), P(D), P(A)].
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        home_ids, away_ids, X, _ = self._prepare_data(features)
        probs, _ = self._forward(home_ids, away_ids, X, training=False)
        return probs
    
    def set_temperature(self, temperature: float) -> None:
        """Set calibration temperature.
        
        Args:
            temperature: New temperature value (>1 = smoother, <1 = sharper).
        """
        self.temperature = temperature
    
    def get_team_embedding(self, team_id: int) -> NDArray[np.float64] | None:
        """Get the learned embedding for a team."""
        if team_id not in self.team_to_idx:
            return None
        idx = self.team_to_idx[team_id]
        return self.team_embeddings[idx].copy()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize model parameters."""
        return {
            "name": self.name,
            "version": self.version,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "temperature": self.temperature,
            "n_teams": self.n_teams,
            "team_to_idx": self.team_to_idx,
            "history": self.history.to_dict() if self.history else None,
        }
