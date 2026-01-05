"""Evaluation metrics for 3-way match prediction.

Metrics:
- Log loss (cross-entropy) - primary metric
- Brier score (multi-class)
- Calibration curves per class
- Bootstrap confidence intervals
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    
    log_loss: float
    brier_score: float
    accuracy: float
    n_samples: int
    
    # Per-class metrics
    class_accuracies: dict[str, float] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    
    # Calibration data (for plotting)
    calibration_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            "class_accuracies": self.class_accuracies,
            "class_counts": self.class_counts,
        }


@dataclass
class BootstrapResult:
    """Bootstrap comparison result."""
    
    mean_diff: float
    std_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float  # Approximate p-value for H0: diff = 0
    n_bootstrap: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_diff": self.mean_diff,
            "std_diff": self.std_diff,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "n_bootstrap": self.n_bootstrap,
        }


def compute_log_loss(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    eps: float = 1e-15,
) -> float:
    """Compute multi-class log loss (cross-entropy).
    
    Args:
        y_true: True labels (0=H, 1=D, 2=A).
        y_proba: Predicted probabilities, shape (n_samples, 3).
        eps: Clipping value to avoid log(0).
        
    Returns:
        Mean log loss.
    """
    y_proba = np.clip(y_proba, eps, 1 - eps)
    n_samples = len(y_true)
    
    # Select probability of true class
    log_probs = np.log(y_proba[np.arange(n_samples), y_true])
    
    return -np.mean(log_probs)


def compute_brier_score(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
) -> float:
    """Compute multi-class Brier score.
    
    Brier score is mean squared error between predicted probabilities
    and one-hot encoded true labels.
    
    Args:
        y_true: True labels (0=H, 1=D, 2=A).
        y_proba: Predicted probabilities, shape (n_samples, 3).
        
    Returns:
        Mean Brier score (lower is better).
    """
    n_samples = len(y_true)
    n_classes = y_proba.shape[1]
    
    # One-hot encode true labels
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_true] = 1
    
    # Mean squared error
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))


def compute_accuracy(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
) -> float:
    """Compute prediction accuracy.
    
    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        
    Returns:
        Accuracy (0-1).
    """
    y_pred = np.argmax(y_proba, axis=1)
    return np.mean(y_pred == y_true)


def compute_calibration_data(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """Compute calibration curve data for each class.
    
    Args:
        y_true: True labels (0=H, 1=D, 2=A).
        y_proba: Predicted probabilities.
        n_bins: Number of probability bins.
        
    Returns:
        Dict with calibration data per class.
    """
    classes = {0: "H", 1: "D", 2: "A"}
    calibration = {}
    
    for class_idx, class_name in classes.items():
        # Binary for this class
        y_binary = (y_true == class_idx).astype(int)
        probs = y_proba[:, class_idx]
        
        # Bin by predicted probability
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_true_freqs = []
        bin_counts = []
        
        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= low) & (probs < high)
            
            if np.sum(mask) > 0:
                bin_centers.append((low + high) / 2)
                bin_true_freqs.append(np.mean(y_binary[mask]))
                bin_counts.append(int(np.sum(mask)))
        
        calibration[class_name] = {
            "bin_centers": bin_centers,
            "bin_true_freqs": bin_true_freqs,
            "bin_counts": bin_counts,
        }
    
    return calibration


def evaluate_predictions(
    y_true: NDArray[np.int64],
    y_proba: NDArray[np.float64],
    n_calibration_bins: int = 10,
) -> MetricsResult:
    """Compute all evaluation metrics.
    
    Args:
        y_true: True labels (0=H, 1=D, 2=A).
        y_proba: Predicted probabilities, shape (n_samples, 3).
        n_calibration_bins: Bins for calibration curves.
        
    Returns:
        MetricsResult with all metrics.
    """
    log_loss = compute_log_loss(y_true, y_proba)
    brier = compute_brier_score(y_true, y_proba)
    accuracy = compute_accuracy(y_true, y_proba)
    calibration = compute_calibration_data(y_true, y_proba, n_calibration_bins)
    
    # Per-class metrics
    classes = {0: "H", 1: "D", 2: "A"}
    class_accuracies = {}
    class_counts = {}
    
    for class_idx, class_name in classes.items():
        mask = y_true == class_idx
        class_counts[class_name] = int(np.sum(mask))
        
        if class_counts[class_name] > 0:
            y_pred = np.argmax(y_proba, axis=1)
            class_accuracies[class_name] = float(np.mean(y_pred[mask] == class_idx))
        else:
            class_accuracies[class_name] = 0.0
    
    return MetricsResult(
        log_loss=log_loss,
        brier_score=brier,
        accuracy=accuracy,
        n_samples=len(y_true),
        class_accuracies=class_accuracies,
        class_counts=class_counts,
        calibration_data=calibration,
    )


def bootstrap_compare(
    y_true: NDArray[np.int64],
    y_proba_model1: NDArray[np.float64],
    y_proba_model0: NDArray[np.float64],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> BootstrapResult:
    """Paired bootstrap comparison of two models on log loss.
    
    Tests H0: Model1 log loss = Model0 log loss.
    Negative difference means Model1 is better.
    
    Args:
        y_true: True labels.
        y_proba_model1: Model1 probabilities.
        y_proba_model0: Model0 (baseline) probabilities.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level for intervals.
        random_state: Random seed.
        
    Returns:
        BootstrapResult with confidence intervals.
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # Compute per-sample log loss differences
    eps = 1e-15
    y_proba_model1 = np.clip(y_proba_model1, eps, 1 - eps)
    y_proba_model0 = np.clip(y_proba_model0, eps, 1 - eps)
    
    ll_model1 = -np.log(y_proba_model1[np.arange(n_samples), y_true])
    ll_model0 = -np.log(y_proba_model0[np.arange(n_samples), y_true])
    
    # Per-sample difference (negative = model1 better)
    diff = ll_model1 - ll_model0
    
    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_diffs.append(np.mean(diff[idx]))
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Statistics
    mean_diff = float(np.mean(bootstrap_diffs))
    std_diff = float(np.std(bootstrap_diffs))
    
    # Confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2)))
    
    # Approximate p-value (two-sided test)
    p_value = float(2 * min(
        np.mean(bootstrap_diffs <= 0),
        np.mean(bootstrap_diffs >= 0),
    ))
    
    return BootstrapResult(
        mean_diff=mean_diff,
        std_diff=std_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_bootstrap=n_bootstrap,
    )


def outcome_to_label(outcomes: list[str]) -> NDArray[np.int64]:
    """Convert outcome strings to numeric labels.
    
    Args:
        outcomes: List of "H", "D", "A".
        
    Returns:
        Array of labels (0, 1, 2).
    """
    mapping = {"H": 0, "D": 1, "A": 2}
    return np.array([mapping[o] for o in outcomes], dtype=np.int64)
