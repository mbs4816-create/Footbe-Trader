"""Probability calibration via temperature scaling.

This module implements temperature scaling for neural network calibration:
- Learn optimal temperature T on validation set
- Apply P_calibrated = softmax(logits / T)
- Minimize negative log-likelihood on held-out data

Temperature scaling preserves accuracy while improving calibration.
T > 1 produces softer probabilities (less confident)
T < 1 produces sharper probabilities (more confident)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


@dataclass
class CalibrationResult:
    """Result of temperature scaling calibration."""
    
    temperature: float
    pre_calibration_ece: float
    post_calibration_ece: float
    pre_calibration_nll: float
    post_calibration_nll: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "pre_calibration_ece": self.pre_calibration_ece,
            "post_calibration_ece": self.post_calibration_ece,
            "pre_calibration_nll": self.pre_calibration_nll,
            "post_calibration_nll": self.post_calibration_nll,
        }


def compute_ece(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.
    
    ECE measures the difference between predicted probabilities and
    actual accuracy across probability bins.
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes).
        y_true: True class labels (n_samples,).
        n_bins: Number of bins for calibration.
        
    Returns:
        Expected calibration error.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    ece = 0.0
    for bin_lower in np.linspace(0, 1 - 1/n_bins, n_bins):
        bin_upper = bin_lower + 1/n_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            bin_weight = np.sum(in_bin) / len(y_true)
            ece += bin_weight * abs(avg_accuracy - avg_confidence)
    
    return ece


def compute_mce(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error.
    
    MCE is the maximum gap between confidence and accuracy across bins.
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes).
        y_true: True class labels (n_samples,).
        n_bins: Number of bins for calibration.
        
    Returns:
        Maximum calibration error.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    mce = 0.0
    for bin_lower in np.linspace(0, 1 - 1/n_bins, n_bins):
        bin_upper = bin_lower + 1/n_bins
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            mce = max(mce, abs(avg_accuracy - avg_confidence))
    
    return mce


def _softmax_with_temperature(
    logits: NDArray[np.float64], temperature: float
) -> NDArray[np.float64]:
    """Apply softmax with temperature scaling."""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def _negative_log_likelihood(
    temperature: float,
    logits: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> float:
    """Compute NLL for temperature scaling optimization."""
    probs = _softmax_with_temperature(logits, temperature)
    eps = 1e-10
    
    # Log probability of true class
    log_probs = np.log(probs[np.arange(len(y_true)), y_true] + eps)
    return -np.mean(log_probs)


def find_optimal_temperature(
    logits: NDArray[np.float64],
    y_true: NDArray[np.int64],
    init_temp: float = 1.5,
) -> float:
    """Find optimal temperature via grid search + refinement.
    
    Args:
        logits: Raw model outputs before softmax (n_samples, n_classes).
        y_true: True class labels (n_samples,).
        init_temp: Initial temperature guess.
        
    Returns:
        Optimal temperature value.
    """
    # Grid search
    temps = np.linspace(0.5, 3.0, 26)
    best_temp = init_temp
    best_nll = float('inf')
    
    for t in temps:
        nll = _negative_log_likelihood(t, logits, y_true)
        if nll < best_nll:
            best_nll = nll
            best_temp = t
    
    # Refine with scipy
    result = optimize.minimize_scalar(
        lambda t: _negative_log_likelihood(t, logits, y_true),
        bounds=(0.1, 5.0),
        method='bounded',
    )
    
    return float(result.x) if result.success else best_temp


def calibrate_probabilities(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> tuple[NDArray[np.float64], float]:
    """Calibrate probabilities using temperature scaling.
    
    Note: This assumes probabilities are already softmax outputs.
    We need to convert back to logits, then apply temperature scaling.
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes).
        y_true: True class labels for calibration.
        
    Returns:
        Tuple of (calibrated_probs, optimal_temperature).
    """
    # Convert probabilities to logits (inverse softmax)
    eps = 1e-10
    logits = np.log(probs + eps)
    
    # Find optimal temperature
    temperature = find_optimal_temperature(logits, y_true)
    
    # Apply temperature scaling
    calibrated = _softmax_with_temperature(logits, temperature)
    
    return calibrated, temperature


def apply_temperature_scaling(
    probs: NDArray[np.float64],
    temperature: float,
) -> NDArray[np.float64]:
    """Apply pre-computed temperature to probabilities.
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes).
        temperature: Temperature value to apply.
        
    Returns:
        Calibrated probabilities.
    """
    eps = 1e-10
    logits = np.log(probs + eps)
    return _softmax_with_temperature(logits, temperature)


class TemperatureScaler:
    """Temperature scaling calibrator for probability outputs.
    
    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_probs, val_labels)
        calibrated = scaler.transform(test_probs)
    """
    
    def __init__(self, init_temperature: float = 1.5):
        """Initialize temperature scaler.
        
        Args:
            init_temperature: Initial temperature for optimization.
        """
        self.init_temperature = init_temperature
        self.temperature: float = 1.0
        self._is_fitted = False
        self._result: CalibrationResult | None = None
    
    def fit(
        self,
        probs: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> "TemperatureScaler":
        """Fit temperature on validation data.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes).
            y_true: True class labels.
            
        Returns:
            Self.
        """
        # Compute pre-calibration metrics
        pre_ece = compute_ece(probs, y_true)
        pre_nll = _negative_log_likelihood(
            1.0, np.log(probs + 1e-10), y_true
        )
        
        # Find optimal temperature
        calibrated, self.temperature = calibrate_probabilities(probs, y_true)
        
        # Compute post-calibration metrics
        post_ece = compute_ece(calibrated, y_true)
        post_nll = _negative_log_likelihood(
            self.temperature, np.log(probs + 1e-10), y_true
        )
        
        self._result = CalibrationResult(
            temperature=self.temperature,
            pre_calibration_ece=pre_ece,
            post_calibration_ece=post_ece,
            pre_calibration_nll=pre_nll,
            post_calibration_nll=post_nll,
        )
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        probs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply temperature scaling to probabilities.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes).
            
        Returns:
            Calibrated probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return apply_temperature_scaling(probs, self.temperature)
    
    def fit_transform(
        self,
        probs: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Fit and transform in one call.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes).
            y_true: True class labels.
            
        Returns:
            Calibrated probabilities.
        """
        self.fit(probs, y_true)
        return self.transform(probs)
    
    @property
    def result(self) -> CalibrationResult | None:
        """Get calibration result."""
        return self._result
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature": self.temperature,
            "is_fitted": self._is_fitted,
            "result": self._result.to_dict() if self._result else None,
        }


def compute_reliability_diagram_data(
    probs: NDArray[np.float64],
    y_true: NDArray[np.int64],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute data for reliability diagram.
    
    Args:
        probs: Predicted probabilities (n_samples, n_classes).
        y_true: True class labels (n_samples,).
        n_bins: Number of bins.
        
    Returns:
        Dictionary with bin centers, accuracies, confidences, and counts.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        bin_centers.append((bin_lower + bin_upper) / 2)
        
        if np.sum(in_bin) > 0:
            bin_accuracies.append(float(np.mean(accuracies[in_bin])))
            bin_confidences.append(float(np.mean(confidences[in_bin])))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": compute_ece(probs, y_true, n_bins),
        "mce": compute_mce(probs, y_true, n_bins),
    }
