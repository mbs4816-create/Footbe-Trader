"""Tests for evaluation metrics."""

import numpy as np
import pytest

from footbe_trader.modeling.metrics import (
    BootstrapResult,
    MetricsResult,
    bootstrap_compare,
    compute_accuracy,
    compute_brier_score,
    compute_calibration_data,
    compute_log_loss,
    evaluate_predictions,
    outcome_to_label,
)


class TestOutcomeToLabel:
    """Tests for outcome_to_label conversion."""
    
    def test_basic_conversion(self):
        """Test H/D/A to 0/1/2 conversion."""
        outcomes = ["H", "D", "A", "H", "A"]
        labels = outcome_to_label(outcomes)
        
        assert list(labels) == [0, 1, 2, 0, 2]
        assert labels.dtype == np.int64
    
    def test_all_home(self):
        """Test all home wins."""
        labels = outcome_to_label(["H", "H", "H"])
        assert list(labels) == [0, 0, 0]
    
    def test_all_draw(self):
        """Test all draws."""
        labels = outcome_to_label(["D", "D", "D"])
        assert list(labels) == [1, 1, 1]
    
    def test_all_away(self):
        """Test all away wins."""
        labels = outcome_to_label(["A", "A", "A"])
        assert list(labels) == [2, 2, 2]


class TestLogLoss:
    """Tests for log loss computation."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should have very low log loss."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.999, 0.0005, 0.0005],  # Confident home
            [0.0005, 0.999, 0.0005],  # Confident draw
            [0.0005, 0.0005, 0.999],  # Confident away
        ])
        
        loss = compute_log_loss(y_true, y_proba)
        assert loss < 0.01  # Very low loss
    
    def test_uniform_predictions(self):
        """Uniform predictions should have log loss of log(3)."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
        ])
        
        loss = compute_log_loss(y_true, y_proba)
        expected = np.log(3)  # ~1.0986
        assert abs(loss - expected) < 0.01
    
    def test_wrong_predictions_high_loss(self):
        """Confidently wrong predictions should have high log loss."""
        y_true = np.array([0])  # True class is 0
        y_proba = np.array([[0.001, 0.001, 0.998]])  # Predicts 2
        
        loss = compute_log_loss(y_true, y_proba)
        assert loss > 5  # Very high loss
    
    def test_clipping_prevents_infinity(self):
        """Extreme probabilities should be clipped."""
        y_true = np.array([0])
        y_proba = np.array([[0.0, 0.5, 0.5]])  # Zero probability for true class
        
        loss = compute_log_loss(y_true, y_proba)
        assert np.isfinite(loss)  # Should not be infinite


class TestBrierScore:
    """Tests for Brier score computation."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score of 0."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        score = compute_brier_score(y_true, y_proba)
        assert abs(score) < 1e-10
    
    def test_uniform_predictions(self):
        """Test Brier score for uniform predictions."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
        ])
        
        # For uniform, MSE = (1-1/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3
        score = compute_brier_score(y_true, y_proba)
        expected = 2/3
        assert abs(score - expected) < 0.01
    
    def test_worst_predictions(self):
        """Completely wrong predictions should have max Brier score."""
        y_true = np.array([0])  # True is home
        y_proba = np.array([[0.0, 0.0, 1.0]])  # Predict away
        
        # MSE = (0-1)^2 + (0-0)^2 + (1-0)^2 = 1 + 0 + 1 = 2
        score = compute_brier_score(y_true, y_proba)
        assert abs(score - 2.0) < 0.01


class TestAccuracy:
    """Tests for accuracy computation."""
    
    def test_perfect_accuracy(self):
        """All correct predictions should give 1.0 accuracy."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [0.8, 0.1, 0.1],  # Predicts 0
            [0.1, 0.8, 0.1],  # Predicts 1
            [0.1, 0.1, 0.8],  # Predicts 2
            [0.6, 0.2, 0.2],  # Predicts 0
            [0.2, 0.6, 0.2],  # Predicts 1
        ])
        
        acc = compute_accuracy(y_true, y_proba)
        assert acc == 1.0
    
    def test_zero_accuracy(self):
        """All wrong predictions should give 0.0 accuracy."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.1, 0.1, 0.8],  # Predicts 2, true is 0
            [0.8, 0.1, 0.1],  # Predicts 0, true is 1
            [0.1, 0.8, 0.1],  # Predicts 1, true is 2
        ])
        
        acc = compute_accuracy(y_true, y_proba)
        assert acc == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        y_true = np.array([0, 1, 2, 0])
        y_proba = np.array([
            [0.8, 0.1, 0.1],  # Correct
            [0.8, 0.1, 0.1],  # Wrong (predicts 0, true is 1)
            [0.1, 0.1, 0.8],  # Correct
            [0.8, 0.1, 0.1],  # Correct
        ])
        
        acc = compute_accuracy(y_true, y_proba)
        assert acc == 0.75


class TestCalibrationData:
    """Tests for calibration curve computation."""
    
    def test_calibration_returns_all_classes(self):
        """Calibration should return data for H, D, A."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.2, 0.3, 0.5],
        ])
        
        cal = compute_calibration_data(y_true, y_proba, n_bins=5)
        
        assert "H" in cal
        assert "D" in cal
        assert "A" in cal
    
    def test_calibration_structure(self):
        """Each class should have bin_centers, bin_true_freqs, bin_counts."""
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_proba = np.array([
            [0.7, 0.2, 0.1],
            [0.8, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.5, 0.2],
            [0.1, 0.2, 0.7],
        ])
        
        cal = compute_calibration_data(y_true, y_proba, n_bins=5)
        
        for cls in ["H", "D", "A"]:
            assert "bin_centers" in cal[cls]
            assert "bin_true_freqs" in cal[cls]
            assert "bin_counts" in cal[cls]


class TestEvaluatePredictions:
    """Tests for full evaluation."""
    
    def test_returns_metrics_result(self):
        """Should return MetricsResult with all fields."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
        ])
        
        result = evaluate_predictions(y_true, y_proba)
        
        assert isinstance(result, MetricsResult)
        assert result.n_samples == 5
        assert result.log_loss > 0
        assert result.brier_score > 0
        assert 0 <= result.accuracy <= 1
        assert "H" in result.class_counts
        assert "D" in result.class_counts
        assert "A" in result.class_counts
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        y_true = np.array([0, 1, 2])
        y_proba = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ])
        
        result = evaluate_predictions(y_true, y_proba)
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert "log_loss" in d
        assert "brier_score" in d
        assert "accuracy" in d


class TestBootstrapCompare:
    """Tests for bootstrap comparison."""
    
    def test_same_model_zero_diff(self):
        """Comparing model to itself should give ~0 difference."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.3, 0.5],
        ])
        
        result = bootstrap_compare(y_true, y_proba, y_proba, n_bootstrap=100)
        
        assert isinstance(result, BootstrapResult)
        assert abs(result.mean_diff) < 0.01  # Should be ~0
    
    def test_better_model_negative_diff(self):
        """Better model should have negative difference (lower log loss)."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        
        # Good model - high confidence in correct class
        y_proba_good = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        
        # Bad model - uniform
        y_proba_bad = np.array([
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
            [1/3, 1/3, 1/3],
        ])
        
        result = bootstrap_compare(y_true, y_proba_good, y_proba_bad, n_bootstrap=500)
        
        assert result.mean_diff < 0  # Good model has lower log loss
    
    def test_returns_bootstrap_result(self):
        """Should return properly structured BootstrapResult."""
        # Use more samples to avoid degenerate CI with tiny datasets
        np.random.seed(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_proba1 = np.array([
            [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
            [0.55, 0.25, 0.2], [0.2, 0.55, 0.25], [0.2, 0.2, 0.6],
            [0.65, 0.2, 0.15], [0.15, 0.65, 0.2], [0.25, 0.15, 0.6],
            [0.5, 0.3, 0.2],
        ])
        y_proba0 = np.array([
            [0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4],
            [0.45, 0.3, 0.25], [0.3, 0.45, 0.25], [0.3, 0.3, 0.4],
            [0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5],
            [0.4, 0.35, 0.25],
        ])
        
        result = bootstrap_compare(y_true, y_proba1, y_proba0, n_bootstrap=100)
        
        assert isinstance(result, BootstrapResult)
        assert result.n_bootstrap == 100
        # Use tolerance for floating point comparison
        assert result.ci_lower <= result.mean_diff + 1e-14
        assert result.mean_diff <= result.ci_upper + 1e-14
        assert 0 <= result.p_value <= 1
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        y_true = np.array([0, 1, 2])
        y_proba1 = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        y_proba0 = np.array([[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        
        result = bootstrap_compare(y_true, y_proba1, y_proba0, n_bootstrap=100)
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert "mean_diff" in d
        assert "ci_lower" in d
        assert "ci_upper" in d
        assert "p_value" in d
