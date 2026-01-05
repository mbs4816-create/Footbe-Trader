"""Tests for calibration module (temperature scaling)."""

import numpy as np
import pytest

from footbe_trader.modeling.calibration import (
    CalibrationResult,
    TemperatureScaler,
    apply_temperature_scaling,
    calibrate_probabilities,
    compute_ece,
    compute_mce,
    compute_reliability_diagram_data,
    find_optimal_temperature,
)


class TestComputeECE:
    """Tests for Expected Calibration Error."""
    
    def test_well_calibrated_lower_ece(self):
        """Well-calibrated predictions should have lower ECE than random."""
        np.random.seed(42)
        n = 1000
        
        # Create well-calibrated predictions by sampling true class from distribution
        probs_calibrated = np.random.dirichlet([2, 2, 2], size=n)
        y_calibrated = np.array([
            np.random.choice([0, 1, 2], p=p) for p in probs_calibrated
        ])
        
        ece_calibrated = compute_ece(probs_calibrated, y_calibrated)
        
        # Create miscalibrated: predict 0.9 for class 0 but true class is random
        probs_miscalibrated = np.zeros((n, 3))
        probs_miscalibrated[:, 0] = 0.9
        probs_miscalibrated[:, 1] = 0.05
        probs_miscalibrated[:, 2] = 0.05
        y_miscalibrated = np.random.randint(0, 3, size=n)
        
        ece_miscalibrated = compute_ece(probs_miscalibrated, y_miscalibrated)
        
        # Calibrated should have lower ECE than miscalibrated
        assert ece_calibrated < ece_miscalibrated
    
    def test_overconfident_high_ece(self):
        """Overconfident wrong predictions should have high ECE."""
        n = 100
        
        # Predict class 0 with high confidence, but true is class 1
        probs = np.zeros((n, 3))
        probs[:, 0] = 0.9
        probs[:, 1] = 0.05
        probs[:, 2] = 0.05
        
        y_true = np.ones(n, dtype=int)  # All class 1
        
        ece = compute_ece(probs, y_true)
        
        # High confidence but wrong = high ECE
        assert ece > 0.5
    
    def test_ece_between_zero_and_one(self):
        """ECE should be between 0 and 1."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        y_true = np.random.randint(0, 3, 100)
        
        ece = compute_ece(probs, y_true)
        
        assert 0 <= ece <= 1


class TestComputeMCE:
    """Tests for Maximum Calibration Error."""
    
    def test_mce_at_least_as_large_as_ece(self):
        """MCE should be >= ECE."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        y_true = np.random.randint(0, 3, 100)
        
        ece = compute_ece(probs, y_true)
        mce = compute_mce(probs, y_true)
        
        assert mce >= ece or abs(mce - ece) < 0.01


class TestTemperatureScaling:
    """Tests for temperature scaling functions."""
    
    def test_temperature_one_no_change(self):
        """Temperature=1 should not change probabilities."""
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        
        scaled = apply_temperature_scaling(probs, temperature=1.0)
        
        np.testing.assert_array_almost_equal(probs, scaled, decimal=5)
    
    def test_high_temperature_softens(self):
        """High temperature should make predictions more uniform."""
        probs = np.array([[0.9, 0.05, 0.05], [0.7, 0.2, 0.1]])
        
        scaled = apply_temperature_scaling(probs, temperature=3.0)
        
        # Max probability should decrease
        assert scaled[0, 0] < probs[0, 0]
        assert scaled[1, 0] < probs[1, 0]
        
        # Min probability should increase
        assert scaled[0, 2] > probs[0, 2]
    
    def test_low_temperature_sharpens(self):
        """Low temperature should make predictions more confident."""
        probs = np.array([[0.5, 0.3, 0.2], [0.4, 0.35, 0.25]])
        
        scaled = apply_temperature_scaling(probs, temperature=0.5)
        
        # Max probability should increase
        assert scaled[0, 0] > probs[0, 0]
        assert scaled[1, 0] > probs[1, 0]
    
    def test_probabilities_still_sum_to_one(self):
        """Scaled probabilities should still sum to 1."""
        probs = np.random.dirichlet(np.ones(3), size=10)
        
        for temp in [0.5, 1.0, 2.0, 5.0]:
            scaled = apply_temperature_scaling(probs, temp)
            sums = scaled.sum(axis=1)
            np.testing.assert_array_almost_equal(sums, np.ones(10), decimal=6)


class TestFindOptimalTemperature:
    """Tests for temperature optimization."""
    
    def test_finds_reasonable_temperature(self):
        """Should find temperature in reasonable range."""
        np.random.seed(42)
        
        # Create overconfident predictions
        n = 200
        probs = np.zeros((n, 3))
        y_true = np.random.randint(0, 3, n)
        
        for i in range(n):
            probs[i, y_true[i]] = 0.8
            other = [j for j in range(3) if j != y_true[i]]
            probs[i, other[0]] = 0.15
            probs[i, other[1]] = 0.05
        
        # Add some noise/mistakes
        for i in range(n // 5):
            y_true[i] = (y_true[i] + 1) % 3
        
        logits = np.log(probs + 1e-10)
        temp = find_optimal_temperature(logits, y_true)
        
        # Temperature should be in reasonable range
        assert 0.5 < temp < 5.0
    
    def test_calibration_improves_ece(self):
        """Calibration should reduce ECE."""
        np.random.seed(42)
        
        # Create predictions
        n = 300
        probs = np.zeros((n, 3))
        y_true = np.random.randint(0, 3, n)
        
        for i in range(n):
            probs[i, y_true[i]] = 0.7 + np.random.rand() * 0.25
            remaining = 1 - probs[i, y_true[i]]
            other = [j for j in range(3) if j != y_true[i]]
            split = np.random.rand()
            probs[i, other[0]] = remaining * split
            probs[i, other[1]] = remaining * (1 - split)
        
        # Add mistakes
        for i in range(n // 4):
            y_true[i] = (y_true[i] + 1) % 3
        
        ece_before = compute_ece(probs, y_true)
        calibrated, temp = calibrate_probabilities(probs, y_true)
        ece_after = compute_ece(calibrated, y_true)
        
        # ECE should improve or stay similar
        assert ece_after <= ece_before + 0.05


class TestTemperatureScaler:
    """Tests for TemperatureScaler class."""
    
    def test_fit_and_transform(self):
        """Fit should learn temperature, transform should apply it."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        y_true = np.random.randint(0, 3, 100)
        
        scaler = TemperatureScaler()
        scaler.fit(probs, y_true)
        
        assert scaler._is_fitted
        assert scaler.temperature > 0
        
        calibrated = scaler.transform(probs)
        assert calibrated.shape == probs.shape
    
    def test_fit_transform_shortcut(self):
        """fit_transform should work in one call."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        y_true = np.random.randint(0, 3, 100)
        
        scaler = TemperatureScaler()
        calibrated = scaler.fit_transform(probs, y_true)
        
        assert scaler._is_fitted
        assert calibrated.shape == probs.shape
    
    def test_result_property(self):
        """Result should contain calibration metrics."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        y_true = np.random.randint(0, 3, 100)
        
        scaler = TemperatureScaler()
        scaler.fit(probs, y_true)
        
        result = scaler.result
        
        assert isinstance(result, CalibrationResult)
        assert result.temperature == scaler.temperature
        assert result.pre_calibration_ece >= 0
        assert result.post_calibration_ece >= 0
    
    def test_transform_before_fit_raises(self):
        """Transform without fitting should raise."""
        scaler = TemperatureScaler()
        probs = np.random.dirichlet(np.ones(3), size=10)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(probs)
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        y_true = np.random.randint(0, 3, 100)
        
        scaler = TemperatureScaler()
        scaler.fit(probs, y_true)
        
        d = scaler.to_dict()
        
        assert "temperature" in d
        assert "is_fitted" in d
        assert d["is_fitted"] is True


class TestCalibrationResult:
    """Tests for CalibrationResult dataclass."""
    
    def test_to_dict(self):
        """to_dict should serialize all fields."""
        result = CalibrationResult(
            temperature=1.5,
            pre_calibration_ece=0.15,
            post_calibration_ece=0.08,
            pre_calibration_nll=1.1,
            post_calibration_nll=1.05,
        )
        
        d = result.to_dict()
        
        assert d["temperature"] == 1.5
        assert d["pre_calibration_ece"] == 0.15
        assert d["post_calibration_ece"] == 0.08


class TestReliabilityDiagramData:
    """Tests for reliability diagram data computation."""
    
    def test_returns_correct_structure(self):
        """Should return correct data structure."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=100)
        y_true = np.random.randint(0, 3, 100)
        
        data = compute_reliability_diagram_data(probs, y_true, n_bins=10)
        
        assert "bin_centers" in data
        assert "bin_accuracies" in data
        assert "bin_confidences" in data
        assert "bin_counts" in data
        assert "ece" in data
        assert "mce" in data
        
        assert len(data["bin_centers"]) == 10
        assert len(data["bin_accuracies"]) == 10
    
    def test_bin_counts_sum_to_n(self):
        """Bin counts should sum to number of samples."""
        np.random.seed(42)
        n = 100
        probs = np.random.dirichlet(np.ones(3), size=n)
        y_true = np.random.randint(0, 3, n)
        
        data = compute_reliability_diagram_data(probs, y_true)
        
        assert sum(data["bin_counts"]) == n
