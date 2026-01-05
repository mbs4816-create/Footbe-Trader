"""Tests for the reporting config module."""

import pytest
from pathlib import Path

from footbe_trader.reporting.config import ReportingConfig


class TestReportingConfig:
    """Tests for ReportingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ReportingConfig()
        
        assert config.reports_dir == Path("reports")
        assert config.generate_html is True
        assert config.generate_markdown is True
        assert config.embed_charts_base64 is True
        assert config.chart_dpi == 100
        assert config.target_weekly_return == 0.10
        assert config.retention_days == 90
    
    def test_directory_properties(self):
        """Test derived directory properties."""
        config = ReportingConfig(reports_dir=Path("/tmp/test_reports"))
        
        assert config.artifacts_dir == Path("/tmp/test_reports/artifacts")
        assert config.charts_dir == Path("/tmp/test_reports/charts")
        assert config.runs_dir == Path("/tmp/test_reports/runs")
        assert config.daily_dir == Path("/tmp/test_reports/daily")
        assert config.weekly_dir == Path("/tmp/test_reports/weekly")
    
    def test_ensure_directories(self, tmp_path):
        """Test directory creation."""
        config = ReportingConfig(reports_dir=tmp_path / "reports")
        config.ensure_directories()
        
        assert config.reports_dir.exists()
        assert config.artifacts_dir.exists()
        assert config.charts_dir.exists()
        assert config.runs_dir.exists()
        assert config.daily_dir.exists()
        assert config.weekly_dir.exists()
    
    def test_get_edge_bucket_negative(self):
        """Test edge bucket for negative edge."""
        config = ReportingConfig()
        assert config.get_edge_bucket(-0.05) == "negative"
    
    def test_get_edge_bucket_low(self):
        """Test edge bucket for low edge."""
        config = ReportingConfig()
        assert config.get_edge_bucket(0.03) == "0-5%"
    
    def test_get_edge_bucket_medium(self):
        """Test edge bucket for medium edge."""
        config = ReportingConfig()
        assert config.get_edge_bucket(0.07) == "5-10%"
    
    def test_get_edge_bucket_high(self):
        """Test edge bucket for high edge."""
        config = ReportingConfig()
        assert config.get_edge_bucket(0.12) == "10-15%"
    
    def test_get_edge_bucket_very_high(self):
        """Test edge bucket for very high edge."""
        config = ReportingConfig()
        assert config.get_edge_bucket(0.25) == "20%+"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ReportingConfig()
        data = config.to_dict()
        
        assert "reports_dir" in data
        assert "generate_html" in data
        assert "target_weekly_return" in data
        assert data["target_weekly_return"] == 0.10
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "reports_dir": "/custom/path",
            "generate_html": False,
            "target_weekly_return": 0.15,
        }
        
        config = ReportingConfig.from_dict(data)
        
        assert config.reports_dir == Path("/custom/path")
        assert config.generate_html is False
        assert config.target_weekly_return == 0.15
    
    def test_round_trip(self):
        """Test to_dict and from_dict round trip."""
        original = ReportingConfig(
            generate_html=False,
            chart_dpi=150,
            target_weekly_return=0.08,
        )
        
        data = original.to_dict()
        restored = ReportingConfig.from_dict(data)
        
        assert restored.generate_html == original.generate_html
        assert restored.chart_dpi == original.chart_dpi
        assert restored.target_weekly_return == original.target_weekly_return
