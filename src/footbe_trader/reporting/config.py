"""Configuration for the reporting subsystem."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportingConfig:
    """Configuration for report generation.
    
    Attributes:
        reports_dir: Base directory for all reports (default: ./reports)
        artifacts_dir: Directory for JSON artifacts (default: ./reports/artifacts)
        charts_dir: Directory for chart images (default: ./reports/charts)
        
        generate_html: Whether to generate HTML reports
        generate_markdown: Whether to generate Markdown reports
        embed_charts_base64: Embed charts as base64 in HTML (vs external files)
        
        chart_dpi: Resolution for generated charts
        chart_width: Default chart width in inches
        chart_height: Default chart height in inches
        chart_style: Matplotlib style to use
        
        edge_bucket_boundaries: Edge percentages for bucketing decisions
        
        target_weekly_return: Target weekly return for pacing display
        target_tolerance: Tolerance band for "on pace" status
    """
    
    # Directory structure
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    
    @property
    def artifacts_dir(self) -> Path:
        """Directory for JSON artifacts."""
        return self.reports_dir / "artifacts"
    
    @property
    def charts_dir(self) -> Path:
        """Directory for generated charts."""
        return self.reports_dir / "charts"
    
    @property
    def runs_dir(self) -> Path:
        """Directory for per-run reports."""
        return self.reports_dir / "runs"
    
    @property
    def daily_dir(self) -> Path:
        """Directory for daily reports."""
        return self.reports_dir / "daily"
    
    @property
    def weekly_dir(self) -> Path:
        """Directory for weekly reports."""
        return self.reports_dir / "weekly"
    
    # Output formats
    generate_html: bool = True
    generate_markdown: bool = True
    embed_charts_base64: bool = True  # Embed in HTML for portability
    
    # Chart settings
    chart_dpi: int = 100
    chart_width: float = 10.0
    chart_height: float = 6.0
    chart_style: str = "seaborn-v0_8-darkgrid"
    
    # Edge bucket boundaries for categorization
    edge_bucket_boundaries: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    
    # Target tracking
    target_weekly_return: float = 0.10  # 10%
    target_tolerance: float = 0.02  # ±2%
    
    # Report retention (days to keep old reports)
    retention_days: int = 90
    
    def ensure_directories(self) -> None:
        """Create all report directories if they don't exist."""
        for directory in [
            self.reports_dir,
            self.artifacts_dir,
            self.charts_dir,
            self.runs_dir,
            self.daily_dir,
            self.weekly_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_edge_bucket(self, edge: float) -> str:
        """Categorize an edge value into a bucket.
        
        Args:
            edge: Edge as a decimal (0.10 = 10%)
            
        Returns:
            Bucket label like "<5%", "5-10%", "10-15%", "15-20%", "20%+"
        """
        if edge < 0:
            return "negative"
        
        boundaries = sorted(self.edge_bucket_boundaries)
        for i, threshold in enumerate(boundaries):
            if edge < threshold:
                if i == 0:
                    return f"<{int(threshold * 100)}%"
                prev = boundaries[i - 1]
                return f"{int(prev * 100)}-{int(threshold * 100)}%"
        
        # Above highest boundary
        highest = boundaries[-1]
        return f"{int(highest * 100)}%+"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "reports_dir": str(self.reports_dir),
            "generate_html": self.generate_html,
            "generate_markdown": self.generate_markdown,
            "embed_charts_base64": self.embed_charts_base64,
            "chart_dpi": self.chart_dpi,
            "chart_width": self.chart_width,
            "chart_height": self.chart_height,
            "chart_style": self.chart_style,
            "edge_bucket_boundaries": self.edge_bucket_boundaries,
            "target_weekly_return": self.target_weekly_return,
            "target_tolerance": self.target_tolerance,
            "retention_days": self.retention_days,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReportingConfig":
        """Create config from dictionary."""
        config = cls()
        if "reports_dir" in data:
            config.reports_dir = Path(data["reports_dir"])
        if "generate_html" in data:
            config.generate_html = data["generate_html"]
        if "generate_markdown" in data:
            config.generate_markdown = data["generate_markdown"]
        if "embed_charts_base64" in data:
            config.embed_charts_base64 = data["embed_charts_base64"]
        if "chart_dpi" in data:
            config.chart_dpi = data["chart_dpi"]
        if "chart_width" in data:
            config.chart_width = data["chart_width"]
        if "chart_height" in data:
            config.chart_height = data["chart_height"]
        if "chart_style" in data:
            config.chart_style = data["chart_style"]
        if "edge_bucket_boundaries" in data:
            config.edge_bucket_boundaries = data["edge_bucket_boundaries"]
        if "target_weekly_return" in data:
            config.target_weekly_return = data["target_weekly_return"]
        if "target_tolerance" in data:
            config.target_tolerance = data["target_tolerance"]
        if "retention_days" in data:
            config.retention_days = data["retention_days"]
        return config
