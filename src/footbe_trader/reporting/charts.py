"""Chart generation for reporting.

Provides matplotlib-based charts for visualizing trading performance:
- Equity curves
- Edge bucket distributions
- Decision pie charts
- Pace vs target tracking
- Drawdown charts
"""

import base64
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

from footbe_trader.common.logging import get_logger
from footbe_trader.reporting.config import ReportingConfig

logger = get_logger(__name__)


@dataclass
class ChartResult:
    """Result of chart generation."""
    
    chart_id: str
    title: str
    png_path: Path | None = None
    base64_data: str | None = None
    
    def get_html_img(self, alt_text: str = "") -> str:
        """Get HTML img tag for the chart.
        
        Args:
            alt_text: Alternative text for the image.
            
        Returns:
            HTML img tag string.
        """
        if self.base64_data:
            return f'<img src="data:image/png;base64,{self.base64_data}" alt="{alt_text or self.title}" />'
        elif self.png_path:
            return f'<img src="{self.png_path}" alt="{alt_text or self.title}" />'
        else:
            return f'<p>[Chart: {self.title}]</p>'
    
    def get_markdown_img(self, alt_text: str = "") -> str:
        """Get Markdown image syntax for the chart.
        
        Args:
            alt_text: Alternative text for the image.
            
        Returns:
            Markdown image string.
        """
        if self.png_path:
            return f'![{alt_text or self.title}]({self.png_path})'
        elif self.base64_data:
            # Markdown doesn't support base64 directly, but some renderers do
            return f'![{alt_text or self.title}](data:image/png;base64,{self.base64_data})'
        else:
            return f'*[Chart: {self.title}]*'


class ChartGenerator:
    """Generates matplotlib charts for reports."""
    
    def __init__(self, config: ReportingConfig | None = None):
        """Initialize chart generator.
        
        Args:
            config: Reporting configuration.
        """
        self.config = config or ReportingConfig()
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Configure matplotlib style."""
        try:
            plt.style.use(self.config.chart_style)
        except OSError:
            # Fallback if style not found
            plt.style.use("seaborn-v0_8-whitegrid")
        
        # Set default figure parameters
        plt.rcParams["figure.figsize"] = (
            self.config.chart_width,
            self.config.chart_height,
        )
        plt.rcParams["figure.dpi"] = self.config.chart_dpi
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["axes.labelsize"] = 10
    
    def _fig_to_result(
        self,
        fig: Figure,
        chart_id: str,
        title: str,
        save_path: Path | None = None,
    ) -> ChartResult:
        """Convert a matplotlib figure to a ChartResult.
        
        Args:
            fig: Matplotlib figure.
            chart_id: Unique identifier for the chart.
            title: Chart title.
            save_path: Optional path to save PNG file.
            
        Returns:
            ChartResult with base64 and/or file path.
        """
        # Generate base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=self.config.chart_dpi)
        buf.seek(0)
        base64_data = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        
        # Optionally save to file
        png_path = None
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), format="png", bbox_inches="tight", dpi=self.config.chart_dpi)
            png_path = save_path
        
        plt.close(fig)
        
        return ChartResult(
            chart_id=chart_id,
            title=title,
            png_path=png_path,
            base64_data=base64_data if self.config.embed_charts_base64 else None,
        )
    
    def equity_curve(
        self,
        timestamps: list[datetime],
        values: list[float],
        title: str = "Equity Curve",
        chart_id: str = "equity_curve",
        target_line: float | None = None,
        save_path: Path | None = None,
    ) -> ChartResult:
        """Generate an equity curve chart.
        
        Args:
            timestamps: List of datetime points.
            values: List of equity values.
            title: Chart title.
            chart_id: Unique chart identifier.
            target_line: Optional target equity to show as reference line.
            save_path: Optional path to save PNG.
            
        Returns:
            ChartResult with the generated chart.
        """
        fig, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height))
        
        if timestamps and values:
            ax.plot(timestamps, values, linewidth=2, color="#2196F3", label="Equity")
            
            if target_line is not None:
                ax.axhline(y=target_line, color="#4CAF50", linestyle="--", 
                          linewidth=1.5, label=f"Target: ${target_line:,.0f}")
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45, ha="right")
            
            # Starting point marker
            if len(values) > 0:
                ax.scatter([timestamps[0]], [values[0]], color="#FF5722", 
                          s=100, zorder=5, label=f"Start: ${values[0]:,.0f}")
                ax.scatter([timestamps[-1]], [values[-1]], color="#4CAF50",
                          s=100, zorder=5, label=f"End: ${values[-1]:,.0f}")
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14, color="gray")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        
        fig.tight_layout()
        
        return self._fig_to_result(fig, chart_id, title, save_path)
    
    def edge_bucket_bar(
        self,
        buckets: list[str],
        counts: list[int],
        title: str = "Edge Distribution",
        chart_id: str = "edge_buckets",
        colors: list[str] | None = None,
        save_path: Path | None = None,
    ) -> ChartResult:
        """Generate a bar chart of edge bucket distribution.
        
        Args:
            buckets: List of bucket labels (e.g., "<5%", "5-10%").
            counts: List of counts per bucket.
            title: Chart title.
            chart_id: Unique chart identifier.
            colors: Optional list of colors for bars.
            save_path: Optional path to save PNG.
            
        Returns:
            ChartResult with the generated chart.
        """
        fig, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height))
        
        if buckets and counts:
            if colors is None:
                # Gradient from red (low edge) to green (high edge)
                colors = ["#f44336", "#ff9800", "#ffeb3b", "#8bc34a", "#4caf50"]
                colors = colors[:len(buckets)] + ["#2196F3"] * (len(buckets) - len(colors))
            
            bars = ax.bar(buckets, counts, color=colors[:len(buckets)], edgecolor="white", linewidth=1)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.annotate(f"{count}",
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha="center", va="bottom",
                           fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14, color="gray")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Edge Bucket")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3, axis="y")
        
        fig.tight_layout()
        
        return self._fig_to_result(fig, chart_id, title, save_path)
    
    def decision_pie(
        self,
        actions: dict[str, int],
        title: str = "Decision Distribution",
        chart_id: str = "decision_pie",
        save_path: Path | None = None,
    ) -> ChartResult:
        """Generate a pie chart of decision types.
        
        Args:
            actions: Dict mapping action name to count.
            title: Chart title.
            chart_id: Unique chart identifier.
            save_path: Optional path to save PNG.
            
        Returns:
            ChartResult with the generated chart.
        """
        fig, ax = plt.subplots(figsize=(self.config.chart_height, self.config.chart_height))
        
        if actions and sum(actions.values()) > 0:
            # Color mapping for actions
            color_map = {
                "buy": "#4CAF50",      # Green
                "sell": "#f44336",     # Red
                "skip": "#9E9E9E",     # Gray
                "hold": "#2196F3",     # Blue
                "exit": "#FF9800",     # Orange
            }
            
            labels = list(actions.keys())
            sizes = list(actions.values())
            colors = [color_map.get(a.lower(), "#607D8B") for a in labels]
            
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct=lambda pct: f"{pct:.1f}%\n({int(pct * sum(sizes) / 100)})",
                startangle=90,
                explode=[0.02] * len(sizes),
            )
            
            # Style the text
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(9)
        else:
            ax.text(0.5, 0.5, "No decisions", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14, color="gray")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        fig.tight_layout()
        
        return self._fig_to_result(fig, chart_id, title, save_path)
    
    def pace_vs_target(
        self,
        dates: list[datetime],
        actual_returns: list[float],
        target_return: float,
        tolerance: float,
        title: str = "Pace vs Target",
        chart_id: str = "pace_vs_target",
        save_path: Path | None = None,
    ) -> ChartResult:
        """Generate a chart comparing actual pace to target.
        
        Args:
            dates: List of datetime points.
            actual_returns: List of cumulative returns.
            target_return: Target return for the period.
            tolerance: Tolerance band around target.
            title: Chart title.
            chart_id: Unique chart identifier.
            save_path: Optional path to save PNG.
            
        Returns:
            ChartResult with the generated chart.
        """
        fig, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height))
        
        if dates and actual_returns:
            # Calculate target line (linear progression)
            days_total = (dates[-1] - dates[0]).days + 1
            target_line = [
                target_return * ((d - dates[0]).days + 1) / days_total
                for d in dates
            ]
            
            # Plot actual returns
            ax.plot(dates, actual_returns, linewidth=2, color="#2196F3", 
                   label="Actual", marker="o", markersize=4)
            
            # Plot target line
            ax.plot(dates, target_line, linewidth=2, color="#4CAF50", 
                   linestyle="--", label=f"Target ({target_return*100:.0f}%)")
            
            # Plot tolerance band
            upper_band = [t + tolerance for t in target_line]
            lower_band = [t - tolerance for t in target_line]
            ax.fill_between(dates, lower_band, upper_band, alpha=0.2, 
                           color="#4CAF50", label="Tolerance Band")
            
            # Format axes
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.1f}%"))
            plt.xticks(rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14, color="gray")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        
        fig.tight_layout()
        
        return self._fig_to_result(fig, chart_id, title, save_path)
    
    def drawdown_chart(
        self,
        timestamps: list[datetime],
        drawdowns: list[float],
        title: str = "Drawdown",
        chart_id: str = "drawdown",
        threshold_lines: list[tuple[float, str]] | None = None,
        save_path: Path | None = None,
    ) -> ChartResult:
        """Generate a drawdown chart.
        
        Args:
            timestamps: List of datetime points.
            drawdowns: List of drawdown values (as positive fractions).
            title: Chart title.
            chart_id: Unique chart identifier.
            threshold_lines: Optional list of (value, label) for horizontal lines.
            save_path: Optional path to save PNG.
            
        Returns:
            ChartResult with the generated chart.
        """
        fig, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height * 0.6))
        
        if timestamps and drawdowns:
            # Convert to negative for display (drawdowns shown below zero)
            neg_drawdowns = [-d for d in drawdowns]
            
            ax.fill_between(timestamps, 0, neg_drawdowns, color="#f44336", alpha=0.5)
            ax.plot(timestamps, neg_drawdowns, linewidth=1.5, color="#d32f2f")
            
            # Add threshold lines
            if threshold_lines:
                for threshold, label in threshold_lines:
                    ax.axhline(y=-threshold, color="#FF9800", linestyle="--",
                              linewidth=1, label=f"{label}: {threshold*100:.0f}%")
            
            # Format axes
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{abs(x)*100:.1f}%"))
            plt.xticks(rotation=45, ha="right")
            
            # Max drawdown annotation
            if drawdowns:
                max_dd = max(drawdowns)
                max_idx = drawdowns.index(max_dd)
                ax.annotate(f"Max: {max_dd*100:.1f}%",
                           xy=(timestamps[max_idx], -max_dd),
                           xytext=(10, -10),
                           textcoords="offset points",
                           fontsize=10, fontweight="bold",
                           color="#d32f2f")
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14, color="gray")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Drawdown")
        if threshold_lines:
            ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(top=0)  # Drawdown always negative/zero
        
        fig.tight_layout()
        
        return self._fig_to_result(fig, chart_id, title, save_path)
    
    def rejection_reasons_bar(
        self,
        reasons: dict[str, int],
        title: str = "Skip Reasons",
        chart_id: str = "rejection_reasons",
        max_reasons: int = 10,
        save_path: Path | None = None,
    ) -> ChartResult:
        """Generate a horizontal bar chart of rejection reasons.
        
        Args:
            reasons: Dict mapping reason to count.
            title: Chart title.
            chart_id: Unique chart identifier.
            max_reasons: Maximum number of reasons to show.
            save_path: Optional path to save PNG.
            
        Returns:
            ChartResult with the generated chart.
        """
        fig, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height))
        
        if reasons:
            # Sort by count and take top N
            sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
            sorted_reasons = sorted_reasons[:max_reasons]
            
            labels = [r[0] for r in sorted_reasons]
            counts = [r[1] for r in sorted_reasons]
            
            # Truncate long labels
            labels = [l[:40] + "..." if len(l) > 40 else l for l in labels]
            
            y_pos = range(len(labels))
            bars = ax.barh(y_pos, counts, color="#FF9800", edgecolor="white")
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()  # Top to bottom
            
            # Add count labels
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax.annotate(f"{count}",
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0),
                           textcoords="offset points",
                           ha="left", va="center",
                           fontsize=9)
        else:
            ax.text(0.5, 0.5, "No rejections", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14, color="gray")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Count")
        ax.grid(True, alpha=0.3, axis="x")
        
        fig.tight_layout()
        
        return self._fig_to_result(fig, chart_id, title, save_path)
    
    def trades_timeline(
        self,
        trade_times: list[datetime],
        trade_pnls: list[float],
        title: str = "Trade Timeline",
        chart_id: str = "trades_timeline",
        save_path: Path | None = None,
    ) -> ChartResult:
        """Generate a timeline scatter plot of trades.
        
        Args:
            trade_times: List of trade timestamps.
            trade_pnls: List of P&L for each trade.
            title: Chart title.
            chart_id: Unique chart identifier.
            save_path: Optional path to save PNG.
            
        Returns:
            ChartResult with the generated chart.
        """
        fig, ax = plt.subplots(figsize=(self.config.chart_width, self.config.chart_height * 0.6))
        
        if trade_times and trade_pnls:
            # Color based on P&L
            colors = ["#4CAF50" if pnl >= 0 else "#f44336" for pnl in trade_pnls]
            sizes = [abs(pnl) * 10 + 50 for pnl in trade_pnls]  # Size based on magnitude
            
            ax.scatter(trade_times, trade_pnls, c=colors, s=sizes, alpha=0.7, edgecolors="white")
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=1)
            
            # Format axes
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
            plt.xticks(rotation=45, ha="right")
            
            # Summary stats
            wins = sum(1 for p in trade_pnls if p > 0)
            losses = sum(1 for p in trade_pnls if p < 0)
            total = sum(trade_pnls)
            ax.text(0.02, 0.98, f"Wins: {wins} | Losses: {losses} | Net: ${total:,.0f}",
                   transform=ax.transAxes, fontsize=9, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                   transform=ax.transAxes, fontsize=14, color="gray")
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("P&L")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        return self._fig_to_result(fig, chart_id, title, save_path)
