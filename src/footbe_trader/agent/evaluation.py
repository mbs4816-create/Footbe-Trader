"""Edge Bucket Evaluation Module.

Evaluates the quality of edge predictions by bucketing trades by predicted edge
and measuring realized outcomes. Answers: "When the agent thinks it has an edge,
is it actually right?"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade for evaluation."""

    trade_id: str
    fixture_id: int
    outcome: str  # home_win, draw, away_win
    ticker: str

    # Entry
    entry_time: datetime
    entry_price: float
    quantity: int
    predicted_edge: float  # Edge at entry
    model_prob: float
    market_prob: float

    # Exit (filled when trade closes)
    exit_time: datetime | None = None
    exit_price: float | None = None
    settlement_value: float | None = None  # 1.0 if won, 0.0 if lost

    # P&L
    realized_pnl: float | None = None
    max_adverse_excursion: float | None = None  # Worst drawdown during hold

    # Context at entry
    hours_to_kickoff: float = 0.0
    drawdown_at_entry: float = 0.0
    pacing_state: str = ""

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None or self.settlement_value is not None

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        if self.realized_pnl is not None:
            return self.realized_pnl > 0
        if self.settlement_value is not None:
            return self.settlement_value > self.entry_price
        return False

    @property
    def return_pct(self) -> float | None:
        """Calculate return percentage."""
        if self.realized_pnl is not None:
            cost = self.entry_price * self.quantity
            if cost > 0:
                return self.realized_pnl / cost
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "fixture_id": self.fixture_id,
            "outcome": self.outcome,
            "ticker": self.ticker,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "predicted_edge": self.predicted_edge,
            "model_prob": self.model_prob,
            "market_prob": self.market_prob,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "settlement_value": self.settlement_value,
            "realized_pnl": self.realized_pnl,
            "max_adverse_excursion": self.max_adverse_excursion,
            "hours_to_kickoff": self.hours_to_kickoff,
            "is_closed": self.is_closed,
            "is_winner": self.is_winner,
            "return_pct": self.return_pct,
        }


@dataclass
class EdgeBucket:
    """Statistics for a specific edge range."""

    # Bucket definition
    edge_min: float  # Inclusive
    edge_max: float  # Exclusive (except for last bucket)
    bucket_name: str

    # Trade counts
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    open_trades: int = 0

    # Returns
    total_pnl: float = 0.0
    average_return: float | None = None
    median_return: float | None = None
    best_return: float | None = None
    worst_return: float | None = None

    # Risk
    average_mae: float | None = None  # Average max adverse excursion
    max_mae: float | None = None  # Worst MAE in bucket

    # Calibration
    average_predicted_edge: float | None = None
    average_realized_return: float | None = None
    calibration_error: float | None = None  # predicted - realized

    @property
    def win_rate(self) -> float | None:
        """Calculate win rate."""
        closed = self.winners + self.losers
        if closed == 0:
            return None
        return self.winners / closed

    @property
    def closed_trades(self) -> int:
        """Count of closed trades."""
        return self.winners + self.losers

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_min": self.edge_min,
            "edge_max": self.edge_max,
            "bucket_name": self.bucket_name,
            "total_trades": self.total_trades,
            "closed_trades": self.closed_trades,
            "winners": self.winners,
            "losers": self.losers,
            "open_trades": self.open_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "average_return": self.average_return,
            "best_return": self.best_return,
            "worst_return": self.worst_return,
            "average_mae": self.average_mae,
            "max_mae": self.max_mae,
            "average_predicted_edge": self.average_predicted_edge,
            "average_realized_return": self.average_realized_return,
            "calibration_error": self.calibration_error,
        }


@dataclass
class EvaluationReport:
    """Complete edge bucket evaluation report."""

    report_id: str
    generated_at: datetime = field(default_factory=utc_now)
    period_start: datetime | None = None
    period_end: datetime | None = None

    # Overall stats
    total_trades: int = 0
    total_closed: int = 0
    overall_win_rate: float | None = None
    overall_pnl: float = 0.0
    overall_return: float | None = None

    # Buckets
    buckets: list[EdgeBucket] = field(default_factory=list)

    # Calibration summary
    is_well_calibrated: bool = False
    calibration_summary: str = ""

    # By outcome
    stats_by_outcome: dict[str, dict[str, Any]] = field(default_factory=dict)

    # By time category
    stats_by_time_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "total_trades": self.total_trades,
            "total_closed": self.total_closed,
            "overall_win_rate": self.overall_win_rate,
            "overall_pnl": self.overall_pnl,
            "overall_return": self.overall_return,
            "buckets": [b.to_dict() for b in self.buckets],
            "is_well_calibrated": self.is_well_calibrated,
            "calibration_summary": self.calibration_summary,
            "stats_by_outcome": self.stats_by_outcome,
            "stats_by_time_category": self.stats_by_time_category,
        }


# Default edge bucket boundaries
DEFAULT_EDGE_BUCKETS = [
    (0.00, 0.01, "0-1%"),
    (0.01, 0.02, "1-2%"),
    (0.02, 0.04, "2-4%"),
    (0.04, 0.06, "4-6%"),
    (0.06, 0.10, "6-10%"),
    (0.10, 1.00, "10%+"),
]


class EdgeBucketEvaluator:
    """Evaluates edge prediction quality.

    Buckets trades by their predicted edge at entry and measures
    how well predictions correlate with realized outcomes.
    """

    def __init__(
        self,
        bucket_definitions: list[tuple[float, float, str]] | None = None,
    ):
        """Initialize evaluator.

        Args:
            bucket_definitions: List of (min, max, name) tuples for buckets.
        """
        self.bucket_definitions = bucket_definitions or DEFAULT_EDGE_BUCKETS
        self._trades: list[TradeRecord] = []

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a trade for evaluation.

        Args:
            trade: Trade record to add.
        """
        self._trades.append(trade)

    def update_trade(
        self,
        trade_id: str,
        exit_time: datetime | None = None,
        exit_price: float | None = None,
        settlement_value: float | None = None,
        realized_pnl: float | None = None,
        max_adverse_excursion: float | None = None,
    ) -> bool:
        """Update a trade with exit information.

        Args:
            trade_id: ID of trade to update.
            exit_time: When trade was closed.
            exit_price: Exit price (if sold before settlement).
            settlement_value: Settlement value (1.0 or 0.0).
            realized_pnl: Final P&L.
            max_adverse_excursion: Worst drawdown during hold.

        Returns:
            True if trade was found and updated.
        """
        for trade in self._trades:
            if trade.trade_id == trade_id:
                if exit_time is not None:
                    trade.exit_time = exit_time
                if exit_price is not None:
                    trade.exit_price = exit_price
                if settlement_value is not None:
                    trade.settlement_value = settlement_value
                if realized_pnl is not None:
                    trade.realized_pnl = realized_pnl
                if max_adverse_excursion is not None:
                    trade.max_adverse_excursion = max_adverse_excursion
                return True
        return False

    def get_bucket_for_edge(self, edge: float) -> str:
        """Get bucket name for an edge value.

        Args:
            edge: Predicted edge value.

        Returns:
            Bucket name.
        """
        for min_edge, max_edge, name in self.bucket_definitions:
            if min_edge <= edge < max_edge:
                return name
        # Default to last bucket for very high edges
        return self.bucket_definitions[-1][2]

    def generate_report(
        self,
        period_days: int | None = None,
        report_id: str | None = None,
    ) -> EvaluationReport:
        """Generate edge bucket evaluation report.

        Args:
            period_days: Only include trades from last N days.
            report_id: Custom report ID.

        Returns:
            EvaluationReport with all statistics.
        """
        import uuid

        now = utc_now()
        report = EvaluationReport(
            report_id=report_id or str(uuid.uuid4())[:8],
            period_end=now,
        )

        # Filter trades by period
        if period_days:
            cutoff = now - timedelta(days=period_days)
            trades = [t for t in self._trades if t.entry_time >= cutoff]
            report.period_start = cutoff
        else:
            trades = self._trades.copy()
            if trades:
                report.period_start = min(t.entry_time for t in trades)

        if not trades:
            report.calibration_summary = "No trades to evaluate."
            return report

        report.total_trades = len(trades)

        # Initialize buckets
        buckets: dict[str, list[TradeRecord]] = {
            name: [] for _, _, name in self.bucket_definitions
        }

        # Assign trades to buckets
        for trade in trades:
            bucket_name = self.get_bucket_for_edge(trade.predicted_edge)
            if bucket_name in buckets:
                buckets[bucket_name].append(trade)

        # Calculate stats per bucket
        report_buckets = []
        for min_edge, max_edge, name in self.bucket_definitions:
            bucket_trades = buckets.get(name, [])
            bucket = self._calculate_bucket_stats(
                min_edge, max_edge, name, bucket_trades
            )
            report_buckets.append(bucket)

        report.buckets = report_buckets

        # Overall stats
        closed_trades = [t for t in trades if t.is_closed]
        report.total_closed = len(closed_trades)

        if closed_trades:
            winners = sum(1 for t in closed_trades if t.is_winner)
            report.overall_win_rate = winners / len(closed_trades)
            report.overall_pnl = sum(t.realized_pnl or 0 for t in closed_trades)

            returns = [t.return_pct for t in closed_trades if t.return_pct is not None]
            if returns:
                report.overall_return = sum(returns) / len(returns)

        # Stats by outcome
        report.stats_by_outcome = self._calculate_outcome_stats(trades)

        # Stats by time category
        report.stats_by_time_category = self._calculate_time_stats(trades)

        # Calibration assessment
        report.is_well_calibrated, report.calibration_summary = (
            self._assess_calibration(report_buckets)
        )

        logger.info(
            "evaluation_report_generated",
            report_id=report.report_id,
            total_trades=report.total_trades,
            closed_trades=report.total_closed,
            overall_pnl=f"${report.overall_pnl:.2f}",
            is_calibrated=report.is_well_calibrated,
        )

        return report

    def _calculate_bucket_stats(
        self,
        min_edge: float,
        max_edge: float,
        name: str,
        trades: list[TradeRecord],
    ) -> EdgeBucket:
        """Calculate statistics for a single bucket."""
        bucket = EdgeBucket(
            edge_min=min_edge,
            edge_max=max_edge,
            bucket_name=name,
            total_trades=len(trades),
        )

        if not trades:
            return bucket

        # Count by status
        closed = [t for t in trades if t.is_closed]
        bucket.winners = sum(1 for t in closed if t.is_winner)
        bucket.losers = len(closed) - bucket.winners
        bucket.open_trades = len(trades) - len(closed)

        # P&L
        bucket.total_pnl = sum(t.realized_pnl or 0 for t in closed)

        # Returns
        returns = [t.return_pct for t in closed if t.return_pct is not None]
        if returns:
            bucket.average_return = sum(returns) / len(returns)
            sorted_returns = sorted(returns)
            mid = len(sorted_returns) // 2
            bucket.median_return = sorted_returns[mid]
            bucket.best_return = max(returns)
            bucket.worst_return = min(returns)

        # MAE
        maes = [t.max_adverse_excursion for t in trades if t.max_adverse_excursion is not None]
        if maes:
            bucket.average_mae = sum(maes) / len(maes)
            bucket.max_mae = max(maes)

        # Calibration
        edges = [t.predicted_edge for t in trades]
        bucket.average_predicted_edge = sum(edges) / len(edges)

        if returns:
            bucket.average_realized_return = sum(returns) / len(returns)
            bucket.calibration_error = (
                bucket.average_predicted_edge - bucket.average_realized_return
            )

        return bucket

    def _calculate_outcome_stats(
        self,
        trades: list[TradeRecord],
    ) -> dict[str, dict[str, Any]]:
        """Calculate stats grouped by outcome type."""
        stats: dict[str, dict[str, Any]] = {}

        for outcome in ["home_win", "draw", "away_win"]:
            outcome_trades = [t for t in trades if t.outcome == outcome]
            closed = [t for t in outcome_trades if t.is_closed]

            if not outcome_trades:
                continue

            winners = sum(1 for t in closed if t.is_winner)
            total_pnl = sum(t.realized_pnl or 0 for t in closed)

            stats[outcome] = {
                "total_trades": len(outcome_trades),
                "closed_trades": len(closed),
                "winners": winners,
                "win_rate": winners / len(closed) if closed else None,
                "total_pnl": total_pnl,
                "average_edge": (
                    sum(t.predicted_edge for t in outcome_trades) / len(outcome_trades)
                ),
            }

        return stats

    def _calculate_time_stats(
        self,
        trades: list[TradeRecord],
    ) -> dict[str, dict[str, Any]]:
        """Calculate stats grouped by time to kickoff."""
        stats: dict[str, dict[str, Any]] = {}

        categories = [
            ("very_early", 48, float("inf")),
            ("early", 24, 48),
            ("standard", 6, 24),
            ("optimal", 2, 6),
            ("late", 0, 2),
        ]

        for name, min_h, max_h in categories:
            cat_trades = [
                t for t in trades
                if min_h <= t.hours_to_kickoff < max_h
            ]
            closed = [t for t in cat_trades if t.is_closed]

            if not cat_trades:
                continue

            winners = sum(1 for t in closed if t.is_winner)
            total_pnl = sum(t.realized_pnl or 0 for t in closed)

            stats[name] = {
                "total_trades": len(cat_trades),
                "closed_trades": len(closed),
                "winners": winners,
                "win_rate": winners / len(closed) if closed else None,
                "total_pnl": total_pnl,
                "average_edge": (
                    sum(t.predicted_edge for t in cat_trades) / len(cat_trades)
                ),
            }

        return stats

    def _assess_calibration(
        self,
        buckets: list[EdgeBucket],
    ) -> tuple[bool, str]:
        """Assess overall calibration quality.

        Returns:
            Tuple of (is_well_calibrated, summary_text).
        """
        # Filter buckets with enough data
        valid_buckets = [
            b for b in buckets
            if b.closed_trades >= 10 and b.calibration_error is not None
        ]

        if not valid_buckets:
            return False, "Insufficient data for calibration assessment (need ≥10 closed trades per bucket)."

        # Check if calibration errors are reasonable
        errors = [abs(b.calibration_error) for b in valid_buckets]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)

        # Analyze patterns
        overconfident = sum(1 for b in valid_buckets if b.calibration_error > 0.02)
        underconfident = sum(1 for b in valid_buckets if b.calibration_error < -0.02)

        lines = []

        if avg_error < 0.02:
            is_calibrated = True
            lines.append("✓ Model is well-calibrated (avg error < 2%)")
        elif avg_error < 0.05:
            is_calibrated = True
            lines.append(f"⚠ Model is reasonably calibrated (avg error: {avg_error:.1%})")
        else:
            is_calibrated = False
            lines.append(f"✗ Model is poorly calibrated (avg error: {avg_error:.1%})")

        if overconfident > len(valid_buckets) / 2:
            lines.append(f"  - Tends to overestimate edge ({overconfident}/{len(valid_buckets)} buckets)")
        elif underconfident > len(valid_buckets) / 2:
            lines.append(f"  - Tends to underestimate edge ({underconfident}/{len(valid_buckets)} buckets)")

        # Per-bucket summary
        lines.append("\nPer-bucket calibration:")
        for b in valid_buckets:
            error_str = f"{b.calibration_error:+.1%}" if b.calibration_error else "N/A"
            wr_str = f"{b.win_rate:.0%}" if b.win_rate else "N/A"
            lines.append(
                f"  {b.bucket_name}: predicted {b.average_predicted_edge:.1%}, "
                f"realized {b.average_realized_return:.1%}, error {error_str}, "
                f"WR {wr_str} ({b.closed_trades} trades)"
            )

        return is_calibrated, "\n".join(lines)

    def generate_text_report(self, period_days: int | None = None) -> str:
        """Generate human-readable text report.

        Args:
            period_days: Only include trades from last N days.

        Returns:
            Formatted text report.
        """
        report = self.generate_report(period_days=period_days)

        lines = [
            "=" * 70,
            "EDGE BUCKET EVALUATION REPORT",
            "=" * 70,
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        ]

        if report.period_start:
            lines.append(
                f"Period: {report.period_start.strftime('%Y-%m-%d')} to "
                f"{report.period_end.strftime('%Y-%m-%d')}"
            )

        lines.extend([
            "",
            f"Total trades: {report.total_trades}",
            f"Closed trades: {report.total_closed}",
            f"Overall P&L: ${report.overall_pnl:,.2f}",
        ])

        if report.overall_win_rate is not None:
            lines.append(f"Overall win rate: {report.overall_win_rate:.1%}")

        lines.extend([
            "",
            "-" * 70,
            "BY EDGE BUCKET",
            "-" * 70,
            f"{'Bucket':<12} {'Trades':<8} {'Closed':<8} {'WR':<8} {'Avg Ret':<10} {'P&L':<12} {'Cal Err':<10}",
            "-" * 70,
        ])

        for b in report.buckets:
            wr = f"{b.win_rate:.0%}" if b.win_rate is not None else "-"
            avg_ret = f"{b.average_return:.1%}" if b.average_return is not None else "-"
            cal_err = f"{b.calibration_error:+.1%}" if b.calibration_error is not None else "-"
            lines.append(
                f"{b.bucket_name:<12} {b.total_trades:<8} {b.closed_trades:<8} "
                f"{wr:<8} {avg_ret:<10} ${b.total_pnl:<11,.2f} {cal_err:<10}"
            )

        lines.extend([
            "",
            "-" * 70,
            "BY OUTCOME",
            "-" * 70,
        ])

        for outcome, stats in report.stats_by_outcome.items():
            wr = f"{stats['win_rate']:.0%}" if stats.get('win_rate') else "-"
            lines.append(
                f"{outcome:<12} {stats['total_trades']} trades, "
                f"{stats['closed_trades']} closed, WR {wr}, "
                f"P&L ${stats['total_pnl']:.2f}"
            )

        lines.extend([
            "",
            "-" * 70,
            "BY TIME TO KICKOFF",
            "-" * 70,
        ])

        for category, stats in report.stats_by_time_category.items():
            wr = f"{stats['win_rate']:.0%}" if stats.get('win_rate') else "-"
            lines.append(
                f"{category:<12} {stats['total_trades']} trades, "
                f"{stats['closed_trades']} closed, WR {wr}, "
                f"P&L ${stats['total_pnl']:.2f}"
            )

        lines.extend([
            "",
            "-" * 70,
            "CALIBRATION ASSESSMENT",
            "-" * 70,
            report.calibration_summary,
            "",
            "=" * 70,
        ])

        return "\n".join(lines)

    def get_trades(self) -> list[TradeRecord]:
        """Get all recorded trades."""
        return self._trades.copy()

    def clear_trades(self) -> None:
        """Clear all recorded trades."""
        self._trades.clear()
