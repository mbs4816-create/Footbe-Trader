"""Backtest Metrics and Report Generation.

Calculates comprehensive metrics from strategy backtest results and
generates formatted reports for analysis.
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.storage.models import BacktestEquity, BacktestTrade, StrategyBacktest

logger = get_logger(__name__)


@dataclass
class ReturnMetrics:
    """Return-based performance metrics."""

    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float | None = None
    best_day: float | None = None
    worst_day: float | None = None
    avg_daily_return: float | None = None
    positive_days: int = 0
    negative_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "avg_daily_return": self.avg_daily_return,
            "positive_days": self.positive_days,
            "negative_days": self.negative_days,
        }


@dataclass
class RiskMetrics:
    """Risk-based performance metrics."""

    max_drawdown: float = 0.0
    max_drawdown_duration_hours: float | None = None
    avg_drawdown: float = 0.0
    volatility: float | None = None
    downside_volatility: float | None = None
    var_95: float | None = None  # Value at Risk (95%)
    cvar_95: float | None = None  # Conditional VaR (95%)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_hours": self.max_drawdown_duration_hours,
            "avg_drawdown": self.avg_drawdown,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }


@dataclass
class RatioMetrics:
    """Risk-adjusted return ratios."""

    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    profit_factor: float | None = None
    expectancy: float | None = None  # Expected value per trade

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
        }


@dataclass
class TradeMetrics:
    """Trade-level statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_hold_time_minutes: float | None = None
    median_hold_time_minutes: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade": self.avg_trade,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_hold_time_minutes": self.avg_hold_time_minutes,
            "median_hold_time_minutes": self.median_hold_time_minutes,
        }


@dataclass
class ExposureMetrics:
    """Exposure and position statistics."""

    avg_exposure: float = 0.0
    max_exposure: float = 0.0
    avg_position_count: float = 0.0
    max_position_count: int = 0
    exposure_utilization: float = 0.0  # % of bankroll used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "avg_exposure": self.avg_exposure,
            "max_exposure": self.max_exposure,
            "avg_position_count": self.avg_position_count,
            "max_position_count": self.max_position_count,
            "exposure_utilization": self.exposure_utilization,
        }


@dataclass
class EdgeCalibrationBucket:
    """Edge calibration for a single bucket."""

    edge_bucket: str = ""
    trade_count: int = 0
    expected_return: float = 0.0  # Based on edge
    actual_return: float = 0.0
    win_rate: float = 0.0
    calibration_error: float = 0.0  # expected - actual

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_bucket": self.edge_bucket,
            "trade_count": self.trade_count,
            "expected_return": self.expected_return,
            "actual_return": self.actual_return,
            "win_rate": self.win_rate,
            "calibration_error": self.calibration_error,
        }


@dataclass
class OutcomeBreakdown:
    """Performance breakdown by outcome type."""

    outcome: str = ""
    trades: int = 0
    winning: int = 0
    losing: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_edge: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.outcome,
            "trades": self.trades,
            "winning": self.winning,
            "losing": self.losing,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "avg_edge": self.avg_edge,
        }


@dataclass
class BacktestMetricsResult:
    """Complete backtest metrics result."""

    backtest_id: str = ""
    strategy_config_hash: str = ""
    backtest_period_start: datetime | None = None
    backtest_period_end: datetime | None = None
    backtest_duration_hours: float = 0.0

    initial_bankroll: float = 0.0
    final_bankroll: float = 0.0

    returns: ReturnMetrics = field(default_factory=ReturnMetrics)
    risk: RiskMetrics = field(default_factory=RiskMetrics)
    ratios: RatioMetrics = field(default_factory=RatioMetrics)
    trades: TradeMetrics = field(default_factory=TradeMetrics)
    exposure: ExposureMetrics = field(default_factory=ExposureMetrics)

    edge_calibration: list[EdgeCalibrationBucket] = field(default_factory=list)
    outcome_breakdown: list[OutcomeBreakdown] = field(default_factory=list)
    fixture_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backtest_id": self.backtest_id,
            "strategy_config_hash": self.strategy_config_hash,
            "backtest_period_start": (
                self.backtest_period_start.isoformat()
                if self.backtest_period_start
                else None
            ),
            "backtest_period_end": (
                self.backtest_period_end.isoformat()
                if self.backtest_period_end
                else None
            ),
            "backtest_duration_hours": self.backtest_duration_hours,
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": self.final_bankroll,
            "returns": self.returns.to_dict(),
            "risk": self.risk.to_dict(),
            "ratios": self.ratios.to_dict(),
            "trades": self.trades.to_dict(),
            "exposure": self.exposure.to_dict(),
            "edge_calibration": [b.to_dict() for b in self.edge_calibration],
            "outcome_breakdown": [o.to_dict() for o in self.outcome_breakdown],
            "fixture_count": self.fixture_count,
        }


class BacktestMetricsCalculator:
    """Calculates comprehensive metrics from backtest results."""

    def calculate(
        self,
        backtest: StrategyBacktest,
        trades: list[BacktestTrade],
        equity_curve: list[BacktestEquity],
    ) -> BacktestMetricsResult:
        """Calculate all metrics from backtest results.

        Args:
            backtest: Backtest result object.
            trades: List of individual trades.
            equity_curve: List of equity snapshots.

        Returns:
            Complete metrics result.
        """
        result = BacktestMetricsResult(
            backtest_id=backtest.backtest_id,
            strategy_config_hash=backtest.strategy_config_hash,
            initial_bankroll=backtest.initial_bankroll,
            final_bankroll=backtest.final_bankroll or backtest.initial_bankroll,
        )

        # Set time period
        if equity_curve:
            result.backtest_period_start = equity_curve[0].timestamp
            result.backtest_period_end = equity_curve[-1].timestamp
            if result.backtest_period_start and result.backtest_period_end:
                result.backtest_duration_hours = (
                    result.backtest_period_end - result.backtest_period_start
                ).total_seconds() / 3600

        # Calculate component metrics
        result.returns = self._calculate_return_metrics(backtest, equity_curve)
        result.risk = self._calculate_risk_metrics(equity_curve)
        result.trades = self._calculate_trade_metrics(trades)
        result.exposure = self._calculate_exposure_metrics(
            equity_curve, backtest.initial_bankroll
        )
        result.ratios = self._calculate_ratio_metrics(
            result.returns, result.risk, result.trades
        )

        # Calculate calibration and breakdowns
        result.edge_calibration = self._calculate_edge_calibration(trades)
        result.outcome_breakdown = self._calculate_outcome_breakdown(trades)
        result.fixture_count = len(set(t.fixture_id for t in trades))

        return result

    def _calculate_return_metrics(
        self,
        backtest: StrategyBacktest,
        equity_curve: list[BacktestEquity],
    ) -> ReturnMetrics:
        """Calculate return-based metrics."""
        metrics = ReturnMetrics()

        initial = backtest.initial_bankroll
        final = backtest.final_bankroll or initial

        metrics.total_return = final - initial
        metrics.total_return_pct = (
            (final - initial) / initial if initial > 0 else 0.0
        )

        if len(equity_curve) < 2:
            return metrics

        # Calculate daily returns (aggregate by day)
        daily_returns = self._aggregate_daily_returns(equity_curve)

        if daily_returns:
            metrics.best_day = max(daily_returns)
            metrics.worst_day = min(daily_returns)
            metrics.avg_daily_return = sum(daily_returns) / len(daily_returns)
            metrics.positive_days = sum(1 for r in daily_returns if r > 0)
            metrics.negative_days = sum(1 for r in daily_returns if r < 0)

            # Annualized return
            if len(daily_returns) >= 2:
                # Compound daily returns
                cumulative = 1.0
                for r in daily_returns:
                    cumulative *= (1 + r)
                total_days = len(daily_returns)
                if total_days > 0 and cumulative > 0:
                    metrics.annualized_return = (
                        cumulative ** (365 / total_days)
                    ) - 1

        return metrics

    def _calculate_risk_metrics(
        self,
        equity_curve: list[BacktestEquity],
    ) -> RiskMetrics:
        """Calculate risk-based metrics."""
        metrics = RiskMetrics()

        if not equity_curve:
            return metrics

        # Max drawdown
        drawdowns = [e.drawdown for e in equity_curve]
        metrics.max_drawdown = max(drawdowns) if drawdowns else 0.0
        metrics.avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 0.0

        # Drawdown duration
        in_drawdown = False
        drawdown_start: datetime | None = None
        max_duration = timedelta(0)

        for equity in equity_curve:
            if equity.drawdown > 0.001:  # 0.1% threshold
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = equity.timestamp
            else:
                if in_drawdown and drawdown_start:
                    duration = equity.timestamp - drawdown_start
                    if duration > max_duration:
                        max_duration = duration
                in_drawdown = False
                drawdown_start = None

        if max_duration.total_seconds() > 0:
            metrics.max_drawdown_duration_hours = max_duration.total_seconds() / 3600

        # Volatility from returns
        returns = self._calculate_period_returns(equity_curve)
        if len(returns) >= 2:
            metrics.volatility = statistics.stdev(returns)

            # Downside volatility (only negative returns)
            downside = [r for r in returns if r < 0]
            if len(downside) >= 2:
                metrics.downside_volatility = statistics.stdev(downside)

            # VaR and CVaR
            sorted_returns = sorted(returns)
            var_idx = int(len(sorted_returns) * 0.05)
            if var_idx > 0:
                metrics.var_95 = sorted_returns[var_idx]
                metrics.cvar_95 = sum(sorted_returns[:var_idx]) / var_idx

        return metrics

    def _calculate_trade_metrics(
        self,
        trades: list[BacktestTrade],
    ) -> TradeMetrics:
        """Calculate trade-level statistics."""
        metrics = TradeMetrics()

        if not trades:
            return metrics

        metrics.total_trades = len(trades)

        winning = [t for t in trades if t.realized_pnl > 0]
        losing = [t for t in trades if t.realized_pnl < 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = len(winning) / len(trades) if trades else 0.0

        if winning:
            metrics.avg_win = sum(t.realized_pnl for t in winning) / len(winning)
            metrics.largest_win = max(t.realized_pnl for t in winning)

        if losing:
            metrics.avg_loss = sum(t.realized_pnl for t in losing) / len(losing)
            metrics.largest_loss = min(t.realized_pnl for t in losing)

        total_pnl = sum(t.realized_pnl for t in trades)
        metrics.avg_trade = total_pnl / len(trades) if trades else 0.0

        # Hold times
        hold_times = [
            t.hold_time_minutes
            for t in trades
            if t.hold_time_minutes is not None
        ]
        if hold_times:
            metrics.avg_hold_time_minutes = sum(hold_times) / len(hold_times)
            metrics.median_hold_time_minutes = statistics.median(hold_times)

        return metrics

    def _calculate_exposure_metrics(
        self,
        equity_curve: list[BacktestEquity],
        initial_bankroll: float,
    ) -> ExposureMetrics:
        """Calculate exposure statistics."""
        metrics = ExposureMetrics()

        if not equity_curve:
            return metrics

        exposures = [e.total_exposure for e in equity_curve]
        positions = [e.position_count for e in equity_curve]

        metrics.avg_exposure = sum(exposures) / len(exposures) if exposures else 0.0
        metrics.max_exposure = max(exposures) if exposures else 0.0
        metrics.avg_position_count = sum(positions) / len(positions) if positions else 0.0
        metrics.max_position_count = max(positions) if positions else 0

        if initial_bankroll > 0:
            metrics.exposure_utilization = metrics.avg_exposure / initial_bankroll

        return metrics

    def _calculate_ratio_metrics(
        self,
        returns: ReturnMetrics,
        risk: RiskMetrics,
        trades: TradeMetrics,
    ) -> RatioMetrics:
        """Calculate risk-adjusted return ratios."""
        metrics = RatioMetrics()

        # Sharpe ratio (from annualized return and volatility)
        if risk.volatility and risk.volatility > 0 and returns.annualized_return:
            # Annualize volatility
            ann_vol = risk.volatility * (252 ** 0.5)  # Assuming daily data
            if ann_vol > 0:
                metrics.sharpe_ratio = returns.annualized_return / ann_vol

        # Sortino ratio (using downside volatility)
        if risk.downside_volatility and risk.downside_volatility > 0:
            if returns.annualized_return:
                ann_downside = risk.downside_volatility * (252 ** 0.5)
                if ann_downside > 0:
                    metrics.sortino_ratio = returns.annualized_return / ann_downside

        # Calmar ratio (return / max drawdown)
        if risk.max_drawdown > 0 and returns.annualized_return:
            metrics.calmar_ratio = returns.annualized_return / risk.max_drawdown

        # Profit factor (gross profit / gross loss)
        if trades.avg_loss < 0 and trades.losing_trades > 0:
            gross_profit = trades.avg_win * trades.winning_trades
            gross_loss = abs(trades.avg_loss * trades.losing_trades)
            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss

        # Expectancy (expected value per trade)
        if trades.total_trades > 0:
            metrics.expectancy = trades.avg_trade

        return metrics

    def _calculate_edge_calibration(
        self,
        trades: list[BacktestTrade],
    ) -> list[EdgeCalibrationBucket]:
        """Calculate edge vs realized return calibration."""
        buckets = {
            "0.00-0.05": {"edges": [], "returns": [], "wins": 0},
            "0.05-0.10": {"edges": [], "returns": [], "wins": 0},
            "0.10-0.15": {"edges": [], "returns": [], "wins": 0},
            "0.15-0.20": {"edges": [], "returns": [], "wins": 0},
            "0.20+": {"edges": [], "returns": [], "wins": 0},
        }

        for trade in trades:
            if trade.entry_edge is None or trade.return_pct is None:
                continue

            edge = trade.entry_edge
            if edge < 0.05:
                key = "0.00-0.05"
            elif edge < 0.10:
                key = "0.05-0.10"
            elif edge < 0.15:
                key = "0.10-0.15"
            elif edge < 0.20:
                key = "0.15-0.20"
            else:
                key = "0.20+"

            buckets[key]["edges"].append(edge)
            buckets[key]["returns"].append(trade.return_pct)
            if trade.return_pct > 0:
                buckets[key]["wins"] += 1

        results = []
        for bucket_name, data in buckets.items():
            if not data["edges"]:
                continue

            expected = sum(data["edges"]) / len(data["edges"])
            actual = sum(data["returns"]) / len(data["returns"])

            results.append(EdgeCalibrationBucket(
                edge_bucket=bucket_name,
                trade_count=len(data["edges"]),
                expected_return=expected,
                actual_return=actual,
                win_rate=data["wins"] / len(data["edges"]),
                calibration_error=expected - actual,
            ))

        return results

    def _calculate_outcome_breakdown(
        self,
        trades: list[BacktestTrade],
    ) -> list[OutcomeBreakdown]:
        """Calculate performance by outcome type."""
        outcomes = {}

        for trade in trades:
            outcome = trade.outcome
            if outcome not in outcomes:
                outcomes[outcome] = {
                    "trades": [],
                    "winning": 0,
                    "edges": [],
                }

            outcomes[outcome]["trades"].append(trade)
            if trade.realized_pnl > 0:
                outcomes[outcome]["winning"] += 1
            if trade.entry_edge is not None:
                outcomes[outcome]["edges"].append(trade.entry_edge)

        results = []
        for outcome, data in outcomes.items():
            trades_list = data["trades"]
            total_pnl = sum(t.realized_pnl for t in trades_list)

            results.append(OutcomeBreakdown(
                outcome=outcome,
                trades=len(trades_list),
                winning=data["winning"],
                losing=len(trades_list) - data["winning"],
                win_rate=data["winning"] / len(trades_list) if trades_list else 0,
                total_pnl=total_pnl,
                avg_pnl=total_pnl / len(trades_list) if trades_list else 0,
                avg_edge=(
                    sum(data["edges"]) / len(data["edges"])
                    if data["edges"]
                    else None
                ),
            ))

        return results

    def _aggregate_daily_returns(
        self,
        equity_curve: list[BacktestEquity],
    ) -> list[float]:
        """Aggregate returns by day."""
        if len(equity_curve) < 2:
            return []

        # Group by date
        by_date: dict[str, list[BacktestEquity]] = {}
        for eq in equity_curve:
            date_str = eq.timestamp.date().isoformat()
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(eq)

        # Get last equity of each day
        daily_values = []
        for date_str in sorted(by_date.keys()):
            day_eqs = by_date[date_str]
            last_eq = day_eqs[-1]
            daily_values.append(last_eq.bankroll + last_eq.total_pnl)

        # Calculate returns
        returns = []
        for i in range(1, len(daily_values)):
            if daily_values[i - 1] > 0:
                ret = (daily_values[i] / daily_values[i - 1]) - 1
                returns.append(ret)

        return returns

    def _calculate_period_returns(
        self,
        equity_curve: list[BacktestEquity],
    ) -> list[float]:
        """Calculate returns between equity snapshots."""
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            curr = equity_curve[i]
            prev_value = prev.bankroll + prev.total_pnl
            curr_value = curr.bankroll + curr.total_pnl
            if prev_value > 0:
                returns.append((curr_value / prev_value) - 1)
        return returns


def generate_backtest_report(
    metrics: BacktestMetricsResult,
    output_dir: Path,
    include_plots: bool = True,
) -> None:
    """Generate backtest report files.

    Args:
        metrics: Calculated metrics.
        output_dir: Directory to write reports.
        include_plots: Whether to generate plots (requires matplotlib).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write summary text file
    _write_summary(metrics, output_dir / "backtest_summary.txt")

    # Write JSON file
    _write_json(metrics, output_dir / "backtest_metrics.json")

    # Write CSV files
    _write_edge_calibration_csv(metrics, output_dir / "edge_calibration.csv")
    _write_outcome_breakdown_csv(metrics, output_dir / "outcome_breakdown.csv")

    logger.info("backtest_report_generated", output_dir=str(output_dir))


def _write_summary(metrics: BacktestMetricsResult, path: Path) -> None:
    """Write summary text file."""
    lines = [
        "=" * 60,
        "STRATEGY BACKTEST REPORT",
        "=" * 60,
        "",
        f"Backtest ID: {metrics.backtest_id}",
        f"Config Hash: {metrics.strategy_config_hash}",
        f"Period: {metrics.backtest_period_start} to {metrics.backtest_period_end}",
        f"Duration: {metrics.backtest_duration_hours:.1f} hours",
        "",
        "-" * 60,
        "RETURN METRICS",
        "-" * 60,
        f"Initial Bankroll:    ${metrics.initial_bankroll:,.2f}",
        f"Final Bankroll:      ${metrics.final_bankroll:,.2f}",
        f"Total Return:        ${metrics.returns.total_return:,.2f} ({metrics.returns.total_return_pct:.2%})",
        f"Annualized Return:   {_fmt_pct(metrics.returns.annualized_return)}",
        f"Best Day:            {_fmt_pct(metrics.returns.best_day)}",
        f"Worst Day:           {_fmt_pct(metrics.returns.worst_day)}",
        f"Positive Days:       {metrics.returns.positive_days}",
        f"Negative Days:       {metrics.returns.negative_days}",
        "",
        "-" * 60,
        "RISK METRICS",
        "-" * 60,
        f"Max Drawdown:        {metrics.risk.max_drawdown:.2%}",
        f"Avg Drawdown:        {metrics.risk.avg_drawdown:.2%}",
        f"Max DD Duration:     {_fmt_hours(metrics.risk.max_drawdown_duration_hours)}",
        f"Volatility:          {_fmt_pct(metrics.risk.volatility)}",
        f"Downside Vol:        {_fmt_pct(metrics.risk.downside_volatility)}",
        f"VaR (95%):           {_fmt_pct(metrics.risk.var_95)}",
        "",
        "-" * 60,
        "RISK-ADJUSTED RATIOS",
        "-" * 60,
        f"Sharpe Ratio:        {_fmt_ratio(metrics.ratios.sharpe_ratio)}",
        f"Sortino Ratio:       {_fmt_ratio(metrics.ratios.sortino_ratio)}",
        f"Calmar Ratio:        {_fmt_ratio(metrics.ratios.calmar_ratio)}",
        f"Profit Factor:       {_fmt_ratio(metrics.ratios.profit_factor)}",
        f"Expectancy:          ${_fmt_value(metrics.ratios.expectancy)}",
        "",
        "-" * 60,
        "TRADE STATISTICS",
        "-" * 60,
        f"Total Trades:        {metrics.trades.total_trades}",
        f"Winning Trades:      {metrics.trades.winning_trades}",
        f"Losing Trades:       {metrics.trades.losing_trades}",
        f"Win Rate:            {metrics.trades.win_rate:.2%}",
        f"Avg Win:             ${metrics.trades.avg_win:,.2f}",
        f"Avg Loss:            ${metrics.trades.avg_loss:,.2f}",
        f"Largest Win:         ${metrics.trades.largest_win:,.2f}",
        f"Largest Loss:        ${metrics.trades.largest_loss:,.2f}",
        f"Avg Hold Time:       {_fmt_minutes(metrics.trades.avg_hold_time_minutes)}",
        "",
        "-" * 60,
        "EXPOSURE METRICS",
        "-" * 60,
        f"Avg Exposure:        ${metrics.exposure.avg_exposure:,.2f}",
        f"Max Exposure:        ${metrics.exposure.max_exposure:,.2f}",
        f"Avg Positions:       {metrics.exposure.avg_position_count:.1f}",
        f"Max Positions:       {metrics.exposure.max_position_count}",
        f"Capital Utilization: {metrics.exposure.exposure_utilization:.2%}",
        "",
        "-" * 60,
        "EDGE CALIBRATION",
        "-" * 60,
    ]

    for bucket in metrics.edge_calibration:
        lines.append(
            f"  {bucket.edge_bucket}: n={bucket.trade_count}, "
            f"expected={bucket.expected_return:.2%}, "
            f"actual={bucket.actual_return:.2%}, "
            f"error={bucket.calibration_error:.2%}"
        )

    lines.extend([
        "",
        "-" * 60,
        "OUTCOME BREAKDOWN",
        "-" * 60,
    ])

    for outcome in metrics.outcome_breakdown:
        lines.append(
            f"  {outcome.outcome}: trades={outcome.trades}, "
            f"win_rate={outcome.win_rate:.2%}, "
            f"pnl=${outcome.total_pnl:,.2f}"
        )

    lines.extend([
        "",
        f"Total Fixtures: {metrics.fixture_count}",
        "",
        "=" * 60,
    ])

    path.write_text("\n".join(lines))


def _write_json(metrics: BacktestMetricsResult, path: Path) -> None:
    """Write JSON file."""
    with open(path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)


def _write_edge_calibration_csv(metrics: BacktestMetricsResult, path: Path) -> None:
    """Write edge calibration CSV."""
    lines = ["edge_bucket,trade_count,expected_return,actual_return,win_rate,calibration_error"]
    for bucket in metrics.edge_calibration:
        lines.append(
            f"{bucket.edge_bucket},{bucket.trade_count},"
            f"{bucket.expected_return:.4f},{bucket.actual_return:.4f},"
            f"{bucket.win_rate:.4f},{bucket.calibration_error:.4f}"
        )
    path.write_text("\n".join(lines))


def _write_outcome_breakdown_csv(metrics: BacktestMetricsResult, path: Path) -> None:
    """Write outcome breakdown CSV."""
    lines = ["outcome,trades,winning,losing,win_rate,total_pnl,avg_pnl,avg_edge"]
    for outcome in metrics.outcome_breakdown:
        lines.append(
            f"{outcome.outcome},{outcome.trades},{outcome.winning},{outcome.losing},"
            f"{outcome.win_rate:.4f},{outcome.total_pnl:.2f},{outcome.avg_pnl:.2f},"
            f"{outcome.avg_edge or 'N/A'}"
        )
    path.write_text("\n".join(lines))


def _fmt_pct(value: float | None) -> str:
    """Format percentage."""
    return f"{value:.2%}" if value is not None else "N/A"


def _fmt_ratio(value: float | None) -> str:
    """Format ratio."""
    return f"{value:.3f}" if value is not None else "N/A"


def _fmt_value(value: float | None) -> str:
    """Format value."""
    return f"{value:,.2f}" if value is not None else "N/A"


def _fmt_hours(value: float | None) -> str:
    """Format hours."""
    return f"{value:.1f} hours" if value is not None else "N/A"


def _fmt_minutes(value: float | None) -> str:
    """Format minutes."""
    if value is None:
        return "N/A"
    if value < 60:
        return f"{value:.1f} min"
    return f"{value / 60:.1f} hours"
