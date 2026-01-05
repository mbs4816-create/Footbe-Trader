"""Report generation for backtest results.

Generates:
- Metrics tables (console + CSV)
- Calibration data (JSON)
- Calibration plots (PNG) - optional matplotlib
- Model comparison reports
- Ablation study results
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from footbe_trader.modeling.backtest import BacktestResult


def generate_report(
    result: BacktestResult,
    output_dir: str | Path,
    generate_plots: bool = True,
    calibration_result: Any | None = None,
    ablation_result: Any | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate full backtest report.
    
    Args:
        result: BacktestResult from backtest.
        output_dir: Directory to write reports.
        generate_plots: Whether to generate PNG plots.
        calibration_result: Optional calibration results.
        ablation_result: Optional ablation study results.
        hyperparams: Optional best hyperparameters.
        
    Returns:
        Dict mapping report type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{result.mode}_{result.model_name}_{timestamp}"
    
    files = {}
    
    # Summary table
    summary_path = output_dir / f"{prefix}_summary.txt"
    _write_summary(result, summary_path, calibration_result, hyperparams)
    files["summary"] = summary_path
    
    # Fold details CSV
    folds_path = output_dir / f"{prefix}_folds.csv"
    _write_folds_csv(result, folds_path)
    files["folds_csv"] = folds_path
    
    # Full results JSON
    json_path = output_dir / f"{prefix}_results.json"
    _write_json(result, json_path, calibration_result, ablation_result, hyperparams)
    files["json"] = json_path
    
    # Calibration data
    if result.aggregate_metrics:
        cal_path = output_dir / f"{prefix}_calibration.json"
        _write_calibration_json(result, cal_path, calibration_result)
        files["calibration"] = cal_path
    
    # Ablation report
    if ablation_result:
        ablation_path = output_dir / f"{prefix}_ablation.txt"
        _write_ablation_report(ablation_result, ablation_path)
        files["ablation"] = ablation_path
    
    # Plots
    if generate_plots and result.aggregate_metrics:
        try:
            plot_path = output_dir / f"{prefix}_calibration.png"
            _generate_calibration_plot(result, plot_path, calibration_result)
            files["calibration_plot"] = plot_path
        except ImportError:
            print("matplotlib not available, skipping plot generation")
    
    return files


def _write_summary(
    result: BacktestResult,
    path: Path,
    calibration_result: Any | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> None:
    """Write summary report to text file."""
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Mode: {result.mode}")
    lines.append(f"Model: {result.model_name}")
    lines.append(f"Number of folds: {len(result.folds)}")
    lines.append("")
    
    # Best hyperparameters
    if hyperparams:
        lines.append("-" * 40)
        lines.append("BEST HYPERPARAMETERS")
        lines.append("-" * 40)
        for key, value in hyperparams.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    
    if result.aggregate_metrics:
        lines.append("-" * 40)
        lines.append("AGGREGATE METRICS (Model)")
        lines.append("-" * 40)
        m = result.aggregate_metrics
        lines.append(f"Log Loss:     {m.log_loss:.4f}")
        lines.append(f"Brier Score:  {m.brier_score:.4f}")
        lines.append(f"Accuracy:     {m.accuracy:.4f}")
        lines.append(f"N Samples:    {m.n_samples}")
        lines.append("")
        lines.append("Class Distribution:")
        for cls, count in m.class_counts.items():
            acc = m.class_accuracies.get(cls, 0)
            lines.append(f"  {cls}: {count} samples, {acc:.4f} accuracy")
        lines.append("")
    
    if result.baseline_metrics:
        lines.append("-" * 40)
        lines.append("BASELINE METRICS (home_advantage)")
        lines.append("-" * 40)
        b = result.baseline_metrics
        lines.append(f"Log Loss:     {b.log_loss:.4f}")
        lines.append(f"Brier Score:  {b.brier_score:.4f}")
        lines.append(f"Accuracy:     {b.accuracy:.4f}")
        lines.append("")
    
    # Calibration results
    if calibration_result:
        lines.append("-" * 40)
        lines.append("CALIBRATION (Temperature Scaling)")
        lines.append("-" * 40)
        lines.append(f"Temperature:  {calibration_result.temperature:.4f}")
        lines.append(f"Pre-cal ECE:  {calibration_result.pre_calibration_ece:.4f}")
        lines.append(f"Post-cal ECE: {calibration_result.post_calibration_ece:.4f}")
        lines.append(f"ECE Improvement: {calibration_result.pre_calibration_ece - calibration_result.post_calibration_ece:.4f}")
        lines.append("")
    
    if result.bootstrap_comparison:
        lines.append("-" * 40)
        lines.append("BOOTSTRAP COMPARISON (Model - Baseline)")
        lines.append("-" * 40)
        c = result.bootstrap_comparison
        lines.append(f"Mean Diff:    {c.mean_diff:.4f}")
        lines.append(f"Std Diff:     {c.std_diff:.4f}")
        lines.append(f"95% CI:       [{c.ci_lower:.4f}, {c.ci_upper:.4f}]")
        lines.append(f"P-value:      {c.p_value:.4f}")
        lines.append("")
        
        if c.mean_diff < 0 and c.ci_upper < 0:
            lines.append(">>> Model SIGNIFICANTLY BETTER than baseline <<<")
        elif c.mean_diff < 0:
            lines.append("Model better than baseline (not statistically significant)")
        elif c.mean_diff > 0 and c.ci_lower > 0:
            lines.append(">>> Model SIGNIFICANTLY WORSE than baseline <<<")
        else:
            lines.append("No significant difference from baseline")
        lines.append("")
    
    lines.append("-" * 40)
    lines.append("PER-FOLD RESULTS")
    lines.append("-" * 40)
    lines.append(f"{'Fold':<40} {'Train':>8} {'Test':>8} {'LogLoss':>10}")
    lines.append("-" * 70)
    
    for fold in result.folds:
        lines.append(
            f"{fold.fold_name:<40} "
            f"{fold.train_size:>8} "
            f"{fold.test_size:>8} "
            f"{fold.metrics.log_loss:>10.4f}"
        )
    
    lines.append("=" * 60)
    
    path.write_text("\n".join(lines))
    print(f"Summary written to {path}")


def _write_folds_csv(result: BacktestResult, path: Path) -> None:
    """Write fold details to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold_name",
            "train_size",
            "test_size",
            "log_loss",
            "brier_score",
            "accuracy",
            "accuracy_H",
            "accuracy_D",
            "accuracy_A",
        ])
        
        for fold in result.folds:
            m = fold.metrics
            writer.writerow([
                fold.fold_name,
                fold.train_size,
                fold.test_size,
                f"{m.log_loss:.6f}",
                f"{m.brier_score:.6f}",
                f"{m.accuracy:.6f}",
                f"{m.class_accuracies.get('H', 0):.6f}",
                f"{m.class_accuracies.get('D', 0):.6f}",
                f"{m.class_accuracies.get('A', 0):.6f}",
            ])
    
    print(f"Folds CSV written to {path}")


def _write_json(
    result: BacktestResult,
    path: Path,
    calibration_result: Any | None = None,
    ablation_result: Any | None = None,
    hyperparams: dict[str, Any] | None = None,
) -> None:
    """Write full results to JSON."""
    data = result.to_dict()
    
    # Add metadata
    data["generated_at"] = datetime.now().isoformat()
    
    # Add calibration results
    if calibration_result:
        data["calibration"] = calibration_result.to_dict()
    
    # Add ablation results
    if ablation_result:
        data["ablation"] = ablation_result.to_dict()
    
    # Add hyperparameters
    if hyperparams:
        data["best_hyperparams"] = hyperparams
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Results JSON written to {path}")


def _write_calibration_json(
    result: BacktestResult,
    path: Path,
    calibration_result: Any | None = None,
) -> None:
    """Write calibration data to JSON for external plotting."""
    if not result.aggregate_metrics:
        return
    
    data = {
        "model_name": result.model_name,
        "mode": result.mode,
        "calibration": result.aggregate_metrics.calibration_data,
    }
    
    # Add temperature scaling results
    if calibration_result:
        data["temperature_scaling"] = calibration_result.to_dict()
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Calibration data written to {path}")


def _write_ablation_report(ablation_result: Any, path: Path) -> None:
    """Write ablation study report to text file."""
    from footbe_trader.modeling.ablation import format_ablation_report
    
    report = format_ablation_report(ablation_result)
    path.write_text(report)
    print(f"Ablation report written to {path}")


def _generate_calibration_plot(
    result: BacktestResult,
    path: Path,
    calibration_result: Any | None = None,
) -> None:
    """Generate calibration plot using matplotlib."""
    import matplotlib.pyplot as plt
    
    if not result.aggregate_metrics:
        return
    
    cal_data = result.aggregate_metrics.calibration_data
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    classes = ["H", "D", "A"]
    class_names = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    colors = {"H": "blue", "D": "gray", "A": "red"}
    
    for i, cls in enumerate(classes):
        ax = axes[i]
        data = cal_data.get(cls, {})
        
        centers = data.get("bin_centers", [])
        freqs = data.get("bin_true_freqs", [])
        counts = data.get("bin_counts", [])
        
        if centers and freqs:
            # Scatter with size proportional to count
            sizes = [max(10, min(100, c / 2)) for c in counts]
            ax.scatter(centers, freqs, s=sizes, c=colors[cls], alpha=0.7)
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(f"{class_names[cls]} Calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    
    # Add calibration info to figure
    if calibration_result:
        fig.suptitle(
            f"Calibration (T={calibration_result.temperature:.2f}, "
            f"ECE: {calibration_result.pre_calibration_ece:.4f} → {calibration_result.post_calibration_ece:.4f})",
            y=1.02
        )
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Calibration plot written to {path}")


def print_comparison_summary(
    result: BacktestResult,
    calibration_result: Any | None = None,
) -> None:
    """Print quick comparison summary to console."""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    
    if result.aggregate_metrics and result.baseline_metrics:
        m = result.aggregate_metrics
        b = result.baseline_metrics
        
        print(f"\n{'Metric':<20} {'Model':>12} {'Baseline':>12} {'Diff':>12}")
        print("-" * 58)
        
        ll_diff = m.log_loss - b.log_loss
        bs_diff = m.brier_score - b.brier_score
        acc_diff = m.accuracy - b.accuracy
        
        print(f"{'Log Loss':<20} {m.log_loss:>12.4f} {b.log_loss:>12.4f} {ll_diff:>+12.4f}")
        print(f"{'Brier Score':<20} {m.brier_score:>12.4f} {b.brier_score:>12.4f} {bs_diff:>+12.4f}")
        print(f"{'Accuracy':<20} {m.accuracy:>12.4f} {b.accuracy:>12.4f} {acc_diff:>+12.4f}")
        
        if result.bootstrap_comparison:
            c = result.bootstrap_comparison
            print(f"\nBootstrap: mean_diff={c.mean_diff:.4f}, "
                  f"95% CI=[{c.ci_lower:.4f}, {c.ci_upper:.4f}], "
                  f"p={c.p_value:.4f}")
            
            if c.mean_diff < 0 and c.ci_upper < 0:
                print("\n✓ Model shows SIGNIFICANT improvement over baseline")
            elif c.mean_diff < 0:
                print("\n~ Model shows improvement (not statistically significant)")
            else:
                print("\n✗ Model does not improve over baseline")
    
    # Calibration summary
    if calibration_result:
        print("\n" + "-" * 50)
        print("CALIBRATION")
        print("-" * 50)
        print(f"Temperature: {calibration_result.temperature:.4f}")
        print(f"ECE: {calibration_result.pre_calibration_ece:.4f} → {calibration_result.post_calibration_ece:.4f}")
        print(f"NLL: {calibration_result.pre_calibration_nll:.4f} → {calibration_result.post_calibration_nll:.4f}")
    
    print("\n" + "=" * 50)


def generate_model_comparison_report(
    results: list[BacktestResult],
    output_dir: str | Path,
) -> Path:
    """Generate comparison report for multiple models.
    
    Args:
        results: List of BacktestResults to compare.
        output_dir: Directory to write report.
        
    Returns:
        Path to comparison report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"model_comparison_{timestamp}.txt"
    
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Sort by log loss
    sorted_results = sorted(results, key=lambda r: r.aggregate_metrics.log_loss if r.aggregate_metrics else float('inf'))
    
    lines.append(f"{'Rank':<6} {'Model':<25} {'Log Loss':>12} {'Brier':>12} {'Accuracy':>12}")
    lines.append("-" * 70)
    
    for i, r in enumerate(sorted_results, 1):
        if r.aggregate_metrics:
            m = r.aggregate_metrics
            lines.append(f"{i:<6} {r.model_name:<25} {m.log_loss:>12.4f} {m.brier_score:>12.4f} {m.accuracy:>12.4f}")
    
    lines.append("")
    lines.append("=" * 70)
    
    path.write_text("\n".join(lines))
    print(f"Model comparison written to {path}")
    
    return path
