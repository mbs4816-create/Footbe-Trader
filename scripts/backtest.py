#!/usr/bin/env python3
"""Backtest CLI for 3-way match prediction models.

Usage:
    python scripts/backtest.py --config configs/dev.yaml
    python scripts/backtest.py --mode season --model multinomial_logistic
    python scripts/backtest.py --mode rolling --min-train-size 200
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.modeling.backtest import run_backtest
from footbe_trader.modeling.reports import generate_report, print_comparison_summary
from footbe_trader.storage.database import Database


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest for 3-way match prediction models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dev.yaml",
        help="Path to configuration file",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["season", "rolling"],
        default="season",
        help="Backtest mode: season-by-season or rolling matchweek",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="multinomial_logistic",
        help="Model to evaluate (home_advantage, multinomial_logistic)",
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        default="home_advantage",
        help="Baseline model for comparison",
    )
    
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Rolling window for feature computation",
    )
    
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=1,
        help="Minimum training seasons for season mode",
    )
    
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=100,
        help="Minimum training samples for rolling mode",
    )
    
    parser.add_argument(
        "--test-window-days",
        type=int,
        default=7,
        help="Test window days for rolling mode",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for gradient descent",
    )
    
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="Training iterations",
    )
    
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.01,
        help="L2 regularization strength",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output reports",
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    parser.add_argument(
        "--season",
        type=str,
        help="Filter to specific season (e.g., 2023)",
    )
    
    parser.add_argument(
        "--league-id",
        type=int,
        default=39,
        help="League ID to backtest (default: 39 = EPL)",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
        db_path = config.database.path
    else:
        print(f"Config not found: {config_path}, using default db path")
        db_path = "data/footbe_dev.db"
    
    # Check database exists
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Run ingestion first: python scripts/ingest_epl.py --config configs/dev.yaml")
        return 1
    
    # Load fixtures
    print(f"Loading fixtures from {db_path}...")
    db = Database(db_path)
    
    # Get all seasons and load fixtures from each
    season_counts = db.get_fixtures_count_by_season()
    fixtures = []
    
    for season in season_counts.keys():
        season_fixtures = db.get_fixtures_by_season(season)
        fixtures.extend(season_fixtures)
    
    # Filter by season if specified
    if args.season:
        fixtures = [f for f in fixtures if f.season == int(args.season)]
    
    print(f"Loaded {len(fixtures)} fixtures")
    
    # Count completed
    completed = [f for f in fixtures if f.status == "FT"]
    print(f"Completed fixtures: {len(completed)}")
    
    if len(completed) < 50:
        print("Not enough completed fixtures for meaningful backtest")
        return 1
    
    # Run backtest
    print(f"\nRunning {args.mode} backtest...")
    print(f"Model: {args.model}")
    print(f"Baseline: {args.baseline}")
    
    result = run_backtest(
        fixtures=fixtures,
        mode=args.mode,
        model_name=args.model,
        baseline_model_name=args.baseline,
        rolling_window=args.rolling_window,
        min_train_seasons=args.min_train_seasons,
        min_train_size=args.min_train_size,
        test_window_days=args.test_window_days,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        regularization=args.regularization,
    )
    
    if len(result.folds) == 0:
        print("No folds completed - not enough data")
        return 1
    
    # Print summary
    print_comparison_summary(result)
    
    # Generate reports
    print(f"\nGenerating reports to {args.output_dir}/...")
    files = generate_report(
        result,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
    )
    
    print("\nGenerated files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
