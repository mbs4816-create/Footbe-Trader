#!/usr/bin/env python3
"""CLI script to build trading agent reports.

This script generates Run, Daily, and Weekly reports from the trading agent's
database. Reports are output in both Markdown and HTML formats.

Usage:
    # Build all recent reports (last 7 days)
    python scripts/build_report.py
    
    # Build report for a specific run
    python scripts/build_report.py --run 123
    
    # Build report for a specific date
    python scripts/build_report.py --date 2024-01-15
    
    # Build weekly report for a specific week
    python scripts/build_report.py --week 2024-01-15
    
    # Rebuild the navigation index only
    python scripts/build_report.py --index
    
    # Build all reports for the last N days
    python scripts/build_report.py --days 14
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.logging import get_logger, setup_logging
from footbe_trader.reporting import ReportBuilder, ReportingConfig
from footbe_trader.storage.database import Database

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build trading agent reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default="data/footbe_trader.db",
        help="Path to database file (default: data/footbe_trader.db)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    
    parser.add_argument(
        "--run",
        type=int,
        help="Build report for a specific run ID",
    )
    
    parser.add_argument(
        "--date",
        type=str,
        help="Build daily report for a specific date (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--week",
        type=str,
        help="Build weekly report starting from date (YYYY-MM-DD, should be Monday)",
    )
    
    parser.add_argument(
        "--index",
        action="store_true",
        help="Only rebuild the navigation index",
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to build reports for (default: 7)",
    )
    
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML generation (Markdown only)",
    )
    
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip Markdown generation (HTML only)",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    logger.info("report_builder_starting", db=args.db, output=args.output)
    
    # Check database exists
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error("database_not_found", path=args.db)
        print(f"Error: Database not found at {args.db}")
        return 1
    
    # Connect to database
    db = Database(db_path)
    db.connect()
    
    try:
        # Configure reporting
        config = ReportingConfig(
            reports_dir=Path(args.output),
            generate_html=not args.no_html,
            generate_markdown=not args.no_markdown,
        )
        
        # Create builder
        builder = ReportBuilder(db.connection, config)
        
        # Determine what to build
        if args.index:
            # Index only
            logger.info("building_index_only")
            md_path, html_path = builder.build_index()
            print(f"Index built: {html_path or md_path}")
            
        elif args.run:
            # Specific run
            logger.info("building_run_report", run_id=args.run)
            report = builder.build_run_report(args.run)
            if report:
                print(f"Run report built: {report.html_path or report.markdown_path}")
                if report.artifact_path:
                    print(f"Artifact saved: {report.artifact_path}")
            else:
                print(f"Error: Run {args.run} not found")
                return 1
            
        elif args.date:
            # Specific date
            try:
                date = datetime.strptime(args.date, "%Y-%m-%d")
            except ValueError:
                print(f"Error: Invalid date format. Use YYYY-MM-DD")
                return 1
            
            logger.info("building_daily_report", date=args.date)
            report = builder.build_daily_report(date)
            print(f"Daily report built: {report.html_path or report.markdown_path}")
            
        elif args.week:
            # Specific week
            try:
                week_start = datetime.strptime(args.week, "%Y-%m-%d")
            except ValueError:
                print(f"Error: Invalid date format. Use YYYY-MM-DD")
                return 1
            
            logger.info("building_weekly_report", week_start=args.week)
            report = builder.build_weekly_report(week_start)
            print(f"Weekly report built: {report.html_path or report.markdown_path}")
            
        else:
            # Build all recent reports
            logger.info("building_all_reports", days=args.days)
            builder.build_all_reports(days_back=args.days)
            print(f"All reports built in {args.output}/")
        
        logger.info("report_builder_complete")
        return 0
        
    except Exception as e:
        logger.exception("report_builder_failed", error=str(e))
        print(f"Error: {e}")
        return 1
        
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
