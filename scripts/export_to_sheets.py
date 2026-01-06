#!/usr/bin/env python3
"""Export trading reports to Google Sheets.

This script exports live trading data with a narrative summary to Google Sheets.
Can be run manually or scheduled to run every 4 hours.

Usage:
    # Run once
    python scripts/export_to_sheets.py
    
    # Run in scheduled mode (every 4 hours)
    python scripts/export_to_sheets.py --scheduled
"""

import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from footbe_trader.reporting.sheets import export_with_narrative, export_all_reports

DB_PATH = Path(__file__).parent.parent / "data" / "footbe_dev.db"
INTERVAL_HOURS = 4


def run_export():
    """Run a single export to Google Sheets."""
    print(f"\n{'='*60}")
    print(f"Starting export at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    try:
        results = export_with_narrative(conn)
        print("\nExport Results:")
        for sheet, count in results.get("exports", {}).items():
            print(f"  - {sheet}: {count} rows")
        print(f"  - Narrative: Generated" if results.get("narrative_generated") else "")
        print("\nExport complete!")
    except Exception as e:
        print(f"Error during export: {e}")
        raise
    finally:
        conn.close()


def run_scheduled():
    """Run exports on a schedule (every 4 hours)."""
    print(f"Starting scheduled export mode (every {INTERVAL_HOURS} hours)")
    print("Press Ctrl+C to stop\n")
    
    while True:
        try:
            run_export()
            
            # Calculate next run time
            next_run = datetime.now().timestamp() + (INTERVAL_HOURS * 3600)
            next_run_str = datetime.fromtimestamp(next_run).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\nNext export scheduled for: {next_run_str}")
            print(f"Sleeping for {INTERVAL_HOURS} hours...")
            
            time.sleep(INTERVAL_HOURS * 3600)
            
        except KeyboardInterrupt:
            print("\n\nScheduled export stopped by user.")
            break
        except Exception as e:
            print(f"Error during scheduled export: {e}")
            print("Will retry at next scheduled interval...")
            time.sleep(INTERVAL_HOURS * 3600)


def main():
    parser = argparse.ArgumentParser(
        description="Export trading reports to Google Sheets"
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help=f"Run in scheduled mode (every {INTERVAL_HOURS} hours)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Export paper trading data instead of live",
    )
    
    args = parser.parse_args()
    
    if args.paper:
        # Paper trading export
        print("Exporting paper trading data...")
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            results = export_all_reports(conn, live_only=False)
            print("Export Results:")
            for sheet, count in results.items():
                print(f"  - {sheet}: {count} rows")
        finally:
            conn.close()
    elif args.scheduled:
        run_scheduled()
    else:
        run_export()


if __name__ == "__main__":
    main()
