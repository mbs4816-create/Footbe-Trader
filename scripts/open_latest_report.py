#!/usr/bin/env python3
"""CLI script to open the latest report in a browser.

This script finds the most recent report (run, daily, or weekly) and opens
it in the default web browser.

Usage:
    # Open the latest report (any type)
    python scripts/open_latest_report.py
    
    # Open the latest run report
    python scripts/open_latest_report.py --type run
    
    # Open the latest daily report
    python scripts/open_latest_report.py --type daily
    
    # Open the latest weekly report
    python scripts/open_latest_report.py --type weekly
    
    # Open the navigation index
    python scripts/open_latest_report.py --index
    
    # Just print the path without opening
    python scripts/open_latest_report.py --print
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Open the latest trading agent report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Reports directory (default: reports)",
    )
    
    parser.add_argument(
        "--type",
        "-t",
        choices=["run", "daily", "weekly", "any"],
        default="any",
        help="Type of report to open (default: any)",
    )
    
    parser.add_argument(
        "--index",
        "-i",
        action="store_true",
        help="Open the navigation index instead of latest report",
    )
    
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        dest="print_only",
        help="Just print the path without opening",
    )
    
    parser.add_argument(
        "--markdown",
        "-m",
        action="store_true",
        help="Open Markdown version instead of HTML",
    )
    
    return parser.parse_args()


def find_latest_report(
    reports_dir: Path,
    report_type: str,
    prefer_html: bool = True,
) -> Path | None:
    """Find the most recent report of the given type.
    
    Args:
        reports_dir: Base reports directory.
        report_type: Type of report ('run', 'daily', 'weekly', or 'any').
        prefer_html: Prefer HTML over Markdown if both exist.
        
    Returns:
        Path to the latest report, or None if not found.
    """
    extension = ".html" if prefer_html else ".md"
    alt_extension = ".md" if prefer_html else ".html"
    
    search_dirs = []
    if report_type in ("run", "any"):
        search_dirs.append(reports_dir / "runs")
    if report_type in ("daily", "any"):
        search_dirs.append(reports_dir / "daily")
    if report_type in ("weekly", "any"):
        search_dirs.append(reports_dir / "weekly")
    
    latest_file: Path | None = None
    latest_mtime: float = 0
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # First try preferred extension
        for file_path in search_dir.glob(f"*{extension}"):
            mtime = file_path.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = file_path
        
        # If no preferred, try alternative
        if latest_file is None:
            for file_path in search_dir.glob(f"*{alt_extension}"):
                mtime = file_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = file_path
    
    return latest_file


def open_file(path: Path) -> bool:
    """Open a file in the default application.
    
    Args:
        path: Path to the file.
        
    Returns:
        True if successful.
    """
    path_str = str(path.absolute())
    
    # Use webbrowser for HTML files
    if path.suffix == ".html":
        webbrowser.open(f"file://{path_str}")
        return True
    
    # Use system default for other files
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", path_str], check=True)
        elif sys.platform == "win32":  # Windows
            subprocess.run(["start", "", path_str], shell=True, check=True)
        else:  # Linux and others
            subprocess.run(["xdg-open", path_str], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    reports_dir = Path(args.reports_dir)
    
    if not reports_dir.exists():
        print(f"Error: Reports directory not found: {reports_dir}")
        print("Run 'python scripts/build_report.py' to generate reports first.")
        return 1
    
    # Determine what to open
    if args.index:
        # Open index
        extension = ".md" if args.markdown else ".html"
        index_path = reports_dir / f"index{extension}"
        
        if not index_path.exists():
            # Try alternative extension
            alt_ext = ".html" if args.markdown else ".md"
            index_path = reports_dir / f"index{alt_ext}"
        
        if not index_path.exists():
            print(f"Error: Index not found in {reports_dir}")
            return 1
        
        target_path = index_path
    else:
        # Find latest report
        target_path = find_latest_report(
            reports_dir,
            args.type,
            prefer_html=not args.markdown,
        )
        
        if target_path is None:
            print(f"Error: No {args.type} reports found in {reports_dir}")
            return 1
    
    # Output or open
    if args.print_only:
        print(target_path.absolute())
    else:
        print(f"Opening: {target_path}")
        if not open_file(target_path):
            print(f"Error: Could not open file")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
