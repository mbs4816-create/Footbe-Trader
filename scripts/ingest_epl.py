#!/usr/bin/env python3
"""EPL fixtures ingestion script.

This script ingests EPL fixtures and results from API-Football into the database.

Usage:
    # Ingest a range of seasons
    python scripts/ingest_epl.py --start 2020 --end 2025

    # Ingest specific seasons
    python scripts/ingest_epl.py --seasons 2023,2024,2025

    # Force re-ingestion (even if already done)
    python scripts/ingest_epl.py --start 2024 --end 2025 --force

    # Show available seasons
    python scripts/ingest_epl.py --list-seasons

    # Generate integrity report only
    python scripts/ingest_epl.py --report

    # Skip standings (faster)
    python scripts/ingest_epl.py --start 2024 --end 2025 --no-standings

Before running:
    1. Set FOOTBALL_API_KEY in your .env file
    2. Ensure config/dev.yaml exists
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

from footbe_trader.common.config import load_config
from footbe_trader.football.client import FootballApiClient, FootballApiError
from footbe_trader.football.ingestion import (
    EPL_LEAGUE_ID,
    EPLIngestion,
    generate_integrity_report,
)
from footbe_trader.storage.database import Database

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)

logger = structlog.get_logger()


async def list_available_seasons(config) -> list[int]:
    """List available EPL seasons from API."""
    async with FootballApiClient(config.football_api) as client:
        league, seasons = await client.get_league_info(EPL_LEAGUE_ID)
        print(f"\n📋 {league.name} ({league.country})")
        print(f"   League ID: {league.league_id}")
        print(f"\n📅 Available seasons:")
        for s in sorted(seasons, key=lambda x: x.year, reverse=True):
            current = " (current)" if s.current else ""
            print(f"   • {s.year}{current}")
        return [s.year for s in seasons]


async def run_ingestion(
    config,
    seasons: list[int],
    include_standings: bool = True,
    force: bool = False,
    raw_data_dir: Path | None = None,
) -> dict:
    """Run the ingestion process.

    Args:
        config: Application config.
        seasons: List of seasons to ingest.
        include_standings: Whether to ingest standings.
        force: Force re-ingestion.
        raw_data_dir: Directory for raw data.

    Returns:
        Summary dictionary.
    """
    print("\n" + "=" * 60)
    print("EPL Fixtures Ingestion")
    print("=" * 60)
    print(f"\n📅 Seasons to ingest: {sorted(seasons)}")
    print(f"📊 Include standings: {include_standings}")
    print(f"🔄 Force re-ingest: {force}")
    if raw_data_dir:
        print(f"💾 Raw data dir: {raw_data_dir}")
    print()

    # Initialize database
    db = Database(config.database.path)
    db.connect()
    db.migrate()

    try:
        async with FootballApiClient(
            config.football_api,
            raw_data_dir=raw_data_dir / "football" if raw_data_dir else None,
        ) as client:
            ingestion = EPLIngestion(
                client=client,
                db=db,
                raw_data_dir=raw_data_dir / "football" if raw_data_dir else None,
            )

            # Process each season
            total_fixtures = 0
            total_teams = 0
            total_standings = 0
            errors = []

            for season in sorted(seasons):
                print(f"\n🏃 Processing season {season}...")

                result = await ingestion.ingest_season(
                    season=season,
                    include_standings=include_standings,
                    force=force,
                )

                if result.fixtures_count > 0:
                    print(f"   ✅ Fixtures: {result.fixtures_count}")
                    print(f"   ✅ Teams: {result.teams_count}")
                    if include_standings:
                        print(f"   ✅ Standings: {result.standings_count}")
                elif result.errors:
                    for error in result.errors:
                        print(f"   ❌ {error}")
                else:
                    print(f"   ⏭️  Already ingested (use --force to re-ingest)")

                total_fixtures += result.fixtures_count
                total_teams += result.teams_count
                total_standings += result.standings_count
                errors.extend(result.errors)

        # Summary
        print("\n" + "-" * 60)
        print("📊 Ingestion Summary")
        print("-" * 60)
        print(f"   Total fixtures: {total_fixtures}")
        print(f"   Total teams: {total_teams}")
        if include_standings:
            print(f"   Total standings: {total_standings}")
        if errors:
            print(f"   ⚠️  Errors: {len(errors)}")

        return {
            "fixtures": total_fixtures,
            "teams": total_teams,
            "standings": total_standings,
            "errors": errors,
        }

    finally:
        db.close()


def show_integrity_report(config) -> None:
    """Show integrity report for ingested data."""
    print("\n" + "=" * 60)
    print("EPL Data Integrity Report")
    print("=" * 60)

    db = Database(config.database.path)
    db.connect()
    db.migrate()

    try:
        report = generate_integrity_report(db)

        print(f"\n📅 Generated at: {report['generated_at']}")
        print(f"\n📊 Totals:")
        print(f"   Total fixtures: {report['total_fixtures']}")
        print(f"   Total teams: {report['total_teams']}")

        print(f"\n📋 Fixtures per season:")
        for season, count in sorted(report["fixtures_per_season"].items()):
            print(f"   {season}: {count} fixtures")

        print(f"\n👥 Teams per season:")
        for season, count in sorted(report["teams_per_season"].items()):
            print(f"   {season}: {count} teams")

        print(f"\n⚠️  Missing scores per season:")
        for season, data in sorted(report["missing_scores_per_season"].items()):
            rate = data["rate"]
            missing = data["missing"]
            total = data["total_finished"]
            if total > 0:
                status = "✅" if missing == 0 else "⚠️"
                print(f"   {season}: {status} {missing}/{total} ({rate})")

        # Check for common issues
        print("\n🔍 Data Quality Checks:")

        # Expected fixtures per season (380 for a 20-team league)
        expected_fixtures = 380
        for season, count in sorted(report["fixtures_per_season"].items()):
            if count < expected_fixtures:
                print(f"   ⚠️  Season {season}: Only {count}/{expected_fixtures} fixtures")
            elif count == expected_fixtures:
                print(f"   ✅ Season {season}: Complete ({count} fixtures)")

        # Expected teams per season
        expected_teams = 20
        for season, count in sorted(report["teams_per_season"].items()):
            if count != expected_teams:
                print(f"   ⚠️  Season {season}: {count} teams (expected {expected_teams})")

    finally:
        db.close()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest EPL fixtures and results from API-Football",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --start 2020 --end 2025       Ingest seasons 2020-2025
  %(prog)s --seasons 2024,2025           Ingest specific seasons
  %(prog)s --list-seasons                Show available seasons
  %(prog)s --report                      Show integrity report
  %(prog)s --start 2024 --end 2025 --force  Force re-ingestion
        """,
    )

    parser.add_argument(
        "--start",
        type=int,
        help="Start season year (e.g., 2020)",
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End season year (e.g., 2025)",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        help="Comma-separated list of seasons (e.g., 2023,2024,2025)",
    )
    parser.add_argument(
        "--list-seasons",
        action="store_true",
        help="List available seasons from API",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show data integrity report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if already done",
    )
    parser.add_argument(
        "--no-standings",
        action="store_true",
        help="Skip standings ingestion (faster)",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw API responses (default: data/raw)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dev.yaml",
        help="Path to config file (default: config/dev.yaml)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        return 1
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return 1

    # Check API key
    if not config.football_api.api_key:
        print("❌ FOOTBALL_API_KEY not set in .env file")
        print("   Get your API key from: https://rapidapi.com/api-sports/api/api-football")
        return 1

    # Handle commands
    if args.list_seasons:
        try:
            asyncio.run(list_available_seasons(config))
            return 0
        except FootballApiError as e:
            print(f"❌ API error: {e}")
            return 1

    if args.report:
        try:
            show_integrity_report(config)
            return 0
        except Exception as e:
            print(f"❌ Error generating report: {e}")
            return 1

    # Determine seasons to ingest
    seasons: list[int] = []

    if args.seasons:
        try:
            seasons = [int(s.strip()) for s in args.seasons.split(",")]
        except ValueError:
            print("❌ Invalid seasons format. Use: --seasons 2023,2024,2025")
            return 1
    elif args.start and args.end:
        if args.start > args.end:
            print("❌ Start season must be <= end season")
            return 1
        seasons = list(range(args.start, args.end + 1))
    else:
        parser.print_help()
        print("\n❌ Please specify seasons with --start/--end or --seasons")
        return 1

    # Run ingestion
    raw_data_dir = Path(args.raw_data_dir) if args.raw_data_dir else None

    try:
        result = asyncio.run(
            run_ingestion(
                config=config,
                seasons=seasons,
                include_standings=not args.no_standings,
                force=args.force,
                raw_data_dir=raw_data_dir,
            )
        )

        # Show integrity report after ingestion
        if result["fixtures"] > 0:
            print()
            show_integrity_report(config)

        print("\n" + "=" * 60)
        if result["errors"]:
            print("⚠️  Ingestion completed with errors")
            return 1
        else:
            print("✅ Ingestion completed successfully!")
            return 0

    except FootballApiError as e:
        print(f"\n❌ API error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("ingestion_failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
