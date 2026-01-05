"""EPL fixtures and results ingestion from API-Football."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.common.time_utils import utc_now
from footbe_trader.football.client import FootballApiClient, FootballApiError
from footbe_trader.football.interfaces import FixtureData, StandingData, TeamData
from footbe_trader.storage.database import Database
from footbe_trader.storage.models import FixtureV2, StandingSnapshot, Team

logger = get_logger(__name__)

# EPL League ID in API-Football
EPL_LEAGUE_ID = 39


class IngestionResult:
    """Result of an ingestion operation."""

    def __init__(self) -> None:
        self.fixtures_count: int = 0
        self.teams_count: int = 0
        self.standings_count: int = 0
        self.errors: list[str] = []
        self.seasons_processed: list[int] = []

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        logger.error("ingestion_error", error=error)

    def summary(self) -> dict[str, Any]:
        """Get summary of ingestion results."""
        return {
            "success": self.success,
            "fixtures_count": self.fixtures_count,
            "teams_count": self.teams_count,
            "standings_count": self.standings_count,
            "seasons_processed": self.seasons_processed,
            "errors": self.errors,
        }


class EPLIngestion:
    """EPL fixtures and results ingestion handler."""

    def __init__(
        self,
        client: FootballApiClient,
        db: Database,
        raw_data_dir: Path | str | None = None,
    ):
        """Initialize ingestion handler.

        Args:
            client: API-Football client.
            db: Database connection.
            raw_data_dir: Directory to save raw API responses.
        """
        self.client = client
        self.db = db
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None

    async def get_available_seasons(self) -> list[int]:
        """Get available EPL seasons from API.

        Returns:
            List of season years.
        """
        _, seasons = await self.client.get_league_info(EPL_LEAGUE_ID)
        return sorted([s.year for s in seasons], reverse=True)

    async def ingest_season(
        self,
        season: int,
        include_standings: bool = True,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest all data for a single season.

        Args:
            season: Season year (e.g., 2024).
            include_standings: Whether to fetch standings.
            force: Re-ingest even if already done.

        Returns:
            IngestionResult with counts and any errors.
        """
        result = IngestionResult()
        result.seasons_processed.append(season)

        # Check if already ingested (unless force)
        if not force:
            existing = self.db.get_ingestion_log("fixtures", season)
            if existing and existing.status == "success":
                logger.info("season_already_ingested", season=season)
                return result

        logger.info("ingesting_season", season=season)

        try:
            # 1. Ingest teams
            teams_count = await self._ingest_teams(season)
            result.teams_count = teams_count
            logger.info("teams_ingested", season=season, count=teams_count)

            # 2. Ingest fixtures
            fixtures_count = await self._ingest_fixtures(season)
            result.fixtures_count = fixtures_count
            logger.info("fixtures_ingested", season=season, count=fixtures_count)

            # 3. Ingest standings (optional)
            if include_standings:
                standings_count = await self._ingest_standings(season)
                result.standings_count = standings_count
                logger.info("standings_ingested", season=season, count=standings_count)

            # Log success
            self.db.log_ingestion("fixtures", season, fixtures_count, "success")
            self.db.log_ingestion("teams", season, teams_count, "success")
            if include_standings:
                self.db.log_ingestion("standings", season, result.standings_count, "success")

        except FootballApiError as e:
            error_msg = f"API error for season {season}: {e}"
            result.add_error(error_msg)
            self.db.log_ingestion("fixtures", season, 0, "failed", str(e))

        except Exception as e:
            error_msg = f"Unexpected error for season {season}: {e}"
            result.add_error(error_msg)
            self.db.log_ingestion("fixtures", season, 0, "failed", str(e))

        return result

    async def ingest_seasons_range(
        self,
        start_season: int,
        end_season: int,
        include_standings: bool = True,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest multiple seasons.

        Args:
            start_season: First season year.
            end_season: Last season year (inclusive).
            include_standings: Whether to fetch standings.
            force: Re-ingest even if already done.

        Returns:
            Combined IngestionResult.
        """
        result = IngestionResult()

        for season in range(start_season, end_season + 1):
            season_result = await self.ingest_season(
                season,
                include_standings=include_standings,
                force=force,
            )

            result.fixtures_count += season_result.fixtures_count
            result.teams_count += season_result.teams_count
            result.standings_count += season_result.standings_count
            result.errors.extend(season_result.errors)
            result.seasons_processed.extend(season_result.seasons_processed)

        return result

    async def _ingest_teams(self, season: int) -> int:
        """Ingest teams for a season.

        Args:
            season: Season year.

        Returns:
            Number of teams ingested.
        """
        teams = await self.client.get_teams(EPL_LEAGUE_ID, season)

        # Save raw data
        if self.raw_data_dir:
            self._save_raw("teams", season, [t.raw_data for t in teams])

        for team in teams:
            db_team = self._api_team_to_model(team)
            self.db.upsert_team(db_team)

        return len(teams)

    async def _ingest_fixtures(self, season: int) -> int:
        """Ingest fixtures for a season.

        Args:
            season: Season year.

        Returns:
            Number of fixtures ingested.
        """
        fixtures = await self.client.get_fixtures(EPL_LEAGUE_ID, season)

        # Save raw data
        if self.raw_data_dir:
            self._save_raw("fixtures", season, [f.raw_data for f in fixtures])

        for fixture in fixtures:
            db_fixture = self._api_fixture_to_model(fixture)
            self.db.upsert_fixture(db_fixture)

        return len(fixtures)

    async def _ingest_standings(self, season: int) -> int:
        """Ingest current standings for a season.

        Args:
            season: Season year.

        Returns:
            Number of standing entries ingested.
        """
        standings = await self.client.get_standings(EPL_LEAGUE_ID, season)

        # Save raw data
        if self.raw_data_dir:
            self._save_raw("standings", season, [s.raw_data for s in standings])

        today = datetime.now().strftime("%Y-%m-%d")
        for standing in standings:
            db_standing = self._api_standing_to_model(standing, season, today)
            self.db.upsert_standing_snapshot(db_standing)

        return len(standings)

    def _api_team_to_model(self, team: TeamData) -> Team:
        """Convert API team data to database model."""
        return Team(
            team_id=team.team_id,
            name=team.name,
            code=team.code,
            country=team.country,
            logo_url=team.logo_url,
            founded=team.founded,
            venue_name=team.venue_name,
            venue_capacity=team.venue_capacity,
            raw_json=team.raw_data,
        )

    def _api_fixture_to_model(self, fixture: FixtureData) -> FixtureV2:
        """Convert API fixture data to database model."""
        return FixtureV2(
            fixture_id=fixture.fixture_id,
            league_id=fixture.league_id,
            season=fixture.season,
            round=fixture.round,
            home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id,
            kickoff_utc=fixture.kickoff_utc,
            status=fixture.status.value,
            home_goals=fixture.home_goals,
            away_goals=fixture.away_goals,
            venue=fixture.venue,
            referee=fixture.referee,
            raw_json=fixture.raw_data,
        )

    def _api_standing_to_model(
        self, standing: StandingData, season: int, snapshot_date: str
    ) -> StandingSnapshot:
        """Convert API standing data to database model."""
        return StandingSnapshot(
            league_id=EPL_LEAGUE_ID,
            season=season,
            snapshot_date=snapshot_date,
            team_id=standing.team_id,
            rank=standing.rank,
            points=standing.points,
            played=standing.played,
            wins=standing.wins,
            draws=standing.draws,
            losses=standing.losses,
            goals_for=standing.goals_for,
            goals_against=standing.goals_against,
            goal_difference=standing.goal_difference,
            form=standing.form,
            raw_json=standing.raw_data,
        )

    def _save_raw(self, data_type: str, season: int, data: list[dict]) -> None:
        """Save raw API response to disk.

        Args:
            data_type: Type of data ('fixtures', 'teams', 'standings').
            season: Season year.
            data: Raw data to save.
        """
        if not self.raw_data_dir:
            return

        dir_path = self.raw_data_dir / data_type
        dir_path.mkdir(parents=True, exist_ok=True)

        filepath = dir_path / f"{data_type}_{season}.json"
        with open(filepath, "w") as f:
            json.dump(
                {
                    "data_type": data_type,
                    "season": season,
                    "fetched_at": utc_now().isoformat(),
                    "count": len(data),
                    "data": data,
                },
                f,
                indent=2,
                default=str,
            )

        logger.debug("raw_data_saved", path=str(filepath), count=len(data))


def generate_integrity_report(db: Database) -> dict[str, Any]:
    """Generate an integrity report for ingested data.

    Args:
        db: Database connection.

    Returns:
        Report dictionary with statistics.
    """
    report: dict[str, Any] = {
        "generated_at": utc_now().isoformat(),
        "fixtures_per_season": {},
        "missing_scores_per_season": {},
        "teams_per_season": {},
        "total_fixtures": 0,
        "total_teams": 0,
    }

    # Fixtures per season
    fixtures_by_season = db.get_fixtures_count_by_season()
    report["fixtures_per_season"] = fixtures_by_season
    report["total_fixtures"] = sum(fixtures_by_season.values())

    # Missing scores per season
    missing_by_season = db.get_missing_scores_by_season()
    report["missing_scores_per_season"] = {
        season: {
            "missing": missing,
            "total_finished": total,
            "rate": f"{missing / total * 100:.1f}%" if total > 0 else "N/A",
        }
        for season, (missing, total) in missing_by_season.items()
    }

    # Teams per season
    teams_per_season: dict[int, int] = {}
    for season in fixtures_by_season.keys():
        team_ids = db.get_teams_by_season(season)
        teams_per_season[season] = len(team_ids)
    report["teams_per_season"] = teams_per_season

    # Total teams
    all_teams = db.get_all_teams()
    report["total_teams"] = len(all_teams)

    return report
