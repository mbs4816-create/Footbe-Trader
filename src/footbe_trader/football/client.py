"""API-Football client implementation with rate limiting and raw response storage."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from footbe_trader.common.config import FootballApiConfig
from footbe_trader.common.logging import get_logger
from footbe_trader.football.interfaces import (
    FixtureData,
    FixtureStatus,
    IFootballClient,
    LeagueData,
    SeasonData,
    StandingData,
    TeamData,
)

logger = get_logger(__name__)


class FootballApiError(Exception):
    """Exception for Football API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class FootballApiClient(IFootballClient):
    """API-Football client implementation.

    Features:
    - Rate limiting (respects API limits)
    - Raw response caching to disk
    - Proper error handling
    """

    # EPL league ID in API-Football
    EPL_LEAGUE_ID = 39

    def __init__(
        self,
        config: FootballApiConfig,
        raw_data_dir: Path | str | None = None,
    ):
        """Initialize client.

        Args:
            config: API configuration.
            raw_data_dir: Directory to save raw API responses (optional).
        """
        self.config = config
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0
        # Rate limit: requests per minute
        self._min_request_interval = 60.0 / config.rate_limit_per_minute

    async def __aenter__(self) -> "FootballApiClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "x-apisports-key": self.config.api_key,
            },
            timeout=self.config.timeout_seconds,
        )
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self._client

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        import time

        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            wait_time = self._min_request_interval - elapsed
            logger.debug("rate_limit_wait", wait_seconds=round(wait_time, 2))
            await asyncio.sleep(wait_time)
        self._last_request_time = time.monotonic()

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        """Make a rate-limited API request.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            cache_key: Key for caching raw response (optional).

        Returns:
            Parsed JSON response.

        Raises:
            FootballApiError: On API errors.
        """
        await self._rate_limit()

        try:
            response = await self.client.get(endpoint, params=params)
            data = response.json()

            # Check for API-level errors
            if response.status_code != 200:
                raise FootballApiError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    response_body=data,
                )

            # Check for API error responses
            errors = data.get("errors", {})
            if errors:
                error_msg = str(errors)
                logger.warning("api_error_response", endpoint=endpoint, errors=errors)
                raise FootballApiError(error_msg, response_body=data)

            # Save raw response if caching enabled
            if cache_key and self.raw_data_dir:
                self._save_raw_response(cache_key, data)

            logger.info(
                "api_request_success",
                endpoint=endpoint,
                results=data.get("results", 0),
            )
            return data

        except httpx.HTTPError as e:
            logger.error("api_request_failed", endpoint=endpoint, error=str(e))
            raise FootballApiError(f"HTTP error: {e}") from e

    def _save_raw_response(self, cache_key: str, data: dict[str, Any]) -> None:
        """Save raw API response to disk.

        Args:
            cache_key: Cache key (becomes filename).
            data: Response data to save.
        """
        if not self.raw_data_dir:
            return

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.raw_data_dir / f"{cache_key}.json"

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug("raw_response_saved", path=str(filepath))

    # --- League & Season Methods ---

    async def get_league_info(self, league_id: int) -> tuple[LeagueData, list[SeasonData]]:
        """Get league information and available seasons."""
        data = await self._request(
            "/leagues",
            params={"id": league_id},
            cache_key=f"league_{league_id}",
        )

        response = data.get("response", [])
        if not response:
            raise FootballApiError(f"League {league_id} not found")

        league_data = response[0]
        league_info = league_data.get("league", {})
        country_info = league_data.get("country", {})
        seasons_info = league_data.get("seasons", [])

        league = LeagueData(
            league_id=league_info.get("id", league_id),
            name=league_info.get("name", ""),
            country=country_info.get("name", ""),
            logo_url=league_info.get("logo", ""),
            type=league_info.get("type", "League"),
            raw_data=league_data,
        )

        seasons = []
        for s in seasons_info:
            start_str = s.get("start")
            end_str = s.get("end")
            seasons.append(
                SeasonData(
                    year=s.get("year", 0),
                    start_date=datetime.fromisoformat(start_str) if start_str else None,
                    end_date=datetime.fromisoformat(end_str) if end_str else None,
                    current=s.get("current", False),
                    coverage=s.get("coverage", {}),
                )
            )

        return league, seasons

    # --- Fixture Methods ---

    async def get_fixtures(
        self,
        league_id: int,
        season: int,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[FixtureData]:
        """Get fixtures for a league and season."""
        params: dict[str, Any] = {
            "league": league_id,
            "season": season,
        }
        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["to"] = to_date.strftime("%Y-%m-%d")

        cache_key = f"fixtures_{league_id}_{season}"
        if from_date:
            cache_key += f"_from_{from_date.strftime('%Y%m%d')}"
        if to_date:
            cache_key += f"_to_{to_date.strftime('%Y%m%d')}"

        data = await self._request("/fixtures", params=params, cache_key=cache_key)
        return self._parse_fixtures(data)

    async def get_fixture(self, fixture_id: int) -> FixtureData | None:
        """Get a specific fixture by ID."""
        data = await self._request(
            "/fixtures",
            params={"id": fixture_id},
            cache_key=f"fixture_{fixture_id}",
        )

        fixtures = self._parse_fixtures(data)
        return fixtures[0] if fixtures else None

    def _parse_fixtures(self, data: dict[str, Any]) -> list[FixtureData]:
        """Parse fixtures from API response."""
        fixtures = []
        for item in data.get("response", []):
            fixture_info = item.get("fixture", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            league = item.get("league", {})

            # Parse kickoff time
            kickoff_str = fixture_info.get("date")
            kickoff = None
            if kickoff_str:
                try:
                    kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning("invalid_kickoff_time", fixture_id=fixture_info.get("id"))

            # Parse status
            status_info = fixture_info.get("status", {})
            status = FixtureStatus.from_short(status_info.get("short", "UNK"))

            fixtures.append(
                FixtureData(
                    fixture_id=fixture_info.get("id", 0),
                    league_id=league.get("id", 0),
                    season=league.get("season", 0),
                    round=league.get("round", ""),
                    home_team_id=teams.get("home", {}).get("id", 0),
                    away_team_id=teams.get("away", {}).get("id", 0),
                    home_team_name=teams.get("home", {}).get("name", ""),
                    away_team_name=teams.get("away", {}).get("name", ""),
                    kickoff_utc=kickoff,
                    status=status,
                    home_goals=goals.get("home"),
                    away_goals=goals.get("away"),
                    venue=fixture_info.get("venue", {}).get("name", ""),
                    referee=fixture_info.get("referee", "") or "",
                    raw_data=item,
                )
            )
        return fixtures

    # --- Team Methods ---

    async def get_teams(self, league_id: int, season: int) -> list[TeamData]:
        """Get teams for a league and season."""
        data = await self._request(
            "/teams",
            params={"league": league_id, "season": season},
            cache_key=f"teams_{league_id}_{season}",
        )
        return self._parse_teams(data)

    def _parse_teams(self, data: dict[str, Any]) -> list[TeamData]:
        """Parse teams from API response."""
        teams = []
        for item in data.get("response", []):
            team_info = item.get("team", {})
            venue_info = item.get("venue", {})

            teams.append(
                TeamData(
                    team_id=team_info.get("id", 0),
                    name=team_info.get("name", ""),
                    code=team_info.get("code", "") or "",
                    country=team_info.get("country", ""),
                    logo_url=team_info.get("logo", ""),
                    founded=team_info.get("founded"),
                    venue_name=venue_info.get("name", "") or "",
                    venue_capacity=venue_info.get("capacity"),
                    raw_data=item,
                )
            )
        return teams

    # --- Standings Methods ---

    async def get_standings(self, league_id: int, season: int) -> list[StandingData]:
        """Get current standings for a league."""
        data = await self._request(
            "/standings",
            params={"league": league_id, "season": season},
            cache_key=f"standings_{league_id}_{season}",
        )
        return self._parse_standings(data)

    def _parse_standings(self, data: dict[str, Any]) -> list[StandingData]:
        """Parse standings from API response."""
        standings = []
        for item in data.get("response", []):
            league_standings = item.get("league", {}).get("standings", [[]])
            # EPL has single group, so take first standings list
            if league_standings:
                for entry in league_standings[0]:
                    team_info = entry.get("team", {})
                    all_stats = entry.get("all", {})
                    goals = all_stats.get("goals", {})

                    standings.append(
                        StandingData(
                            team_id=team_info.get("id", 0),
                            team_name=team_info.get("name", ""),
                            rank=entry.get("rank", 0),
                            points=entry.get("points", 0),
                            played=all_stats.get("played", 0),
                            wins=all_stats.get("win", 0),
                            draws=all_stats.get("draw", 0),
                            losses=all_stats.get("lose", 0),
                            goals_for=goals.get("for", 0),
                            goals_against=goals.get("against", 0),
                            goal_difference=entry.get("goalsDiff", 0),
                            form=entry.get("form", ""),
                            description=entry.get("description", "") or "",
                            raw_data=entry,
                        )
                    )
        return standings

    # --- Health Check ---

    async def health_check(self) -> bool:
        """Check if API is accessible."""
        try:
            # Use /status endpoint or minimal request
            data = await self._request("/status")
            return data.get("response", {}).get("account", {}).get("status") == "Active"
        except FootballApiError:
            return False
