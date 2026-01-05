"""NBA API client implementation with rate limiting."""

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx

from footbe_trader.common.config import FootballApiConfig
from footbe_trader.common.logging import get_logger
from footbe_trader.nba.interfaces import NBAGame, NBAGameStatus, NBATeam

logger = get_logger(__name__)


class NBAApiError(Exception):
    """Exception for NBA API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class NBAApiClient:
    """NBA API client from api-sports.io.

    Features:
    - Rate limiting (respects API limits)
    - Raw response caching to disk
    - Proper error handling
    
    Uses the same API key as the Football API.
    """

    # Base URL for NBA API
    BASE_URL = "https://v2.nba.api-sports.io"

    def __init__(
        self,
        config: FootballApiConfig,
        raw_data_dir: Path | str | None = None,
    ):
        """Initialize client.

        Args:
            config: API configuration (uses same key as football).
            raw_data_dir: Directory to save raw API responses (optional).
        """
        self.config = config
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else None
        self._client: httpx.AsyncClient | None = None
        self._last_request_time: float = 0.0
        # Rate limit: requests per minute
        self._min_request_interval = 60.0 / config.rate_limit_per_minute

    async def __aenter__(self) -> "NBAApiClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
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
            NBAApiError: On API errors.
        """
        await self._rate_limit()

        try:
            response = await self.client.get(endpoint, params=params)
            data = response.json()

            # Check for API-level errors
            if response.status_code != 200:
                raise NBAApiError(
                    f"API error: {response.status_code}",
                    status_code=response.status_code,
                    response_body=data,
                )

            # Check for API error responses
            errors = data.get("errors", {})
            if errors:
                error_msg = str(errors)
                logger.warning("api_error_response", endpoint=endpoint, errors=errors)
                raise NBAApiError(error_msg, response_body=data)

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
            raise NBAApiError(f"HTTP error: {e}") from e

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

    # --- Teams Methods ---

    async def get_teams(self) -> list[NBATeam]:
        """Get all NBA teams."""
        data = await self._request(
            "/teams",
            cache_key="nba_teams",
        )

        teams = []
        for item in data.get("response", []):
            try:
                team = NBATeam.from_api_response(item)
                teams.append(team)
            except Exception as e:
                logger.warning("team_parse_error", error=str(e), data=item)

        return teams

    # --- Games Methods ---

    async def get_games_by_date(
        self,
        game_date: date,
        season: int | None = None,
    ) -> list[NBAGame]:
        """Get games for a specific date.
        
        Args:
            game_date: Date to get games for.
            season: NBA season year (e.g., 2024 for 2024-25 season).
            
        Returns:
            List of games on that date.
        """
        params: dict[str, Any] = {
            "date": game_date.isoformat(),
        }
        if season:
            params["season"] = season

        cache_key = f"nba_games_{game_date.isoformat()}"
        if season:
            cache_key += f"_s{season}"

        data = await self._request(
            "/games",
            params=params,
            cache_key=cache_key,
        )

        return self._parse_games_response(data)

    async def get_games_by_season(
        self,
        season: int,
        league: str = "standard",
    ) -> list[NBAGame]:
        """Get all games for a season.
        
        Args:
            season: NBA season year (e.g., 2024 for 2024-25 season).
            league: League type (standard, vegas, sacramento, etc).
            
        Returns:
            List of games in the season.
        """
        data = await self._request(
            "/games",
            params={"season": season, "league": league},
            cache_key=f"nba_games_season_{season}_{league}",
        )

        return self._parse_games_response(data)

    async def get_upcoming_games(
        self,
        days_ahead: int = 7,
        season: int | None = None,
    ) -> list[NBAGame]:
        """Get upcoming games for the next N days.
        
        Args:
            days_ahead: Number of days to look ahead.
            season: NBA season year (optional).
            
        Returns:
            List of upcoming games.
        """
        from datetime import timedelta
        
        all_games: list[NBAGame] = []
        today = date.today()
        
        for i in range(days_ahead):
            game_date = today + timedelta(days=i)
            games = await self.get_games_by_date(game_date, season)
            # Filter to only upcoming games
            upcoming = [g for g in games if g.status == NBAGameStatus.NOT_STARTED]
            all_games.extend(upcoming)
        
        return all_games

    async def get_live_games(self) -> list[NBAGame]:
        """Get currently live games."""
        data = await self._request(
            "/games",
            params={"live": "all"},
            cache_key=f"nba_games_live_{datetime.now().strftime('%Y%m%d_%H%M')}",
        )

        games = self._parse_games_response(data)
        return [g for g in games if g.status == NBAGameStatus.LIVE]

    async def get_game_by_id(self, game_id: int) -> NBAGame | None:
        """Get a specific game by ID.
        
        Args:
            game_id: NBA API game ID.
            
        Returns:
            Game data or None if not found.
        """
        data = await self._request(
            "/games",
            params={"id": game_id},
            cache_key=f"nba_game_{game_id}",
        )

        games = self._parse_games_response(data)
        return games[0] if games else None

    def _parse_games_response(self, data: dict[str, Any]) -> list[NBAGame]:
        """Parse games from API response.
        
        Args:
            data: Raw API response.
            
        Returns:
            List of parsed games.
        """
        games = []
        for item in data.get("response", []):
            try:
                game = NBAGame.from_api_response(item)
                games.append(game)
            except Exception as e:
                logger.warning("game_parse_error", error=str(e), data=item)

        return games

    # --- Standings Methods ---

    async def get_standings(
        self,
        season: int,
        league: str = "standard",
        conference: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get standings for a season.
        
        Args:
            season: NBA season year.
            league: League type.
            conference: Filter by conference (East/West).
            
        Returns:
            List of standing entries (raw dict for now).
        """
        params: dict[str, Any] = {
            "season": season,
            "league": league,
        }
        if conference:
            params["conference"] = conference

        data = await self._request(
            "/standings",
            params=params,
            cache_key=f"nba_standings_{season}_{league}",
        )

        return data.get("response", [])

    # --- Seasons Methods ---

    async def get_seasons(self) -> list[int]:
        """Get available seasons."""
        data = await self._request(
            "/seasons",
            cache_key="nba_seasons",
        )

        return data.get("response", [])
