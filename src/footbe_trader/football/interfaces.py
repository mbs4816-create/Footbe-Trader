"""Football API interfaces and data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class FixtureStatus(str, Enum):
    """Fixture status codes from API-Football."""

    # Not started
    TIME_TO_BE_DEFINED = "TBD"
    NOT_STARTED = "NS"
    # In play
    FIRST_HALF = "1H"
    HALFTIME = "HT"
    SECOND_HALF = "2H"
    EXTRA_TIME = "ET"
    BREAK_TIME = "BT"
    PENALTY = "P"
    LIVE = "LIVE"
    # Finished
    FULL_TIME = "FT"
    AFTER_EXTRA_TIME = "AET"
    AFTER_PENALTIES = "PEN"
    # Postponed/cancelled
    POSTPONED = "PST"
    CANCELLED = "CANC"
    ABANDONED = "ABD"
    TECHNICAL_LOSS = "AWD"
    WALKOVER = "WO"
    # Unknown
    UNKNOWN = "UNK"

    @classmethod
    def from_short(cls, short: str) -> "FixtureStatus":
        """Get status from short code."""
        try:
            return cls(short)
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_finished(self) -> bool:
        """Check if match is finished."""
        return self in {
            self.FULL_TIME,
            self.AFTER_EXTRA_TIME,
            self.AFTER_PENALTIES,
            self.TECHNICAL_LOSS,
            self.WALKOVER,
        }

    @property
    def is_live(self) -> bool:
        """Check if match is in progress."""
        return self in {
            self.FIRST_HALF,
            self.HALFTIME,
            self.SECOND_HALF,
            self.EXTRA_TIME,
            self.BREAK_TIME,
            self.PENALTY,
            self.LIVE,
        }

    @property
    def is_scheduled(self) -> bool:
        """Check if match is scheduled but not started."""
        return self in {self.TIME_TO_BE_DEFINED, self.NOT_STARTED}


@dataclass
class TeamData:
    """Team data from API."""

    team_id: int
    name: str
    code: str = ""
    country: str = ""
    logo_url: str = ""
    founded: int | None = None
    venue_name: str = ""
    venue_capacity: int | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class FixtureData:
    """Fixture data from API."""

    fixture_id: int
    league_id: int
    season: int
    round: str
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    kickoff_utc: datetime | None
    status: FixtureStatus
    home_goals: int | None = None
    away_goals: int | None = None
    venue: str = ""
    referee: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    @property
    def has_result(self) -> bool:
        """Check if fixture has a final result."""
        return self.status.is_finished and self.home_goals is not None


@dataclass
class LeagueData:
    """League data from API."""

    league_id: int
    name: str
    country: str
    logo_url: str = ""
    type: str = "League"  # League or Cup
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeasonData:
    """Season data from API."""

    year: int
    start_date: datetime | None = None
    end_date: datetime | None = None
    current: bool = False
    coverage: dict[str, bool] = field(default_factory=dict)


@dataclass
class StandingData:
    """League standings entry."""

    team_id: int
    team_name: str
    rank: int
    points: int
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_difference: int
    form: str = ""
    description: str = ""  # e.g., "Champions League"
    raw_data: dict[str, Any] = field(default_factory=dict)


class IFootballClient(ABC):
    """Interface for football data client."""

    @abstractmethod
    async def get_league_info(self, league_id: int) -> tuple[LeagueData, list[SeasonData]]:
        """Get league information and available seasons.

        Args:
            league_id: API-Football league ID (39 for EPL).

        Returns:
            Tuple of (league data, list of available seasons).
        """
        ...

    @abstractmethod
    async def get_fixtures(
        self,
        league_id: int,
        season: int,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[FixtureData]:
        """Get fixtures for a league and season.

        Args:
            league_id: API-Football league ID (39 for EPL).
            season: Season year (e.g., 2025).
            from_date: Filter fixtures from this date.
            to_date: Filter fixtures to this date.

        Returns:
            List of fixture data.
        """
        ...

    @abstractmethod
    async def get_fixture(self, fixture_id: int) -> FixtureData | None:
        """Get a specific fixture by ID.

        Args:
            fixture_id: External fixture ID.

        Returns:
            Fixture data or None if not found.
        """
        ...

    @abstractmethod
    async def get_teams(self, league_id: int, season: int) -> list[TeamData]:
        """Get teams for a league and season.

        Args:
            league_id: API-Football league ID.
            season: Season year.

        Returns:
            List of team data.
        """
        ...

    @abstractmethod
    async def get_standings(self, league_id: int, season: int) -> list[StandingData]:
        """Get current standings for a league.

        Args:
            league_id: API-Football league ID.
            season: Season year.

        Returns:
            List of standings data.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if API is accessible.

        Returns:
            True if API is healthy.
        """
        ...
