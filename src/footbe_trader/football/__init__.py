"""Football API client module."""

from footbe_trader.football.client import FootballApiClient, FootballApiError
from footbe_trader.football.field_catalog import (
    FIELD_CATALOG,
    FieldAvailability,
    FieldInfo,
    get_pre_match_safe_fields,
    get_post_match_only_fields,
    is_field_pre_match_safe,
    validate_pre_match_fields,
)
from footbe_trader.football.interfaces import (
    FixtureData,
    FixtureStatus,
    IFootballClient,
    LeagueData,
    SeasonData,
    StandingData,
    TeamData,
)

__all__ = [
    # Client
    "FootballApiClient",
    "FootballApiError",
    # Data types
    "FixtureData",
    "FixtureStatus",
    "IFootballClient",
    "LeagueData",
    "SeasonData",
    "StandingData",
    "TeamData",
    # Field catalog
    "FIELD_CATALOG",
    "FieldAvailability",
    "FieldInfo",
    "get_pre_match_safe_fields",
    "get_post_match_only_fields",
    "is_field_pre_match_safe",
    "validate_pre_match_fields",
]
