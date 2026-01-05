"""NBA API client and data ingestion.

This module provides interfaces for the API-NBA (api-sports.io) service
and ingestion of NBA game data.
"""

from footbe_trader.nba.client import NBAApiClient, NBAApiError
from footbe_trader.nba.interfaces import (
    NBAGame,
    NBAGameTeam,
    NBATeam,
    NBAGameStatus,
)

__all__ = [
    "NBAApiClient",
    "NBAApiError",
    "NBAGame",
    "NBAGameTeam",
    "NBATeam",
    "NBAGameStatus",
]
