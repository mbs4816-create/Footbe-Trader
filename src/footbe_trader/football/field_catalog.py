"""Field catalog for API-Football endpoints.

This module defines field availability labels and provides a registry of all fields
from API-Football endpoints that may be used for modeling, along with their
availability relative to match kickoff.

Field availability categories:
- PRE_MATCH_SAFE: Available before kickoff reliably (e.g., team names, standings)
- PRE_MATCH_UNCERTAIN: May be available but often missing (e.g., lineups, injuries)
- POST_MATCH_ONLY: Only available after match completion (e.g., goals, stats)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FieldAvailability(str, Enum):
    """Field availability relative to match kickoff."""

    PRE_MATCH_SAFE = "PRE_MATCH_SAFE"
    PRE_MATCH_UNCERTAIN = "PRE_MATCH_UNCERTAIN"
    POST_MATCH_ONLY = "POST_MATCH_ONLY"


@dataclass
class FieldInfo:
    """Information about a single API field."""

    path: str  # JSONPath-like path (e.g., "fixture.goals.home")
    python_type: str  # Inferred Python type
    availability: FieldAvailability
    description: str = ""
    example_value: Any = None
    nullable: bool = False
    endpoint: str = ""  # Which endpoint this field comes from


# =============================================================================
# FIELD CATALOG - All fields from API-Football endpoints used for modeling
# =============================================================================
# This is the authoritative source for field availability in the project.
# Any feature engineering must reference fields from this catalog.
# =============================================================================

FIELD_CATALOG: dict[str, FieldInfo] = {}


def _register_field(
    path: str,
    python_type: str,
    availability: FieldAvailability,
    description: str = "",
    example_value: Any = None,
    nullable: bool = False,
    endpoint: str = "",
) -> None:
    """Register a field in the catalog."""
    FIELD_CATALOG[path] = FieldInfo(
        path=path,
        python_type=python_type,
        availability=availability,
        description=description,
        example_value=example_value,
        nullable=nullable,
        endpoint=endpoint,
    )


# -----------------------------------------------------------------------------
# FIXTURES ENDPOINT - /fixtures
# -----------------------------------------------------------------------------

# Fixture identification (PRE_MATCH_SAFE)
_register_field(
    "fixture.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Unique fixture identifier", 1035037, endpoint="fixtures"
)
_register_field(
    "fixture.referee", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Match referee name", "Michael Oliver, England", nullable=True, endpoint="fixtures"
)
_register_field(
    "fixture.timezone", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Timezone of date field", "UTC", endpoint="fixtures"
)
_register_field(
    "fixture.date", "datetime", FieldAvailability.PRE_MATCH_SAFE,
    "Match date/time ISO format", "2023-08-11T19:00:00+00:00", endpoint="fixtures"
)
_register_field(
    "fixture.timestamp", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Unix timestamp of kickoff", 1691780400, endpoint="fixtures"
)

# Venue (PRE_MATCH_SAFE - known in advance)
_register_field(
    "fixture.venue.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Venue unique ID", 10503, nullable=True, endpoint="fixtures"
)
_register_field(
    "fixture.venue.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Venue name", "Vitality Stadium", nullable=True, endpoint="fixtures"
)
_register_field(
    "fixture.venue.city", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Venue city", "Bournemouth", nullable=True, endpoint="fixtures"
)

# Status (PRE_MATCH_SAFE for scheduled, changes during/after match)
_register_field(
    "fixture.status.long", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Full status description", "Not Started", endpoint="fixtures"
)
_register_field(
    "fixture.status.short", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Short status code (NS, FT, etc.)", "NS", endpoint="fixtures"
)
_register_field(
    "fixture.status.elapsed", "int", FieldAvailability.POST_MATCH_ONLY,
    "Minutes elapsed in match", 90, nullable=True, endpoint="fixtures"
)

# League info (PRE_MATCH_SAFE)
_register_field(
    "league.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "League unique ID (39 for EPL)", 39, endpoint="fixtures"
)
_register_field(
    "league.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "League name", "Premier League", endpoint="fixtures"
)
_register_field(
    "league.country", "str", FieldAvailability.PRE_MATCH_SAFE,
    "League country", "England", endpoint="fixtures"
)
_register_field(
    "league.logo", "str", FieldAvailability.PRE_MATCH_SAFE,
    "League logo URL", "https://media.api-sports.io/...", endpoint="fixtures"
)
_register_field(
    "league.flag", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Country flag URL", "https://media.api-sports.io/...", endpoint="fixtures"
)
_register_field(
    "league.season", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Season year", 2023, endpoint="fixtures"
)
_register_field(
    "league.round", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Match round/gameweek", "Regular Season - 1", endpoint="fixtures"
)

# Teams (PRE_MATCH_SAFE)
_register_field(
    "teams.home.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Home team unique ID", 35, endpoint="fixtures"
)
_register_field(
    "teams.home.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Home team name", "Bournemouth", endpoint="fixtures"
)
_register_field(
    "teams.home.logo", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Home team logo URL", "https://media.api-sports.io/...", endpoint="fixtures"
)
_register_field(
    "teams.home.winner", "bool", FieldAvailability.POST_MATCH_ONLY,
    "Whether home team won", True, nullable=True, endpoint="fixtures"
)
_register_field(
    "teams.away.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Away team unique ID", 48, endpoint="fixtures"
)
_register_field(
    "teams.away.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Away team name", "West Ham", endpoint="fixtures"
)
_register_field(
    "teams.away.logo", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Away team logo URL", "https://media.api-sports.io/...", endpoint="fixtures"
)
_register_field(
    "teams.away.winner", "bool", FieldAvailability.POST_MATCH_ONLY,
    "Whether away team won", False, nullable=True, endpoint="fixtures"
)

# Goals (POST_MATCH_ONLY)
_register_field(
    "goals.home", "int", FieldAvailability.POST_MATCH_ONLY,
    "Home team final goals", 1, nullable=True, endpoint="fixtures"
)
_register_field(
    "goals.away", "int", FieldAvailability.POST_MATCH_ONLY,
    "Away team final goals", 1, nullable=True, endpoint="fixtures"
)

# Score details (POST_MATCH_ONLY)
_register_field(
    "score.halftime.home", "int", FieldAvailability.POST_MATCH_ONLY,
    "Home goals at halftime", 0, nullable=True, endpoint="fixtures"
)
_register_field(
    "score.halftime.away", "int", FieldAvailability.POST_MATCH_ONLY,
    "Away goals at halftime", 1, nullable=True, endpoint="fixtures"
)
_register_field(
    "score.fulltime.home", "int", FieldAvailability.POST_MATCH_ONLY,
    "Home goals at fulltime", 1, nullable=True, endpoint="fixtures"
)
_register_field(
    "score.fulltime.away", "int", FieldAvailability.POST_MATCH_ONLY,
    "Away goals at fulltime", 1, nullable=True, endpoint="fixtures"
)
_register_field(
    "score.extratime.home", "int", FieldAvailability.POST_MATCH_ONLY,
    "Home goals in extra time", None, nullable=True, endpoint="fixtures"
)
_register_field(
    "score.extratime.away", "int", FieldAvailability.POST_MATCH_ONLY,
    "Away goals in extra time", None, nullable=True, endpoint="fixtures"
)
_register_field(
    "score.penalty.home", "int", FieldAvailability.POST_MATCH_ONLY,
    "Home penalty shootout goals", None, nullable=True, endpoint="fixtures"
)
_register_field(
    "score.penalty.away", "int", FieldAvailability.POST_MATCH_ONLY,
    "Away penalty shootout goals", None, nullable=True, endpoint="fixtures"
)

# -----------------------------------------------------------------------------
# STANDINGS ENDPOINT - /standings
# -----------------------------------------------------------------------------

_register_field(
    "standings.rank", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Team league position", 1, endpoint="standings"
)
_register_field(
    "standings.team.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Team unique ID", 42, endpoint="standings"
)
_register_field(
    "standings.team.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Team name", "Arsenal", endpoint="standings"
)
_register_field(
    "standings.team.logo", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Team logo URL", "https://media.api-sports.io/...", endpoint="standings"
)
_register_field(
    "standings.points", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Total points", 84, endpoint="standings"
)
_register_field(
    "standings.goalsDiff", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Goal difference", 45, endpoint="standings"
)
_register_field(
    "standings.group", "str", FieldAvailability.PRE_MATCH_SAFE,
    "League group (for cup formats)", "Premier League", endpoint="standings"
)
_register_field(
    "standings.form", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Last 5 match results (WWDLW)", "WDWWW", nullable=True, endpoint="standings"
)
_register_field(
    "standings.status", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Promotion/relegation status", "same", nullable=True, endpoint="standings"
)
_register_field(
    "standings.description", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Position description", "Champions League", nullable=True, endpoint="standings"
)
_register_field(
    "standings.all.played", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Total matches played", 38, endpoint="standings"
)
_register_field(
    "standings.all.win", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Total wins", 26, endpoint="standings"
)
_register_field(
    "standings.all.draw", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Total draws", 6, endpoint="standings"
)
_register_field(
    "standings.all.lose", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Total losses", 6, endpoint="standings"
)
_register_field(
    "standings.all.goals.for", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Goals scored", 88, endpoint="standings"
)
_register_field(
    "standings.all.goals.against", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Goals conceded", 43, endpoint="standings"
)
_register_field(
    "standings.home.played", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Home matches played", 19, endpoint="standings"
)
_register_field(
    "standings.home.win", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Home wins", 14, endpoint="standings"
)
_register_field(
    "standings.home.draw", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Home draws", 3, endpoint="standings"
)
_register_field(
    "standings.home.lose", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Home losses", 2, endpoint="standings"
)
_register_field(
    "standings.home.goals.for", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Home goals scored", 45, endpoint="standings"
)
_register_field(
    "standings.home.goals.against", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Home goals conceded", 18, endpoint="standings"
)
_register_field(
    "standings.away.played", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Away matches played", 19, endpoint="standings"
)
_register_field(
    "standings.away.win", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Away wins", 12, endpoint="standings"
)
_register_field(
    "standings.away.draw", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Away draws", 3, endpoint="standings"
)
_register_field(
    "standings.away.lose", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Away losses", 4, endpoint="standings"
)
_register_field(
    "standings.away.goals.for", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Away goals scored", 43, endpoint="standings"
)
_register_field(
    "standings.away.goals.against", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Away goals conceded", 25, endpoint="standings"
)
_register_field(
    "standings.update", "datetime", FieldAvailability.PRE_MATCH_SAFE,
    "Last update timestamp", "2024-01-15T00:00:00+00:00", endpoint="standings"
)

# -----------------------------------------------------------------------------
# TEAMS ENDPOINT - /teams
# -----------------------------------------------------------------------------

_register_field(
    "team.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Team unique ID", 33, endpoint="teams"
)
_register_field(
    "team.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Team name", "Manchester United", endpoint="teams"
)
_register_field(
    "team.code", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Team short code", "MUN", nullable=True, endpoint="teams"
)
_register_field(
    "team.country", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Team country", "England", endpoint="teams"
)
_register_field(
    "team.founded", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Year team founded", 1878, nullable=True, endpoint="teams"
)
_register_field(
    "team.national", "bool", FieldAvailability.PRE_MATCH_SAFE,
    "Is national team", False, endpoint="teams"
)
_register_field(
    "team.logo", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Team logo URL", "https://media.api-sports.io/...", endpoint="teams"
)
_register_field(
    "venue.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Team venue ID", 556, nullable=True, endpoint="teams"
)
_register_field(
    "venue.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Team venue name", "Old Trafford", nullable=True, endpoint="teams"
)
_register_field(
    "venue.address", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Venue address", "Sir Matt Busby Way", nullable=True, endpoint="teams"
)
_register_field(
    "venue.city", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Venue city", "Manchester", nullable=True, endpoint="teams"
)
_register_field(
    "venue.capacity", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Venue capacity", 76212, nullable=True, endpoint="teams"
)
_register_field(
    "venue.surface", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Pitch surface type", "grass", nullable=True, endpoint="teams"
)
_register_field(
    "venue.image", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Venue image URL", "https://media.api-sports.io/...", nullable=True, endpoint="teams"
)

# -----------------------------------------------------------------------------
# FIXTURE STATISTICS ENDPOINT - /fixtures/statistics
# -----------------------------------------------------------------------------

_register_field(
    "statistics.team.id", "int", FieldAvailability.POST_MATCH_ONLY,
    "Team ID for these stats", 33, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.team.name", "str", FieldAvailability.POST_MATCH_ONLY,
    "Team name for these stats", "Manchester United", endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Shots on Goal", "int", FieldAvailability.POST_MATCH_ONLY,
    "Shots on target", 5, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Shots off Goal", "int", FieldAvailability.POST_MATCH_ONLY,
    "Shots off target", 3, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Total Shots", "int", FieldAvailability.POST_MATCH_ONLY,
    "Total shots", 12, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Blocked Shots", "int", FieldAvailability.POST_MATCH_ONLY,
    "Shots blocked", 4, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Shots insidebox", "int", FieldAvailability.POST_MATCH_ONLY,
    "Shots inside box", 8, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Shots outsidebox", "int", FieldAvailability.POST_MATCH_ONLY,
    "Shots outside box", 4, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Fouls", "int", FieldAvailability.POST_MATCH_ONLY,
    "Fouls committed", 10, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Corner Kicks", "int", FieldAvailability.POST_MATCH_ONLY,
    "Corners taken", 6, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Offsides", "int", FieldAvailability.POST_MATCH_ONLY,
    "Offsides", 2, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Ball Possession", "str", FieldAvailability.POST_MATCH_ONLY,
    "Ball possession percentage", "54%", nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Yellow Cards", "int", FieldAvailability.POST_MATCH_ONLY,
    "Yellow cards", 2, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Red Cards", "int", FieldAvailability.POST_MATCH_ONLY,
    "Red cards", 0, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Goalkeeper Saves", "int", FieldAvailability.POST_MATCH_ONLY,
    "Goalkeeper saves", 4, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Total passes", "int", FieldAvailability.POST_MATCH_ONLY,
    "Total passes", 456, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Passes accurate", "int", FieldAvailability.POST_MATCH_ONLY,
    "Accurate passes", 387, nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.Passes %", "str", FieldAvailability.POST_MATCH_ONLY,
    "Pass accuracy percentage", "85%", nullable=True, endpoint="fixtures/statistics"
)
_register_field(
    "statistics.expected_goals", "float", FieldAvailability.POST_MATCH_ONLY,
    "Expected goals (xG)", 1.45, nullable=True, endpoint="fixtures/statistics"
)

# -----------------------------------------------------------------------------
# LINEUPS ENDPOINT - /fixtures/lineups
# -----------------------------------------------------------------------------

_register_field(
    "lineups.team.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Team ID", 33, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.team.name", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Team name", "Manchester United", endpoint="fixtures/lineups"
)
_register_field(
    "lineups.team.logo", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Team logo URL", "https://...", endpoint="fixtures/lineups"
)
_register_field(
    "lineups.formation", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Team formation (4-3-3)", "4-3-3", nullable=True, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.startXI.player.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Starting player ID", 882, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.startXI.player.name", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Starting player name", "Bruno Fernandes", endpoint="fixtures/lineups"
)
_register_field(
    "lineups.startXI.player.number", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Player shirt number", 8, nullable=True, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.startXI.player.pos", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Player position (G/D/M/F)", "M", nullable=True, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.startXI.player.grid", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Player grid position", "3:2", nullable=True, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.substitutes.player.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Substitute player ID", 747, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.substitutes.player.name", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Substitute player name", "Harry Maguire", endpoint="fixtures/lineups"
)
_register_field(
    "lineups.substitutes.player.number", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Substitute shirt number", 5, nullable=True, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.substitutes.player.pos", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Substitute position", "D", nullable=True, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.coach.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Coach ID", 19, endpoint="fixtures/lineups"
)
_register_field(
    "lineups.coach.name", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Coach name", "Erik ten Hag", endpoint="fixtures/lineups"
)

# -----------------------------------------------------------------------------
# INJURIES ENDPOINT - /injuries
# -----------------------------------------------------------------------------

_register_field(
    "injuries.player.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Injured player ID", 882, endpoint="injuries"
)
_register_field(
    "injuries.player.name", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Injured player name", "Bruno Fernandes", endpoint="injuries"
)
_register_field(
    "injuries.player.photo", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Player photo URL", "https://...", endpoint="injuries"
)
_register_field(
    "injuries.player.type", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Injury type", "Hamstring", nullable=True, endpoint="injuries"
)
_register_field(
    "injuries.player.reason", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Injury reason/description", "Muscle Injury", nullable=True, endpoint="injuries"
)
_register_field(
    "injuries.team.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Injured player team ID", 33, endpoint="injuries"
)
_register_field(
    "injuries.team.name", "str", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Injured player team name", "Manchester United", endpoint="injuries"
)
_register_field(
    "injuries.fixture.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Fixture ID for injury report", 1035037, endpoint="injuries"
)
_register_field(
    "injuries.fixture.date", "datetime", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Fixture date for injury report", "2024-01-15T15:00:00", endpoint="injuries"
)
_register_field(
    "injuries.league.id", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "League ID", 39, endpoint="injuries"
)
_register_field(
    "injuries.league.season", "int", FieldAvailability.PRE_MATCH_UNCERTAIN,
    "Season year", 2023, endpoint="injuries"
)

# -----------------------------------------------------------------------------
# ODDS ENDPOINT - /odds
# -----------------------------------------------------------------------------

_register_field(
    "odds.fixture.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Fixture ID for odds", 1035037, endpoint="odds"
)
_register_field(
    "odds.fixture.date", "datetime", FieldAvailability.PRE_MATCH_SAFE,
    "Fixture date", "2024-01-15T15:00:00", endpoint="odds"
)
_register_field(
    "odds.league.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "League ID", 39, endpoint="odds"
)
_register_field(
    "odds.bookmaker.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Bookmaker ID", 8, endpoint="odds"
)
_register_field(
    "odds.bookmaker.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Bookmaker name", "Bet365", endpoint="odds"
)
_register_field(
    "odds.bet.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Bet type ID", 1, endpoint="odds"
)
_register_field(
    "odds.bet.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Bet type name (Match Winner)", "Match Winner", endpoint="odds"
)
_register_field(
    "odds.value.value", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Bet selection (Home/Draw/Away)", "Home", endpoint="odds"
)
_register_field(
    "odds.value.odd", "str", FieldAvailability.PRE_MATCH_SAFE,
    "Decimal odds", "1.85", endpoint="odds"
)

# -----------------------------------------------------------------------------
# FIXTURE EVENTS ENDPOINT - /fixtures/events
# -----------------------------------------------------------------------------

_register_field(
    "events.time.elapsed", "int", FieldAvailability.POST_MATCH_ONLY,
    "Minute of event", 45, endpoint="fixtures/events"
)
_register_field(
    "events.time.extra", "int", FieldAvailability.POST_MATCH_ONLY,
    "Extra time minutes", 2, nullable=True, endpoint="fixtures/events"
)
_register_field(
    "events.team.id", "int", FieldAvailability.POST_MATCH_ONLY,
    "Team ID of event", 33, endpoint="fixtures/events"
)
_register_field(
    "events.team.name", "str", FieldAvailability.POST_MATCH_ONLY,
    "Team name of event", "Manchester United", endpoint="fixtures/events"
)
_register_field(
    "events.player.id", "int", FieldAvailability.POST_MATCH_ONLY,
    "Player ID of event", 882, endpoint="fixtures/events"
)
_register_field(
    "events.player.name", "str", FieldAvailability.POST_MATCH_ONLY,
    "Player name of event", "Bruno Fernandes", endpoint="fixtures/events"
)
_register_field(
    "events.type", "str", FieldAvailability.POST_MATCH_ONLY,
    "Event type (Goal, Card, Subst)", "Goal", endpoint="fixtures/events"
)
_register_field(
    "events.detail", "str", FieldAvailability.POST_MATCH_ONLY,
    "Event detail (Normal Goal, Penalty)", "Normal Goal", endpoint="fixtures/events"
)
_register_field(
    "events.comments", "str", FieldAvailability.POST_MATCH_ONLY,
    "Event comments", None, nullable=True, endpoint="fixtures/events"
)
_register_field(
    "events.assist.id", "int", FieldAvailability.POST_MATCH_ONLY,
    "Assist player ID", 747, nullable=True, endpoint="fixtures/events"
)
_register_field(
    "events.assist.name", "str", FieldAvailability.POST_MATCH_ONLY,
    "Assist player name", "Marcus Rashford", nullable=True, endpoint="fixtures/events"
)

# -----------------------------------------------------------------------------
# HEAD TO HEAD ENDPOINT - /fixtures/headtohead
# (Historical data - all available pre-match)
# -----------------------------------------------------------------------------

_register_field(
    "h2h.fixture.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "Historical fixture ID", 867946, endpoint="fixtures/headtohead"
)
_register_field(
    "h2h.fixture.date", "datetime", FieldAvailability.PRE_MATCH_SAFE,
    "Historical fixture date", "2023-03-12T15:00:00", endpoint="fixtures/headtohead"
)
_register_field(
    "h2h.teams.home.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "H2H home team ID", 33, endpoint="fixtures/headtohead"
)
_register_field(
    "h2h.teams.home.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "H2H home team name", "Manchester United", endpoint="fixtures/headtohead"
)
_register_field(
    "h2h.teams.away.id", "int", FieldAvailability.PRE_MATCH_SAFE,
    "H2H away team ID", 40, endpoint="fixtures/headtohead"
)
_register_field(
    "h2h.teams.away.name", "str", FieldAvailability.PRE_MATCH_SAFE,
    "H2H away team name", "Liverpool", endpoint="fixtures/headtohead"
)
_register_field(
    "h2h.goals.home", "int", FieldAvailability.PRE_MATCH_SAFE,
    "H2H home goals (historical)", 2, nullable=True, endpoint="fixtures/headtohead"
)
_register_field(
    "h2h.goals.away", "int", FieldAvailability.PRE_MATCH_SAFE,
    "H2H away goals (historical)", 1, nullable=True, endpoint="fixtures/headtohead"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_fields_by_availability(availability: FieldAvailability) -> list[FieldInfo]:
    """Get all fields with a specific availability level."""
    return [f for f in FIELD_CATALOG.values() if f.availability == availability]


def get_fields_by_endpoint(endpoint: str) -> list[FieldInfo]:
    """Get all fields from a specific endpoint."""
    return [f for f in FIELD_CATALOG.values() if f.endpoint == endpoint]


def get_pre_match_safe_fields() -> list[FieldInfo]:
    """Get all fields safe to use for pre-match predictions."""
    return get_fields_by_availability(FieldAvailability.PRE_MATCH_SAFE)


def get_post_match_only_fields() -> list[FieldInfo]:
    """Get all fields only available after match completion."""
    return get_fields_by_availability(FieldAvailability.POST_MATCH_ONLY)


def is_field_pre_match_safe(path: str) -> bool:
    """Check if a field path is safe for pre-match use."""
    if path not in FIELD_CATALOG:
        # Unknown field - be conservative
        return False
    return FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_SAFE


def validate_pre_match_fields(field_paths: list[str]) -> list[str]:
    """Validate that all field paths are safe for pre-match use.

    Args:
        field_paths: List of field paths to validate.

    Returns:
        List of field paths that are NOT pre-match safe (violations).
    """
    violations = []
    for path in field_paths:
        if path not in FIELD_CATALOG:
            violations.append(f"{path} (unknown field)")
        elif FIELD_CATALOG[path].availability == FieldAvailability.POST_MATCH_ONLY:
            violations.append(f"{path} (POST_MATCH_ONLY)")
        elif FIELD_CATALOG[path].availability == FieldAvailability.PRE_MATCH_UNCERTAIN:
            # Allow but could warn
            pass
    return violations


def get_catalog_summary() -> dict[str, int]:
    """Get summary counts of field catalog."""
    summary = {
        "total_fields": len(FIELD_CATALOG),
        "pre_match_safe": 0,
        "pre_match_uncertain": 0,
        "post_match_only": 0,
    }
    for field in FIELD_CATALOG.values():
        if field.availability == FieldAvailability.PRE_MATCH_SAFE:
            summary["pre_match_safe"] += 1
        elif field.availability == FieldAvailability.PRE_MATCH_UNCERTAIN:
            summary["pre_match_uncertain"] += 1
        else:
            summary["post_match_only"] += 1
    return summary


def get_endpoints() -> list[str]:
    """Get list of all endpoints in catalog."""
    return sorted(set(f.endpoint for f in FIELD_CATALOG.values()))
