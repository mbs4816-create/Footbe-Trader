"""NBA data interfaces and models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any


class NBAGameStatus(IntEnum):
    """NBA game status codes from API."""
    
    NOT_STARTED = 1
    LIVE = 2
    FINISHED = 3
    POSTPONED = 4
    DELAYED = 5
    CANCELED = 6


@dataclass
class NBATeam:
    """NBA team data."""
    
    team_id: int
    name: str
    nickname: str
    code: str  # 3-letter code like "LAL", "BOS"
    city: str
    logo: str = ""
    conference: str = ""  # "East" or "West"
    division: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "NBATeam":
        """Create NBATeam from API response."""
        return cls(
            team_id=data.get("id", 0),
            name=data.get("name", ""),
            nickname=data.get("nickname", ""),
            code=data.get("code", ""),
            city=data.get("city", ""),
            logo=data.get("logo", ""),
            conference=data.get("leagues", {}).get("standard", {}).get("conference", ""),
            division=data.get("leagues", {}).get("standard", {}).get("division", ""),
            raw_data=data,
        )


@dataclass
class NBAGameTeam:
    """Team info within a game context."""
    
    team_id: int
    name: str
    nickname: str
    code: str
    logo: str = ""
    score: int | None = None


@dataclass 
class NBAGame:
    """NBA game data."""
    
    game_id: int
    date: datetime
    timestamp: int  # Unix timestamp
    status: NBAGameStatus
    
    # Teams
    home_team: NBAGameTeam
    away_team: NBAGameTeam
    
    # Scores (None if not started)
    home_score: int | None = None
    away_score: int | None = None
    
    # Game info
    league: str = "standard"
    season: int = 2025
    stage: int | None = None
    
    # Arena info
    arena: str = ""
    city: str = ""
    
    raw_data: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "NBAGame":
        """Create NBAGame from API response."""
        # Parse teams
        teams = data.get("teams", {})
        home_data = teams.get("home", {})
        away_data = teams.get("visitors", {})
        
        home_team = NBAGameTeam(
            team_id=home_data.get("id", 0),
            name=home_data.get("name", ""),
            nickname=home_data.get("nickname", ""),
            code=home_data.get("code", ""),
            logo=home_data.get("logo", ""),
        )
        
        away_team = NBAGameTeam(
            team_id=away_data.get("id", 0),
            name=away_data.get("name", ""),
            nickname=away_data.get("nickname", ""),
            code=away_data.get("code", ""),
            logo=away_data.get("logo", ""),
        )
        
        # Parse scores
        scores = data.get("scores", {})
        home_score = scores.get("home", {}).get("points")
        away_score = scores.get("visitors", {}).get("points")
        
        # Parse date
        date_str = data.get("date", {}).get("start", "")
        if date_str:
            # Parse ISO format: "2026-01-07T00:00:00.000Z"
            try:
                game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                game_date = datetime.now()
        else:
            game_date = datetime.now()
        
        # Parse status
        status_code = data.get("status", {}).get("short", 1)
        try:
            status = NBAGameStatus(status_code)
        except ValueError:
            status = NBAGameStatus.NOT_STARTED
        
        # Arena
        arena_data = data.get("arena", {})
        
        return cls(
            game_id=data.get("id", 0),
            date=game_date,
            timestamp=data.get("timestamp", 0),
            status=status,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            league=data.get("league", "standard"),
            season=data.get("season", 2025),
            stage=data.get("stage"),
            arena=arena_data.get("name", ""),
            city=arena_data.get("city", ""),
            raw_data=data,
        )
    
    @property
    def is_upcoming(self) -> bool:
        """Check if game hasn't started yet."""
        return self.status == NBAGameStatus.NOT_STARTED
    
    @property
    def is_finished(self) -> bool:
        """Check if game is finished."""
        return self.status == NBAGameStatus.FINISHED
    
    @property
    def display_name(self) -> str:
        """Human-readable game name."""
        return f"{self.away_team.name} @ {self.home_team.name}"
