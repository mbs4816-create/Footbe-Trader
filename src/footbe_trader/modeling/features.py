"""Leakage-safe feature engineering for match prediction.

This module implements feature computation with strict time integrity:
- For each fixture, only data from fixtures with kickoff_utc < target kickoff is used.
- All features are validated against the field catalog to ensure PRE_MATCH_SAFE fields.

The feature pipeline is designed to be deterministic and reproducible for backtesting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from footbe_trader.football.field_catalog import (
    FieldAvailability,
    validate_pre_match_fields,
)
from footbe_trader.storage.models import FixtureV2


# Fields used by this feature module - validated at import time
FEATURE_FIELDS_USED = [
    # Fixture scheduling (PRE_MATCH_SAFE)
    "fixture.date",
    "fixture.venue.name",
    "teams.home.id",
    "teams.away.id",
    "league.round",
    # Historical data for feature computation (PRE_MATCH_SAFE via h2h)
    "h2h.goals.home",
    "h2h.goals.away",
]


@dataclass
class MatchFeatureVector:
    """Feature vector for a single match prediction.
    
    All features are computed using only data available before kickoff.
    """
    
    fixture_id: int
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    season: int
    round_str: str
    
    # Target (only set for completed matches)
    outcome: str | None = None  # "H", "D", "A" or None if not finished
    home_goals: int | None = None
    away_goals: int | None = None
    
    # Features
    home_advantage: float = 1.0  # Constant indicating home team
    
    # Rolling goals for home team (when playing at home)
    home_team_home_goals_scored_avg: float = 0.0
    home_team_home_goals_conceded_avg: float = 0.0
    
    # Rolling goals for home team (when playing away) 
    home_team_away_goals_scored_avg: float = 0.0
    home_team_away_goals_conceded_avg: float = 0.0
    
    # Rolling goals for away team (when playing at home)
    away_team_home_goals_scored_avg: float = 0.0
    away_team_home_goals_conceded_avg: float = 0.0
    
    # Rolling goals for away team (when playing away)
    away_team_away_goals_scored_avg: float = 0.0
    away_team_away_goals_conceded_avg: float = 0.0
    
    # Rest days
    home_rest_days: float = 7.0  # Days since last match
    away_rest_days: float = 7.0
    rest_days_diff: float = 0.0  # home_rest - away_rest
    
    # Match counts for feature stability
    home_team_matches_used: int = 0
    away_team_matches_used: int = 0
    
    def to_feature_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.home_advantage,
            self.home_team_home_goals_scored_avg,
            self.home_team_home_goals_conceded_avg,
            self.home_team_away_goals_scored_avg,
            self.home_team_away_goals_conceded_avg,
            self.away_team_home_goals_scored_avg,
            self.away_team_home_goals_conceded_avg,
            self.away_team_away_goals_scored_avg,
            self.away_team_away_goals_conceded_avg,
            self.rest_days_diff,
        ])
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get feature names in same order as to_feature_array."""
        return [
            "home_advantage",
            "home_team_home_goals_scored_avg",
            "home_team_home_goals_conceded_avg",
            "home_team_away_goals_scored_avg",
            "home_team_away_goals_conceded_avg",
            "away_team_home_goals_scored_avg",
            "away_team_home_goals_conceded_avg",
            "away_team_away_goals_scored_avg",
            "away_team_away_goals_conceded_avg",
            "rest_days_diff",
        ]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_id": self.fixture_id,
            "kickoff_utc": self.kickoff_utc.isoformat() if self.kickoff_utc else None,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "season": self.season,
            "outcome": self.outcome,
            **{name: val for name, val in zip(self.feature_names(), self.to_feature_array())},
        }


class FeatureBuilder:
    """Leakage-safe feature builder for match prediction.
    
    Computes features for each fixture using ONLY data from matches
    that occurred BEFORE the target fixture's kickoff time.
    
    Example:
        >>> builder = FeatureBuilder(rolling_window=5)
        >>> features = builder.build_features(fixtures, target_fixture)
    """
    
    def __init__(
        self,
        rolling_window: int = 5,
        min_matches: int = 1,
        default_goals_avg: float = 1.3,
        default_rest_days: float = 7.0,
    ):
        """Initialize feature builder.
        
        Args:
            rolling_window: Number of recent matches to use for rolling averages.
            min_matches: Minimum matches required before using rolling averages.
            default_goals_avg: Default goals average when insufficient history.
            default_rest_days: Default rest days when no previous match found.
        """
        self.rolling_window = rolling_window
        self.min_matches = min_matches
        self.default_goals_avg = default_goals_avg
        self.default_rest_days = default_rest_days
    
    def build_features(
        self,
        all_fixtures: list[FixtureV2],
        target_fixture: FixtureV2,
    ) -> MatchFeatureVector:
        """Build features for a target fixture using only prior data.
        
        CRITICAL: Only fixtures with kickoff_utc < target.kickoff_utc are used.
        This prevents data leakage in backtesting.
        
        Args:
            all_fixtures: All fixtures (will be filtered by time).
            target_fixture: The fixture to build features for.
            
        Returns:
            MatchFeatureVector with computed features.
        """
        if target_fixture.kickoff_utc is None:
            raise ValueError(f"Target fixture {target_fixture.fixture_id} has no kickoff_utc")
        
        target_kickoff = target_fixture.kickoff_utc
        
        # STRICT TIME FILTER - only use fixtures BEFORE target kickoff
        prior_fixtures = [
            f for f in all_fixtures
            if f.kickoff_utc is not None
            and f.kickoff_utc < target_kickoff
            and f.home_goals is not None  # Must be finished
            and f.away_goals is not None
        ]
        
        # Sort by kickoff (most recent first for rolling window)
        prior_fixtures.sort(key=lambda x: x.kickoff_utc, reverse=True)
        
        # Compute features for home team
        home_team_id = target_fixture.home_team_id
        home_home_scored, home_home_conceded, home_home_count = self._rolling_goals(
            prior_fixtures, home_team_id, is_home=True
        )
        home_away_scored, home_away_conceded, home_away_count = self._rolling_goals(
            prior_fixtures, home_team_id, is_home=False
        )
        
        # Compute features for away team
        away_team_id = target_fixture.away_team_id
        away_home_scored, away_home_conceded, away_home_count = self._rolling_goals(
            prior_fixtures, away_team_id, is_home=True
        )
        away_away_scored, away_away_conceded, away_away_count = self._rolling_goals(
            prior_fixtures, away_team_id, is_home=False
        )
        
        # Compute rest days
        home_rest = self._compute_rest_days(prior_fixtures, home_team_id, target_kickoff)
        away_rest = self._compute_rest_days(prior_fixtures, away_team_id, target_kickoff)
        
        # Determine outcome (only for finished matches)
        outcome = None
        if target_fixture.home_goals is not None and target_fixture.away_goals is not None:
            if target_fixture.home_goals > target_fixture.away_goals:
                outcome = "H"
            elif target_fixture.home_goals < target_fixture.away_goals:
                outcome = "A"
            else:
                outcome = "D"
        
        return MatchFeatureVector(
            fixture_id=target_fixture.fixture_id,
            kickoff_utc=target_kickoff,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            season=target_fixture.season,
            round_str=target_fixture.round,
            outcome=outcome,
            home_goals=target_fixture.home_goals,
            away_goals=target_fixture.away_goals,
            home_advantage=1.0,
            home_team_home_goals_scored_avg=home_home_scored,
            home_team_home_goals_conceded_avg=home_home_conceded,
            home_team_away_goals_scored_avg=home_away_scored,
            home_team_away_goals_conceded_avg=home_away_conceded,
            away_team_home_goals_scored_avg=away_home_scored,
            away_team_home_goals_conceded_avg=away_home_conceded,
            away_team_away_goals_scored_avg=away_away_scored,
            away_team_away_goals_conceded_avg=away_away_conceded,
            home_rest_days=home_rest,
            away_rest_days=away_rest,
            rest_days_diff=home_rest - away_rest,
            home_team_matches_used=home_home_count + home_away_count,
            away_team_matches_used=away_home_count + away_away_count,
        )
    
    def build_features_batch(
        self,
        all_fixtures: list[FixtureV2],
        target_fixtures: list[FixtureV2],
    ) -> list[MatchFeatureVector]:
        """Build features for multiple target fixtures.
        
        Args:
            all_fixtures: All fixtures (will be filtered by time for each target).
            target_fixtures: Fixtures to build features for.
            
        Returns:
            List of MatchFeatureVectors.
        """
        return [self.build_features(all_fixtures, f) for f in target_fixtures]
    
    def _rolling_goals(
        self,
        prior_fixtures: list[FixtureV2],
        team_id: int,
        is_home: bool,
    ) -> tuple[float, float, int]:
        """Compute rolling average goals for a team.
        
        Args:
            prior_fixtures: Fixtures sorted by kickoff descending.
            team_id: Team to compute for.
            is_home: If True, only use matches where team was home.
            
        Returns:
            Tuple of (goals_scored_avg, goals_conceded_avg, match_count).
        """
        matches = []
        
        for f in prior_fixtures:
            if len(matches) >= self.rolling_window:
                break
                
            if is_home and f.home_team_id == team_id:
                matches.append((f.home_goals, f.away_goals))
            elif not is_home and f.away_team_id == team_id:
                matches.append((f.away_goals, f.home_goals))
        
        if len(matches) < self.min_matches:
            return self.default_goals_avg, self.default_goals_avg, 0
        
        scored = np.mean([m[0] for m in matches])
        conceded = np.mean([m[1] for m in matches])
        
        return scored, conceded, len(matches)
    
    def _compute_rest_days(
        self,
        prior_fixtures: list[FixtureV2],
        team_id: int,
        target_kickoff: datetime,
    ) -> float:
        """Compute days since team's last match.
        
        Args:
            prior_fixtures: Fixtures sorted by kickoff descending.
            team_id: Team to find last match for.
            target_kickoff: Target match kickoff time.
            
        Returns:
            Days since last match (capped at 14).
        """
        for f in prior_fixtures:
            if f.home_team_id == team_id or f.away_team_id == team_id:
                if f.kickoff_utc is not None:
                    delta = target_kickoff - f.kickoff_utc
                    days = delta.total_seconds() / 86400
                    return min(days, 14.0)  # Cap at 14 days
        
        return self.default_rest_days


def validate_feature_module() -> None:
    """Validate that this module only uses PRE_MATCH_SAFE fields.
    
    Raises:
        ValueError: If any POST_MATCH_ONLY fields are referenced.
    """
    violations = validate_pre_match_fields(FEATURE_FIELDS_USED)
    if violations:
        raise ValueError(f"Feature module uses unsafe fields: {violations}")


# Validate on import
validate_feature_module()
