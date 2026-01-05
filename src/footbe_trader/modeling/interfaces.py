"""Modeling interfaces and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionResult:
    """Result of a model prediction."""

    prediction_type: str  # e.g., "home_win_prob", "draw_prob", "away_win_prob"
    value: float  # probability or expected value
    confidence: float  # model confidence (0-1)
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchFeatures:
    """Features for match prediction."""

    home_team: str
    away_team: str
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_form: list[str] = field(default_factory=list)  # Last N results: W/D/L
    away_form: list[str] = field(default_factory=list)
    home_goals_scored_avg: float = 0.0
    home_goals_conceded_avg: float = 0.0
    away_goals_scored_avg: float = 0.0
    away_goals_conceded_avg: float = 0.0
    home_advantage: float = 0.0
    days_since_last_match_home: int = 0
    days_since_last_match_away: int = 0


class IModel(ABC):
    """Interface for prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Model version."""
        ...

    @abstractmethod
    def predict(self, features: MatchFeatures) -> list[PredictionResult]:
        """Generate predictions for a match.

        Args:
            features: Match features.

        Returns:
            List of predictions (e.g., win/draw/loss probabilities).
        """
        ...

    @abstractmethod
    def fit(self, training_data: list[dict[str, Any]]) -> None:
        """Train the model.

        Args:
            training_data: Historical match data.
        """
        ...

    def is_trained(self) -> bool:
        """Check if model is trained.

        Returns:
            True if model is ready for predictions.
        """
        return True
