"""Placeholder model implementation."""

from typing import Any

from footbe_trader.common.logging import get_logger
from footbe_trader.modeling.interfaces import IModel, MatchFeatures, PredictionResult

logger = get_logger(__name__)


class PlaceholderModel(IModel):
    """Placeholder model that returns uniform probabilities.

    This is a stub implementation for testing the pipeline.
    Real implementation would use ML models like:
    - Elo rating system
    - Poisson regression
    - Gradient boosting (XGBoost, LightGBM)
    - Neural networks
    """

    def __init__(self) -> None:
        """Initialize placeholder model."""
        self._trained = False

    @property
    def name(self) -> str:
        """Model name."""
        return "placeholder"

    @property
    def version(self) -> str:
        """Model version."""
        return "0.1.0"

    def predict(self, features: MatchFeatures) -> list[PredictionResult]:
        """Generate uniform 1/3 probabilities for home/draw/away.

        Args:
            features: Match features (mostly ignored in placeholder).

        Returns:
            List of predictions with uniform probabilities.
        """
        logger.debug(
            "placeholder_prediction",
            home_team=features.home_team,
            away_team=features.away_team,
        )

        # Placeholder: Return uniform probabilities
        # Real model would compute actual probabilities
        base_prob = 1.0 / 3.0

        # Slight home advantage adjustment for demo
        home_prob = base_prob + 0.05
        away_prob = base_prob - 0.03
        draw_prob = 1.0 - home_prob - away_prob

        return [
            PredictionResult(
                prediction_type="home_win_prob",
                value=home_prob,
                confidence=0.5,
                features={
                    "home_team": features.home_team,
                    "away_team": features.away_team,
                },
                metadata={"model": self.name, "version": self.version},
            ),
            PredictionResult(
                prediction_type="draw_prob",
                value=draw_prob,
                confidence=0.5,
                features={
                    "home_team": features.home_team,
                    "away_team": features.away_team,
                },
                metadata={"model": self.name, "version": self.version},
            ),
            PredictionResult(
                prediction_type="away_win_prob",
                value=away_prob,
                confidence=0.5,
                features={
                    "home_team": features.home_team,
                    "away_team": features.away_team,
                },
                metadata={"model": self.name, "version": self.version},
            ),
        ]

    def fit(self, training_data: list[dict[str, Any]]) -> None:
        """Placeholder training (does nothing).

        Args:
            training_data: Historical match data.
        """
        logger.info(
            "placeholder_fit",
            num_samples=len(training_data),
        )
        self._trained = True

    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._trained
