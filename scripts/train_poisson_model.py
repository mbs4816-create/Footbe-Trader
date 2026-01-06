#!/usr/bin/env python3
"""Train and save Poisson model for match prediction."""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from footbe_trader.common.config import load_config
from footbe_trader.common.logging import get_logger
from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.poisson_model import PoissonModel
from footbe_trader.storage.database import Database

logger = get_logger(__name__)


def load_training_data(db: Database) -> list[MatchFeatureVector]:
    """Load completed fixtures as training data."""
    cursor = db.connection.cursor()

    # Get all completed fixtures with scores
    cursor.execute("""
        SELECT
            id,
            fixture_id,
            home_team_id,
            away_team_id,
            kickoff_utc,
            season,
            round,
            home_goals,
            away_goals,
            status,
            league_id
        FROM fixtures_v2
        WHERE home_goals IS NOT NULL
          AND away_goals IS NOT NULL
          AND status IN ('FT', '90')
        ORDER BY kickoff_utc
    """)

    features = []
    for row in cursor.fetchall():
        id_, fixture_id, home_id, away_id, kickoff, season, round_str, home_goals, away_goals, status, league_id = row

        # Determine outcome
        if home_goals > away_goals:
            outcome = "H"
        elif home_goals < away_goals:
            outcome = "A"
        else:
            outcome = "D"

        features.append(MatchFeatureVector(
            fixture_id=fixture_id,
            kickoff_utc=datetime.fromisoformat(kickoff),
            home_team_id=home_id,
            away_team_id=away_id,
            season=season,
            round_str=round_str or "",
            outcome=outcome,
            home_goals=home_goals,
            away_goals=away_goals,
        ))

    logger.info("training_data_loaded", count=len(features))
    return features


def main():
    """Train and save Poisson model."""
    config = load_config("configs/dev.yaml")
    db = Database(config.database.path)
    db.connect()

    try:
        # Load training data
        logger.info("loading_training_data")
        features = load_training_data(db)

        if len(features) < 100:
            logger.error("insufficient_training_data", count=len(features))
            return 1

        # Train model
        logger.info("training_poisson_model", fixtures=len(features))
        model = PoissonModel(
            half_life_days=180.0,
            home_advantage=1.35,
            use_dixon_coles=True,
        )
        model.fit(features)

        # Log model summary
        logger.info(
            "model_trained",
            home_advantage=model.home_advantage,
            league_avg_goals=model.league_avg_goals,
            teams=len(model.team_ratings),
            rho=model.rho if model.use_dixon_coles else None,
        )

        # Get top teams
        rankings = model.get_team_rankings()
        logger.info("top_teams", top_5=[r for r in rankings[:5]])

        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "poisson_v1.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info("model_saved", path=str(model_path))

        # Save metadata
        metadata = {
            "name": model.name,
            "version": model.version,
            "trained_at": datetime.utcnow().isoformat(),
            "training_samples": len(features),
            "home_advantage": model.home_advantage,
            "league_avg_goals": model.league_avg_goals,
            "rho": model.rho if model.use_dixon_coles else None,
            "num_teams": len(model.team_ratings),
        }

        metadata_path = model_dir / "poisson_v1_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("metadata_saved", path=str(metadata_path))

        return 0

    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
