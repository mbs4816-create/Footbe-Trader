"""Model Lifecycle Manager - Automated retraining and deployment.

This module enables continuous model improvement by:
1. Monitoring model performance drift
2. Automatically retraining on recent data
3. A/B testing new models in paper mode
4. Deploying improvements when statistically significant
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from footbe_trader.common.config import AppConfig
from footbe_trader.common.logging import get_logger
from footbe_trader.modeling.backtest import Backtester, BacktestResult
from footbe_trader.modeling.models import BaseModel, create_model
from footbe_trader.storage.database import Database
from footbe_trader.storage.models import FixtureV2

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """Versioned model with metadata."""

    model_id: str
    model_name: str
    version: str
    created_at: datetime
    training_window_days: int
    training_samples: int

    # Hyperparameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Performance metrics on validation set
    validation_accuracy: float = 0.0
    validation_log_loss: float = 0.0
    validation_sharpe: float = 0.0

    # A/B test results (if deployed)
    ab_test_sharpe: float | None = None
    ab_test_trades: int = 0
    ab_test_pnl: float = 0.0

    # Status
    status: str = "training"  # training, testing, deployed, retired
    deployed_at: datetime | None = None
    retired_at: datetime | None = None

    # Model artifact path
    artifact_path: str | None = None

    def config_hash(self) -> str:
        """Generate hash for this model configuration."""
        config_str = json.dumps(self.hyperparameters, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class PerformanceDrift:
    """Detected performance degradation."""

    current_sharpe: float
    baseline_sharpe: float
    degradation_pct: float
    recent_accuracy: float
    baseline_accuracy: float

    should_retrain: bool = False
    reason: str = ""


class ModelLifecycleManager:
    """Manages the full lifecycle of prediction models.

    Responsibilities:
    - Monitor model performance for drift
    - Trigger retraining when performance degrades
    - A/B test new models before deployment
    - Version and archive all models
    - Rollback if new model underperforms
    """

    def __init__(
        self,
        db: Database,
        config: AppConfig,
        models_dir: Path | None = None,
    ):
        """Initialize lifecycle manager.

        Args:
            db: Database connection.
            config: System configuration.
            models_dir: Directory to store model artifacts.
        """
        self.db = db
        self.config = config
        self.models_dir = models_dir or Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Active models (one per sport)
        self.active_models: dict[str, ModelVersion] = {}

        # Performance tracking
        self.performance_window_days = 7
        self.retrain_threshold_sharpe_drop = 0.3  # Retrain if Sharpe drops 30%
        self.retrain_threshold_accuracy_drop = 0.02  # Or accuracy drops 2%

        # A/B testing
        self.ab_test_duration_days = 7
        self.ab_test_min_trades = 50
        self.ab_test_confidence_level = 0.95

    async def start_monitoring_loop(self, interval_hours: int = 24):
        """Start continuous monitoring and retraining loop.

        Args:
            interval_hours: Hours between monitoring checks.
        """
        logger.info("model_lifecycle_started", interval_hours=interval_hours)

        while True:
            try:
                # Check for drift in football models
                await self._check_and_retrain("football")

                # Check for drift in NBA models
                await self._check_and_retrain("nba")

                # Check A/B tests
                await self._evaluate_ab_tests()

            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e), exc_info=True)

            await asyncio.sleep(interval_hours * 3600)

    async def _check_and_retrain(self, sport: str):
        """Check model performance and retrain if needed.

        Args:
            sport: "football" or "nba"
        """
        logger.info("checking_model_performance", sport=sport)

        # Get current active model
        active_model = self.active_models.get(sport)
        if not active_model:
            logger.warning("no_active_model", sport=sport)
            await self._train_initial_model(sport)
            return

        # Detect drift
        drift = await self._detect_drift(sport, active_model)

        if drift.should_retrain:
            logger.info(
                "performance_drift_detected",
                sport=sport,
                reason=drift.reason,
                current_sharpe=drift.current_sharpe,
                degradation_pct=drift.degradation_pct,
            )

            # Trigger retraining
            new_model = await self._retrain_model(sport)

            if new_model:
                # Start A/B test
                await self._start_ab_test(new_model, active_model)

    async def _detect_drift(
        self,
        sport: str,
        active_model: ModelVersion,
    ) -> PerformanceDrift:
        """Detect if model performance has degraded.

        Args:
            sport: Sport type.
            active_model: Currently active model.

        Returns:
            PerformanceDrift analysis.
        """
        # Get recent fixtures (last N days)
        cutoff = datetime.now(UTC) - timedelta(days=self.performance_window_days)
        recent_fixtures = await self._load_recent_fixtures(sport, cutoff)

        if len(recent_fixtures) < 20:
            return PerformanceDrift(
                current_sharpe=0.0,
                baseline_sharpe=0.0,
                degradation_pct=0.0,
                recent_accuracy=0.0,
                baseline_accuracy=0.0,
                should_retrain=False,
                reason="Insufficient recent data",
            )

        # Get model's baseline performance (from when it was trained)
        baseline_sharpe = active_model.validation_sharpe
        baseline_accuracy = active_model.validation_accuracy

        # Measure current performance
        current_sharpe, current_accuracy = await self._evaluate_model_on_fixtures(
            active_model, recent_fixtures
        )

        # Calculate degradation
        sharpe_degradation = (baseline_sharpe - current_sharpe) / max(baseline_sharpe, 0.01)
        accuracy_degradation = baseline_accuracy - current_accuracy

        should_retrain = False
        reason = ""

        if sharpe_degradation > self.retrain_threshold_sharpe_drop:
            should_retrain = True
            reason = f"Sharpe degraded {sharpe_degradation:.1%} (from {baseline_sharpe:.2f} to {current_sharpe:.2f})"
        elif accuracy_degradation > self.retrain_threshold_accuracy_drop:
            should_retrain = True
            reason = f"Accuracy degraded {accuracy_degradation:.1%} (from {baseline_accuracy:.1%} to {current_accuracy:.1%})"

        return PerformanceDrift(
            current_sharpe=current_sharpe,
            baseline_sharpe=baseline_sharpe,
            degradation_pct=sharpe_degradation,
            recent_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            should_retrain=should_retrain,
            reason=reason,
        )

    async def _evaluate_model_on_fixtures(
        self,
        model: ModelVersion,
        fixtures: list[FixtureV2],
    ) -> tuple[float, float]:
        """Evaluate model on a set of fixtures.

        Args:
            model: Model to evaluate.
            fixtures: Fixtures to evaluate on.

        Returns:
            (sharpe_ratio, accuracy)
        """
        # Load model from artifact
        loaded_model = await self._load_model_artifact(model)

        # Run backtest
        backtester = Backtester(
            model_name=model.model_name,
            rolling_window=model.hyperparameters.get("rolling_window", 10),
        )

        # Use rolling mode for recent evaluation
        result = backtester.run_rolling_backtest(
            fixtures,
            min_train_size=50,
            test_window_days=7,
        )

        if result.aggregate_metrics:
            sharpe = self._calculate_sharpe_from_backtest(result)
            accuracy = result.aggregate_metrics.accuracy
            return sharpe, accuracy

        return 0.0, 0.0

    def _calculate_sharpe_from_backtest(self, result: BacktestResult) -> float:
        """Calculate Sharpe ratio from backtest results.

        This is a simplified calculation - in production you'd track
        actual trade-by-trade P&L and compute realized Sharpe.
        """
        # Placeholder - would need actual P&L data
        # For now, use log loss as proxy (lower is better)
        if result.aggregate_metrics and result.aggregate_metrics.log_loss < 0.6:
            return 1.5
        return 0.5

    async def _retrain_model(self, sport: str) -> ModelVersion | None:
        """Retrain model on recent data.

        Args:
            sport: Sport type.

        Returns:
            New model version or None if training failed.
        """
        logger.info("retraining_model", sport=sport)

        # Load training data (last 180 days)
        cutoff = datetime.now(UTC) - timedelta(days=180)
        training_fixtures = await self._load_recent_fixtures(sport, cutoff)

        if len(training_fixtures) < 100:
            logger.warning("insufficient_training_data", sport=sport, count=len(training_fixtures))
            return None

        # Hyperparameters to try (could use Bayesian optimization here)
        hyperparams = {
            "model_name": "multinomial_logistic",
            "rolling_window": 10,
            "learning_rate": 0.01,
            "n_iterations": 1000,
            "regularization": 0.01,
        }

        # Train model
        backtester = Backtester(**hyperparams)
        result = backtester.run_season_backtest(training_fixtures, min_train_seasons=1)

        if not result.aggregate_metrics:
            logger.error("training_failed", sport=sport)
            return None

        # Create model version
        model_version = ModelVersion(
            model_id=f"{sport}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            model_name=hyperparams["model_name"],
            version=datetime.now(UTC).strftime("%Y.%m.%d"),
            created_at=datetime.now(UTC),
            training_window_days=180,
            training_samples=len(training_fixtures),
            hyperparameters=hyperparams,
            validation_accuracy=result.aggregate_metrics.accuracy,
            validation_log_loss=result.aggregate_metrics.log_loss,
            validation_sharpe=self._calculate_sharpe_from_backtest(result),
            status="testing",
        )

        # Save model artifact
        artifact_path = await self._save_model_artifact(model_version, result)
        model_version.artifact_path = str(artifact_path)

        logger.info(
            "model_trained",
            model_id=model_version.model_id,
            accuracy=model_version.validation_accuracy,
            log_loss=model_version.validation_log_loss,
            sharpe=model_version.validation_sharpe,
        )

        return model_version

    async def _train_initial_model(self, sport: str):
        """Train initial model for a sport."""
        logger.info("training_initial_model", sport=sport)

        new_model = await self._retrain_model(sport)
        if new_model:
            new_model.status = "deployed"
            new_model.deployed_at = datetime.now(UTC)
            self.active_models[sport] = new_model
            await self._persist_model_version(new_model)

    async def _start_ab_test(
        self,
        challenger: ModelVersion,
        champion: ModelVersion,
    ):
        """Start A/B test between new model and current model.

        Args:
            challenger: New model to test.
            champion: Current active model.
        """
        logger.info(
            "starting_ab_test",
            challenger_id=challenger.model_id,
            champion_id=champion.model_id,
            duration_days=self.ab_test_duration_days,
        )

        # In production, this would:
        # 1. Deploy challenger alongside champion
        # 2. Randomly assign fixtures to each model
        # 3. Track performance separately
        # 4. After duration, compare and promote winner

        # For now, we'll mark it as testing
        challenger.status = "testing"
        await self._persist_model_version(challenger)

    async def _evaluate_ab_tests(self):
        """Evaluate ongoing A/B tests and promote winners."""
        # Load all models in "testing" status
        testing_models = await self._load_testing_models()

        for model in testing_models:
            # Check if test duration has passed
            if model.created_at + timedelta(days=self.ab_test_duration_days) > datetime.now(UTC):
                continue

            # Check if we have enough trades
            if model.ab_test_trades < self.ab_test_min_trades:
                logger.info("ab_test_insufficient_data", model_id=model.model_id)
                continue

            # Compare to champion
            sport = "football" if "football" in model.model_id else "nba"
            champion = self.active_models.get(sport)

            if not champion:
                # No champion, promote challenger
                await self._promote_model(model, sport)
                continue

            # Statistical test: is challenger significantly better?
            if self._is_significantly_better(model, champion):
                logger.info(
                    "promoting_challenger",
                    challenger_id=model.model_id,
                    champion_id=champion.model_id,
                    challenger_sharpe=model.ab_test_sharpe,
                    champion_sharpe=champion.validation_sharpe,
                )
                await self._promote_model(model, sport)
            else:
                logger.info(
                    "keeping_champion",
                    challenger_id=model.model_id,
                    reason="No significant improvement",
                )
                model.status = "retired"
                model.retired_at = datetime.now(UTC)
                await self._persist_model_version(model)

    def _is_significantly_better(
        self,
        challenger: ModelVersion,
        champion: ModelVersion,
    ) -> bool:
        """Test if challenger is statistically better than champion.

        Uses simple threshold for now - in production would use
        bootstrap hypothesis test or Bayesian comparison.
        """
        if not challenger.ab_test_sharpe or not champion.validation_sharpe:
            return False

        improvement = (challenger.ab_test_sharpe - champion.validation_sharpe) / max(champion.validation_sharpe, 0.01)

        # Require at least 10% improvement to promote
        return improvement > 0.10

    async def _promote_model(self, model: ModelVersion, sport: str):
        """Promote a model to production.

        Args:
            model: Model to promote.
            sport: Sport type.
        """
        # Retire old champion
        if sport in self.active_models:
            old_champion = self.active_models[sport]
            old_champion.status = "retired"
            old_champion.retired_at = datetime.now(UTC)
            await self._persist_model_version(old_champion)

        # Promote new model
        model.status = "deployed"
        model.deployed_at = datetime.now(UTC)
        self.active_models[sport] = model
        await self._persist_model_version(model)

        logger.info("model_promoted", model_id=model.model_id, sport=sport)

    # --- Data loading ---

    async def _load_recent_fixtures(
        self,
        sport: str,
        since: datetime,
    ) -> list[FixtureV2]:
        """Load recent fixtures for a sport."""
        cursor = self.db.connection.cursor()

        if sport == "football":
            cursor.execute(
                """
                SELECT * FROM fixtures_v2
                WHERE kickoff_utc >= ?
                AND status = 'FT'
                AND home_goals IS NOT NULL
                AND away_goals IS NOT NULL
                ORDER BY kickoff_utc ASC
                """,
                (since.isoformat(),),
            )
        else:
            cursor.execute(
                """
                SELECT
                    game_id as fixture_id,
                    season,
                    date_utc as kickoff_utc,
                    home_team_id,
                    away_team_id,
                    home_score as home_goals,
                    away_score as away_goals,
                    status_short as status
                FROM nba_games
                WHERE date_utc >= ?
                AND status = 3  -- FINISHED
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
                ORDER BY date_utc ASC
                """,
                (since.isoformat(),),
            )

        rows = cursor.fetchall()
        fixtures = []
        for row in rows:
            fixture = FixtureV2(
                fixture_id=row["fixture_id"],
                season=row.get("season", 2025),
                kickoff_utc=datetime.fromisoformat(row["kickoff_utc"]),
                home_team_id=row["home_team_id"],
                away_team_id=row["away_team_id"],
                home_goals=row["home_goals"],
                away_goals=row["away_goals"],
                status=row["status"],
            )
            fixtures.append(fixture)

        return fixtures

    async def _load_testing_models(self) -> list[ModelVersion]:
        """Load all models currently in A/B testing."""
        # Would load from database - placeholder
        return []

    # --- Model persistence ---

    async def _save_model_artifact(
        self,
        model: ModelVersion,
        backtest_result: BacktestResult,
    ) -> Path:
        """Save model artifact to disk.

        Args:
            model: Model version.
            backtest_result: Training results.

        Returns:
            Path to saved artifact.
        """
        artifact_dir = self.models_dir / model.model_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = artifact_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "model_id": model.model_id,
                "model_name": model.model_name,
                "version": model.version,
                "created_at": model.created_at.isoformat(),
                "hyperparameters": model.hyperparameters,
                "validation_accuracy": model.validation_accuracy,
                "validation_log_loss": model.validation_log_loss,
            }, f, indent=2)

        # Save backtest results
        results_path = artifact_dir / "backtest_results.json"
        with open(results_path, "w") as f:
            json.dump(backtest_result.to_dict(), f, indent=2)

        return artifact_dir

    async def _load_model_artifact(self, model: ModelVersion) -> BaseModel:
        """Load model from artifact.

        Args:
            model: Model version.

        Returns:
            Loaded model instance.
        """
        # In production, would deserialize model weights
        # For now, create fresh model with same hyperparameters
        return create_model(
            model.model_name,
            **model.hyperparameters,
        )

    async def _persist_model_version(self, model: ModelVersion):
        """Persist model version to database."""
        cursor = self.db.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO model_versions (
                model_id, model_name, version, created_at,
                training_window_days, training_samples,
                hyperparameters, validation_accuracy, validation_log_loss,
                validation_sharpe, status, deployed_at, retired_at,
                artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model.model_id,
                model.model_name,
                model.version,
                model.created_at.isoformat(),
                model.training_window_days,
                model.training_samples,
                json.dumps(model.hyperparameters),
                model.validation_accuracy,
                model.validation_log_loss,
                model.validation_sharpe,
                model.status,
                model.deployed_at.isoformat() if model.deployed_at else None,
                model.retired_at.isoformat() if model.retired_at else None,
                model.artifact_path,
            ),
        )
        self.db.connection.commit()
