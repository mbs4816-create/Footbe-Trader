"""Time-decayed Poisson model with Dixon-Coles correction.

This module implements a goals-based model for match outcome prediction:
- Team attack/defense ratings with exponential time decay
- Home advantage factor
- Scoreline probability matrix P(home_goals, away_goals)
- Optional Dixon-Coles low-score correction for 0-0, 1-0, 0-1, 1-1

The model outputs:
- lambda_home, lambda_away (expected goals)
- P(H), P(D), P(A) via integration over scoreline matrix
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from footbe_trader.modeling.features import MatchFeatureVector
from footbe_trader.modeling.models import BaseModel, PredictionOutput


@dataclass
class TeamRatings:
    """Attack and defense ratings for a team."""
    
    team_id: int
    attack: float = 1.0  # Goals scored relative to league average
    defense: float = 1.0  # Goals conceded relative to league average
    home_matches: int = 0
    away_matches: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team_id": self.team_id,
            "attack": self.attack,
            "defense": self.defense,
            "home_matches": self.home_matches,
            "away_matches": self.away_matches,
        }


@dataclass
class PoissonParameters:
    """Parameters for a single match prediction."""
    
    lambda_home: float  # Expected home goals
    lambda_away: float  # Expected away goals
    scoreline_matrix: NDArray[np.float64]  # P(home_goals, away_goals)
    prob_home: float
    prob_draw: float
    prob_away: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            "prob_home": self.prob_home,
            "prob_draw": self.prob_draw,
            "prob_away": self.prob_away,
        }


class PoissonModel(BaseModel):
    """Time-decayed Poisson model for match outcome prediction.
    
    The model estimates team attack/defense ratings using maximum likelihood
    on historical results with exponential time decay. Match outcomes are
    predicted via a scoreline probability matrix.
    
    Model equations:
    - lambda_home = home_advantage * home_attack * away_defense * league_avg
    - lambda_away = away_attack * home_defense * league_avg
    - P(home=h, away=a) = Poisson(h; lambda_home) * Poisson(a; lambda_away) * rho_correction
    
    Where rho_correction is the Dixon-Coles adjustment for low scores.
    """
    
    name = "poisson"
    version = "1.0"
    
    def __init__(
        self,
        half_life_days: float = 180.0,
        home_advantage: float = 1.35,
        max_goals: int = 10,
        use_dixon_coles: bool = True,
        rho_initial: float = -0.1,
        regularization: float = 0.001,
        max_iterations: int = 100,
    ):
        """Initialize Poisson model.
        
        Args:
            half_life_days: Half-life for exponential time decay.
            home_advantage: Initial home advantage multiplier.
            max_goals: Maximum goals to consider in scoreline matrix.
            use_dixon_coles: Whether to apply Dixon-Coles correction.
            rho_initial: Initial rho parameter for Dixon-Coles.
            regularization: L2 regularization for team ratings.
            max_iterations: Maximum optimization iterations.
        """
        super().__init__()
        self.half_life_days = half_life_days
        self.home_advantage = home_advantage
        self.max_goals = max_goals
        self.use_dixon_coles = use_dixon_coles
        self.rho = rho_initial
        self.regularization = regularization
        self.max_iterations = max_iterations
        
        # Learned parameters
        self.team_ratings: dict[int, TeamRatings] = {}
        self.league_avg_goals: float = 1.3  # Goals per team per match
        self.reference_date: datetime | None = None
        
        # Decay constant: lambda = ln(2) / half_life
        self.decay_rate = np.log(2) / half_life_days
    
    def _compute_time_weight(self, match_date: datetime, reference_date: datetime) -> float:
        """Compute exponential time decay weight."""
        days_ago = (reference_date - match_date).days
        return np.exp(-self.decay_rate * max(0, days_ago))
    
    def _poisson_pmf(self, k: int, lam: float) -> float:
        """Poisson probability mass function."""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return np.exp(-lam + k * np.log(lam) - np.sum(np.log(np.arange(1, k + 1))))
    
    def _dixon_coles_correction(
        self, home_goals: int, away_goals: int, lambda_h: float, lambda_a: float, rho: float
    ) -> float:
        """Dixon-Coles correction factor for low scores.
        
        Adjusts probabilities for 0-0, 1-0, 0-1, 1-1 scorelines
        which are empirically under/over-predicted by independent Poisson.
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - lambda_h * lambda_a * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + lambda_h * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + lambda_a * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _compute_scoreline_matrix(
        self, lambda_home: float, lambda_away: float
    ) -> NDArray[np.float64]:
        """Compute probability matrix P(home_goals, away_goals)."""
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                p_h = self._poisson_pmf(h, lambda_home)
                p_a = self._poisson_pmf(a, lambda_away)
                
                if self.use_dixon_coles:
                    correction = self._dixon_coles_correction(
                        h, a, lambda_home, lambda_away, self.rho
                    )
                else:
                    correction = 1.0
                
                matrix[h, a] = p_h * p_a * correction
        
        # Normalize to sum to 1
        total = matrix.sum()
        if total > 0:
            matrix /= total
        
        return matrix
    
    def _extract_outcomes(
        self, scoreline_matrix: NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Extract P(H), P(D), P(A) from scoreline matrix."""
        prob_home = 0.0
        prob_draw = 0.0
        prob_away = 0.0
        
        for h in range(self.max_goals + 1):
            for a in range(self.max_goals + 1):
                if h > a:
                    prob_home += scoreline_matrix[h, a]
                elif h == a:
                    prob_draw += scoreline_matrix[h, a]
                else:
                    prob_away += scoreline_matrix[h, a]
        
        return prob_home, prob_draw, prob_away
    
    def _log_likelihood(
        self,
        params: NDArray[np.float64],
        team_ids: list[int],
        matches: list[tuple[int, int, int, int, float]],
    ) -> float:
        """Compute negative log-likelihood for optimization.
        
        Args:
            params: Flattened array of [attack_ratings, defense_ratings, home_adv, rho]
            team_ids: List of unique team IDs
            matches: List of (home_id, away_id, home_goals, away_goals, weight)
        """
        n_teams = len(team_ids)
        team_idx = {tid: i for i, tid in enumerate(team_ids)}
        
        attacks = params[:n_teams]
        defenses = params[n_teams:2*n_teams]
        home_adv = params[2*n_teams]
        rho = params[2*n_teams + 1] if self.use_dixon_coles else 0.0
        
        # Constraint: average attack and defense should be 1
        attacks = attacks / np.mean(attacks)
        defenses = defenses / np.mean(defenses)
        
        log_lik = 0.0
        for home_id, away_id, home_goals, away_goals, weight in matches:
            home_idx = team_idx.get(home_id)
            away_idx = team_idx.get(away_id)
            
            if home_idx is None or away_idx is None:
                continue
            
            lambda_h = home_adv * attacks[home_idx] * defenses[away_idx] * self.league_avg_goals
            lambda_a = attacks[away_idx] * defenses[home_idx] * self.league_avg_goals
            
            # Clip lambdas
            lambda_h = max(0.1, min(lambda_h, 6.0))
            lambda_a = max(0.1, min(lambda_a, 6.0))
            
            # Poisson log-likelihood
            p_h = self._poisson_pmf(home_goals, lambda_h)
            p_a = self._poisson_pmf(away_goals, lambda_a)
            
            if self.use_dixon_coles:
                correction = self._dixon_coles_correction(
                    home_goals, away_goals, lambda_h, lambda_a, rho
                )
            else:
                correction = 1.0
            
            prob = max(p_h * p_a * correction, 1e-10)
            log_lik += weight * np.log(prob)
        
        # L2 regularization on attack/defense ratings
        reg = self.regularization * (np.sum(attacks**2) + np.sum(defenses**2))
        
        return -(log_lik - reg)
    
    def fit(self, features: list[MatchFeatureVector]) -> "PoissonModel":
        """Fit team ratings using maximum likelihood.
        
        Args:
            features: Training data with outcomes.
            
        Returns:
            Self.
        """
        # Filter to completed matches
        train_data = [
            f for f in features
            if f.outcome is not None and f.home_goals is not None and f.away_goals is not None
        ]
        
        if len(train_data) < 20:
            raise ValueError(f"Insufficient training data: {len(train_data)} samples")
        
        # Find reference date (most recent match)
        self.reference_date = max(f.kickoff_utc for f in train_data)
        
        # Compute league average goals
        total_goals = sum(f.home_goals + f.away_goals for f in train_data)
        self.league_avg_goals = total_goals / (2 * len(train_data))
        
        # Get unique teams
        team_ids = sorted(set(
            tid for f in train_data for tid in [f.home_team_id, f.away_team_id]
        ))
        n_teams = len(team_ids)
        
        # Prepare match data with time weights
        matches = []
        for f in train_data:
            weight = self._compute_time_weight(f.kickoff_utc, self.reference_date)
            matches.append((f.home_team_id, f.away_team_id, f.home_goals, f.away_goals, weight))
        
        # Initial parameters
        n_params = 2 * n_teams + (2 if self.use_dixon_coles else 1)
        x0 = np.ones(n_params)
        x0[2*n_teams] = self.home_advantage  # Initial home advantage
        if self.use_dixon_coles:
            x0[2*n_teams + 1] = self.rho  # Initial rho
        
        # Bounds
        bounds = (
            [(0.2, 3.0)] * n_teams +  # Attack ratings
            [(0.2, 3.0)] * n_teams +  # Defense ratings
            [(1.0, 2.0)] +  # Home advantage
            ([(-.3, 0.0)] if self.use_dixon_coles else [])  # rho (typically negative)
        )
        
        # Optimize
        result = optimize.minimize(
            self._log_likelihood,
            x0,
            args=(team_ids, matches),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations},
        )
        
        # Extract parameters
        attacks = result.x[:n_teams]
        defenses = result.x[n_teams:2*n_teams]
        self.home_advantage = result.x[2*n_teams]
        if self.use_dixon_coles:
            self.rho = result.x[2*n_teams + 1]
        
        # Normalize to average 1
        attacks = attacks / np.mean(attacks)
        defenses = defenses / np.mean(defenses)
        
        # Store team ratings
        self.team_ratings = {}
        for i, tid in enumerate(team_ids):
            # Count home/away matches
            home_count = sum(1 for f in train_data if f.home_team_id == tid)
            away_count = sum(1 for f in train_data if f.away_team_id == tid)
            
            self.team_ratings[tid] = TeamRatings(
                team_id=tid,
                attack=attacks[i],
                defense=defenses[i],
                home_matches=home_count,
                away_matches=away_count,
            )
        
        self._is_trained = True
        return self
    
    def _get_team_rating(self, team_id: int) -> TeamRatings:
        """Get team rating, returning average if unknown."""
        if team_id in self.team_ratings:
            return self.team_ratings[team_id]
        return TeamRatings(team_id=team_id, attack=1.0, defense=1.0)
    
    def predict_params(self, feature: MatchFeatureVector) -> PoissonParameters:
        """Get full Poisson parameters for a match."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        home_rating = self._get_team_rating(feature.home_team_id)
        away_rating = self._get_team_rating(feature.away_team_id)
        
        # Compute expected goals
        lambda_home = (
            self.home_advantage *
            home_rating.attack *
            away_rating.defense *
            self.league_avg_goals
        )
        lambda_away = (
            away_rating.attack *
            home_rating.defense *
            self.league_avg_goals
        )
        
        # Clip to reasonable range
        lambda_home = max(0.3, min(lambda_home, 5.0))
        lambda_away = max(0.3, min(lambda_away, 5.0))
        
        # Compute scoreline matrix
        matrix = self._compute_scoreline_matrix(lambda_home, lambda_away)
        
        # Extract outcome probabilities
        prob_home, prob_draw, prob_away = self._extract_outcomes(matrix)
        
        return PoissonParameters(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            scoreline_matrix=matrix,
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
        )
    
    def predict(self, features: list[MatchFeatureVector]) -> list[PredictionOutput]:
        """Generate predictions.
        
        Args:
            features: Feature vectors.
            
        Returns:
            Predictions with probabilities.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        predictions = []
        for f in features:
            params = self.predict_params(f)
            
            # Determine predicted outcome
            probs = [params.prob_home, params.prob_draw, params.prob_away]
            outcomes = ["H", "D", "A"]
            predicted = outcomes[np.argmax(probs)]
            
            predictions.append(
                PredictionOutput(
                    fixture_id=f.fixture_id,
                    prob_home=params.prob_home,
                    prob_draw=params.prob_draw,
                    prob_away=params.prob_away,
                    predicted_outcome=predicted,
                )
            )
        
        return predictions
    
    def get_team_rankings(self) -> list[dict[str, Any]]:
        """Get team rankings by attack and defense strength."""
        if not self._is_trained:
            return []
        
        # Sort by overall strength (attack / defense)
        rankings = []
        for tid, rating in self.team_ratings.items():
            rankings.append({
                "team_id": tid,
                "attack": rating.attack,
                "defense": rating.defense,
                "overall": rating.attack / max(rating.defense, 0.1),
                "matches": rating.home_matches + rating.away_matches,
            })
        
        rankings.sort(key=lambda x: x["overall"], reverse=True)
        return rankings
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize model parameters."""
        return {
            "name": self.name,
            "version": self.version,
            "home_advantage": self.home_advantage,
            "rho": self.rho if self.use_dixon_coles else None,
            "league_avg_goals": self.league_avg_goals,
            "half_life_days": self.half_life_days,
            "team_ratings": {
                str(tid): rating.to_dict()
                for tid, rating in self.team_ratings.items()
            },
        }
