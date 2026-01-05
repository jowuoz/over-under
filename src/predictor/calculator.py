"""
calculator.py - Probability calculator for Over/Under Predictor system
Core engine for calculating probabilities using multiple statistical models
"""
import math
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
import logging
from collections import defaultdict

# Import models and configuration
try:
    from .models import (
        Game, ProbabilityMetrics, RiskAssessment, Team, League,
        GameStatus, PredictionConfidence, RiskLevel,
        create_prediction_id, calculate_implied_probability
    )
    from config import get_predictor_config
except ImportError:
    from src.predictor.models import (
        Game, ProbabilityMetrics, RiskAssessment, Team, League,
        GameStatus, PredictionConfidence, RiskLevel,
        create_prediction_id, calculate_implied_probability
    )
    from src.config import get_predictor_config


@dataclass
class ModelWeights:
    """Weights for different prediction models"""
    time_based: float = 0.35      # Time decay and current score
    statistical: float = 0.25     # Historical statistics
    momentum: float = 0.20        # Recent match momentum
    league_based: float = 0.10    # League tendencies
    odds_based: float = 0.10      # Market odds (if available)
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = sum([
            self.time_based, self.statistical, 
            self.momentum, self.league_based, self.odds_based
        ])
        if total > 0:
            self.time_based /= total
            self.statistical /= total
            self.momentum /= total
            self.league_based /= total
            self.odds_based /= total


@dataclass
class CalculationResult:
    """Result of probability calculation"""
    probabilities: Dict[str, float]  # over_05, over_15, over_25, over_35, over_45
    expected_additional_goals: float
    expected_total_goals: float
    confidence_score: float
    model_contributions: Dict[str, float]
    key_factors: List[str]
    warnings: List[str]


class ProbabilityCalculator:
    """
    Main calculator for over/under probabilities using multiple models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize probability calculator
        
        Args:
            config: Calculator configuration
        """
        self.config = config or self._load_config()
        self.logger = logging.getLogger("predictor.calculator")
        
        # Model weights
        self.weights = ModelWeights(**self.config.get('model_weights', {}))
        self.weights.normalize()
        
        # Model instances
        self.models = {
            'time_based': TimeBasedModel(self.config),
            'statistical': StatisticalModel(self.config),
            'momentum': MomentumModel(self.config),
            'league_based': LeagueBasedModel(self.config),
            'odds_based': OddsBasedModel(self.config),
        }
        
        # Cache for performance
        self._cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Statistical distributions
        self._setup_distributions()
        
        self.logger.info(f"ProbabilityCalculator initialized with weights: {self.weights}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load calculator configuration"""
        try:
            return get_predictor_config()
        except:
            # Default configuration
            return {
                'model_weights': {
                    'time_based': 0.35,
                    'statistical': 0.25,
                    'momentum': 0.20,
                    'league_based': 0.10,
                    'odds_based': 0.10,
                },
                'min_confidence': 0.6,
                'min_data_quality': 0.5,
                'poisson_lambda_min': 0.1,
                'poisson_lambda_max': 5.0,
                'time_decay_factor': 0.02,
                'momentum_window': 10,  # minutes
                'historical_matches': 10,
                'league_factor_weight': 0.3,
            }
    
    def _setup_distributions(self):
        """Setup statistical distributions"""
        # Pre-calculate Poisson probabilities for common lambdas
        self._poisson_cache = {}
        
    def calculate_probabilities(self, game: Game, historical_data: Optional[Dict] = None) -> ProbabilityMetrics:
        """
        Calculate probabilities for all over/under lines
        
        Args:
            game: Game object with match data
            historical_data: Optional historical performance data
            
        Returns:
            ProbabilityMetrics object with all calculations
        """
        # Check cache first
        cache_key = self._create_cache_key(game)
        cached = self._get_from_cache(cache_key)
        if cached:
            self.logger.debug(f"Using cached probabilities for game {game.id}")
            return cached
        
        self.logger.info(f"Calculating probabilities for {game.home_team.name} vs {game.away_team.name}")
        
        try:
            # Run all models
            model_results = {}
            for model_name, model in self.models.items():
                try:
                    result = model.calculate(game, historical_data)
                    model_results[model_name] = result
                    self.logger.debug(f"Model '{model_name}' completed: {result}")
                except Exception as e:
                    self.logger.warning(f"Model '{model_name}' failed: {e}")
                    model_results[model_name] = None
            
            # Ensemble results
            ensemble_result = self._ensemble_predictions(model_results)
            
            # Calculate confidence
            confidence_score = self._calculate_confidence(game, model_results, ensemble_result)
            
            # Extract key factors
            key_factors = self._extract_key_factors(game, model_results, ensemble_result)
            
            # Calculate expected values
            expected_additional = self._calculate_expected_additional_goals(game, ensemble_result)
            expected_total = game.total_goals + expected_additional
            
            # Create probability metrics
            metrics = ProbabilityMetrics(
                game_id=game.id,
                probability_over_05=ensemble_result.probabilities['over_05'],
                probability_over_15=ensemble_result.probabilities['over_15'],
                probability_over_25=ensemble_result.probabilities['over_25'],
                probability_over_35=ensemble_result.probabilities['over_35'],
                probability_over_45=ensemble_result.probabilities.get('over_45', 0.0),
                implied_prob_over_25=(
                    calculate_implied_probability(game.odds_over_25) 
                    if game.odds_over_25 else None
                ),
                implied_prob_under_25=(
                    calculate_implied_probability(game.odds_under_25) 
                    if game.odds_under_25 else None
                ),
                confidence_score=confidence_score,
                expected_additional_goals=expected_additional,
                expected_total_goals=expected_total,
                model_contributions={
                    name: weight for name, weight in zip(
                        ['time_based', 'statistical', 'momentum', 'league_based', 'odds_based'],
                        [self.weights.time_based, self.weights.statistical, 
                         self.weights.momentum, self.weights.league_based, self.weights.odds_based]
                    )
                }
            )
            
            # Cache the result
            self._add_to_cache(cache_key, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating probabilities: {e}")
            # Return default metrics
            return self._get_default_metrics(game)
    
    def _ensemble_predictions(self, model_results: Dict[str, Any]) -> CalculationResult:
        """
        Ensemble predictions from multiple models
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Ensemble calculation result
        """
        # Collect predictions from each model
        all_predictions = {}
        model_contributions = {}
        key_factors = []
        warnings = []
        
        for model_name, result in model_results.items():
            if result is None:
                continue
            
            weight = getattr(self.weights, model_name, 0.0)
            
            # Add to ensemble
            for line, prob in result.probabilities.items():
                if line not in all_predictions:
                    all_predictions[line] = []
                all_predictions[line].append(prob * weight)
            
            # Track contributions
            model_contributions[model_name] = weight
            
            # Collect factors and warnings
            key_factors.extend(result.key_factors)
            warnings.extend(result.warnings)
        
        # Calculate weighted averages
        ensemble_probs = {}
        for line, weighted_probs in all_predictions.items():
            ensemble_probs[line] = sum(weighted_probs)
        
        # Calculate expected values (weighted average)
        expected_additional = sum(
            result.expected_additional_goals * getattr(self.weights, model_name, 0.0)
            for model_name, result in model_results.items()
            if result is not None
        )
        
        expected_total = sum(
            result.expected_total_goals * getattr(self.weights, model_name, 0.0)
            for model_name, result in model_results.items()
            if result is not None
        )
        
        # Calculate confidence (average of model confidences)
        confidence = sum(
            result.confidence_score * getattr(self.weights, model_name, 0.0)
            for model_name, result in model_results.items()
            if result is not None
        )
        
        # Remove duplicates from factors and warnings
        key_factors = list(dict.fromkeys(key_factors))
        warnings = list(dict.fromkeys(warnings))
        
        return CalculationResult(
            probabilities=ensemble_probs,
            expected_additional_goals=expected_additional,
            expected_total_goals=expected_total,
            confidence_score=confidence,
            model_contributions=model_contributions,
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _calculate_confidence(self, game: Game, model_results: Dict, ensemble_result: CalculationResult) -> float:
        """
        Calculate overall confidence score
        
        Args:
            game: Game object
            model_results: Individual model results
            ensemble_result: Ensemble result
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []
        
        # 1. Data completeness (0.0 to 0.3)
        data_score = self._calculate_data_completeness_score(game)
        confidence_factors.append(data_score * 0.3)
        
        # 2. Model agreement (0.0 to 0.3)
        agreement_score = self._calculate_model_agreement_score(model_results)
        confidence_factors.append(agreement_score * 0.3)
        
        # 3. Match stage (0.0 to 0.2)
        stage_score = self._calculate_match_stage_score(game)
        confidence_factors.append(stage_score * 0.2)
        
        # 4. Historical data quality (0.0 to 0.2)
        history_score = self._calculate_historical_data_score(game)
        confidence_factors.append(history_score * 0.2)
        
        # Calculate weighted average
        confidence = sum(confidence_factors)
        
        # Apply ensemble confidence
        confidence = (confidence + ensemble_result.confidence_score) / 2
        
        # Ensure within bounds
        return max(0.0, min(1.0, confidence))
    
    def _calculate_data_completeness_score(self, game: Game) -> float:
        """Calculate score based on data completeness"""
        factors = []
        
        # Basic data
        if game.home_team.name and game.away_team.name:
            factors.append(0.2)
        
        # Scores
        if game.home_score is not None and game.away_score is not None:
            factors.append(0.2)
        
        # Match minute
        if game.current_minute > 0:
            factors.append(0.1)
        
        # Statistics
        stat_factors = [
            game.shots_on_target != (0, 0),
            game.shots_total != (0, 0),
            game.possession != (50.0, 50.0),
            game.expected_goals != (0.0, 0.0),
        ]
        factors.append(sum(stat_factors) / len(stat_factors) * 0.5)
        
        return sum(factors)
    
    def _calculate_model_agreement_score(self, model_results: Dict) -> float:
        """Calculate score based on model agreement"""
        valid_results = [r for r in model_results.values() if r is not None]
        
        if len(valid_results) < 2:
            return 0.5  # Neutral if only one model
        
        # Compare predictions for over 2.5
        predictions = []
        for result in valid_results:
            if 'over_25' in result.probabilities:
                predictions.append(result.probabilities['over_25'])
        
        if not predictions:
            return 0.5
        
        # Calculate variance
        variance = statistics.variance(predictions) if len(predictions) > 1 else 0
        
        # Higher variance = lower agreement
        agreement = 1.0 - min(variance * 4, 1.0)  # Scale variance
        
        return max(0.0, min(1.0, agreement))
    
    def _calculate_match_stage_score(self, game: Game) -> float:
        """Calculate score based on match stage"""
        if not game.is_live:
            return 0.5  # Neutral for non-live matches
        
        # More minutes = more reliable data
        minute_factor = min(game.current_minute / 70, 1.0)
        
        # Adjust based on score stability
        score_factor = 1.0
        if game.total_goals > 0:
            # Games with goals have more reliable patterns
            score_factor = min(game.total_goals / 3, 1.0)
        
        return minute_factor * score_factor
    
    def _calculate_historical_data_score(self, game: Game) -> float:
        """Calculate score based on historical data availability"""
        # This would check if we have historical data for teams/league
        # For now, return default based on league tier
        if game.league.tier == 1:
            return 0.8  # Good data for top leagues
        elif game.league.tier == 2:
            return 0.6  # Moderate data
        else:
            return 0.4  # Limited data
    
    def _extract_key_factors(self, game: Game, model_results: Dict, ensemble_result: CalculationResult) -> List[str]:
        """Extract key factors influencing the prediction"""
        factors = []
        
        # Add factors from ensemble result
        factors.extend(ensemble_result.key_factors)
        
        # Add game-specific factors
        if game.is_live:
            factors.append(f"Live match ({game.current_minute}' elapsed)")
            
            if game.total_goals > 0:
                goal_rate = game.goal_rate
                if goal_rate > 0.05:  # More than 1 goal every 20 minutes
                    factors.append(f"High goal rate ({goal_rate:.2f} goals/min)")
                elif goal_rate < 0.02:  # Less than 1 goal every 50 minutes
                    factors.append(f"Low goal rate ({goal_rate:.2f} goals/min)")
            
            if game.momentum_score > 7.0:
                factors.append("High attacking momentum")
            elif game.momentum_score < 3.0:
                factors.append("Low attacking momentum")
        
        # League factors
        if game.league.avg_goals_per_game > 2.8:
            factors.append(f"High-scoring league ({game.league.avg_goals_per_game:.1f} avg goals)")
        elif game.league.avg_goals_per_game < 2.2:
            factors.append(f"Low-scoring league ({game.league.avg_goals_per_game:.1f} avg goals)")
        
        # Team factors
        if game.home_team.avg_goals_scored > 2.0:
            factors.append(f"{game.home_team.name} strong attack ({game.home_team.avg_goals_scored:.1f} avg)")
        if game.away_team.avg_goals_scored > 2.0:
            factors.append(f"{game.away_team.name} strong attack ({game.away_team.avg_goals_scored:.1f} avg)")
        
        # Limit to top 5 factors
        return factors[:5]
    
    def _calculate_expected_additional_goals(self, game: Game, result: CalculationResult) -> float:
        """Calculate expected additional goals in remaining time"""
        if not game.is_live:
            # For non-live games, use expected total minus historical average
            return max(0.0, result.expected_total_goals - game.league.avg_goals_per_game)
        
        # For live games, consider time remaining
        minutes_remaining = game.minutes_remaining
        if minutes_remaining <= 0:
            return 0.0
        
        # Adjust expected rate based on current match dynamics
        current_rate = game.goal_rate
        expected_rate = result.expected_additional_goals / minutes_remaining if minutes_remaining > 0 else 0
        
        # Blend current rate with expected rate
        time_decay = game.time_decay_factor
        blended_rate = (current_rate * time_decay) + (expected_rate * (1 - time_decay))
        
        return blended_rate * minutes_remaining
    
    def _create_cache_key(self, game: Game) -> str:
        """Create cache key for game"""
        return f"{game.id}_{game.current_minute}_{game.home_score}_{game.away_score}"
    
    def _get_from_cache(self, key: str) -> Optional[ProbabilityMetrics]:
        """Get item from cache"""
        if key in self._cache:
            entry = self._cache[key]
            if (datetime.now() - entry['timestamp']).seconds < self.cache_ttl:
                return entry['metrics']
            else:
                # Remove expired entry
                del self._cache[key]
        return None
    
    def _add_to_cache(self, key: str, metrics: ProbabilityMetrics):
        """Add item to cache"""
        self._cache[key] = {
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
    
    def _get_default_metrics(self, game: Game) -> ProbabilityMetrics:
        """Get default metrics when calculation fails"""
        # Use league average as baseline
        league_avg = game.league.avg_goals_per_game
        
        # Simple Poisson calculation
        poisson_probs = self._calculate_poisson_probabilities(league_avg)
        
        return ProbabilityMetrics(
            game_id=game.id,
            probability_over_05=poisson_probs['over_05'],
            probability_over_15=poisson_probs['over_15'],
            probability_over_25=poisson_probs['over_25'],
            probability_over_35=poisson_probs['over_35'],
            confidence_score=0.5,  # Medium confidence
            expected_additional_goals=league_avg * 0.5,
            expected_total_goals=league_avg,
        )
    
    def _calculate_poisson_probabilities(self, lambda_param: float) -> Dict[str, float]:
        """Calculate Poisson probabilities for over/under lines"""
        lambda_param = max(self.config.get('poisson_lambda_min', 0.1), 
                          min(lambda_param, self.config.get('poisson_lambda_max', 5.0)))
        
        # Cache Poisson calculations
        cache_key = f"poisson_{lambda_param:.2f}"
        if cache_key in self._poisson_cache:
            return self._poisson_cache[cache_key]
        
        # Calculate cumulative probabilities
        probs = {}
        
        # P(X > k) = 1 - P(X ≤ k)
        for k in [0, 1, 2, 3, 4]:
            prob_leq = stats.poisson.cdf(k, lambda_param)
            prob_over = 1 - prob_leq
            
            # Map to over lines
            if k == 0:
                probs['over_05'] = prob_over
            elif k == 1:
                probs['over_15'] = prob_over
            elif k == 2:
                probs['over_25'] = prob_over
            elif k == 3:
                probs['over_35'] = prob_over
            elif k == 4:
                probs['over_45'] = prob_over
        
        # Cache result
        self._poisson_cache[cache_key] = probs
        
        return probs


# Base Model Class

class BaseModel:
    """Base class for all prediction models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"predictor.model.{self.__class__.__name__.lower()}")
    
    def calculate(self, game: Game, historical_data: Optional[Dict] = None) -> CalculationResult:
        """
        Calculate probabilities using this model
        
        Args:
            game: Game object
            historical_data: Optional historical data
            
        Returns:
            CalculationResult object
        """
        raise NotImplementedError("Subclasses must implement calculate method")
    
    def _calculate_confidence(self, game: Game, data_quality: float = 1.0) -> float:
        """Calculate model-specific confidence score"""
        base_confidence = 0.7  # Base confidence
        
        # Adjust for match stage
        if game.is_live:
            # More minutes = more confidence
            minute_factor = min(game.current_minute / 60, 1.0)
            base_confidence *= 0.5 + (minute_factor * 0.5)
        else:
            # Less confidence for non-live matches
            base_confidence *= 0.6
        
        # Adjust for data quality
        base_confidence *= data_quality
        
        return max(0.0, min(1.0, base_confidence))


# Time-Based Model

class TimeBasedModel(BaseModel):
    """
    Model based on time decay and current match state
    Uses Poisson distribution adjusted for time remaining
    """
    
    def calculate(self, game: Game, historical_data: Optional[Dict] = None) -> CalculationResult:
        """Calculate probabilities based on time and current score"""
        if not game.is_live:
            return self._calculate_for_non_live(game)
        
        return self._calculate_for_live(game)
    
    def _calculate_for_live(self, game: Game) -> CalculationResult:
        """Calculate for live matches"""
        # Current goal rate
        current_rate = game.goal_rate
        
        # Expected rate for remaining time (based on league average)
        league_rate = game.league.avg_goals_per_game / 90
        expected_rate = (league_rate + current_rate) / 2
        
        # Adjust for time decay (later in game, current rate matters more)
        time_decay = game.time_decay_factor
        adjusted_rate = (current_rate * time_decay) + (expected_rate * (1 - time_decay))
        
        # Calculate expected additional goals
        minutes_remaining = game.minutes_remaining
        expected_additional = adjusted_rate * minutes_remaining
        
        # Calculate Poisson probabilities for additional goals
        poisson_probs = self._calculate_poisson_probabilities(expected_additional)
        
        # Adjust for goals already scored
        adjusted_probs = self._adjust_for_existing_goals(game.total_goals, poisson_probs)
        
        # Calculate expected total goals
        expected_total = game.total_goals + expected_additional
        
        # Confidence factors
        confidence_factors = []
        
        # More minutes played = higher confidence
        if game.current_minute > 20:
            confidence_factors.append(min(game.current_minute / 90, 1.0))
        
        # Stable scoreline = higher confidence
        if game.total_goals > 0:
            confidence_factors.append(min(game.total_goals / 4, 1.0))
        
        confidence = statistics.mean(confidence_factors) if confidence_factors else 0.6
        
        # Key factors
        key_factors = [
            f"Current goal rate: {current_rate:.2f}/min",
            f"Time remaining: {minutes_remaining}'",
            f"Time decay factor: {time_decay:.2f}"
        ]
        
        # Warnings
        warnings = []
        if minutes_remaining < 15:
            warnings.append(f"Limited time remaining ({minutes_remaining}')")
        if current_rate < 0.01 and game.total_goals == 0:
            warnings.append("Very low scoring match")
        
        return CalculationResult(
            probabilities=adjusted_probs,
            expected_additional_goals=expected_additional,
            expected_total_goals=expected_total,
            confidence_score=confidence,
            model_contributions={'time_based': 1.0},
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _calculate_for_non_live(self, game: Game) -> CalculationResult:
        """Calculate for non-live (scheduled) matches"""
        # Use league average as baseline
        league_avg = game.league.avg_goals_per_game
        
        # Adjust based on team strengths
        team_adjustment = self._calculate_team_adjustment(game)
        adjusted_lambda = league_avg * team_adjustment
        
        # Calculate Poisson probabilities
        poisson_probs = self._calculate_poisson_probabilities(adjusted_lambda)
        
        # Confidence (lower for non-live matches)
        confidence = 0.5
        
        # Key factors
        key_factors = [
            f"League average: {league_avg:.1f} goals/game",
            f"Team adjustment factor: {team_adjustment:.2f}",
            f"Expected total: {adjusted_lambda:.1f} goals"
        ]
        
        # Warnings
        warnings = ["Match not yet started - prediction based on historical data"]
        
        return CalculationResult(
            probabilities=poisson_probs,
            expected_additional_goals=adjusted_lambda,
            expected_total_goals=adjusted_lambda,
            confidence_score=confidence,
            model_contributions={'time_based': 1.0},
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _calculate_team_adjustment(self, game: Game) -> float:
        """Calculate adjustment factor based on team strengths"""
        # Combine offensive strengths and defensive weaknesses
        home_attack = game.home_team.avg_goals_scored
        home_defense = game.home_team.avg_goals_conceded
        away_attack = game.away_team.avg_goals_scored
        away_defense = game.away_team.avg_goals_conceded
        
        # Expected goals = (home attack * away defense + away attack * home defense) / 2
        expected_goals = ((home_attack * away_defense) + (away_attack * home_defense)) / 2
        
        # Adjustment factor relative to league average
        league_avg = game.league.avg_goals_per_game
        adjustment = expected_goals / league_avg if league_avg > 0 else 1.0
        
        # Cap adjustment
        return max(0.5, min(adjustment, 2.0))
    
    def _calculate_poisson_probabilities(self, lambda_param: float) -> Dict[str, float]:
        """Calculate Poisson probabilities for over/under lines"""
        # Cache key
        cache_key = f"time_poisson_{lambda_param:.2f}"
        if hasattr(self, '_cache') and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate probabilities
        probs = {}
        
        # P(X > k) = 1 - P(X ≤ k)
        k_values = [0.5, 1.5, 2.5, 3.5, 4.5]
        
        for k in k_values:
            # For non-integer k, use floor(k) for Poisson CDF
            k_floor = math.floor(k)
            prob_leq = stats.poisson.cdf(k_floor, lambda_param)
            prob_over = 1 - prob_leq
            
            # Map to line names
            if k == 0.5:
                probs['over_05'] = prob_over
            elif k == 1.5:
                probs['over_15'] = prob_over
            elif k == 2.5:
                probs['over_25'] = prob_over
            elif k == 3.5:
                probs['over_35'] = prob_over
            elif k == 4.5:
                probs['over_45'] = prob_over
        
        # Cache result
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[cache_key] = probs
        
        return probs
    
    def _adjust_for_existing_goals(self, existing_goals: int, poisson_probs: Dict) -> Dict[str, float]:
        """Adjust probabilities for goals already scored"""
        adjusted = {}
        
        for line_name, prob_over in poisson_probs.items():
            # Extract the line value (e.g., 2.5 from 'over_25')
            line_value = float(line_name.split('_')[1]) / 10
            
            # Goals needed for over
            goals_needed = line_value - existing_goals
            
            if goals_needed <= 0:
                # Already over the line
                adjusted[line_name] = 1.0
            else:
                # Probability of scoring at least goals_needed more goals
                # This is a simplification - in reality we'd need to recalculate Poisson
                # with adjusted lambda for remaining time
                adjusted[line_name] = prob_over
        
        return adjusted


# Statistical Model

class StatisticalModel(BaseModel):
    """
    Model based on historical statistics and team performance
    """
    
    def calculate(self, game: Game, historical_data: Optional[Dict] = None) -> CalculationResult:
        """Calculate probabilities based on historical statistics"""
        # Get historical data for teams
        home_history = self._get_team_history(game.home_team, historical_data)
        away_history = self._get_team_history(game.away_team, historical_data)
        
        # Calculate expected goals based on historical performance
        expected_goals = self._calculate_expected_goals(game, home_history, away_history)
        
        # Adjust for current match state if live
        if game.is_live:
            expected_goals = self._adjust_for_live_match(game, expected_goals)
        
        # Calculate Poisson probabilities
        poisson_probs = self._calculate_poisson_probabilities(expected_goals)
        
        # Confidence based on data quality
        data_quality = self._calculate_data_quality(home_history, away_history)
        confidence = self._calculate_confidence(game, data_quality)
        
        # Key factors
        key_factors = self._extract_key_factors(game, home_history, away_history, expected_goals)
        
        # Warnings
        warnings = self._extract_warnings(data_quality, home_history, away_history)
        
        return CalculationResult(
            probabilities=poisson_probs,
            expected_additional_goals=expected_goals,
            expected_total_goals=game.total_goals + expected_goals if game.is_live else expected_goals,
            confidence_score=confidence,
            model_contributions={'statistical': 1.0},
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _get_team_history(self, team: Team, historical_data: Optional[Dict]) -> Dict:
        """Get historical data for team"""
        if historical_data and team.id in historical_data:
            return historical_data[team.id]
        
        # Return default historical data based on team statistics
        return {
            'total_matches': 10,
            'avg_goals_scored': team.avg_goals_scored,
            'avg_goals_conceded': team.avg_goals_conceded,
            'over_25_rate': 0.5,  # Default
            'home_advantage': 1.1,  # 10% home advantage
        }
    
    def _calculate_expected_goals(self, game: Game, home_history: Dict, away_history: Dict) -> float:
        """Calculate expected total goals for the match"""
        # Base: League average
        base_goals = game.league.avg_goals_per_game
        
        # Team adjustments
        home_attack = home_history['avg_goals_scored']
        home_defense = home_history['avg_goals_conceded']
        away_attack = away_history['avg_goals_scored']
        away_defense = away_history['avg_goals_conceded']
        
        # Expected goals formula
        expected_goals = (
            (home_attack * away_defense * home_history.get('home_advantage', 1.1)) +
            (away_attack * home_defense / home_history.get('home_advantage', 1.1))
        ) / 2
        
        # Blend with league average
        blended = (expected_goals * 0.7) + (base_goals * 0.3)
        
        return blended
    
    def _adjust_for_live_match(self, game: Game, expected_additional: float) -> float:
        """Adjust expected additional goals for live match"""
        if not game.is_live:
            return expected_additional
        
        # Consider goals already scored
        if game.total_goals > expected_additional:
            # Match is scoring faster than expected
            return expected_additional * 1.2
        elif game.total_goals < expected_additional * 0.5:
            # Match is scoring slower than expected
            return expected_additional * 0.8
        
        return expected_additional
    
    def _calculate_data_quality(self, home_history: Dict, away_history: Dict) -> float:
        """Calculate data quality score"""
        scores = []
        
        # Number of historical matches
        home_matches = home_history.get('total_matches', 0)
        away_matches = away_history.get('total_matches', 0)
        
        if home_matches >= 5:
            scores.append(0.5)
        elif home_matches >= 2:
            scores.append(0.3)
        else:
            scores.append(0.1)
        
        if away_matches >= 5:
            scores.append(0.5)
        elif away_matches >= 2:
            scores.append(0.3)
        else:
            scores.append(0.1)
        
        return statistics.mean(scores) if scores else 0.2
    
    def _extract_key_factors(self, game: Game, home_history: Dict, away_history: Dict, expected_goals: float) -> List[str]:
        """Extract key factors from statistical analysis"""
        factors = []
        
        # Team strength factors
        if home_history['avg_goals_scored'] > 2.0:
            factors.append(f"{game.home_team.name} strong attack ({home_history['avg_goals_scored']:.1f} avg)")
        
        if away_history['avg_goals_scored'] > 2.0:
            factors.append(f"{game.away_team.name} strong attack ({away_history['avg_goals_scored']:.1f} avg)")
        
        if home_history['avg_goals_conceded'] > 1.5:
            factors.append(f"{game.home_team.name} weak defense ({home_history['avg_goals_conceded']:.1f} avg conceded)")
        
        if away_history['avg_goals_conceded'] > 1.5:
            factors.append(f"{game.away_team.name} weak defense ({away_history['avg_goals_conceded']:.1f} avg conceded)")
        
        # Historical over/under rate
        home_over_rate = home_history.get('over_25_rate', 0.5)
        away_over_rate = away_history.get('over_25_rate', 0.5)
        
        if home_over_rate > 0.6 or away_over_rate > 0.6:
            factors.append("Teams have high historical over 2.5 rates")
        elif home_over_rate < 0.4 and away_over_rate < 0.4:
            factors.append("Teams have low historical over 2.5 rates")
        
        # Expected goals comparison
        if expected_goals > 3.0:
            factors.append(f"High expected goals ({expected_goals:.1f})")
        elif expected_goals < 2.0:
            factors.append(f"Low expected goals ({expected_goals:.1f})")
        
        return factors[:4]  # Limit to 4 factors
    
    def _extract_warnings(self, data_quality: float, home_history: Dict, away_history: Dict) -> List[str]:
        """Extract warnings from statistical analysis"""
        warnings = []
        
        if data_quality < 0.3:
            warnings.append("Limited historical data available")
        
        home_matches = home_history.get('total_matches', 0)
        away_matches = away_history.get('total_matches', 0)
        
        if home_matches < 3:
            warnings.append(f"Limited data for {home_history.get('team_name', 'home team')} ({home_matches} matches)")
        
        if away_matches < 3:
            warnings.append(f"Limited data for {away_history.get('team_name', 'away team')} ({away_matches} matches)")
        
        return warnings


# Momentum Model

class MomentumModel(BaseModel):
    """
    Model based on recent match momentum and in-game events
    """
    
    def calculate(self, game: Game, historical_data: Optional[Dict] = None) -> CalculationResult:
        """Calculate probabilities based on match momentum"""
        if not game.is_live:
            # For non-live matches, use default
            return self._default_calculation(game)
        
        # Calculate momentum score
        momentum_score = game.momentum_score
        
        # Calculate intensity
        intensity = game.match_intensity
        
        # Calculate expected additional goals based on momentum
        expected_additional = self._calculate_expected_from_momentum(game, momentum_score, intensity)
        
        # Calculate probabilities
        poisson_probs = self._calculate_poisson_probabilities(expected_additional)
        
        # Adjust for existing goals
        adjusted_probs = self._adjust_for_existing_goals(game.total_goals, poisson_probs)
        
        # Confidence based on data quality
        confidence = self._calculate_momentum_confidence(game, momentum_score, intensity)
        
        # Key factors
        key_factors = self._extract_momentum_factors(game, momentum_score, intensity)
        
        # Warnings
        warnings = self._extract_momentum_warnings(game, momentum_score, intensity)
        
        return CalculationResult(
            probabilities=adjusted_probs,
            expected_additional_goals=expected_additional,
            expected_total_goals=game.total_goals + expected_additional,
            confidence_score=confidence,
            model_contributions={'momentum': 1.0},
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _calculate_expected_from_momentum(self, game: Game, momentum_score: float, intensity: float) -> float:
        """Calculate expected additional goals from momentum"""
        minutes_remaining = game.minutes_remaining
        
        # Base rate from current goal rate
        current_rate = game.goal_rate
        
        # Adjust based on momentum
        momentum_factor = momentum_score / 10  # Convert to 0-1 scale
        intensity_factor = min(intensity * 2, 1.0)  # Scale intensity
        
        # Combined factor
        combined_factor = (momentum_factor * 0.6) + (intensity_factor * 0.4)
        
        # Adjust current rate
        adjusted_rate = current_rate * (1.0 + combined_factor)
        
        # Calculate expected additional goals
        expected_additional = adjusted_rate * minutes_remaining
        
        # Cap at reasonable values
        return max(0.0, min(expected_additional, minutes_remaining * 0.15))  # Max 0.15 goals/minute
    
    def _calculate_momentum_confidence(self, game: Game, momentum_score: float, intensity: float) -> float:
        """Calculate confidence for momentum model"""
        confidence_factors = []
        
        # Match duration
        if game.current_minute > 30:
            confidence_factors.append(0.8)
        elif game.current_minute > 15:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Event density
        if intensity > 0.5:
            confidence_factors.append(0.7)
        elif intensity > 0.2:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
        
        # Data completeness
        if (game.shots_on_target != (0, 0) and 
            game.possession != (50.0, 50.0) and 
            game.expected_goals != (0.0, 0.0)):
            confidence_factors.append(0.9)
        elif game.shots_on_target != (0, 0):
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    def _extract_momentum_factors(self, game: Game, momentum_score: float, intensity: float) -> List[str]:
        """Extract key factors from momentum analysis"""
        factors = []
        
        # Momentum level
        if momentum_score > 7.0:
            factors.append("High attacking momentum")
        elif momentum_score > 4.0:
            factors.append("Moderate attacking momentum")
        else:
            factors.append("Low attacking momentum")
        
        # Match intensity
        if intensity > 0.8:
            factors.append("High match intensity")
        elif intensity > 0.5:
            factors.append("Moderate match intensity")
        
        # Recent events
        if game.total_goals > 0:
            factors.append(f"{game.total_goals} goal(s) already scored")
        
        if sum(game.shots_on_target) > 10:
            factors.append(f"High shots on target ({sum(game.shots_on_target)})")
        
        if game.expected_goals_total > 2.0:
            factors.append(f"High expected goals ({game.expected_goals_total:.1f})")
        
        return factors[:3]
    
    def _extract_momentum_warnings(self, game: Game, momentum_score: float, intensity: float) -> List[str]:
        """Extract warnings from momentum analysis"""
        warnings = []
        
        if game.current_minute < 20:
            warnings.append("Early stage of match - limited data")
        
        if intensity < 0.2:
            warnings.append("Low match intensity")
        
        if momentum_score < 3.0 and game.total_goals == 0:
            warnings.append("Low attacking momentum with no goals")
        
        return warnings
    
    def _default_calculation(self, game: Game) -> CalculationResult:
        """Default calculation for non-live matches"""
        # Use league average
        league_avg = game.league.avg_goals_per_game
        
        poisson_probs = self._calculate_poisson_probabilities(league_avg)
        
        return CalculationResult(
            probabilities=poisson_probs,
            expected_additional_goals=league_avg,
            expected_total_goals=league_avg,
            confidence_score=0.4,
            model_contributions={'momentum': 1.0},
            key_factors=["Match not yet started"],
            warnings=["No live momentum data available"]
        )


# League-Based Model

class LeagueBasedModel(BaseModel):
    """
    Model based on league tendencies and characteristics
    """
    
    def calculate(self, game: Game, historical_data: Optional[Dict] = None) -> CalculationResult:
        """Calculate probabilities based on league characteristics"""
        # Get league statistics
        league_stats = self._get_league_statistics(game.league)
        
        # Adjust for match status
        if game.is_live:
            expected_additional = self._adjust_for_live_league(game, league_stats)
        else:
            expected_additional = league_stats['expected_goals']
        
        # Calculate probabilities
        poisson_probs = self._calculate_poisson_probabilities(expected_additional)
        
        # Adjust for existing goals if live
        if game.is_live:
            adjusted_probs = self._adjust_for_existing_goals(game.total_goals, poisson_probs)
        else:
            adjusted_probs = poisson_probs
        
        # Confidence
        confidence = self._calculate_league_confidence(game.league, league_stats)
        
        # Key factors
        key_factors = self._extract_league_factors(game.league, league_stats)
        
        # Warnings
        warnings = self._extract_league_warnings(game.league, league_stats)
        
        return CalculationResult(
            probabilities=adjusted_probs,
            expected_additional_goals=expected_additional,
            expected_total_goals=game.total_goals + expected_additional if game.is_live else expected_additional,
            confidence_score=confidence,
            model_contributions={'league_based': 1.0},
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _get_league_statistics(self, league: League) -> Dict[str, float]:
        """Get comprehensive league statistics"""
        return {
            'avg_goals_per_game': league.avg_goals_per_game,
            'over_25_rate': league.over_25_rate,
            'btts_rate': league.btts_rate,
            'home_win_rate': 0.45,  # Example values
            'draw_rate': 0.25,
            'away_win_rate': 0.30,
            'expected_goals': league.avg_goals_per_game,
            'goal_variance': 1.5,  # Measure of scoring volatility
        }
    
    def _adjust_for_live_league(self, game: Game, league_stats: Dict) -> float:
        """Adjust expected goals for live match based on league tendencies"""
        base_expected = league_stats['expected_goals']
        
        # Adjust for current score relative to league average
        league_avg = league_stats['avg_goals_per_game']
        expected_full_match = league_avg
        
        # Calculate expected goals for remaining time
        proportion_remaining = game.minutes_remaining / 90
        expected_remaining = expected_full_match * proportion_remaining
        
        # Adjust based on current scoring rate
        current_rate = game.goal_rate
        league_rate = league_avg / 90
        
        if current_rate > league_rate * 1.5:
            # Scoring faster than league average
            expected_remaining *= 1.3
        elif current_rate < league_rate * 0.7:
            # Scoring slower than league average
            expected_remaining *= 0.7
        
        return expected_remaining
    
    def _calculate_league_confidence(self, league: League, league_stats: Dict) -> float:
        """Calculate confidence based on league data quality"""
        confidence = 0.7  # Base confidence
        
        # Adjust for league tier
        if league.tier == 1:
            confidence *= 1.1  # Top leagues have better data
        elif league.tier == 2:
            confidence *= 1.0
        else:
            confidence *= 0.8  # Lower tiers have less reliable data
        
        # Adjust for data availability
        if league_stats['avg_goals_per_game'] > 0:
            confidence *= 1.0
        else:
            confidence *= 0.6
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_league_factors(self, league: League, league_stats: Dict) -> List[str]:
        """Extract key factors from league analysis"""
        factors = []
        
        # Scoring level
        avg_goals = league_stats['avg_goals_per_game']
        if avg_goals > 2.8:
            factors.append(f"High-scoring league ({avg_goals:.1f} avg goals)")
        elif avg_goals > 2.3:
            factors.append(f"Moderate-scoring league ({avg_goals:.1f} avg goals)")
        else:
            factors.append(f"Low-scoring league ({avg_goals:.1f} avg goals)")
        
        # Over 2.5 tendency
        over_rate = league_stats['over_25_rate']
        if over_rate > 0.6:
            factors.append(f"High over 2.5 rate ({over_rate:.0%})")
        elif over_rate < 0.4:
            factors.append(f"Low over 2.5 rate ({over_rate:.0%})")
        
        # Both teams to score tendency
        btts_rate = league_stats['btts_rate']
        if btts_rate > 0.6:
            factors.append(f"High BTTS rate ({btts_rate:.0%})")
        
        # League tier
        if league.tier == 1:
            factors.append("Top tier league (reliable data)")
        
        return factors[:3]
    
    def _extract_league_warnings(self, league: League, league_stats: Dict) -> List[str]:
        """Extract warnings from league analysis"""
        warnings = []
        
        if league.tier > 2:
            warnings.append(f"Lower tier league (tier {league.tier}) - less reliable data")
        
        if league_stats['avg_goals_per_game'] < 2.0:
            warnings.append("Very low-scoring league")
        
        return warnings


# Odds-Based Model

class OddsBasedModel(BaseModel):
    """
    Model based on betting market odds
    Uses implied probabilities from bookmakers
    """
    
    def calculate(self, game: Game, historical_data: Optional[Dict] = None) -> CalculationResult:
        """Calculate probabilities based on market odds"""
        # Check if odds are available
        if not game.odds_over_25 or not game.odds_under_25:
            return self._default_calculation(game)
        
        # Calculate implied probabilities
        implied_prob_over = calculate_implied_probability(game.odds_over_25)
        implied_prob_under = calculate_implied_probability(game.odds_under_25)
        
        # Adjust for bookmaker margin
        total_prob = implied_prob_over + implied_prob_under
        if total_prob > 1.0:
            # Remove margin by normalizing
            implied_prob_over /= total_prob
            implied_prob_under /= total_prob
        
        # Calculate other over/under lines based on implied probability
        probabilities = self._calculate_all_probabilities(implied_prob_over, game)
        
        # Calculate expected goals from implied probability
        expected_additional = self._calculate_expected_from_odds(implied_prob_over, game)
        
        # Confidence based on odds quality
        confidence = self._calculate_odds_confidence(game)
        
        # Key factors
        key_factors = self._extract_odds_factors(game, implied_prob_over)
        
        # Warnings
        warnings = self._extract_odds_warnings(game, implied_prob_over)
        
        return CalculationResult(
            probabilities=probabilities,
            expected_additional_goals=expected_additional,
            expected_total_goals=game.total_goals + expected_additional if game.is_live else expected_additional,
            confidence_score=confidence,
            model_contributions={'odds_based': 1.0},
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _calculate_all_probabilities(self, implied_prob_over_25: float, game: Game) -> Dict[str, float]:
        """Calculate probabilities for all over/under lines based on implied probability"""
        # This is a simplified approach - in reality you'd need to model
        # the entire distribution based on multiple odds lines
        
        # Use implied probability as base
        base_prob = implied_prob_over_25
        
        # Estimate other lines (simplified linear scaling)
        probabilities = {
            'over_05': min(base_prob * 1.4, 0.99),  # Higher probability for over 0.5
            'over_15': min(base_prob * 1.2, 0.95),  # Slightly higher for over 1.5
            'over_25': base_prob,
            'over_35': max(base_prob * 0.8, 0.01),  # Lower for over 3.5
            'over_45': max(base_prob * 0.6, 0.01),  # Even lower for over 4.5
        }
        
        # Adjust for existing goals if live
        if game.is_live:
            probabilities = self._adjust_for_existing_goals(game.total_goals, probabilities)
        
        return probabilities
    
    def _calculate_expected_from_odds(self, implied_prob_over_25: float, game: Game) -> float:
        """Calculate expected additional goals from odds"""
        # Convert probability to expected goals using inverse Poisson
        # This is an approximation
        
        # Find lambda such that P(X > 2) = implied_prob_over_25
        # We solve: 1 - P(X ≤ 2) = implied_prob_over_25
        # Where P(X ≤ 2) = e^-λ * (1 + λ + λ²/2)
        
        target_prob = implied_prob_over_25
        
        # Binary search for lambda
        low, high = 0.1, 5.0
        for _ in range(20):  # 20 iterations for precision
            mid = (low + high) / 2
            prob = 1 - (math.exp(-mid) * (1 + mid + (mid**2)/2))
            
            if prob < target_prob:
                low = mid
            else:
                high = mid
        
        lambda_est = (low + high) / 2
        
        # Adjust for match status
        if game.is_live:
            # Adjust for time remaining
            proportion_remaining = game.minutes_remaining / 90
            lambda_est *= proportion_remaining
        
        return lambda_est
    
    def _calculate_odds_confidence(self, game: Game) -> float:
        """Calculate confidence based on odds quality"""
        confidence = 0.6  # Base confidence for odds
        
        # Adjust for odds availability
        if game.odds_over_25 and game.odds_under_25:
            confidence *= 1.2
        
        # Adjust for odds reliability (extreme odds are less reliable)
        if game.odds_over_25:
            if game.odds_over_25 < 1.2 or game.odds_over_25 > 5.0:
                confidence *= 0.8  # Extreme odds less reliable
        
        # Adjust for match stage
        if game.is_live and game.current_minute > 60:
            confidence *= 1.1  # More reliable later in match
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_odds_factors(self, game: Game, implied_prob_over: float) -> List[str]:
        """Extract key factors from odds analysis"""
        factors = []
        
        if game.odds_over_25:
            factors.append(f"Market odds: {game.odds_over_25:.2f} for Over 2.5")
        
        # Market sentiment
        if implied_prob_over > 0.6:
            factors.append("Market favors Over 2.5")
        elif implied_prob_over < 0.4:
            factors.append("Market favors Under 2.5")
        else:
            factors.append("Market neutral on Over/Under 2.5")
        
        # Value assessment
        if hasattr(game, 'probability_metrics') and game.probability_metrics:
            market_prob = implied_prob_over
            model_prob = game.probability_metrics.probability_over_25
            
            if model_prob > market_prob + 0.1:
                factors.append("Potential value in Over 2.5")
            elif model_prob < market_prob - 0.1:
                factors.append("Potential value in Under 2.5")
        
        return factors[:2]
    
    def _extract_odds_warnings(self, game: Game, implied_prob_over: float) -> List[str]:
        """Extract warnings from odds analysis"""
        warnings = []
        
        if not game.odds_over_25 or not game.odds_under_25:
            warnings.append("Limited odds data available")
        
        if game.odds_over_25 and (game.odds_over_25 < 1.1 or game.odds_over_25 > 10.0):
            warnings.append("Extreme odds values - market may be illiquid")
        
        # Check for arbitrage opportunity (shouldn't happen in efficient markets)
        if game.odds_over_25 and game.odds_under_25:
            total_prob = (1/game.odds_over_25) + (1/game.odds_under_25)
            if total_prob < 0.95:
                warnings.append("Potential arbitrage opportunity detected")
        
        return warnings
    
    def _default_calculation(self, game: Game) -> CalculationResult:
        """Default calculation when odds are not available"""
        # Use league average
        league_avg = game.league.avg_goals_per_game
        
        poisson_probs = self._calculate_poisson_probabilities(league_avg)
        
        return CalculationResult(
            probabilities=poisson_probs,
            expected_additional_goals=league_avg,
            expected_total_goals=league_avg,
            confidence_score=0.3,  # Low confidence without odds
            model_contributions={'odds_based': 1.0},
            key_factors=["No odds data available"],
            warnings=["Using league average as proxy for market expectations"]
        )


# Utility functions

def create_calculator(config: Optional[Dict] = None) -> ProbabilityCalculator:
    """
    Create a ProbabilityCalculator instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ProbabilityCalculator instance
    """
    return ProbabilityCalculator(config)


def calculate_risk_assessment(game: Game, metrics: ProbabilityMetrics) -> RiskAssessment:
    """
    Calculate risk assessment for a prediction
    
    Args:
        game: Game object
        metrics: ProbabilityMetrics object
        
    Returns:
        RiskAssessment object
    """
    # Time risk (higher later in game)
    time_risk = 0.0
    if game.is_live:
        # Risk increases as match progresses
        time_risk = min(game.current_minute / 90, 0.8)
    
    # Data risk (based on data completeness)
    data_factors = [
        1.0 if game.shots_on_target != (0, 0) else 0.5,
        1.0 if game.expected_goals != (0.0, 0.0) else 0.5,
        1.0 if game.possession != (50.0, 50.0) else 0.5,
        1.0 if game.odds_over_25 else 0.7,
    ]
    data_risk = 1.0 - (sum(data_factors) / len(data_factors))
    
    # Volatility risk (based on match characteristics)
    volatility_risk = 0.0
    if game.is_live:
        if game.total_goals > 3:
            volatility_risk = 0.7  # High scoring = volatile
        elif game.momentum_score > 7.0:
            volatility_risk = 0.6  # High momentum = more volatile
    
    # League risk (lower tiers = higher risk)
    league_risk = 0.0
    if game.league.tier == 1:
        league_risk = 0.2
    elif game.league.tier == 2:
        league_risk = 0.4
    else:
        league_risk = 0.6
    
    # Confidence adjustment (higher confidence = lower risk)
    confidence_adjustment = 1.0 - metrics.confidence_score
    
    # Apply adjustments
    time_risk *= confidence_adjustment
    data_risk *= confidence_adjustment
    volatility_risk *= confidence_adjustment
    league_risk *= confidence_adjustment
    
    return RiskAssessment(
        game_id=game.id,
        time_risk=time_risk,
        data_risk=data_risk,
        volatility_risk=volatility_risk,
        league_risk=league_risk
    )


# Example usage
if __name__ == "__main__":
    # Test the calculator
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample game
    from src.predictor.models import Team, League, Game, GameStatus
    
    home_team = Team(
        id="team_1",
        name="Manchester City",
        avg_goals_scored=2.3,
        avg_goals_conceded=0.8
    )
    
    away_team = Team(
        id="team_2",
        name="Liverpool FC",
        avg_goals_scored=2.1,
        avg_goals_conceded=1.0
    )
    
    league = League(
        id="league_1",
        name="English Premier League",
        country="England",
        avg_goals_per_game=2.8,
        over_25_rate=0.55
    )
    
    game = Game(
        id="test_game_1",
        home_team=home_team,
        away_team=away_team,
        league=league,
        start_time=datetime.now(),
        current_minute=65,
        status=GameStatus.LIVE,
        home_score=2,
        away_score=1,
        shots_on_target=(7, 4),
        shots_total=(15, 10),
        possession=(58, 42),
        expected_goals=(2.1, 1.4),
        odds_over_25=1.85,
        odds_under_25=1.95
    )
    
    # Create calculator
    calculator = ProbabilityCalculator()
    
    # Calculate probabilities
    print("Calculating probabilities...")
    metrics = calculator.calculate_probabilities(game)
    
    print(f"\nResults for {game.home_team.name} vs {game.away_team.name}:")
    print(f"Over 0.5: {metrics.probability_over_05:.1%}")
    print(f"Over 1.5: {metrics.probability_over_15:.1%}")
    print(f"Over 2.5: {metrics.probability_over_25:.1%}")
    print(f"Over 3.5: {metrics.probability_over_35:.1%}")
    print(f"Confidence: {metrics.confidence_score:.1%}")
    print(f"Expected Total Goals: {metrics.expected_total_goals:.2f}")
    print(f"Expected Additional Goals: {metrics.expected_additional_goals:.2f}")
    
    # Calculate risk assessment
    risk = calculate_risk_assessment(game, metrics)
    print(f"\nRisk Assessment:")
    print(f"Overall Risk: {risk.overall_risk_score:.2f} ({risk.risk_level.value})")
    print(f"Time Risk: {risk.time_risk:.2f}")
    print(f"Data Risk: {risk.data_risk:.2f}")
    print(f"Volatility Risk: {risk.volatility_risk:.2f}")
    print(f"League Risk: {risk.league_risk:.2f}")
    
    if risk.recommended_stake:
        print(f"Recommended Stake: {risk.recommended_stake:.2%}")
