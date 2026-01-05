"""
models.py - Data models for Over/Under Predictor system
Defines all data structures used throughout the prediction pipeline
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
from decimal import Decimal, ROUND_HALF_UP


class GameStatus(Enum):
    """Status of a football match"""
    SCHEDULED = "scheduled"
    LIVE = "live"
    HALFTIME = "halftime"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


class MarketType(Enum):
    """Types of betting markets"""
    OVER_UNDER = "over_under"
    ASIAN_HANDICAP = "asian_handicap"
    MONEYLINE = "moneyline"
    BOTH_TEAMS_SCORE = "both_teams_score"
    DOUBLE_CHANCE = "double_chance"


class OverUnderLine(Enum):
    """Common over/under lines"""
    OVER_0_5 = 0.5
    OVER_1_5 = 1.5
    OVER_2_5 = 2.5
    OVER_3_5 = 3.5
    OVER_4_5 = 4.5
    UNDER_0_5 = -0.5
    UNDER_1_5 = -1.5
    UNDER_2_5 = -2.5
    UNDER_3_5 = -3.5
    UNDER_4_5 = -4.5


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"            # 75-89%
    MEDIUM = "medium"        # 60-74%
    LOW = "low"              # 40-59%
    VERY_LOW = "very_low"    # 0-39%


class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Team:
    """Football team data"""
    id: str
    name: str
    short_name: Optional[str] = None
    country: Optional[str] = None
    league: Optional[str] = None
    
    # Performance metrics
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    form_last_5: List[str] = field(default_factory=list)  # W/W/D/L/W
    clean_sheets: int = 0
    failed_to_score: int = 0
    
    def __post_init__(self):
        if not self.short_name:
            self.short_name = ''.join([word[0].upper() for word in self.name.split()[:3]])
    
    @property
    def offensive_strength(self) -> float:
        """Calculate offensive strength metric"""
        return self.avg_goals_scored
    
    @property
    def defensive_strength(self) -> float:
        """Calculate defensive strength metric"""
        return self.avg_goals_conceded
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class League:
    """Football league data"""
    id: str
    name: str
    country: str
    tier: int = 1  # 1 for top division
    
    # League statistics
    avg_goals_per_game: float = 2.5
    home_advantage_factor: float = 1.1
    over_25_rate: float = 0.5
    btts_rate: float = 0.5
    
    def __str__(self) -> str:
        return f"{self.name} ({self.country})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Game:
    """Complete game/match data"""
    # Basic identification
    id: str
    home_team: Team
    away_team: Team
    league: League
    start_time: datetime
    current_minute: int = 0
    status: GameStatus = GameStatus.SCHEDULED
    
    # Score
    home_score: int = 0
    away_score: int = 0
    
    # Match statistics
    shots_on_target: Tuple[int, int] = (0, 0)  # (home, away)
    shots_total: Tuple[int, int] = (0, 0)
    possession: Tuple[float, float] = (50.0, 50.0)  # (home %, away %)
    corners: Tuple[int, int] = (0, 0)
    fouls: Tuple[int, int] = (0, 0)
    yellow_cards: Tuple[int, int] = (0, 0)
    red_cards: Tuple[int, int] = (0, 0)
    
    # Advanced metrics
    expected_goals: Tuple[float, float] = (0.0, 0.0)  # (xG_home, xG_away)
    expected_goals_total: float = 0.0
    
    # Odds data (if available)
    odds_over_25: Optional[float] = None
    odds_under_25: Optional[float] = None
    odds_home_win: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away_win: Optional[float] = None
    
    # Additional data
    venue: Optional[str] = None
    referee: Optional[str] = None
    attendance: Optional[int] = None
    weather: Optional[Dict[str, Any]] = None
    
    # Tracking
    last_updated: datetime = field(default_factory=datetime.now)
    data_source: str = "unknown"
    
    @property
    def total_goals(self) -> int:
        """Total goals in the match"""
        return self.home_score + self.away_score
    
    @property
    def goal_difference(self) -> int:
        """Goal difference (home - away)"""
        return self.home_score - self.away_score
    
    @property
    def is_live(self) -> bool:
        """Check if game is currently live"""
        return self.status in [GameStatus.LIVE, GameStatus.HALFTIME]
    
    @property
    def is_finished(self) -> bool:
        """Check if game is finished"""
        return self.status == GameStatus.FINISHED
    
    @property
    def minutes_remaining(self) -> int:
        """Calculate minutes remaining in the match"""
        if not self.is_live:
            return 0
        
        if self.status == GameStatus.HALFTIME:
            return 45
        
        if self.current_minute <= 45:
            return 45 - self.current_minute
        else:
            return 90 - self.current_minute
    
    @property
    def time_decay_factor(self) -> float:
        """
        Factor based on time elapsed in the match
        Higher value = more weight to current score
        """
        if not self.is_live:
            return 0.0
        
        elapsed_ratio = self.current_minute / 90.0
        return min(0.9, elapsed_ratio * 1.2)  # Cap at 0.9
    
    @property
    def goal_rate(self) -> float:
        """Goals per minute rate"""
        if self.current_minute > 0:
            return self.total_goals / self.current_minute
        return 0.0
    
    @property
    def expected_goal_rate(self) -> float:
        """Expected goals per minute rate"""
        if self.current_minute > 0:
            return self.expected_goals_total / self.current_minute
        return 0.0
    
    @property
    def shots_on_target_rate(self) -> float:
        """Shots on target per minute"""
        total_shots_on_target = self.shots_on_target[0] + self.shots_on_target[1]
        if self.current_minute > 0:
            return total_shots_on_target / self.current_minute
        return 0.0
    
    @property
    def momentum_score(self) -> float:
        """
        Calculate match momentum based on recent events
        Higher score = more attacking momentum
        """
        score = 0.0
        
        # Recent goals boost momentum
        if self.total_goals > 0:
            score += min(self.total_goals * 0.3, 1.0)
        
        # Shots on target indicate attacking intent
        shots_score = min(self.shots_on_target_rate * 10, 2.0)
        score += shots_score
        
        # Expected goals indicate quality chances
        xg_score = min(self.expected_goal_rate * 15, 2.0)
        score += xg_score
        
        # Possession advantage
        possession_diff = self.possession[0] - self.possession[1]
        score += abs(possession_diff) / 100
        
        # Normalize to 0-10 scale
        return min(score, 10.0)
    
    @property
    def match_intensity(self) -> float:
        """
        Calculate match intensity based on events per minute
        """
        total_events = (
            self.total_goals +
            self.shots_total[0] + self.shots_total[1] +
            self.corners[0] + self.corners[1] +
            self.fouls[0] + self.fouls[1]
        )
        
        if self.current_minute > 0:
            return total_events / self.current_minute
        return 0.0
    
    def get_game_context(self) -> Dict[str, Any]:
        """Get contextual information about the game"""
        return {
            "is_live": self.is_live,
            "is_finished": self.is_finished,
            "minutes_played": self.current_minute,
            "minutes_remaining": self.minutes_remaining,
            "total_goals": self.total_goals,
            "goal_rate": self.goal_rate,
            "momentum": self.momentum_score,
            "intensity": self.match_intensity,
            "time_decay": self.time_decay_factor,
            "league_avg_goals": self.league.avg_goals_per_game,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        
        # Handle datetime serialization
        data['start_time'] = self.start_time.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        
        # Handle enum serialization
        data['status'] = self.status.value
        
        # Handle nested objects
        data['home_team'] = self.home_team.to_dict()
        data['away_team'] = self.away_team.to_dict()
        data['league'] = self.league.to_dict()
        
        return data
    
    def __str__(self) -> str:
        status_symbol = {
            GameStatus.LIVE: "⚽",
            GameStatus.HALFTIME: "⏸️",
            GameStatus.FINISHED: "✅",
            GameStatus.SCHEDULED: "🕐",
            GameStatus.POSTPONED: "❌",
        }.get(self.status, "❓")
        
        return (f"{status_symbol} {self.home_team.name} {self.home_score}-{self.away_score} {self.away_team.name} "
                f"({self.current_minute}') - {self.league.name}")


@dataclass
class HistoricalPerformance:
    """Historical performance data for a team or league"""
    team_id: str
    total_matches: int = 0
    
    # Over/Under statistics
    over_05_count: int = 0
    over_15_count: int = 0
    over_25_count: int = 0
    over_35_count: int = 0
    
    # Average statistics
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    avg_total_goals: float = 0.0
    
    # Recent form (last 10 matches)
    recent_matches: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def over_05_rate(self) -> float:
        """Probability of over 0.5 goals"""
        if self.total_matches == 0:
            return 0.0
        return self.over_05_count / self.total_matches
    
    @property
    def over_15_rate(self) -> float:
        """Probability of over 1.5 goals"""
        if self.total_matches == 0:
            return 0.0
        return self.over_15_count / self.total_matches
    
    @property
    def over_25_rate(self) -> float:
        """Probability of over 2.5 goals"""
        if self.total_matches == 0:
            return 0.0
        return self.over_25_count / self.total_matches
    
    @property
    def over_35_rate(self) -> float:
        """Probability of over 3.5 goals"""
        if self.total_matches == 0:
            return 0.0
        return self.over_35_count / self.total_matches
    
    def update_from_match(self, match_data: Dict[str, Any]):
        """Update statistics from a new match"""
        total_goals = match_data.get('total_goals', 0)
        
        self.total_matches += 1
        
        # Update over/under counts
        if total_goals > 0.5:
            self.over_05_count += 1
        if total_goals > 1.5:
            self.over_15_count += 1
        if total_goals > 2.5:
            self.over_25_count += 1
        if total_goals > 3.5:
            self.over_35_count += 1
        
        # Update averages (moving average)
        goals_scored = match_data.get('goals_scored', 0)
        goals_conceded = match_data.get('goals_conceded', 0)
        
        self.avg_goals_scored = (
            (self.avg_goals_scored * (self.total_matches - 1) + goals_scored) 
            / self.total_matches
        )
        self.avg_goals_conceded = (
            (self.avg_goals_conceded * (self.total_matches - 1) + goals_conceded) 
            / self.total_matches
        )
        self.avg_total_goals = (
            (self.avg_total_goals * (self.total_matches - 1) + total_goals) 
            / self.total_matches
        )
        
        # Add to recent matches (keep last 10)
        self.recent_matches.append(match_data)
        if len(self.recent_matches) > 10:
            self.recent_matches = self.recent_matches[-10:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "team_id": self.team_id,
            "total_matches": self.total_matches,
            "over_05_rate": self.over_05_rate,
            "over_15_rate": self.over_15_rate,
            "over_25_rate": self.over_25_rate,
            "over_35_rate": self.over_35_rate,
            "avg_goals_scored": self.avg_goals_scored,
            "avg_goals_conceded": self.avg_goals_conceded,
            "avg_total_goals": self.avg_total_goals,
            "recent_matches_count": len(self.recent_matches),
        }


@dataclass
class ProbabilityMetrics:
    """Probability metrics for a game"""
    game_id: str
    
    # Base probabilities (0.0 to 1.0)
    probability_over_05: float = 0.0
    probability_over_15: float = 0.0
    probability_over_25: float = 0.0
    probability_over_35: float = 0.0
    probability_over_45: float = 0.0
    
    # Implied probabilities from odds (if available)
    implied_prob_over_25: Optional[float] = None
    implied_prob_under_25: Optional[float] = None
    
    # Confidence metrics
    confidence_score: float = 0.0  # 0.0 to 1.0
    confidence_level: PredictionConfidence = PredictionConfidence.VERY_LOW
    
    # Expected values
    expected_additional_goals: float = 0.0
    expected_total_goals: float = 0.0
    
    # Model contributions (what factors contributed to the prediction)
    model_contributions: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Calculate confidence level from score
        if self.confidence_score >= 0.9:
            self.confidence_level = PredictionConfidence.VERY_HIGH
        elif self.confidence_score >= 0.75:
            self.confidence_level = PredictionConfidence.HIGH
        elif self.confidence_score >= 0.6:
            self.confidence_level = PredictionConfidence.MEDIUM
        elif self.confidence_score >= 0.4:
            self.confidence_level = PredictionConfidence.LOW
    
    @property
    def value_bet_score(self) -> Optional[float]:
        """
        Calculate value bet score (difference between our probability and implied probability)
        Positive = potential value bet
        """
        if self.implied_prob_over_25 is None:
            return None
        
        # Kelly Criterion based value
        value = self.probability_over_25 - self.implied_prob_over_25
        return value
    
    @property
    def recommended_bet(self) -> Optional[str]:
        """Get recommended bet based on probabilities"""
        if self.probability_over_25 >= 0.75 and self.confidence_level in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
            return "OVER 2.5"
        elif self.probability_over_25 <= 0.25 and self.confidence_level in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
            return "UNDER 2.5"
        return None
    
    def get_probability_for_line(self, line: float) -> float:
        """Get probability for a specific over/under line"""
        line_to_prob = {
            0.5: self.probability_over_05,
            1.5: self.probability_over_15,
            2.5: self.probability_over_25,
            3.5: self.probability_over_35,
            4.5: self.probability_over_45,
        }
        return line_to_prob.get(abs(line), 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "game_id": self.game_id,
            "probabilities": {
                "over_0.5": self.probability_over_05,
                "over_1.5": self.probability_over_15,
                "over_2.5": self.probability_over_25,
                "over_3.5": self.probability_over_35,
                "over_4.5": self.probability_over_45,
            },
            "implied_probabilities": {
                "over_2.5": self.implied_prob_over_25,
                "under_2.5": self.implied_prob_under_25,
            },
            "confidence": {
                "score": self.confidence_score,
                "level": self.confidence_level.value,
            },
            "expected_values": {
                "additional_goals": self.expected_additional_goals,
                "total_goals": self.expected_total_goals,
            },
            "value_bet_score": self.value_bet_score,
            "recommended_bet": self.recommended_bet,
            "model_contributions": self.model_contributions,
        }


@dataclass
class RiskAssessment:
    """Risk assessment for a prediction"""
    game_id: str
    
    # Risk factors (0.0 to 1.0, higher = more risk)
    time_risk: float = 0.0  # Risk from time remaining
    data_risk: float = 0.0  # Risk from incomplete data
    volatility_risk: float = 0.0  # Risk from game volatility
    league_risk: float = 0.0  # Risk from league unpredictability
    
    # Overall risk metrics
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Recommendations
    recommended_stake: Optional[float] = None  # As percentage of bankroll
    max_stake: float = 0.05  # 5% maximum
    
    def __post_init__(self):
        # Calculate overall risk score (weighted average)
        weights = {
            'time_risk': 0.4,
            'data_risk': 0.3,
            'volatility_risk': 0.2,
            'league_risk': 0.1,
        }
        
        self.overall_risk_score = (
            weights['time_risk'] * self.time_risk +
            weights['data_risk'] * self.data_risk +
            weights['volatility_risk'] * self.volatility_risk +
            weights['league_risk'] * self.league_risk
        )
        
        # Determine risk level
        if self.overall_risk_score <= 0.2:
            self.risk_level = RiskLevel.VERY_LOW
        elif self.overall_risk_score <= 0.4:
            self.risk_level = RiskLevel.LOW
        elif self.overall_risk_score <= 0.6:
            self.risk_level = RiskLevel.MEDIUM
        elif self.overall_risk_score <= 0.8:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.VERY_HIGH
        
        # Calculate recommended stake using Kelly Criterion adjusted for risk
        if self.recommended_stake is None:
            self._calculate_recommended_stake()
    
    def _calculate_recommended_stake(self):
        """Calculate recommended stake based on risk"""
        # Base Kelly would be: (bp - q) / b
        # Simplified version adjusted for risk
        base_kelly = 0.02  # 2% base
        
        # Adjust for risk level
        risk_adjustment = {
            RiskLevel.VERY_LOW: 1.5,
            RiskLevel.LOW: 1.2,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.7,
            RiskLevel.VERY_HIGH: 0.4,
        }
        
        adjusted_stake = base_kelly * risk_adjustment.get(self.risk_level, 1.0)
        
        # Cap at max stake
        self.recommended_stake = min(adjusted_stake, self.max_stake)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "game_id": self.game_id,
            "risk_factors": {
                "time_risk": self.time_risk,
                "data_risk": self.data_risk,
                "volatility_risk": self.volatility_risk,
                "league_risk": self.league_risk,
            },
            "overall_risk": {
                "score": self.overall_risk_score,
                "level": self.risk_level.value,
            },
            "stake_recommendation": {
                "recommended_stake": self.recommended_stake,
                "max_stake": self.max_stake,
            }
        }


@dataclass
class Prediction:
    """Complete prediction for a game"""
    # Identification
    id: str
    game: Game
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Core prediction
    probability_metrics: ProbabilityMetrics
    risk_assessment: RiskAssessment
    
    # Additional analysis
    key_factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Betting recommendations
    recommended_action: Optional[str] = None
    confidence_summary: str = ""
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction has high confidence"""
        return self.probability_metrics.confidence_level in [
            PredictionConfidence.HIGH,
            PredictionConfidence.VERY_HIGH
        ]
    
    @property
    def is_value_bet(self) -> bool:
        """Check if this represents a value bet opportunity"""
        value_score = self.probability_metrics.value_bet_score
        return (
            value_score is not None and 
            value_score > 0.05 and  # At least 5% edge
            self.is_high_confidence
        )
    
    @property
    def alert_priority(self) -> int:
        """Calculate alert priority (higher = more urgent)"""
        priority = 0
        
        # High confidence high probability gets highest priority
        if self.probability_metrics.probability_over_25 >= 0.75 and self.is_high_confidence:
            priority += 100
        
        # Value bet opportunity
        if self.is_value_bet:
            priority += 50
        
        # Time urgency (earlier in game = higher priority)
        if self.game.is_live and self.game.current_minute <= 60:
            priority += (60 - self.game.current_minute)
        
        return priority
    
    def generate_summary(self) -> str:
        """Generate human-readable summary"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(
            f"🎯 Prediction for: {self.game.home_team.name} vs {self.game.away_team.name}"
        )
        
        # Current state
        if self.game.is_live:
            summary_parts.append(
                f"📊 Current: {self.game.home_score}-{self.game.away_score} ({self.game.current_minute}')"
            )
        else:
            summary_parts.append(f"🕐 Status: {self.game.status.value}")
        
        # Key prediction
        over_25_prob = self.probability_metrics.probability_over_25
        confidence = self.probability_metrics.confidence_level.value
        
        summary_parts.append(
            f"📈 Over 2.5 Probability: {over_25_prob:.1%} ({confidence} confidence)"
        )
        
        # Recommendation
        if self.recommended_action:
            summary_parts.append(f"💡 Recommendation: {self.recommended_action}")
        
        # Risk
        risk_level = self.risk_assessment.risk_level.value
        summary_parts.append(f"⚠️  Risk Level: {risk_level}")
        
        # Key factors (first 3)
        if self.key_factors:
            factors = ", ".join(self.key_factors[:3])
            summary_parts.append(f"🔑 Key Factors: {factors}")
        
        return "\n".join(summary_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction_id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "game": self.game.to_dict(),
            "probability_metrics": self.probability_metrics.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "analysis": {
                "key_factors": self.key_factors,
                "warnings": self.warnings,
                "notes": self.notes,
            },
            "recommendations": {
                "recommended_action": self.recommended_action,
                "confidence_summary": self.confidence_summary,
                "is_high_confidence": self.is_high_confidence,
                "is_value_bet": self.is_value_bet,
                "alert_priority": self.alert_priority,
            },
            "summary": self.generate_summary(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def __str__(self) -> str:
        return self.generate_summary()


@dataclass
class BatchPrediction:
    """Batch of predictions for multiple games"""
    predictions: List[Prediction]
    batch_id: str = field(default_factory=lambda: f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Statistics
    total_games: int = 0
    high_confidence_predictions: int = 0
    value_bet_opportunities: int = 0
    
    def __post_init__(self):
        self.total_games = len(self.predictions)
        self.high_confidence_predictions = sum(1 for p in self.predictions if p.is_high_confidence)
        self.value_bet_opportunities = sum(1 for p in self.predictions if p.is_value_bet)
    
    @property
    def success_rate_estimate(self) -> float:
        """Estimate success rate based on confidence levels"""
        if not self.predictions:
            return 0.0
        
        total_weight = 0.0
        weighted_success = 0.0
        
        for prediction in self.predictions:
            confidence = prediction.probability_metrics.confidence_score
            probability = prediction.probability_metrics.probability_over_25
            
            weight = confidence
            success_estimate = probability * confidence
            
            total_weight += weight
            weighted_success += success_estimate
        
        if total_weight == 0:
            return 0.0
        
        return weighted_success / total_weight
    
    def get_high_priority_predictions(self, limit: int = 5) -> List[Prediction]:
        """Get predictions sorted by alert priority"""
        sorted_predictions = sorted(
            self.predictions,
            key=lambda p: p.alert_priority,
            reverse=True
        )
        return sorted_predictions[:limit]
    
    def filter_by_confidence(self, min_confidence: PredictionConfidence) -> 'BatchPrediction':
        """Filter predictions by minimum confidence level"""
        confidence_values = {
            PredictionConfidence.VERY_LOW: 0,
            PredictionConfidence.LOW: 1,
            PredictionConfidence.MEDIUM: 2,
            PredictionConfidence.HIGH: 3,
            PredictionConfidence.VERY_HIGH: 4,
        }
        
        min_value = confidence_values[min_confidence]
        
        filtered = [
            p for p in self.predictions
            if confidence_values[p.probability_metrics.confidence_level] >= min_value
        ]
        
        return BatchPrediction(predictions=filtered)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp.isoformat(),
            "statistics": {
                "total_games": self.total_games,
                "high_confidence_predictions": self.high_confidence_predictions,
                "value_bet_opportunities": self.value_bet_opportunities,
                "success_rate_estimate": self.success_rate_estimate,
            },
            "predictions": [p.to_dict() for p in self.predictions],
            "high_priority_predictions": [
                p.to_dict() for p in self.get_high_priority_predictions(5)
            ],
        }
    
    def save_to_file(self, filepath: str):
        """Save batch predictions to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BatchPrediction':
        """Load batch predictions from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Note: This is simplified - you'd need to reconstruct objects properly
        return cls(predictions=[])


# Factory functions for easy object creation

def create_game_from_scraped_data(scraped_data: Dict[str, Any]) -> Game:
    """Create Game object from scraped data"""
    from src.scrapers.base_scraper import ScrapedGame
    
    # Extract teams
    home_team = Team(
        id=f"team_{scraped_data.get('home_team', '').lower().replace(' ', '_')}",
        name=scraped_data.get('home_team', 'Unknown'),
        country=scraped_data.get('country', 'Unknown'),
    )
    
    away_team = Team(
        id=f"team_{scraped_data.get('away_team', '').lower().replace(' ', '_')}",
        name=scraped_data.get('away_team', 'Unknown'),
        country=scraped_data.get('country', 'Unknown'),
    )
    
    # Extract league
    league = League(
        id=f"league_{scraped_data.get('league', '').lower().replace(' ', '_')}",
        name=scraped_data.get('league', 'Unknown League'),
        country=scraped_data.get('country', 'Unknown'),
    )
    
    # Parse status
    status_map = {
        'live': GameStatus.LIVE,
        'halftime': GameStatus.HALFTIME,
        'finished': GameStatus.FINISHED,
        'scheduled': GameStatus.SCHEDULED,
    }
    
    status = status_map.get(
        scraped_data.get('status', '').lower(),
        GameStatus.SCHEDULED
    )
    
    # Create Game object
    game = Game(
        id=scraped_data.get('id', 'unknown'),
        home_team=home_team,
        away_team=away_team,
        league=league,
        start_time=datetime.fromisoformat(scraped_data.get('timestamp', datetime.now().isoformat())),
        current_minute=scraped_data.get('minute', 0),
        status=status,
        home_score=scraped_data.get('home_score', 0),
        away_score=scraped_data.get('away_score', 0),
        data_source=scraped_data.get('source', 'unknown'),
    )
    
    return game


def create_prediction_id(game: Game) -> str:
    """Create unique prediction ID for a game"""
    import hashlib
    
    unique_string = f"{game.id}_{game.start_time.isoformat()}_{datetime.now().timestamp()}"
    return f"pred_{hashlib.md5(unique_string.encode()).hexdigest()[:12]}"


# Utility functions for probability calculations

def calculate_implied_probability(odds: float) -> float:
    """
    Calculate implied probability from decimal odds
    
    Args:
        odds: Decimal odds (e.g., 1.85)
        
    Returns:
        Implied probability (0.0 to 1.0)
    """
    if odds <= 1.0:
        return 1.0  # Invalid odds, default to certainty
    
    return 1.0 / odds


def calculate_fair_odds(probability: float, margin: float = 0.05) -> float:
    """
    Calculate fair odds with bookmaker margin
    
    Args:
        probability: True probability (0.0 to 1.0)
        margin: Bookmaker margin (default: 5%)
        
    Returns:
        Fair decimal odds
    """
    if probability <= 0.0:
        return 100.0  # Very high odds for impossible events
    
    fair_probability = probability * (1 - margin)
    return 1.0 / fair_probability


def round_probability(probability: float, decimals: int = 3) -> float:
    """Round probability to specified decimal places"""
    return float(Decimal(str(probability)).quantize(
        Decimal(f'1.{ "0" * decimals }'),
        rounding=ROUND_HALF_UP
    ))


# Type aliases for convenience
GameDict = Dict[str, Any]
PredictionDict = Dict[str, Any]
ProbabilityDict = Dict[str, float]
