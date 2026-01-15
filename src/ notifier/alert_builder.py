"""
alert_builder.py - Alert builder for Over/Under Predictor system
Intelligently builds, prioritizes, and manages alerts for predictions
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import json
import logging
from enum import Enum
from collections import defaultdict
import asyncio

# Import models and formatters
try:
    # First try absolute imports (when running from scripts/)
    from src.predictor.models import Prediction, BatchPrediction, Game, ProbabilityMetrics, RiskAssessment
    from src.predictor.formatter import PredictionFormatter, OutputFormat, AlertLevel
    from src.notifier.telegram_client import TelegramMessage
except ImportError:
    try:
        # Then try relative imports (when module is imported as part of package)
        from ..predictor.models import Prediction, BatchPrediction, Game, ProbabilityMetrics, RiskAssessment
        from ..predictor.formatter import PredictionFormatter, OutputFormat, AlertLevel
        from .telegram_client import TelegramMessage
    except ImportError:
        # Last resort: try direct imports (for testing)
        from predictor.models import Prediction, BatchPrediction, Game, ProbabilityMetrics, RiskAssessment
        from predictor.formatter import PredictionFormatter, OutputFormat, AlertLevel
        from telegram_client import TelegramMessage


class AlertType(Enum):
    """Types of alerts"""
    HIGH_PROBABILITY = "high_probability"
    VALUE_BET = "value_bet"
    MATCH_STARTING = "match_starting"
    MATCH_LIVE = "match_live"
    MATCH_ENDING = "match_ending"
    SYSTEM_ALERT = "system_alert"
    DAILY_SUMMARY = "daily_summary"
    PERFORMANCE_UPDATE = "performance_update"


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "critical"      # Immediate attention required
    HIGH = "high"              # Important alert
    MEDIUM = "medium"          # Regular alert  
    LOW = "low"                # Informational
    INFO = "info"              # Background information


@dataclass
class AlertRule:
    """Rule for triggering alerts"""
    name: str
    alert_type: AlertType
    conditions: List[Dict[str, Any]]
    priority: AlertPriority
    cooldown_minutes: int = 5
    max_alerts_per_day: int = 10
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.alert_type.value,
            "priority": self.priority.value,
            "cooldown_minutes": self.cooldown_minutes,
            "max_alerts_per_day": self.max_alerts_per_day,
            "enabled": self.enabled,
            "conditions": self.conditions
        }


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    alert_type: AlertType
    priority: AlertPriority
    prediction: Optional[Prediction] = None
    game: Optional[Game] = None
    title: str = ""
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    sent: bool = False
    sent_at: Optional[datetime] = None
    recipients: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.title:
            self.title = self._generate_title()
        
        if not self.message and self.prediction:
            self.message = self._generate_message()
        
        if not self.expires_at:
            self.expires_at = self.timestamp + timedelta(hours=1)
    
    def _generate_title(self) -> str:
        """Generate alert title based on type and priority"""
        if self.alert_type == AlertType.HIGH_PROBABILITY:
            return f"🚨 HIGH PROBABILITY ALERT"
        elif self.alert_type == AlertType.VALUE_BET:
            return f"💰 VALUE BET DETECTED"
        elif self.alert_type == AlertType.MATCH_STARTING:
            return f"⏰ MATCH STARTING SOON"
        elif self.alert_type == AlertType.MATCH_LIVE:
            return f"⚽ MATCH NOW LIVE"
        elif self.alert_type == AlertType.MATCH_ENDING:
            return f"⏱️ MATCH ENDING SOON"
        elif self.alert_type == AlertType.SYSTEM_ALERT:
            return f"🔧 SYSTEM ALERT"
        elif self.alert_type == AlertType.DAILY_SUMMARY:
            return f"📊 DAILY SUMMARY"
        elif self.alert_type == AlertType.PERFORMANCE_UPDATE:
            return f"📈 PERFORMANCE UPDATE"
        
        return f"📢 ALERT"
    
    def _generate_message(self) -> str:
        """Generate alert message from prediction"""
        if not self.prediction:
            return self.message
        
        game = self.prediction.game
        metrics = self.prediction.probability_metrics
        
        if self.alert_type == AlertType.HIGH_PROBABILITY:
            return (
                f"🎯 <b>High Probability Opportunity</b>\n\n"
                f"⚽ {game.home_team.name} vs {game.away_team.name}\n"
                f"📊 Score: {game.home_score}-{game.away_score} ({game.current_minute if game.is_live else 'Not started'}')\n"
                f"🎯 Over 2.5 Probability: <b>{metrics.probability_over_25:.1%}</b>\n"
                f"✅ Confidence: {metrics.confidence_score:.1%}\n"
                f"⚠️ Risk: {self.prediction.risk_assessment.risk_level.value.upper()}"
            )
        
        elif self.alert_type == AlertType.VALUE_BET:
            value_score = metrics.value_bet_score or 0
            return (
                f"💰 <b>Value Bet Detected!</b>\n\n"
                f"⚽ {game.home_team.name} vs {game.away_team.name}\n"
                f"📈 Our Probability: {metrics.probability_over_25:.1%}\n"
                f"🎰 Market Probability: {(1/game.odds_over_25):.1% if game.odds_over_25 else 'N/A'}\n"
                f"📊 Value Edge: <b>{value_score:.1%}</b>\n"
                f"✅ Confidence: {metrics.confidence_score:.1%}"
            )
        
        return self.message
    
    def is_expired(self) -> bool:
        """Check if alert has expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "id": self.id,
            "type": self.alert_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "sent": self.sent,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "recipients": self.recipients,
            "data": self.data,
            "metadata": self.metadata
        }
        
        if self.prediction:
            data["prediction_id"] = self.prediction.id
            data["game_id"] = self.prediction.game.id
        
        if self.game:
            data["game_id"] = self.game.id
        
        return data
    
    def to_telegram_message(self, chat_id: str) -> TelegramMessage:
        """Convert to Telegram message"""
        # Build full message
        full_message = f"{self.title}\n\n{self.message}"
        
        # Add metadata if available
        if self.metadata:
            metadata_str = "\n\n📊 Metadata:\n"
            for key, value in self.metadata.items():
                if key not in ['raw_data', 'internal']:
                    metadata_str += f"• {key}: {value}\n"
            full_message += metadata_str
        
        # Add footer
        footer = f"\n\n🕐 {self.timestamp.strftime('%H:%M:%S')}"
        if self.alert_type in [AlertType.HIGH_PROBABILITY, AlertType.VALUE_BET]:
            footer += f" | 📍 Alert ID: {self.id[:8]}"
        
        full_message += footer
        
        # Determine if notification should make sound
        disable_notification = self.priority in [AlertPriority.LOW, AlertPriority.INFO]
        
        return TelegramMessage(
            chat_id=chat_id,
            text=full_message,
            parse_mode="HTML",
            disable_web_page_preview=True,
            disable_notification=disable_notification
        )


@dataclass
class AlertStatistics:
    """Alert statistics"""
    total_alerts: int = 0
    alerts_sent: int = 0
    alerts_expired: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    last_alert_time: Optional[datetime] = None
    
    def update(self, alert: Alert, sent: bool = True):
        """Update statistics with new alert"""
        self.total_alerts += 1
        
        if sent:
            self.alerts_sent += 1
        else:
            self.alerts_expired += 1
        
        # Update type counts
        alert_type = alert.alert_type.value
        self.by_type[alert_type] = self.by_type.get(alert_type, 0) + 1
        
        # Update priority counts
        priority = alert.priority.value
        self.by_priority[priority] = self.by_priority.get(priority, 0) + 1
        
        # Update success rate
        if self.total_alerts > 0:
            self.success_rate = self.alerts_sent / self.total_alerts
        
        self.last_alert_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_alerts": self.total_alerts,
            "alerts_sent": self.alerts_sent,
            "alerts_expired": self.alerts_expired,
            "by_type": self.by_type,
            "by_priority": self.by_priority,
            "success_rate": self.success_rate,
            "last_alert_time": self.last_alert_time.isoformat() if self.last_alert_time else None
        }


class AlertBuilder:
    """
    Intelligent alert builder and manager
    Creates alerts based on predictions and manages alert lifecycle
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize alert builder
        
        Args:
            config: Builder configuration
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger("notifier.alert_builder")
        
        # Formatter for creating alert messages
        self.formatter = PredictionFormatter()
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000
        
        # Alert rules
        self.rules: List[AlertRule] = self._create_default_rules()
        
        # Statistics
        self.statistics = AlertStatistics()
        
        # Cooldown tracking
        self.cooldowns: Dict[Tuple[str, str], datetime] = {}  # (rule_name, game_id) -> timestamp
        
        # Daily limits tracking
        self.daily_counts: Dict[Tuple[str, str], int] = defaultdict(int)  # (rule_name, game_id) -> count
        self.last_daily_reset = datetime.now()
        
        # Recipient management
        self.recipients: Dict[str, Dict[str, Any]] = {}  # chat_id -> preferences
        
        self.logger.info("AlertBuilder initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'alert_thresholds': {
                'high_probability': 0.75,
                'value_bet_edge': 0.05,
                'match_starting_minutes': 15,
                'match_ending_minutes': 10,
            },
            'cooldowns': {
                'same_game_minutes': 10,
                'same_rule_minutes': 5,
            },
            'priorities': {
                'critical': ['system_down', 'high_value_bet'],
                'high': ['high_probability', 'match_ending'],
                'medium': ['value_bet', 'match_starting'],
                'low': ['match_live', 'performance_update'],
                'info': ['daily_summary']
            },
            'max_alerts_per_hour': 20,
            'max_alerts_per_day': 100,
            'alert_expiry_hours': 1,
            'enable_duplicate_filtering': True,
            'min_confidence_for_alert': 0.6,
            'min_prediction_priority': 50,
        }
    
    def _create_default_rules(self) -> List[AlertRule]:
        """Create default alert rules"""
        return [
            # High probability alerts
            AlertRule(
                name="high_probability_over_25",
                alert_type=AlertType.HIGH_PROBABILITY,
                conditions=[
                    {"field": "probability_metrics.probability_over_25", "operator": ">=", "value": 0.75},
                    {"field": "probability_metrics.confidence_score", "operator": ">=", "value": 0.7},
                    {"field": "risk_assessment.overall_risk_score", "operator": "<=", "value": 0.7},
                    {"field": "game.is_live", "operator": "==", "value": True},
                    {"field": "game.current_minute", "operator": "<=", "value": 80},
                ],
                priority=AlertPriority.HIGH,
                cooldown_minutes=10,
                max_alerts_per_day=5
            ),
            
            # Value bet alerts
            AlertRule(
                name="value_bet_detected",
                alert_type=AlertType.VALUE_BET,
                conditions=[
                    {"field": "probability_metrics.value_bet_score", "operator": ">=", "value": 0.05},
                    {"field": "probability_metrics.confidence_score", "operator": ">=", "value": 0.6},
                    {"field": "game.odds_over_25", "operator": "!=", "value": None},
                    {"field": "game.is_live", "operator": "==", "value": True},
                ],
                priority=AlertPriority.MEDIUM,
                cooldown_minutes=15,
                max_alerts_per_day=3
            ),
            
            # Match starting soon
            AlertRule(
                name="match_starting_soon",
                alert_type=AlertType.MATCH_STARTING,
                conditions=[
                    {"field": "game.status", "operator": "==", "value": "scheduled"},
                    {"field": "game.start_time", "operator": "within_minutes", "value": 15},
                    {"field": "probability_metrics.probability_over_25", "operator": ">=", "value": 0.65},
                ],
                priority=AlertPriority.MEDIUM,
                cooldown_minutes=30,
                max_alerts_per_day=2
            ),
            
            # Match now live
            AlertRule(
                name="match_now_live",
                alert_type=AlertType.MATCH_LIVE,
                conditions=[
                    {"field": "game.status", "operator": "in", "value": ["live", "halftime"]},
                    {"field": "game.current_minute", "operator": "<=", "value": 5},
                    {"field": "probability_metrics.probability_over_25", "operator": ">=", "value": 0.6},
                ],
                priority=AlertPriority.LOW,
                cooldown_minutes=60,
                max_alerts_per_day=1
            ),
            
            # Match ending soon (potential late goal opportunity)
            AlertRule(
                name="match_ending_soon",
                alert_type=AlertType.MATCH_ENDING,
                conditions=[
                    {"field": "game.is_live", "operator": "==", "value": True},
                    {"field": "game.current_minute", "operator": ">=", "value": 75},
                    {"field": "game.total_goals", "operator": "<=", "value": 2},
                    {"field": "probability_metrics.probability_over_25", "operator": ">=", "value": 0.4},
                ],
                priority=AlertPriority.HIGH,
                cooldown_minutes=5,
                max_alerts_per_day=3
            ),
        ]
    
    def build_from_prediction(self, prediction: Prediction) -> List[Alert]:
        """
        Build alerts from a prediction
        
        Args:
            prediction: Prediction object
            
        Returns:
            List of Alert objects
        """
        alerts = []
        
        # Check each rule
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check if rule conditions are met
            if self._check_rule_conditions(rule, prediction):
                
                # Check cooldown
                if not self._check_cooldown(rule, prediction):
                    continue
                
                # Check daily limit
                if not self._check_daily_limit(rule, prediction):
                    continue
                
                # Create alert
                alert = self._create_alert_from_rule(rule, prediction)
                if alert:
                    alerts.append(alert)
                    
                    # Update cooldown
                    self._update_cooldown(rule, prediction)
                    
                    # Update daily count
                    self._update_daily_count(rule, prediction)
        
        return alerts
    
    def build_from_batch(self, batch: BatchPrediction) -> List[Alert]:
        """
        Build alerts from a batch of predictions
        
        Args:
            batch: BatchPrediction object
            
        Returns:
            List of Alert objects
        """
        alerts = []
        
        for prediction in batch.predictions:
            prediction_alerts = self.build_from_prediction(prediction)
            alerts.extend(prediction_alerts)
        
        return alerts
    
    def _check_rule_conditions(self, rule: AlertRule, prediction: Prediction) -> bool:
        """
        Check if prediction meets rule conditions
        
        Args:
            rule: AlertRule to check
            prediction: Prediction to evaluate
            
        Returns:
            True if all conditions are met
        """
        for condition in rule.conditions:
            field_path = condition["field"]
            operator = condition["operator"]
            expected_value = condition["value"]
            
            # Get actual value from prediction
            actual_value = self._get_field_value(prediction, field_path)
            
            # Apply operator
            if not self._apply_operator(actual_value, operator, expected_value):
                return False
        
        return True
    
    def _get_field_value(self, prediction: Prediction, field_path: str) -> Any:
        """
        Get field value from prediction using dot notation
        
        Args:
            prediction: Prediction object
            field_path: Dot notation path to field
            
        Returns:
            Field value
        """
        parts = field_path.split('.')
        current = prediction
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Try to access via get method
                if hasattr(current, 'get'):
                    current = current.get(part)
                else:
                    return None
        
        return current
    
    def _apply_operator(self, actual: Any, operator: str, expected: Any) -> bool:
        """
        Apply comparison operator
        
        Args:
            actual: Actual value
            operator: Comparison operator
            expected: Expected value
            
        Returns:
            True if condition is met
        """
        if operator == "==":
            return actual == expected
        elif operator == "!=":
            return actual != expected
        elif operator == ">":
            return actual > expected
        elif operator == ">=":
            return actual >= expected
        elif operator == "<":
            return actual < expected
        elif operator == "<=":
            return actual <= expected
        elif operator == "in":
            return actual in expected if isinstance(expected, list) else False
        elif operator == "not in":
            return actual not in expected if isinstance(expected, list) else True
        elif operator == "within_minutes":
            if isinstance(actual, datetime) and isinstance(expected, (int, float)):
                time_diff = (actual - datetime.now()).total_seconds() / 60
                return 0 <= time_diff <= expected
            return False
        elif operator == "contains":
            return expected in str(actual)
        else:
            self.logger.warning(f"Unknown operator: {operator}")
            return False
    
    def _check_cooldown(self, rule: AlertRule, prediction: Prediction) -> bool:
        """
        Check if cooldown period has passed for this rule and game
        
        Args:
            rule: AlertRule
            prediction: Prediction
            
        Returns:
            True if cooldown has passed
        """
        game_id = prediction.game.id
        key = (rule.name, game_id)
        
        if key in self.cooldowns:
            last_alert_time = self.cooldowns[key]
            cooldown_end = last_alert_time + timedelta(minutes=rule.cooldown_minutes)
            
            if datetime.now() < cooldown_end:
                self.logger.debug(f"Cooldown active for {rule.name} on game {game_id}")
                return False
        
        return True
    
    def _check_daily_limit(self, rule: AlertRule, prediction: Prediction) -> bool:
        """
        Check daily alert limit for this rule and game
        
        Args:
            rule: AlertRule
            prediction: Prediction
            
        Returns:
            True if under daily limit
        """
        # Reset daily counts if it's a new day
        self._reset_daily_counts_if_needed()
        
        game_id = prediction.game.id
        key = (rule.name, game_id)
        
        if self.daily_counts[key] >= rule.max_alerts_per_day:
            self.logger.debug(f"Daily limit reached for {rule.name} on game {game_id}")
            return False
        
        return True
    
    def _update_cooldown(self, rule: AlertRule, prediction: Prediction):
        """Update cooldown timestamp for rule and game"""
        game_id = prediction.game.id
        key = (rule.name, game_id)
        self.cooldowns[key] = datetime.now()
    
    def _update_daily_count(self, rule: AlertRule, prediction: Prediction):
        """Update daily alert count for rule and game"""
        game_id = prediction.game.id
        key = (rule.name, game_id)
        self.daily_counts[key] += 1
    
    def _reset_daily_counts_if_needed(self):
        """Reset daily counts if it's a new day"""
        now = datetime.now()
        if now.date() > self.last_daily_reset.date():
            self.daily_counts.clear()
            self.last_daily_reset = now
            self.logger.info("Daily alert counts reset")
    
    def _create_alert_from_rule(self, rule: AlertRule, prediction: Prediction) -> Optional[Alert]:
        """
        Create alert from rule and prediction
        
        Args:
            rule: AlertRule
            prediction: Prediction
            
        Returns:
            Alert object or None
        """
        try:
            # Generate alert ID
            alert_id = f"alert_{prediction.id}_{rule.name}_{datetime.now().strftime('%H%M%S')}"
            
            # Create alert data
            alert_data = {
                "probability_over_25": prediction.probability_metrics.probability_over_25,
                "confidence": prediction.probability_metrics.confidence_score,
                "risk_score": prediction.risk_assessment.overall_risk_score,
                "game_status": prediction.game.status.value,
                "current_minute": prediction.game.current_minute,
                "total_goals": prediction.game.total_goals,
            }
            
            # Add value bet data if applicable
            if rule.alert_type == AlertType.VALUE_BET:
                value_score = prediction.probability_metrics.value_bet_score
                alert_data["value_bet_score"] = value_score
                alert_data["market_odds"] = prediction.game.odds_over_25
            
            # Create alert
            alert = Alert(
                id=alert_id,
                alert_type=rule.alert_type,
                priority=rule.priority,
                prediction=prediction,
                game=prediction.game,
                data=alert_data,
                metadata={
                    "rule_name": rule.name,
                    "prediction_id": prediction.id,
                    "game_id": prediction.game.id,
                    "alert_priority": prediction.alert_priority,
                }
            )
            
            # Set expiry based on alert type
            if rule.alert_type in [AlertType.MATCH_STARTING, AlertType.MATCH_LIVE, AlertType.MATCH_ENDING]:
                # Short expiry for match-related alerts
                alert.expires_at = datetime.now() + timedelta(minutes=30)
            elif rule.alert_type == AlertType.HIGH_PROBABILITY:
                # Medium expiry for probability alerts
                alert.expires_at = datetime.now() + timedelta(minutes=45)
            else:
                # Default expiry
                alert.expires_at = datetime.now() + timedelta(hours=1)
            
            self.logger.info(f"Created {rule.alert_type.value} alert for {prediction.game.home_team.name} vs {prediction.game.away_team.name}")
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error creating alert from rule {rule.name}: {e}")
            return None
    
    def create_system_alert(self, title: str, message: str, priority: AlertPriority = AlertPriority.MEDIUM, 
                          data: Optional[Dict] = None) -> Alert:
        """
        Create a system alert (not prediction-based)
        
        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority
            data: Additional data
            
        Returns:
            Alert object
        """
        alert_id = f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = Alert(
            id=alert_id,
            alert_type=AlertType.SYSTEM_ALERT,
            priority=priority,
            title=title,
            message=message,
            data=data or {},
            metadata={
                "source": "system",
                "created_at": datetime.now().isoformat()
            }
        )
        
        return alert
    
    def create_daily_summary_alert(self, summary_data: Dict[str, Any]) -> Alert:
        """
        Create daily summary alert
        
        Args:
            summary_data: Summary data dictionary
            
        Returns:
            Alert object
        """
        # Format summary message
        total_predictions = summary_data.get('total_predictions', 0)
        high_confidence = summary_data.get('high_confidence_predictions', 0)
        value_bets = summary_data.get('value_bet_opportunities', 0)
        success_rate = summary_data.get('success_rate_estimate', 0)
        
        message = (
            f"📊 <b>Daily Summary Report</b>\n\n"
            f"📈 <b>Performance Statistics:</b>\n"
            f"• Total Predictions: {total_predictions}\n"
            f"• High Confidence: {high_confidence}\n"
            f"• Value Bets Detected: {value_bets}\n"
            f"• Estimated Success Rate: {success_rate:.1%}\n\n"
        )
        
        # Add top performing leagues if available
        if 'top_leagues' in summary_data:
            message += f"🏆 <b>Top Performing Leagues:</b>\n"
            for league in summary_data['top_leagues'][:3]:
                message += f"• {league.get('name')}: {league.get('success_rate', 0):.1%}\n"
            message += "\n"
        
        # Add alert statistics
        message += f"🔔 <b>Alert Statistics:</b>\n"
        message += f"• Total Alerts Sent: {self.statistics.total_alerts}\n"
        message += f"• Alert Success Rate: {self.statistics.success_rate:.1%}\n\n"
        
        message += f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        message += f"🔄 Next update tomorrow"
        
        alert = Alert(
            id=f"daily_summary_{datetime.now().strftime('%Y%m%d')}",
            alert_type=AlertType.DAILY_SUMMARY,
            priority=AlertPriority.INFO,
            title="📊 Daily Summary Report",
            message=message,
            data=summary_data,
            metadata={
                "report_date": datetime.now().date().isoformat(),
                "report_type": "daily_summary"
            }
        )
        
        return alert
    
    def create_performance_update_alert(self, performance_data: Dict[str, Any]) -> Alert:
        """
        Create performance update alert
        
        Args:
            performance_data: Performance data
            
        Returns:
            Alert object
        """
        message = (
            f"📈 <b>Performance Update</b>\n\n"
            f"🔄 <b>System Status:</b>\n"
            f"• Uptime: {performance_data.get('uptime', 'N/A')}\n"
            f"• Predictions/hour: {performance_data.get('predictions_per_hour', 0)}\n"
            f"• Success Rate: {performance_data.get('success_rate', 0):.1%}\n\n"
        )
        
        if 'recent_alerts' in performance_data:
            message += f"🔔 <b>Recent Alerts:</b>\n"
            for alert in performance_data['recent_alerts'][:3]:
                message += f"• {alert.get('type')}: {alert.get('count', 0)} sent\n"
        
        alert = Alert(
            id=f"performance_{datetime.now().strftime('%Y%m%d_%H%M')}",
            alert_type=AlertType.PERFORMANCE_UPDATE,
            priority=AlertPriority.LOW,
            title="📈 Performance Update",
            message=message,
            data=performance_data
        )
        
        return alert
    
    def add_alert(self, alert: Alert):
        """
        Add alert to active alerts
        
        Args:
            alert: Alert to add
        """
        self.active_alerts[alert.id] = alert
        self.logger.debug(f"Added alert {alert.id} to active alerts")
    
    def remove_alert(self, alert_id: str):
        """
        Remove alert from active alerts
        
        Args:
            alert_id: ID of alert to remove
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            self.alert_history.append(alert)
            
            # Trim history if needed
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size:]
            
            self.logger.debug(f"Removed alert {alert_id} from active alerts")
    
    def mark_alert_sent(self, alert_id: str, recipients: List[str]):
        """
        Mark alert as sent
        
        Args:
            alert_id: ID of alert
            recipients: List of recipient chat IDs
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.sent = True
            alert.sent_at = datetime.now()
            alert.recipients = recipients
            
            # Update statistics
            self.statistics.update(alert, sent=True)
            
            self.logger.info(f"Marked alert {alert_id} as sent to {len(recipients)} recipients")
    
    def mark_alert_expired(self, alert_id: str):
        """
        Mark alert as expired
        
        Args:
            alert_id: ID of alert
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.expires_at = datetime.now()  # Force expiry
            
            # Update statistics
            self.statistics.update(alert, sent=False)
            
            self.logger.debug(f"Marked alert {alert_id} as expired")
    
    def get_active_alerts(self, alert_type: Optional[AlertType] = None, 
                         priority: Optional[AlertPriority] = None) -> List[Alert]:
        """
        Get active alerts filtered by type and/or priority
        
        Args:
            alert_type: Filter by alert type
            priority: Filter by priority
            
        Returns:
            List of filtered alerts
        """
        alerts = list(self.active_alerts.values())
        
        # Filter by type
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        # Filter by priority
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        # Sort by priority (critical first) and timestamp (newest first)
        priority_order = {AlertPriority.CRITICAL: 0, AlertPriority.HIGH: 1, 
                         AlertPriority.MEDIUM: 2, AlertPriority.LOW: 3, AlertPriority.INFO: 4}
        
        alerts.sort(key=lambda a: (priority_order[a.priority], a.timestamp), reverse=True)
        
        return alerts
    
    def get_expired_alerts(self) -> List[Alert]:
        """Get expired alerts that are still active"""
        now = datetime.now()
        expired = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.is_expired():
                expired.append(alert)
        
        return expired
    
    def cleanup_expired_alerts(self):
        """Remove expired alerts from active alerts"""
        expired_ids = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.is_expired():
                expired_ids.append(alert_id)
        
        for alert_id in expired_ids:
            self.remove_alert(alert_id)
        
        if expired_ids:
            self.logger.info(f"Cleaned up {len(expired_ids)} expired alerts")
    
    def add_recipient(self, chat_id: str, preferences: Optional[Dict] = None):
        """
        Add recipient with preferences
        
        Args:
            chat_id: Recipient chat ID
            preferences: Alert preferences
        """
        default_preferences = {
            "receive_high_probability": True,
            "receive_value_bets": True,
            "receive_match_starting": True,
            "receive_match_live": False,
            "receive_match_ending": True,
            "receive_system_alerts": False,
            "receive_daily_summary": True,
            "receive_performance_updates": False,
            "min_priority": "medium",  # 'critical', 'high', 'medium', 'low', 'info'
            "quiet_hours_start": 23,  # 11 PM
            "quiet_hours_end": 8,     # 8 AM
        }
        
        if preferences:
            default_preferences.update(preferences)
        
        self.recipients[chat_id] = default_preferences
        self.logger.info(f"Added recipient {chat_id}")
    
    def remove_recipient(self, chat_id: str):
        """
        Remove recipient
        
        Args:
            chat_id: Recipient chat ID to remove
        """
        if chat_id in self.recipients:
            del self.recipients[chat_id]
            self.logger.info(f"Removed recipient {chat_id}")
    
    def should_send_to_recipient(self, alert: Alert, chat_id: str) -> bool:
        """
        Check if alert should be sent to recipient based on preferences
        
        Args:
            alert: Alert to check
            chat_id: Recipient chat ID
            
        Returns:
            True if should send
        """
        if chat_id not in self.recipients:
            return False
        
        preferences = self.recipients[chat_id]
        
        # Check quiet hours
        if self._in_quiet_hours(preferences):
            self.logger.debug(f"Quiet hours for {chat_id}, skipping alert")
            return False
        
        # Check minimum priority
        min_priority = preferences.get("min_priority", "medium")
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        
        if priority_order.get(alert.priority.value, 5) > priority_order.get(min_priority, 2):
            return False
        
        # Check alert type preferences
        alert_type_pref = f"receive_{alert.alert_type.value}"
        if not preferences.get(alert_type_pref, True):
            return False
        
        return True
    
    def _in_quiet_hours(self, preferences: Dict) -> bool:
        """
        Check if current time is within recipient's quiet hours
        
        Args:
            preferences: Recipient preferences
            
        Returns:
            True if in quiet hours
        """
        quiet_start = preferences.get("quiet_hours_start", 23)
        quiet_end = preferences.get("quiet_hours_end", 8)
        
        current_hour = datetime.now().hour
        
        if quiet_end > quiet_start:
            # Quiet hours don't cross midnight
            return quiet_start <= current_hour < quiet_end
        else:
            # Quiet hours cross midnight
            return current_hour >= quiet_start or current_hour < quiet_end
    
    def filter_alerts_for_recipient(self, alerts: List[Alert], chat_id: str) -> List[Alert]:
        """
        Filter alerts for specific recipient based on preferences
        
        Args:
            alerts: List of alerts to filter
            chat_id: Recipient chat ID
            
        Returns:
            Filtered list of alerts
        """
        if chat_id not in self.recipients:
            return []
        
        return [alert for alert in alerts if self.should_send_to_recipient(alert, chat_id)]
    
    def get_recipients_for_alert(self, alert: Alert) -> List[str]:
        """
        Get recipients who should receive this alert
        
        Args:
            alert: Alert to check
            
        Returns:
            List of recipient chat IDs
        """
        recipients = []
        
        for chat_id in self.recipients.keys():
            if self.should_send_to_recipient(alert, chat_id):
                recipients.append(chat_id)
        
        return recipients
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics
        
        Returns:
            Statistics dictionary
        """
        stats = self.statistics.to_dict()
        
        # Add additional stats
        stats.update({
            "active_alerts": len(self.active_alerts),
            "alert_history": len(self.alert_history),
            "recipients": len(self.recipients),
            "rules_enabled": len([r for r in self.rules if r.enabled]),
            "rules_total": len(self.rules),
            "daily_counts": dict(self.daily_counts),
            "cooldowns_active": len(self.cooldowns),
        })
        
        return stats
    
    def save_state(self, filepath: str):
        """
        Save alert builder state to file
        
        Args:
            filepath: Path to save file
        """
        try:
            state = {
                "active_alerts": {aid: alert.to_dict() for aid, alert in self.active_alerts.items()},
                "alert_history": [alert.to_dict() for alert in self.alert_history[-100:]],  # Last 100
                "statistics": self.statistics.to_dict(),
                "recipients": self.recipients,
                "cooldowns": {str(k): v.isoformat() for k, v in self.cooldowns.items()},
                "daily_counts": {str(k): v for k, v in self.daily_counts.items()},
                "last_daily_reset": self.last_daily_reset.isoformat(),
                "rules": [rule.to_dict() for rule in self.rules],
                "saved_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Saved state to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def load_state(self, filepath: str):
        """
        Load alert builder state from file
        
        Args:
            filepath: Path to load file from
        """
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"State file not found: {filepath}")
                return
            
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load active alerts
            self.active_alerts.clear()
            for aid, alert_data in state.get("active_alerts", {}).items():
                try:
                    alert = self._dict_to_alert(alert_data)
                    self.active_alerts[aid] = alert
                except Exception as e:
                    self.logger.warning(f"Error loading alert {aid}: {e}")
            
            # Load alert history
            self.alert_history.clear()
            for alert_data in state.get("alert_history", []):
                try:
                    alert = self._dict_to_alert(alert_data)
                    self.alert_history.append(alert)
                except Exception as e:
                    self.logger.warning(f"Error loading historical alert: {e}")
            
            # Load statistics
            stats_data = state.get("statistics", {})
            self.statistics = AlertStatistics(
                total_alerts=stats_data.get("total_alerts", 0),
                alerts_sent=stats_data.get("alerts_sent", 0),
                alerts_expired=stats_data.get("alerts_expired", 0),
                by_type=stats_data.get("by_type", {}),
                by_priority=stats_data.get("by_priority", {}),
                success_rate=stats_data.get("success_rate", 0.0)
            )
            if stats_data.get("last_alert_time"):
                self.statistics.last_alert_time = datetime.fromisoformat(stats_data["last_alert_time"])
            
            # Load recipients
            self.recipients = state.get("recipients", {})
            
            # Load cooldowns
            self.cooldowns.clear()
            for key_str, timestamp_str in state.get("cooldowns", {}).items():
                try:
                    key = eval(key_str)  # Convert string back to tuple
                    timestamp = datetime.fromisoformat(timestamp_str)
                    self.cooldowns[key] = timestamp
                except Exception as e:
                    self.logger.warning(f"Error loading cooldown {key_str}: {e}")
            
            # Load daily counts
            self.daily_counts.clear()
            for key_str, count in state.get("daily_counts", {}).items():
                try:
                    key = eval(key_str)  # Convert string back to tuple
                    self.daily_counts[key] = count
                except Exception as e:
                    self.logger.warning(f"Error loading daily count {key_str}: {e}")
            
            # Load last daily reset
            if state.get("last_daily_reset"):
                self.last_daily_reset = datetime.fromisoformat(state["last_daily_reset"])
            
            # Load rules (optional - only if not already set)
            if not self.rules and "rules" in state:
                self.rules = []
                for rule_data in state["rules"]:
                    try:
                        rule = AlertRule(
                            name=rule_data["name"],
                            alert_type=AlertType(rule_data["type"]),
                            conditions=rule_data["conditions"],
                            priority=AlertPriority(rule_data["priority"]),
                            cooldown_minutes=rule_data.get("cooldown_minutes", 5),
                            max_alerts_per_day=rule_data.get("max_alerts_per_day", 10),
                            enabled=rule_data.get("enabled", True)
                        )
                        self.rules.append(rule)
                    except Exception as e:
                        self.logger.warning(f"Error loading rule: {e}")
            
            self.logger.info(f"Loaded state from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
    
    def _dict_to_alert(self, alert_data: Dict) -> Alert:
        """Convert dictionary to Alert object"""
        # Note: This is a simplified version
        # In production, you'd need to reconstruct prediction objects too
        
        alert = Alert(
            id=alert_data["id"],
            alert_type=AlertType(alert_data["type"]),
            priority=AlertPriority(alert_data["priority"]),
            title=alert_data["title"],
            message=alert_data["message"],
            data=alert_data["data"],
            timestamp=datetime.fromisoformat(alert_data["timestamp"]),
            sent=alert_data["sent"],
            recipients=alert_data["recipients"],
            metadata=alert_data["metadata"]
        )
        
        if alert_data.get("expires_at"):
            alert.expires_at = datetime.fromisoformat(alert_data["expires_at"])
        
        if alert_data.get("sent_at"):
            alert.sent_at = datetime.fromisoformat(alert_data["sent_at"])
        
        return alert
    
    def add_custom_rule(self, rule: AlertRule):
        """
        Add custom alert rule
        
        Args:
            rule: AlertRule to add
        """
        self.rules.append(rule)
        self.logger.info(f"Added custom rule: {rule.name}")
    
    def enable_rule(self, rule_name: str):
        """Enable alert rule by name"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                self.logger.info(f"Enabled rule: {rule_name}")
                return
        
        self.logger.warning(f"Rule not found: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """Disable alert rule by name"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                self.logger.info(f"Disabled rule: {rule_name}")
                return
        
        self.logger.warning(f"Rule not found: {rule_name}")


# Factory function

def create_alert_builder(config: Optional[Dict] = None) -> AlertBuilder:
    """
    Create AlertBuilder instance
    
    Args:
        config: Optional configuration
        
    Returns:
        AlertBuilder instance
    """
    return AlertBuilder(config)


# Example usage

async def main():
    """Test the alert builder"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing AlertBuilder...")
    print("=" * 60)
    
    # Create alert builder
    builder = create_alert_builder()
    
    # Create a sample prediction for testing
    from datetime import datetime
    
    # Note: In reality, you'd import your actual models
    # This is just for testing
    
    class MockGame:
        id = "game_123"
        home_team = type('obj', (object,), {'name': 'Manchester City'})()
        away_team = type('obj', (object,), {'name': 'Liverpool FC'})()
        status = "live"
        is_live = True
        current_minute = 65
        home_score = 2
        away_score = 1
        total_goals = 3
        start_time = datetime.now()
        odds_over_25 = 1.85
    
    class MockProbabilityMetrics:
        probability_over_25 = 0.78
        confidence_score = 0.82
        value_bet_score = 0.07
    
    class MockRiskAssessment:
        overall_risk_score = 0.45
        risk_level = type('obj', (object,), {'value': 'medium'})()
    
    class MockPrediction:
        id = "pred_123"
        game = MockGame()
        probability_metrics = MockProbabilityMetrics()
        risk_assessment = MockRiskAssessment()
        alert_priority = 85
    
    prediction = MockPrediction()
    
    # Build alerts from prediction
    print("\n1. Building alerts from prediction...")
    alerts = builder.build_from_prediction(prediction)
    
    print(f"   Generated {len(alerts)} alerts")
    
    for i, alert in enumerate(alerts, 1):
        print(f"   {i}. {alert.alert_type.value} - {alert.priority.value}")
        print(f"      Title: {alert.title}")
        print(f"      Expires: {alert.expires_at.strftime('%H:%M:%S')}")
    
    # Create system alert
    print("\n2. Creating system alert...")
    system_alert = builder.create_system_alert(
        title="🔧 System Maintenance",
        message="Scheduled maintenance in 30 minutes. System may be briefly unavailable.",
        priority=AlertPriority.MEDIUM,
        data={"maintenance_window": "30 minutes", "impact": "minimal"}
    )
    
    builder.add_alert(system_alert)
    print(f"   Created system alert: {system_alert.title}")
    
    # Create daily summary
    print("\n3. Creating daily summary...")
    summary_data = {
        "total_predictions": 42,
        "high_confidence_predictions": 18,
        "value_bet_opportunities": 7,
        "success_rate_estimate": 0.68,
        "top_leagues": [
            {"name": "Premier League", "success_rate": 0.72},
            {"name": "La Liga", "success_rate": 0.65},
            {"name": "Bundesliga", "success_rate": 0.61},
        ]
    }
    
    daily_alert = builder.create_daily_summary_alert(summary_data)
    builder.add_alert(daily_alert)
    print(f"   Created daily summary alert")
    
    # Add recipient
    print("\n4. Adding test recipient...")
    builder.add_recipient("test_chat_123", {
        "receive_high_probability": True,
        "receive_value_bets": True,
        "receive_system_alerts": True,
        "min_priority": "medium"
    })
    
    # Check which alerts should be sent to recipient
    active_alerts = builder.get_active_alerts()
    filtered = builder.filter_alerts_for_recipient(active_alerts, "test_chat_123")
    
    print(f"   Recipient would receive {len(filtered)} of {len(active_alerts)} active alerts")
    
    # Get statistics
    print("\n5. Getting statistics...")
    stats = builder.get_alert_statistics()
    print(f"   Active alerts: {stats['active_alerts']}")
    print(f"   Total alerts sent: {stats['total_alerts']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Recipients: {stats['recipients']}")
    
    # Cleanup expired alerts
    print("\n6. Cleaning up expired alerts...")
    builder.cleanup_expired_alerts()
    
    print("\n" + "=" * 60)
    print("AlertBuilder test completed!")
    print("=" * 60)


if __name__ == "__main__":
    import os
    import asyncio
    
    # Run test
    asyncio.run(main())
