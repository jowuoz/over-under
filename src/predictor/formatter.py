"""
formatter.py - Formatters for Over/Under Predictor system
Handles formatting of predictions, alerts, reports, and data for various outputs
"""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import logging
from enum import Enum
import html

# Import models
try:
    from .models import (
        Prediction, Game, ProbabilityMetrics, RiskAssessment,
        BatchPrediction, GameStatus, PredictionConfidence, RiskLevel,
        Team, League
    )
except ImportError:
    from src.predictor.models import (
        Prediction, Game, ProbabilityMetrics, RiskAssessment,
        BatchPrediction, GameStatus, PredictionConfidence, RiskLevel,
        Team, League
    )


class OutputFormat(Enum):
    """Output formats for predictions"""
    TELEGRAM = "telegram"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    CONSOLE = "console"
    EMAIL = "email"
    DISCORD = "discord"
    SLACK = "slack"


class AlertLevel(Enum):
    """Alert levels for notifications"""
    HIGH = "high"      # High probability, high confidence
    MEDIUM = "medium"  # Good opportunity
    LOW = "low"        # Informational
    INFO = "info"      # General information


class PredictionFormatter:
    """
    Formats predictions for various output channels
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize formatter with configuration
        
        Args:
            config: Formatter configuration
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger("predictor.formatter")
        
        # Emoji mappings
        self.emoji = {
            'live': '⚽',
            'finished': '✅',
            'scheduled': '🕐',
            'halftime': '⏸️',
            'high_confidence': '🎯',
            'medium_confidence': '📊',
            'low_confidence': '📉',
            'risk_low': '🟢',
            'risk_medium': '🟡',
            'risk_high': '🔴',
            'value_bet': '💰',
            'warning': '⚠️',
            'success': '✅',
            'error': '❌',
            'clock': '⏰',
            'trophy': '🏆',
            'fire': '🔥',
            'chart': '📈',
            'money': '💵',
            'star': '⭐',
            'bell': '🔔',
        }
        
        # Color codes (for HTML/console)
        self.colors = {
            'success': '#10B981',  # Green
            'warning': '#F59E0B',  # Yellow
            'error': '#EF4444',    # Red
            'info': '#3B82F6',     # Blue
            'high': '#DC2626',     # Dark Red
            'medium': '#F97316',   # Orange
            'low': '#84CC16',      # Light Green
            'text': '#1F2937',     # Dark Gray
            'muted': '#6B7280',    # Gray
        }
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_predictions_per_alert': 5,
            'truncate_team_names': 20,
            'include_risk_assessment': True,
            'include_key_factors': True,
            'probability_thresholds': {
                'very_high': 0.85,
                'high': 0.75,
                'medium': 0.60,
                'low': 0.40,
            },
            'alert_triggers': {
                'min_confidence': 0.70,
                'min_probability': 0.65,
                'max_risk': 0.70,
            }
        }
    
    def format_prediction(
        self, 
        prediction: Prediction, 
        format_type: OutputFormat = OutputFormat.TELEGRAM,
        alert_level: Optional[AlertLevel] = None
    ) -> str:
        """
        Format a single prediction for the specified output
        
        Args:
            prediction: Prediction object to format
            format_type: Output format
            alert_level: Alert level (for notifications)
            
        Returns:
            Formatted string
        """
        if alert_level is None:
            alert_level = self._determine_alert_level(prediction)
        
        formatter_map = {
            OutputFormat.TELEGRAM: self._format_for_telegram,
            OutputFormat.HTML: self._format_for_html,
            OutputFormat.JSON: self._format_for_json,
            OutputFormat.CSV: self._format_for_csv,
            OutputFormat.CONSOLE: self._format_for_console,
            OutputFormat.EMAIL: self._format_for_email,
            OutputFormat.DISCORD: self._format_for_discord,
            OutputFormat.SLACK: self._format_for_slack,
        }
        
        formatter = formatter_map.get(format_type, self._format_for_telegram)
        return formatter(prediction, alert_level)
    
    def format_batch(
        self, 
        batch: BatchPrediction, 
        format_type: OutputFormat = OutputFormat.TELEGRAM,
        include_all: bool = False
    ) -> str:
        """
        Format a batch of predictions
        
        Args:
            batch: BatchPrediction object
            format_type: Output format
            include_all: Include all predictions or just high priority
            
        Returns:
            Formatted string
        """
        if format_type == OutputFormat.JSON:
            return batch.to_json()
        
        if include_all:
            predictions = batch.predictions
        else:
            predictions = batch.get_high_priority_predictions(
                self.config['max_predictions_per_alert']
            )
        
        if not predictions:
            return self._format_no_predictions(format_type)
        
        formatter_map = {
            OutputFormat.TELEGRAM: self._format_batch_for_telegram,
            OutputFormat.HTML: self._format_batch_for_html,
            OutputFormat.CONSOLE: self._format_batch_for_console,
            OutputFormat.EMAIL: self._format_batch_for_email,
            OutputFormat.DISCORD: self._format_batch_for_discord,
            OutputFormat.SLACK: self._format_batch_for_slack,
        }
        
        formatter = formatter_map.get(format_type, self._format_batch_for_telegram)
        return formatter(batch, predictions)
    
    def _determine_alert_level(self, prediction: Prediction) -> AlertLevel:
        """Determine alert level based on prediction characteristics"""
        metrics = prediction.probability_metrics
        risk = prediction.risk_assessment
        
        # High alert: High probability + high confidence + acceptable risk
        if (metrics.probability_over_25 >= 0.75 and 
            metrics.confidence_level in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH] and
            risk.overall_risk_score <= 0.6):
            return AlertLevel.HIGH
        
        # Medium alert: Good probability + medium confidence
        if (metrics.probability_over_25 >= 0.65 and 
            metrics.confidence_level in [PredictionConfidence.MEDIUM, PredictionConfidence.HIGH]):
            return AlertLevel.MEDIUM
        
        # Low alert: Everything else worth noting
        return AlertLevel.LOW
    
    # Telegram Formatters
    
    def _format_for_telegram(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction for Telegram"""
        game = prediction.game
        metrics = prediction.probability_metrics
        risk = prediction.risk_assessment
        
        # Header based on alert level
        if alert_level == AlertLevel.HIGH:
            header = f"🚨 {self.emoji['fire']} HIGH CONFIDENCE ALERT {self.emoji['fire']}\n\n"
        elif alert_level == AlertLevel.MEDIUM:
            header = f"📢 {self.emoji['bell']} GOOD OPPORTUNITY {self.emoji['bell']}\n\n"
        else:
            header = f"📋 {self.emoji['chart']} PREDICTION UPDATE {self.emoji['chart']}\n\n"
        
        # Match information
        status_emoji = self._get_status_emoji(game.status)
        match_info = (
            f"{status_emoji} <b>{self._truncate(game.home_team.name)} vs {self._truncate(game.away_team.name)}</b>\n"
            f"🏆 {game.league.name}\n"
        )
        
        if game.is_live:
            match_info += f"📊 {game.home_score}-{game.away_score} ({game.current_minute}')\n"
        else:
            match_info += f"🕐 Starts: {game.start_time.strftime('%H:%M')}\n"
        
        # Probability information
        over_25_prob = metrics.probability_over_25
        confidence_emoji = self._get_confidence_emoji(metrics.confidence_level)
        risk_emoji = self._get_risk_emoji(risk.risk_level)
        
        prob_info = (
            f"\n{self.emoji['chart']} <b>PREDICTION ANALYSIS</b>\n"
            f"🎯 Over 2.5 Goals: <b>{over_25_prob:.1%}</b>\n"
            f"{confidence_emoji} Confidence: {metrics.confidence_level.value.replace('_', ' ').title()}\n"
            f"{risk_emoji} Risk Level: {risk.risk_level.value.replace('_', ' ').title()}\n"
        )
        
        # Additional probabilities
        prob_details = (
            f"📈 Over 1.5: {metrics.probability_over_15:.1%} | "
            f"Over 3.5: {metrics.probability_over_35:.1%}\n"
        )
        
        # Expected goals
        exp_goals = (
            f"🔮 Expected Total Goals: <b>{metrics.expected_total_goals:.2f}</b>\n"
        )
        
        # Key factors (if any)
        factors = ""
        if prediction.key_factors and self.config['include_key_factors']:
            factors = "\n🔑 <b>Key Factors:</b>\n"
            for i, factor in enumerate(prediction.key_factors[:3], 1):
                factors += f"{i}. {factor}\n"
        
        # Recommendation
        recommendation = ""
        if prediction.recommended_action:
            if prediction.is_value_bet:
                recommendation = f"\n{self.emoji['value_bet']} <b>VALUE BET DETECTED!</b>\n"
            recommendation += f"💡 <b>Recommendation:</b> {prediction.recommended_action}\n"
            
            if risk.recommended_stake:
                recommendation += f"💰 Suggested Stake: {risk.recommended_stake:.2%} of bankroll\n"
        
        # Warnings (if any)
        warnings = ""
        if prediction.warnings:
            warnings = "\n⚠️  <b>Warnings:</b>\n"
            for warning in prediction.warnings[:2]:
                warnings += f"• {warning}\n"
        
        # Footer
        footer = f"\n🔄 Updated: {prediction.timestamp.strftime('%H:%M:%S')}"
        footer += f" | 📍 ID: {prediction.id[:8]}"
        
        # Combine all parts
        message = header + match_info + prob_info + prob_details + exp_goals
        
        if factors:
            message += factors
        
        if recommendation:
            message += recommendation
        
        if warnings:
            message += warnings
        
        message += footer
        
        # Add hashtags
        hashtags = self._generate_hashtags(game, metrics)
        message += f"\n\n{hashtags}"
        
        return message
    
    def _format_batch_for_telegram(self, batch: BatchPrediction, predictions: List[Prediction]) -> str:
        """Format batch for Telegram"""
        if not predictions:
            return self._format_no_predictions(OutputFormat.TELEGRAM)
        
        # Header with batch statistics
        header = (
            f"📊 <b>PREDICTION BATCH UPDATE</b>\n"
            f"🕐 {batch.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
            f"🎯 {len(predictions)} High Priority Predictions\n"
            f"⭐ {batch.high_confidence_predictions} High Confidence\n"
            f"💰 {batch.value_bet_opportunities} Value Bets\n"
            f"📈 Estimated Success Rate: {batch.success_rate_estimate:.1%}\n\n"
            f"{'='*30}\n\n"
        )
        
        # Format each prediction
        prediction_texts = []
        for i, prediction in enumerate(predictions, 1):
            game = prediction.game
            metrics = prediction.probability_metrics
            
            # Compact format for batch
            status_emoji = self._get_status_emoji(game.status)
            confidence_emoji = self._get_confidence_emoji(metrics.confidence_level)
            
            if game.is_live:
                score = f"{game.home_score}-{game.away_score} ({game.current_minute}')"
            else:
                score = "vs"
            
            pred_text = (
                f"{i}. {status_emoji} <b>{self._truncate(game.home_team.name, 15)} {score} {self._truncate(game.away_team.name, 15)}</b>\n"
                f"   🎯 Over 2.5: <b>{metrics.probability_over_25:.1%}</b> {confidence_emoji}\n"
                f"   🏆 {self._truncate(game.league.name, 20)}\n"
            )
            
            # Add recommendation if available
            if prediction.recommended_action:
                pred_text += f"   💡 {prediction.recommended_action}\n"
            
            prediction_texts.append(pred_text + "\n")
        
        # Footer
        footer = (
            f"\n{'='*30}\n"
            f"🔔 Total games analyzed: {batch.total_games}\n"
            f"📋 Full report available at dashboard\n"
            f"🔄 Next update in ~5 minutes"
        )
        
        return header + "".join(prediction_texts) + footer
    
    # HTML Formatters
    
    def _format_for_html(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction as HTML"""
        game = prediction.game
        metrics = prediction.probability_metrics
        risk = prediction.risk_assessment
        
        # Determine CSS classes based on alert level
        alert_class = f"alert-{alert_level.value}"
        card_class = f"prediction-card {alert_class}"
        
        # Create HTML
        html_content = f"""
        <div class="{card_class}">
            <div class="prediction-header">
                <h3>{self._escape(game.home_team.name)} vs {self._escape(game.away_team.name)}</h3>
                <span class="league">{self._escape(game.league.name)}</span>
            </div>
            
            <div class="match-status">
                <span class="status {game.status.value}">{game.status.value.upper()}</span>
                {self._format_game_status_html(game)}
            </div>
            
            <div class="probability-section">
                <div class="probability-main">
                    <h4>Over 2.5 Goals Probability</h4>
                    <div class="probability-value {self._get_probability_class(metrics.probability_over_25)}">
                        {metrics.probability_over_25:.1%}
                    </div>
                    <div class="confidence-badge {metrics.confidence_level.value}">
                        {metrics.confidence_level.value.replace('_', ' ').title()} Confidence
                    </div>
                </div>
                
                <div class="probability-details">
                    <div class="detail-row">
                        <span>Over 1.5:</span>
                        <span class="value">{metrics.probability_over_15:.1%}</span>
                    </div>
                    <div class="detail-row">
                        <span>Over 3.5:</span>
                        <span class="value">{metrics.probability_over_35:.1%}</span>
                    </div>
                    <div class="detail-row">
                        <span>Expected Goals:</span>
                        <span class="value">{metrics.expected_total_goals:.2f}</span>
                    </div>
                </div>
            </div>
            
            {self._format_risk_html(risk)}
            
            {self._format_recommendation_html(prediction)}
            
            {self._format_factors_html(prediction.key_factors) if prediction.key_factors else ''}
            
            <div class="prediction-footer">
                <small>Updated: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
                <small class="prediction-id">ID: {prediction.id[:8]}</small>
            </div>
        </div>
        """
        
        return html_content
    
    def _format_batch_for_html(self, batch: BatchPrediction, predictions: List[Prediction]) -> str:
        """Format batch as HTML"""
        if not predictions:
            return self._format_no_predictions(OutputFormat.HTML)
        
        # Batch header
        html_content = f"""
        <div class="batch-container">
            <div class="batch-header">
                <h2>Prediction Batch Report</h2>
                <div class="batch-meta">
                    <div class="meta-item">
                        <span class="label">Generated:</span>
                        <span class="value">{batch.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    <div class="meta-item">
                        <span class="label">Total Games:</span>
                        <span class="value">{batch.total_games}</span>
                    </div>
                    <div class="meta-item">
                        <span class="label">High Confidence:</span>
                        <span class="value">{batch.high_confidence_predictions}</span>
                    </div>
                    <div class="meta-item">
                        <span class="label">Value Bets:</span>
                        <span class="value">{batch.value_bet_opportunities}</span>
                    </div>
                </div>
            </div>
            
            <div class="batch-predictions">
        """
        
        # Add each prediction
        for i, prediction in enumerate(predictions, 1):
            pred_html = self._format_for_html(prediction, self._determine_alert_level(prediction))
            html_content += f'<div class="batch-item">{pred_html}</div>'
        
        # Footer
        html_content += """
            </div>
            
            <div class="batch-footer">
                <p class="disclaimer">
                    Predictions are based on statistical models and should not be considered financial advice.
                    Always gamble responsibly.
                </p>
            </div>
        </div>
        """
        
        return html_content
    
    # Console Formatters
    
    def _format_for_console(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction for console output"""
        game = prediction.game
        metrics = prediction.probability_metrics
        
        # Header
        header = "=" * 80 + "\n"
        
        if alert_level == AlertLevel.HIGH:
            header += "🚨 HIGH CONFIDENCE PREDICTION 🚨\n"
        elif alert_level == AlertLevel.MEDIUM:
            header += "📢 GOOD OPPORTUNITY 📢\n"
        else:
            header += "📋 PREDICTION UPDATE 📋\n"
        
        header += "=" * 80 + "\n\n"
        
        # Match info
        match_info = (
            f"MATCH: {game.home_team.name} vs {game.away_team.name}\n"
            f"LEAGUE: {game.league.name}\n"
            f"STATUS: {game.status.value.upper()}"
        )
        
        if game.is_live:
            match_info += f" | SCORE: {game.home_score}-{game.away_score} ({game.current_minute}')\n"
        else:
            match_info += f" | STARTS: {game.start_time.strftime('%H:%M')}\n"
        
        match_info += "-" * 40 + "\n"
        
        # Probability info
        prob_info = (
            "PROBABILITY ANALYSIS:\n"
            f"  Over 2.5 Goals: {metrics.probability_over_25:6.1%}  [Confidence: {metrics.confidence_level.value}]\n"
            f"  Over 1.5 Goals: {metrics.probability_over_15:6.1%}\n"
            f"  Over 3.5 Goals: {metrics.probability_over_35:6.1%}\n"
            f"  Expected Goals: {metrics.expected_total_goals:.2f}\n"
        )
        
        # Risk info
        risk = prediction.risk_assessment
        risk_info = (
            f"RISK ASSESSMENT: {risk.risk_level.value.upper()} (Score: {risk.overall_risk_score:.2f})\n"
            f"  Time Risk:     {risk.time_risk:.2f}\n"
            f"  Data Risk:     {risk.data_risk:.2f}\n"
            f"  Volatility:    {risk.volatility_risk:.2f}\n"
        )
        
        # Recommendation
        recommendation = ""
        if prediction.recommended_action:
            recommendation = (
                f"\nRECOMMENDATION: {prediction.recommended_action}\n"
            )
            if risk.recommended_stake:
                recommendation += f"STAKE SUGGESTION: {risk.recommended_stake:.2%} of bankroll\n"
        
        # Key factors
        factors = ""
        if prediction.key_factors:
            factors = "\nKEY FACTORS:\n"
            for factor in prediction.key_factors[:3]:
                factors += f"  • {factor}\n"
        
        # Footer
        footer = (
            "\n" + "-" * 40 + "\n"
            f"Prediction ID: {prediction.id}\n"
            f"Generated: {prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "=" * 80
        )
        
        return header + match_info + prob_info + risk_info + recommendation + factors + footer
    
    def _format_batch_for_console(self, batch: BatchPrediction, predictions: List[Prediction]) -> str:
        """Format batch for console"""
        if not predictions:
            return self._format_no_predictions(OutputFormat.CONSOLE)
        
        header = "=" * 80 + "\n"
        header += "📊 PREDICTION BATCH REPORT 📊\n"
        header += "=" * 80 + "\n\n"
        
        stats = (
            f"Generated:     {batch.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Total Games:   {batch.total_games}\n"
            f"High Confidence Predictions: {batch.high_confidence_predictions}\n"
            f"Value Bet Opportunities: {batch.value_bet_opportunities}\n"
            f"Estimated Success Rate: {batch.success_rate_estimate:.1%}\n\n"
            + "-" * 80 + "\n"
            "HIGH PRIORITY PREDICTIONS:\n"
            + "-" * 80 + "\n"
        )
        
        predictions_text = ""
        for i, prediction in enumerate(predictions, 1):
            game = prediction.game
            metrics = prediction.probability_metrics
            
            # Compact format
            if game.is_live:
                score = f"{game.home_score}-{game.away_score} ({game.current_minute}')"
            else:
                score = "vs"
            
            pred_line = (
                f"{i:2d}. {game.home_team.name[:20]:20} {score:15} {game.away_team.name[:20]:20}\n"
                f"     Over 2.5: {metrics.probability_over_25:5.1%} | Confidence: {metrics.confidence_level.value:10} | "
            )
            
            if prediction.recommended_action:
                pred_line += f"Rec: {prediction.recommended_action}\n"
            else:
                pred_line += "\n"
            
            predictions_text += pred_line + "\n"
        
        footer = "=" * 80 + "\n"
        
        return header + stats + predictions_text + footer
    
    # JSON Formatter
    
    def _format_for_json(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction as JSON"""
        data = prediction.to_dict()
        data['alert_level'] = alert_level.value
        data['formatted'] = {
            'telegram': self._format_for_telegram(prediction, alert_level),
            'html': self._format_for_html(prediction, alert_level),
            'console': self._format_for_console(prediction, alert_level),
        }
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    # CSV Formatter
    
    def _format_for_csv(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction as CSV"""
        game = prediction.game
        metrics = prediction.probability_metrics
        risk = prediction.risk_assessment
        
        fields = [
            prediction.timestamp.isoformat(),
            prediction.id,
            game.home_team.name,
            game.away_team.name,
            game.league.name,
            game.status.value,
            str(game.home_score),
            str(game.away_score),
            str(game.current_minute),
            f"{metrics.probability_over_25:.4f}",
            f"{metrics.probability_over_15:.4f}",
            f"{metrics.probability_over_35:.4f}",
            metrics.confidence_level.value,
            f"{metrics.confidence_score:.4f}",
            f"{metrics.expected_total_goals:.4f}",
            risk.risk_level.value,
            f"{risk.overall_risk_score:.4f}",
            prediction.recommended_action or "",
            alert_level.value,
        ]
        
        # Escape commas and quotes
        escaped_fields = []
        for field in fields:
            if field is None:
                field = ""
            field_str = str(field)
            if ',' in field_str or '"' in field_str:
                field_str = f'"{field_str.replace(\"\", "\"\"")}"'
            escaped_fields.append(field_str)
        
        return ",".join(escaped_fields)
    
    def get_csv_header(self) -> str:
        """Get CSV header row"""
        headers = [
            "timestamp",
            "prediction_id",
            "home_team",
            "away_team",
            "league",
            "status",
            "home_score",
            "away_score",
            "current_minute",
            "probability_over_25",
            "probability_over_15",
            "probability_over_35",
            "confidence_level",
            "confidence_score",
            "expected_total_goals",
            "risk_level",
            "risk_score",
            "recommendation",
            "alert_level",
        ]
        return ",".join(headers)
    
    # Email Formatter
    
    def _format_for_email(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction for email"""
        # Use HTML format for email with proper styling
        html_content = self._format_for_html(prediction, alert_level)
        
        # Wrap in email template
        email_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Over/Under Prediction Alert</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .email-container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 5px; }}
                .content {{ padding: 20px; }}
                .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; text-align: center; }}
                .disclaimer {{ font-size: 11px; color: #999; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <h2>⚽ Over/Under Prediction Alert</h2>
                    <p>Automated prediction from your Over/Under Predictor System</p>
                </div>
                
                <div class="content">
                    {html_content}
                </div>
                
                <div class="footer">
                    <p>This is an automated message from your Over/Under Predictor System.</p>
                    <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="disclaimer">
                        <p><strong>Disclaimer:</strong> These predictions are based on statistical models and historical data.
                        They should not be considered as financial advice. Always gamble responsibly and within your means.
                        Past performance is not indicative of future results.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return email_template
    
    def _format_batch_for_email(self, batch: BatchPrediction, predictions: List[Prediction]) -> str:
        """Format batch for email"""
        html_content = self._format_batch_for_html(batch, predictions)
        
        email_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Batch Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .email-container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 5px; margin-bottom: 20px; }}
                .batch-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }}
                .stat-item {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #3B82F6; }}
                .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #666; text-align: center; }}
                .disclaimer {{ font-size: 11px; color: #999; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <h2>📊 Prediction Batch Report</h2>
                    <p>Analysis of {batch.total_games} football matches</p>
                </div>
                
                <div class="batch-stats">
                    <div class="stat-item">
                        <div class="stat-value">{batch.total_games}</div>
                        <div class="stat-label">Total Games</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{batch.high_confidence_predictions}</div>
                        <div class="stat-label">High Confidence</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{batch.value_bet_opportunities}</div>
                        <div class="stat-label">Value Bets</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{batch.success_rate_estimate:.1%}</div>
                        <div class="stat-label">Success Rate</div>
                    </div>
                </div>
                
                {html_content}
                
                <div class="footer">
                    <p>This is an automated report from your Over/Under Predictor System.</p>
                    <p>Report generated at: {batch.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="disclaimer">
                        <p><strong>Disclaimer:</strong> These predictions are based on statistical models and historical data.
                        They should not be considered as financial advice. Always gamble responsibly and within your means.
                        Past performance is not indicative of future results.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return email_template
    
    # Discord Formatter
    
    def _format_for_discord(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction for Discord"""
        game = prediction.game
        metrics = prediction.probability_metrics
        
        # Discord uses markdown-like formatting
        if alert_level == AlertLevel.HIGH:
            header = "**🚨 HIGH CONFIDENCE ALERT 🚨**\n\n"
        elif alert_level == AlertLevel.MEDIUM:
            header = "**📢 GOOD OPPORTUNITY 📢**\n\n"
        else:
            header = "**📋 PREDICTION UPDATE 📋**\n\n"
        
        # Match info
        status_emoji = self._get_status_emoji(game.status)
        match_info = (
            f"{status_emoji} **{game.home_team.name} vs {game.away_team.name}**\n"
            f"🏆 {game.league.name}\n"
        )
        
        if game.is_live:
            match_info += f"📊 {game.home_score}-{game.away_score} ({game.current_minute}')\n"
        else:
            match_info += f"🕐 Starts: {game.start_time.strftime('%H:%M')}\n"
        
        # Probability
        prob_info = (
            f"\n**Probability Analysis**\n"
            f"🎯 **Over 2.5 Goals:** {metrics.probability_over_25:.1%}\n"
            f"📊 Confidence: {metrics.confidence_level.value.replace('_', ' ').title()}\n"
            f"⚠️ Risk Level: {prediction.risk_assessment.risk_level.value.replace('_', ' ').title()}\n"
        )
        
        # Recommendation
        recommendation = ""
        if prediction.recommended_action:
            recommendation = f"\n**Recommendation:** {prediction.recommended_action}\n"
        
        # Footer
        footer = f"\n*Updated: {prediction.timestamp.strftime('%H:%M:%S')}*"
        
        return header + match_info + prob_info + recommendation + footer
    
    # Slack Formatter
    
    def _format_for_slack(self, prediction: Prediction, alert_level: AlertLevel) -> str:
        """Format prediction for Slack"""
        # Similar to Discord but with Slack-specific formatting
        discord_format = self._format_for_discord(prediction, alert_level)
        
        # Add Slack-specific formatting
        lines = discord_format.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('**') and line.endswith('**'):
                # Convert to Slack bold
                line = f"*{line[2:-2]}*"
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    # Helper methods
    
    def _get_status_emoji(self, status: GameStatus) -> str:
        """Get emoji for game status"""
        emoji_map = {
            GameStatus.LIVE: self.emoji['live'],
            GameStatus.HALFTIME: self.emoji['halftime'],
            GameStatus.FINISHED: self.emoji['finished'],
            GameStatus.SCHEDULED: self.emoji['scheduled'],
        }
        return emoji_map.get(status, '⚽')
    
    def _get_confidence_emoji(self, confidence: PredictionConfidence) -> str:
        """Get emoji for confidence level"""
        if confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
            return self.emoji['high_confidence']
        elif confidence == PredictionConfidence.MEDIUM:
            return self.emoji['medium_confidence']
        else:
            return self.emoji['low_confidence']
    
    def _get_risk_emoji(self, risk: RiskLevel) -> str:
        """Get emoji for risk level"""
        if risk in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            return self.emoji['risk_low']
        elif risk == RiskLevel.MEDIUM:
            return self.emoji['risk_medium']
        else:
            return self.emoji['risk_high']
    
    def _get_probability_class(self, probability: float) -> str:
        """Get CSS class for probability value"""
        if probability >= 0.75:
            return "probability-high"
        elif probability >= 0.60:
            return "probability-medium"
        else:
            return "probability-low"
    
    def _truncate(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text to specified length"""
        if max_length is None:
            max_length = self.config['truncate_team_names']
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - 3] + "..."
    
    def _escape(self, text: str) -> str:
        """Escape HTML special characters"""
        return html.escape(str(text))
    
    def _format_game_status_html(self, game: Game) -> str:
        """Format game status as HTML"""
        if game.is_live:
            return f"""
                <div class="score">
                    <span class="home-score">{game.home_score}</span>
                    <span class="divider">-</span>
                    <span class="away-score">{game.away_score}</span>
                    <span class="minute">({game.current_minute}')</span>
                </div>
            """
        elif game.status == GameStatus.FINISHED:
            return f"""
                <div class="score final">
                    <span class="home-score">{game.home_score}</span>
                    <span class="divider">-</span>
                    <span class="away-score">{game.away_score}</span>
                    <span class="label">FT</span>
                </div>
            """
        else:
            return f"""
                <div class="start-time">
                    Starts at {game.start_time.strftime('%H:%M')}
                </div>
            """
    
    def _format_risk_html(self, risk: RiskAssessment) -> str:
        """Format risk assessment as HTML"""
        risk_class = f"risk-{risk.risk_level.value}"
        
        return f"""
            <div class="risk-assessment {risk_class}">
                <h4>Risk Assessment</h4>
                <div class="risk-meter">
                    <div class="risk-bar" style="width: {risk.overall_risk_score * 100}%"></div>
                </div>
                <div class="risk-details">
                    <span class="risk-level">{risk.risk_level.value.replace('_', ' ').title()}</span>
                    <span class="risk-score">Score: {risk.overall_risk_score:.2f}</span>
                </div>
                
                <div class="risk-factors">
                    <div class="risk-factor">
                        <span class="label">Time Risk:</span>
                        <span class="value">{risk.time_risk:.2f}</span>
                    </div>
                    <div class="risk-factor">
                        <span class="label">Data Risk:</span>
                        <span class="value">{risk.data_risk:.2f}</span>
                    </div>
                    <div class="risk-factor">
                        <span class="label">Volatility:</span>
                        <span class="value">{risk.volatility_risk:.2f}</span>
                    </div>
                </div>
                
                {self._format_stake_recommendation_html(risk) if risk.recommended_stake else ''}
            </div>
        """
    
    def _format_stake_recommendation_html(self, risk: RiskAssessment) -> str:
        """Format stake recommendation as HTML"""
        return f"""
            <div class="stake-recommendation">
                <span class="label">Suggested Stake:</span>
                <span class="value">{risk.recommended_stake:.2%}</span>
                <small>of bankroll (max: {risk.max_stake:.0%})</small>
            </div>
        """
    
    def _format_recommendation_html(self, prediction: Prediction) -> str:
        """Format recommendation as HTML"""
        if not prediction.recommended_action:
            return ""
        
        recommendation_class = "recommendation"
        if prediction.is_value_bet:
            recommendation_class += " value-bet"
        
        return f"""
            <div class="{recommendation_class}">
                <h4>Recommendation</h4>
                <div class="recommendation-content">
                    <strong>{prediction.recommended_action}</strong>
                    {self._format_value_bet_html(prediction) if prediction.is_value_bet else ''}
                </div>
            </div>
        """
    
    def _format_value_bet_html(self, prediction: Prediction) -> str:
        """Format value bet indicator as HTML"""
        value_score = prediction.probability_metrics.value_bet_score
        if value_score:
            return f"""
                <div class="value-bet-indicator">
                    <span class="value-bet-badge">VALUE BET</span>
                    <span class="value-bet-score">Edge: {value_score:.1%}</span>
                </div>
            """
        return ""
    
    def _format_factors_html(self, factors: List[str]) -> str:
        """Format key factors as HTML"""
        if not factors:
            return ""
        
        factor_items = "".join([f'<li>{self._escape(factor)}</li>' for factor in factors[:3]])
        
        return f"""
            <div class="key-factors">
                <h4>Key Factors</h4>
                <ul>
                    {factor_items}
                </ul>
            </div>
        """
    
    def _generate_hashtags(self, game: Game, metrics: ProbabilityMetrics) -> str:
        """Generate hashtags for social media"""
        hashtags = [
            "#FootballPredictions",
            "#OverUnder",
            "#BettingTips",
        ]
        
        # Add league hashtag
        league_name = game.league.name.replace(' ', '')
        hashtags.append(f"#{league_name}")
        
        # Add country hashtag
        country = game.league.country.replace(' ', '')
        hashtags.append(f"#{country}Football")
        
        # Add probability hashtag
        if metrics.probability_over_25 >= 0.75:
            hashtags.append("#HighProbability")
        elif metrics.probability_over_25 <= 0.25:
            hashtags.append("#LowProbability")
        
        # Add confidence hashtag
        if metrics.confidence_level in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
            hashtags.append("#HighConfidence")
        
        return " ".join(hashtags)
    
    def _format_no_predictions(self, format_type: OutputFormat) -> str:
        """Format message when no predictions are available"""
        messages = {
            OutputFormat.TELEGRAM: (
                f"{self.emoji['info']} <b>No High Priority Predictions</b>\n\n"
                f"Currently no games meet the alert criteria. "
                f"Monitoring continues...\n\n"
                f"🔄 Next update in ~5 minutes"
            ),
            OutputFormat.HTML: """
                <div class="no-predictions">
                    <h3>No High Priority Predictions</h3>
                    <p>Currently no games meet the alert criteria. Monitoring continues...</p>
                    <p class="next-update">Next update in ~5 minutes</p>
                </div>
            """,
            OutputFormat.CONSOLE: (
                "=" * 80 + "\n"
                "NO HIGH PRIORITY PREDICTIONS\n"
                "=" * 80 + "\n\n"
                "Currently no games meet the alert criteria.\n"
                "Monitoring continues...\n\n"
                "Next update in ~5 minutes\n"
                + "=" * 80
            ),
            OutputFormat.JSON: json.dumps({
                "status": "no_predictions",
                "message": "No high priority predictions available",
                "timestamp": datetime.now().isoformat()
            }, indent=2),
        }
        
        return messages.get(format_type, "No predictions available")


# CSS Styles for HTML output

HTML_STYLES = """
<style>
    .prediction-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .alert-high {
        border-left: 4px solid #dc2626;
        background-color: #fef2f2;
    }
    
    .alert-medium {
        border-left: 4px solid #f59e0b;
        background-color: #fffbeb;
    }
    
    .alert-low {
        border-left: 4px solid #3b82f6;
        background-color: #eff6ff;
    }
    
    .prediction-header {
        margin-bottom: 15px;
    }
    
    .prediction-header h3 {
        margin: 0 0 5px 0;
        color: #1f2937;
        font-size: 18px;
    }
    
    .league {
        color: #6b7280;
        font-size: 14px;
    }
    
    .match-status {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        padding: 10px;
        background: #f9fafb;
        border-radius: 4px;
    }
    
    .status {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .status.live {
        background-color: #dc2626;
        color: white;
    }
    
    .status.finished {
        background-color: #10b981;
        color: white;
    }
    
    .status.scheduled {
        background-color: #6b7280;
        color: white;
    }
    
    .score {
        display: flex;
        align-items: center;
        gap: 5px;
        font-weight: bold;
    }
    
    .home-score, .away-score {
        font-size: 18px;
    }
    
    .divider {
        color: #6b7280;
    }
    
    .minute, .label {
        font-size: 12px;
        color: #6b7280;
    }
    
    .probability-section {
        margin-bottom: 20px;
    }
    
    .probability-main {
        text-align: center;
        margin-bottom: 15px;
    }
    
    .probability-main h4 {
        margin: 0 0 10px 0;
        color: #374151;
        font-size: 14px;
    }
    
    .probability-value {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .probability-high {
        color: #10b981;
    }
    
    .probability-medium {
        color: #f59e0b;
    }
    
    .probability-low {
        color: #ef4444;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .confidence-badge.high, .confidence-badge.very_high {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .confidence-badge.medium {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .confidence-badge.low, .confidence-badge.very_low {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .probability-details {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        font-size: 14px;
    }
    
    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .detail-row .value {
        font-weight: bold;
    }
    
    .risk-assessment {
        padding: 15px;
        background: #f8fafc;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    
    .risk-assessment h4 {
        margin: 0 0 10px 0;
        color: #374151;
        font-size: 14px;
    }
    
    .risk-meter {
        height: 6px;
        background: #e5e7eb;
        border-radius: 3px;
        margin-bottom: 10px;
        overflow: hidden;
    }
    
    .risk-bar {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
        transition: width 0.3s ease;
    }
    
    .risk-details {
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
        font-size: 14px;
    }
    
    .risk-level {
        font-weight: bold;
    }
    
    .risk-score {
        color: #6b7280;
    }
    
    .risk-factors {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        font-size: 12px;
        margin-bottom: 10px;
    }
    
    .risk-factor {
        display: flex;
        justify-content: space-between;
    }
    
    .risk-factor .label {
        color: #6b7280;
    }
    
    .risk-factor .value {
        font-weight: bold;
    }
    
    .stake-recommendation {
        text-align: center;
        padding: 10px;
        background: white;
        border-radius: 4px;
        border: 1px solid #e5e7eb;
        font-size: 14px;
    }
    
    .stake-recommendation .value {
        font-weight: bold;
        color: #10b981;
    }
    
    .stake-recommendation small {
        color: #6b7280;
        display: block;
        margin-top: 2px;
    }
    
    .recommendation {
        padding: 15px;
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        border-radius: 6px;
        margin-bottom: 20px;
        border-left: 4px solid #0ea5e9;
    }
    
    .recommendation.value-bet {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-left-color: #10b981;
    }
    
    .recommendation h4 {
        margin: 0 0 10px 0;
        color: #0c4a6e;
        font-size: 14px;
    }
    
    .value-bet-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
    }
    
    .value-bet-badge {
        background-color: #10b981;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .value-bet-score {
        font-size: 12px;
        color: #065f46;
        font-weight: bold;
    }
    
    .key-factors {
        margin-bottom: 20px;
    }
    
    .key-factors h4 {
        margin: 0 0 10px 0;
        color: #374151;
        font-size: 14px;
    }
    
    .key-factors ul {
        margin: 0;
        padding-left: 20px;
        color: #4b5563;
        font-size: 14px;
    }
    
    .key-factors li {
        margin-bottom: 5px;
    }
    
    .prediction-footer {
        display: flex;
        justify-content: space-between;
        padding-top: 15px;
        border-top: 1px solid #e5e7eb;
        font-size: 12px;
        color: #6b7280;
    }
    
    .prediction-id {
        font-family: monospace;
    }
    
    /* Batch container styles */
    .batch-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .batch-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .batch-header h2 {
        color: #1f2937;
        margin-bottom: 15px;
    }
    
    .batch-meta {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
    }
    
    .meta-item {
        text-align: center;
    }
    
    .meta-item .label {
        display: block;
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    .meta-item .value {
        display: block;
        font-size: 18px;
        font-weight: bold;
        color: #3b82f6;
    }
    
    .batch-predictions {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 20px;
    }
    
    .batch-item {
        break-inside: avoid;
    }
    
    .batch-footer {
        margin-top: 30px;
        text-align: center;
    }
    
    .disclaimer {
        font-size: 12px;
        color: #9ca3af;
        line-height: 1.5;
        max-width: 600px;
        margin: 20px auto 0;
    }
    
    /* No predictions styles */
    .no-predictions {
        text-align: center;
        padding: 40px 20px;
        background: #f9fafb;
        border-radius: 8px;
        border: 2px dashed #d1d5db;
    }
    
    .no-predictions h3 {
        color: #6b7280;
        margin-bottom: 10px;
    }
    
    .no-predictions p {
        color: #9ca3af;
        margin-bottom: 10px;
    }
    
    .next-update {
        font-style: italic;
        color: #6b7280;
    }
</style>
"""


# Utility functions

def get_formatted_timestamp(dt: Optional[datetime] = None) -> str:
    """Get formatted timestamp"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_probability(probability: float, decimals: int = 1) -> str:
    """Format probability as percentage string"""
    return f"{probability * 100:.{decimals}f}%"


def format_odds(odds: float) -> str:
    """Format odds as string"""
    return f"{odds:.2f}"


def format_time_remaining(minutes: int) -> str:
    """Format time remaining in match"""
    if minutes <= 0:
        return "Match ended"
    
    if minutes < 90:
        return f"{minutes}' remaining"
    
    return f"{minutes}' (including extra time)"


# Factory function for easy instantiation

def create_formatter(config: Optional[Dict] = None) -> PredictionFormatter:
    """
    Create a PredictionFormatter instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        PredictionFormatter instance
    """
    return PredictionFormatter(config)


# Example usage
if __name__ == "__main__":
    # Create a sample prediction for testing
    from datetime import datetime
    
    # Create sample teams
    home_team = Team(
        id="team_man_city",
        name="Manchester City",
        short_name="MCI",
        country="England",
        avg_goals_scored=2.3,
        avg_goals_conceded=0.8
    )
    
    away_team = Team(
        id="team_liverpool",
        name="Liverpool FC",
        short_name="LIV",
        country="England",
        avg_goals_scored=2.1,
        avg_goals_conceded=1.0
    )
    
    # Create sample league
    league = League(
        id="league_epl",
        name="English Premier League",
        country="England",
        avg_goals_per_game=2.8,
        over_25_rate=0.55
    )
    
    # Create sample game
    game = Game(
        id="match_12345",
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
        expected_goals_total=3.5,
        odds_over_25=1.85,
        odds_under_25=1.95
    )
    
    # Create sample probability metrics
    metrics = ProbabilityMetrics(
        game_id="match_12345",
        probability_over_05=0.95,
        probability_over_15=0.85,
        probability_over_25=0.72,
        probability_over_35=0.45,
        implied_prob_over_25=0.54,  # 1/1.85
        implied_prob_under_25=0.51,  # 1/1.95
        confidence_score=0.78,
        expected_additional_goals=0.8,
        expected_total_goals=3.3,
        model_contributions={
            "time_based": 0.35,
            "statistical": 0.40,
            "momentum": 0.25
        }
    )
    
    # Create sample risk assessment
    risk = RiskAssessment(
        game_id="match_12345",
        time_risk=0.4,
        data_risk=0.2,
        volatility_risk=0.3,
        league_risk=0.1
    )
    
    # Create sample prediction
    prediction = Prediction(
        id="pred_abc123",
        game=game,
        probability_metrics=metrics,
        risk_assessment=risk,
        key_factors=[
            "High expected goals total (3.5)",
            "Both teams creating chances",
            "League average goals: 2.8 per game"
        ],
        recommended_action="OVER 2.5 GOALS",
        confidence_summary="Good confidence based on attacking statistics"
    )
    
    # Test the formatter
    formatter = PredictionFormatter()
    
    print("Testing Prediction Formatter\n")
    print("=" * 80)
    
    # Test Telegram format
    print("Telegram Format:")
    print("-" * 40)
    telegram_output = formatter.format_prediction(prediction, OutputFormat.TELEGRAM)
    print(telegram_output)
    print("\n" + "=" * 80)
    
    # Test HTML format
    print("\nHTML Format (first 500 chars):")
    print("-" * 40)
    html_output = formatter.format_prediction(prediction, OutputFormat.HTML)
    print(html_output[:500] + "...")
    print("\n" + "=" * 80)
    
    # Test Console format
    print("\nConsole Format:")
    print("-" * 40)
    console_output = formatter.format_prediction(prediction, OutputFormat.CONSOLE)
    print(console_output)
    print("\n" + "=" * 80)
    
    # Test JSON format
    print("\nJSON Format (first 300 chars):")
    print("-" * 40)
    json_output = formatter.format_prediction(prediction, OutputFormat.JSON)
    print(json_output[:300] + "...")
    print("\n" + "=" * 80)
    
    # Test CSV format
    print("\nCSV Format:")
    print("-" * 40)
    csv_output = formatter.format_prediction(prediction, OutputFormat.CSV)
    print(csv_output)
    print("\n" + "=" * 80)
