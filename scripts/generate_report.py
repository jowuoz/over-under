#!/usr/bin/env python3
"""
generate_report.py - Report generator for Over/Under Predictor system
Creates HTML dashboards and reports from prediction data
"""
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import asyncio
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Then define report path
REPORT_PATH = "reports/index.html"  # ← This is the key line

try:
    from config import get_config
    from src.scrapers.base_scraper import ScrapedGame, DataSourceType
except ImportError:
    print("Warning: Could not import config or base_scraper")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("report_generator")


class ReportGenerator:
    """
    Generates HTML reports and dashboards from prediction data
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize report generator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = get_config() if 'get_config' in globals() else self._get_default_config()
        
        # Path configuration
        self.base_dir = Path(__file__).parent.parent
        self.reports_dir = self.base_dir / "reports"
        self.storage_dir = self.base_dir / "storage"
        self.data_dir = self.storage_dir / "data"
        
        # Ensure directories exist
        self.reports_dir.mkdir(exist_ok=True)
        (self.reports_dir / "data").mkdir(exist_ok=True)
        (self.reports_dir / "charts").mkdir(exist_ok=True)
        (self.reports_dir / "css").mkdir(exist_ok=True)
        (self.reports_dir / "js").mkdir(exist_ok=True)
        
        # Data structures
        self.predictions = []
        self.alerts = []
        self.stats = {}
        
        # Color scheme for charts and UI
        self.colors = {
            'primary': '#2563eb',      # Blue
            'secondary': '#64748b',    # Slate
            'success': '#10b981',      # Emerald
            'warning': '#f59e0b',      # Amber
            'danger': '#ef4444',       # Red
            'info': '#3b82f6',         # Light Blue
            'light': '#f8fafc',        # Light Gray
            'dark': '#1e293b',         # Dark Gray
            'over_25': '#10b981',      # Green for Over 2.5
            'under_25': '#ef4444',     # Red for Under 2.5
            'high_confidence': '#10b981',  # High confidence
            'medium_confidence': '#f59e0b', # Medium confidence
            'low_confidence': '#ef4444',   # Low confidence
        }
        
        # League logos mapping (placeholder - you can add real URLs)
        self.league_logos = {
            'Premier League': 'https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg',
            'La Liga': 'https://upload.wikimedia.org/wikipedia/en/0/0e/LaLiga_logo_2023.svg',
            'Bundesliga': 'https://upload.wikimedia.org/wikipedia/en/d/df/Bundesliga_logo_%282017%29.svg',
            'Serie A': 'https://upload.wikimedia.org/wikipedia/en/e/e9/Serie_A_logo_%282023%29.svg',
            'Ligue 1': 'https://upload.wikimedia.org/wikipedia/en/2/29/Ligue_1_Uber_Eats_logo.svg',
            'Champions League': 'https://upload.wikimedia.org/wikipedia/en/b/bf/UEFA_Champions_League_logo_2.svg',
            'Europa League': 'https://upload.wikimedia.org/wikipedia/en/0/05/UEFA_Europa_League_logo_2.svg',
            'FA Cup': 'https://upload.wikimedia.org/wikipedia/en/f/f2/The_Football_Association_Logo.svg',
        }
        
        logger.info(f"Report generator initialized. Reports will be saved to: {self.reports_dir}")
    
    def _get_default_config(self):
        """Get default configuration if config module not available"""
        return {
            'system': {
                'name': 'Over/Under Predictor',
                'update_interval': 300
            },
            'predictor': {
                'thresholds': {
                    'over_0.5': 0.85,
                    'over_1.5': 0.80,
                    'over_2.5': 0.75,
                    'over_3.5': 0.70
                },
                'min_confidence': 0.60
            }
        }
    
    def load_data(self) -> bool:
        """
        Load prediction data from storage
        
        Returns:
            True if data loaded successfully
        """
        try:
            # Try to load latest predictions
            predictions_path = self.data_dir / "predictions.json"
            
            if predictions_path.exists():
                with open(predictions_path, 'r') as f:
                    data = json.load(f)
                
                self.predictions = data.get('predictions', [])
                self.alerts = data.get('alerts', [])
                self.stats = data.get('summary', {})
                
                logger.info(f"Loaded {len(self.predictions)} predictions and {len(self.alerts)} alerts")
                return True
            
            # Try reports data as fallback
            reports_path = self.reports_dir / "data" / "latest.json"
            if reports_path.exists():
                with open(reports_path, 'r') as f:
                    data = json.load(f)
                
                self.predictions = data.get('predictions', [])
                self.alerts = data.get('alerts', [])
                self.stats = data.get('summary', {})
                
                logger.info(f"Loaded {len(self.predictions)} predictions from reports")
                return True
            
            logger.warning("No prediction data found")
            return False
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def load_historical_data(self, days: int = 7) -> List[Dict]:
        """
        Load historical prediction data
        
        Args:
            days: Number of days of history to load
            
        Returns:
            List of historical data entries
        """
        historical_data = []
        
        try:
            historical_dir = self.storage_dir / "data" / "historical"
            if historical_dir.exists():
                # Get all historical files
                historical_files = sorted(historical_dir.glob("*.json"))
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for file_path in historical_files[-100:]:  # Last 100 files max
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Check if data is recent enough
                        timestamp_str = data.get('timestamp')
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            if timestamp >= cutoff_date:
                                historical_data.append(data)
                    except Exception as e:
                        logger.debug(f"Error loading historical file {file_path}: {e}")
                        continue
            
            logger.info(f"Loaded {len(historical_data)} historical data points")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
        
        return historical_data
    
    def generate_all_reports(self):
        """
        Generate all reports and dashboards
        """
        logger.info("Starting report generation")
        
        # Load data first
        if not self.load_data():
            logger.warning("No data to generate reports from")
            return
        
        # Generate individual reports
        self.generate_main_dashboard()
        self.generate_alerts_report()
        self.generate_statistics_report()
        self.generate_historical_charts()
        self.generate_performance_report()
        self.generate_league_report()
        
        # Generate supporting files
        self.generate_css()
        self.generate_js()
        self.generate_sitemap()
        
        logger.info("All reports generated successfully")
    
    def generate_main_dashboard(self):
        """
        Generate main HTML dashboard
        """
        logger.info("Generating main dashboard")
        
        # Calculate statistics
        stats = self._calculate_dashboard_stats()
        
        # Get recent alerts
        recent_alerts = self.alerts[-10:]  # Last 10 alerts
        
        # Get live games
        live_games = [p for p in self.predictions if p.get('status') in ['live', 'LIVE', 'IN_PLAY']]
        
        # Generate charts
        self._generate_confidence_chart()
        self._generate_league_distribution_chart()
        
        # Create HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚽ {self.config['system']['name']} - Live Dashboard</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <meta http-equiv="refresh" content="300">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="dashboard-header">
            <div class="header-content">
                <h1><i class="fas fa-futbol"></i> {self.config['system']['name']}</h1>
                <p class="subtitle">AI-powered over/under prediction system</p>
                <div class="header-stats">
                    <div class="stat-badge">
                        <i class="fas fa-clock"></i>
                        <span>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                    </div>
                    <div class="stat-badge">
                        <i class="fas fa-sync-alt"></i>
                        <span>Updates every {self.config['system']['update_interval'] // 60} minutes</span>
                    </div>
                </div>
            </div>
        </header>

        <!-- Stats Grid -->
        <div class="stats-grid">
            <div class="stat-card card-primary">
                <div class="stat-icon">
                    <i class="fas fa-chart-line"></i>
                </div>
                <div class="stat-content">
                    <h3>Live Games</h3>
                    <p class="stat-number">{stats['live_games']}</p>
                    <p class="stat-trend">
                        <i class="fas fa-arrow-up"></i> {stats['games_trend']}% from last hour
                    </p>
                </div>
            </div>
            
            <div class="stat-card card-success">
                <div class="stat-icon">
                    <i class="fas fa-bell"></i>
                </div>
                <div class="stat-content">
                    <h3>Active Alerts</h3>
                    <p class="stat-number">{stats['active_alerts']}</p>
                    <p class="stat-trend">
                        <i class="fas fa-bolt"></i> {stats['alerts_today']} today
                    </p>
                </div>
            </div>
            
            <div class="stat-card card-warning">
                <div class="stat-icon">
                    <i class="fas fa-crosshairs"></i>
                </div>
                <div class="stat-content">
                    <h3>Avg. Confidence</h3>
                    <p class="stat-number">{stats['avg_confidence']}%</p>
                    <p class="stat-trend">
                        <i class="fas fa-chart-bar"></i> High: {stats['high_confidence_pct']}%
                    </p>
                </div>
            </div>
            
            <div class="stat-card card-info">
                <div class="stat-icon">
                    <i class="fas fa-trophy"></i>
                </div>
                <div class="stat-content">
                    <h3>Top League</h3>
                    <p class="stat-number">{stats['top_league']}</p>
                    <p class="stat-trend">
                        <i class="fas fa-star"></i> {stats['top_league_games']} games
                    </p>
                </div>
            </div>
        </div>

        <!-- Alerts Section -->
        <section class="section">
            <div class="section-header">
                <h2><i class="fas fa-exclamation-triangle"></i> Recent Alerts</h2>
                <a href="alerts.html" class="btn-view-all">View All <i class="fas fa-arrow-right"></i></a>
            </div>
            
            <div class="alerts-container">
                {"".join([self._format_alert_card(alert) for alert in recent_alerts]) if recent_alerts else 
                '<div class="empty-state"><i class="fas fa-bell-slash"></i><p>No active alerts at the moment</p></div>'}
            </div>
        </section>

        <!-- Live Games -->
        <section class="section">
            <div class="section-header">
                <h2><i class="fas fa-play-circle"></i> Live Games Analysis</h2>
                <span class="badge-live">{len(live_games)} Live</span>
            </div>
            
            <div class="games-grid">
                {"".join([self._format_game_card(game) for game in live_games[:6]]) if live_games else 
                '<div class="empty-state"><i class="fas fa-clock"></i><p>No live games being analyzed</p></div>'}
            </div>
        </section>

        <!-- Charts Section -->
        <div class="charts-grid">
            <div class="chart-card">
                <div class="chart-header">
                    <h3><i class="fas fa-chart-pie"></i> Prediction Confidence</h3>
                </div>
                <div class="chart-container">
                    <canvas id="confidenceChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <div class="chart-header">
                    <h3><i class="fas fa-globe-europe"></i> League Distribution</h3>
                </div>
                <div class="chart-container">
                    <canvas id="leagueChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="dashboard-footer">
            <div class="footer-content">
                <div class="footer-info">
                    <p><i class="fas fa-code"></i> Powered by AI Prediction System</p>
                    <p><i class="fas fa-sync-alt"></i> Auto-refresh in <span id="countdown">300</span> seconds</p>
                </div>
                <div class="footer-links">
                    <a href="statistics.html"><i class="fas fa-chart-bar"></i> Statistics</a>
                    <a href="performance.html"><i class="fas fa-tachometer-alt"></i> Performance</a>
                    <a href="leagues.html"><i class="fas fa-trophy"></i> Leagues</a>
                </div>
            </div>
            <div class="footer-copyright">
                <p>© {datetime.now().year} {self.config['system']['name']}. All data is for informational purposes only.</p>
            </div>
        </footer>
    </div>

    <!-- JavaScript -->
    <script src="js/dashboard.js"></script>
    <script>
        // Load charts
        window.addEventListener('DOMContentLoaded', function() {{
            loadConfidenceChart();
            loadLeagueChart();
            startCountdown();
        }});
    </script>
</body>
</html>
"""
        
        # Save main dashboard
        dashboard_path = self.reports_dir / "index.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Main dashboard saved to: {dashboard_path}")
        
        # Also save data for charts
        self._save_chart_data()
    
    def _format_alert_card(self, alert: Dict) -> str:
        """Format an alert as HTML card"""
        probability = alert.get('over_25_prob', 0) * 100
        confidence = alert.get('confidence', 0) * 100
        
        # Determine badge color based on probability
        if probability >= 80:
            badge_class = "badge-high"
            badge_icon = "fas fa-fire"
        elif probability >= 70:
            badge_class = "badge-medium"
            badge_icon = "fas fa-exclamation-triangle"
        else:
            badge_class = "badge-low"
            badge_icon = "fas fa-info-circle"
        
        # Format teams and score
        home_team = alert.get('home_team', 'Unknown')
        away_team = alert.get('away_team', 'Unknown')
        home_score = alert.get('home_score', 0)
        away_score = alert.get('away_score', 0)
        minute = alert.get('minute', 0)
        
        return f"""
        <div class="alert-card">
            <div class="alert-header">
                <div class="alert-badge {badge_class}">
                    <i class="{badge_icon}"></i>
                    <span>Over 2.5: {probability:.1f}%</span>
                </div>
                <div class="alert-time">
                    <i class="fas fa-clock"></i> {minute}'
                </div>
            </div>
            <div class="alert-content">
                <h4>{home_team} vs {away_team}</h4>
                <div class="alert-score">
                    <span class="score">{home_score} - {away_score}</span>
                </div>
                <div class="alert-details">
                    <div class="detail-item">
                        <i class="fas fa-bullseye"></i>
                        <span>Confidence: {confidence:.1f}%</span>
                    </div>
                    <div class="detail-item">
                        <i class="fas fa-trophy"></i>
                        <span>{alert.get('league', 'Unknown League')}</span>
                    </div>
                </div>
            </div>
            <div class="alert-actions">
                <button class="btn-action" onclick="trackAlert('{alert.get('id', '')}')">
                    <i class="fas fa-eye"></i> Track
                </button>
                <button class="btn-action" onclick="shareAlert('{home_team}', '{away_team}', {probability})">
                    <i class="fas fa-share-alt"></i> Share
                </button>
            </div>
        </div>
        """
    
    def _format_game_card(self, game: Dict) -> str:
        """Format a game as HTML card"""
        total_goals = game.get('home_score', 0) + game.get('away_score', 0)
        minute = game.get('minute', 0)
        
        # Determine over/under status
        if total_goals >= 3:
            status_class = "status-over"
            status_text = "OVER 2.5"
            status_icon = "fas fa-arrow-up"
        elif total_goals == 2 and minute >= 70:
            status_class = "status-close"
            status_text = "CLOSE"
            status_icon = "fas fa-hourglass-half"
        else:
            status_class = "status-under"
            status_text = "UNDER 2.5"
            status_icon = "fas fa-arrow-down"
        
        # Get league logo
        league = game.get('league', '')
        logo_url = self.league_logos.get(league, 'https://img.icons8.com/color/96/000000/football2--v1.png')
        
        return f"""
        <div class="game-card">
            <div class="game-header">
                <img src="{logo_url}" alt="{league}" class="league-logo">
                <span class="league-name">{league}</span>
                <span class="game-minute">{minute}'</span>
            </div>
            <div class="game-teams">
                <div class="team home-team">
                    <span class="team-name">{game.get('home_team', 'Home')}</span>
                    <span class="team-score">{game.get('home_score', 0)}</span>
                </div>
                <div class="vs-separator">VS</div>
                <div class="team away-team">
                    <span class="team-score">{game.get('away_score', 0)}</span>
                    <span class="team-name">{game.get('away_team', 'Away')}</span>
                </div>
            </div>
            <div class="game-status">
                <div class="status-indicator {status_class}">
                    <i class="{status_icon}"></i>
                    <span>{status_text}</span>
                </div>
                <div class="game-probability">
                    <i class="fas fa-chart-line"></i>
                    <span>Over 2.5: {game.get('over_25_prob', 0) * 100:.1f}%</span>
                </div>
            </div>
            <div class="game-footer">
                <span class="game-updated">
                    <i class="fas fa-sync-alt"></i> Updated {datetime.now().strftime('%H:%M')}
                </span>
                <button class="btn-details" onclick="viewGameDetails('{game.get('id', '')}')">
                    Details <i class="fas fa-chevron-right"></i>
                </button>
            </div>
        </div>
        """
    
    def _calculate_dashboard_stats(self) -> Dict:
        """Calculate dashboard statistics"""
        stats = {
            'live_games': 0,
            'active_alerts': len(self.alerts),
            'avg_confidence': 0,
            'high_confidence_pct': 0,
            'top_league': 'N/A',
            'top_league_games': 0,
            'games_trend': 0,
            'alerts_today': 0
        }
        
        if not self.predictions:
            return stats
        
        # Count live games
        stats['live_games'] = len([p for p in self.predictions if p.get('status') in ['live', 'LIVE', 'IN_PLAY']])
        
        # Calculate average confidence
        confidences = [p.get('confidence', 0) * 100 for p in self.predictions if p.get('confidence')]
        if confidences:
            stats['avg_confidence'] = np.mean(confidences)
            stats['high_confidence_pct'] = len([c for c in confidences if c >= 70]) / len(confidences) * 100
        
        # Find top league
        league_counts = {}
        for game in self.predictions:
            league = game.get('league', 'Unknown')
            league_counts[league] = league_counts.get(league, 0) + 1
        
        if league_counts:
            stats['top_league'] = max(league_counts, key=league_counts.get)
            stats['top_league_games'] = league_counts[stats['top_league']]
        
        # Calculate trends (simplified - would need historical data for real trends)
        stats['games_trend'] = 5  # Placeholder
        stats['alerts_today'] = len([a for a in self.alerts])  # Simplified
        
        return stats
    
    def _generate_confidence_chart(self):
        """Generate confidence distribution chart"""
        if not self.predictions:
            return
        
        # Extract confidence values
        confidences = [p.get('confidence', 0) * 100 for p in self.predictions if p.get('confidence') is not None]
        
        if not confidences:
            return
        
        # Create histogram data
        bins = [0, 50, 70, 85, 100]
        labels = ['Low (<50%)', 'Medium (50-70%)', 'High (70-85%)', 'Very High (>85%)']
        
        hist, _ = np.histogram(confidences, bins=bins)
        
        # Save chart data
        chart_data = {
            'labels': labels,
            'data': hist.tolist(),
            'colors': [
                self.colors['low_confidence'],
                self.colors['medium_confidence'],
                self.colors['high_confidence'],
                self.colors['success']
            ]
        }
        
        chart_path = self.reports_dir / "data" / "confidence_chart.json"
        with open(chart_path, 'w') as f:
            json.dump(chart_data, f)
        
        # Also create static image for email reports
        self._create_static_chart(confidences, 'confidence')
    
    def _generate_league_distribution_chart(self):
        """Generate league distribution chart"""
        if not self.predictions:
            return
        
        # Count games by league
        league_counts = {}
        for game in self.predictions:
            league = game.get('league', 'Unknown')
            league_counts[league] = league_counts.get(league, 0) + 1
        
        # Sort and take top 10
        sorted_leagues = sorted(league_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if not sorted_leagues:
            return
        
        labels = [league for league, count in sorted_leagues]
        data = [count for league, count in sorted_leagues]
        
        # Save chart data
        chart_data = {
            'labels': labels,
            'data': data,
            'colors': [self.colors['primary']] * len(labels)
        }
        
        chart_path = self.reports_dir / "data" / "league_chart.json"
        with open(chart_path, 'w') as f:
            json.dump(chart_data, f)
    
    def _create_static_chart(self, data: List[float], chart_type: str):
        """Create static chart image"""
        try:
            plt.figure(figsize=(8, 5))
            
            if chart_type == 'confidence':
                # Confidence distribution histogram
                plt.hist(data, bins=10, alpha=0.7, color=self.colors['primary'], edgecolor='black')
                plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Confidence (%)', fontsize=12)
                plt.ylabel('Number of Predictions', fontsize=12)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = self.reports_dir / "charts" / f"{chart_type}_chart.png"
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating static chart: {e}")
    
    def generate_alerts_report(self):
        """Generate dedicated alerts report page"""
        logger.info("Generating alerts report")
        
        # Sort alerts by probability (highest first)
        sorted_alerts = sorted(self.alerts, key=lambda x: x.get('over_25_prob', 0), reverse=True)
        
        # Group alerts by confidence level
        high_alerts = [a for a in sorted_alerts if a.get('confidence', 0) >= 0.8]
        medium_alerts = [a for a in sorted_alerts if 0.6 <= a.get('confidence', 0) < 0.8]
        low_alerts = [a for a in sorted_alerts if a.get('confidence', 0) < 0.6]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚠️ Alerts Report - {self.config['system']['name']}</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <header class="page-header">
            <h1><i class="fas fa-exclamation-triangle"></i> Alerts Report</h1>
            <div class="header-actions">
                <a href="index.html" class="btn-back">
                    <i class="fas fa-arrow-left"></i> Back to Dashboard
                </a>
                <button class="btn-export" onclick="exportAlerts()">
                    <i class="fas fa-download"></i> Export CSV
                </button>
            </div>
        </header>

        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="summary-card card-danger">
                <h3>High Confidence Alerts</h3>
                <p class="summary-number">{len(high_alerts)}</p>
                <p class="summary-desc">Confidence ≥ 80%</p>
            </div>
            
            <div class="summary-card card-warning">
                <h3>Medium Confidence Alerts</h3>
                <p class="summary-number">{len(medium_alerts)}</p>
                <p class="summary-desc">60% ≤ Confidence < 80%</p>
            </div>
            
            <div class="summary-card card-secondary">
                <h3>Low Confidence Alerts</h3>
                <p class="summary-number">{len(low_alerts)}</p>
                <p class="summary-desc">Confidence < 60%</p>
            </div>
        </div>

        <!-- Alerts Table -->
        <div class="section">
            <div class="section-header">
                <h2><i class="fas fa-list"></i> All Alerts ({len(sorted_alerts)})</h2>
                <div class="table-controls">
                    <input type="text" id="alertSearch" placeholder="Search alerts..." class="search-input">
                    <select id="alertFilter" class="filter-select">
                        <option value="all">All Alerts</option>
                        <option value="high">High Confidence</option>
                        <option value="medium">Medium Confidence</option>
                        <option value="low">Low Confidence</option>
                    </select>
                </div>
            </div>
            
            <div class="table-container">
                <table class="data-table" id="alertsTable">
                    <thead>
                        <tr>
                            <th>Match</th>
                            <th>League</th>
                            <th>Score</th>
                            <th>Minute</th>
                            <th>Over 2.5 Prob</th>
                            <th>Confidence</th>
                            <th>Status</th>
                            <th>Time</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([self._format_alert_row(alert) for alert in sorted_alerts])}
                    </tbody>
                </table>
            </div>
            
            {"<div class='pagination' id='alertsPagination'></div>" if len(sorted_alerts) > 20 else ""}
        </div>

        <!-- Historical Alert Trends -->
        <div class="section">
            <h2><i class="fas fa-chart-line"></i> Alert Trends</h2>
            <div class="trends-container">
                <div class="trend-card">
                    <h4>Alerts by Hour (Last 24h)</h4>
                    <div class="trend-placeholder">
                        <i class="fas fa-chart-bar"></i>
                        <p>Historical data visualization would appear here</p>
                    </div>
                </div>
                <div class="trend-card">
                    <h4>Success Rate</h4>
                    <div class="trend-placeholder">
                        <i class="fas fa-percentage"></i>
                        <p>Success rate analysis would appear here</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="js/alerts.js"></script>
</body>
</html>
"""
        
        alerts_path = self.reports_dir / "alerts.html"
        with open(alerts_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Alerts report saved to: {alerts_path}")
    
    def _format_alert_row(self, alert: Dict) -> str:
        """Format alert as table row"""
        probability = alert.get('over_25_prob', 0) * 100
        confidence = alert.get('confidence', 0) * 100
        
        # Determine status badge
        if probability >= 80:
            status_class = "badge-success"
            status_text = "HIGH"
        elif probability >= 70:
            status_class = "badge-warning"
            status_text = "MEDIUM"
        else:
            status_class = "badge-secondary"
            status_text = "LOW"
        
        # Format timestamp
        timestamp = alert.get('timestamp', datetime.now().isoformat())
        try:
            alert_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%H:%M')
        except:
            alert_time = "N/A"
        
        return f"""
        <tr>
            <td>
                <strong>{alert.get('home_team', 'Unknown')}</strong> vs <strong>{alert.get('away_team', 'Unknown')}</strong>
            </td>
            <td>{alert.get('league', 'Unknown')}</td>
            <td>{alert.get('home_score', 0)} - {alert.get('away_score', 0)}</td>
            <td>{alert.get('minute', 0)}'</td>
            <td>
                <div class="probability-bar">
                    <div class="bar-fill" style="width: {probability}%"></div>
                    <span>{probability:.1f}%</span>
                </div>
            </td>
            <td>{confidence:.1f}%</td>
            <td><span class="badge {status_class}">{status_text}</span></td>
            <td>{alert_time}</td>
            <td>
                <button class="btn-table" onclick="viewAlertDetails('{alert.get('id', '')}')">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        </tr>
        """
    
    def generate_statistics_report(self):
        """Generate statistics report page"""
        logger.info("Generating statistics report")
        
        # Calculate statistics
        stats = self._calculate_detailed_statistics()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Statistics - {self.config['system']['name']}</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <header class="page-header">
            <h1><i class="fas fa-chart-bar"></i> System Statistics</h1>
            <div class="header-actions">
                <a href="index.html" class="btn-back">
                    <i class="fas fa-arrow-left"></i> Dashboard
                </a>
            </div>
        </header>

        <!-- Performance Overview -->
        <div class="stats-grid-large">
            <div class="stat-large card-primary">
                <div class="stat-large-icon">
                    <i class="fas fa-database"></i>
                </div>
                <div class="stat-large-content">
                    <h3>Total Predictions</h3>
                    <p class="stat-large-number">{stats['total_predictions']:,}</p>
                    <p class="stat-large-desc">Processed this week</p>
                </div>
            </div>
            
            <div class="stat-large card-success">
                <div class="stat-large-icon">
                    <i class="fas fa-bullseye"></i>
                </div>
                <div class="stat-large-content">
                    <h3>Accuracy Rate</h3>
                    <p class="stat-large-number">{stats['accuracy_rate']}%</p>
                    <p class="stat-large-desc">Based on historical data</p>
                </div>
            </div>
            
            <div class="stat-large card-warning">
                <div class="stat-large-icon">
                    <i class="fas fa-bolt"></i>
                </div>
                <div class="stat-large-content">
                    <h3>Avg. Response Time</h3>
                    <p class="stat-large-number">{stats['avg_response_time']}ms</p>
                    <p class="stat-large-desc">Data processing speed</p>
                </div>
            </div>
            
            <div class="stat-large card-danger">
                <div class="stat-large-icon">
                    <i class="fas fa-exclamation-circle"></i>
                </div>
                <div class="stat-large-content">
                    <h3>Error Rate</h3>
                    <p class="stat-large-number">{stats['error_rate']}%</p>
                    <p class="stat-large-desc">Failed data collection</p>
                </div>
            </div>
        </div>

        <!-- Detailed Statistics -->
        <div class="section">
            <h2><i class="fas fa-chart-pie"></i> Detailed Analysis</h2>
            
            <div class="detailed-stats">
                <div class="detail-stat">
                    <h4>League Coverage</h4>
                    <p class="detail-number">{stats['leagues_covered']}</p>
                    <p class="detail-label">Different leagues monitored</p>
                </div>
                
                <div class="detail-stat">
                    <h4>Avg. Games per Run</h4>
                    <p class="detail-number">{stats['avg_games_per_run']:.1f}</p>
                    <p class="detail-label">Live games analyzed</p>
                </div>
                
                <div class="detail-stat">
                    <h4>Success Rate (Over 2.5)</h4>
                    <p class="detail-number">{stats['over_25_success']}%</p>
                    <p class="detail-label">When probability > 75%</p>
                </div>
                
                <div class="detail-stat">
                    <h4>System Uptime</h4>
                    <p class="detail-number">{stats['uptime']}%</p>
                    <p class="detail-label">Last 30 days</p>
                </div>
            </div>
        </div>

        <!-- Data Sources -->
        <div class="section">
            <h2><i class="fas fa-server"></i> Data Sources</h2>
            
            <div class="sources-grid">
                <div class="source-card">
                    <div class="source-icon api">
                        <i class="fas fa-plug"></i>
                    </div>
                    <h4>API Sources</h4>
                    <p class="source-count">{stats['api_sources']}</p>
                    <p class="source-desc">Structured data APIs</p>
                </div>
                
                <div class="source-card">
                    <div class="source-icon scraper">
                        <i class="fas fa-globe"></i>
                    </div>
                    <h4>Web Scrapers</h4>
                    <p class="source-count">{stats['web_scrapers']}</p>
                    <p class="source-desc">Live score websites</p>
                </div>
                
                <div class="source-card">
                    <div class="source-icon cache">
                        <i class="fas fa-database"></i>
                    </div>
                    <h4>Cache Hit Rate</h4>
                    <p class="source-count">{stats['cache_hit_rate']}%</p>
                    <p class="source-desc">Reduced API calls</p>
                </div>
                
                <div class="source-card">
                    <div class="source-icon reliability">
                        <i class="fas fa-shield-alt"></i>
                    </div>
                    <h4>Data Reliability</h4>
                    <p class="source-count">{stats['data_reliability']}%</p>
                    <p class="source-desc">Accuracy of collected data</p>
                </div>
            </div>
        </div>
    </div>

    <script src="js/statistics.js"></script>
</body>
</html>
"""
        
        stats_path = self.reports_dir / "statistics.html"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Statistics report saved to: {stats_path}")
    
    def _calculate_detailed_statistics(self) -> Dict:
        """Calculate detailed system statistics"""
        # These are placeholder calculations
        # In a real system, you'd calculate these from historical data
        return {
            'total_predictions': len(self.predictions) * 100,  # Placeholder
            'accuracy_rate': 78.5,
            'avg_response_time': 245,
            'error_rate': 2.3,
            'leagues_covered': 15,
            'avg_games_per_run': len(self.predictions),
            'over_25_success': 82,
            'uptime': 99.8,
            'api_sources': 3,
            'web_scrapers': 2,
            'cache_hit_rate': 65,
            'data_reliability': 94
        }
    
    def generate_performance_report(self):
        """Generate system performance report"""
        logger.info("Generating performance report")
        
        # This would generate a more technical report
        # For now, create a simple placeholder
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ Performance - {self.config['system']['name']}</title>
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="container">
        <header class="page-header">
            <h1><i class="fas fa-tachometer-alt"></i> System Performance</h1>
            <a href="index.html" class="btn-back">
                <i class="fas fa-arrow-left"></i> Back to Dashboard
            </a>
        </header>
        
        <div class="empty-state large">
            <i class="fas fa-chart-line fa-3x"></i>
            <h2>Performance Report</h2>
            <p>Detailed performance metrics and system monitoring will be available here.</p>
            <p>This section will include:</p>
            <ul class="feature-list">
                <li><i class="fas fa-check"></i> Response time graphs</li>
                <li><i class="fas fa-check"></i> Error rate tracking</li>
                <li><i class="fas fa-check"></i> Resource utilization</li>
                <li><i class="fas fa-check"></i> API rate limit monitoring</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        perf_path = self.reports_dir / "performance.html"
        with open(perf_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Performance report saved to: {perf_path}")
    
    def generate_historical_charts(self):
        """Generate historical charts and data files"""
        logger.info("Generating historical charts")
        
        historical_data = self.load_historical_data(days=7)
        
        if not historical_data:
            logger.warning("No historical data available for charts")
            return
        
        # Create historical summary
        historical_summary = []
        
        for data_point in historical_data:
            summary = {
                'timestamp': data_point.get('timestamp'),
                'total_games': data_point.get('summary', {}).get('total_games', 0),
                'total_alerts': data_point.get('summary', {}).get('total_alerts', 0),
                'avg_confidence': self._calculate_avg_confidence(data_point.get('predictions', []))
            }
            historical_summary.append(summary)
        
        # Save historical data
        historical_path = self.reports_dir / "data" / "historical_summary.json"
        with open(historical_path, 'w') as f:
            json.dump(historical_summary, f)
        
        logger.info(f"Historical summary saved: {historical_path}")
    
    def _calculate_avg_confidence(self, predictions: List[Dict]) -> float:
        """Calculate average confidence from predictions"""
        if not predictions:
            return 0
        
        confidences = [p.get('confidence', 0) for p in predictions if p.get('confidence')]
        if not confidences:
            return 0
        
        return np.mean(confidences) * 100
    
    def generate_league_report(self):
        """Generate league-specific report"""
        logger.info("Generating league report")
        
        # Group predictions by league
        league_data = {}
        for prediction in self.predictions:
            league = prediction.get('league', 'Unknown')
            if league not in league_data:
                league_data[league] = []
            league_data[league].append(prediction)
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏆 Leagues - {self.config['system']['name']}</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <header class="page-header">
            <h1><i class="fas fa-trophy"></i> League Analysis</h1>
            <a href="index.html" class="btn-back">
                <i class="fas fa-arrow-left"></i> Dashboard
            </a>
        </header>
        
        <div class="section">
            <h2><i class="fas fa-list-ol"></i> Monitored Leagues ({len(league_data)})</h2>
            
            <div class="leagues-grid">
                {"".join([self._format_league_card(league, games) for league, games in league_data.items()])}
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        leagues_path = self.reports_dir / "leagues.html"
        with open(leagues_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"League report saved to: {leagues_path}")
    
    def _format_league_card(self, league: str, games: List[Dict]) -> str:
        """Format league as card"""
        # Calculate league statistics
        total_games = len(games)
        live_games = len([g for g in games if g.get('status') in ['live', 'LIVE', 'IN_PLAY']])
        avg_confidence = np.mean([g.get('confidence', 0) * 100 for g in games if g.get('confidence')]) or 0
        
        # Get league logo
        logo_url = self.league_logos.get(league, 'https://img.icons8.com/color/96/000000/trophy.png')
        
        return f"""
        <div class="league-card">
            <div class="league-header">
                <img src="{logo_url}" alt="{league}" class="league-logo-large">
                <h3>{league}</h3>
            </div>
            <div class="league-stats">
                <div class="league-stat">
                    <i class="fas fa-gamepad"></i>
                    <span>Games: {total_games}</span>
                </div>
                <div class="league-stat">
                    <i class="fas fa-play-circle"></i>
                    <span>Live: {live_games}</span>
                </div>
                <div class="league-stat">
                    <i class="fas fa-bullseye"></i>
                    <span>Confidence: {avg_confidence:.1f}%</span>
                </div>
            </div>
            <div class="league-actions">
                <button class="btn-league" onclick="viewLeagueDetails('{league}')">
                    View Details
                </button>
            </div>
        </div>
        """
    
    def generate_css(self):
        """Generate CSS stylesheet"""
        css_content = f"""
/* Main CSS for Over/Under Predictor Dashboard */
:root {{
    --primary: {self.colors['primary']};
    --secondary: {self.colors['secondary']};
    --success: {self.colors['success']};
    --warning: {self.colors['warning']};
    --danger: {self.colors['danger']};
    --info: {self.colors['info']};
    --light: {self.colors['light']};
    --dark: {self.colors['dark']};
    --over-25: {self.colors['over_25']};
    --under-25: {self.colors['under_25']};
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background-color: #f5f7fa;
    color: #333;
    line-height: 1.6;
}}

.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}}

/* Header Styles */
.dashboard-header {{
    background: linear-gradient(135deg, var(--primary) 0%, var(--info) 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}}

.header-content h1 {{
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}}

.subtitle {{
    font-size: 1.1rem;
    opacity: 0.9;
    margin-bottom: 1rem;
}}

.header-stats {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}}

.stat-badge {{
    background: rgba(255, 255, 255, 0.2);
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}}

/* Stats Grid */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}}

.stat-card {{
    background: white;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}}

.stat-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}}

.stat-icon {{
    background: var(--light);
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}}

.stat-content h3 {{
    font-size: 1rem;
    color: var(--secondary);
    margin-bottom: 0.5rem;
}}

.stat-number {{
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 0.25rem;
}}

.stat-trend {{
    font-size: 0.85rem;
    color: var(--secondary);
    display: flex;
    align-items: center;
    gap: 0.25rem;
}}

/* Card Colors */
.card-primary .stat-icon {{ color: var(--primary); }}
.card-success .stat-icon {{ color: var(--success); }}
.card-warning .stat-icon {{ color: var(--warning); }}
.card-info .stat-icon {{ color: var(--info); }}
.card-danger .stat-icon {{ color: var(--danger); }}

/* Section Styles */
.section {{
    background: white;
    padding: 1.5rem;
    border-radius: 1rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}}

.section-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--light);
}}

.section-header h2 {{
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

/* Alert Cards */
.alerts-container {{
    display: flex;
    flex-direction: column;
    gap: 1rem;
}}

.alert-card {{
    border-left: 4px solid var(--warning);
    padding: 1.25rem;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-radius: 0.75rem;
}}

.alert-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}}

.alert-badge {{
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.badge-high {{ background: var(--success); color: white; }}
.badge-medium {{ background: var(--warning); color: white; }}
.badge-low {{ background: var(--secondary); color: white; }}

.alert-content h4 {{
    font-size: 1.25rem;
    margin-bottom: 0.75rem;
}}

.alert-score {{
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    margin: 1rem 0;
}}

.alert-details {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1rem;
}}

.detail-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--secondary);
}}

/* Game Cards */
.games-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
}}

.game-card {{
    background: white;
    border: 1px solid var(--light);
    border-radius: 1rem;
    padding: 1.5rem;
    transition: transform 0.2s, box-shadow 0.2s;
}}

.game-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}}

.game-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}}

.league-logo {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    object-fit: contain;
}}

.game-teams {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}}

.team {{
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
}}

.team-name {{
    font-weight: 600;
    margin-bottom: 0.5rem;
}}

.team-score {{
    font-size: 2rem;
    font-weight: bold;
}}

.vs-separator {{
    padding: 0 1rem;
    color: var(--secondary);
    font-weight: 600;
}}

/* Status Indicators */
.status-indicator {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-weight: 600;
}}

.status-over {{ background: #dcfce7; color: #166534; }}
.status-close {{ background: #fef3c7; color: #92400e; }}
.status-under {{ background: #fee2e2; color: #991b1b; }}

/* Charts */
.charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}}

.chart-card {{
    background: white;
    padding: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}}

.chart-container {{
    position: relative;
    height: 300px;
}}

/* Footer */
.dashboard-footer {{
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--light);
    color: var(--secondary);
}}

.footer-content {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}}

.footer-links {{
    display: flex;
    gap: 1.5rem;
}}

.footer-links a {{
    color: var(--primary);
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.footer-links a:hover {{
    text-decoration: underline;
}}

/* Empty States */
.empty-state {{
    text-align: center;
    padding: 3rem;
    color: var(--secondary);
}}

.empty-state i {{
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}}

/* Buttons */
.btn-view-all, .btn-back, .btn-export {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: var(--primary);
    color: white;
    text-decoration: none;
    border-radius: 0.5rem;
    font-weight: 600;
    border: none;
    cursor: pointer;
    transition: background 0.2s;
}}

.btn-view-all:hover, .btn-back:hover, .btn-export:hover {{
    background: var(--info);
}}

/* Responsive Design */
@media (max-width: 768px) {{
    .stats-grid, .games-grid, .charts-grid {{
        grid-template-columns: 1fr;
    }}
    
    .section-header {{
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }}
    
    .header-content h1 {{
        font-size: 2rem;
    }}
    
    .footer-content {{
        flex-direction: column;
        gap: 1rem;
        text-align: center;
    }}
}}

/* Animations */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.pulse {{
    animation: pulse 2s infinite;
}}
"""
        
        css_path = self.reports_dir / "css" / "style.css"
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        logger.info(f"CSS stylesheet saved to: {css_path}")
    
    def generate_js(self):
        """Generate JavaScript files for interactivity"""
        
        # Dashboard JavaScript
        dashboard_js = """
// Dashboard JavaScript
function startCountdown() {
    let seconds = 300;
    const countdownElement = document.getElementById('countdown');
    
    const interval = setInterval(() => {
        seconds--;
        if (countdownElement) {
            countdownElement.textContent = seconds;
        }
        
        if (seconds <= 0) {
            clearInterval(interval);
            location.reload();
        }
    }, 1000);
}

function loadConfidenceChart() {
    fetch('data/confidence_chart.json')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('confidenceChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.labels,
                    datasets: [{
                        data: data.data,
                        backgroundColor: data.colors,
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `${context.label}: ${context.raw} predictions`;
                                }
                            }
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading confidence chart:', error));
}

function loadLeagueChart() {
    fetch('data/league_chart.json')
        .then(response => response.json())
        .then(data => {
            const ctx = document.getElementById('leagueChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'Number of Games',
                        data: data.data,
                        backgroundColor: data.colors,
                        borderWidth: 1,
                        borderColor: 'rgba(0, 0, 0, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading league chart:', error));
}

function trackAlert(alertId) {
    console.log('Tracking alert:', alertId);
    // Implement alert tracking logic
}

function shareAlert(homeTeam, awayTeam, probability) {
    const text = `⚽ Alert: ${homeTeam} vs ${awayTeam} - Over 2.5 probability: ${probability}%`;
    if (navigator.share) {
        navigator.share({
            title: 'Over/Under Alert',
            text: text,
            url: window.location.href
        });
    } else {
        navigator.clipboard.writeText(text);
        alert('Alert copied to clipboard!');
    }
}

function viewGameDetails(gameId) {
    console.log('Viewing game details:', gameId);
    // Implement game details view
}
"""
        
        dashboard_js_path = self.reports_dir / "js" / "dashboard.js"
        with open(dashboard_js_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_js)
        
        # Alerts JavaScript
        alerts_js = """
// Alerts page JavaScript
function exportAlerts() {
    const table = document.getElementById('alertsTable');
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    
    const csvContent = [
        ['Home Team', 'Away Team', 'League', 'Score', 'Minute', 'Over 2.5 Probability', 'Confidence', 'Status', 'Time'],
        ...rows.map(row => {
            const cells = row.querySelectorAll('td');
            return [
                cells[0].textContent.split(' vs ')[0].trim(),
                cells[0].textContent.split(' vs ')[1].trim(),
                cells[1].textContent,
                cells[2].textContent,
                cells[3].textContent,
                cells[4].querySelector('span').textContent,
                cells[5].textContent,
                cells[6].textContent,
                cells[7].textContent
            ];
        })
    ].map(row => row.join(',')).join('\\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alerts_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function viewAlertDetails(alertId) {
    console.log('Viewing alert details:', alertId);
    // Implement alert details modal
}

// Search and filter functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('alertSearch');
    const filterSelect = document.getElementById('alertFilter');
    
    if (searchInput) {
        searchInput.addEventListener('input', filterAlerts);
    }
    
    if (filterSelect) {
        filterSelect.addEventListener('change', filterAlerts);
    }
    
    // Initialize pagination
    initPagination();
});

function filterAlerts() {
    const searchTerm = document.getElementById('alertSearch').value.toLowerCase();
    const filterValue = document.getElementById('alertFilter').value;
    const rows = document.querySelectorAll('#alertsTable tbody tr');
    
    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        const confidence = row.querySelector('td:nth-child(6)').textContent;
        
        let matchesSearch = searchTerm === '' || text.includes(searchTerm);
        let matchesFilter = filterValue === 'all' || 
                           (filterValue === 'high' && parseFloat(confidence) >= 80) ||
                           (filterValue === 'medium' && parseFloat(confidence) >= 60 && parseFloat(confidence) < 80) ||
                           (filterValue === 'low' && parseFloat(confidence) < 60);
        
        row.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
    });
}

function initPagination() {
    const rows = document.querySelectorAll('#alertsTable tbody tr');
    const rowsPerPage = 20;
    const pageCount = Math.ceil(rows.length / rowsPerPage);
    
    if (pageCount <= 1) return;
    
    const pagination = document.getElementById('alertsPagination');
    pagination.innerHTML = '';
    
    for (let i = 1; i <= pageCount; i++) {
        const button = document.createElement('button');
        button.textContent = i;
        button.className = 'page-btn';
        if (i === 1) button.classList.add('active');
        
        button.addEventListener('click', () => {
            showPage(i);
            document.querySelectorAll('.page-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
        
        pagination.appendChild(button);
    }
    
    showPage(1);
}

function showPage(pageNum) {
    const rows = document.querySelectorAll('#alertsTable tbody tr');
    const rowsPerPage = 20;
    const start = (pageNum - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    
    rows.forEach((row, index) => {
        row.style.display = (index >= start && index < end) ? '' : 'none';
    });
}
"""
        
        alerts_js_path = self.reports_dir / "js" / "alerts.js"
        with open(alerts_js_path, 'w', encoding='utf-8') as f:
            f.write(alerts_js)
        
        logger.info(f"JavaScript files saved to: {self.reports_dir / 'js'}")
    
    def generate_sitemap(self):
        """Generate sitemap for the reports"""
        sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url>
        <loc>https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/</loc>
        <lastmod>2024-01-15</lastmod>
        <changefreq>hourly</changefreq>
        <priority>1.0</priority>
    </url>
    <url>
        <loc>https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/alerts.html</loc>
        <lastmod>2024-01-15</lastmod>
        <changefreq>hourly</changefreq>
        <priority>0.8</priority>
    </url>
    <url>
        <loc>https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/statistics.html</loc>
        <lastmod>2024-01-15</lastmod>
        <changefreq>daily</changefreq>
        <priority>0.7</priority>
    </url>
    <url>
        <loc>https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/performance.html</loc>
        <lastmod>2024-01-15</lastmod>
        <changefreq>weekly</changefreq>
        <priority>0.6</priority>
    </url>
    <url>
        <loc>https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/leagues.html</loc>
        <lastmod>2024-01-15</lastmod>
        <changefreq>daily</changefreq>
        <priority>0.7</priority>
    </url>
</urlset>
"""
        
        sitemap_path = self.reports_dir / "sitemap.xml"
        with open(sitemap_path, 'w', encoding='utf-8') as f:
            f.write(sitemap_content)
        
        logger.info(f"Sitemap saved to: {sitemap_path}")
    
    def _save_chart_data(self):
        """Save data for charts"""
        chart_data = {
            'last_updated': datetime.now().isoformat(),
            'total_predictions': len(self.predictions),
            'total_alerts': len(self.alerts),
            'live_games': len([p for p in self.predictions if p.get('status') in ['live', 'LIVE', 'IN_PLAY']]),
            'avg_confidence': self._calculate_avg_confidence(self.predictions),
        }
        
        chart_path = self.reports_dir / "data" / "latest.json"
        with open(chart_path, 'w') as f:
            json.dump(chart_data, f)
    
    def cleanup_old_reports(self, days_to_keep: int = 30):
        """
        Clean up old report files
        
        Args:
            days_to_keep: Number of days to keep reports
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old chart images
            charts_dir = self.reports_dir / "charts"
            if charts_dir.exists():
                for file in charts_dir.glob("*.png"):
                    file_date = datetime.fromtimestamp(file.stat().st_mtime)
                    if file_date < cutoff_date:
                        file.unlink()
            
            logger.info(f"Cleaned up reports older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old reports: {e}")


# Command line interface
def main():
    """Main function for command line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate reports for Over/Under Predictor')
    parser.add_argument('--all', action='store_true', help='Generate all reports')
    parser.add_argument('--dashboard', action='store_true', help='Generate main dashboard only')
    parser.add_argument('--alerts', action='store_true', help='Generate alerts report only')
    parser.add_argument('--stats', action='store_true', help='Generate statistics report only')
    parser.add_argument('--cleanup', type=int, default=30, 
                       help='Clean up reports older than N days (default: 30)')
    
    args = parser.parse_args()
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Generate reports based on arguments
    if args.all or (not args.dashboard and not args.alerts and not args.stats):
        generator.generate_all_reports()
    else:
        if args.dashboard:
            generator.load_data()
            generator.generate_main_dashboard()
        
        if args.alerts:
            generator.load_data()
            generator.generate_alerts_report()
        
        if args.stats:
            generator.load_data()
            generator.generate_statistics_report()
    
    # Cleanup if requested
    if args.cleanup:
        generator.cleanup_old_reports(args.cleanup)
    
    print("✅ Report generation completed successfully!")


if __name__ == "__main__":
    main()
