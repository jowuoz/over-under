"""
Configuration module for Over/Under Predictor
Provides centralized configuration management
"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dataclasses import dataclass, field

@dataclass
class ScraperConfig:
    """Scraper configuration - UPDATED VERSION"""
    active: list
    request_delay: float  # Changed from int to float
    timeout: int
    user_agent: str
    max_retries: int = 3  # NEW: Add this
    cache_duration: int = 300  # NEW: Add this
    
    def __post_init__(self):
        """Validate configuration"""
        if self.request_delay < 0.5:
            self.request_delay = 1.0  # Minimum delay
        if self.timeout < 5:
            self.timeout = 10  # Minimum timeout

@dataclass
class PredictorConfig:
    """Predictor configuration"""
    thresholds: Dict[str, float]
    min_minute: int
    max_minute: int
    min_confidence: float
    model_weights: Dict[str, float] = field(default_factory=lambda: {   # ← add this
        'time_based': 0.35,
        'statistical': 0.25,
        'momentum': 0.20,
        'league_based': 0.10,
        'odds_based': 0.10,
    })

@dataclass
class NotificationConfig:
    """Notification configuration"""
    telegram_enabled: bool
    min_confidence: float

@dataclass
class SystemConfig:
    """Main system configuration"""
    name: str
    update_interval: int
    timezone: str
    scraper: ScraperConfig
    predictor: PredictorConfig
    notifications: NotificationConfig

class ConfigManager:
    """Manages configuration loading and access"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from YAML file"""
        config_path = os.path.join(os.path.dirname(__file__), 'settings.yaml')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Build configuration objects
        scraper_config = ScraperConfig(
            active=raw_config['scrapers']['active'],
            request_delay=raw_config['scrapers']['request_delay'],
            timeout=raw_config['scrapers']['timeout'],
            user_agent=raw_config['scrapers'].get('user_agent', 'Mozilla/5.0')
        )
        
        predictor_config = PredictorConfig(
            thresholds=raw_config['predictor']['thresholds'],
            min_minute=raw_config['predictor']['min_minute'],
            max_minute=raw_config['predictor']['max_minute'],
            min_confidence=raw_config['predictor']['min_confidence']
        )
        
        notification_config = NotificationConfig(
            telegram_enabled=raw_config['notifications']['telegram']['enabled'],
            min_confidence=raw_config['notifications']['min_confidence']
        )
        
        system_config = SystemConfig(
            name=raw_config['system']['name'],
            update_interval=raw_config['system']['update_interval'],
            timezone=raw_config['system']['timezone'],
            scraper=scraper_config,
            predictor=predictor_config,
            notifications=notification_config
        )
        
        return system_config
    
    def get_config(self) -> SystemConfig:
        """Get the current configuration"""
        return self._config
    
    def reload(self):
        """Reload configuration from file"""
        self._config = self._load_config()
    
    def get_telegram_config(self) -> Optional[Dict[str, str]]:
        """Get Telegram configuration from environment"""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if bot_token and chat_id:
            return {
                'bot_token': bot_token,
                'chat_id': chat_id
            }
        return None

# Global configuration instance
config = ConfigManager().get_config()

# Convenience functions
def get_scraper_config() -> ScraperConfig:
    """Get scraper configuration"""
    return config.scraper

def get_predictor_config() -> PredictorConfig:
    """Get predictor configuration"""
    return config.predictor

def get_notification_config() -> NotificationConfig:
    """Get notification configuration"""
    return config.notifications

def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return config
