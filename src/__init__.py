"""
Over/Under Predictor - Main Package
Version: 1.0.0
"""
__version__ = "1.0.0"
__author__ = "Predictor System"

# Import main components for easy access
from .scrapers import ScraperManager, FlashScoreScraper
from .predictor import ProbabilityCalculator, Game, Prediction
from .notifier import TelegramNotifier, AlertBuilder

# Package-level exports
__all__ = [
    'ScraperManager',
    'FlashScoreScraper',
    'ProbabilityCalculator',
    'Game',
    'Prediction',
    'TelegramNotifier',
    'AlertBuilder'
]
