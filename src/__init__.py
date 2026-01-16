# src/__init__.py
"""
Over/Under Predictor - src package
"""
__version__ = "0.1.0"

# Optional: expose important names at package level
from .scrapers.scraper_manager import ScraperManager
from .predictor.calculator import ProbabilityCalculator
from .notifier.telegram_client import TelegramNotifier

# Prevents accidental double-import issues in some environments
if __name__ == "__main__":
    print("src package loaded directly — usually not intended")
