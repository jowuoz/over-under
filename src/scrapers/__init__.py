"""
Scrapers module - Live data collection from various sources
"""
# Import only what's safe - avoid circular imports
# Don't import everything at module level

__all__ = [
    'BaseScraper',
    'FlashScoreScraper',
    'ScraperManager'
]

# Use try-except to handle missing imports gracefully
try:
    from .base_scraper import BaseScraper
except ImportError:
    BaseScraper = None

try:
    from .flashscore_scraper import FlashScoreScraper
except ImportError:
    FlashScoreScraper = None

try:
    from .scraper_manager import ScraperManager
except ImportError:
    ScraperManager = None
