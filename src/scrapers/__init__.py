"""
Scrapers module - Live data collection from various sources
"""
from .base_scraper import BaseScraper
from .flashscore_scraper import FlashScoreScraper
from .scraper_manager import ScraperManager

__all__ = [
    'BaseScraper',
    'FlashScoreScraper',
    'ScraperManager'
]
