"""
Over/Under Predictor - Main Package
Version: 1.0.0
"""
__version__ = "1.0.0"
__author__ = "Predictor System"

# Define what should be available
__all__ = [
    'ScraperManager',
    'FlashScoreScraper',
    'ProbabilityCalculator',
    'Game',
    'Prediction',
    'TelegramNotifier',
    'AlertBuilder'
]

# Lazy imports to avoid circular import issues
import sys
from types import ModuleType

class _LazyLoader(ModuleType):
    """Lazy loader that imports modules only when accessed"""
    
    _module_cache = {}
    
    def __getattr__(self, name):
        if name not in __all__:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        
        # Cache the imported module to avoid re-importing
        if name not in self._module_cache:
            if name in ['ScraperManager', 'FlashScoreScraper']:
                from .scrapers import ScraperManager, FlashScoreScraper
                self._module_cache['ScraperManager'] = ScraperManager
                self._module_cache['FlashScoreScraper'] = FlashScoreScraper
            
            elif name in ['ProbabilityCalculator', 'Game', 'Prediction']:
                from .predictor import ProbabilityCalculator, Game, Prediction
                self._module_cache['ProbabilityCalculator'] = ProbabilityCalculator
                self._module_cache['Game'] = Game
                self._module_cache['Prediction'] = Prediction
            
            elif name == 'TelegramNotifier':
                from .notifier import TelegramNotifier
                self._module_cache['TelegramNotifier'] = TelegramNotifier
            
            elif name == 'AlertBuilder':
                from .notifier import AlertBuilder
                self._module_cache['AlertBuilder'] = AlertBuilder
        
        return self._module_cache[name]
    
    def __dir__(self):
        return __all__

# Replace the module with lazy loader
sys.modules[__name__] = _LazyLoader(__name__)
