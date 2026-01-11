"""
scraper_manager.py - Manager for all data scrapers in Over/Under Predictor system
Orchestrates multiple data sources with smart prioritization and fallback logic.
"""
import asyncio
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import logging
from enum import Enum
import os

# Import scrapers and base classes
try:
    from .base_scraper import (
        BaseScraper, ScrapedGame, DataSourceType, 
        deduplicate_games, filter_live_games, merge_games_from_sources,
        FootballDataScraper, ApiFootballScraper
    )
    from .flashscore_scraper import FlashScoreScraper
except ImportError:
    from src.scrapers.base_scraper import (
        BaseScraper, ScrapedGame, DataSourceType,
        deduplicate_games, filter_live_games, merge_games_from_sources,
        FootballDataScraper, ApiFootballScraper
    )
    from src.scrapers.flashscore_scraper import FlashScoreScraper, create_flashscore_scraper


class ScraperPriority(Enum):
    """Priority levels for scrapers"""
    HIGH = 3    # Primary data sources (structured APIs)
    MEDIUM = 2  # Reliable web scrapers
    LOW = 1     # Fallback scrapers
    TEST = 0    # Testing only


class ScraperStatus(Enum):
    """Status of a scraper"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


@dataclass
class ScraperInfo:
    """Information about a registered scraper"""
    name: str
    instance: BaseScraper
    priority: ScraperPriority
    status: ScraperStatus
    last_run: Optional[datetime] = None
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    errors_today: int = 0
    games_found_total: int = 0
    enabled: bool = True
    weight: float = 1.0  # Weight for load balancing
    
    def __post_init__(self):
        if self.last_run is None:
            self.last_run = datetime.now()


class ScraperManager:
    """
    Main manager for all data scrapers
    Handles registration, orchestration, load balancing, and failover
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the scraper manager
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_default_config()
        
        # Setup logging
        self.logger = logging.getLogger("scraper_manager")
        self.setup_logging()
        
        # Scraper registry
        self.scrapers: Dict[str, ScraperInfo] = {}
        
        # Results cache
        self.results_cache: Dict[str, Dict] = {
            'games': [],
            'timestamp': None,
            'scrapers_used': []
        }
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'total_games_found': 0,
            'total_errors': 0,
            'start_time': datetime.now(),
            'last_full_run': None,
            'cache_hits': 0
        }
        
        # Load balancing weights
        self.load_weights = {
            ScraperPriority.HIGH: 1.0,
            ScraperPriority.MEDIUM: 0.7,
            ScraperPriority.LOW: 0.4,
            ScraperPriority.TEST: 0.1
        }
        
        # Health check thresholds
        self.health_thresholds = {
            'min_success_rate': 0.6,  # 60% success rate minimum
            'max_errors_per_hour': 10,  # Max 10 errors per hour
            'max_response_time': 30.0,  # 30 seconds max response time
            'cache_ttl': 300  # 5 minutes cache TTL
        }
        
        # Initialize default scrapers
        self._initialize_default_scrapers()
        
        self.logger.info(f"ScraperManager initialized with {len(self.scrapers)} scrapers")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'max_concurrent_scrapers': 3,
            'request_timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 2,
            'enable_caching': True,
            'cache_duration': 300,  # 5 minutes
            'enable_fallbacks': True,
            'min_games_threshold': 5,
            'league_priorities': {
                'PREMIER LEAGUE': 100,
                'CHAMPIONS LEAGUE': 100,
                'LA LIGA': 95,
                'BUNDESLIGA': 95,
                'SERIE A': 90,
                'LIGUE 1': 90,
                'EREDIVISIE': 80,
                'PRIMEIRA LIGA': 80,
            },
            'blacklisted_leagues': [
                'FRIENDLY',
                'RESERVES',
                'YOUTH',
                'U21',
                'U19',
                'TEST'
            ]
        }
    
    def setup_logging(self):
        """Setup logging for scraper manager"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_default_scrapers(self):
        """Initialize default scrapers based on configuration"""
        self.logger.info("Initializing default scrapers...")
        
        # Register API scrapers (HIGH priority)
        try:
            football_data = FootballDataScraper()
            self.register_scraper(
                name="football_data_api",
                instance=football_data,
                priority=ScraperPriority.HIGH,
                weight=1.2  # Higher weight for structured API
            )
            self.logger.info("Registered FootballData API scraper")
        except Exception as e:
            self.logger.warning(f"Failed to register FootballData scraper: {e}")
        
        try:
            api_football = ApiFootballScraper()
            self.register_scraper(
                name="api_football",
                instance=api_football,
                priority=ScraperPriority.HIGH,
                weight=1.0
            )
            self.logger.info("Registered ApiFootball API scraper")
        except Exception as e:
            self.logger.warning(f"Failed to register ApiFootball scraper: {e}")
        
        # Register web scrapers (MEDIUM priority)
        try:
            flashscore = create_flashscore_scraper()
            self.register_scraper(
                name="flashscore",
                instance=flashscore,
                priority=ScraperPriority.MEDIUM,
                weight=0.8
            )
            self.logger.info("Registered FlashScore scraper")
        except Exception as e:
            self.logger.warning(f"Failed to register FlashScore scraper: {e}")
    
    def register_scraper(self, name: str, instance: BaseScraper, 
                        priority: ScraperPriority = ScraperPriority.MEDIUM,
                        weight: float = 1.0,
                        enabled: bool = True):
        """
        Register a new scraper
        
        Args:
            name: Unique name for the scraper
            instance: BaseScraper instance
            priority: Scraper priority level
            weight: Load balancing weight
            enabled: Whether scraper is enabled
        """
        if name in self.scrapers:
            self.logger.warning(f"Scraper '{name}' already registered, updating")
        
        scraper_info = ScraperInfo(
            name=name,
            instance=instance,
            priority=priority,
            status=ScraperStatus.ACTIVE,
            weight=weight,
            enabled=enabled
        )
        
        self.scrapers[name] = scraper_info
        self.logger.debug(f"Registered scraper: {name} (priority: {priority}, weight: {weight})")
    
    def unregister_scraper(self, name: str):
        """
        Unregister a scraper
        
        Args:
            name: Name of scraper to unregister
        """
        if name in self.scrapers:
            del self.scrapers[name]
            self.logger.info(f"Unregistered scraper: {name}")
        else:
            self.logger.warning(f"Scraper '{name}' not found")
    
    def enable_scraper(self, name: str, enabled: bool = True):
        """
        Enable or disable a scraper
        
        Args:
            name: Name of scraper
            enabled: Whether to enable the scraper
        """
        if name in self.scrapers:
            self.scrapers[name].enabled = enabled
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"{status} scraper: {name}")
        else:
            self.logger.warning(f"Scraper '{name}' not found")
    
    def update_scraper_status(self, name: str, status: ScraperStatus):
        """
        Update scraper status
        
        Args:
            name: Name of scraper
            status: New status
        """
        if name in self.scrapers:
            old_status = self.scrapers[name].status
            self.scrapers[name].status = status
            self.scrapers[name].last_run = datetime.now()
            
            if old_status != status:
                self.logger.info(f"Updated scraper '{name}' status: {old_status} -> {status}")
        else:
            self.logger.warning(f"Scraper '{name}' not found")
    
    async def run_scraper(self, scraper_info: ScraperInfo) -> Tuple[List[ScrapedGame], Dict[str, Any]]:
        """
        Run a single scraper and collect results
        
        Args:
            scraper_info: Scraper information
            
        Returns:
            Tuple of (games list, performance metrics)
        """
        scraper_name = scraper_info.name
        scraper = scraper_info.instance
        
        self.logger.debug(f"Running scraper: {scraper_name}")
        
        start_time = time.time()
        metrics = {
            'success': False,
            'games_found': 0,
            'response_time': 0,
            'error': None,
            'cache_used': False
        }
        
        try:
            # Check cache first
            cache_key = f"{scraper_name}_{datetime.now().strftime('%Y%m%d_%H')}"
            if self.config['enable_caching']:
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    metrics['cache_used'] = True
                    metrics['response_time'] = time.time() - start_time
                    metrics['success'] = True
                    metrics['games_found'] = len(cached_result)
                    self.stats['cache_hits'] += 1
                    self.logger.debug(f"Using cached results for {scraper_name}")
                    return cached_result, metrics
            
            # Run the scraper with timeout
            timeout = self.config['request_timeout']
            
            try:
                games = await asyncio.wait_for(
                    scraper.fetch_live_games(),
                    timeout=timeout
                )
                
                response_time = time.time() - start_time
                
                # Update scraper statistics
                scraper_info.success_rate = self._calculate_success_rate(
                    scraper_info.success_rate, 
                    True, 
                    scraper_info.games_found_total
                )
                scraper_info.avg_response_time = self._calculate_avg_response_time(
                    scraper_info.avg_response_time,
                    response_time,
                    scraper_info.games_found_total
                )
                scraper_info.games_found_total += len(games)
                scraper_info.last_run = datetime.now()
                
                metrics.update({
                    'success': True,
                    'games_found': len(games),
                    'response_time': response_time
                })
                
                # Cache the results
                if self.config['enable_caching']:
                    self._cache_result(cache_key, games)
                
                self.logger.debug(f"Scraper {scraper_name} found {len(games)} games in {response_time:.2f}s")
                
                return games, metrics
                
            except asyncio.TimeoutError:
                error_msg = f"Scraper {scraper_name} timed out after {timeout}s"
                metrics['error'] = error_msg
                self.logger.warning(error_msg)
                
            except Exception as e:
                error_msg = f"Scraper {scraper_name} error: {str(e)}"
                metrics['error'] = error_msg
                self.logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"Unexpected error running scraper {scraper_name}: {str(e)}"
            metrics['error'] = error_msg
            self.logger.error(error_msg)
        
        # If we reach here, scraper failed
        response_time = time.time() - start_time
        
        scraper_info.success_rate = self._calculate_success_rate(
            scraper_info.success_rate, 
            False, 
            scraper_info.games_found_total
        )
        scraper_info.errors_today += 1
        scraper_info.last_run = datetime.now()
        
        metrics['response_time'] = response_time
        
        self.update_scraper_status(scraper_name, ScraperStatus.ERROR)
        
        return [], metrics
    
    def _calculate_success_rate(self, current_rate: float, success: bool, total_runs: int) -> float:
        """Calculate moving average success rate"""
        if total_runs == 0:
            return 1.0 if success else 0.0
        
        alpha = 0.3  # Smoothing factor
        new_rate = 1.0 if success else 0.0
        return (alpha * new_rate) + ((1 - alpha) * current_rate)
    
    def _calculate_avg_response_time(self, current_avg: float, new_time: float, total_runs: int) -> float:
        """Calculate moving average response time"""
        if total_runs == 0:
            return new_time
        
        alpha = 0.3  # Smoothing factor
        return (alpha * new_time) + ((1 - alpha) * current_avg)
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[ScrapedGame]]:
        """Get cached results if available and fresh"""
        cache_file = os.path.join('storage', 'cache', f'{cache_key}.json')
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            cache_age = (datetime.now() - cache_time).seconds
            
            if cache_age <= self.health_thresholds['cache_ttl']:
                # Convert cached data back to ScrapedGame objects
                games = []
                for game_data in cache_data['games']:
                    game = ScrapedGame(
                        id=game_data['id'],
                        home_team=game_data['home_team'],
                        away_team=game_data['away_team'],
                        home_score=game_data['home_score'],
                        away_score=game_data['away_score'],
                        minute=game_data['minute'],
                        status=game_data['status'],
                        league=game_data['league'],
                        country=game_data['country'],
                        timestamp=datetime.fromisoformat(game_data['timestamp']),
                        source=game_data['source'],
                        source_type=DataSourceType(game_data['source_type']),
                        metadata=game_data['metadata'],
                        odds_over_25=game_data.get('odds_over_25'),
                        odds_under_25=game_data.get('odds_under_25')
                    )
                    games.append(game)
                
                return games
        
        except Exception as e:
            self.logger.debug(f"Cache read error for {cache_key}: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, games: List[ScrapedGame]):
        """Cache scraper results"""
        try:
            os.makedirs(os.path.join('storage', 'cache'), exist_ok=True)
            cache_file = os.path.join('storage', 'cache', f'{cache_key}.json')
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'games': [game.to_dict() for game in games],
                'count': len(games)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug(f"Cached {len(games)} games for {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache results: {e}")
    
    def _get_scrapers_for_run(self) -> List[ScraperInfo]:
        """
        Select scrapers to run based on priority, health, and load balancing
        
        Returns:
            List of scraper info objects to run
        """
        # Filter enabled scrapers
        enabled_scrapers = [
            info for info in self.scrapers.values() 
            if info.enabled and info.status != ScraperStatus.MAINTENANCE
        ]
        
        if not enabled_scrapers:
            self.logger.warning("No enabled scrapers available")
            return []
        
        # Sort by priority and weight
        sorted_scrapers = sorted(
            enabled_scrapers,
            key=lambda x: (
                self.load_weights.get(x.priority, 1.0) * x.weight,
                x.success_rate,
                -x.avg_response_time
            ),
            reverse=True
        )
        
        # Limit concurrent scrapers
        max_concurrent = self.config['max_concurrent_scrapers']
        selected_scrapers = sorted_scrapers[:max_concurrent]
        
        # Log selection
        selected_names = [s.name for s in selected_scrapers]
        self.logger.debug(f"Selected scrapers for run: {selected_names}")
        
        return selected_scrapers
    
    async def fetch_all_games(self) -> Dict[str, Any]:
        """
        Main method: Fetch games from all scrapers
        
        Returns:
            Dictionary with games and metadata
        """
        self.logger.info("Starting fetch from all scrapers")
        self.stats['total_runs'] += 1
        
        start_time = time.time()
        
        # Get scrapers to run
        scrapers_to_run = self._get_scrapers_for_run()
        
        if not scrapers_to_run:
            self.logger.error("No scrapers available to run")
            self.stats['total_errors'] += 1
            return self._create_empty_result()
        
        # Run scrapers concurrently
        tasks = []
        for scraper_info in scrapers_to_run:
            task = self.run_scraper(scraper_info)
            tasks.append(task)
        
        # Wait for all scrapers to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_games = []
        scraper_results = {}
        successful_scrapers = []
        
        for i, (scraper_info, result) in enumerate(zip(scrapers_to_run, results)):
            scraper_name = scraper_info.name
            
            if isinstance(result, Exception):
                self.logger.error(f"Scraper {scraper_name} failed with exception: {result}")
                scraper_results[scraper_name] = {
                    'success': False,
                    'error': str(result),
                    'games_found': 0
                }
                self.stats['total_errors'] += 1
                continue
            
            games, metrics = result
            scraper_results[scraper_name] = metrics
            
            if metrics['success']:
                successful_scrapers.append(scraper_name)
                all_games.extend(games)
                
                # Update scraper status based on performance
                if metrics.get('games_found', 0) > 0:
                    self.update_scraper_status(scraper_name, ScraperStatus.ACTIVE)
                else:
                    self.logger.warning(f"Scraper {scraper_name} succeeded but found 0 games")
            else:
                self.stats['total_errors'] += 1
        
        # Merge and deduplicate games from different sources
        merged_games = deduplicate_games(all_games)
        
        # Filter for live games
        live_games = filter_live_games(
            merged_games,
            min_minute=1,
            max_minute=90
        )
        
        # Apply league filtering
        filtered_games = self._filter_games_by_league(live_games)
        
        # Sort games by priority
        sorted_games = self._sort_games_by_priority(filtered_games)
        
        # Calculate performance metrics
        total_response_time = time.time() - start_time
        success_rate = len(successful_scrapers) / len(scrapers_to_run) if scrapers_to_run else 0
        
        # Update statistics
        self.stats['total_games_found'] += len(sorted_games)
        if len(successful_scrapers) > 0:
            self.stats['successful_runs'] += 1
        self.stats['last_full_run'] = datetime.now()
        
        # Update cache with results
        self.results_cache = {
            'games': sorted_games,
            'timestamp': datetime.now(),
            'scrapers_used': successful_scrapers,
            'total_games': len(sorted_games)
        }
        
        # Log summary
        self.logger.info(
            f"Fetch completed: {len(sorted_games)} games from {len(successful_scrapers)}/"
            f"{len(scrapers_to_run)} scrapers in {total_response_time:.2f}s "
            f"(success rate: {success_rate:.1%})"
        )
        
        return {
            'games': sorted_games,
            'metadata': {
                'total_games': len(sorted_games),
                'live_games': len(live_games),
                'scrapers_used': successful_scrapers,
                'total_scrapers': len(scrapers_to_run),
                'success_rate': success_rate,
                'response_time': total_response_time,
                'timestamp': datetime.now().isoformat(),
                'scraper_results': scraper_results
            },
            'raw_games_by_scraper': {
                scraper_name: games 
                for scraper_name, (games, _) in zip(
                    [s.name for s in scrapers_to_run], 
                    results
                ) 
                if not isinstance(games, Exception)
            }
        }
    
    def _filter_games_by_league(self, games: List[ScrapedGame]) -> List[ScrapedGame]:
        """
        Filter games based on league priorities and blacklists
        
        Args:
            games: List of games to filter
            
        Returns:
            Filtered list of games
        """
        if not self.config.get('league_priorities') and not self.config.get('blacklisted_leagues'):
            return games
        
        filtered_games = []
        
        for game in games:
            league_upper = game.league.upper()
            
            # Check if league is blacklisted
            blacklisted = False
            for blacklisted_league in self.config.get('blacklisted_leagues', []):
                if blacklisted_league.upper() in league_upper:
                    blacklisted = True
                    break
            
            if blacklisted:
                self.logger.debug(f"Filtered out blacklisted league: {game.league}")
                continue
            
            filtered_games.append(game)
        
        return filtered_games
    
    def _sort_games_by_priority(self, games: List[ScrapedGame]) -> List[ScrapedGame]:
        """
        Sort games by league priority and game status
        
        Args:
            games: List of games to sort
            
        Returns:
            Sorted list of games
        """
        if not games:
            return []
        
        def get_game_priority(game: ScrapedGame) -> Tuple[int, int, int]:
            """
            Calculate priority score for a game
            Returns: (league_priority, status_priority, minute)
            """
            # League priority
            league_priority = 50  # Default
            league_upper = game.league.upper()
            
            for league_name, priority in self.config.get('league_priorities', {}).items():
                if league_name.upper() in league_upper:
                    league_priority = priority
                    break
            
            # Status priority (higher for live games)
            status_priority = {
                'live': 100,
                'halftime': 80,
                'finished': 0,
                'scheduled': 0
            }.get(game.status, 0)
            
            # Minute priority (later in game = higher priority for predictions)
            minute_priority = game.minute if game.status == 'live' else 0
            
            return (-league_priority, -status_priority, -minute_priority)
        
        return sorted(games, key=get_game_priority)
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            'games': [],
            'metadata': {
                'total_games': 0,
                'live_games': 0,
                'scrapers_used': [],
                'total_scrapers': 0,
                'success_rate': 0,
                'response_time': 0,
                'timestamp': datetime.now().isoformat(),
                'scraper_results': {}
            },
            'raw_games_by_scraper': {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all scrapers
        
        Returns:
            Health status dictionary
        """
        self.logger.info("Performing health check on all scrapers")
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'total_scrapers': len(self.scrapers),
            'enabled_scrapers': len([s for s in self.scrapers.values() if s.enabled]),
            'active_scrapers': len([s for s in self.scrapers.values() if s.status == ScraperStatus.ACTIVE]),
            'scrapers': {},
            'overall_health': 'healthy',
            'issues': []
        }
        
        # Check each scraper
        for scraper_name, scraper_info in self.scrapers.items():
            scraper_health = {
                'enabled': scraper_info.enabled,
                'status': scraper_info.status.value,
                'success_rate': scraper_info.success_rate,
                'avg_response_time': scraper_info.avg_response_time,
                'errors_today': scraper_info.errors_today,
                'games_found_total': scraper_info.games_found_total,
                'last_run': scraper_info.last_run.isoformat() if scraper_info.last_run else None,
                'health': 'healthy',
                'issues': []
            }
            
            # Check success rate
            if scraper_info.success_rate < self.health_thresholds['min_success_rate']:
                scraper_health['health'] = 'warning'
                issue = f"Low success rate: {scraper_info.success_rate:.1%}"
                scraper_health['issues'].append(issue)
                health_status['issues'].append(f"{scraper_name}: {issue}")
            
            # Check response time
            if scraper_info.avg_response_time > self.health_thresholds['max_response_time']:
                scraper_health['health'] = 'warning'
                issue = f"High response time: {scraper_info.avg_response_time:.2f}s"
                scraper_health['issues'].append(issue)
                health_status['issues'].append(f"{scraper_name}: {issue}")
            
            # Check error rate
            if scraper_info.errors_today > self.health_thresholds['max_errors_per_hour']:
                scraper_health['health'] = 'error'
                issue = f"High error count: {scraper_info.errors_today}"
                scraper_health['issues'].append(issue)
                health_status['issues'].append(f"{scraper_name}: {issue}")
                health_status['overall_health'] = 'unhealthy'
            
            health_status['scrapers'][scraper_name] = scraper_health
        
        # Log health status
        if health_status['issues']:
            self.logger.warning(f"Health check found issues: {health_status['issues']}")
        else:
            self.logger.info("Health check passed: all scrapers healthy")
        
        return health_status
    
    def get_scraper_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all scrapers
        
        Returns:
            Statistics dictionary
        """
        scraper_stats = {}
        
        for scraper_name, scraper_info in self.scrapers.items():
            scraper_stats[scraper_name] = {
                'priority': scraper_info.priority.value,
                'status': scraper_info.status.value,
                'enabled': scraper_info.enabled,
                'success_rate': scraper_info.success_rate,
                'avg_response_time': scraper_info.avg_response_time,
                'errors_today': scraper_info.errors_today,
                'games_found_total': scraper_info.games_found_total,
                'last_run': scraper_info.last_run.isoformat() if scraper_info.last_run else None,
                'weight': scraper_info.weight
            }
        
        return {
            'scrapers': scraper_stats,
            'manager_stats': self.stats,
            'total_scrapers': len(self.scrapers),
            'enabled_scrapers': len([s for s in self.scrapers.values() if s.enabled]),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_stats_to_file(self, filepath: str = None):
        """
        Save statistics to file
        
        Args:
            filepath: Path to save statistics file
        """
        if filepath is None:
            os.makedirs('storage/logs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'storage/logs/scraper_stats_{timestamp}.json'
        
        try:
            stats = self.get_scraper_stats()
            
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"Statistics saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")
    
    def get_last_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the last fetched results from cache
        
        Returns:
            Last results or None
        """
        if not self.results_cache['games']:
            return None
        
        cache_age = None
        if self.results_cache['timestamp']:
            cache_age = (datetime.now() - self.results_cache['timestamp']).seconds
        
        return {
            'games': self.results_cache['games'],
            'timestamp': self.results_cache['timestamp'].isoformat() if self.results_cache['timestamp'] else None,
            'cache_age_seconds': cache_age,
            'scrapers_used': self.results_cache.get('scrapers_used', []),
            'total_games': len(self.results_cache['games'])
        }
    
    def reset_scraper_stats(self, scraper_name: str = None):
        """
        Reset statistics for a scraper or all scrapers
        
        Args:
            scraper_name: Name of scraper to reset (None for all)
        """
        if scraper_name:
            if scraper_name in self.scrapers:
                self.scrapers[scraper_name].success_rate = 0.0
                self.scrapers[scraper_name].avg_response_time = 0.0
                self.scrapers[scraper_name].errors_today = 0
                self.scrapers[scraper_name].games_found_total = 0
                self.logger.info(f"Reset statistics for scraper: {scraper_name}")
            else:
                self.logger.warning(f"Scraper '{scraper_name}' not found")
        else:
            for name, scraper_info in self.scrapers.items():
                scraper_info.success_rate = 0.0
                scraper_info.avg_response_time = 0.0
                scraper_info.errors_today = 0
                scraper_info.games_found_total = 0
            self.logger.info("Reset statistics for all scrapers")


# Factory function for easy instantiation
def create_scraper_manager(config: Optional[Dict] = None) -> ScraperManager:
    """
    Create and initialize a scraper manager
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ScraperManager instance
    """
    return ScraperManager(config)


# Example usage and testing
async def test_scraper_manager():
    """Test the scraper manager"""
    print("Testing Scraper Manager...")
    
    # Create manager
    manager = create_scraper_manager()
    
    # Show registered scrapers
    print(f"\nRegistered scrapers: {len(manager.scrapers)}")
    for name, info in manager.scrapers.items():
        print(f"  - {name}: {info.priority.value} (enabled: {info.enabled})")
    
    # Perform health check
    print("\nPerforming health check...")
    health = await manager.health_check()
    print(f"Overall health: {health['overall_health']}")
    
    # Fetch games
    print("\nFetching games from all scrapers...")
    start_time = time.time()
    results = await manager.fetch_all_games()
    elapsed = time.time() - start_time
    
    games = results['games']
    metadata = results['metadata']
    
    print(f"\nFetch completed in {elapsed:.2f}s")
    print(f"Success rate: {metadata['success_rate']:.1%}")
    print(f"Scrapers used: {len(metadata['scrapers_used'])}/{metadata['total_scrapers']}")
    print(f"Total games found: {len(games)}")
    print(f"Live games: {metadata['live_games']}")
    
    if games:
        print("\nSample games:")
        for i, game in enumerate(games[:5]):  # Show first 5 games
            status_symbol = "⚽" if game.status == 'live' else "⏸️"
            print(f"{i+1}. {status_symbol} {game.home_team} {game.home_score}-{game.away_score} {game.away_team}")
            print(f"   {game.league} - {game.minute}' - Source: {game.source}")
    
    # Show statistics
    print("\nScraper statistics:")
    stats = manager.get_scraper_stats()
    for scraper_name, scraper_stats in stats['scrapers'].items():
        print(f"  {scraper_name}:")
        print(f"    Success rate: {scraper_stats['success_rate']:.1%}")
        print(f"    Avg response: {scraper_stats['avg_response_time']:.2f}s")
        print(f"    Games found: {scraper_stats['games_found_total']}")
        print(f"    Errors today: {scraper_stats['errors_today']}")
    
    return manager, results


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_scraper_manager())
