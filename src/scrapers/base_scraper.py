"""
base_scraper.py - Base class for all scrapers in Over/Under Predictor system
Now includes structured API clients alongside web scrapers.
"""
import abc
import asyncio
import aiohttp
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
import os
from enum import Enum
from urllib.parse import urlparse

# Import configuration
try:
    from config import get_scraper_config
except ImportError:
    # Fallback for testing
    from ...config import get_scraper_config


class DataSourceType(Enum):
    """Type of data source"""
    WEB_SCRAPER = "web_scraper"
    STRUCTURED_API = "structured_api"
    OPEN_API = "open_api"


@dataclass
class ScrapedGame:
    """Standardized game data structure"""
    id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    minute: int
    status: str  # 'live', 'halftime', 'finished', 'scheduled'
    league: str
    country: str
    timestamp: datetime
    source: str
    source_type: DataSourceType
    metadata: Dict[str, Any] = None
    odds_over_25: Optional[float] = None
    odds_under_25: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def total_goals(self) -> int:
        """Calculate total goals"""
        return self.home_score + self.away_score
    
    @property
    def is_live(self) -> bool:
        """Check if game is currently live"""
        return self.status in ['LIVE', 'IN_PLAY', '1H', '2H', 'HT', 'live', 'halftime']
    
    @property
    def game_key(self) -> str:
        """Create unique key for deduplication"""
        return f"{self.home_team.lower()}_{self.away_team.lower()}_{self.timestamp.strftime('%Y%m%d')}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'home_score': self.home_score,
            'away_score': self.away_score,
            'minute': self.minute,
            'status': self.status,
            'league': self.league,
            'country': self.country,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'source_type': self.source_type.value,
            'total_goals': self.total_goals,
            'is_live': self.is_live,
            'odds_over_25': self.odds_over_25,
            'odds_under_25': self.odds_under_25,
            'metadata': self.metadata
        }


class BaseScraper(abc.ABC):
    """
    Abstract base class for all data collectors
    Supports both web scraping and structured APIs
    """
    
    def __init__(self, name: str, base_url: str = None, source_type: DataSourceType = DataSourceType.WEB_SCRAPER):
        """
        Initialize base data collector
        
        Args:
            name: Collector name (e.g., 'flashscore', 'football_data_api')
            base_url: Base URL for the source
            source_type: Type of data source
        """
        self.name = name
        self.base_url = base_url
        self.source_type = source_type
        self.config = get_scraper_config()
        
        # Setup logging
        self.logger = logging.getLogger(f"scraper.{name}")
        self.setup_logging()
        
        # API Configuration - YOUR KEYS GO HERE
        self.api_keys = {
            'football_data': 'e460c4df45fc41fe8fca16623ee3f733',  # Your football-data.org key
            'api_football': '43b03752fdae250ddf40f592d5490388',   # Your api-football.com key
            'thesportsdb': '123'  # Your thesportsdb.com key (example)
        }
        
        # API Endpoints
        self.api_endpoints = {
            'football_data': {
                'base': 'https://api.football-data.org/v4',
                'matches': '/matches',
                'live': '/matches?status=LIVE',
                'competitions': '/competitions'
            },
            'api_football': {
                'base': 'https://v3.football.api-sports.io',
                'live': '/fixtures?live=all',
                'fixtures': '/fixtures',
                'odds': '/odds'
            },
            'thesportsdb': {
                'base': 'https://www.thesportsdb.com/api/v1/json',
                'livescores': '/livescore.php',
                'events': '/eventsseason.php',
                'latest': '/eventslast.php'
            }
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.request_limit = 100  # Per hour
        
        # Cache setup
        self.cache_enabled = True
        self.cache_duration = 300  # 5 minutes
        self._cache = {}
        
        # User agents for web scrapers
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # Statistics
        self.stats = {
            'requests': 0,
            'games_found': 0,
            'errors': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'start_time': datetime.now()
        }
    
    def setup_logging(self):
        """Setup scraper-specific logging"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abc.abstractmethod
    async def fetch_live_games(self) -> List[ScrapedGame]:
        """
        Abstract method to fetch live games
        Must be implemented by each collector
        """
        pass
    
    # === NEW API-SPECIFIC METHODS ===
    
    async def fetch_football_data_api(self, endpoint: str = 'live') -> Optional[Dict]:
        """
        Fetch data from football-data.org API
        
        Args:
            endpoint: API endpoint to call
            
        Returns:
            JSON response or None if failed
        """
        api_config = self.api_endpoints['football_data']
        url = f"{api_config['base']}{api_config.get(endpoint, endpoint)}"
        
        headers = {
            'X-Auth-Token': self.api_keys['football_data'],
            'Accept': 'application/json'
        }
        
        return await self._make_api_request(url, headers, 'football_data')
    
    async def fetch_api_football(self, endpoint: str = 'live', params: Dict = None) -> Optional[Dict]:
        """
        Fetch data from api-football.com API
        
        Args:
            endpoint: API endpoint to call
            params: Additional query parameters
            
        Returns:
            JSON response or None if failed
        """
        api_config = self.api_endpoints['api_football']
        url = f"{api_config['base']}{api_config.get(endpoint, endpoint)}"
        
        headers = {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': self.api_keys['api_football']
        }
        
        return await self._make_api_request(url, headers, 'api_football', params)
    
    async def fetch_thesportsdb(self, endpoint: str = 'livescores', params: Dict = None) -> Optional[Dict]:
        """
        Fetch data from TheSportsDB.com API
        
        Args:
            endpoint: API endpoint to call
            params: Additional query parameters
            
        Returns:
            JSON response or None if failed
        """
        api_config = self.api_endpoints['thesportsdb']
        url = f"{api_config['base']}{api_config.get(endpoint, endpoint)}"
        
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(params)}"
        
        return await self._make_api_request(url, {}, 'thesportsdb')
    
    async def _make_api_request(self, url: str, headers: Dict, api_name: str, 
                              params: Dict = None, max_retries: int = 3) -> Optional[Dict]:
        """
        Make API request with rate limiting and error handling
        
        Args:
            url: API endpoint URL
            headers: Request headers
            api_name: Name of API for logging
            params: Query parameters
            max_retries: Maximum retry attempts
            
        Returns:
            JSON response or None
        """
        cache_key = f"api_{api_name}_{url}"
        
        # Check cache
        if self.cache_enabled and cache_key in self._cache:
            cache_data = self._cache[cache_key]
            if time.time() - cache_data['timestamp'] < self.cache_duration:
                self.stats['cache_hits'] += 1
                self.logger.debug(f"API cache hit for: {api_name}")
                return cache_data['data']
        
        # Rate limiting
        await self._respect_rate_limit()
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Calling {api_name} API: {url} (attempt {attempt + 1})")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params, 
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        
                        self.stats['requests'] += 1
                        self.stats['api_calls'] += 1
                        self.last_request_time = time.time()
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            # Cache successful response
                            if self.cache_enabled:
                                self._cache[cache_key] = {
                                    'data': data,
                                    'timestamp': time.time()
                                }
                            
                            self.logger.debug(f"Successfully fetched from {api_name}")
                            return data
                            
                        elif response.status == 429:  # Rate limited
                            retry_after = int(response.headers.get('Retry-After', 60))
                            self.logger.warning(f"{api_name} rate limited. Waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        elif response.status == 403:  # Forbidden (invalid key)
                            self.logger.error(f"{api_name} API key rejected. Status: {response.status}")
                            return None
                            
                        else:
                            self.logger.warning(f"{api_name} API error: HTTP {response.status}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                
            except aiohttp.ClientError as e:
                self.logger.error(f"{api_name} network error: {e}")
                self.stats['errors'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                self.logger.error(f"{api_name} API timeout")
                self.stats['errors'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"{api_name} unexpected error: {e}")
                self.stats['errors'] += 1
                break
        
        self.logger.error(f"Failed to fetch from {api_name} after {max_retries} attempts")
        return None
    
    # === API DATA PARSERS ===
    
    def parse_football_data_response(self, data: Dict) -> List[ScrapedGame]:
        """
        Parse football-data.org API response
        
        Args:
            data: API response JSON
            
        Returns:
            List of ScrapedGame objects
        """
        games = []
        
        if not data or 'matches' not in data:
            return games
        
        for match in data['matches']:
            try:
                # Extract basic match info
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                
                # Extract score
                score = match.get('score', {})
                full_time = score.get('fullTime', {})
                home_score = full_time.get('home', 0)
                away_score = full_time.get('away', 0)
                
                # Determine status and minute
                status = match.get('status', 'SCHEDULED')
                minute = 0
                
                if status == 'LIVE':
                    minute = match.get('minute', 0)
                elif status == 'IN_PLAY':
                    minute = match.get('minute', 45)  # Default to 45 for in-play
                elif status == 'PAUSED':
                    minute = 45  # Halftime
                elif status == 'FINISHED':
                    minute = 90
                
                # Create game object
                game = ScrapedGame(
                    id=str(match['id']),
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    minute=minute,
                    status=status,
                    league=match.get('competition', {}).get('name', 'Unknown'),
                    country=match.get('area', {}).get('name', 'Unknown'),
                    timestamp=datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00')),
                    source='football-data.org',
                    source_type=DataSourceType.STRUCTURED_API,
                    metadata={
                        'competition_id': match.get('competition', {}).get('id'),
                        'matchday': match.get('matchday'),
                        'stage': match.get('stage'),
                        'group': match.get('group'),
                        'last_updated': match.get('lastUpdated')
                    }
                )
                
                if self.validate_game(game):
                    games.append(game)
                    
            except KeyError as e:
                self.logger.warning(f"Missing key in football-data response: {e}")
                continue
        
        self.stats['games_found'] += len(games)
        return games
    
    def parse_api_football_response(self, data: Dict) -> List[ScrapedGame]:
        """
        Parse api-football.com API response
        
        Args:
            data: API response JSON
            
        Returns:
            List of ScrapedGame objects
        """
        games = []
        
        if not data or 'response' not in data:
            return games
        
        for fixture in data['response']:
            try:
                fixture_data = fixture.get('fixture', {})
                teams_data = fixture.get('teams', {})
                goals_data = fixture.get('goals', {})
                league_data = fixture.get('league', {})
                
                # Extract status and minute
                status_info = fixture_data.get('status', {})
                status = status_info.get('long', 'Not Started')
                elapsed = status_info.get('elapsed')
                minute = elapsed if elapsed is not None else 0
                
                # Adjust minute for halftime
                if status == 'Halftime':
                    minute = 45
                
                # Create game object
                game = ScrapedGame(
                    id=str(fixture_data['id']),
                    home_team=teams_data['home']['name'],
                    away_team=teams_data['away']['name'],
                    home_score=goals_data.get('home', 0),
                    away_score=goals_data.get('away', 0),
                    minute=minute,
                    status=status,
                    league=league_data.get('name', 'Unknown'),
                    country=league_data.get('country', 'Unknown'),
                    timestamp=datetime.fromisoformat(fixture_data['date'].replace('Z', '+00:00')),
                    source='api-football.com',
                    source_type=DataSourceType.STRUCTURED_API,
                    metadata={
                        'fixture_id': fixture_data['id'],
                        'league_id': league_data.get('id'),
                        'round': fixture.get('league', {}).get('round'),
                        'venue': fixture_data.get('venue', {}).get('name'),
                        'referee': fixture_data.get('referee')
                    }
                )
                
                # Try to extract odds if available
                if 'odds' in fixture:
                    game.odds_over_25 = self._extract_over_25_odds(fixture['odds'])
                
                if self.validate_game(game):
                    games.append(game)
                    
            except KeyError as e:
                self.logger.warning(f"Missing key in api-football response: {e}")
                continue
        
        self.stats['games_found'] += len(games)
        return games
    
    def _extract_over_25_odds(self, odds_data: Dict) -> Optional[float]:
        """Extract Over 2.5 odds from API response"""
        try:
            # Try different possible structures
            if 'bookmakers' in odds_data:
                for bookmaker in odds_data['bookmakers']:
                    if 'bets' in bookmaker:
                        for bet in bookmaker['bets']:
                            if bet.get('name') == 'Total Goals Over/Under':
                                for value in bet.get('values', []):
                                    if value.get('value') == 'Over 2.5':
                                        return float(value.get('odd', 0))
        except Exception:
            pass
        return None
    
    # === ORIGINAL BASE METHODS (UPDATED) ===
    
    async def get_with_retry(self, url: str, max_retries: int = 3, **kwargs) -> Optional[str]:
        """
        HTTP GET request with retry logic and rate limiting
        Used for web scraping (not APIs)
        """
        # Check cache first
        cache_key = f"web_{url}"
        if self.cache_enabled and cache_key in self._cache:
            cache_data = self._cache[cache_key]
            if time.time() - cache_data['timestamp'] < self.cache_duration:
                self.stats['cache_hits'] += 1
                self.logger.debug(f"Web cache hit for: {url}")
                return cache_data['content']
        
        # Rate limiting
        await self._respect_rate_limit()
        
        headers = kwargs.get('headers', {})
        headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        kwargs['headers'] = headers
        
        # Add timeout if not specified
        if 'timeout' not in kwargs:
            kwargs['timeout'] = aiohttp.ClientTimeout(total=self.config.timeout)
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Scraping {url} (attempt {attempt + 1}/{max_retries})")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, **kwargs) as response:
                        self.stats['requests'] += 1
                        self.last_request_time = time.time()
                        
                        if response.status == 200:
                            content = await response.text()
                            
                            # Cache successful response
                            if self.cache_enabled:
                                self._cache[cache_key] = {
                                    'content': content,
                                    'timestamp': time.time()
                                }
                            
                            self.logger.debug(f"Successfully scraped {url}")
                            return content
                        elif response.status == 429:  # Too Many Requests
                            retry_after = int(response.headers.get('Retry-After', 30))
                            self.logger.warning(f"Rate limited, waiting {retry_after} seconds")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            self.logger.warning(f"HTTP {response.status} for {url}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                
            except aiohttp.ClientError as e:
                self.logger.error(f"Network error for {url}: {e}")
                self.stats['errors'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout for {url}")
                self.stats['errors'] += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Unexpected error for {url}: {e}")
                self.stats['errors'] += 1
                break
        
        self.logger.error(f"Failed to scrape {url} after {max_retries} attempts")
        return None
    
    async def _respect_rate_limit(self):
        """Respect rate limiting between requests"""
        delay = self.config.request_delay
        
        # Add jitter to avoid pattern detection
        jitter = random.uniform(0.5, 1.5)
        actual_delay = delay * jitter
        
        time_since_last = time.time() - self.last_request_time
        if time_since_last < actual_delay:
            wait_time = actual_delay - time_since_last
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
    
    def validate_game(self, game: ScrapedGame) -> bool:
        """
        Validate scraped game data
        """
        if not game.home_team or not game.away_team:
            return False
        
        if game.home_team == game.away_team:
            return False
        
        if game.minute < 0 or game.minute > 120:
            return False
        
        if game.home_score < 0 or game.away_score < 0:
            return False
        
        # For live games, check minute range
        if game.is_live and (game.minute < 1 or game.minute > 90):
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get scraper statistics
        """
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'scraper_name': self.name,
            'source_type': self.source_type.value,
            'requests': self.stats['requests'],
            'api_calls': self.stats['api_calls'],
            'games_found': self.stats['games_found'],
            'errors': self.stats['errors'],
            'cache_hits': self.stats['cache_hits'],
            'uptime_seconds': uptime,
            'requests_per_hour': self.stats['requests'] / (uptime / 3600) if uptime > 0 else 0,
            'success_rate': (self.stats['requests'] - self.stats['errors']) / self.stats['requests'] 
                          if self.stats['requests'] > 0 else 0,
            'last_update': datetime.now().isoformat()
        }


# === CONCRETE API SCRAPER CLASSES ===

class FootballDataScraper(BaseScraper):
    """
    Concrete scraper for football-data.org API
    """
    
    def __init__(self):
        super().__init__(
            name="football_data_api",
            base_url="https://api.football-data.org",
            source_type=DataSourceType.STRUCTURED_API
        )
    
    async def fetch_live_games(self) -> List[ScrapedGame]:
        """Fetch live games from football-data.org"""
        self.logger.info("Fetching live games from football-data.org")
        
        data = await self.fetch_football_data_api('live')
        if data:
            return self.parse_football_data_response(data)
        return []


class ApiFootballScraper(BaseScraper):
    """
    Concrete scraper for api-football.com API
    """
    
    def __init__(self):
        super().__init__(
            name="api_football",
            base_url="https://v3.football.api-sports.io",
            source_type=DataSourceType.STRUCTURED_API
        )
    
    async def fetch_live_games(self) -> List[ScrapedGame]:
        """Fetch live games from api-football.com"""
        self.logger.info("Fetching live games from api-football.com")
        
        data = await self.fetch_api_football('live')
        if data:
            return self.parse_api_football_response(data)
        return []


# === HELPER FUNCTIONS ===

def create_game_id(home_team: str, away_team: str, timestamp: datetime) -> str:
    """Create a unique game ID"""
    import hashlib
    
    game_str = f"{home_team.lower()}_{away_team.lower()}_{timestamp.strftime('%Y%m%d_%H%M')}"
    return hashlib.md5(game_str.encode()).hexdigest()[:12]


def filter_live_games(games: List[ScrapedGame], min_minute: int = 1, max_minute: int = 90) -> List[ScrapedGame]:
    """Filter games that are actually live"""
    return [
        game for game in games 
        if game.is_live and min_minute <= game.minute <= max_minute
    ]


def deduplicate_games(games: List[ScrapedGame]) -> List[ScrapedGame]:
    """Remove duplicate games based on game_key"""
    seen = set()
    unique_games = []
    
    for game in games:
        key = game.game_key
        if key not in seen:
            seen.add(key)
            unique_games.append(game)
    
    return unique_games


def merge_games_from_sources(games_list: List[List[ScrapedGame]]) -> List[ScrapedGame]:
    """
    Merge games from multiple sources, preferring structured API data
    """
    all_games = []
    
    for games in games_list:
        all_games.extend(games)
    
    # Group by game key
    games_by_key = {}
    for game in all_games:
        key = game.game_key
        if key not in games_by_key:
            games_by_key[key] = []
        games_by_key[key].append(game)
    
    # Merge games, preferring structured API sources
    merged = []
    for key, game_versions in games_by_key.items():
        # Sort by source reliability: structured API > open API > web scraper
        game_versions.sort(key=lambda g: {
            DataSourceType.STRUCTURED_API: 0,
            DataSourceType.OPEN_API: 1,
            DataSourceType.WEB_SCRAPER: 2
        }[g.source_type])
        
        # Take the most reliable version
        merged.append(game_versions[0])
    
    return merged
