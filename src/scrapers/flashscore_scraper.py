"""
flashscore_scraper.py - FlashScore scraper for Over/Under Predictor system
Uses web scraping to extract live match data from FlashScore.com
"""
import asyncio
import aiohttp
import re
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse
import logging

# Import base scraper and models
try:
    from .base_scraper import BaseScraper, ScrapedGame, DataSourceType, create_game_id
except ImportError:
    from src.scrapers.base_scraper import BaseScraper, ScrapedGame, DataSourceType, create_game_id


class FlashScoreScraper(BaseScraper):
    """
    FlashScore.com web scraper for live football matches
    Extracts match data using web scraping techniques
    """
    
    def __init__(self):
        """Initialize FlashScore scraper"""
        super().__init__(
            name="flashscore",
            base_url="https://www.flashscore.com",
            source_type=DataSourceType.WEB_SCRAPER
        )
        
        # FlashScore specific configuration
        self.match_detail_urls = {
            'match_summary': '/match/{match_id}/#/match-summary',
            'match_details': '/match/{match_id}/#/match-summary/match-statistics/0',
            'h2h': '/match/{match_id}/#/h2h/overall'
        }
        
        # XHR endpoints discovered from network traffic
        self.api_endpoints = {
            'live_matches': 'https://www.flashscore.com/x/feed/df_scores_1_1',
            'match_details': 'https://www.flashscore.com/x/feed/df_match_detail_1_{match_id}',
            'match_summary': 'https://www.flashscore.com/x/feed/df_match_summary_1_{match_id}'
        }
        
        # Common headers for FlashScore requests
        self.flashscore_headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'DNT': '1',
            'Pragma': 'no-cache',
            'Referer': 'https://www.flashscore.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'X-Fsign': 'SW9D1eZo',  # FlashScore signature
            'X-GeoIP': '1',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        # Match status mapping
        self.status_mapping = {
            '1H': 'live',  # First Half
            '2H': 'live',  # Second Half
            'HT': 'halftime',  # Half Time
            'ET': 'live',  # Extra Time
            'P': 'live',  # Penalties
            'FT': 'finished',  # Full Time
            'AET': 'finished',  # After Extra Time
            'AP': 'finished',  # After Penalties
            'POST': 'finished',  # Postponed
            'INT': 'interrupted',  # Interrupted
            'ABD': 'abandoned',  # Abandoned
            'AWD': 'awarded',  # Awarded
            'WO': 'walkover',  # Walkover
            'LIVE': 'live',  # Live
            'NS': 'scheduled',  # Not Started
            'CNCL': 'cancelled',  # Cancelled
            'TBA': 'scheduled',  # To Be Announced
        }
        
        # League priority (higher priority leagues are scraped first)
        self.priority_leagues = {
            'EPL': 100,  # English Premier League
            'UCL': 100,  # UEFA Champions League
            'UEL': 90,   # UEFA Europa League
            'LA LIGA': 95,  # Spanish La Liga
            'BUNDESLIGA': 95,  # German Bundesliga
            'SERIE A': 90,  # Italian Serie A
            'LIGUE 1': 90,  # French Ligue 1
            'EREDIVISIE': 80,  # Dutch Eredivisie
            'PRIMEIRA LIGA': 80,  # Portuguese Primeira Liga
        }
        
        # Statistics tracking
        self.match_cache = {}
        self.league_cache = {}
        self.last_full_scrape = None
        
        self.logger.info(f"Initialized FlashScore scraper with base URL: {self.base_url}")
    
    async def fetch_live_games(self) -> List[ScrapedGame]:
        """
        Main method to fetch live games from FlashScore
        
        Returns:
            List of ScrapedGame objects
        """
        self.logger.info("Starting FlashScore live games fetch")
        
        try:
            # Method 1: Try XHR API first (more reliable)
            games = await self._fetch_via_xhr_api()
            
            if not games:
                self.logger.warning("XHR API failed, falling back to HTML scraping")
                games = await self._fetch_via_html_scraping()
            
            # Update statistics
            self.stats['games_found'] += len(games)
            self.last_full_scrape = datetime.now()
            
            self.logger.info(f"Successfully fetched {len(games)} live games from FlashScore")
            return games
            
        except Exception as e:
            self.logger.error(f"Error fetching live games from FlashScore: {e}")
            self.stats['errors'] += 1
            return []
    
    async def _fetch_via_xhr_api(self) -> List[ScrapedGame]:
        """
        Fetch live games using FlashScore's XHR API
        This is the preferred method as it returns structured data
        """
        self.logger.debug("Attempting to fetch via XHR API")
        
        try:
            # First, get the main page to get cookies and initial data
            main_page = await self.get_with_retry(
                self.base_url,
                headers=self.flashscore_headers
            )
            
            if not main_page:
                self.logger.warning("Failed to fetch main page for cookies")
                return []
            
            # Extract XHR feed URL parameters from the page
            feed_params = self._extract_feed_params(main_page)
            
            # Build the XHR feed URL
            xhr_url = self._build_xhr_feed_url(feed_params)
            
            if not xhr_url:
                self.logger.warning("Could not build XHR feed URL")
                return []
            
            # Fetch the XHR feed
            xhr_response = await self.get_with_retry(
                xhr_url,
                headers=self.flashscore_headers
            )
            
            if not xhr_response:
                self.logger.warning("XHR feed request failed")
                return []
            
            # Parse the XHR response
            games = self._parse_xhr_response(xhr_response)
            
            # Enrich with additional details for live games
            enriched_games = []
            for game in games:
                if game.status == 'live':
                    try:
                        enriched_game = await self._enrich_game_details(game)
                        enriched_games.append(enriched_game)
                    except Exception as e:
                        self.logger.debug(f"Could not enrich game {game.id}: {e}")
                        enriched_games.append(game)
                else:
                    enriched_games.append(game)
            
            return enriched_games
            
        except Exception as e:
            self.logger.error(f"Error in XHR API fetch: {e}")
            return []
    
    def _extract_feed_params(self, html: str) -> Dict[str, str]:
        """
        Extract XHR feed parameters from HTML page
        
        Args:
            html: HTML content of main page
            
        Returns:
            Dictionary of feed parameters
        """
        params = {}
        
        try:
            # Look for feed configuration in script tags
            feed_patterns = [
                r'window\.fs_scoreboard_config\s*=\s*({[^}]+})',
                r'FEED_URL\s*:\s*["\']([^"\']+)["\']',
                r'feedUrl\s*:\s*["\']([^"\']+)["\']',
                r'x/feed/([^"\']+)["\']',
            ]
            
            for pattern in feed_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    if pattern.startswith('window'):
                        try:
                            config = json.loads(matches[0] + '}')
                            if 'feedUrl' in config:
                                params['feedUrl'] = config['feedUrl']
                        except json.JSONDecodeError:
                            pass
                    else:
                        params['feedUrl'] = matches[0]
                    break
            
            # Extract additional parameters
            param_patterns = {
                'key': r'["\']key["\']\s*:\s*["\']([^"\']+)["\']',
                'secret': r'["\']secret["\']\s*:\s*["\']([^"\']+)["\']',
                'timestamp': r'["\']timestamp["\']\s*:\s*["\']([^"\']+)["\']',
            }
            
            for param_name, pattern in param_patterns.items():
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    params[param_name] = matches[0]
            
            self.logger.debug(f"Extracted feed params: {list(params.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Error extracting feed params: {e}")
        
        return params
    
    def _build_xhr_feed_url(self, params: Dict[str, str]) -> Optional[str]:
        """
        Build XHR feed URL from extracted parameters
        
        Args:
            params: Extracted feed parameters
            
        Returns:
            Complete XHR feed URL or None
        """
        if 'feedUrl' in params:
            feed_url = params['feedUrl']
            if not feed_url.startswith('http'):
                feed_url = urljoin(self.base_url, feed_url)
            return feed_url
        
        # Fallback to known feed URL patterns
        base_feeds = [
            'https://www.flashscore.com/x/feed/df_scores_1_1',
            'https://www.flashscore.com/x/feed/f_1_1_3_en_1',
            'https://www.flashscore.com/x/feed/ss_1_1',
        ]
        
        # Add timestamp to avoid caching
        timestamp = int(time.time())
        
        for feed in base_feeds:
            test_url = f"{feed}?_={timestamp}"
            return test_url
        
        return None
    
    def _parse_xhr_response(self, response_text: str) -> List[ScrapedGame]:
        """
        Parse XHR API response
        
        Args:
            response_text: Raw XHR response text
            
        Returns:
            List of ScrapedGame objects
        """
        games = []
        
        try:
            # FlashScore XHR responses are often in a custom format
            # Try to parse as JSON first
            try:
                data = json.loads(response_text)
                games = self._parse_json_response(data)
            except json.JSONDecodeError:
                # Try parsing as key-value format
                games = self._parse_key_value_response(response_text)
            
        except Exception as e:
            self.logger.error(f"Error parsing XHR response: {e}")
        
        return games
    
    def _parse_json_response(self, data: Dict) -> List[ScrapedGame]:
        """
        Parse JSON formatted response
        
        Args:
            data: JSON response data
            
        Returns:
            List of ScrapedGame objects
        """
        games = []
        
        try:
            # Structure varies, try different possible formats
            if 'events' in data:
                events = data['events']
            elif 'data' in data and 'events' in data['data']:
                events = data['data']['events']
            elif isinstance(data, list):
                events = data
            else:
                events = []
            
            for event in events:
                try:
                    game = self._parse_single_event(event)
                    if game and self.validate_game(game):
                        games.append(game)
                except Exception as e:
                    self.logger.debug(f"Error parsing event: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error in JSON response parsing: {e}")
        
        return games
    
    def _parse_key_value_response(self, response_text: str) -> List[ScrapedGame]:
        """
        Parse key-value formatted response (common in FlashScore)
        
        Args:
            response_text: Key-value response text
            
        Returns:
            List of ScrapedGame objects
        """
        games = []
        
        try:
            # FlashScore often uses format like: AA÷value¬BB÷value¬
            lines = response_text.split('¬')
            event_data = {}
            
            for line in lines:
                if '÷' in line:
                    key, value = line.split('÷', 1)
                    event_data[key] = value
            
            # Group data into events
            events = self._group_key_value_data(event_data)
            
            for event in events:
                try:
                    game = self._parse_single_key_value_event(event)
                    if game and self.validate_game(game):
                        games.append(game)
                except Exception as e:
                    self.logger.debug(f"Error parsing key-value event: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error in key-value parsing: {e}")
        
        return games
    
    def _group_key_value_data(self, data: Dict) -> List[Dict]:
        """
        Group key-value data into individual events
        
        Args:
            data: Flat key-value dictionary
            
        Returns:
            List of event dictionaries
        """
        events = []
        current_event = {}
        
        # Simple grouping logic - FlashScore's actual format is more complex
        for key, value in data.items():
            if key.startswith('AA'):
                if current_event:
                    events.append(current_event)
                    current_event = {}
            
            current_event[key] = value
        
        if current_event:
            events.append(current_event)
        
        return events
    
    def _parse_single_event(self, event: Dict) -> Optional[ScrapedGame]:
        """
        Parse a single event from structured data
        
        Args:
            event: Event dictionary
            
        Returns:
            ScrapedGame object or None
        """
        try:
            # Extract basic information
            home_team = event.get('homeTeam', {}).get('name', '')
            away_team = event.get('awayTeam', {}).get('name', '')
            
            if not home_team or not away_team:
                return None
            
            # Extract score
            score = event.get('score', {})
            home_score = int(score.get('home', 0))
            away_score = int(score.get('away', 0))
            
            # Extract status and minute
            status_info = event.get('status', {})
            status_code = status_info.get('code', 'NS')
            status = self.status_mapping.get(status_code, 'scheduled')
            
            minute = 0
            if status == 'live':
                minute = int(status_info.get('elapsed', 0))
            elif status == 'halftime':
                minute = 45
            elif status == 'finished':
                minute = 90
            
            # Extract league/country
            tournament = event.get('tournament', {})
            league = tournament.get('name', 'Unknown League')
            country = tournament.get('category', {}).get('name', 'Unknown')
            
            # Create timestamp
            start_time = event.get('startTime')
            if start_time:
                try:
                    timestamp = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Create game ID
            game_id = str(event.get('id', create_game_id(home_team, away_team, timestamp)))
            
            # Create ScrapedGame object
            game = ScrapedGame(
                id=game_id,
                home_team=home_team.strip(),
                away_team=away_team.strip(),
                home_score=home_score,
                away_score=away_score,
                minute=minute,
                status=status,
                league=league.strip(),
                country=country.strip(),
                timestamp=timestamp,
                source='flashscore.com',
                source_type=DataSourceType.WEB_SCRAPER,
                metadata={
                    'tournament_id': tournament.get('id'),
                    'status_code': status_code,
                    'round': event.get('round'),
                    'venue': event.get('venue', {}).get('name'),
                    'has_statistics': event.get('hasStatistics', False),
                    'has_events': event.get('hasEvents', False),
                    'has_lineups': event.get('hasLineups', False),
                }
            )
            
            return game
            
        except Exception as e:
            self.logger.debug(f"Error parsing single event: {e}")
            return None
    
    def _parse_single_key_value_event(self, event: Dict) -> Optional[ScrapedGame]:
        """
        Parse a single event from key-value data
        
        Args:
            event: Key-value event dictionary
            
        Returns:
            ScrapedGame object or None
        """
        try:
            # Map FlashScore keys to our fields
            # This is a simplified mapping - actual keys vary
            field_map = {
                'AE': 'home_team',
                'AF': 'away_team',
                'AG': 'home_score',
                'AH': 'away_score',
                'AD': 'minute',
                'AC': 'status_code',
                'B': 'league',
                'C': 'country',
                'W': 'start_time',
                'I': 'game_id',
            }
            
            # Extract fields
            extracted = {}
            for key, field_name in field_map.items():
                if key in event:
                    extracted[field_name] = event[key]
            
            if 'home_team' not in extracted or 'away_team' not in extracted:
                return None
            
            # Parse status
            status_code = extracted.get('status_code', 'NS')
            status = self.status_mapping.get(status_code, 'scheduled')
            
            minute = 0
            if status == 'live':
                try:
                    minute = int(extracted.get('minute', 0))
                except:
                    minute = 0
            elif status == 'halftime':
                minute = 45
            elif status == 'finished':
                minute = 90
            
            # Parse scores
            try:
                home_score = int(extracted.get('home_score', 0))
                away_score = int(extracted.get('away_score', 0))
            except:
                home_score = 0
                away_score = 0
            
            # Parse timestamp
            start_time_str = extracted.get('start_time')
            if start_time_str:
                try:
                    timestamp = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                except:
                    try:
                        timestamp = datetime.fromtimestamp(int(start_time_str))
                    except:
                        timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Create game ID
            game_id = extracted.get('game_id', 
                                  create_game_id(extracted['home_team'], extracted['away_team'], timestamp))
            
            # Create ScrapedGame object
            game = ScrapedGame(
                id=game_id,
                home_team=extracted['home_team'].strip(),
                away_team=extracted['away_team'].strip(),
                home_score=home_score,
                away_score=away_score,
                minute=minute,
                status=status,
                league=extracted.get('league', 'Unknown League').strip(),
                country=extracted.get('country', 'Unknown').strip(),
                timestamp=timestamp,
                source='flashscore.com',
                source_type=DataSourceType.WEB_SCRAPER,
                metadata={
                    'parsed_from': 'key_value',
                    'status_code': status_code,
                }
            )
            
            return game
            
        except Exception as e:
            self.logger.debug(f"Error parsing key-value event: {e}")
            return None
    
    async def _fetch_via_html_scraping(self) -> List[ScrapedGame]:
        """
        Fallback method: Fetch live games by scraping HTML
        
        Returns:
            List of ScrapedGame objects
        """
        self.logger.debug("Using HTML scraping fallback")
        
        try:
            # Fetch the main live scores page
            html = await self.get_with_retry(
                f"{self.base_url}/football/",
                headers={
                    'User-Agent': self.user_agents[0],
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
            )
            
            if not html:
                self.logger.warning("Failed to fetch HTML page")
                return []
            
            # Parse HTML for live games
            games = self._parse_html_for_live_games(html)
            
            return games
            
        except Exception as e:
            self.logger.error(f"Error in HTML scraping: {e}")
            return []
    
    def _parse_html_for_live_games(self, html: str) -> List[ScrapedGame]:
        """
        Parse HTML to extract live games
        
        Args:
            html: HTML content of the page
            
        Returns:
            List of ScrapedGame objects
        """
        games = []
        
        try:
            # Look for live game containers
            # FlashScore uses various class names
            patterns = [
                r'<div[^>]*class="[^"]*event__match[^"]*live[^"]*"[^>]*>.*?</div></div></div>',
                r'<div[^>]*data-testid="[^"]*match-row[^"]*"[^>]*>.*?</div></div>',
                r'<div[^>]*class="[^"]*live[^"]*"[^>]*data-testid="[^"]*match-row[^"]*"[^>]*>',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html, re.DOTALL)
                if matches:
                    for match_html in matches:
                        try:
                            game = self._parse_single_html_match(match_html)
                            if game and self.validate_game(game):
                                games.append(game)
                        except Exception as e:
                            self.logger.debug(f"Error parsing HTML match: {e}")
                            continue
                    break
            
            self.logger.debug(f"Found {len(games)} games via HTML parsing")
            
        except Exception as e:
            self.logger.error(f"Error parsing HTML for live games: {e}")
        
        return games
    
    def _parse_single_html_match(self, match_html: str) -> Optional[ScrapedGame]:
        """
        Parse a single match from HTML
        
        Args:
            match_html: HTML snippet for a single match
            
        Returns:
            ScrapedGame object or None
        """
        try:
            # Extract team names
            home_team_match = re.search(r'class="[^"]*event__participant--home[^"]*"[^>]*>([^<]+)<', match_html)
            away_team_match = re.search(r'class="[^"]*event__participant--away[^"]*"[^>]*>([^<]+)<', match_html)
            
            if not home_team_match or not away_team_match:
                return None
            
            home_team = home_team_match.group(1).strip()
            away_team = away_team_match.group(1).strip()
            
            # Extract score
            score_match = re.search(r'class="[^"]*event__score[^"]*"[^>]*>([^<]+)<', match_html)
            if score_match:
                score_text = score_match.group(1).strip()
                home_score, away_score = self.extract_score(score_text)
            else:
                home_score, away_score = 0, 0
            
            # Extract minute/status
            minute_match = re.search(r'class="[^"]*event__stage[^"]*"[^>]*>([^<]+)<', match_html)
            if minute_match:
                minute_text = minute_match.group(1).strip()
                minute = self.extract_minute(minute_text)
                
                # Determine status from minute text
                if 'HT' in minute_text.upper():
                    status = 'halftime'
                elif 'FT' in minute_text.upper() or 'FIN' in minute_text.upper():
                    status = 'finished'
                elif 'LIVE' in minute_text.upper() or "'" in minute_text:
                    status = 'live'
                else:
                    status = 'scheduled'
            else:
                minute = 0
                status = 'scheduled'
            
            # Extract league information (if available in same snippet)
            league_match = re.search(r'class="[^"]*event__title[^"]*"[^>]*>([^<]+)<', match_html)
            league = league_match.group(1).strip() if league_match else 'Unknown League'
            
            # Create timestamp (current time for live games)
            timestamp = datetime.now()
            
            # Create game ID
            game_id = create_game_id(home_team, away_team, timestamp)
            
            # Create ScrapedGame object
            game = ScrapedGame(
                id=game_id,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                minute=minute,
                status=status,
                league=league,
                country=self._extract_country_from_league(league),
                timestamp=timestamp,
                source='flashscore.com',
                source_type=DataSourceType.WEB_SCRAPER,
                metadata={
                    'parsed_from': 'html',
                    'html_snippet_length': len(match_html),
                }
            )
            
            return game
            
        except Exception as e:
            self.logger.debug(f"Error parsing single HTML match: {e}")
            return None
    
    def _extract_country_from_league(self, league: str) -> str:
        """
        Extract country from league name
        
        Args:
            league: League name
            
        Returns:
            Country name
        """
        # Simple mapping based on common league names
        country_map = {
            'PREMIER LEAGUE': 'England',
            'LA LIGA': 'Spain',
            'BUNDESLIGA': 'Germany',
            'SERIE A': 'Italy',
            'LIGUE 1': 'France',
            'EREDIVISIE': 'Netherlands',
            'PRIMEIRA LIGA': 'Portugal',
            'CHAMPIONS LEAGUE': 'Europe',
            'EUROPA LEAGUE': 'Europe',
            'CONFERENCE LEAGUE': 'Europe',
        }
        
        league_upper = league.upper()
        for key, country in country_map.items():
            if key in league_upper:
                return country
        
        return 'Unknown'
    
    async def _enrich_game_details(self, game: ScrapedGame) -> ScrapedGame:
        """
        Enrich game with additional details from match page
        
        Args:
            game: Basic game object
            
        Returns:
            Enriched game object
        """
        # Check cache first
        cache_key = f"details_{game.id}"
        if cache_key in self.match_cache:
            cached_time = self.match_cache[cache_key]['timestamp']
            if (datetime.now() - cached_time).seconds < 300:  # 5 minutes cache
                return self.match_cache[cache_key]['game']
        
        try:
            # Try to fetch match details page
            match_url = f"{self.base_url}/match/{game.id}/"
            
            html = await self.get_with_retry(
                match_url,
                headers={
                    'User-Agent': self.user_agents[0],
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                }
            )
            
            if html:
                # Parse additional details from match page
                enriched_metadata = await self._parse_match_details(html, game)
                game.metadata.update(enriched_metadata)
            
            # Cache the enriched game
            self.match_cache[cache_key] = {
                'game': game,
                'timestamp': datetime.now()
            }
            
            return game
            
        except Exception as e:
            self.logger.debug(f"Could not enrich game details for {game.id}: {e}")
            return game
    
    async def _parse_match_details(self, html: str, game: ScrapedGame) -> Dict[str, Any]:
        """
        Parse additional details from match page
        
        Args:
            html: Match page HTML
            game: Existing game object
            
        Returns:
            Dictionary of additional metadata
        """
        metadata = {}
        
        try:
            # Extract statistics if available
            stats_patterns = {
                'shots_on_target': r'Shots on target.*?(\d+)\s*-\s*(\d+)',
                'total_shots': r'Total shots.*?(\d+)\s*-\s*(\d+)',
                'possession': r'Possession.*?(\d+)%\s*-\s*(\d+)%',
                'corners': r'Corner kicks.*?(\d+)\s*-\s*(\d+)',
                'fouls': r'Fouls.*?(\d+)\s*-\s*(\d+)',
            }
            
            for stat_name, pattern in stats_patterns.items():
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    metadata[f'{stat_name}_home'] = int(match.group(1))
                    metadata[f'{stat_name}_away'] = int(match.group(2))
            
            # Extract events (goals, cards, substitutions)
            events_section = re.search(r'id="[^"]*events[^"]*"[^>]*>.*?</div>', html, re.DOTALL)
            if events_section:
                events_html = events_section.group(0)
                metadata['events_count'] = events_html.count('event__icon')
            
            # Extract lineups if available
            if 'Lineups' in html or 'line-ups' in html:
                metadata['has_lineups'] = True
            
            # Extract weather/pitch conditions
            if 'weather' in html.lower() or 'condition' in html.lower():
                metadata['has_weather_info'] = True
            
        except Exception as e:
            self.logger.debug(f"Error parsing match details: {e}")
        
        return metadata
    
    async def get_league_priority(self, league_name: str) -> int:
        """
        Get priority score for a league
        
        Args:
            league_name: Name of the league
            
        Returns:
            Priority score (higher = more important)
        """
        if league_name in self.league_cache:
            return self.league_cache[league_name]
        
        # Calculate priority
        priority = 50  # Default
        
        league_upper = league_name.upper()
        for key, score in self.priority_leagues.items():
            if key in league_upper:
                priority = score
                break
        
        # Cache the result
        self.league_cache[league_name] = priority
        
        return priority
    
    async def fetch_specific_leagues(self, league_names: List[str]) -> List[ScrapedGame]:
        """
        Fetch games from specific leagues only
        
        Args:
            league_names: List of league names to fetch
            
        Returns:
            List of ScrapedGame objects
        """
        self.logger.info(f"Fetching specific leagues: {league_names}")
        
        all_games = await self.fetch_live_games()
        
        if not league_names:
            return all_games
        
        filtered_games = []
        for game in all_games:
            game_league_upper = game.league.upper()
            for league in league_names:
                if league.upper() in game_league_upper:
                    filtered_games.append(game)
                    break
        
        self.logger.info(f"Filtered to {len(filtered_games)} games from specified leagues")
        return filtered_games
    
    async def test_connectivity(self) -> bool:
        """
        Test connectivity to FlashScore
        
        Returns:
            True if connection successful
        """
        try:
            test_url = f"{self.base_url}/football/"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, 
                                     headers={'User-Agent': self.user_agents[0]},
                                     timeout=10) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Connectivity test failed: {e}")
            return False


# Factory function for easy instantiation
def create_flashscore_scraper() -> FlashScoreScraper:
    """
    Factory function to create FlashScore scraper
    
    Returns:
        FlashScoreScraper instance
    """
    return FlashScoreScraper()


# Quick test function
async def test_flashscore_scraper():
    """Test the FlashScore scraper"""
    scraper = FlashScoreScraper()
    
    print("Testing FlashScore scraper...")
    print(f"Base URL: {scraper.base_url}")
    
    # Test connectivity
    if await scraper.test_connectivity():
        print("✓ Connectivity test passed")
    else:
        print("✗ Connectivity test failed")
        return
    
    # Fetch live games
    print("\nFetching live games...")
    games = await scraper.fetch_live_games()
    
    print(f"Found {len(games)} live games")
    
    if games:
        print("\nSample games:")
        for i, game in enumerate(games[:3]):  # Show first 3 games
            print(f"{i+1}. {game.home_team} {game.home_score}-{game.away_score} {game.away_team}")
            print(f"   {game.league} - {game.minute}' - Status: {game.status}")
    
    # Show statistics
    stats = scraper.get_stats()
    print(f"\nStatistics:")
    print(f"  Requests: {stats['requests']}")
    print(f"  Games found: {stats['games_found']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Cache hits: {stats['cache_hits']}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_flashscore_scraper())
