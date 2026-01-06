#!/usr/bin/env python3
"""
test_scraper.py - Comprehensive tests for scraper system
Tests all scraper components including API and web scrapers
"""
import asyncio
import sys
import os
import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import scraper components
from src.scrapers.base_scraper import (
    BaseScraper, 
    ScrapedGame, 
    DataSourceType, 
    FootballDataScraper, 
    ApiFootballScraper,
    create_game_id,
    filter_live_games,
    deduplicate_games,
    merge_games_from_sources
)
from src.scrapers.flashscore_scraper import FlashScoreScraper, create_flashscore_scraper
from src.utils.logger import SystemLogger
from config import get_scraper_config


class TestBaseScraper:
    """Test base scraper functionality"""
    
    def setup_method(self):
        """Setup before each test"""
        # Mock config to avoid file loading
        with patch('src.scrapers.base_scraper.get_scraper_config') as mock_config:
            mock_config.return_value = Mock(
                request_delay=1,
                timeout=10,
                active=['flashscore']
            )
            self.scraper = BaseScraper(name="test_scraper", base_url="https://test.com")
    
    def test_initialization(self):
        """Test scraper initialization"""
        assert self.scraper.name == "test_scraper"
        assert self.scraper.base_url == "https://test.com"
        assert self.scraper.source_type == DataSourceType.WEB_SCRAPER
        assert self.scraper.cache_enabled == True
        assert self.scraper.cache_duration == 300
    
    def test_scraped_game_dataclass(self):
        """Test ScrapedGame data structure"""
        timestamp = datetime.now()
        game = ScrapedGame(
            id="test_123",
            home_team="Team A",
            away_team="Team B",
            home_score=2,
            away_score=1,
            minute=45,
            status="live",
            league="Test League",
            country="Test Country",
            timestamp=timestamp,
            source="test_source",
            source_type=DataSourceType.STRUCTURED_API,
            metadata={"key": "value"},
            odds_over_25=1.85,
            odds_under_25=1.95
        )
        
        assert game.id == "test_123"
        assert game.home_team == "Team A"
        assert game.away_team == "Team B"
        assert game.home_score == 2
        assert game.away_score == 1
        assert game.total_goals == 3
        assert game.minute == 45
        assert game.status == "live"
        assert game.is_live == True
        assert game.league == "Test League"
        assert game.country == "Test Country"
        assert game.timestamp == timestamp
        assert game.source == "test_source"
        assert game.source_type == DataSourceType.STRUCTURED_API
        assert game.metadata == {"key": "value"}
        assert game.odds_over_25 == 1.85
        assert game.odds_under_25 == 1.95
    
    def test_game_key_generation(self):
        """Test unique game key generation"""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        game = ScrapedGame(
            id="test_123",
            home_team="Manchester United",
            away_team="Liverpool",
            home_score=1,
            away_score=0,
            minute=45,
            status="live",
            league="Premier League",
            country="England",
            timestamp=timestamp,
            source="test",
            source_type=DataSourceType.STRUCTURED_API
        )
        
        expected_key = "manchester_united_liverpool_20240115"
        assert game.game_key == expected_key.lower()
    
    def test_create_game_id(self):
        """Test game ID creation helper"""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        game_id = create_game_id("Team A", "Team B", timestamp)
        
        assert isinstance(game_id, str)
        assert len(game_id) == 12  # MD5 hash truncated to 12 chars
    
    def test_extract_score(self):
        """Test score extraction from various formats"""
        test_cases = [
            ("2-1", (2, 1)),
            ("2:1", (2, 1)),
            ("2 - 1", (2, 1)),
            ("2–1", (2, 1)),
            ("2—1", (2, 1)),
            ("2 1", (2, 1)),
            ("0-0", (0, 0)),
            ("", (0, 0)),
            (None, (0, 0)),
        ]
        
        for score_text, expected in test_cases:
            result = self.scraper.extract_score(score_text)
            assert result == expected, f"Failed for: {score_text}"
    
    def test_extract_minute(self):
        """Test minute extraction from various formats"""
        test_cases = [
            ("45'", 45),
            ("HT", 45),
            ("HALF", 45),
            ("FT", 90),
            ("FULL", 90),
            ("90+3", 90),
            ("LIVE", 0),
            ("NS", 0),
            ("", 0),
            ("PEN", 0),
        ]
        
        for minute_text, expected in test_cases:
            result = self.scraper.extract_minute(minute_text)
            assert result == expected, f"Failed for: {minute_text}"
    
    def test_normalize_team_name(self):
        """Test team name normalization"""
        test_cases = [
            ("  Manchester United  ", "Manchester United"),
            ("manchester united", "manchester united"),
            ("FC Barcelona", "FC Barcelona"),
            ("", ""),
            (None, ""),
        ]
        
        for name, expected in test_cases:
            result = self.scraper.normalize_team_name(name)
            assert result == expected, f"Failed for: {name}"
    
    def test_validate_game(self):
        """Test game validation"""
        valid_game = ScrapedGame(
            id="test",
            home_team="Team A",
            away_team="Team B",
            home_score=1,
            away_score=0,
            minute=45,
            status="live",
            league="Test",
            country="Test",
            timestamp=datetime.now(),
            source="test",
            source_type=DataSourceType.STRUCTURED_API
        )
        
        # Test valid game
        assert self.scraper.validate_game(valid_game) == True
        
        # Test invalid games
        invalid_games = [
            # Missing team names
            ScrapedGame(
                id="test", home_team="", away_team="Team B",
                home_score=0, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
            # Same team
            ScrapedGame(
                id="test", home_team="Team A", away_team="Team A",
                home_score=0, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
            # Negative minute
            ScrapedGame(
                id="test", home_team="Team A", away_team="Team B",
                home_score=0, away_score=0, minute=-10, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
            # Negative score
            ScrapedGame(
                id="test", home_team="Team A", away_team="Team B",
                home_score=-1, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
        ]
        
        for game in invalid_games:
            assert self.scraper.validate_game(game) == False
    
    def test_filter_live_games(self):
        """Test live game filtering"""
        games = [
            ScrapedGame(
                id="1", home_team="A", away_team="B",
                home_score=0, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
            ScrapedGame(
                id="2", home_team="C", away_team="D",
                home_score=0, away_score=0, minute=0, status="scheduled",
                league="Test", country="Test", timestamp=datetime.now(),
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
            ScrapedGame(
                id="3", home_team="E", away_team="F",
                home_score=0, away_score=0, minute=90, status="finished",
                league="Test", country="Test", timestamp=datetime.now(),
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
        ]
        
        live_games = filter_live_games(games)
        assert len(live_games) == 1
        assert live_games[0].id == "1"
    
    def test_deduplicate_games(self):
        """Test game deduplication"""
        timestamp = datetime.now()
        
        games = [
            ScrapedGame(
                id="1", home_team="Team A", away_team="Team B",
                home_score=1, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=timestamp,
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
            ScrapedGame(
                id="2", home_team="Team A", away_team="Team B",
                home_score=1, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=timestamp,
                source="test2", source_type=DataSourceType.STRUCTURED_API
            ),
            ScrapedGame(
                id="3", home_team="Team C", away_team="Team D",
                home_score=0, away_score=0, minute=30, status="live",
                league="Test", country="Test", timestamp=timestamp,
                source="test", source_type=DataSourceType.STRUCTURED_API
            ),
        ]
        
        unique_games = deduplicate_games(games)
        assert len(unique_games) == 2  # Should have 2 unique games
    
    def test_get_stats(self):
        """Test statistics collection"""
        stats = self.scraper.get_stats()
        
        assert 'scraper_name' in stats
        assert 'requests' in stats
        assert 'games_found' in stats
        assert 'errors' in stats
        assert 'cache_hits' in stats
        assert stats['scraper_name'] == 'test_scraper'


class TestFootballDataScraper:
    """Test football-data.org API scraper"""
    
    def setup_method(self):
        """Setup before each test"""
        self.scraper = FootballDataScraper()
    
    @pytest.mark.asyncio
    async def test_fetch_live_games_success(self):
        """Test successful fetch from football-data.org"""
        mock_response = {
            "matches": [
                {
                    "id": 123456,
                    "homeTeam": {"name": "Manchester United"},
                    "awayTeam": {"name": "Liverpool"},
                    "score": {"fullTime": {"home": 1, "away": 0}},
                    "status": "LIVE",
                    "minute": 45,
                    "competition": {"name": "Premier League"},
                    "area": {"name": "England"},
                    "utcDate": "2024-01-15T14:30:00Z",
                    "lastUpdated": "2024-01-15T14:29:00Z"
                }
            ]
        }
        
        with patch.object(self.scraper, 'fetch_football_data_api') as mock_fetch:
            mock_fetch.return_value = mock_response
            games = await self.scraper.fetch_live_games()
            
            assert len(games) == 1
            assert games[0].home_team == "Manchester United"
            assert games[0].away_team == "Liverpool"
            assert games[0].home_score == 1
            assert games[0].away_score == 0
            assert games[0].minute == 45
            assert games[0].status == "LIVE"
            assert games[0].league == "Premier League"
            assert games[0].country == "England"
            assert games[0].source == "football-data.org"
    
    @pytest.mark.asyncio
    async def test_fetch_live_games_empty(self):
        """Test fetch with empty response"""
        with patch.object(self.scraper, 'fetch_football_data_api') as mock_fetch:
            mock_fetch.return_value = None
            games = await self.scraper.fetch_live_games()
            
            assert len(games) == 0
    
    def test_parse_football_data_response(self):
        """Test football-data.org response parsing"""
        response_data = {
            "matches": [
                {
                    "id": 123456,
                    "homeTeam": {"name": "Team A"},
                    "awayTeam": {"name": "Team B"},
                    "score": {"fullTime": {"home": 2, "away": 1}},
                    "status": "LIVE",
                    "minute": 65,
                    "competition": {"name": "Test League", "id": 2021},
                    "area": {"name": "Test Country"},
                    "utcDate": "2024-01-15T14:30:00Z",
                    "lastUpdated": "2024-01-15T14:29:00Z",
                    "matchday": 21,
                    "stage": "REGULAR_SEASON"
                }
            ]
        }
        
        games = self.scraper.parse_football_data_response(response_data)
        
        assert len(games) == 1
        game = games[0]
        assert game.id == "123456"
        assert game.home_team == "Team A"
        assert game.away_team == "Team B"
        assert game.home_score == 2
        assert game.away_score == 1
        assert game.total_goals == 3
        assert game.minute == 65
        assert game.status == "LIVE"
        assert game.league == "Test League"
        assert game.country == "Test Country"
        assert game.source == "football-data.org"
        assert game.source_type == DataSourceType.STRUCTURED_API
        assert "competition_id" in game.metadata
        assert game.metadata["matchday"] == 21


class TestApiFootballScraper:
    """Test api-football.com API scraper"""
    
    def setup_method(self):
        """Setup before each test"""
        self.scraper = ApiFootballScraper()
    
    @pytest.mark.asyncio
    async def test_fetch_live_games_success(self):
        """Test successful fetch from api-football.com"""
        mock_response = {
            "response": [
                {
                    "fixture": {
                        "id": 592342,
                        "date": "2024-01-15T14:30:00Z",
                        "status": {"long": "First Half", "elapsed": 35}
                    },
                    "teams": {
                        "home": {"name": "Barcelona"},
                        "away": {"name": "Real Madrid"}
                    },
                    "goals": {"home": 2, "away": 0},
                    "league": {
                        "name": "La Liga",
                        "country": "Spain",
                        "id": 140,
                        "round": "Regular Season - 21"
                    }
                }
            ]
        }
        
        with patch.object(self.scraper, 'fetch_api_football') as mock_fetch:
            mock_fetch.return_value = mock_response
            games = await self.scraper.fetch_live_games()
            
            assert len(games) == 1
            assert games[0].home_team == "Barcelona"
            assert games[0].away_team == "Real Madrid"
            assert games[0].home_score == 2
            assert games[0].away_score == 0
            assert games[0].minute == 35
            assert games[0].status == "First Half"
            assert games[0].league == "La Liga"
            assert games[0].country == "Spain"
            assert games[0].source == "api-football.com"
    
    def test_parse_api_football_response(self):
        """Test api-football.com response parsing"""
        response_data = {
            "response": [
                {
                    "fixture": {
                        "id": 592342,
                        "date": "2024-01-15T14:30:00Z",
                        "status": {"long": "First Half", "elapsed": 35},
                        "venue": {"name": "Camp Nou"},
                        "referee": "Antonio Mateu"
                    },
                    "teams": {
                        "home": {"name": "Barcelona"},
                        "away": {"name": "Real Madrid"}
                    },
                    "goals": {"home": 2, "away": 0},
                    "league": {
                        "name": "La Liga",
                        "country": "Spain",
                        "id": 140,
                        "round": "Regular Season - 21"
                    },
                    "odds": {
                        "bookmakers": [
                            {
                                "bets": [
                                    {
                                        "name": "Total Goals Over/Under",
                                        "values": [
                                            {"value": "Over 2.5", "odd": 1.85},
                                            {"value": "Under 2.5", "odd": 1.95}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                }
            ]
        }
        
        games = self.scraper.parse_api_football_response(response_data)
        
        assert len(games) == 1
        game = games[0]
        assert game.id == "592342"
        assert game.home_team == "Barcelona"
        assert game.away_team == "Real Madrid"
        assert game.home_score == 2
        assert game.away_score == 0
        assert game.minute == 35
        assert game.status == "First Half"
        assert game.league == "La Liga"
        assert game.country == "Spain"
        assert game.odds_over_25 == 1.85
        assert game.source == "api-football.com"
        assert "fixture_id" in game.metadata
        assert game.metadata["round"] == "Regular Season - 21"


class TestFlashScoreScraper:
    """Test FlashScore web scraper"""
    
    def setup_method(self):
        """Setup before each test"""
        self.scraper = FlashScoreScraper()
    
    def test_initialization(self):
        """Test FlashScore scraper initialization"""
        assert self.scraper.name == "flashscore"
        assert self.scraper.base_url == "https://www.flashscore.com"
        assert self.scraper.source_type == DataSourceType.WEB_SCRAPER
        assert "flashscore_headers" in self.scraper.__dict__
        assert "status_mapping" in self.scraper.__dict__
    
    def test_status_mapping(self):
        """Test FlashScore status mapping"""
        mapping = self.scraper.status_mapping
        
        assert mapping["1H"] == "live"
        assert mapping["2H"] == "live"
        assert mapping["HT"] == "halftime"
        assert mapping["FT"] == "finished"
        assert mapping["NS"] == "scheduled"
    
    @pytest.mark.asyncio
    async def test_fetch_live_games_xhr(self):
        """Test XHR API fetch method"""
        mock_html = '<script>window.fs_scoreboard_config = {"feedUrl": "/x/feed/test"}</script>'
        mock_xhr_response = 'AA÷match1¬AE÷Team A¬AF÷Team B¬AG÷1¬AH÷0¬AD÷45¬AC÷1H¬'
        
        with patch.object(self.scraper, 'get_with_retry') as mock_get:
            # First call returns HTML
            mock_get.side_effect = [mock_html, mock_xhr_response]
            
            games = await self.scraper.fetch_live_games()
            
            assert mock_get.call_count >= 2
            # Should return empty list since parsing mock data won't create valid games
    
    def test_parse_key_value_response(self):
        """Test FlashScore key-value response parsing"""
        response_text = 'AA÷match1¬AE÷Manchester United¬AF÷Liverpool¬AG÷1¬AH÷0¬AD÷45¬AC÷1H¬B÷Premier League¬C÷England¬W÷2024-01-15T14:30:00Z¬'
        
        # This is a simplified test since actual parsing is more complex
        games = self.scraper._parse_key_value_response(response_text)
        
        # Should handle parsing without errors
        assert isinstance(games, list)
    
    def test_extract_country_from_league(self):
        """Test country extraction from league names"""
        test_cases = [
            ("PREMIER LEAGUE", "England"),
            ("LA LIGA", "Spain"),
            ("BUNDESLIGA", "Germany"),
            ("SERIE A", "Italy"),
            ("Unknown League", "Unknown"),
        ]
        
        for league, expected_country in test_cases:
            country = self.scraper._extract_country_from_league(league)
            assert country == expected_country


class TestIntegration:
    """Integration tests for scraper system"""
    
    def test_merge_games_from_sources(self):
        """Test merging games from multiple sources"""
        timestamp = datetime.now()
        
        # Create games from different sources
        api_game = ScrapedGame(
            id="api_1",
            home_team="Team A",
            away_team="Team B",
            home_score=1,
            away_score=0,
            minute=45,
            status="live",
            league="Test",
            country="Test",
            timestamp=timestamp,
            source="football-data.org",
            source_type=DataSourceType.STRUCTURED_API
        )
        
        scraper_game_same = ScrapedGame(
            id="scraper_1",
            home_team="Team A",
            away_team="Team B",
            home_score=1,
            away_score=0,
            minute=45,
            status="live",
            league="Test",
            country="Test",
            timestamp=timestamp,
            source="flashscore.com",
            source_type=DataSourceType.WEB_SCRAPER
        )
        
        scraper_game_different = ScrapedGame(
            id="scraper_2",
            home_team="Team C",
            away_team="Team D",
            home_score=0,
            away_score=0,
            minute=30,
            status="live",
            league="Test",
            country="Test",
            timestamp=timestamp,
            source="flashscore.com",
            source_type=DataSourceType.WEB_SCRAPER
        )
        
        games_list = [
            [api_game],                    # From API source
            [scraper_game_same, scraper_game_different]  # From web scraper
        ]
        
        merged_games = merge_games_from_sources(games_list)
        
        # Should have 2 unique games (deduplicated Team A vs Team B)
        assert len(merged_games) == 2
        
        # API game should be preferred over web scraper for same game
        team_a_b_game = next(g for g in merged_games if g.home_team == "Team A")
        assert team_a_b_game.source_type == DataSourceType.STRUCTURED_API
        assert team_a_b_game.source == "football-data.org"
    
    @pytest.mark.asyncio
    async def test_scraper_manager_integration(self):
        """Test integration with ScraperManager"""
        # Import here to avoid circular imports
        from src.scrapers.scraper_manager import ScraperManager
        
        # Create mock scrapers
        mock_football_data = AsyncMock()
        mock_football_data.fetch_live_games.return_value = [
            ScrapedGame(
                id="1", home_team="Team A", away_team="Team B",
                home_score=1, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="football-data.org", source_type=DataSourceType.STRUCTURED_API
            )
        ]
        
        mock_api_football = AsyncMock()
        mock_api_football.fetch_live_games.return_value = [
            ScrapedGame(
                id="2", home_team="Team C", away_team="Team D",
                home_score=0, away_score=0, minute=30, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="api-football.com", source_type=DataSourceType.STRUCTURED_API
            )
        ]
        
        mock_flashscore = AsyncMock()
        mock_flashscore.fetch_live_games.return_value = [
            ScrapedGame(
                id="3", home_team="Team A", away_team="Team B",  # Same as football-data
                home_score=1, away_score=0, minute=45, status="live",
                league="Test", country="Test", timestamp=datetime.now(),
                source="flashscore.com", source_type=DataSourceType.WEB_SCRAPER
            )
        ]
        
        # Create ScraperManager with mocked scrapers
        with patch('src.scrapers.scraper_manager.FootballDataScraper', return_value=mock_football_data), \
             patch('src.scrapers.scraper_manager.ApiFootballScraper', return_value=mock_api_football), \
             patch('src.scrapers.scraper_manager.FlashScoreScraper', return_value=mock_flashscore):
            
            manager = ScraperManager()
            games = await manager.get_live_games()
            
            # Should have 2 unique games (deduplicated)
            assert len(games) == 2
            
            # Verify API sources are preferred
            sources = [g.source for g in games]
            assert "football-data.org" in sources
            assert "api-football.com" in sources
            assert "flashscore.com" not in sources  # Should be deduplicated in favor of API


class TestPerformance:
    """Performance tests for scrapers"""
    
    @pytest.mark.asyncio
    async def test_scraper_performance(self):
        """Test scraper performance metrics"""
        scraper = BaseScraper(name="perf_test", base_url="https://test.com")
        
        # Simulate some requests
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{}')
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Make multiple requests
            for _ in range(5):
                await scraper.get_with_retry("https://test.com/api")
        
        stats = scraper.get_stats()
        
        assert stats['requests'] >= 5
        assert stats['success_rate'] >= 0  # Should be calculated
        assert 'requests_per_hour' in stats


# Helper function to run all tests
def run_all_tests():
    """Run all scraper tests"""
    print("🧪 Running scraper tests...")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestBaseScraper,
        TestFootballDataScraper,
        TestApiFootballScraper,
        TestFlashScoreScraper,
        TestIntegration,
        TestPerformance
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 Testing: {test_class.__name__}")
        print("-" * 40)
        
        # Create instance
        test_instance = test_class()
        
        # Find test methods
        test_methods = [m for m in dir(test_instance) 
                       if m.startswith('test_') and callable(getattr(test_instance, m))]
        
        for method_name in test_methods:
            total_tests += 1
            method = getattr(test_instance, method_name)
            
            try:
                # Handle async methods
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    # Setup method if exists
                    if hasattr(test_instance, 'setup_method'):
                        test_instance.setup_method()
                    
                    method()
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name} - {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{method_name}: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")
        return False
    else:
        print("\n🎉 All tests passed!")
        return True


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_all_tests()
    sys.exit(0 if success else 1)
