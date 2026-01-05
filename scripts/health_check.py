#!/usr/bin/env python3
"""
health_check.py - Comprehensive system health check for Over/Under Predictor
Checks all components: APIs, scrapers, database, notifications, and GitHub Actions
"""
import asyncio
import aiohttp
import sys
import os
import json
import yaml
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try imports with fallbacks
try:
    from src.scrapers.base_scraper import FootballDataScraper, ApiFootballScraper, create_game_id
    from src.scrapers.flashscore_scraper import FlashScoreScraper
    from config import get_config, load_config
except ImportError:
    print("Warning: Could not import all modules. Some checks may be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("health_check")


class HealthCheck:
    """
    Main health check class for the Over/Under Predictor system
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize health check system
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            try:
                self.config = get_config()
            except:
                self.config = self._load_default_config()
        
        # Setup paths
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = os.path.join(self.project_root, 'storage', 'logs')
        self.data_dir = os.path.join(self.project_root, 'storage', 'data')
        self.reports_dir = os.path.join(self.project_root, 'reports')
        
        # Ensure directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Health check results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {},
            'summary': {},
            'recommendations': []
        }
        
        # API keys from environment or config
        self.api_keys = {
            'football_data': os.getenv('FOOTBALL_DATA_API_KEY', 
                                     self.config.get('api_keys', {}).get('football_data', 'e460c4df45fc41fe8fca16623ee3f733')),
            'api_football': os.getenv('API_FOOTBALL_KEY',
                                    self.config.get('api_keys', {}).get('api_football', '43b03752fdae250ddf40f592d5490388')),
            'telegram_bot': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat': os.getenv('TELEGRAM_CHAT_ID', '')
        }
        
        # Status colors for console output
        self.status_colors = {
            'HEALTHY': '🟢',
            'WARNING': '🟡',
            'CRITICAL': '🔴',
            'UNKNOWN': '⚪'
        }
        
        logger.info(f"Health check initialized for project: {self.project_root}")
    
    def _load_default_config(self) -> Dict:
        """Load default configuration if config module not available"""
        return {
            'system': {
                'name': 'Over/Under Predictor',
                'update_interval': 300
            },
            'api_keys': {
                'football_data': 'e460c4df45fc41fe8fca16623ee3f733',
                'api_football': '43b03752fdae250ddf40f592d5490388'
            }
        }
    
    async def run_comprehensive_check(self) -> Dict:
        """
        Run comprehensive health check on all system components
        
        Returns:
            Dictionary with all health check results
        """
        logger.info("Starting comprehensive health check...")
        
        # Run all checks concurrently
        checks = [
            self.check_api_connectivity(),
            self.check_scrapers(),
            self.check_file_system(),
            self.check_database(),
            self.check_notifications(),
            self.check_github_actions(),
            self.check_performance(),
            self.check_security(),
            self.check_dependencies()
        ]
        
        # Run checks
        check_results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Process results
        check_names = [
            'api_connectivity', 'scrapers', 'file_system', 'database',
            'notifications', 'github_actions', 'performance', 'security', 'dependencies'
        ]
        
        for name, result in zip(check_names, check_results):
            if isinstance(result, Exception):
                self.results['checks'][name] = {
                    'status': 'CRITICAL',
                    'message': f'Check failed with error: {str(result)}',
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"Check {name} failed: {result}")
            else:
                self.results['checks'][name] = result
        
        # Generate overall status
        self._generate_overall_status()
        
        # Generate summary and recommendations
        self._generate_summary()
        self._generate_recommendations()
        
        # Save results
        self.save_results()
        
        # Send notifications if configured
        await self.send_notifications()
        
        logger.info(f"Health check completed. Overall status: {self.results['overall_status']}")
        return self.results
    
    async def check_api_connectivity(self) -> Dict:
        """
        Check connectivity to all external APIs
        
        Returns:
            Dictionary with API connectivity status
        """
        logger.info("Checking API connectivity...")
        
        checks = {}
        
        # 1. football-data.org API
        football_data_status = await self._check_football_data_api()
        checks['football_data'] = football_data_status
        
        # 2. api-football.com API
        api_football_status = await self._check_api_football_api()
        checks['api_football'] = api_football_status
        
        # 3. FlashScore website
        flashscore_status = await self._check_flashscore_website()
        checks['flashscore'] = flashscore_status
        
        # Determine overall API status
        all_healthy = all(check['status'] == 'HEALTHY' for check in checks.values())
        any_critical = any(check['status'] == 'CRITICAL' for check in checks.values())
        
        overall_status = 'HEALTHY'
        if any_critical:
            overall_status = 'CRITICAL'
        elif not all_healthy:
            overall_status = 'WARNING'
        
        failed_apis = [name for name, check in checks.items() if check['status'] != 'HEALTHY']
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"API connectivity: {overall_status}. Failed: {failed_apis}" if failed_apis else "All APIs healthy",
            'timestamp': datetime.now().isoformat(),
            'failed_apis': failed_apis
        }
    
    async def _check_football_data_api(self) -> Dict:
        """Check football-data.org API"""
        try:
            url = "https://api.football-data.org/v4/matches"
            headers = {'X-Auth-Token': self.api_keys['football_data']}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        matches_count = len(data.get('matches', []))
                        return {
                            'status': 'HEALTHY',
                            'response_time': response.elapsed.total_seconds(),
                            'matches_found': matches_count,
                            'rate_limit': response.headers.get('X-Requests-Available', 'Unknown'),
                            'message': f"API healthy. Found {matches_count} matches."
                        }
                    elif response.status == 429:
                        return {
                            'status': 'WARNING',
                            'message': 'Rate limit exceeded',
                            'rate_limit': response.headers.get('X-Requests-Available', '0')
                        }
                    else:
                        return {
                            'status': 'CRITICAL',
                            'message': f'HTTP {response.status}: {await response.text()}'
                        }
        except asyncio.TimeoutError:
            return {
                'status': 'CRITICAL',
                'message': 'Timeout connecting to API'
            }
        except Exception as e:
            return {
                'status': 'CRITICAL',
                'message': f'Error: {str(e)}'
            }
    
    async def _check_api_football_api(self) -> Dict:
        """Check api-football.com API"""
        try:
            url = "https://v3.football.api-sports.io/status"
            headers = {
                'x-rapidapi-host': 'v3.football.api-sports.io',
                'x-rapidapi-key': self.api_keys['api_football']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        status_info = data.get('response', {})
                        return {
                            'status': 'HEALTHY',
                            'response_time': response.elapsed.total_seconds(),
                            'account': status_info.get('account', {}).get('firstname', 'Unknown'),
                            'requests_day': f"{status_info.get('requests', {}).get('current', 0)}/{status_info.get('requests', {}).get('limit_day', 0)}",
                            'message': 'API healthy'
                        }
                    elif response.status == 429:
                        return {
                            'status': 'WARNING',
                            'message': 'Rate limit exceeded'
                        }
                    else:
                        return {
                            'status': 'CRITICAL',
                            'message': f'HTTP {response.status}'
                        }
        except asyncio.TimeoutError:
            return {
                'status': 'CRITICAL',
                'message': 'Timeout connecting to API'
            }
        except Exception as e:
            return {
                'status': 'CRITICAL',
                'message': f'Error: {str(e)}'
            }
    
    async def _check_flashscore_website(self) -> Dict:
        """Check FlashScore website accessibility"""
        try:
            url = "https://www.flashscore.com/football/"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10, 
                                     headers={'User-Agent': 'Mozilla/5.0'}) as response:
                    if response.status == 200:
                        content = await response.text()
                        has_live_games = 'live' in content.lower() or 'match' in content.lower()
                        return {
                            'status': 'HEALTHY',
                            'response_time': response.elapsed.total_seconds(),
                            'content_length': len(content),
                            'has_live_content': has_live_games,
                            'message': 'Website accessible'
                        }
                    else:
                        return {
                            'status': 'WARNING',
                            'message': f'HTTP {response.status}'
                        }
        except asyncio.TimeoutError:
            return {
                'status': 'WARNING',
                'message': 'Timeout connecting to website'
            }
        except Exception as e:
            return {
                'status': 'WARNING',
                'message': f'Error: {str(e)}'
            }
    
    async def check_scrapers(self) -> Dict:
        """
        Check all scraper components
        
        Returns:
            Dictionary with scraper status
        """
        logger.info("Checking scrapers...")
        
        checks = {}
        
        try:
            # Test FootballDataScraper
            football_data_scraper = FootballDataScraper()
            football_data_health = await football_data_scraper.health_check()
            checks['football_data_scraper'] = football_data_health
            
            # Test ApiFootballScraper
            api_football_scraper = ApiFootballScraper()
            api_football_health = await api_football_scraper.health_check()
            checks['api_football_scraper'] = api_football_health
            
            # Test FlashScoreScraper connectivity
            flashscore_scraper = FlashScoreScraper()
            flashscore_connected = await flashscore_scraper.test_connectivity()
            checks['flashscore_scraper'] = {
                'status': 'HEALTHY' if flashscore_connected else 'WARNING',
                'connectivity': flashscore_connected,
                'message': 'Connected' if flashscore_connected else 'Connection failed'
            }
            
            # Check if scrapers can fetch data
            test_games = await self._test_scraper_data_fetch()
            checks['data_fetch'] = test_games
            
            # Determine overall status
            all_healthy = all(
                check.get('status') == 'HEALTHY' 
                for check in checks.values() 
                if isinstance(check, dict) and 'status' in check
            )
            
            overall_status = 'HEALTHY' if all_healthy else 'WARNING'
            
            return {
                'status': overall_status,
                'details': checks,
                'message': f"Scrapers: {overall_status}. Test fetch: {test_games.get('games_found', 0)} games",
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking scrapers: {e}")
            return {
                'status': 'CRITICAL',
                'message': f'Error checking scrapers: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    async def _test_scraper_data_fetch(self) -> Dict:
        """Test if scrapers can fetch actual data"""
        try:
            # Use football-data API as test
            url = "https://api.football-data.org/v4/matches"
            headers = {'X-Auth-Token': self.api_keys['football_data']}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        matches = data.get('matches', [])
                        live_matches = [m for m in matches if m.get('status') in ['LIVE', 'IN_PLAY']]
                        
                        return {
                            'status': 'HEALTHY',
                            'total_matches': len(matches),
                            'live_matches': len(live_matches),
                            'games_found': len(live_matches),
                            'message': f'Found {len(live_matches)} live matches'
                        }
                    else:
                        return {
                            'status': 'WARNING',
                            'games_found': 0,
                            'message': f'HTTP {response.status}'
                        }
        except Exception as e:
            return {
                'status': 'WARNING',
                'games_found': 0,
                'message': f'Error: {str(e)}'
            }
    
    async def check_file_system(self) -> Dict:
        """
        Check file system health and permissions
        
        Returns:
            Dictionary with file system status
        """
        logger.info("Checking file system...")
        
        checks = {}
        
        # Check required directories
        required_dirs = [
            ('storage', self.data_dir),
            ('logs', self.log_dir),
            ('reports', self.reports_dir),
            ('config', os.path.join(self.project_root, 'config'))
        ]
        
        for dir_name, dir_path in required_dirs:
            if os.path.exists(dir_path):
                # Check if writable
                test_file = os.path.join(dir_path, '.health_check_test')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    checks[dir_name] = {'status': 'HEALTHY', 'writable': True}
                except Exception as e:
                    checks[dir_name] = {'status': 'CRITICAL', 'writable': False, 'error': str(e)}
            else:
                checks[dir_name] = {'status': 'CRITICAL', 'exists': False}
        
        # Check important files
        required_files = [
            ('requirements.txt', os.path.join(self.project_root, 'requirements.txt')),
            ('config/settings.yaml', os.path.join(self.project_root, 'config', 'settings.yaml')),
            ('.github/workflows/predictor.yml', os.path.join(self.project_root, '.github', 'workflows', 'predictor.yml'))
        ]
        
        for file_name, file_path in required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                checks[file_name] = {'status': 'HEALTHY', 'size': file_size}
            else:
                checks[file_name] = {'status': 'WARNING', 'exists': False}
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_root)
            free_gb = free / (2**30)
            
            checks['disk_space'] = {
                'status': 'HEALTHY' if free_gb > 1 else 'WARNING',
                'total_gb': round(total / (2**30), 2),
                'used_gb': round(used / (2**30), 2),
                'free_gb': round(free_gb, 2),
                'message': f'{free_gb:.1f}GB free'
            }
        except Exception as e:
            checks['disk_space'] = {'status': 'UNKNOWN', 'error': str(e)}
        
        # Check recent log files
        log_files = []
        if os.path.exists(self.log_dir):
            for file in os.listdir(self.log_dir):
                if file.endswith('.log'):
                    file_path = os.path.join(self.log_dir, file)
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age_hours = (datetime.now() - mtime).total_seconds() / 3600
                    log_files.append({
                        'file': file,
                        'size_mb': os.path.getsize(file_path) / (1024*1024),
                        'age_hours': round(age_hours, 1)
                    })
        
        checks['log_files'] = {
            'count': len(log_files),
            'files': log_files[:5],  # First 5 files
            'status': 'HEALTHY' if log_files else 'WARNING'
        }
        
        # Determine overall status
        critical_issues = sum(1 for check in checks.values() 
                            if isinstance(check, dict) and check.get('status') == 'CRITICAL')
        warning_issues = sum(1 for check in checks.values() 
                           if isinstance(check, dict) and check.get('status') == 'WARNING')
        
        overall_status = 'HEALTHY'
        if critical_issues > 0:
            overall_status = 'CRITICAL'
        elif warning_issues > 0:
            overall_status = 'WARNING'
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"File system: {overall_status}. Critical: {critical_issues}, Warnings: {warning_issues}",
            'timestamp': datetime.now().isoformat(),
            'critical_issues': critical_issues,
            'warning_issues': warning_issues
        }
    
    async def check_database(self) -> Dict:
        """
        Check database connectivity and status
        
        Returns:
            Dictionary with database status
        """
        logger.info("Checking database...")
        
        checks = {}
        
        # Check if predictions file exists and is recent
        predictions_file = os.path.join(self.data_dir, 'predictions.json')
        if os.path.exists(predictions_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(predictions_file))
            age_minutes = (datetime.now() - mtime).total_seconds() / 60
            
            try:
                with open(predictions_file, 'r') as f:
                    data = json.load(f)
                
                predictions_count = len(data.get('predictions', []))
                last_update = data.get('timestamp', 'Unknown')
                
                checks['predictions_file'] = {
                    'status': 'HEALTHY' if age_minutes < 60 else 'WARNING',
                    'age_minutes': round(age_minutes, 1),
                    'predictions_count': predictions_count,
                    'last_update': last_update,
                    'file_size_mb': os.path.getsize(predictions_file) / (1024*1024)
                }
            except Exception as e:
                checks['predictions_file'] = {
                    'status': 'CRITICAL',
                    'error': str(e)
                }
        else:
            checks['predictions_file'] = {
                'status': 'WARNING',
                'message': 'File does not exist'
            }
        
        # Check historical data
        historical_dir = os.path.join(self.data_dir, 'historical')
        if os.path.exists(historical_dir):
            historical_files = [f for f in os.listdir(historical_dir) if f.endswith('.json')]
            checks['historical_data'] = {
                'status': 'HEALTHY' if historical_files else 'WARNING',
                'file_count': len(historical_files),
                'oldest_file': self._get_oldest_file_age(historical_dir) if historical_files else 'N/A'
            }
        else:
            checks['historical_data'] = {
                'status': 'WARNING',
                'message': 'Directory does not exist'
            }
        
        # Determine overall status
        statuses = [check.get('status', 'UNKNOWN') for check in checks.values()]
        
        overall_status = 'HEALTHY'
        if 'CRITICAL' in statuses:
            overall_status = 'CRITICAL'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"Database: {overall_status}. Predictions: {checks.get('predictions_file', {}).get('predictions_count', 0)}",
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_oldest_file_age(self, directory: str) -> str:
        """Get age of oldest file in directory"""
        oldest = None
        for file in os.listdir(directory):
            if file.endswith('.json'):
                file_path = os.path.join(directory, file)
                mtime = os.path.getmtime(file_path)
                if oldest is None or mtime < oldest:
                    oldest = mtime
        
        if oldest:
            age_days = (time.time() - oldest) / (24 * 3600)
            return f"{age_days:.1f} days"
        return "N/A"
    
    async def check_notifications(self) -> Dict:
        """
        Check notification systems (Telegram, email, etc.)
        
        Returns:
            Dictionary with notification status
        """
        logger.info("Checking notifications...")
        
        checks = {}
        
        # Check Telegram
        if self.api_keys['telegram_bot'] and self.api_keys['telegram_chat']:
            telegram_status = await self._check_telegram_bot()
            checks['telegram'] = telegram_status
        else:
            checks['telegram'] = {
                'status': 'WARNING',
                'message': 'Telegram credentials not configured'
            }
        
        # Check if notification log exists
        notification_log = os.path.join(self.log_dir, 'notifications.log')
        if os.path.exists(notification_log):
            log_size = os.path.getsize(notification_log)
            checks['notification_log'] = {
                'status': 'HEALTHY',
                'size_kb': round(log_size / 1024, 2),
                'exists': True
            }
        else:
            checks['notification_log'] = {
                'status': 'WARNING',
                'message': 'Notification log does not exist'
            }
        
        # Determine overall status
        statuses = [check.get('status', 'UNKNOWN') for check in checks.values()]
        
        overall_status = 'HEALTHY'
        if 'CRITICAL' in statuses:
            overall_status = 'CRITICAL'
        elif statuses.count('WARNING') > 0:
            overall_status = 'WARNING'
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"Notifications: {overall_status}",
            'timestamp': datetime.now().isoformat()
        }
    
    async def _check_telegram_bot(self) -> Dict:
        """Check Telegram bot connectivity"""
        try:
            url = f"https://api.telegram.org/bot{self.api_keys['telegram_bot']}/getMe"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        bot_name = data.get('result', {}).get('first_name', 'Unknown')
                        
                        # Try to send a test message
                        test_url = f"https://api.telegram.org/bot{self.api_keys['telegram_bot']}/sendMessage"
                        test_data = {
                            'chat_id': self.api_keys['telegram_chat'],
                            'text': f'🤖 Health check test at {datetime.now().strftime("%H:%M")}\nSystem is running health checks.',
                            'parse_mode': 'HTML'
                        }
                        
                        async with session.post(test_url, json=test_data, timeout=10) as test_response:
                            if test_response.status == 200:
                                return {
                                    'status': 'HEALTHY',
                                    'bot_name': bot_name,
                                    'can_send': True,
                                    'message': f'Bot "{bot_name}" is operational'
                                }
                            else:
                                return {
                                    'status': 'WARNING',
                                    'bot_name': bot_name,
                                    'can_send': False,
                                    'message': 'Bot exists but cannot send messages'
                                }
                    else:
                        return {
                            'status': 'CRITICAL',
                            'message': f'HTTP {response.status}: Invalid bot token'
                        }
        except asyncio.TimeoutError:
            return {
                'status': 'WARNING',
                'message': 'Timeout connecting to Telegram API'
            }
        except Exception as e:
            return {
                'status': 'CRITICAL',
                'message': f'Error: {str(e)}'
            }
    
    async def check_github_actions(self) -> Dict:
        """
        Check GitHub Actions workflow status
        
        Returns:
            Dictionary with GitHub Actions status
        """
        logger.info("Checking GitHub Actions...")
        
        checks = {}
        
        # Check if workflow file exists
        workflow_file = os.path.join(self.project_root, '.github', 'workflows', 'predictor.yml')
        if os.path.exists(workflow_file):
            with open(workflow_file, 'r') as f:
                workflow_content = f.read()
            
            checks['workflow_file'] = {
                'status': 'HEALTHY',
                'size': len(workflow_content),
                'has_schedule': 'schedule:' in workflow_content,
                'has_manual': 'workflow_dispatch:' in workflow_content
            }
        else:
            checks['workflow_file'] = {
                'status': 'CRITICAL',
                'message': 'Workflow file not found'
            }
        
        # Check last run by examining log files
        log_pattern = os.path.join(self.log_dir, 'app.log')
        if os.path.exists(log_pattern):
            try:
                # Get last 10 lines of main log
                with open(log_pattern, 'r') as f:
                    lines = f.readlines()[-10:]
                
                last_runs = []
                for line in lines:
                    if 'Completed:' in line or 'predictions' in line.lower():
                        last_runs.append(line.strip())
                
                checks['last_runs'] = {
                    'status': 'HEALTHY' if last_runs else 'WARNING',
                    'recent_logs': last_runs[-3:] if last_runs else [],
                    'has_recent_activity': len(last_runs) > 0
                }
            except Exception as e:
                checks['last_runs'] = {
                    'status': 'WARNING',
                    'error': str(e)
                }
        
        # Check if reports are being generated
        if os.path.exists(self.reports_dir):
            report_files = [f for f in os.listdir(self.reports_dir) 
                          if f.endswith(('.html', '.json'))]
            
            if report_files:
                latest_report = max(report_files, 
                                  key=lambda f: os.path.getmtime(os.path.join(self.reports_dir, f)))
                report_age = (datetime.now() - 
                            datetime.fromtimestamp(os.path.getmtime(os.path.join(self.reports_dir, latest_report))))
                
                checks['reports'] = {
                    'status': 'HEALTHY' if report_age.total_seconds() < 3600 else 'WARNING',
                    'count': len(report_files),
                    'latest': latest_report,
                    'age_minutes': round(report_age.total_seconds() / 60, 1)
                }
            else:
                checks['reports'] = {
                    'status': 'WARNING',
                    'message': 'No report files found'
                }
        
        # Determine overall status
        statuses = [check.get('status', 'UNKNOWN') for check in checks.values()]
        
        overall_status = 'HEALTHY'
        if 'CRITICAL' in statuses:
            overall_status = 'CRITICAL'
        elif statuses.count('WARNING') > 0:
            overall_status = 'WARNING'
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"GitHub Actions: {overall_status}",
            'timestamp': datetime.now().isoformat()
        }
    
    async def check_performance(self) -> Dict:
        """
        Check system performance metrics
        
        Returns:
            Dictionary with performance status
        """
        logger.info("Checking performance...")
        
        checks = {}
        
        # Check response times
        response_times = await self._measure_response_times()
        checks['response_times'] = response_times
        
        # Check memory usage (approximate)
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            checks['memory_usage'] = {
                'status': 'HEALTHY' if memory_mb < 100 else 'WARNING',
                'memory_mb': round(memory_mb, 2),
                'message': f'Using {memory_mb:.1f}MB RAM'
            }
        except ImportError:
            checks['memory_usage'] = {
                'status': 'UNKNOWN',
                'message': 'psutil not installed'
            }
        
        # Check if system is overloaded (CPU)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            
            checks['cpu_usage'] = {
                'status': 'HEALTHY' if cpu_percent < 80 else 'WARNING',
                'cpu_percent': cpu_percent,
                'message': f'CPU usage: {cpu_percent}%'
            }
        except ImportError:
            checks['cpu_usage'] = {
                'status': 'UNKNOWN',
                'message': 'psutil not installed'
            }
        
        # Determine overall status
        warning_count = sum(1 for check in checks.values() 
                          if isinstance(check, dict) and check.get('status') == 'WARNING')
        
        overall_status = 'HEALTHY' if warning_count == 0 else 'WARNING'
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"Performance: {overall_status}",
            'timestamp': datetime.now().isoformat()
        }
    
    async def _measure_response_times(self) -> Dict:
        """Measure API response times"""
        response_times = {}
        
        # Test football-data.org
        try:
            start = time.time()
            url = "https://api.football-data.org/v4/matches"
            headers = {'X-Auth-Token': self.api_keys['football_data']}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    response_time = time.time() - start
                    
                    response_times['football_data'] = {
                        'time_seconds': round(response_time, 3),
                        'status': 'HEALTHY' if response_time < 2 else 'WARNING',
                        'http_status': response.status
                    }
        except Exception as e:
            response_times['football_data'] = {
                'time_seconds': None,
                'status': 'CRITICAL',
                'error': str(e)
            }
        
        # Test FlashScore
        try:
            start = time.time()
            url = "https://www.flashscore.com/football/"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    response_time = time.time() - start
                    
                    response_times['flashscore'] = {
                        'time_seconds': round(response_time, 3),
                        'status': 'HEALTHY' if response_time < 3 else 'WARNING',
                        'http_status': response.status
                    }
        except Exception as e:
            response_times['flashscore'] = {
                'time_seconds': None,
                'status': 'WARNING',
                'error': str(e)
            }
        
        # Calculate average
        valid_times = [rt['time_seconds'] for rt in response_times.values() 
                      if rt['time_seconds'] is not None]
        
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            response_times['average'] = round(avg_time, 3)
        
        return response_times
    
    async def check_security(self) -> Dict:
        """
        Check security aspects of the system
        
        Returns:
            Dictionary with security status
        """
        logger.info("Checking security...")
        
        checks = {}
        
        # Check for exposed API keys
        exposed_keys = []
        
        # Check environment variables
        env_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 
                   'FOOTBALL_DATA_API_KEY', 'API_FOOTBALL_KEY']
        
        for var in env_vars:
            if os.getenv(var):
                # Check if it's in plain text in any files
                exposed = self._check_exposed_key(var, os.getenv(var))
                if exposed:
                    exposed_keys.append(var)
        
        checks['api_keys'] = {
            'status': 'CRITICAL' if exposed_keys else 'HEALTHY',
            'exposed_keys': exposed_keys,
            'message': f'Exposed keys: {exposed_keys}' if exposed_keys else 'API keys secure'
        }
        
        # Check file permissions
        sensitive_files = [
            os.path.join(self.project_root, 'config', 'settings.yaml'),
            os.path.join(self.project_root, '.env'),
            os.path.join(self.data_dir, 'predictions.json')
        ]
        
        permission_issues = []
        for file in sensitive_files:
            if os.path.exists(file):
                mode = os.stat(file).st_mode
                # Check if world-readable
                if mode & 0o004:
                    permission_issues.append(file)
        
        checks['file_permissions'] = {
            'status': 'CRITICAL' if permission_issues else 'HEALTHY',
            'world_readable_files': permission_issues,
            'message': f'World readable: {len(permission_issues)} files' if permission_issues else 'Permissions OK'
        }
        
        # Check for .env file
        env_file = os.path.join(self.project_root, '.env')
        if os.path.exists(env_file):
            env_size = os.path.getsize(env_file)
            checks['env_file'] = {
                'status': 'HEALTHY' if env_size > 0 else 'WARNING',
                'exists': True,
                'size': env_size
            }
        else:
            checks['env_file'] = {
                'status': 'WARNING',
                'message': '.env file not found'
            }
        
        # Determine overall status
        critical_issues = sum(1 for check in checks.values() 
                            if isinstance(check, dict) and check.get('status') == 'CRITICAL')
        
        overall_status = 'HEALTHY'
        if critical_issues > 0:
            overall_status = 'CRITICAL'
        elif any(check.get('status') == 'WARNING' for check in checks.values()):
            overall_status = 'WARNING'
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"Security: {overall_status}. Critical issues: {critical_issues}",
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_exposed_key(self, key_name: str, key_value: str) -> bool:
        """Check if API key is exposed in source files"""
        if not key_value:
            return False
        
        # Search for key in source files
        search_dirs = [
            os.path.join(self.project_root, 'src'),
            os.path.join(self.project_root, 'scripts'),
            self.project_root
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(('.py', '.yaml', '.yml', '.json')):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    if key_value in content:
                                        return True
                            except:
                                continue
        
        return False
    
    async def check_dependencies(self) -> Dict:
        """
        Check Python dependencies and versions
        
        Returns:
            Dictionary with dependencies status
        """
        logger.info("Checking dependencies...")
        
        checks = {}
        
        # Check requirements.txt
        requirements_file = os.path.join(self.project_root, 'requirements.txt')
        if os.path.exists(requirements_file):
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            checks['requirements_file'] = {
                'status': 'HEALTHY',
                'count': len(requirements),
                'requirements': requirements[:5]  # First 5
            }
        else:
            checks['requirements_file'] = {
                'status': 'CRITICAL',
                'message': 'requirements.txt not found'
            }
        
        # Check Python version
        python_version = sys.version.split()[0]
        checks['python_version'] = {
            'status': 'HEALTHY',
            'version': python_version,
            'message': f'Python {python_version}'
        }
        
        # Check critical dependencies
        critical_deps = ['aiohttp', 'requests', 'beautifulsoup4', 'pandas', 'numpy']
        installed_deps = {}
        
        for dep in critical_deps:
            try:
                module = __import__(dep)
                installed_deps[dep] = {
                    'installed': True,
                    'version': getattr(module, '__version__', 'Unknown')
                }
            except ImportError:
                installed_deps[dep] = {'installed': False}
        
        missing_deps = [dep for dep, info in installed_deps.items() if not info.get('installed')]
        
        checks['critical_dependencies'] = {
            'status': 'CRITICAL' if missing_deps else 'HEALTHY',
            'installed': installed_deps,
            'missing': missing_deps,
            'message': f'Missing: {missing_deps}' if missing_deps else 'All critical deps installed'
        }
        
        # Determine overall status
        statuses = [checks.get('requirements_file', {}).get('status', 'UNKNOWN'),
                   checks.get('critical_dependencies', {}).get('status', 'UNKNOWN')]
        
        overall_status = 'HEALTHY'
        if 'CRITICAL' in statuses:
            overall_status = 'CRITICAL'
        elif 'WARNING' in statuses:
            overall_status = 'WARNING'
        
        return {
            'status': overall_status,
            'details': checks,
            'message': f"Dependencies: {overall_status}. Missing: {len(missing_deps)}",
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_overall_status(self):
        """Generate overall system status from individual checks"""
        status_weights = {
            'CRITICAL': 3,
            'WARNING': 2,
            'HEALTHY': 1,
            'UNKNOWN': 0
        }
        
        check_statuses = [check.get('status', 'UNKNOWN') for check in self.results['checks'].values()]
        
        if not check_statuses:
            self.results['overall_status'] = 'UNKNOWN'
            return
        
        # Calculate weighted status
        status_scores = [status_weights.get(status, 0) for status in check_statuses]
        avg_score = sum(status_scores) / len(status_scores)
        
        if avg_score >= 2.5:
            overall_status = 'CRITICAL'
        elif avg_score >= 1.5:
            overall_status = 'WARNING'
        elif avg_score >= 0.5:
            overall_status = 'HEALTHY'
        else:
            overall_status = 'UNKNOWN'
        
        self.results['overall_status'] = overall_status
    
    def _generate_summary(self):
        """Generate summary statistics"""
        total_checks = len(self.results['checks'])
        healthy_checks = sum(1 for check in self.results['checks'].values() 
                           if check.get('status') == 'HEALTHY')
        warning_checks = sum(1 for check in self.results['checks'].values() 
                           if check.get('status') == 'WARNING')
        critical_checks = sum(1 for check in self.results['checks'].values() 
                            if check.get('status') == 'CRITICAL')
        
        self.results['summary'] = {
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'warning_checks': warning_checks,
            'critical_checks': critical_checks,
            'health_percentage': round((healthy_checks / total_checks) * 100, 1) if total_checks > 0 else 0
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on check results"""
        recommendations = []
        
        # Check-specific recommendations
        for check_name, check_result in self.results['checks'].items():
            if check_result.get('status') in ['WARNING', 'CRITICAL']:
                message = check_result.get('message', 'Issue detected')
                
                if check_name == 'api_connectivity':
                    failed_apis = check_result.get('failed_apis', [])
                    if failed_apis:
                        recommendations.append(f"🔧 Fix API connectivity for: {', '.join(failed_apis)}")
                
                elif check_name == 'file_system':
                    if check_result.get('critical_issues', 0) > 0:
                        recommendations.append("🔧 Fix file system permissions or create missing directories")
                
                elif check_name == 'notifications':
                    if 'Telegram' in message and 'not configured' in message:
                        recommendations.append("🔧 Configure Telegram bot credentials in GitHub Secrets")
                
                elif check_name == 'security':
                    if 'Exposed keys' in message:
                        recommendations.append("🔒 Remove exposed API keys from source files")
                    if 'World readable' in message:
                        recommendations.append("🔒 Fix file permissions on sensitive files")
        
        # General recommendations based on overall status
        if self.results['overall_status'] == 'CRITICAL':
            recommendations.insert(0, "🚨 IMMEDIATE ACTION REQUIRED: Critical issues detected!")
        elif self.results['overall_status'] == 'WARNING':
            recommendations.insert(0, "⚠️  Attention needed: System has warnings")
        
        # Add maintenance recommendations
        if self.results['summary']['health_percentage'] < 80:
            recommendations.append("🛠️  Schedule system maintenance soon")
        
        self.results['recommendations'] = recommendations
    
    def save_results(self):
        """Save health check results to file"""
        # Save detailed results
        results_file = os.path.join(self.data_dir, 'health_check_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary for dashboard
        summary_file = os.path.join(self.reports_dir, 'health_summary.json')
        summary = {
            'timestamp': self.results['timestamp'],
            'overall_status': self.results['overall_status'],
            'health_percentage': self.results['summary']['health_percentage'],
            'critical_issues': self.results['summary']['critical_checks'],
            'warning_issues': self.results['summary']['warning_checks'],
            'recommendations': self.results['recommendations'][:3]  # Top 3
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log results
        log_file = os.path.join(self.log_dir, 'health_check.log')
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Health Check: {self.results['timestamp']}\n")
            f.write(f"Overall Status: {self.results['overall_status']}\n")
            f.write(f"Health Percentage: {self.results['summary']['health_percentage']}%\n")
            f.write(f"Critical Issues: {self.results['summary']['critical_checks']}\n")
            f.write(f"Warning Issues: {self.results['summary']['warning_checks']}\n")
            f.write(f"Recommendations: {', '.join(self.results['recommendations'][:3])}\n")
        
        logger.info(f"Results saved to {results_file}")
    
    async def send_notifications(self):
        """Send notifications based on health check results"""
        if not self.api_keys['telegram_bot'] or not self.api_keys['telegram_chat']:
            logger.warning("Telegram credentials not configured, skipping notifications")
            return
        
        # Only send notifications for critical issues or if explicitly requested
        if self.results['overall_status'] in ['HEALTHY', 'WARNING']:
            logger.info("No critical issues, skipping Telegram notification")
            return
        
        try:
            message = self._format_telegram_message()
            
            url = f"https://api.telegram.org/bot{self.api_keys['telegram_bot']}/sendMessage"
            data = {
                'chat_id': self.api_keys['telegram_chat'],
                'text': message,
                'parse_mode': 'HTML',
                'disable_notification': self.results['overall_status'] == 'WARNING'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=10) as response:
                    if response.status == 200:
                        logger.info("Health check notification sent to Telegram")
                    else:
                        logger.error(f"Failed to send Telegram notification: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    def _format_telegram_message(self) -> str:
        """Format health check results for Telegram"""
        status_emoji = self.status_colors.get(self.results['overall_status'], '⚪')
        
        message = f"""
{status_emoji} <b>System Health Check</b> {status_emoji}

📊 <b>Overall Status:</b> {self.results['overall_status']}
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📈 <b>Health Score:</b> {self.results['summary']['health_percentage']}%

🔴 <b>Critical:</b> {self.results['summary']['critical_checks']}
🟡 <b>Warnings:</b> {self.results['summary']['warning_checks']}
🟢 <b>Healthy:</b> {self.results['summary']['healthy_checks']}

<b>Top Issues:</b>
"""
        
        # Add top 3 critical/warning issues
        issues_added = 0
        for check_name, check_result in self.results['checks'].items():
            if check_result.get('status') in ['CRITICAL', 'WARNING'] and issues_added < 3:
                emoji = '🔴' if check_result['status'] == 'CRITICAL' else '🟡'
                message += f"{emoji} {check_name}: {check_result.get('message', 'Issue')}\n"
                issues_added += 1
        
        # Add recommendations
        if self.results['recommendations']:
            message += "\n<b>Recommendations:</b>\n"
            for i, rec in enumerate(self.results['recommendations'][:3], 1):
                message += f"{i}. {rec}\n"
        
        # Add footer
        message += f"\n📁 <i>Detailed report saved to storage/</i>"
        
        return message
    
    def print_results(self):
        """Print results to console in readable format"""
        print("\n" + "="*60)
        print("OVER/UNDER PREDICTOR - SYSTEM HEALTH CHECK")
        print("="*60)
        
        # Overall status
        status_emoji = self.status_colors.get(self.results['overall_status'], '⚪')
        print(f"\n{status_emoji} OVERALL STATUS: {self.results['overall_status']}")
        print(f"   Time: {self.results['timestamp']}")
        print(f"   Health Score: {self.results['summary']['health_percentage']}%")
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"   Total Checks: {self.results['summary']['total_checks']}")
        print(f"   🟢 Healthy: {self.results['summary']['healthy_checks']}")
        print(f"   🟡 Warnings: {self.results['summary']['warning_checks']}")
        print(f"   🔴 Critical: {self.results['summary']['critical_checks']}")
        
        # Detailed check results
        print(f"\n🔍 DETAILED CHECK RESULTS:")
        for check_name, check_result in self.results['checks'].items():
            status_emoji = self.status_colors.get(check_result.get('status', 'UNKNOWN'), '⚪')
            print(f"   {status_emoji} {check_name.upper()}: {check_result.get('status', 'UNKNOWN')}")
            print(f"      Message: {check_result.get('message', 'No message')}")
        
        # Recommendations
        if self.results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
        print("Health check complete. Results saved to storage/health_check_results.json")
        print("="*60)


# Command line interface
async def main():
    """Main function for command line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Over/Under Predictor Health Check')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (no console output)')
    parser.add_argument('--notify', '-n', action='store_true', help='Always send notifications')
    parser.add_argument('--check', help='Run specific check (api, scrapers, files, etc.)')
    
    args = parser.parse_args()
    
    # Initialize health check
    health_check = HealthCheck(args.config)
    
    if args.check:
        # Run specific check
        check_method = getattr(health_check, f'check_{args.check}', None)
        if check_method:
            result = await check_method()
            print(json.dumps(result, indent=2))
        else:
            print(f"Unknown check: {args.check}")
            print("Available checks: api_connectivity, scrapers, file_system, database, notifications, github_actions, performance, security, dependencies")
    else:
        # Run comprehensive check
        results = await health_check.run_comprehensive_check()
        
        if not args.quiet:
            health_check.print_results()
        
        # Exit with appropriate code
        if results['overall_status'] == 'CRITICAL':
            sys.exit(1)
        elif results['overall_status'] == 'WARNING':
            sys.exit(2)
        else:
            sys.exit(0)


# Quick test function
async def quick_test():
    """Quick test of the health check system"""
    print("Running quick health check test...")
    
    health_check = HealthCheck()
    
    # Test individual components
    print("\n1. Testing API connectivity...")
    api_result = await health_check.check_api_connectivity()
    print(f"   Status: {api_result['status']}")
    
    print("\n2. Testing file system...")
    fs_result = await health_check.check_file_system()
    print(f"   Status: {fs_result['status']}")
    
    print("\n3. Testing dependencies...")
    deps_result = await health_check.check_dependencies()
    print(f"   Status: {deps_result['status']}")
    
    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    # Run main function
    asyncio.run(main())
