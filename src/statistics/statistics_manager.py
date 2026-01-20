"""
Statistics Manager - Collects and manages historical team/league statistics
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TeamStats:
    """Team statistics data class"""
    team_id: str
    team_name: str
    matches_played: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    home_matches: int = 0
    away_matches: int = 0
    over_25_wins: int = 0
    under_25_wins: int = 0
    last_updated: datetime = None
    
    @property
    def avg_goals_scored(self) -> float:
        return self.goals_scored / max(self.matches_played, 1)
    
    @property
    def avg_goals_conceded(self) -> float:
        return self.goals_conceded / max(self.matches_played, 1)
    
    @property
    def avg_total_goals(self) -> float:
        return (self.goals_scored + self.goals_conceded) / max(self.matches_played, 1)
    
    @property
    def over_25_rate(self) -> float:
        return self.over_25_wins / max(self.matches_played, 1)
    
    def to_dict(self) -> Dict:
        return {
            'team_id': self.team_id,
            'team_name': self.team_name,
            'matches_played': self.matches_played,
            'goals_scored': self.goals_scored,
            'goals_conceded': self.goals_conceded,
            'home_matches': self.home_matches,
            'away_matches': self.away_matches,
            'over_25_wins': self.over_25_wins,
            'under_25_wins': self.under_25_wins,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'averages': {
                'goals_scored': self.avg_goals_scored,
                'goals_conceded': self.avg_goals_conceded,
                'total_goals': self.avg_total_goals,
                'over_25_rate': self.over_25_rate
            }
        }

@dataclass
class LeagueStats:
    """League statistics data class"""
    league_id: str
    league_name: str
    total_matches: int = 0
    total_goals: int = 0
    over_25_matches: int = 0
    avg_goals_per_game: float = 2.5
    over_25_rate: float = 0.55
    last_updated: datetime = None
    
    def update(self, goals_home: int, goals_away: int):
        self.total_matches += 1
        self.total_goals += goals_home + goals_away
        if goals_home + goals_away > 2.5:
            self.over_25_matches += 1
        
        self.avg_goals_per_game = self.total_goals / max(self.total_matches, 1)
        self.over_25_rate = self.over_25_matches / max(self.total_matches, 1)
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'league_id': self.league_id,
            'league_name': self.league_name,
            'total_matches': self.total_matches,
            'total_goals': self.total_goals,
            'over_25_matches': self.over_25_matches,
            'avg_goals_per_game': self.avg_goals_per_game,
            'over_25_rate': self.over_25_rate,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class StatisticsManager:
    """Manages collection and retrieval of historical statistics"""
    
    def __init__(self, data_dir: str = "storage/statistics"):
        self.data_dir = data_dir
        self.team_stats_file = os.path.join(data_dir, "team_stats.json")
        self.league_stats_file = os.path.join(data_dir, "league_stats.json")
        self.match_history_file = os.path.join(data_dir, "match_history.json")
        
        self.team_stats: Dict[str, TeamStats] = {}
        self.league_stats: Dict[str, LeagueStats] = {}
        self.match_history: List[Dict] = []
        
        self._load_data()
    
    def _load_data(self):
        """Load existing statistics data"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        try:
            if os.path.exists(self.team_stats_file):
                with open(self.team_stats_file, 'r') as f:
                    data = json.load(f)
                    for team_id, team_data in data.items():
                        stats = TeamStats(
                            team_id=team_data['team_id'],
                            team_name=team_data['team_name'],
                            matches_played=team_data['matches_played'],
                            goals_scored=team_data['goals_scored'],
                            goals_conceded=team_data['goals_conceded'],
                            home_matches=team_data['home_matches'],
                            away_matches=team_data['away_matches'],
                            over_25_wins=team_data['over_25_wins'],
                            under_25_wins=team_data['under_25_wins'],
                            last_updated=datetime.fromisoformat(team_data['last_updated']) if team_data['last_updated'] else None
                        )
                        self.team_stats[team_id] = stats
        except Exception as e:
            logger.warning(f"Could not load team stats: {e}")
            self.team_stats = {}
        
        try:
            if os.path.exists(self.league_stats_file):
                with open(self.league_stats_file, 'r') as f:
                    data = json.load(f)
                    for league_id, league_data in data.items():
                        stats = LeagueStats(
                            league_id=league_data['league_id'],
                            league_name=league_data['league_name'],
                            total_matches=league_data['total_matches'],
                            total_goals=league_data['total_goals'],
                            over_25_matches=league_data['over_25_matches'],
                            avg_goals_per_game=league_data['avg_goals_per_game'],
                            over_25_rate=league_data['over_25_rate'],
                            last_updated=datetime.fromisoformat(league_data['last_updated']) if league_data['last_updated'] else None
                        )
                        self.league_stats[league_id] = stats
        except Exception as e:
            logger.warning(f"Could not load league stats: {e}")
            self.league_stats = {}
    
    def save(self):
        """Save statistics to files"""
        try:
            # Save team stats
            team_data = {team_id: stats.to_dict() for team_id, stats in self.team_stats.items()}
            with open(self.team_stats_file, 'w') as f:
                json.dump(team_data, f, indent=2)
            
            # Save league stats
            league_data = {league_id: stats.to_dict() for league_id, stats in self.league_stats.items()}
            with open(self.league_stats_file, 'w') as f:
                json.dump(league_data, f, indent=2)
            
            logger.info(f"Statistics saved: {len(self.team_stats)} teams, {len(self.league_stats)} leagues")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
    
    def record_match(self, home_team: str, away_team: str, league: str, 
                    home_score: int, away_score: int, match_date: datetime = None):
        """Record a completed match"""
        if match_date is None:
            match_date = datetime.now()
        
        # Generate IDs
        home_id = self._generate_team_id(home_team)
        away_id = self._generate_team_id(away_team)
        league_id = self._generate_league_id(league)
        
        # Update team stats
        self._update_team_stats(home_id, home_team, home_score, away_score, is_home=True)
        self._update_team_stats(away_id, away_team, away_score, home_score, is_home=False)
        
        # Update league stats
        if league_id not in self.league_stats:
            self.league_stats[league_id] = LeagueStats(
                league_id=league_id,
                league_name=league
            )
        
        self.league_stats[league_id].update(home_score, away_score)
        
        # Record match history
        match_record = {
            'date': match_date.isoformat(),
            'home_team': home_team,
            'away_team': away_team,
            'league': league,
            'home_score': home_score,
            'away_score': away_score,
            'total_goals': home_score + away_score,
            'over_25': home_score + away_score > 2.5
        }
        self.match_history.append(match_record)
        
        # Keep only last 1000 matches to avoid huge files
        if len(self.match_history) > 1000:
            self.match_history = self.match_history[-1000:]
        
        self.save()
    
    def _update_team_stats(self, team_id: str, team_name: str, 
                          goals_scored: int, goals_conceded: int, is_home: bool):
        """Update statistics for a team"""
        if team_id not in self.team_stats:
            self.team_stats[team_id] = TeamStats(
                team_id=team_id,
                team_name=team_name,
                last_updated=datetime.now()
            )
        
        stats = self.team_stats[team_id]
        stats.matches_played += 1
        stats.goals_scored += goals_scored
        stats.goals_conceded += goals_conceded
        
        if is_home:
            stats.home_matches += 1
        else:
            stats.away_matches += 1
        
        if goals_scored + goals_conceded > 2.5:
            stats.over_25_wins += 1
        else:
            stats.under_25_wins += 1
        
        stats.last_updated = datetime.now()
    
    def get_team_stats(self, team_name: str, is_home: bool = True) -> Optional[TeamStats]:
        """Get statistics for a team"""
        team_id = self._generate_team_id(team_name)
        return self.team_stats.get(team_id)
    
    def get_league_stats(self, league_name: str) -> Optional[LeagueStats]:
        """Get statistics for a league"""
        league_id = self._generate_league_id(league_name)
        return self.league_stats.get(league_id)
    
    def get_team_averages(self, team_name: str, is_home: bool = True) -> Dict[str, float]:
        """Get average statistics for a team"""
        stats = self.get_team_stats(team_name)
        
        if stats:
            return {
                'goals_scored': stats.avg_goals_scored,
                'goals_conceded': stats.avg_goals_conceded,
                'total_goals': stats.avg_total_goals,
                'over_25_rate': stats.over_25_rate
            }
        
        # Default averages if no data
        return self._get_default_averages(is_home)
    
    def _get_default_averages(self, is_home: bool) -> Dict[str, float]:
        """Get reasonable default averages"""
        if is_home:
            return {
                'goals_scored': 1.6,
                'goals_conceded': 1.2,
                'total_goals': 2.8,
                'over_25_rate': 0.55
            }
        else:
            return {
                'goals_scored': 1.3,
                'goals_conceded': 1.5,
                'total_goals': 2.8,
                'over_25_rate': 0.55
            }
    
    def _generate_team_id(self, team_name: str) -> str:
        """Generate consistent team ID from name"""
        return team_name.lower().replace(' ', '_').replace('-', '_')
    
    def _generate_league_id(self, league_name: str) -> str:
        """Generate consistent league ID from name"""
        return league_name.lower().replace(' ', '_').replace('-', '_')
