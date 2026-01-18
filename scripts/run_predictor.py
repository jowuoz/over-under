#!/usr/bin/env python3
"""
Main script that runs in GitHub Actions
"""
import sys
import os
import asyncio


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from scripts/
sys.path.insert(0, project_root)

print(f"🚀 Starting prediction run...")
print(f"📁 Project root: {project_root}")

try:
    # Import directly from submodules
    from src.scrapers.scraper_manager import ScraperManager
    print("✅ Imported ScraperManager")
except ImportError as e:
    print(f"❌ Failed to import ScraperManager: {e}")
    sys.exit(1)

try:
    from src.predictor.calculator import ProbabilityCalculator
    print("✅ Imported ProbabilityCalculator")
except ImportError as e:
    print(f"❌ Failed to import ProbabilityCalculator: {e}")
    sys.exit(1)

try:
    # Import TelegramNotifier directly from notifier module
    from src.notifier.telegram_client import TelegramNotifier
    print("✅ Imported TelegramNotifier")
except ImportError as e:
    print(f"❌ Failed to import TelegramNotifier: {e}")
    # Create a simple fallback
    class TelegramNotifier:
        async def send_alert(self, prediction):
            print(f"📨 Telegram alert (simulated): {prediction.get('home_team', 'Team1')} vs {prediction.get('away_team', 'Team2')}")
            return True
    print("⚠️ Using fallback TelegramNotifier")

# Check for other imports
try:
    from src.utils.logger import SystemLogger
    print("✅ Imported SystemLogger")
except ImportError as e:
    print(f"⚠️ Failed to import SystemLogger: {e}")
    # Simple fallback logger
    class SystemLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    print("⚠️ Using fallback SystemLogger")

try:
    from src.storage.data_manager import DataManager
    print("✅ Imported DataManager")
except ImportError as e:
    print(f"⚠️ Failed to import DataManager: {e}")
    # Simple fallback data manager
    class DataManager:
        def save_predictions(self, predictions, high_prob_games):
            import json
            os.makedirs('storage/data', exist_ok=True)
            data = {
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'high_probability_games': high_prob_games
            }
            with open('storage/data/predictions.json', 'w') as f:
                json.dump(data, f, indent=2)
            return 'storage/data/predictions.json'
    print("⚠️ Using fallback DataManager")

import json
from datetime import datetime
import shutil

from src.predictor.models import Team, League, Game, GameStatus

async def main():
    # Initialize logger and data manager
    logger = SystemLogger()
    data_manager = DataManager()
    
    logger.info(f"🚀 Starting prediction run at {datetime.now()}")
    print(f"🚀 Starting prediction run at {datetime.now()}")
    
    try:
        # 1. Scrape live games
        scraper = ScraperManager()
        result = await scraper.fetch_all_games()
        
        games = result.get('games', [])   # extract the actual list of games
        metadata = result.get('metadata', {})

        # After: games = result.get('games', [])
        converted_games = []
        
        for scraped in games:
            # Create dummy/minimal Team objects (replace placeholders with real data when available)
            home_team = Team(
                id=f"home_{scraped.home_team.lower().replace(' ', '_')}",
                name=scraped.home_team,
                avg_goals_scored=1.6,          # ← placeholder (use historical data later)
                avg_goals_conceded=1.2
            )
            
            away_team = Team(
                id=f"away_{scraped.away_team.lower().replace(' ', '_')}",
                name=scraped.away_team,
                avg_goals_scored=1.4,
                avg_goals_conceded=1.3
            )
            
            # Minimal League object
            league = League(
                id=scraped.league.lower().replace(' ', '_'),
                name=scraped.league,
                country=scraped.country,
                avg_goals_per_game=2.7,        # ← placeholder
                over_25_rate=0.55
            )
            
            # Convert to full Game object
            game = Game(
                id=scraped.id,
                home_team=home_team,
                away_team=away_team,
                league=league,
                start_time=scraped.timestamp,
                current_minute=scraped.minute or 0,
                status=GameStatus.LIVE if 'live' in scraped.status.lower() else GameStatus.SCHEDULED,
                home_score=scraped.home_score,
                away_score=scraped.away_score,
                # Add defaults for missing fields to avoid further crashes
                shots_on_target=(0, 0),
                shots_total=(0, 0),
                possession=(50.0, 50.0),
                expected_goals=(0.0, 0.0),
            )
            
            converted_games.append(game)
        
        games = converted_games  # Now games are ready for prediction
        logger.info(f"Converted {len(games)} games to full Game objects")
        
        logger.info(f"📊 Found {len(games)} live games from {metadata.get('total_scrapers', '?')} scrapers")
        print(f"📊 Found {len(games)} live games")
        
        # 2. Calculate probabilities
        calculator = ProbabilityCalculator()
        predictions = []

        # Debug print — add this block
        print(f"DEBUG - Game: {game.home_team.name} vs {game.away_team.name}")
        print(f"  Over 2.5 prob: {metrics.probability_over_25:.1%}")
        print(f"  Confidence score: {metrics.confidence_score:.2f}")
        print(f"  Expected total goals: {metrics.expected_total_goals:.2f}")
        print(f"  Confidence level: {metrics.confidence_level.value}")
        
        for game in games:
           try:
                metrics = calculator.calculate_probabilities(game)
                
                # Debug print — keep this
                print(f"DEBUG - Game: {game.home_team.name} vs {game.away_team.name}")
                print(f"  Over 2.5 prob: {metrics.probability_over_25:.1%}")
                print(f"  Confidence score: {metrics.confidence_score:.2f}")
                print(f"  Expected total goals: {metrics.expected_total_goals:.2f}")
                print(f"  Confidence level: {metrics.confidence_level.value}")
                
                # Lower threshold for testing
                if metrics.confidence_score >= 0.0:  # ← temporary 0.0 to see all
                    prediction = {
                        'home_team': game.home_team.name,
                        'away_team': game.away_team.name,
                        'over_2.5_probability': metrics.probability_over_25,
                        'confidence': metrics.confidence_score,
                        # add more if needed
                    }
                    predictions.append(prediction)
                    logger.info(f"Added prediction for {game.home_team.name} vs {game.away_team.name} (conf: {metrics.confidence_score:.2f})")
                else:
                    logger.info(f"Skipped {game.home_team.name} vs {game.away_team.name} - low confidence ({metrics.confidence_score:.2f})")
                    
           except Exception as calc_err:
                logger.error(f"Failed to calculate for {game.home_team.name} vs {game.away_team.name}: {calc_err}")
                # Optional: add fallback prediction or skip
        
        logger.info(f"🎯 Calculated {len(predictions)} predictions with confidence ≥ 0.6")
        
        # 3. Send alerts for high probability games
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        if telegram_token:
            notifier = TelegramNotifier(bot_token=telegram_token)
            logger.info("✅ Telegram notifier initialized with token")
        else:
            class DummyNotifier:
                async def send_alert(self, game):
                    logger.info(f"📨 [DUMMY] Would send alert for {game.get('home_team')} vs {game.get('away_team')}")
                    return True
            notifier = DummyNotifier()
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN not set, using dummy notifier")
        
        high_prob_games = [
            p for p in predictions 
            if p['over_2.5_probability'] >= 0.75
        ]
        
        alert_count = 0
        for game in high_prob_games:
            try:
                await notifier.send_alert(game)
                alert_count += 1
                logger.info(f"📨 Sent alert for {game['home_team']} vs {game['away_team']}")
            except Exception as e:
                logger.error(f"❌ Failed to send Telegram alert: {e}")
        
        # 4. Save results using DataManager
        saved_file = data_manager.save_predictions(predictions, high_prob_games)
        logger.info(f"💾 Saved {len(predictions)} predictions, {len(high_prob_games)} alerts to {saved_file}")
        
        # 5. Copy to reports folder for GitHub Pages
        os.makedirs('reports/data', exist_ok=True)
        shutil.copy(
            'storage/data/predictions.json',
            'reports/data/latest.json'
        )
        logger.info(f"📄 Copied data to reports/data/latest.json for dashboard")
        
        # ── NEW: Extra safety write (optional but good practice) ─────────────
        # This writes directly to latest.json even if copy fails
        try:
            with open('reports/data/latest.json', 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'predictions': predictions,
                    'high_probability_games': high_prob_games,
                    'summary': {
                        'total_games': len(games),
                        'total_predictions': len(predictions),
                        'total_alerts': len(high_prob_games),
                        'alerts_sent': alert_count
                    }
                }, f, indent=2)
            logger.info("Extra safety write to latest.json completed")
        except Exception as extra_err:
            logger.warning(f"Extra safety write failed: {extra_err}")
        
        # 6. Log completion
        completion_msg = f"✅ Run completed: {len(predictions)} predictions, {len(high_prob_games)} alerts, {alert_count} Telegram notifications sent"
        logger.info(completion_msg)
        print(completion_msg)
        
        return True
        
    except Exception as e:
        error_msg = f"❌ Error in prediction run: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        
        # ── FORCED FALLBACK JSON WRITE ───────────────────────────────────────
        # This runs even if the script crashes anywhere above
        try:
            os.makedirs('reports/data', exist_ok=True)
            fallback_data = {
                "timestamp": datetime.now().isoformat(),
                "predictions": [],
                "high_probability_games": [],
                "summary": {
                    "total_games": 0,
                    "total_predictions": 0,
                    "total_alerts": 0,
                    "alerts_sent": 0,
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                },
                "message": "No predictions this run - script failed. Check GitHub Actions log."
            }
            with open('reports/data/latest.json', 'w') as f:
                json.dump(fallback_data, f, indent=2)
            logger.info("Forced fallback JSON written to reports/data/latest.json")
            print("DEBUG: Forced fallback JSON written")
        except Exception as write_err:
            logger.error(f"Failed to write fallback JSON: {write_err}")
            print(f"DEBUG: Failed to write fallback JSON: {write_err}")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
