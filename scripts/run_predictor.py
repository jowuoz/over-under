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

async def main():
    # Initialize logger and data manager
    logger = SystemLogger()
    data_manager = DataManager()
    
    logger.info(f"🚀 Starting prediction run at {datetime.now()}")
    print(f"🚀 Starting prediction run at {datetime.now()}")
    
    try:
        # 1. Scrape live games
        scraper = ScraperManager()
        games = await scraper.get_live_games()
        logger.info(f"📊 Found {len(games)} live games")
        print(f"📊 Found {len(games)} live games")
        
        # 2. Calculate probabilities
        calculator = ProbabilityCalculator()
        predictions = []
        
        for game in games:
            prediction = calculator.predict(game)
            if prediction['confidence'] >= 0.6:  # Minimum confidence
                predictions.append(prediction)
        
        logger.info(f"🎯 Calculated {len(predictions)} predictions with confidence ≥ 0.6")
        
        # 3. Send alerts for high probability games
        notifier = TelegramNotifier()
        high_prob_games = [
            p for p in predictions 
            if p['over_2.5_probability'] >= 0.75  # Your threshold
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
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
