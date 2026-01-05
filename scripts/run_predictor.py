#!/usr/bin/env python3
"""
Main script that runs in GitHub Actions
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.scrapers.scraper_manager import ScraperManager
from src.predictor.calculator import ProbabilityCalculator
from src.notifier.telegram_client import TelegramNotifier
import json
from datetime import datetime

def main():
    print(f" Starting prediction run at {datetime.now()}")
    
    # 1. Scrape live games
    scraper = ScraperManager()
    games = scraper.get_live_games()
    print(f" Found {len(games)} live games")
    
    # 2. Calculate probabilities
    calculator = ProbabilityCalculator()
    predictions = []
    
    for game in games:
        prediction = calculator.predict(game)
        if prediction['confidence'] >= 0.6:  # Minimum confidence
            predictions.append(prediction)
    
    # 3. Send alerts for high probability games
    notifier = TelegramNotifier()
    high_prob_games = [
        p for p in predictions 
        if p['over_2.5_probability'] >= 0.75  # Your threshold
    ]
    
    for game in high_prob_games:
        notifier.send_alert(game)
    
    # 4. Save results
    save_results(predictions, high_prob_games)
    
    print(f" Run completed: {len(predictions)} predictions, {len(high_prob_games)} alerts")

def save_results(predictions, high_prob_games):
    """Save results for dashboard"""
    os.makedirs('storage/data', exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions,
        'alerts': high_prob_games,
        'summary': {
            'total_games': len(predictions),
            'total_alerts': len(high_prob_games)
        }
    }
    
    with open('storage/data/predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save to reports folder for GitHub Pages
    os.makedirs('reports/data', exist_ok=True)
    with open('reports/data/latest.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
