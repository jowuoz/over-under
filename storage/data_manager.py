import json
import os
from datetime import datetime
from typing import Dict, List, Any

class DataManager:
    def __init__(self, data_dir="storage/data"):
        self.data_dir = data_dir
        self.historical_dir = os.path.join(data_dir, "historical")
        os.makedirs(self.historical_dir, exist_ok=True)
    
    def save_predictions(self, predictions: List[Dict], alerts: List[Dict]):
        """Save latest predictions and archive historical data"""
        # Save latest predictions
        latest_data = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "alerts": alerts,
            "summary": {
                "total_games": len(predictions),
                "total_alerts": len(alerts)
            }
        }
        
        latest_file = os.path.join(self.data_dir, "predictions.json")
        with open(latest_file, 'w') as f:
            json.dump(latest_data, f, indent=2)
        
        # Archive to historical folder
        self._archive_to_historical(latest_data)
        
        return latest_file
    
    def _archive_to_historical(self, data: Dict):
        """Archive data to historical folder by month"""
        timestamp = datetime.fromisoformat(data["timestamp"])
        year_month = timestamp.strftime("%Y-%m")
        
        # Create month folder
        month_dir = os.path.join(self.historical_dir, year_month)
        os.makedirs(month_dir, exist_ok=True)
        
        # Save with timestamp filename
        filename = f"predictions_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        archive_file = os.path.join(month_dir, filename)
        
        with open(archive_file, 'w') as f:
            json.dump(data, f, indent=2)
