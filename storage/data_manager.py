import json
import os
from datetime import datetime
from typing import Dict, List, Any

class DataManager:
    def __init__(self, data_dir="storage/data"):
        self.data_dir = data_dir
        self.historical_dir = os.path.join(data_dir, "historical")
        self._safe_makedirs()
    
    def _safe_makedirs(self):
        """Safely create directories, handling file/directory conflicts"""
        try:
            # First create the base data directory
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Check if historical "directory" is actually a file
            if os.path.exists(self.historical_dir):
                if os.path.isfile(self.historical_dir):
                    # It's a file, not a directory - remove it
                    print(f"⚠️  Removing file at {self.historical_dir} (should be directory)")
                    os.remove(self.historical_dir)
                    os.makedirs(self.historical_dir, exist_ok=True)
                elif os.path.isdir(self.historical_dir):
                    # It's already a directory, do nothing
                    pass
            else:
                # Doesn't exist, create it
                os.makedirs(self.historical_dir, exist_ok=True)
                
        except FileExistsError as e:
            # Handle edge cases
            print(f"⚠️  FileExistsError: {e}")
            # Try alternative approach
            if not os.path.exists(self.historical_dir):
                os.mkdir(self.historical_dir)
    
    def save_predictions(self, predictions: List[Dict], alerts: List[Dict]):
        """Save latest predictions and archive historical data"""
        # Ensure directories exist before saving
        self._safe_makedirs()
        
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
