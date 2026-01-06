import logging
import os
from datetime import datetime

class SystemLogger:
    def __init__(self, log_dir="storage/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Main application log
        self.app_logger = self._setup_logger(
            "app", 
            os.path.join(log_dir, "app.log")
        )
        
        # Error log
        self.error_logger = self._setup_logger(
            "error", 
            os.path.join(log_dir, "errors.log")
        )
    
    def _setup_logger(self, name, log_file):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message):
        self.app_logger.info(message)
    
    def error(self, message):
        self.app_logger.error(message)
        self.error_logger.error(message)
