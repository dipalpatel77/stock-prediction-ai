"""
Simple Logger — Lightweight variant for standalone scripts.
For pipeline components use pipeline_logger.py (PipelineLogger).
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime


class SimpleLogger:
    """
    Simplified logger for the stock prediction pipeline
    
    This logger provides:
    - Basic logging with timestamps
    - File and console output
    - Simple error tracking
    """
    
    def __init__(self, name: str = "pipeline", log_level: str = "INFO"):
        """
        Initialize Simple Logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler — write to main/logs/ not the cwd
        _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(_log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(_log_dir, 'pipeline.log'), encoding='utf-8')
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
