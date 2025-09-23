#!/usr/bin/env python3
"""
Centralized Logging System
Provides consistent logging across the entire application
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class PipelineLogger:
    """Centralized logging system for the pipeline"""
    
    def __init__(self, name: str = "pipeline", log_level: str = "INFO", 
                 log_file: Optional[str] = None, enable_console: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self.console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup console handler
        if enable_console:
            self._setup_console_handler()
        
        # Setup file handler
        if log_file:
            self._setup_file_handler(log_file)
    
    def _setup_console_handler(self):
        """Setup console handler with colors"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: str):
        """Setup file handler"""
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with additional context"""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {context}"
        return message
    
    # Convenience methods for common pipeline operations
    def data_loaded(self, ticker: str, records: int, source: str = "unknown"):
        """Log successful data loading"""
        self.info(f"✅ Data loaded for {ticker}", 
                 records=records, source=source)
    
    def data_failed(self, ticker: str, error: str, source: str = "unknown"):
        """Log data loading failure"""
        self.error(f"❌ Data loading failed for {ticker}", 
                  error=error, source=source)
    
    def model_trained(self, model_name: str, accuracy: float = None, 
                     training_time: float = None):
        """Log successful model training"""
        context = {}
        if accuracy is not None:
            context['accuracy'] = f"{accuracy:.2%}"
        if training_time is not None:
            context['training_time'] = f"{training_time:.2f}s"
        
        self.info(f"🤖 Model trained: {model_name}", **context)
    
    def model_failed(self, model_name: str, error: str):
        """Log model training failure"""
        self.error(f"❌ Model training failed: {model_name}", error=error)
    
    def prediction_generated(self, ticker: str, predictions: int, 
                           confidence: float = None):
        """Log successful prediction generation"""
        context = {'predictions': predictions}
        if confidence is not None:
            context['confidence'] = f"{confidence:.2%}"
        
        self.info(f"🔮 Predictions generated for {ticker}", **context)
    
    def api_call(self, api_name: str, endpoint: str, status_code: int = None,
                response_time: float = None):
        """Log API call"""
        context = {'endpoint': endpoint}
        if status_code is not None:
            context['status_code'] = status_code
        if response_time is not None:
            context['response_time'] = f"{response_time:.2f}s"
        
        self.info(f"📡 API call: {api_name}", **context)
    
    def database_operation(self, operation: str, table: str, 
                          records_affected: int = None, duration: float = None):
        """Log database operation"""
        context = {'table': table}
        if records_affected is not None:
            context['records'] = records_affected
        if duration is not None:
            context['duration'] = f"{duration:.2f}s"
        
        self.info(f"💾 Database {operation}", **context)
    
    def performance_metric(self, operation: str, duration: float, 
                          memory_usage: float = None, cpu_usage: float = None):
        """Log performance metrics"""
        context = {'duration': f"{duration:.2f}s"}
        if memory_usage is not None:
            context['memory'] = f"{memory_usage:.1f}MB"
        if cpu_usage is not None:
            context['cpu'] = f"{cpu_usage:.1f}%"
        
        self.info(f"⚡ Performance: {operation}", **context)

# Global logger instance
def get_logger(name: str = "pipeline", log_level: str = "INFO") -> PipelineLogger:
    """Get a logger instance"""
    return PipelineLogger(name, log_level, 
                         log_file=f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log")

# Convenience functions for backward compatibility
def log_success(message: str, details: str = ""):
    """Log success message (backward compatibility)"""
    logger = get_logger()
    logger.info(f"✅ {message}", details=details)

def log_warning(message: str, details: str = ""):
    """Log warning message (backward compatibility)"""
    logger = get_logger()
    logger.warning(f"⚠️ {message}", details=details)

def log_error(message: str, details: str = ""):
    """Log error message (backward compatibility)"""
    logger = get_logger()
    logger.error(f"❌ {message}", details=details)

# Suppress third-party library logs
def suppress_third_party_logs():
    """Suppress verbose logging from third-party libraries"""
    logging.getLogger('lightgbm').setLevel(logging.CRITICAL)
    logging.getLogger('xgboost').setLevel(logging.CRITICAL)
    logging.getLogger('catboost').setLevel(logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
    logging.getLogger('sklearn').setLevel(logging.CRITICAL)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    logging.getLogger('PIL').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('mysql.connector').setLevel(logging.WARNING)

# Initialize logging
suppress_third_party_logs()
