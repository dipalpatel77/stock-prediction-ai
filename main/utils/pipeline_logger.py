"""
Pipeline Logger
Enhanced logging for the polylithic pipeline
"""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path


class UnicodeFormatter(logging.Formatter):
    """Custom formatter that handles Unicode characters properly"""
    
    def format(self, record):
        try:
            # Get the formatted message
            msg = super().format(record)
            # Replace Unicode characters that cause issues
            msg = msg.replace('🚀', '[ROCKET]')
            msg = msg.replace('📊', '[CHART]')
            msg = msg.replace('✅', '[CHECK]')
            msg = msg.replace('❌', '[CROSS]')
            msg = msg.replace('⚠️', '[WARNING]')
            msg = msg.replace('🔍', '[SEARCH]')
            msg = msg.replace('📈', '[UP]')
            msg = msg.replace('📉', '[DOWN]')
            msg = msg.replace('💰', '[MONEY]')
            msg = msg.replace('🎯', '[TARGET]')
            msg = msg.replace('🎉', '[PARTY]')
            msg = msg.replace('🇮🇳', '[INDIA]')
            msg = msg.replace('🌍', '[GLOBE]')
            msg = msg.replace('💱', '[EXCHANGE]')
            msg = msg.replace('📅', '[CALENDAR]')
            msg = msg.replace('🧪', '[TEST]')
            msg = msg.replace('🔧', '[TOOL]')
            msg = msg.replace('📋', '[CLIPBOARD]')
            msg = msg.replace('⏱️', '[CLOCK]')
            msg = msg.replace('📄', '[DOCUMENT]')
            msg = msg.replace('🏢', '[BUILDING]')
            msg = msg.replace('📝', '[NOTE]')
            msg = msg.replace('🎯', '[TARGET]')
            return msg
        except Exception:
            # Ultimate fallback
            return f"{record.levelname}: {record.getMessage()}"


class PipelineLogger:
    """
    Enhanced logging for the polylithic pipeline
    
    This logger provides:
    - Structured logging with timestamps
    - Log level management
    - File and console logging
    - Performance metrics logging
    - Error tracking and reporting
    - Log rotation and management
    """
    
    def __init__(self, name: str = "pipeline", log_level: str = "INFO", log_dir: str = "logs"):
        """
        Initialize Pipeline Logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup formatters
        self._setup_formatters()
        
        # Setup handlers
        self._setup_handlers()
        
        # Performance metrics
        self.metrics = {
            'start_time': datetime.now(),
            'operations': [],
            'errors': [],
            'warnings': []
        }
    
    def _setup_formatters(self):
        """Setup log formatters with Unicode support"""
        # Console formatter with Unicode support
        self.console_formatter = UnicodeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File formatter
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # JSON formatter for structured logging
        self.json_formatter = logging.Formatter(
            '%(message)s'
        )
    
    def _setup_handlers(self):
        """Setup log handlers"""
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.console_formatter)
        # Set encoding for console output
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.file_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance metrics handler
        metrics_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d')}.log"
        )
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(metrics_handler)
    
    def log_operation(self, operation: str, duration: float, status: str = "success", **kwargs):
        """
        Log an operation with performance metrics
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            status: Operation status (success, error, warning)
            **kwargs: Additional operation data
        """
        try:
            operation_data = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation,
                'duration': duration,
                'status': status,
                **kwargs
            }
            
            self.metrics['operations'].append(operation_data)
            
            # Log to metrics file
            self.logger.info(json.dumps(operation_data))
            
            # Log to console based on status
            if status == "error":
                self.logger.error(f"Operation '{operation}' failed after {duration:.2f}s")
            elif status == "warning":
                self.logger.warning(f"Operation '{operation}' completed with warnings after {duration:.2f}s")
            else:
                self.logger.info(f"Operation '{operation}' completed successfully in {duration:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Failed to log operation: {e}")
    
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """
        Log an error with context
        
        Args:
            error: Error message
            context: Additional context information
        """
        try:
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'error': error,
                'context': context or {}
            }
            
            self.metrics['errors'].append(error_data)
            self.logger.error(f"Error: {error}", extra={'context': context})
            
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")
    
    def error(self, message: str, context: Dict[str, Any] = None):
        """
        Log an error message (alias for log_error)
        
        Args:
            message: Error message
            context: Additional context information
        """
        self.log_error(message, context)
    
    def info(self, message: str, context: Dict[str, Any] = None):
        """
        Log an info message (alias for log_info)
        
        Args:
            message: Info message
            context: Additional context information
        """
        self.log_info(message, context)
    
    def warning(self, message: str, context: Dict[str, Any] = None):
        """
        Log a warning message (alias for log_warning)
        
        Args:
            message: Warning message
            context: Additional context information
        """
        self.log_warning(message, context)
    
    def debug(self, message: str, context: Dict[str, Any] = None):
        """
        Log a debug message (alias for log_debug)
        
        Args:
            message: Debug message
            context: Additional context information
        """
        self.log_debug(message, context)
    
    def log_warning(self, warning: str, context: Dict[str, Any] = None):
        """
        Log a warning with context
        
        Args:
            warning: Warning message
            context: Additional context information
        """
        try:
            warning_data = {
                'timestamp': datetime.now().isoformat(),
                'warning': warning,
                'context': context or {}
            }
            
            self.metrics['warnings'].append(warning_data)
            self.logger.warning(f"Warning: {warning}", extra={'context': context})
            
        except Exception as e:
            self.logger.error(f"Failed to log warning: {e}")
    
    def _clean_unicode(self, message: str) -> str:
        """Clean Unicode characters from message"""
        replacements = {
            '🚀': '[ROCKET]',
            '📊': '[CHART]',
            '✅': '[CHECK]',
            '❌': '[CROSS]',
            '⚠️': '[WARNING]',
            '🔍': '[SEARCH]',
            '📈': '[UP]',
            '📉': '[DOWN]',
            '💰': '[MONEY]',
            '🎯': '[TARGET]',
            '🎉': '[PARTY]',
            '🇮🇳': '[INDIA]',
            '🌍': '[GLOBE]',
            '💱': '[EXCHANGE]',
            '📅': '[CALENDAR]',
            '🧪': '[TEST]',
            '🔧': '[TOOL]',
            '📋': '[CLIPBOARD]',
            '⏱️': '[CLOCK]',
            '📄': '[DOCUMENT]',
            '🏢': '[BUILDING]',
            '📝': '[NOTE]'
        }
        
        for unicode_char, replacement in replacements.items():
            message = message.replace(unicode_char, replacement)
        
        return message
    
    def log_info(self, message: str, context: Dict[str, Any] = None):
        """
        Log an info message with context
        
        Args:
            message: Info message
            context: Additional context information
        """
        try:
            clean_message = self._clean_unicode(message)
            self.logger.info(f"Info: {clean_message}", extra={'context': context})
            
        except Exception as e:
            self.logger.error(f"Failed to log info: {e}")
    
    def log_debug(self, message: str, context: Dict[str, Any] = None):
        """
        Log a debug message with context
        
        Args:
            message: Debug message
            context: Additional context information
        """
        try:
            self.logger.debug(f"Debug: {message}", extra={'context': context})
            
        except Exception as e:
            self.logger.error(f"Failed to log debug: {e}")
    
    def log_performance(self, component: str, metrics: Dict[str, Any]):
        """
        Log performance metrics for a component
        
        Args:
            component: Component name
            metrics: Performance metrics
        """
        try:
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'component': component,
                'metrics': metrics
            }
            
            self.logger.info(json.dumps(performance_data))
            
        except Exception as e:
            self.logger.error(f"Failed to log performance: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get performance metrics summary
        
        Returns:
            Dictionary with metrics summary
        """
        try:
            total_operations = len(self.metrics['operations'])
            successful_operations = len([op for op in self.metrics['operations'] if op['status'] == 'success'])
            failed_operations = len([op for op in self.metrics['operations'] if op['status'] == 'error'])
            warning_operations = len([op for op in self.metrics['operations'] if op['status'] == 'warning'])
            
            total_duration = sum(op['duration'] for op in self.metrics['operations'])
            avg_duration = total_duration / total_operations if total_operations > 0 else 0
            
            summary = {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'failed_operations': failed_operations,
                'warning_operations': warning_operations,
                'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
                'total_duration': total_duration,
                'average_duration': avg_duration,
                'total_errors': len(self.metrics['errors']),
                'total_warnings': len(self.metrics['warnings']),
                'start_time': self.metrics['start_time'].isoformat(),
                'current_time': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {'error': str(e)}
    
    def save_metrics_to_file(self, filename: str = None):
        """
        Save metrics to a JSON file
        
        Args:
            filename: Optional filename for the metrics file
        """
        try:
            if filename is None:
                filename = f"{self.name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            metrics_file = self.log_dir / filename
            
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            
            self.logger.info(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def clear_metrics(self):
        """Clear all metrics"""
        try:
            self.metrics = {
                'start_time': datetime.now(),
                'operations': [],
                'errors': [],
                'warnings': []
            }
            self.logger.info("Metrics cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear metrics: {e}")
    
    def set_log_level(self, level: str):
        """
        Set logging level
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        try:
            new_level = getattr(logging, level.upper(), logging.INFO)
            self.logger.setLevel(new_level)
            
            for handler in self.logger.handlers:
                handler.setLevel(new_level)
            
            self.log_level = new_level
            self.logger.info(f"Log level set to {level}")
            
        except Exception as e:
            self.logger.error(f"Failed to set log level: {e}")
    
    def get_logger(self) -> logging.Logger:
        """
        Get the underlying logger instance
        
        Returns:
            Logger instance
        """
        return self.logger
