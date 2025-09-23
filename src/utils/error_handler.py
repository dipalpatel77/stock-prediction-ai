#!/usr/bin/env python3
"""
Centralized Error Handling System
Provides consistent error handling across the entire application
"""

import logging
import traceback
from typing import Optional, Dict, Any
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PipelineError(Exception):
    """Base exception for pipeline errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: str = None, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context
        self.error_code = error_code

class DataLoadError(PipelineError):
    """Data loading specific errors"""
    def __init__(self, message: str, ticker: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.ticker = ticker

class ModelTrainingError(PipelineError):
    """Model training specific errors"""
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model_name = model_name

class DatabaseError(PipelineError):
    """Database specific errors"""
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation

class APIError(PipelineError):
    """API specific errors"""
    def __init__(self, message: str, api_name: str = None, status_code: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.api_name = api_name
        self.status_code = status_code

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
    
    def handle_error(self, error: Exception, context: str, 
                    fallback_action: str = None, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    additional_info: Dict[str, Any] = None) -> Optional[str]:
        """
        Handle errors with proper logging and fallback actions
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            fallback_action: Action to take as fallback
            severity: Severity level of the error
            additional_info: Additional information about the error
            
        Returns:
            Fallback action if executed, None otherwise
        """
        # Log the error
        self._log_error(error, context, severity, additional_info)
        
        # Track error statistics
        self._track_error(error, context)
        
        # Execute fallback action if provided
        if fallback_action:
            logger.info(f"Executing fallback action: {fallback_action}")
            return fallback_action
        
        return None
    
    def _log_error(self, error: Exception, context: str, 
                   severity: ErrorSeverity, additional_info: Dict[str, Any] = None):
        """Log error with appropriate level based on severity"""
        error_msg = f"Error in {context}: {str(error)}"
        
        if additional_info:
            error_msg += f" | Additional Info: {additional_info}"
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(error_msg)
            logger.critical(f"Stack trace: {traceback.format_exc()}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(error_msg)
            logger.error(f"Stack trace: {traceback.format_exc()}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(error_msg)
        else:  # LOW
            logger.info(error_msg)
    
    def _track_error(self, error: Exception, context: str):
        """Track error statistics for monitoring"""
        error_key = f"{type(error).__name__}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Keep error history (last 100 errors)
        self.error_history.append({
            'error_type': type(error).__name__,
            'context': context,
            'message': str(error),
            'timestamp': logging.Formatter().formatTime(logging.LogRecord(
                name='', level=0, pathname='', lineno=0, msg='', args=(), exc_info=None
            ))
        })
        
        if len(self.error_history) > 100:
            self.error_history.pop(0)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            'error_counts': self.error_counts,
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def reset_statistics(self):
        """Reset error statistics"""
        self.error_counts.clear()
        self.error_history.clear()

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions
def handle_data_error(error: Exception, ticker: str, context: str = "data_loading") -> Optional[str]:
    """Handle data loading errors"""
    return error_handler.handle_error(
        error, 
        f"{context}:{ticker}", 
        fallback_action="Use cached data if available",
        severity=ErrorSeverity.MEDIUM,
        additional_info={'ticker': ticker}
    )

def handle_model_error(error: Exception, model_name: str, context: str = "model_training") -> Optional[str]:
    """Handle model training errors"""
    return error_handler.handle_error(
        error,
        f"{context}:{model_name}",
        fallback_action="Use alternative model",
        severity=ErrorSeverity.HIGH,
        additional_info={'model_name': model_name}
    )

def handle_database_error(error: Exception, operation: str, context: str = "database") -> Optional[str]:
    """Handle database errors"""
    return error_handler.handle_error(
        error,
        f"{context}:{operation}",
        fallback_action="Retry with exponential backoff",
        severity=ErrorSeverity.HIGH,
        additional_info={'operation': operation}
    )

def handle_api_error(error: Exception, api_name: str, context: str = "api_call") -> Optional[str]:
    """Handle API errors"""
    return error_handler.handle_error(
        error,
        f"{context}:{api_name}",
        fallback_action="Use cached data or alternative API",
        severity=ErrorSeverity.MEDIUM,
        additional_info={'api_name': api_name}
    )
