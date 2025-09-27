"""
Error Handler
Enhanced error handling for the polylithic pipeline
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    DATA = "data"
    API = "api"
    DATABASE = "database"
    PROCESSING = "processing"
    VALIDATION = "validation"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorHandler:
    """
    Enhanced error handling for the polylithic pipeline
    
    This handler provides:
    - Error categorization and severity assessment
    - Error recovery mechanisms
    - Error reporting and logging
    - Fallback strategies
    - Error metrics and analytics
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize Error Handler
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_history = []
        self.recovery_strategies = {}
        self.fallback_handlers = {}
        
        # Setup default recovery strategies
        self._setup_default_recovery_strategies()
        
        self.logger.info("Error Handler initialized")
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies"""
        self.recovery_strategies = {
            ErrorCategory.API: self._handle_api_error,
            ErrorCategory.DATABASE: self._handle_database_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.DATA: self._handle_data_error,
            ErrorCategory.PROCESSING: self._handle_processing_error,
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.CONFIGURATION: self._handle_configuration_error,
            ErrorCategory.UNKNOWN: self._handle_unknown_error
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None, 
                    category: ErrorCategory = None, severity: ErrorSeverity = None) -> Dict[str, Any]:
        """
        Handle an error with appropriate recovery strategies
        
        Args:
            error: Exception instance
            context: Additional context information
            category: Error category (auto-detected if None)
            severity: Error severity (auto-assessed if None)
            
        Returns:
            Dictionary with error handling results
        """
        try:
            # Auto-detect category if not provided
            if category is None:
                category = self._detect_error_category(error)
            
            # Auto-assess severity if not provided
            if severity is None:
                severity = self._assess_error_severity(error, category)
            
            # Create error record
            error_record = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'category': category.value,
                'severity': severity.value,
                'context': context or {},
                'traceback': traceback.format_exc()
            }
            
            # Add to error history
            self.error_history.append(error_record)
            
            # Log error
            self._log_error(error_record)
            
            # Apply recovery strategy
            recovery_result = self._apply_recovery_strategy(error, category, context)
            
            # Return handling result
            return {
                'handled': True,
                'category': category.value,
                'severity': severity.value,
                'recovery_applied': recovery_result['success'],
                'recovery_strategy': recovery_result['strategy'],
                'fallback_used': recovery_result.get('fallback_used', False),
                'error_record': error_record
            }
            
        except Exception as e:
            self.logger.error(f"Failed to handle error: {e}")
            return {
                'handled': False,
                'error': str(e),
                'original_error': str(error)
            }
    
    def _detect_error_category(self, error: Exception) -> ErrorCategory:
        """Detect error category from exception"""
        try:
            error_type = type(error).__name__.lower()
            error_message = str(error).lower()
            
            # API errors
            if any(keyword in error_type or keyword in error_message for keyword in 
                   ['connection', 'timeout', 'request', 'http', 'api', 'rate']):
                return ErrorCategory.API
            
            # Database errors
            elif any(keyword in error_type or keyword in error_message for keyword in 
                     ['database', 'sql', 'mysql', 'sqlite', 'connection']):
                return ErrorCategory.DATABASE
            
            # Network errors
            elif any(keyword in error_type or keyword in error_message for keyword in 
                     ['network', 'socket', 'dns', 'connection']):
                return ErrorCategory.NETWORK
            
            # Data errors
            elif any(keyword in error_type or keyword in error_message for keyword in 
                     ['data', 'value', 'type', 'format', 'missing']):
                return ErrorCategory.DATA
            
            # Validation errors
            elif any(keyword in error_type or keyword in error_message for keyword in 
                     ['validation', 'invalid', 'required', 'format']):
                return ErrorCategory.VALIDATION
            
            # Configuration errors
            elif any(keyword in error_type or keyword in error_message for keyword in 
                     ['config', 'setting', 'parameter', 'option']):
                return ErrorCategory.CONFIGURATION
            
            # Processing errors
            elif any(keyword in error_type or keyword in error_message for keyword in 
                     ['processing', 'calculation', 'algorithm', 'model']):
                return ErrorCategory.PROCESSING
            
            else:
                return ErrorCategory.UNKNOWN
                
        except Exception:
            return ErrorCategory.UNKNOWN
    
    def _assess_error_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity"""
        try:
            error_type = type(error).__name__
            error_message = str(error).lower()
            
            # Critical errors
            if any(keyword in error_message for keyword in 
                   ['critical', 'fatal', 'system', 'memory', 'disk']):
                return ErrorSeverity.CRITICAL
            
            # High severity errors
            elif any(keyword in error_message for keyword in 
                     ['connection', 'timeout', 'database', 'api']):
                return ErrorSeverity.HIGH
            
            # Medium severity errors
            elif any(keyword in error_message for keyword in 
                     ['validation', 'format', 'missing', 'invalid']):
                return ErrorSeverity.MEDIUM
            
            # Low severity errors
            else:
                return ErrorSeverity.LOW
                
        except Exception:
            return ErrorSeverity.MEDIUM
    
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error with appropriate level"""
        try:
            severity = error_record['severity']
            message = f"Error in {error_record['category']}: {error_record['error_message']}"
            
            if severity == 'critical':
                self.logger.critical(message, extra={'error_record': error_record})
            elif severity == 'high':
                self.logger.error(message, extra={'error_record': error_record})
            elif severity == 'medium':
                self.logger.warning(message, extra={'error_record': error_record})
            else:
                self.logger.info(message, extra={'error_record': error_record})
                
        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")
    
    def _apply_recovery_strategy(self, error: Exception, category: ErrorCategory, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply appropriate recovery strategy"""
        try:
            strategy_func = self.recovery_strategies.get(category, self._handle_unknown_error)
            return strategy_func(error, context)
            
        except Exception as e:
            self.logger.error(f"Failed to apply recovery strategy: {e}")
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_api_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API errors"""
        try:
            # Try fallback data source
            if 'fallback_source' in context:
                self.logger.info(f"Using fallback data source: {context['fallback_source']}")
                return {'success': True, 'strategy': 'fallback_source', 'fallback_used': True}
            
            # Try retry with backoff
            if 'retry_count' not in context:
                context['retry_count'] = 0
            
            if context['retry_count'] < 3:
                context['retry_count'] += 1
                self.logger.info(f"Retrying API call (attempt {context['retry_count']})")
                return {'success': True, 'strategy': 'retry', 'retry_count': context['retry_count']}
            
            return {'success': False, 'strategy': 'none', 'error': 'Max retries exceeded'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_database_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database errors"""
        try:
            # Try connection reset
            if 'connection_reset' in context:
                self.logger.info("Attempting database connection reset")
                return {'success': True, 'strategy': 'connection_reset'}
            
            # Try alternative database
            if 'alternative_db' in context:
                self.logger.info(f"Using alternative database: {context['alternative_db']}")
                return {'success': True, 'strategy': 'alternative_db', 'fallback_used': True}
            
            return {'success': False, 'strategy': 'none', 'error': 'No database recovery options'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_network_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network errors"""
        try:
            # Try different network interface
            if 'network_interface' in context:
                self.logger.info(f"Switching network interface: {context['network_interface']}")
                return {'success': True, 'strategy': 'network_switch'}
            
            # Try offline mode
            if 'offline_mode' in context:
                self.logger.info("Switching to offline mode")
                return {'success': True, 'strategy': 'offline_mode', 'fallback_used': True}
            
            return {'success': False, 'strategy': 'none', 'error': 'No network recovery options'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_data_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data errors"""
        try:
            # Try data cleaning
            if 'data_cleaning' in context:
                self.logger.info("Attempting data cleaning")
                return {'success': True, 'strategy': 'data_cleaning'}
            
            # Try alternative data source
            if 'alternative_data' in context:
                self.logger.info("Using alternative data source")
                return {'success': True, 'strategy': 'alternative_data', 'fallback_used': True}
            
            return {'success': False, 'strategy': 'none', 'error': 'No data recovery options'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_processing_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle processing errors"""
        try:
            # Try simplified processing
            if 'simplified_processing' in context:
                self.logger.info("Using simplified processing")
                return {'success': True, 'strategy': 'simplified_processing', 'fallback_used': True}
            
            # Try alternative algorithm
            if 'alternative_algorithm' in context:
                self.logger.info("Using alternative algorithm")
                return {'success': True, 'strategy': 'alternative_algorithm', 'fallback_used': True}
            
            return {'success': False, 'strategy': 'none', 'error': 'No processing recovery options'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation errors"""
        try:
            # Try data correction
            if 'data_correction' in context:
                self.logger.info("Attempting data correction")
                return {'success': True, 'strategy': 'data_correction'}
            
            # Try relaxed validation
            if 'relaxed_validation' in context:
                self.logger.info("Using relaxed validation")
                return {'success': True, 'strategy': 'relaxed_validation', 'fallback_used': True}
            
            return {'success': False, 'strategy': 'none', 'error': 'No validation recovery options'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_configuration_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configuration errors"""
        try:
            # Try default configuration
            if 'default_config' in context:
                self.logger.info("Using default configuration")
                return {'success': True, 'strategy': 'default_config', 'fallback_used': True}
            
            # Try configuration reset
            if 'config_reset' in context:
                self.logger.info("Resetting configuration")
                return {'success': True, 'strategy': 'config_reset'}
            
            return {'success': False, 'strategy': 'none', 'error': 'No configuration recovery options'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def _handle_unknown_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown errors"""
        try:
            # Try generic fallback
            if 'generic_fallback' in context:
                self.logger.info("Using generic fallback")
                return {'success': True, 'strategy': 'generic_fallback', 'fallback_used': True}
            
            return {'success': False, 'strategy': 'none', 'error': 'No recovery strategy for unknown error'}
            
        except Exception as e:
            return {'success': False, 'strategy': 'none', 'error': str(e)}
    
    def add_recovery_strategy(self, category: ErrorCategory, strategy_func: Callable):
        """
        Add custom recovery strategy
        
        Args:
            category: Error category
            strategy_func: Recovery strategy function
        """
        try:
            self.recovery_strategies[category] = strategy_func
            self.logger.info(f"Added recovery strategy for {category.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to add recovery strategy: {e}")
    
    def add_fallback_handler(self, category: ErrorCategory, fallback_func: Callable):
        """
        Add fallback handler
        
        Args:
            category: Error category
            fallback_func: Fallback handler function
        """
        try:
            self.fallback_handlers[category] = fallback_func
            self.logger.info(f"Added fallback handler for {category.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to add fallback handler: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary statistics
        
        Returns:
            Dictionary with error summary
        """
        try:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {'total_errors': 0, 'categories': {}, 'severities': {}}
            
            # Count by category
            categories = {}
            for error in self.error_history:
                category = error['category']
                categories[category] = categories.get(category, 0) + 1
            
            # Count by severity
            severities = {}
            for error in self.error_history:
                severity = error['severity']
                severities[severity] = severities.get(severity, 0) + 1
            
            # Recent errors (last 24 hours)
            recent_errors = [
                error for error in self.error_history
                if (datetime.now() - datetime.fromisoformat(error['timestamp'])).days < 1
            ]
            
            return {
                'total_errors': total_errors,
                'recent_errors': len(recent_errors),
                'categories': categories,
                'severities': severities,
                'last_error': self.error_history[-1] if self.error_history else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get error summary: {e}")
            return {'error': str(e)}
    
    def clear_error_history(self):
        """Clear error history"""
        try:
            self.error_history.clear()
            self.logger.info("Error history cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear error history: {e}")
    
    def export_error_report(self, filename: str = None) -> str:
        """
        Export error report to file
        
        Args:
            filename: Optional filename for the report
            
        Returns:
            Path to the exported file
        """
        try:
            if filename is None:
                filename = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_data = {
                'summary': self.get_error_summary(),
                'error_history': self.error_history,
                'exported_at': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Error report exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
            return None
