"""
Input Validator
Input validation for Angel One parameters and user inputs
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputValidator:
    """
    Input validator for Angel One parameters and user inputs
    
    This validator provides:
    - Ticker symbol validation
    - Angel One configuration validation
    - Parameter validation
    - Data format validation
    - Range validation
    """
    
    def __init__(self):
        """Initialize Input Validator"""
        self.valid_exchanges = ['NSE', 'BSE']
        self.valid_intervals = [
            'ONE_MINUTE', 'THREE_MINUTE', 'FIVE_MINUTE', 'TEN_MINUTE',
            'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'ONE_DAY'
        ]
        self.valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
        self.valid_analysis_types = [
            'short_term', 'mid_term', 'long_term', 'intraday',
            'swing', 'position', 'comprehensive'
        ]
        
        logger.info("Input Validator initialized")
    
    def validate_user_inputs(self, user_inputs: Dict[str, Any]) -> bool:
        """
        Validate user inputs
        
        Args:
            user_inputs: Dictionary with user inputs
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate ticker
            if not self.validate_ticker(user_inputs.get('ticker', '')):
                logger.error("Invalid ticker symbol")
                return False
            
            # Validate analysis type
            if not self.validate_analysis_type(user_inputs.get('analysis_type', '')):
                logger.error("Invalid analysis type")
                return False
            
            # Validate Angel One config if present
            if user_inputs.get('angel_config'):
                if not self.validate_angel_one_config(user_inputs['angel_config']):
                    logger.error("Invalid Angel One configuration")
                    return False
            
            # Validate parameters
            if not self.validate_parameters(user_inputs.get('parameters', {})):
                logger.error("Invalid parameters")
                return False
            
            logger.info("User inputs validation successful")
            return True
            
        except Exception as e:
            logger.error(f"User inputs validation failed: {e}")
            return False
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate ticker symbol
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not ticker or not isinstance(ticker, str):
                return False
            
            # Remove whitespace and convert to uppercase
            ticker = ticker.strip().upper()
            
            # Check minimum length
            if len(ticker) < 1:
                return False
            
            # Check for valid characters (letters, numbers, dots, hyphens)
            if not re.match(r'^[A-Z0-9.\-]+$', ticker):
                return False
            
            # Check for numeric-only tickers (not allowed)
            if ticker.isdigit():
                return False
            
            # Check for common invalid patterns
            invalid_patterns = ['', 'N/A', 'NULL', 'NONE']
            if ticker in invalid_patterns:
                return False
            
            logger.info(f"Ticker validation successful: {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Ticker validation failed: {e}")
            return False
    
    def validate_analysis_type(self, analysis_type: str) -> bool:
        """
        Validate analysis type
        
        Args:
            analysis_type: Analysis type string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not analysis_type or not isinstance(analysis_type, str):
                return False
            
            if analysis_type not in self.valid_analysis_types:
                logger.error(f"Invalid analysis type: {analysis_type}")
                return False
            
            logger.info(f"Analysis type validation successful: {analysis_type}")
            return True
            
        except Exception as e:
            logger.error(f"Analysis type validation failed: {e}")
            return False
    
    def validate_angel_one_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate Angel One configuration
        
        Args:
            config: Angel One configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['api_key', 'api_secret', 'access_token', 'exchange', 'interval']
            
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field: {field}")
                    return False
                
                if not config[field] or not isinstance(config[field], str):
                    logger.error(f"Invalid field value: {field}")
                    return False
            
            # Validate exchange
            if config['exchange'] not in self.valid_exchanges:
                logger.error(f"Invalid exchange: {config['exchange']}")
                return False
            
            # Validate interval
            if config['interval'] not in self.valid_intervals:
                logger.error(f"Invalid interval: {config['interval']}")
                return False
            
            # Validate API credentials format
            if not self._validate_api_credentials(config):
                return False
            
            logger.info("Angel One configuration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Angel One configuration validation failed: {e}")
            return False
    
    def _validate_api_credentials(self, config: Dict[str, Any]) -> bool:
        """
        Validate API credentials format
        
        Args:
            config: Angel One configuration
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check API key format (Angel One API keys are typically 8-10 characters)
            api_key = config.get('api_key', '')
            if len(api_key) < 6:  # Angel One API key minimum length
                logger.error("API key too short")
                return False
            
            # Check API secret format (Angel One client codes are typically 4-8 characters)
            api_secret = config.get('api_secret', '')
            if len(api_secret) < 4:  # Angel One client code minimum length
                logger.error("API secret too short")
                return False
            
            # Check access token format (Angel One client PINs are typically 4-6 characters)
            access_token = config.get('access_token', '')
            if len(access_token) < 4:  # Angel One client PIN minimum length
                logger.error("Access token too short")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"API credentials validation failed: {e}")
            return False
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate analysis parameters
        
        Args:
            parameters: Parameters dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate timeframe
            timeframe = parameters.get('timeframe', '')
            if timeframe and timeframe not in self.valid_periods:
                logger.error(f"Invalid timeframe: {timeframe}")
                return False
            
            # Validate indicators
            indicators = parameters.get('indicators', [])
            if not isinstance(indicators, list):
                logger.error("Indicators must be a list")
                return False
            
            # Validate models
            models = parameters.get('models', [])
            if not isinstance(models, list):
                logger.error("Models must be a list")
                return False
            
            # Validate strategies
            strategies = parameters.get('strategies', [])
            if not isinstance(strategies, list):
                logger.error("Strategies must be a list")
                return False
            
            logger.info("Parameters validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Parameters validation failed: {e}")
            return False
    
    def validate_data_parameters(self, data_params: Dict[str, Any]) -> bool:
        """
        Validate data parameters
        
        Args:
            data_params: Data parameters dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate ticker
            if not self.validate_ticker(data_params.get('ticker', '')):
                return False
            
            # Validate period
            period = data_params.get('period', '')
            if period and period not in self.valid_periods:
                logger.error(f"Invalid period: {period}")
                return False
            
            # Validate interval
            interval = data_params.get('interval', '')
            if interval and interval not in self.valid_intervals:
                logger.error(f"Invalid interval: {interval}")
                return False
            
            # Validate data source
            data_source = data_params.get('data_source', '')
            valid_sources = ['angel_one', 'yahoo_finance', 'both']
            if data_source and data_source not in valid_sources:
                logger.error(f"Invalid data source: {data_source}")
                return False
            
            logger.info("Data parameters validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Data parameters validation failed: {e}")
            return False
    
    def validate_enhanced_features(self, features: Dict[str, bool]) -> bool:
        """
        Validate enhanced features configuration
        
        Args:
            features: Enhanced features dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(features, dict):
                logger.error("Enhanced features must be a dictionary")
                return False
            
            # Check for valid feature keys
            valid_features = [
                'technical_indicators', 'economic_data', 'sentiment_analysis',
                'news_analysis', 'social_analysis', 'ml_models',
                'backtesting', 'risk_analysis'
            ]
            
            for key, value in features.items():
                if key not in valid_features:
                    logger.warning(f"Unknown feature: {key}")
                
                if not isinstance(value, bool):
                    logger.error(f"Feature value must be boolean: {key}")
                    return False
            
            logger.info("Enhanced features validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced features validation failed: {e}")
            return False
    
    def validate_range(self, value: Any, min_val: float, max_val: float, field_name: str) -> bool:
        """
        Validate numeric range
        
        Args:
            value: Value to validate
            min_val: Minimum value
            max_val: Maximum value
            field_name: Field name for logging
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(value, (int, float)):
                logger.error(f"{field_name} must be numeric")
                return False
            
            if value < min_val or value > max_val:
                logger.error(f"{field_name} must be between {min_val} and {max_val}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Range validation failed for {field_name}: {e}")
            return False
    
    def validate_date_format(self, date_string: str, format_string: str = "%Y-%m-%d") -> bool:
        """
        Validate date format
        
        Args:
            date_string: Date string to validate
            format_string: Expected format
            
        Returns:
            True if valid, False otherwise
        """
        try:
            datetime.strptime(date_string, format_string)
            return True
            
        except ValueError:
            logger.error(f"Invalid date format: {date_string}")
            return False
        except Exception as e:
            logger.error(f"Date validation failed: {e}")
            return False
    
    def get_validation_summary(self, user_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get validation summary
        
        Args:
            user_inputs: User inputs dictionary
            
        Returns:
            Validation summary
        """
        try:
            summary = {
                'ticker_valid': self.validate_ticker(user_inputs.get('ticker', '')),
                'analysis_type_valid': self.validate_analysis_type(user_inputs.get('analysis_type', '')),
                'angel_config_valid': True,
                'parameters_valid': self.validate_parameters(user_inputs.get('parameters', {})),
                'overall_valid': True
            }
            
            # Check Angel One config if present
            if user_inputs.get('angel_config'):
                summary['angel_config_valid'] = self.validate_angel_one_config(user_inputs['angel_config'])
            
            # Overall validation
            summary['overall_valid'] = all([
                summary['ticker_valid'],
                summary['analysis_type_valid'],
                summary['angel_config_valid'],
                summary['parameters_valid']
            ])
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get validation summary: {e}")
            return {'error': str(e)}
    
    def get_validator_status(self) -> Dict[str, Any]:
        """
        Get validator status
        
        Returns:
            Status information
        """
        try:
            status = {
                'valid_exchanges': self.valid_exchanges,
                'valid_intervals': self.valid_intervals,
                'valid_periods': self.valid_periods,
                'valid_analysis_types': self.valid_analysis_types,
                'initialized': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get validator status: {e}")
            return {'error': str(e)}
