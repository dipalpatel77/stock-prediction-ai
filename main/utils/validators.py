"""
Validators
Data and configuration validation utilities
"""

import logging
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, date
import json
from decimal import Decimal, InvalidOperation


class DataValidator:
    """
    Data validation utilities
    
    This validator provides:
    - Data type validation
    - Range validation
    - Format validation
    - Business rule validation
    - Data quality assessment
    """
    
    def __init__(self):
        """Initialize Data Validator"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Data Validator initialized")
    
    def validate_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Validate stock ticker symbol
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Validation result dictionary
        """
        try:
            if not ticker or not isinstance(ticker, str):
                return {'valid': False, 'error': 'Ticker must be a non-empty string'}
            
            ticker = ticker.strip().upper()
            
            # Check length
            if len(ticker) < 1 or len(ticker) > 20:
                return {'valid': False, 'error': 'Ticker length must be between 1 and 20 characters'}
            
            # Check for valid characters
            if not re.match(r'^[A-Z0-9.\-]+$', ticker):
                return {'valid': False, 'error': 'Ticker contains invalid characters'}
            
            # Check for numeric-only tickers
            if ticker.isdigit():
                return {'valid': False, 'error': 'Ticker cannot be numeric only'}
            
            # Check for common invalid patterns
            invalid_patterns = ['', 'N/A', 'NULL', 'NONE', 'TEST']
            if ticker in invalid_patterns:
                return {'valid': False, 'error': f'Ticker "{ticker}" is not allowed'}
            
            return {'valid': True, 'ticker': ticker}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_price(self, price: Union[float, int, str, Decimal]) -> Dict[str, Any]:
        """
        Validate price value
        
        Args:
            price: Price value
            
        Returns:
            Validation result dictionary
        """
        try:
            # Convert to Decimal for precise validation
            if isinstance(price, str):
                price = Decimal(price)
            elif isinstance(price, (int, float)):
                price = Decimal(str(price))
            elif not isinstance(price, Decimal):
                return {'valid': False, 'error': 'Price must be a number'}
            
            # Check if positive
            if price < 0:
                return {'valid': False, 'error': 'Price cannot be negative'}
            
            # Check if reasonable range (0.01 to 1,000,000)
            if price < Decimal('0.01'):
                return {'valid': False, 'error': 'Price too small (minimum 0.01)'}
            
            if price > Decimal('1000000'):
                return {'valid': False, 'error': 'Price too large (maximum 1,000,000)'}
            
            return {'valid': True, 'price': float(price)}
            
        except (ValueError, InvalidOperation) as e:
            return {'valid': False, 'error': f'Invalid price format: {str(e)}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_volume(self, volume: Union[float, int, str, Decimal]) -> Dict[str, Any]:
        """
        Validate volume value
        
        Args:
            volume: Volume value
            
        Returns:
            Validation result dictionary
        """
        try:
            # Convert to Decimal for precise validation
            if isinstance(volume, str):
                volume = Decimal(volume)
            elif isinstance(volume, (int, float)):
                volume = Decimal(str(volume))
            elif not isinstance(volume, Decimal):
                return {'valid': False, 'error': 'Volume must be a number'}
            
            # Check if non-negative
            if volume < 0:
                return {'valid': False, 'error': 'Volume cannot be negative'}
            
            # Check if reasonable range (0 to 10 billion)
            if volume > Decimal('10000000000'):
                return {'valid': False, 'error': 'Volume too large (maximum 10 billion)'}
            
            return {'valid': True, 'volume': int(volume)}
            
        except (ValueError, InvalidOperation) as e:
            return {'valid': False, 'error': f'Invalid volume format: {str(e)}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_date(self, date_value: Union[str, datetime, date], 
                     format_string: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate date value
        
        Args:
            date_value: Date value
            format_string: Expected date format
            
        Returns:
            Validation result dictionary
        """
        try:
            if isinstance(date_value, str):
                if format_string:
                    parsed_date = datetime.strptime(date_value, format_string)
                else:
                    # Try common formats
                    formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%m/%d/%Y']
                    parsed_date = None
                    
                    for fmt in formats:
                        try:
                            parsed_date = datetime.strptime(date_value, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if parsed_date is None:
                        return {'valid': False, 'error': 'Invalid date format'}
                
            elif isinstance(date_value, datetime):
                parsed_date = date_value
            elif isinstance(date_value, date):
                parsed_date = datetime.combine(date_value, datetime.min.time())
            else:
                return {'valid': False, 'error': 'Date must be a string, datetime, or date object'}
            
            # Check if date is reasonable (not too far in past or future)
            now = datetime.now()
            if parsed_date.year < 1900:
                return {'valid': False, 'error': 'Date too far in the past'}
            
            if parsed_date > now:
                return {'valid': False, 'error': 'Date cannot be in the future'}
            
            return {'valid': True, 'date': parsed_date}
            
        except ValueError as e:
            return {'valid': False, 'error': f'Invalid date format: {str(e)}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_percentage(self, percentage: Union[float, int, str, Decimal]) -> Dict[str, Any]:
        """
        Validate percentage value
        
        Args:
            percentage: Percentage value
            
        Returns:
            Validation result dictionary
        """
        try:
            # Convert to Decimal for precise validation
            if isinstance(percentage, str):
                percentage = Decimal(percentage)
            elif isinstance(percentage, (int, float)):
                percentage = Decimal(str(percentage))
            elif not isinstance(percentage, Decimal):
                return {'valid': False, 'error': 'Percentage must be a number'}
            
            # Check if reasonable range (-100% to 1000%)
            if percentage < Decimal('-100'):
                return {'valid': False, 'error': 'Percentage too low (minimum -100%)'}
            
            if percentage > Decimal('1000'):
                return {'valid': False, 'error': 'Percentage too high (maximum 1000%)'}
            
            return {'valid': True, 'percentage': float(percentage)}
            
        except (ValueError, InvalidOperation) as e:
            return {'valid': False, 'error': f'Invalid percentage format: {str(e)}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_dataframe(self, df, required_columns: List[str] = None,
                          min_rows: int = 1, max_rows: int = None) -> Dict[str, Any]:
        """
        Validate pandas DataFrame
        
        Args:
            df: DataFrame to validate
            required_columns: List of required columns
            min_rows: Minimum number of rows
            max_rows: Maximum number of rows
            
        Returns:
            Validation result dictionary
        """
        try:
            import pandas as pd
            
            if not isinstance(df, pd.DataFrame):
                return {'valid': False, 'error': 'Data must be a pandas DataFrame'}
            
            # Check if empty
            if df.empty:
                return {'valid': False, 'error': 'DataFrame is empty'}
            
            # Check row count
            if len(df) < min_rows:
                return {'valid': False, 'error': f'Too few rows (minimum {min_rows})'}
            
            if max_rows and len(df) > max_rows:
                return {'valid': False, 'error': f'Too many rows (maximum {max_rows})'}
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    return {'valid': False, 'error': f'Missing required columns: {missing_columns}'}
            
            # Check for all NaN columns
            all_nan_columns = df.columns[df.isnull().all()].tolist()
            if all_nan_columns:
                return {'valid': False, 'error': f'Columns with all NaN values: {all_nan_columns}'}
            
            return {
                'valid': True,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_json(self, json_data: Union[str, dict, list]) -> Dict[str, Any]:
        """
        Validate JSON data
        
        Args:
            json_data: JSON data to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            if isinstance(json_data, str):
                parsed_data = json.loads(json_data)
            else:
                parsed_data = json_data
            
            # Check if it's a valid JSON structure
            if not isinstance(parsed_data, (dict, list)):
                return {'valid': False, 'error': 'JSON must be an object or array'}
            
            return {'valid': True, 'data': parsed_data}
            
        except json.JSONDecodeError as e:
            return {'valid': False, 'error': f'Invalid JSON format: {str(e)}'}
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}


class ConfigValidator:
    """
    Configuration validation utilities
    
    This validator provides:
    - Configuration structure validation
    - Required field validation
    - Type validation
    - Range validation
    - Dependency validation
    """
    
    def __init__(self):
        """Initialize Config Validator"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Config Validator initialized")
    
    def validate_config(self, config: Dict[str, Any], 
                       required_fields: List[str] = None,
                       field_types: Dict[str, type] = None,
                       field_ranges: Dict[str, tuple] = None) -> Dict[str, Any]:
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary
            required_fields: List of required fields
            field_types: Dictionary mapping field names to expected types
            field_ranges: Dictionary mapping field names to (min, max) ranges
            
        Returns:
            Validation result dictionary
        """
        try:
            if not isinstance(config, dict):
                return {'valid': False, 'error': 'Config must be a dictionary'}
            
            errors = []
            
            # Check required fields
            if required_fields:
                missing_fields = [field for field in required_fields if field not in config]
                if missing_fields:
                    errors.append(f'Missing required fields: {missing_fields}')
            
            # Check field types
            if field_types:
                for field, expected_type in field_types.items():
                    if field in config:
                        if not isinstance(config[field], expected_type):
                            errors.append(f'Field "{field}" must be of type {expected_type.__name__}')
            
            # Check field ranges
            if field_ranges:
                for field, (min_val, max_val) in field_ranges.items():
                    if field in config:
                        value = config[field]
                        if isinstance(value, (int, float)):
                            if value < min_val or value > max_val:
                                errors.append(f'Field "{field}" must be between {min_val} and {max_val}')
            
            if errors:
                return {'valid': False, 'errors': errors}
            
            return {'valid': True, 'config': config}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_angel_one_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Angel One configuration
        
        Args:
            config: Angel One configuration
            
        Returns:
            Validation result dictionary
        """
        try:
            required_fields = ['api_key', 'api_secret', 'access_token', 'exchange', 'interval']
            field_types = {
                'api_key': str,
                'api_secret': str,
                'access_token': str,
                'exchange': str,
                'interval': str
            }
            
            # Validate basic structure
            result = self.validate_config(config, required_fields, field_types)
            if not result['valid']:
                return result
            
            # Validate exchange
            valid_exchanges = ['NSE', 'BSE']
            if config['exchange'] not in valid_exchanges:
                return {'valid': False, 'error': f'Invalid exchange: {config["exchange"]}'}
            
            # Validate interval
            valid_intervals = [
                'ONE_MINUTE', 'THREE_MINUTE', 'FIVE_MINUTE', 'TEN_MINUTE',
                'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'ONE_DAY'
            ]
            if config['interval'] not in valid_intervals:
                return {'valid': False, 'error': f'Invalid interval: {config["interval"]}'}
            
            # Validate API credentials length
            if len(config['api_key']) < 8:
                return {'valid': False, 'error': 'API key too short'}
            
            if len(config['api_secret']) < 8:
                return {'valid': False, 'error': 'API secret too short'}
            
            if len(config['access_token']) < 8:
                return {'valid': False, 'error': 'Access token too short'}
            
            return {'valid': True, 'config': config}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_analysis_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis configuration
        
        Args:
            config: Analysis configuration
            
        Returns:
            Validation result dictionary
        """
        try:
            required_fields = ['analysis_type', 'timeframe']
            field_types = {
                'analysis_type': str,
                'timeframe': str,
                'use_enhanced': bool,
                'use_database': bool
            }
            
            # Validate basic structure
            result = self.validate_config(config, required_fields, field_types)
            if not result['valid']:
                return result
            
            # Validate analysis type
            valid_types = [
                'short_term', 'mid_term', 'long_term', 'intraday',
                'swing', 'position', 'comprehensive'
            ]
            if config['analysis_type'] not in valid_types:
                return {'valid': False, 'error': f'Invalid analysis type: {config["analysis_type"]}'}
            
            # Validate timeframe
            valid_timeframes = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
            if config['timeframe'] not in valid_timeframes:
                return {'valid': False, 'error': f'Invalid timeframe: {config["timeframe"]}'}
            
            return {'valid': True, 'config': config}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def validate_data_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data configuration
        
        Args:
            config: Data configuration
            
        Returns:
            Validation result dictionary
        """
        try:
            required_fields = ['ticker', 'period', 'interval']
            field_types = {
                'ticker': str,
                'period': str,
                'interval': str,
                'data_source': str
            }
            
            # Validate basic structure
            result = self.validate_config(config, required_fields, field_types)
            if not result['valid']:
                return result
            
            # Validate ticker
            data_validator = DataValidator()
            ticker_result = data_validator.validate_ticker(config['ticker'])
            if not ticker_result['valid']:
                return {'valid': False, 'error': f'Ticker validation failed: {ticker_result["error"]}'}
            
            # Validate period
            valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
            if config['period'] not in valid_periods:
                return {'valid': False, 'error': f'Invalid period: {config["period"]}'}
            
            # Validate interval
            valid_intervals = [
                '1m', '5m', '15m', '30m', '1h', '1d',
                'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE',
                'ONE_HOUR', 'ONE_DAY'
            ]
            if config['interval'] not in valid_intervals:
                return {'valid': False, 'error': f'Invalid interval: {config["interval"]}'}
            
            # Validate data source
            if 'data_source' in config:
                valid_sources = ['angel_one', 'yahoo_finance', 'both']
                if config['data_source'] not in valid_sources:
                    return {'valid': False, 'error': f'Invalid data source: {config["data_source"]}'}
            
            return {'valid': True, 'config': config}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
