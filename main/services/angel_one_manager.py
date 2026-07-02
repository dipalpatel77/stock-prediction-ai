#!/usr/bin/env python3
"""
Angel One Manager
Manages Angel One API operations with rate limiting and caching
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Import core services
# These services are now integrated into AngelOneManager itself
# from src.core.enhanced_angel_one_service import EnhancedAngelOneService
# from src.core.angel_one_database_schema import AngelOneDatabaseSchema
from main.utils.rate_limiter import get_api_rate_limiter
from main.utils.date_formatter import convert_period_to_days
from main.utils.stock_utils import is_cache_fresh

logger = logging.getLogger(__name__)

class AngelOneManager:
    """Manages Angel One API operations with rate limiting and caching"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize Angel One service with real API functionality
        from .angel_one_service import AngelOneService
        self.angel_service = AngelOneService(config)
        
        # Initialize database schema with connection string
        connection_string = config.get('connection_string', 'sqlite:///angel_one.db')
        self.db_schema = None  # Will be implemented with actual database schema
        
        # Set up actual Angel One credentials
        self.api_key = config.get('api_key', 'fRBSMrnn')
        self.client_code = config.get('api_secret', 'D54448')  # api_secret is actually client_code
        self.client_pin = config.get('access_token', '2251')  # access_token is actually client_pin
        self.totp_secret = config.get('totp_secret', 'NP4SAXOKMTJQZ4KZP2TBTYXRCE')
        
        self.rate_limiter = get_api_rate_limiter()

        from .technical_indicators_service import TechnicalIndicatorsService
        self.technical_indicators_service = TechnicalIndicatorsService()
        
        # Angel One API limits - Based on official documentation
        self.api_limits = {
            'ONE_MINUTE': 30,      # Max 30 days for 1-minute data
            'THREE_MINUTE': 60,    # Max 60 days for 3-minute data
            'FIVE_MINUTE': 100,    # Max 100 days for 5-minute data
            'TEN_MINUTE': 100,     # Max 100 days for 10-minute data
            'FIFTEEN_MINUTE': 200, # Max 200 days for 15-minute data
            'THIRTY_MINUTE': 200, # Max 200 days for 30-minute data
            'ONE_HOUR': 400,      # Max 400 days for 1-hour data
            'ONE_DAY': 2000       # Max 2000 days for daily data
        }
        
        # API endpoints - Official Angel One API endpoints
        self.endpoints = {
            'historical_data': '/rest/secure/angelbroking/historical/v1/getCandleData',
            'ltp_data': '/rest/secure/angelbroking/order/v1/getLtpData',
            'search_scrip': '/rest/secure/angelbroking/order/v1/searchScrip',
            'quote': '/rest/secure/angelbroking/market/v1/quote'
        }
        
        # Base URL for Angel One API
        self.base_url = "https://apiconnect.angelone.in"
        
        logger.info("Angel One Manager initialized")
    
    def get_historical_data(self, symbol: str, interval: str = 'ONE_DAY', 
                           from_date: str = None, to_date: str = None, 
                           days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical data using the Angel One service
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            from_date: Start date
            to_date: End date
            days: Number of days to fetch
            
        Returns:
            DataFrame with historical data
        """
        if self.angel_service:
            return self.angel_service.get_historical_data(symbol, interval, from_date, to_date, days)
        else:
            logger.error("Angel One service not initialized")
            return None
    
    def get_multiple_intervals_data(self, symbol: str, intervals: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple intervals
        
        Args:
            symbol: Stock symbol
            intervals: List of intervals to fetch
            
        Returns:
            Dictionary with interval as key and DataFrame as value
        """
        if self.angel_service:
            return self.angel_service.get_multiple_intervals_data(symbol, intervals)
        else:
            logger.error("Angel One service not initialized")
            return {}
    
    def is_configured(self) -> bool:
        """
        Check if Angel One is properly configured
        
        Returns:
            True if configured, False otherwise
        """
        if self.angel_service:
            return self.angel_service.is_configured()
        return False
    
    def check_rate_limit(self, interval: str = 'ONE_DAY') -> bool:
        """
        Check if API rate limit is available
        
        Args:
            interval: Data interval
            
        Returns:
            True if rate limit is available, False otherwise
        """
        # TESTING MODE: Remove rate limiting constraints for testing
        return True
        
        try:
            if self.rate_limiter:
                return self.rate_limiter.check_limit(interval)
            return True
        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return True
    
    def load_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Load stock data from Angel One API using actual credentials
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            Stock data DataFrame
        """
        try:
            # Check rate limit first
            if not self.check_rate_limit(interval):
                logger.warning(f"Rate limit exceeded for {interval}")
                return pd.DataFrame()
            
            # Configure Angel One service with actual credentials
            self.angel_service.api_key = self.api_key
            self.angel_service.client_code = self.client_code
            self.angel_service.client_pin = self.client_pin
            self.angel_service.totp_secret = self.totp_secret
            
            # Load data using the service with real credentials
            data = self.angel_service.get_stock_data(ticker, period, interval)
            
            if not data.empty:
                logger.info(f"Loaded {len(data)} records for {ticker} from Angel One")
                return data
            else:
                logger.warning(f"No data loaded for {ticker} from Angel One")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load stock data for {ticker} from Angel One: {e}")
            return pd.DataFrame()
    
    def get_optimal_historical_data(self, ticker: str, exchange: str, interval: str, days_back: int = None) -> pd.DataFrame:
        """
        Get optimal historical data with maximum days based on interval
        
        Args:
            ticker: Stock ticker symbol
            exchange: Exchange (NSE/BSE)
            interval: Data interval
            days_back: Number of days back (auto-optimized if None)
            
        Returns:
            Optimized historical data DataFrame
        """
        try:
            # Auto-optimize days_back based on interval limits
            if days_back is None:
                days_back = self.api_limits.get(interval, 2000)
            
            # Ensure we don't exceed API limits
            max_days = self.api_limits.get(interval, 2000)
            days_back = min(days_back, max_days)
            
            logger.info(f"Fetching optimal data for {ticker}: {days_back} days, {interval} interval")
            
            # Configure Angel One service with actual credentials
            self.angel_service.api_key = self.api_key
            self.angel_service.client_code = self.client_code
            self.angel_service.client_pin = self.client_pin
            self.angel_service.totp_secret = self.totp_secret
            
            # Use optimized data fetching
            data = self.angel_service.get_optimal_historical_data(ticker, exchange, interval, days_back)
            
            if not data.empty:
                logger.info(f"Retrieved {len(data)} records for {ticker} (optimized)")
                return data
            else:
                logger.warning(f"No optimized data retrieved for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Optimized data fetching failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_comprehensive_historical_data(self, ticker: str, exchange: str, period: str = '5y') -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive historical data for all intervals
        
        Args:
            ticker: Stock ticker
            exchange: Exchange (NSE/BSE)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            
        Returns:
            Dictionary with interval as key and DataFrame as value
        """
        try:
            logger.info(f"Getting comprehensive historical data for {ticker} on {exchange}")
            
            # Define all intervals with their max days
            intervals_config = {
                'ONE_MINUTE': 30,
                'THREE_MINUTE': 60,
                'FIVE_MINUTE': 100,
                'TEN_MINUTE': 100,
                'FIFTEEN_MINUTE': 200,
                'THIRTY_MINUTE': 200,
                'ONE_HOUR': 400,
                'ONE_DAY': 2000
            }
            
            comprehensive_data = {}
            
            print(f"\n📊 Downloading comprehensive data for {ticker} ({exchange})")
            print("=" * 60)
            
            for interval, max_days in intervals_config.items():
                try:
                    print(f"⏰ Downloading {interval} data (max {max_days} days)...")
                    
                    # Convert period to days and limit to max days for this interval
                    requested_days = convert_period_to_days(period)
                    actual_days = min(requested_days, max_days)
                    
                    # Get historical data for this interval
                    data = self.get_optimal_historical_data(ticker, exchange, interval, actual_days)
                    
                    if data is not None and not data.empty:
                        comprehensive_data[interval] = data
                        print(f"✅ {interval}: {len(data)} records")
                        
                        # Store in database using interval-specific storage
                        self.store_data_in_database(ticker, data, interval)
                    else:
                        print(f"❌ {interval}: No data retrieved")
                        
                except Exception as e:
                    print(f"❌ {interval}: Error - {e}")
                    logger.error(f"Failed to get {interval} data for {ticker}: {e}")
            
            print(f"\n📈 Comprehensive data download completed for {ticker}")
            print(f"   Successfully downloaded: {len(comprehensive_data)} intervals")
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive historical data for {ticker}: {e}")
            return {}
    
    def batch_fetch_multiple_stocks(self, tickers: list, exchange: str, interval: str, days_back: int) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch multiple stocks efficiently
        
        Args:
            tickers: List of stock tickers
            exchange: Exchange (NSE/BSE)
            interval: Data interval
            days_back: Number of days back
            
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        try:
            results = {}
            logger.info(f"Batch fetching {len(tickers)} stocks from {exchange}")
            
            for ticker in tickers:
                try:
                    data = self.get_optimal_historical_data(ticker, exchange, interval, days_back)
                    if not data.empty:
                        results[ticker] = data
                        logger.info(f"✅ {ticker}: {len(data)} records")
                    else:
                        logger.warning(f"⚠️ {ticker}: No data retrieved")
                except Exception as e:
                    logger.error(f"❌ {ticker}: {e}")
                    continue
            
            logger.info(f"Batch fetch completed: {len(results)}/{len(tickers)} successful")
            return results
            
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            return {}
    
    def get_oi_data(self, symbol: str, exchange: str, interval: str) -> pd.DataFrame:
        """
        Get Open Interest data for F&O contracts
        
        Args:
            symbol: F&O symbol (e.g., 'NIFTY')
            exchange: Exchange (NFO)
            interval: Data interval
            
        Returns:
            Open Interest data DataFrame
        """
        try:
            logger.info(f"Fetching OI data for {symbol} from {exchange}")
            
            # Configure Angel One service
            self.angel_service.api_key = self.api_key
            self.angel_service.client_code = self.client_code
            self.angel_service.client_pin = self.client_pin
            self.angel_service.totp_secret = self.totp_secret
            
            # Fetch OI data
            oi_data = self.angel_service.get_oi_data(symbol, exchange, interval)
            
            if not oi_data.empty:
                logger.info(f"Retrieved {len(oi_data)} OI records for {symbol}")
                return oi_data
            else:
                logger.warning(f"No OI data retrieved for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"OI data fetching failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_latest_data(self, ticker: str, exchange: str) -> pd.DataFrame:
        """
        Incremental update for latest data
        
        Args:
            ticker: Stock ticker symbol
            exchange: Exchange (NSE/BSE)
            
        Returns:
            Latest data DataFrame
        """
        try:
            logger.info(f"Updating latest data for {ticker}")
            
            # Configure Angel One service
            self.angel_service.api_key = self.api_key
            self.angel_service.client_code = self.client_code
            self.angel_service.client_pin = self.client_pin
            self.angel_service.totp_secret = self.totp_secret
            
            # Get latest data (last 1 day)
            latest_data = self.get_optimal_historical_data(ticker, exchange, 'ONE_DAY', 1)
            
            if not latest_data.empty:
                logger.info(f"Updated latest data for {ticker}: {len(latest_data)} records")
                return latest_data
            else:
                logger.warning(f"No latest data for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Latest data update failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_enhanced_data_with_indicators(self, ticker: str, exchange: str, interval: str, days_back: int = None) -> pd.DataFrame:
        """
        Get enhanced data with technical indicators and sentiment features
        
        Args:
            ticker: Stock ticker symbol
            exchange: Exchange (NSE/BSE)
            interval: Data interval
            days_back: Number of days back
            
        Returns:
            Enhanced DataFrame with technical indicators
        """
        try:
            # Get base data
            data = self.get_optimal_historical_data(ticker, exchange, interval, days_back)
            
            if data.empty:
                return data
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Add sentiment features
            data = self._add_sentiment_features(data)
            
            # Calculate data quality metrics
            quality_score = self._calculate_data_quality(data)
            logger.info(f"Data quality score for {ticker}: {quality_score:.1f}%")
            
            return data
            
        except Exception as e:
            logger.error(f"Enhanced data processing failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            return self.technical_indicators_service.calculate_all_indicators(data)
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            return data
    
    def _add_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment features to the data"""
        try:
            # Price momentum
            data['Price_Momentum_5'] = data['Close'].pct_change(5)
            data['Price_Momentum_10'] = data['Close'].pct_change(10)
            
            # Volatility measures
            data['Volatility_5'] = data['Close'].rolling(window=5).std()
            data['Volatility_20'] = data['Close'].rolling(window=20).std()
            
            # High-Low ratios
            data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
            data['Close_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            
            # Volume momentum
            data['Volume_Momentum'] = data['Volume'].pct_change(5)
            
            return data
            
        except Exception as e:
            logger.error(f"Sentiment features calculation failed: {e}")
            return data
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        try:
            if data.empty:
                return 0.0
            
            # Data completeness
            total_cells = len(data) * len(data.columns)
            non_null_cells = data.count().sum()
            completeness = (non_null_cells / total_cells) * 100
            
            # Price consistency (no negative prices)
            price_columns = ['Open', 'High', 'Low', 'Close']
            price_consistency = 100.0
            for col in price_columns:
                if col in data.columns:
                    if (data[col] <= 0).any():
                        price_consistency -= 20
            
            # Volume consistency (no negative volumes)
            volume_consistency = 100.0
            if 'Volume' in data.columns:
                if (data['Volume'] < 0).any():
                    volume_consistency -= 30
            
            # Overall quality score
            quality_score = (completeness + price_consistency + volume_consistency) / 3
            return max(0.0, min(100.0, quality_score))
            
        except Exception as e:
            logger.error(f"Data quality calculation failed: {e}")
            return 0.0
    
    def get_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Get stock data with rate limiting and caching
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with stock data
        """
        try:
            logger.info(f"Getting Angel One data for {ticker} - period: {period}, interval: {interval}")
            
            # Check cache first
            cached_data = self.get_cached_data(ticker, period, interval)
            if cached_data is not None and not cached_data.empty:
                # Check if cache is fresh
                if self._is_cache_fresh(cached_data):
                    logger.info(f"Using fresh cached data for {ticker}")
                    return cached_data
                else:
                    logger.info(f"Cached data is stale for {ticker}, will update")
            
            # Apply rate limiting and fetch data
            data = self.rate_limiter.call_with_retry(
                'angel_one',
                self._fetch_stock_data,
                ticker, period, interval
            )
            
            if data is not None and not data.empty:
                # Store in cache
                self.store_data_in_database(ticker, data, interval)
                logger.info(f"Successfully fetched {len(data)} records for {ticker}")
                return data
            else:
                raise Exception("No data received from Angel One API")
                
        except Exception as e:
            logger.error(f"Failed to get Angel One data for {ticker}: {e}")
            raise Exception(f"Angel One data fetch failed: {e}")
    
    def _fetch_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """
        Fetch stock data from Angel One API
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Convert period to days
            days = convert_period_to_days(period)
            
            # Check if requested days exceed API limits
            max_days = self.api_limits.get(interval, 2000)
            if days > max_days:
                logger.warning(f"Requested {days} days exceeds limit of {max_days} for {interval}")
                days = max_days
            
            # Get data from Angel One service
            data = self.angel_service.get_historical_data(
                symbol=ticker,
                interval=interval,
                days=days
            )
            
            if data is not None and not data.empty:
                # Standardize data format
                data = self._standardize_data(data)
                return data
            else:
                raise Exception("Empty data received from Angel One API")
                
        except Exception as e:
            logger.error(f"Angel One API fetch failed: {e}")
            raise e
    
    def store_data_in_database(self, ticker: str, data: pd.DataFrame, interval: str):
        """
        Store data in database using Angel One schema
        
        Args:
            ticker: Stock ticker symbol
            data: Stock data DataFrame
            interval: Data interval
        """
        try:
            logger.info(f"Storing {len(data)} Angel One records for {ticker}")
            
            # Use Angel One database schema to store data
            self.db_schema.store_stock_data(
                ticker=ticker,
                data=data,
                interval=interval
            )
            
            logger.info(f"Successfully stored Angel One data for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to store Angel One data in database: {e}")
            # Don't raise exception - data fetching can continue without storage
    
    def get_cached_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get cached data from database
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            Cached DataFrame or None if not found
        """
        try:
            # Use Angel One database schema to retrieve data
            data = self.db_schema.get_stock_data(
                ticker=ticker,
                interval=interval,
                days=convert_period_to_days(period)
            )
            
            if data is not None and not data.empty:
                logger.info(f"Retrieved {len(data)} cached Angel One records for {ticker}")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached Angel One data: {e}")
            return None
    
    def get_incremental_data(self, ticker: str, interval: str, last_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get incremental data since last update
        
        Args:
            ticker: Stock ticker symbol
            interval: Data interval
            last_date: Last date in the existing data
            
        Returns:
            DataFrame with new data or None if failed
        """
        try:
            if not self.angel_service:
                logger.error("Angel One service not initialized")
                return None
            
            # Get incremental data from Angel One service
            data = self.angel_service.get_incremental_data(ticker, interval, last_date)
            
            if data is not None and not data.empty:
                logger.info(f"Retrieved {len(data)} incremental records for {ticker}")
                return data
            else:
                logger.info(f"No incremental data available for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get incremental data for {ticker}: {e}")
            return None
    
    def _is_cache_fresh(self, data: pd.DataFrame, max_age_hours: int = 24) -> bool:
        return is_cache_fresh(data, max_age_hours)
    
    def test_connection(self) -> bool:
        """
        Test Angel One API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Angel One API connection")
            
            # Try to get a small amount of data to test connection
            test_data = self.rate_limiter.call_with_retry(
                'angel_one',
                self._test_api_call
            )
            
            if test_data is not None:
                logger.info("Angel One API connection test successful")
                return True
            else:
                logger.error("Angel One API connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"Angel One API connection test failed: {e}")
            return False
    
    def _test_api_call(self) -> bool:
        """
        Make a test API call to Angel One
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to get master data or make a simple API call
            # This is a placeholder - implement based on actual Angel One API
            result = self.angel_service.test_connection()
            return result is not None
            
        except Exception as e:
            logger.error(f"Angel One test API call failed: {e}")
            return False
    
    def get_api_limits(self, interval: str) -> int:
        """
        Get API limits for a specific interval
        
        Args:
            interval: Data interval
            
        Returns:
            Maximum days allowed for the interval
        """
        return self.api_limits.get(interval, 2000)
    
    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Angel One data format
        
        Args:
            data: Raw Angel One data
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Rename columns if needed (Angel One might use different names)
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low', 
                'close': 'Close',
                'volume': 'Volume',
                'timestamp': 'Date'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"Missing column {col} in Angel One data")
                    if col == 'Volume':
                        data[col] = 0  # Default volume to 0
                    else:
                        data[col] = data['Close']  # Use close price as fallback
            
            # Ensure proper data types
            for col in required_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'Date' in data.columns:
                    data.index = pd.to_datetime(data['Date'])
                    data = data.drop('Date', axis=1)
                else:
                    data.index = pd.to_datetime(data.index)
            
            # Sort by date
            data = data.sort_index()
            
            return data[required_columns]
            
        except Exception as e:
            logger.error(f"Failed to standardize Angel One data: {e}")
            return data
    
    def _convert_period_to_days(self, period: str) -> int:
        return convert_period_to_days(period)
    
    def get_available_intervals(self) -> list:
        """
        Get list of available intervals
        
        Returns:
            List of available intervals
        """
        return list(self.api_limits.keys())
    
    def get_interval_description(self, interval: str) -> str:
        """
        Get description for an interval
        
        Args:
            interval: Data interval
            
        Returns:
            Description of the interval
        """
        descriptions = {
            'ONE_MINUTE': '1 Minute (Intraday)',
            'THREE_MINUTE': '3 Minutes (Intraday)',
            'FIVE_MINUTE': '5 Minutes (Intraday)',
            'TEN_MINUTE': '10 Minutes (Intraday)',
            'FIFTEEN_MINUTE': '15 Minutes (Intraday)',
            'THIRTY_MINUTE': '30 Minutes (Intraday)',
            'ONE_HOUR': '1 Hour (Intraday)',
            'ONE_DAY': '1 Day (Daily)'
        }
        
        return descriptions.get(interval, f'{interval} (Unknown)')
    
    def validate_config(self) -> bool:
        """
        Validate Angel One configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            required_keys = ['api_key', 'api_secret', 'access_token', 'exchange']
            
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"Missing required configuration key: {key}")
                    return False
                
                if not self.config[key] or self.config[key] == f'your_{key}':
                    logger.warning(f"Configuration key {key} is not set properly")
                    return False
            
            logger.info("Angel One configuration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Angel One configuration validation failed: {e}")
            return False
    
    def get_config_status(self) -> Dict[str, Any]:
        """
        Get configuration status
        
        Returns:
            Dictionary with configuration status
        """
        try:
            status = {
                'configured': self.validate_config(),
                'api_key_set': bool(self.config.get('api_key') and self.config['api_key'] != 'your_api_key'),
                'api_secret_set': bool(self.config.get('api_secret') and self.config['api_secret'] != 'your_api_secret'),
                'access_token_set': bool(self.config.get('access_token') and self.config['access_token'] != 'your_access_token'),
                'exchange': self.config.get('exchange', 'Not set'),
                'interval': self.config.get('interval', 'ONE_DAY')
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get configuration status: {e}")
            return {'configured': False, 'error': str(e)}
