#!/usr/bin/env python3
"""
Data Service Wrapper
Wrapper for data service with Angel One and database integration
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Import core services
# Use existing services from main.services
from .data_service import DataService
from .database_manager import DatabaseManager
from .angel_one_manager import AngelOneManager
from ..utils.date_formatter import convert_period_to_days
from ..utils.stock_utils import is_indian_stock
# Smart data fetcher removed (duplicate functionality)

logger = logging.getLogger(__name__)

class DataServiceWrapper:
    """Wrapper for data service with Angel One and database integration"""
    
    def __init__(self, ticker: str = "AAPL", config: Dict[str, Any] = None):
        if config is None:
            config = {
                'database_url': 'sqlite:///default.db',
                'use_angel_one': False,
                'cache_enabled': True
            }
        self.ticker = ticker
        self.config = config
        self.data_service = DataService(use_database=True)
        self.angel_service = None
        self.database_service = DatabaseManager()
        # Smart data fetcher removed (duplicate functionality)
        self.smart_fetcher = None
        
        # Initialize Angel One service - try dynamic lookup first
        try:
            self.angel_service = AngelOneManager(config)
            # Check if this is an Indian stock using dynamic lookup
            if hasattr(self.angel_service, 'angel_service') and hasattr(self.angel_service.angel_service, 'dynamic_lookup'):
                token, exchange = self.angel_service.angel_service.get_dynamic_token_and_exchange(ticker)
                if not token or not exchange:
                    # Not an Indian stock, set manager to None
                    self.angel_service = None
                    logger.info(f"{ticker} not found in Indian stocks, using Yahoo Finance")
                else:
                    logger.info(f"{ticker} found as Indian stock: Token={token}, Exchange={exchange}")
            else:
                # Fallback to hardcoded check
                if self._is_indian_stock(ticker):
                    logger.info(f"{ticker} found as Indian stock (hardcoded)")
                else:
                    self.angel_service = None
                    logger.info(f"{ticker} not found in hardcoded Indian stocks")
        except Exception as e:
            logger.warning(f"Failed to initialize Angel One manager: {e}")
            self.angel_service = None
    
    def load_stock_data(self, period: str, interval: str = 'ONE_DAY', force_refresh: bool = False) -> pd.DataFrame:
        """
        Load stock data from appropriate source with intelligent fetching
        
        Args:
            period: Data period (e.g., '1y', '6mo', '3mo')
            interval: Data interval (e.g., 'ONE_DAY', 'ONE_HOUR')
            force_refresh: Force refresh regardless of timing
            
        Returns:
            DataFrame with stock data
        """
        try:
            # TESTING MODE: Remove fetch constraints for testing
            should_fetch = True
            reason = "Testing mode - no constraints"
            
            # Check if we should fetch data based on interval update frequency
            # should_fetch, reason = self.smart_fetcher.should_fetch_data(
            #     self.ticker, interval, force_refresh
            # )
            
            if not should_fetch:
                logger.info(f"⏰ Skipping fetch for {self.ticker} ({interval}): {reason}")
                # Try to load from cache/database instead
                return self._load_cached_data(period, interval)
            
            logger.info(f"📊 Fetching data for {self.ticker} - period: {period}, interval: {interval}")
            logger.info(f"   Reason: {reason}")
            
            data = None
            success = False

            is_indian = self._is_indian_stock(self.ticker)

            if is_indian:
                # For Indian stocks, use Angel One API
                if not self.angel_service:
                    raise Exception(f"Angel One service not available for Indian stock {self.ticker}. Please configure Angel One API.")

                data = self._load_angel_one_data(period, interval)
                if data is not None and not data.empty:
                    logger.info(f"Successfully loaded {len(data)} records from Angel One for Indian stock {self.ticker}")
                    success = True
                else:
                    raise Exception(f"Failed to load data from Angel One for Indian stock {self.ticker}.")
            else:
                # For non-Indian stocks, use Yahoo Finance
                data = self._load_yahoo_finance_data(period)
                if data is not None and not data.empty:
                    logger.info(f"Successfully loaded {len(data)} records from Yahoo Finance for {self.ticker}")
                    success = True
                else:
                    raise Exception(f"Failed to load Yahoo Finance data for {self.ticker}.")

            if self.smart_fetcher:
                self.smart_fetcher.record_fetch(self.ticker, interval, success)

            return data

        except Exception as e:
            logger.error(f"Failed to load data for {self.ticker}: {e}")
            if self.smart_fetcher:
                self.smart_fetcher.record_fetch(self.ticker, interval, False)
            # Final fallback to basic data service
            return self._load_basic_data(period)
    
    def load_comprehensive_stock_data(self, period: str = '5y', force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load comprehensive stock data for all Angel One intervals
        
        Args:
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            force_refresh: Force refresh data even if cached
            
        Returns:
            Dictionary with interval as key and DataFrame as value
        """
        try:
            logger.info(f"Loading comprehensive stock data for {self.ticker} - Period: {period}")
            
            # Define all Angel One intervals with their max days
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
            
            # Only download comprehensive data for Indian stocks with Angel One
            if self._is_indian_stock(self.ticker) and self.is_angel_one_configured():
                print(f"\n📊 Downloading comprehensive data for {self.ticker}")
                print("=" * 50)
                
                for interval, max_days in intervals_config.items():
                    try:
                        print(f"⏰ Downloading {interval} data (max {max_days} days)...")
                        
                        # Convert period to days and limit to max days for this interval
                        requested_days = convert_period_to_days(period)
                        actual_days = min(requested_days, max_days)
                        
                        # Load data for this interval
                        interval_data = self._load_angel_one_data(period, interval)
                        
                        if interval_data is not None and not interval_data.empty:
                            comprehensive_data[interval] = interval_data
                            print(f"✅ {interval}: {len(interval_data)} records")
                            
                            # Store in database
                            self.store_data_in_database(interval_data, interval)
                        else:
                            print(f"❌ {interval}: Failed to download")
                            
                    except Exception as e:
                        print(f"❌ {interval}: Error - {e}")
                        logger.error(f"Failed to load {interval} data for {self.ticker}: {e}")
                
                print(f"\n📈 Comprehensive data download completed for {self.ticker}")
                print(f"   Successfully downloaded: {len(comprehensive_data)} intervals")
                
            else:
                # For non-Indian stocks, just load ONE_DAY data
                print(f"📊 Downloading standard data for {self.ticker} (non-Indian stock)")
                standard_data = self.load_stock_data(period, 'ONE_DAY', force_refresh)
                if standard_data is not None and not standard_data.empty:
                    comprehensive_data['ONE_DAY'] = standard_data
                    self.store_data_in_database(standard_data, 'ONE_DAY')
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Failed to load comprehensive stock data for {self.ticker}: {e}")
            return {}
    
    def _load_cached_data(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache/database when fetch is not needed
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with cached data or None
        """
        try:
            logger.info(f"🔄 Loading cached data for {self.ticker} ({interval})")
            
            # For Indian stocks, ensure we only use Angel One cached data
            if self._is_indian_stock(self.ticker):
                logger.info(f"🇮🇳 Loading cached Angel One data for Indian stock {self.ticker}")
                
                # Try to load from Angel One database cache
                if hasattr(self.database_service, 'get_stock_data'):
                    cached_data = self.database_service.get_stock_data(
                        ticker=self.ticker,
                        period=period,
                        interval=interval,
                        source='angel_one'  # Specify Angel One source
                    )
                    if cached_data is not None and not cached_data.empty:
                        logger.info(f"✅ Loaded {len(cached_data)} records from Angel One database cache")
                        return cached_data
                
                # Try Angel One file cache
                if self.angel_service and hasattr(self.angel_service, 'get_cached_data'):
                    cached_data = self.angel_service.get_cached_data(
                        ticker=self.ticker,
                        period=period,
                        interval=interval
                    )
                    if cached_data is not None and not cached_data.empty:
                        logger.info(f"✅ Loaded {len(cached_data)} records from Angel One file cache")
                        return cached_data
                
                logger.warning(f"No Angel One cached data available for Indian stock {self.ticker}")
                return None
            else:
                # For US/International stocks, use regular cache
                # Try to load from database first
                if hasattr(self.database_service, 'get_stock_data'):
                    cached_data = self.database_service.get_stock_data(
                        ticker=self.ticker,
                        period=period,
                        interval=interval
                    )
                    if cached_data is not None and not cached_data.empty:
                        logger.info(f"✅ Loaded {len(cached_data)} records from database cache")
                        return cached_data
                
                # Try to load from file cache
                if hasattr(self.data_service, 'load_stock_data'):
                    cached_data = self.data_service.load_stock_data(
                        ticker=self.ticker,
                        period=period,
                        interval=interval,
                        force_refresh=False
                    )
                    if cached_data is not None and not cached_data.empty:
                        logger.info(f"✅ Loaded {len(cached_data)} records from file cache")
                        return cached_data
                
                logger.warning("No cached data available, will need to fetch")
                return None
            
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
            return None
    
    def _load_angel_one_data(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load data from Angel One API
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with Angel One data or None if failed
        """
        try:
            if not self.angel_service:
                return None
            
            # Convert period to days for Angel One API
            days = convert_period_to_days(period)
            
            # Get data from Angel One service
            data = self.angel_service.get_historical_data(
                symbol=self.ticker,
                interval=interval,
                days=days
            )
            
            if data is not None and not data.empty:
                # Ensure proper column names and data types
                data = self._standardize_angel_one_data(data)
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Angel One data loading failed: {e}")
            return None
    
    def _load_yahoo_finance_data(self, period: str) -> pd.DataFrame:
        """
        Load data from Yahoo Finance
        
        Args:
            period: Data period
            
        Returns:
            DataFrame with Yahoo Finance data
        """
        try:
            # Use the existing data service for Yahoo Finance
            data = self.data_service.load_stock_data(self.ticker, period=period)
            
            if data is not None and not data.empty:
                # Ensure proper column names and data types
                data = self._standardize_yahoo_data(data)
                return data
            
            # If data service fails, try direct yfinance
            return self._load_direct_yfinance_data(period)
            
        except Exception as e:
            logger.error(f"Yahoo Finance data loading failed: {e}")
            return self._load_direct_yfinance_data(period)
    
    def _load_direct_yfinance_data(self, period: str) -> pd.DataFrame:
        """
        Load data directly from yfinance as fallback
        
        Args:
            period: Data period
            
        Returns:
            DataFrame with yfinance data
        """
        try:
            import yfinance as yf
            
            ticker_obj = yf.Ticker(self.ticker)
            data = ticker_obj.history(period=period)
            
            if data is not None and not data.empty:
                data = self._standardize_yahoo_data(data)
                return data
            
            raise Exception("No data received from yfinance")
            
        except Exception as e:
            logger.error(f"Direct yfinance loading failed: {e}")
            raise Exception(f"All data loading methods failed for {self.ticker}")
    
    def _load_basic_data(self, period: str) -> pd.DataFrame:
        """
        Load data using basic data service as final fallback
        
        Args:
            period: Data period
            
        Returns:
            DataFrame with basic data
        """
        try:
            # Use basic data service without database
            basic_service = DataService(use_database=False)
            data = basic_service.load_stock_data(self.ticker, period=period)
            
            if data is not None and not data.empty:
                return data
            
            raise Exception("Basic data service also failed")
            
        except Exception as e:
            logger.error(f"Basic data loading failed: {e}")
            raise Exception(f"All data loading methods failed for {self.ticker}")
    
    def _standardize_angel_one_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
                'volume': 'Volume'
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
                data.index = pd.to_datetime(data.index)
            
            return data[required_columns]
            
        except Exception as e:
            logger.error(f"Failed to standardize Angel One data: {e}")
            return data
    
    def _standardize_yahoo_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Yahoo Finance data format
        
        Args:
            data: Raw Yahoo Finance data
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Yahoo Finance typically has the correct column names
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in data.columns:
                    logger.warning(f"Missing column {col} in Yahoo Finance data")
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
                data.index = pd.to_datetime(data.index)
            
            return data[required_columns]
            
        except Exception as e:
            logger.error(f"Failed to standardize Yahoo Finance data: {e}")
            return data
    
    def _convert_period_to_days(self, period: str) -> int:
        return convert_period_to_days(period)
    
    def _is_indian_stock(self, ticker: str) -> bool:
        return is_indian_stock(ticker)
    
    def is_angel_one_configured(self) -> bool:
        """
        Check if Angel One API is properly configured
        
        Returns:
            True if Angel One is configured, False otherwise
        """
        return self.angel_service is not None
    
    def get_data_source_info(self) -> Dict[str, Any]:
        """
        Get information about data sources for the current ticker
        
        Returns:
            Dictionary with data source information
        """
        is_indian = self._is_indian_stock(self.ticker)
        angel_configured = self.is_angel_one_configured()
        
        return {
            'ticker': self.ticker,
            'is_indian_stock': is_indian,
            'angel_one_configured': angel_configured,
            'primary_source': 'Angel One' if is_indian else 'Yahoo Finance',
            'fallback_available': not is_indian,  # No fallback for Indian stocks
            'status': 'Ready' if (is_indian and angel_configured) or not is_indian else 'Angel One Required'
        }
    
    def get_data_source(self) -> str:
        """
        Get the data source being used
        
        Returns:
            Data source name ('angel_one' or 'yahoo_finance')
        """
        logger.info(f"🔍 Determining data source for {self.ticker}")
        
        # Use dynamic lookup if Angel One service is available
        if self.angel_service and hasattr(self.angel_service, 'angel_service'):
            if hasattr(self.angel_service.angel_service, 'dynamic_lookup'):
                token, exchange = self.angel_service.angel_service.get_dynamic_token_and_exchange(self.ticker)
                if token and exchange:
                    logger.info(f"✅ Dynamic lookup confirms {self.ticker} is Indian stock: Token={token}, Exchange={exchange}")
                    return 'angel_one'
                else:
                    logger.warning(f"❌ Dynamic lookup failed for {self.ticker}")
            else:
                logger.warning(f"❌ Dynamic lookup not available for {self.ticker}")
        else:
            logger.warning(f"❌ Angel One service not available for {self.ticker}")
        
        # Fallback to hardcoded check
        is_indian = self._is_indian_stock(self.ticker)
        logger.info(f"🔍 Hardcoded check for {self.ticker}: is_indian={is_indian}, angel_service={self.angel_service is not None}")
        
        if is_indian and self.angel_service:
            logger.info(f"✅ Using Angel One for {self.ticker} (hardcoded)")
            return 'angel_one'
        else:
            logger.info(f"❌ Using Yahoo Finance for {self.ticker}")
            return 'yahoo_finance'
    
    def store_data_in_database(self, data: pd.DataFrame, interval: str = 'ONE_DAY'):
        """
        Store data in database
        
        Args:
            data: Stock data DataFrame
            interval: Data interval
        """
        try:
            source = self.get_data_source()
            logger.info(f"Storing {len(data)} records in database from {source}")
            
            # Use database service to store data
            self.database_service.store_stock_data(
                ticker=self.ticker,
                data=data,
                source=source,
                interval=interval
            )
            
            logger.info(f"Successfully stored data for {self.ticker}")
            
        except Exception as e:
            logger.error(f"Failed to store data in database: {e}")
            # Don't raise exception - data loading can continue without storage
    
    def get_cached_data(self, period: str, interval: str = 'ONE_DAY') -> Optional[pd.DataFrame]:
        """
        Get cached data from database
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            Cached DataFrame or None if not found
        """
        try:
            source = self.get_data_source()
            data = self.database_service.get_stock_data(
                ticker=self.ticker,
                period=period,
                source=source,
                interval=interval
            )
            
            if data is not None and not data.empty:
                logger.info(f"Retrieved {len(data)} cached records for {self.ticker}")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            return None
    
    def load_data_with_fallback(self, period: str = '1y', interval: str = 'ONE_DAY') -> pd.DataFrame:
        """
        Load data with fallback mechanisms
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            Stock data DataFrame
        """
        try:
            # Try primary data source
            if self.angel_service and self._is_indian_stock(self.ticker):
                try:
                    data = self.angel_service.get_stock_data(self.ticker, period, interval)
                    if not data.empty:
                        logger.info(f"Loaded data from Angel One for {self.ticker}")
                        return data
                except Exception as e:
                    logger.warning(f"Angel One data loading failed: {e}")
                    # For Indian stocks, don't fallback to Yahoo Finance
                    # since we have dynamic lookup that should work
                    logger.error(f"No fallback for Indian stock {self.ticker} - Angel One should work with dynamic lookup")
                    return pd.DataFrame()
            
            # For non-Indian stocks, try Yahoo Finance
            if not self._is_indian_stock(self.ticker):
                try:
                    data = self.data_service.get_stock_data(self.ticker, period, interval)
                    if not data.empty:
                        logger.info(f"Loaded data from Yahoo Finance for {self.ticker}")
                        return data
                except Exception as e:
                    logger.error(f"Yahoo Finance data loading failed: {e}")
            
            # Final fallback - return empty DataFrame
            logger.warning(f"All data sources failed for {self.ticker}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Data loading with fallback failed: {e}")
            return pd.DataFrame()
    
    def degrade_gracefully(self, error: Exception) -> Dict[str, Any]:
        """
        Handle graceful degradation when services fail
        
        Args:
            error: The error that occurred
            
        Returns:
            Dictionary with degraded service information
        """
        try:
            logger.warning(f"Graceful degradation triggered: {error}")
            
            return {
                'status': 'degraded',
                'error': str(error),
                'fallback_active': True,
                'available_services': ['yahoo_finance'],
                'degraded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'fallback_active': False
            }
    
    def get_fetch_status(self, interval: str = 'ONE_DAY') -> Dict[str, Any]:
        """
        Get fetch status for the current ticker and interval
        
        Args:
            interval: Data interval
            
        Returns:
            Dictionary with fetch status information
        """
        return self.smart_fetcher.get_fetch_status(self.ticker, interval)
    
    def get_fetch_recommendations(self) -> Dict[str, Any]:
        """
        Get fetch recommendations for all intervals for the current ticker
        
        Returns:
            Dictionary with recommendations for each interval
        """
        return self.smart_fetcher.get_fetch_recommendations(self.ticker)
    
    def get_optimal_fetch_time(self, interval: str = 'ONE_DAY') -> Optional[datetime]:
        """
        Get the optimal time for the next fetch
        
        Args:
            interval: Data interval
            
        Returns:
            Optimal fetch time or None if can fetch now
        """
        return self.smart_fetcher.get_optimal_fetch_time(self.ticker, interval)
    
    def reset_fetch_tracker(self, interval: str = None):
        """
        Reset fetch tracker for the current ticker
        
        Args:
            interval: Specific interval to reset (None for all)
        """
        self.smart_fetcher.reset_fetch_tracker(self.ticker, interval)
        logger.info(f"Reset fetch tracker for {self.ticker}" + (f" ({interval})" if interval else " (all intervals)"))
    
    def get_max_days_per_request(self, interval: str = 'ONE_DAY') -> int:
        """
        Get the maximum days that can be requested in one API call for an interval
        
        Args:
            interval: Data interval
            
        Returns:
            Maximum days per request
        """
        return self.smart_fetcher.get_max_days_per_request(interval)
    
    def get_optimal_request_period(self, interval: str = 'ONE_DAY', requested_days: int = None) -> int:
        """
        Get the optimal number of days to request for an interval
        
        Args:
            interval: Data interval
            requested_days: Number of days requested (None for max)
            
        Returns:
            Optimal number of days to request
        """
        return self.smart_fetcher.get_optimal_request_period(interval, requested_days)
    
    def get_interval_limits(self, interval: str = 'ONE_DAY') -> Dict[str, Any]:
        """
        Get all limits and rules for an interval
        
        Args:
            interval: Data interval
            
        Returns:
            Dictionary with all interval limits
        """
        return self.smart_fetcher.get_interval_limits(interval)
