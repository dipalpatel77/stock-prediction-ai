#!/usr/bin/env python3
"""
Data Service
Shared data loading and preprocessing service for all analysis tools
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import pickle
import json
from pathlib import Path
import threading
import time

warnings.filterwarnings('ignore')

class DataService:
    """Shared data loading and preprocessing service."""
    
    def __init__(self, cache_dir: str = "data/cache", period_config: str = "recommended"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        self.current_prices = {}
        
        # Data validation settings
        self.min_data_points = 100
        self.max_missing_ratio = 0.1
        
        # Load period configuration
        try:
            from config.data_periods_config import get_period_config
            self.period_config = get_period_config(period_config)
        except ImportError:
            # Fallback to default periods
            self.period_config = {
                'default': '1y',
                'angel_one': '6mo',
                'yfinance': '1y',
                'quick_check': '3mo',
                'comprehensive': '2y'
            }
        
    def load_stock_data(self, ticker: str, period: str = None, 
                        interval: str = '1d', force_refresh: bool = False,
                        start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load stock data with intelligent caching and fallback mechanisms.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: Force refresh data from source
            start_date: Custom start date (YYYY-MM-DD format)
            end_date: Custom end date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with stock data
        """
        ticker = ticker.upper()
        
        # Use configured period if none provided
        if period is None:
            if self._is_indian_stock(ticker):
                period = self.period_config.get('angel_one', '6mo')
            else:
                period = self.period_config.get('yfinance', '1y')
        
        # Handle custom date range
        if start_date and end_date:
            return self.load_stock_data_custom_dates(ticker, start_date, end_date, interval, force_refresh)
        
        cache_key = f"{ticker}_{period}_{interval}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache first (unless force refresh)
        if not force_refresh and cache_file.exists():
            try:
                with self.cache_lock:
                    if cache_key in self.data_cache:
                        return self.data_cache[cache_key].copy()
                    
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Validate cached data
                    if self._validate_data(data):
                        self.data_cache[cache_key] = data
                        print(f"📊 Loaded cached data for {ticker} ({len(data)} records)")
                        return data.copy()
                    else:
                        print(f"⚠️ Cached data for {ticker} is invalid, refreshing...")
            except Exception as e:
                print(f"⚠️ Cache loading error for {ticker}: {e}")
        
        # Load from source
        print(f"📥 Downloading data for {ticker}...")
        data = self._download_stock_data(ticker, period, interval)
        
        if data is not None and self._validate_data(data):
            # Cache the data
            try:
                with self.cache_lock:
                    self.data_cache[cache_key] = data
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                print(f"✅ Data cached for {ticker}")
            except Exception as e:
                print(f"⚠️ Caching error for {ticker}: {e}")
            
            return data.copy()
        else:
            raise ValueError(f"Failed to load valid data for {ticker}")
    
    def load_stock_data_custom_dates(self, ticker: str, start_date: str, end_date: str, 
                                   interval: str = '1d', force_refresh: bool = False) -> pd.DataFrame:
        """
        Load stock data for custom date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            interval: Data interval
            force_refresh: Force refresh data from source
            
        Returns:
            DataFrame with stock data
        """
        ticker = ticker.upper()
        cache_key = f"{ticker}_custom_{start_date}_{end_date}_{interval}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache first (unless force refresh)
        if not force_refresh and cache_file.exists():
            try:
                with self.cache_lock:
                    if cache_key in self.data_cache:
                        return self.data_cache[cache_key].copy()
                    
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Validate cached data
                    if self._validate_data(data):
                        self.data_cache[cache_key] = data
                        print(f"📊 Loaded cached custom data for {ticker} ({len(data)} records)")
                        return data.copy()
                    else:
                        print(f"⚠️ Cached custom data for {ticker} is invalid, refreshing...")
            except Exception as e:
                print(f"⚠️ Cache loading error for {ticker}: {e}")
        
        # Load from source
        print(f"📥 Downloading custom date range data for {ticker} ({start_date} to {end_date})...")
        data = self._download_stock_data_custom_dates(ticker, start_date, end_date, interval)
        
        if data is not None and self._validate_data(data):
            # Cache the data
            try:
                with self.cache_lock:
                    self.data_cache[cache_key] = data
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                print(f"✅ Custom data cached for {ticker}")
            except Exception as e:
                print(f"⚠️ Caching error for {ticker}: {e}")
            
            return data.copy()
        else:
            raise ValueError(f"Failed to load valid data for {ticker}")
    
    def _download_stock_data_custom_dates(self, ticker: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Download stock data for custom date range with intelligent source selection."""
        try:
            # For Indian stocks: Try Angel One first, then yfinance as fallback
            if self._is_indian_stock(ticker):
                print(f"🇮🇳 Indian stock detected: {ticker}")
                print(f"🔄 Trying Angel One first for {ticker}...")
                
                # Try Angel One first
                angel_data = self._download_from_angel_one_custom_dates(ticker, start_date, end_date, interval)
                if angel_data is not None and not angel_data.empty:
                    print(f"✅ Successfully downloaded {len(angel_data)} records for {ticker} from Angel One")
                    return angel_data
                
                # Fallback to yfinance
                print(f"⚠️ Angel One failed for {ticker}, trying yfinance as fallback...")
                yahoo_data = self._download_from_yahoo_custom_dates(ticker, start_date, end_date, interval)
                if yahoo_data is not None and not yahoo_data.empty:
                    print(f"✅ Successfully downloaded {len(yahoo_data)} records for {ticker} from yfinance (fallback)")
                    return yahoo_data
                
                print(f"❌ Both Angel One and yfinance failed for {ticker}")
                return None
            
            # For non-Indian stocks: Use yfinance
            else:
                print(f"🌍 Non-Indian stock detected: {ticker}")
                print(f"🔄 Using yfinance for {ticker}...")
                
                yahoo_data = self._download_from_yahoo_custom_dates(ticker, start_date, end_date, interval)
                if yahoo_data is not None and not yahoo_data.empty:
                    print(f"✅ Successfully downloaded {len(yahoo_data)} records for {ticker} from yfinance")
                    return yahoo_data
                
                print(f"❌ yfinance failed for {ticker}")
                return None
            
        except Exception as e:
            print(f"❌ Error downloading data for {ticker}: {e}")
            return None
    
    def _download_stock_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Download stock data with intelligent source selection."""
        try:
            # For Indian stocks: Try Angel One first, then yfinance as fallback
            if self._is_indian_stock(ticker):
                print(f"🇮🇳 Indian stock detected: {ticker}")
                print(f"🔄 Trying Angel One first for {ticker}...")
                
                # Try Angel One first
                angel_data = self._download_from_angel_one(ticker, period, interval)
                if angel_data is not None and not angel_data.empty:
                    print(f"✅ Successfully downloaded {len(angel_data)} records for {ticker} from Angel One")
                    return angel_data
                
                # Fallback to yfinance
                print(f"⚠️ Angel One failed for {ticker}, trying yfinance as fallback...")
                yahoo_data = self._download_from_yahoo(ticker, period, interval)
                if yahoo_data is not None and not yahoo_data.empty:
                    print(f"✅ Successfully downloaded {len(yahoo_data)} records for {ticker} from yfinance (fallback)")
                    return yahoo_data
                
                # Both sources failed - provide detailed error information
                print(f"❌ Both Angel One and yfinance failed for {ticker}")
                print(f"   💡 This could be due to:")
                print(f"   - Angel One authentication issues (check API credentials)")
                print(f"   - Stock not available on international exchanges (yfinance)")
                print(f"   - Stock may be delisted or suspended")
                print(f"   - Network connectivity issues")
                
                # Try with .NS suffix for yfinance as additional fallback
                print(f"🔄 Trying with .NS suffix for {ticker}...")
                yahoo_data_ns = self._download_from_yahoo(f"{ticker}.NS", period, interval)
                if yahoo_data_ns is not None and not yahoo_data_ns.empty:
                    print(f"✅ Successfully downloaded {len(yahoo_data_ns)} records for {ticker}.NS from yfinance")
                    return yahoo_data_ns
                
                print(f"❌ All data sources failed for {ticker}")
                return None
            
            # For non-Indian stocks: Use yfinance first
            else:
                print(f"🌍 Non-Indian stock detected: {ticker}")
                print(f"🔄 Using yfinance for {ticker}...")
                
                yahoo_data = self._download_from_yahoo(ticker, period, interval)
                if yahoo_data is not None and not yahoo_data.empty:
                    print(f"✅ Successfully downloaded {len(yahoo_data)} records for {ticker} from yfinance")
                    return yahoo_data
                
                print(f"❌ yfinance failed for {ticker}")
                return None
            
        except Exception as e:
            print(f"❌ Error downloading data for {ticker}: {e}")
            return None
    
    def _preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing of raw stock data."""
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Ensure Date column exists and is datetime
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Add basic features
        df = self._add_basic_features(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in stock data."""
        # Forward fill for OHLCV data
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Drop rows with too many missing values
        missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
        df = df[missing_ratio < self.max_missing_ratio]
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic features to the dataset."""
        if 'Close' in df.columns:
            # Price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_Abs'] = df['Price_Change'].abs()
            
            # Moving averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Volatility
            df['Volatility'] = df['Price_Change'].rolling(window=20).std()
            
            # Volume features
            if 'Volume' in df.columns:
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality."""
        if df is None or df.empty:
            return False
        
        # Adjust minimum data points based on data length
        # For short periods (1mo, 3mo), allow fewer data points
        min_required = min(self.min_data_points, max(10, len(df) // 2))
        
        if len(df) < min_required:
            print(f"⚠️ Insufficient data points: {len(df)} < {min_required}")
            return False
        
        # Check for required columns
        required_cols = ['Date', 'Close']
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️ Missing required columns: {required_cols}")
            return False
        
        # Check for reasonable price values
        if 'Close' in df.columns:
            if df['Close'].min() <= 0 or df['Close'].max() > 100000:
                print(f"⚠️ Unreasonable price values: min={df['Close'].min()}, max={df['Close'].max()}")
                return False
        
        return True
    
    def get_current_price(self, ticker: str, force_refresh: bool = False) -> float:
        """Get current stock price with intelligent source selection."""
        ticker = ticker.upper()
        
        # Check cache first
        if not force_refresh and ticker in self.current_prices:
            cache_time, price = self.current_prices[ticker]
            # Cache valid for 5 minutes
            if time.time() - cache_time < 300:
                return price
        
        try:
            # For Indian stocks: Try Angel One first, then yfinance as fallback
            if self._is_indian_stock(ticker):
                print(f"🇮🇳 Getting current price for Indian stock: {ticker}")
                
                # Try Angel One first
                angel_price = self._get_current_price_from_angel_one(ticker)
                if angel_price is not None:
                    self.current_prices[ticker] = (time.time(), angel_price)
                    return angel_price
                
                # Fallback to yfinance
                print(f"⚠️ Angel One failed for {ticker}, trying yfinance...")
                yahoo_price = self._get_current_price_from_yahoo(ticker)
                if yahoo_price is not None:
                    self.current_prices[ticker] = (time.time(), yahoo_price)
                    return yahoo_price
                
                print(f"❌ Both Angel One and yfinance failed for {ticker}")
                return None
            
            # For non-Indian stocks: Use yfinance
            else:
                print(f"🌍 Getting current price for non-Indian stock: {ticker}")
                yahoo_price = self._get_current_price_from_yahoo(ticker)
                if yahoo_price is not None:
                    self.current_prices[ticker] = (time.time(), yahoo_price)
                    return yahoo_price
                
                print(f"❌ yfinance failed for {ticker}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting current price for {ticker}: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame, timeframe: str = 'daily', 
                       target_col: str = 'Close') -> pd.DataFrame:
        """
        Preprocess data for different timeframes and analysis types.
        
        Args:
            df: Input DataFrame
            timeframe: 'intraday', 'daily', 'weekly', 'monthly'
            target_col: Target column for prediction
            
        Returns:
            Preprocessed DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Copy to avoid modifying original
        df = df.copy()
        
        # Ensure Date column is properly formatted as datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        
        # Resample data based on timeframe
        if timeframe == 'intraday':
            # Keep original frequency
            pass
        elif timeframe == 'daily':
            # Ensure daily frequency
            if 'Date' in df.columns:
                df = df.set_index('Date').resample('D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 
                    'Close': 'last', 'Volume': 'sum'
                }).dropna().reset_index()
        elif timeframe == 'weekly':
            if 'Date' in df.columns:
                df = df.set_index('Date').resample('W').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 
                    'Close': 'last', 'Volume': 'sum'
                }).dropna().reset_index()
        elif timeframe == 'monthly':
            if 'Date' in df.columns:
                df = df.set_index('Date').resample('M').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 
                    'Close': 'last', 'Volume': 'sum'
                }).dropna().reset_index()
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Prepare target variable
        if target_col in df.columns:
            df['Target'] = df[target_col].shift(-1)  # Next day's price
            df['Target_Change'] = df['Target'].pct_change()
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        if 'Close' not in df.columns:
            return df
        
        close = df['Close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic
        if 'High' in df.columns and 'Low' in df.columns:
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['Stoch_K'] = 100 * ((close - low_min) / (high_max - low_min))
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'Date' not in df.columns:
            return df
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['Day_of_Year'] = df['Date'].dt.dayofyear
        
        return df
    
    def get_data_summary(self, ticker: str) -> Dict:
        """Get summary statistics for a stock."""
        try:
            data = self.load_stock_data(ticker, period='1y')
            if data is None or data.empty:
                return {}
            
            summary = {
                'ticker': ticker,
                'data_points': len(data),
                'date_range': {
                    'start': data['Date'].min().strftime('%Y-%m-%d'),
                    'end': data['Date'].max().strftime('%Y-%m-%d')
                },
                'price_stats': {
                    'current': data['Close'].iloc[-1],
                    'min': data['Close'].min(),
                    'max': data['Close'].max(),
                    'mean': data['Close'].mean(),
                    'std': data['Close'].std()
                },
                'volume_stats': {
                    'current': data['Volume'].iloc[-1] if 'Volume' in data.columns else None,
                    'mean': data['Volume'].mean() if 'Volume' in data.columns else None
                }
            }
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting data summary for {ticker}: {e}")
            return {}
    
    def clear_cache(self, ticker: str = None):
        """Clear data cache."""
        with self.cache_lock:
            if ticker:
                # Clear specific ticker cache
                keys_to_remove = [k for k in self.data_cache.keys() if k.startswith(ticker)]
                for key in keys_to_remove:
                    del self.data_cache[key]
                
                # Remove cache files
                cache_files = list(self.cache_dir.glob(f"{ticker}_*.pkl"))
                for file in cache_files:
                    file.unlink()
                
                print(f"🗑️ Cleared cache for {ticker}")
            else:
                # Clear all cache
                self.data_cache.clear()
                cache_files = list(self.cache_dir.glob("*.pkl"))
                for file in cache_files:
                    file.unlink()
                print("🗑️ Cleared all cache")
    
    def get_available_tickers(self) -> List[str]:
        """Get list of tickers with cached data."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        tickers = set()
        
        for file in cache_files:
            ticker = file.stem.split('_')[0]
            tickers.add(ticker)
        
        return sorted(list(tickers))
    
    def _is_indian_stock(self, ticker: str) -> bool:
        """Check if ticker is an Indian stock using the Indian Stock Mapper."""
        try:
            # First check if it has Indian suffixes (fast check)
            indian_suffixes = ['.NS', '.BO', '.NSE', '.BSE']
            ticker_upper = ticker.upper()
            
            for suffix in indian_suffixes:
                if ticker_upper.endswith(suffix):
                    print(f"✅ '{ticker}' identified as Indian stock (suffix: {suffix})")
                    return True
            
            # Remove suffix for lookup in Angel One database
            base_ticker = ticker_upper
            for suffix in indian_suffixes:
                if base_ticker.endswith(suffix):
                    base_ticker = base_ticker[:-len(suffix)]
                    break
            
            # Try to use the Indian Stock Mapper
            from data_downloaders.indian_stock_mapper import get_symbol_info, load_angel_master
            
            # Load Angel master data (uses smart caching)
            angel_master = load_angel_master()
            
            # Look up the symbol (try both with and without suffix)
            symbol_info = get_symbol_info(base_ticker, angel_master)
            if not symbol_info:
                symbol_info = get_symbol_info(ticker, angel_master)
            
            if symbol_info:
                print(f"✅ Found '{ticker}' in Angel One database: {symbol_info['exchange']}")
                return True
            else:
                print(f"❌ '{ticker}' not found in Angel One database")
                return False
                
        except Exception as e:
            print(f"⚠️ Error checking Indian stock mapper for '{ticker}': {e}")
            print("🔄 Falling back to hardcoded list...")
            
            # Fallback to hardcoded list
            return self._is_indian_stock_fallback(ticker)
    
    def _is_indian_stock_fallback(self, ticker: str) -> bool:
        """Fallback method using hardcoded Indian stock patterns."""
        # Common Indian stock patterns (fallback)
        indian_patterns = [
            # NSE symbols (no suffix)
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
            'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'AXISBANK', 'ASIANPAINT',
            'MARUTI', 'SUNPHARMA', 'TATAMOTORS', 'WIPRO', 'ULTRACEMCO', 'TITAN',
            'BAJFINANCE', 'NESTLEIND', 'HCLTECH', 'POWERGRID', 'NTPC', 'TECHM',
            'ADANIENT', 'ADANIPORTS', 'BAJAJFINSV', 'BRITANNIA', 'CIPLA', 'COALINDIA',
            'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM', 'HDFC', 'HDFCLIFE',
            'HEROMOTOCO', 'HINDALCO', 'JSWSTEEL', 'LT', 'M&M', 'ONGC', 'SHREECEM',
            'TATACONSUM', 'TATASTEEL', 'UPL', 'VEDL', 'WIPRO',
            # Additional Indian stocks including SWIGGY
            'SWIGGY', 'ZOMATO', 'PAYTM', 'NYKAA', 'DELHIVERY', 'POLICYBZR',
            'CARTRADE', 'FINO', 'MAPMYINDIA', 'TATAPOWER', 'ADANIGREEN', 'ADANITRANS',
            'ADANIGAS', 'ADANIPOWER', 'ADANIENT', 'ADANIPORTS', 'ADANIRETAIL',
            'ADANITOTAL', 'ADANIWILMAR', 'AMBUJACEM', 'APOLLOHOSP', 'APOLLOTYRE',
            'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE', 'BAJAJHLDNG', 'BERGEPAINT',
            'BHARATFORG', 'BIOCON', 'BPCL', 'BRITANNIA', 'CADILAHC', 'CIPLA',
            'COLPAL', 'DLF', 'DABUR', 'EICHERMOT', 'GAIL', 'GODREJCP', 'HCLTECH',
            'HDFCAMC', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDPETRO', 'ICICIGI',
            'ICICIPRULI', 'INDUSINDBK', 'IOC', 'JINDALSTEL', 'JSWSTEEL', 'KOTAKBANK',
            'LT', 'M&M', 'MARICO', 'NESTLEIND', 'NMDC', 'NTPC', 'ONGC', 'PEL',
            'PIDILITIND', 'POWERGRID', 'PFC', 'RECLTD', 'SAIL', 'SBILIFE', 'SHREECEM',
            'SIEMENS', 'SUNTV', 'TATACONSUM', 'TATAMOTORS', 'TATAPOWER', 'TATASTEEL',
            'TECHM', 'TITAN', 'TORNTPHARM', 'ULTRACEMCO', 'UPL', 'VEDL', 'VOLTAS',
            'WIPRO', 'ZEEL'
        ]
        
        # Also check for common Indian stock suffixes
        indian_suffixes = ['.NS', '.BO', '.NSE', '.BSE']
        
        ticker_upper = ticker.upper()
        
        # Check if ticker is in the patterns list
        if ticker_upper in indian_patterns:
            return True
        
        # Check if ticker has Indian suffixes
        for suffix in indian_suffixes:
            if ticker_upper.endswith(suffix):
                return True
        
        # Additional check for common Indian company names
        indian_keywords = ['RELIANCE', 'TCS', 'HDFC', 'ICICI', 'INFY', 'WIPRO', 
                          'TATA', 'ADANI', 'BAJAJ', 'MARUTI', 'HINDALCO', 'ONGC',
                          'SWIGGY', 'ZOMATO', 'PAYTM', 'NYKAA', 'DELHIVERY']
        
        for keyword in indian_keywords:
            if keyword in ticker_upper:
                return True
        
        return False
    
    def _download_from_yahoo(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Download data from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"⚠️ No data returned for {ticker} from yfinance")
                return None
            
            # Basic preprocessing
            data = self._preprocess_raw_data(data)
            return data
            
        except Exception as e:
            print(f"❌ Error downloading data for {ticker} from yfinance: {e}")
            return None
    
    def _download_from_angel_one(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Download data from Angel One API using Indian Stock Mapper for symbol lookup."""
        try:
            from .angel_one_data_downloader import AngelOneDataDownloader
            from data_downloaders.indian_stock_mapper import get_symbol_info, load_angel_master
            
            # Initialize Angel One downloader
            angel_downloader = AngelOneDataDownloader()
            
            # Try to authenticate (this will use TOTP if configured)
            if not angel_downloader.authenticate():
                print(f"⚠️ Angel One authentication failed for {ticker}")
                return None
            
            # Use Indian Stock Mapper to get symbol info
            # Remove suffix for Angel One lookup
            base_ticker = ticker
            indian_suffixes = ['.NS', '.BO', '.NSE', '.BSE']
            for suffix in indian_suffixes:
                if base_ticker.upper().endswith(suffix):
                    base_ticker = base_ticker[:-len(suffix)]
                    break
            
            print(f"🔍 Looking up '{base_ticker}' in Angel One database...")
            angel_master = load_angel_master()
            symbol_info = get_symbol_info(base_ticker, angel_master)
            
            if not symbol_info:
                print(f"⚠️ Symbol '{base_ticker}' not available in Angel One database")
                print(f"   This is normal for unlisted/new companies")
                print(f"   Falling back to yfinance...")
                return None
            
            print(f"✅ Found '{base_ticker}' in Angel One: Token={symbol_info['token']}, Exchange={symbol_info['exchange']}")
            
            # Download data using the symbol info from mapper
            df = angel_downloader.get_historical_data(
                symbol_name=base_ticker,
                exchange=symbol_info['exchange'],
                interval=interval,
                from_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                to_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if df is not None and not df.empty:
                # Preprocess the data to match yfinance format
                df = self._preprocess_raw_data(df)
                print(f"✅ Successfully downloaded {len(df)} records for {ticker} from Angel One")
                return df
            else:
                print(f"⚠️ No data available for {ticker} from Angel One")
                return None
                
        except ImportError:
            print("⚠️ Angel One integration not available (missing dependencies)")
            return None
        except Exception as e:
            print(f"❌ Angel One download error for {ticker}: {e}")
            return None
    
    def _get_current_price_from_yahoo(self, ticker: str) -> Optional[float]:
        """Get current price from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('regularMarketPrice')
            
            if current_price is None:
                # Fallback: get latest close price
                data = self.load_stock_data(ticker, period='5d')
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
            
            return current_price
            
        except Exception as e:
            print(f"❌ Error getting current price from yfinance for {ticker}: {e}")
            return None
    
    def _get_current_price_from_angel_one(self, ticker: str) -> Optional[float]:
        """Get current price from Angel One API."""
        try:
            from .angel_one_data_downloader import AngelOneDataDownloader
            
            # Initialize Angel One downloader
            angel_downloader = AngelOneDataDownloader()
            
            # Try to authenticate
            if not angel_downloader.authenticate():
                print(f"⚠️ Angel One authentication failed for {ticker}")
                return None
            
            # Get symbol info - remove suffix for lookup
            base_ticker = ticker
            indian_suffixes = ['.NS', '.BO', '.NSE', '.BSE']
            for suffix in indian_suffixes:
                if base_ticker.upper().endswith(suffix):
                    base_ticker = base_ticker[:-len(suffix)]
                    break
            
            symbol_info = angel_downloader.config.get_symbol_info(base_ticker)
            if not symbol_info:
                print(f"⚠️ Symbol info not found for {base_ticker}")
                return None
            
            # Get LTP data
            ltp_data = angel_downloader.get_ltp_data(base_ticker, symbol_info['exchange'])
            if ltp_data and 'ltp' in ltp_data:
                return float(ltp_data['ltp'])
            
            return None
            
        except ImportError:
            print("⚠️ Angel One integration not available (missing dependencies)")
            return None
        except Exception as e:
            print(f"❌ Error getting current price from Angel One for {ticker}: {e}")
            return None

    def _download_from_yahoo_custom_dates(self, ticker: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Download data from yfinance for custom date range."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                print(f"⚠️ No data returned for {ticker} from yfinance for period {start_date} to {end_date}")
                return None
            
            # Basic preprocessing
            data = self._preprocess_raw_data(data)
            return data
            
        except Exception as e:
            print(f"❌ Error downloading data for {ticker} from yfinance: {e}")
            return None
    
    def _download_from_angel_one_custom_dates(self, ticker: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Download data from Angel One API for custom date range using Indian Stock Mapper."""
        try:
            from .angel_one_data_downloader import AngelOneDataDownloader
            from data_downloaders.indian_stock_mapper import get_symbol_info, load_angel_master
            
            # Initialize Angel One downloader
            angel_downloader = AngelOneDataDownloader()
            
            # Try to authenticate (this will use TOTP if configured)
            if not angel_downloader.authenticate():
                print(f"⚠️ Angel One authentication failed for {ticker}")
                return None
            
            # Use Indian Stock Mapper to get symbol info
            # Remove suffix for Angel One lookup
            base_ticker = ticker
            indian_suffixes = ['.NS', '.BO', '.NSE', '.BSE']
            for suffix in indian_suffixes:
                if base_ticker.upper().endswith(suffix):
                    base_ticker = base_ticker[:-len(suffix)]
                    break
            
            print(f"🔍 Looking up '{base_ticker}' in Angel One database...")
            angel_master = load_angel_master()
            symbol_info = get_symbol_info(base_ticker, angel_master)
            
            if not symbol_info:
                print(f"⚠️ Symbol '{base_ticker}' not available in Angel One database")
                print(f"   This is normal for unlisted/new companies")
                print(f"   Falling back to yfinance...")
                return None
            
            print(f"✅ Found '{base_ticker}' in Angel One: Token={symbol_info['token']}, Exchange={symbol_info['exchange']}")
            
            # Download data using the symbol info from mapper with custom dates
            df = angel_downloader.get_historical_data(
                symbol_name=base_ticker,
                exchange=symbol_info['exchange'],
                interval=interval,
                from_date=start_date,
                to_date=end_date
            )
            
            if df is not None and not df.empty:
                # Preprocess the data to match yfinance format
                df = self._preprocess_raw_data(df)
                print(f"✅ Successfully downloaded {len(df)} records for {ticker} from Angel One")
                return df
            else:
                print(f"⚠️ No data available for {ticker} from Angel One")
                return None
                
        except ImportError:
            print("⚠️ Angel One integration not available (missing dependencies)")
            return None
        except Exception as e:
            print(f"❌ Angel One download error for {ticker}: {e}")
            return None
