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
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        self.current_prices = {}
        
        # Data validation settings
        self.min_data_points = 100
        self.max_missing_ratio = 0.1
        
    def load_stock_data(self, ticker: str, period: str = '2y', 
                       interval: str = '1d', force_refresh: bool = False) -> pd.DataFrame:
        """
        Load stock data with intelligent caching and fallback mechanisms.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_refresh: Force refresh data from source
            
        Returns:
            DataFrame with stock data
        """
        ticker = ticker.upper()
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
    
    def _download_stock_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Download stock data from yfinance with error handling."""
        try:
            # Try yfinance first
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                print(f"⚠️ No data returned for {ticker}")
                return None
            
            # Basic preprocessing
            data = self._preprocess_raw_data(data)
            return data
            
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
        
        df['Date'] = pd.to_datetime(df['Date'])
        
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
        
        if len(df) < self.min_data_points:
            return False
        
        # Check for required columns
        required_cols = ['Date', 'Close']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check for reasonable price values
        if 'Close' in df.columns:
            if df['Close'].min() <= 0 or df['Close'].max() > 100000:
                return False
        
        return True
    
    def get_current_price(self, ticker: str, force_refresh: bool = False) -> float:
        """Get current stock price with caching."""
        ticker = ticker.upper()
        
        # Check cache first
        if not force_refresh and ticker in self.current_prices:
            cache_time, price = self.current_prices[ticker]
            # Cache valid for 5 minutes
            if time.time() - cache_time < 300:
                return price
        
        try:
            # Get current price
            stock = yf.Ticker(ticker)
            current_price = stock.info.get('regularMarketPrice')
            
            if current_price is None:
                # Fallback: get latest close price
                data = self.load_stock_data(ticker, period='5d')
                if data is not None and not data.empty:
                    current_price = data['Close'].iloc[-1]
            
            if current_price is not None:
                self.current_prices[ticker] = (time.time(), current_price)
                return current_price
            else:
                raise ValueError(f"Could not get current price for {ticker}")
                
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
        
        # Resample data based on timeframe
        if timeframe == 'intraday':
            # Keep original frequency
            pass
        elif timeframe == 'daily':
            # Ensure daily frequency
            df = df.set_index('Date').resample('D').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 
                'Close': 'last', 'Volume': 'sum'
            }).dropna().reset_index()
        elif timeframe == 'weekly':
            df = df.set_index('Date').resample('W').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 
                'Close': 'last', 'Volume': 'sum'
            }).dropna().reset_index()
        elif timeframe == 'monthly':
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
