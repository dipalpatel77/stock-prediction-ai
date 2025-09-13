#!/usr/bin/env python3
"""
Incremental Data Service
Handles smart incremental updates of stock data to improve efficiency
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import yfinance as yf
import warnings
from pathlib import Path
import logging
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')

# Import database service
from .database_service import DatabaseService
from config.database_config import get_database_config

class IncrementalDataService:
    """
    Service for handling incremental stock data updates.
    Maintains data integrity while minimizing API calls.
    """
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "cache", 
                 use_database: bool = True, db_config_preset: str = "local"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self.data_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        
        # Configuration
        self.max_gap_days = 7  # Maximum gap to allow before full refresh
        self.min_records = 10  # Minimum records required for incremental update
        self.cache_expiry_hours = 24  # Cache expiry time
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Database integration
        self.use_database = use_database
        if self.use_database:
            try:
                db_config = get_database_config(db_config_preset)
                self.db_service = DatabaseService(
                    db_type=db_config.db_type,
                    connection_string=db_config.connection_string
                )
                self.logger.info(f"Database service initialized: {db_config.db_type}")
            except Exception as e:
                self.logger.warning(f"Database initialization failed: {e}, falling back to file system")
                self.use_database = False
    
    def get_incremental_data(self, ticker: str, period: str = "2y", 
                           interval: str = "1d", force_refresh: bool = False) -> pd.DataFrame:
        """
        Get stock data with incremental updates.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force full data refresh
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Try database first if available
            if self.use_database:
                return self._get_incremental_data_database(ticker, period, interval, force_refresh)
            else:
                # Fallback to file-based system
                is_indian = self._is_indian_stock(ticker)
                
                if is_indian:
                    return self._get_incremental_data_indian(ticker, period, interval, force_refresh)
                else:
                    return self._get_incremental_data_yahoo(ticker, period, interval, force_refresh)
                
        except Exception as e:
            self.logger.error(f"Error getting incremental data for {ticker}: {e}")
            # Fallback to full download
            return self._fallback_full_download(ticker, period, interval)
    
    def _get_incremental_data_database(self, ticker: str, period: str, 
                                     interval: str, force_refresh: bool) -> pd.DataFrame:
        """Get incremental data using database storage."""
        try:
            # Check existing data in database
            if not force_refresh:
                existing_info = self.db_service.get_data_info(ticker)
                if existing_info.get('exists', False):
                    # Check if we need incremental update
                    update_needed, update_info = self._check_update_needed_database(existing_info, period)
                    
                    if not update_needed:
                        self.logger.info(f"Using existing database data for {ticker} ({existing_info['records']} records)")
                        return self.db_service.get_stock_data(ticker)
                    
                    # Perform incremental update
                    if update_info['can_incremental']:
                        return self._perform_incremental_update_database(ticker, existing_info, update_info)
                    else:
                        self.logger.info(f"Gap too large for {ticker}, performing full refresh")
            
            # Full download and store in database
            return self._perform_full_download_database(ticker, period, interval)
            
        except Exception as e:
            self.logger.error(f"Error in database incremental update for {ticker}: {e}")
            # Fallback to file-based system
            return self._get_incremental_data_yahoo(ticker, period, interval, force_refresh)
    
    def _check_update_needed_database(self, existing_info: Dict, period: str) -> Tuple[bool, Dict]:
        """Check if database data update is needed."""
        try:
            last_date = existing_info['last_date']
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, '%Y-%m-%d').date()
            
            end_date = datetime.now().date()
            days_since_last = (end_date - last_date).days
            
            update_info = {
                'last_date': last_date,
                'end_date': end_date,
                'days_since_last': days_since_last,
                'can_incremental': days_since_last <= self.max_gap_days,
                'needs_full_refresh': days_since_last > self.max_gap_days or existing_info['records'] < self.min_records
            }
            
            # Determine if update is needed
            update_needed = days_since_last > 0
            
            return update_needed, update_info
            
        except Exception as e:
            self.logger.error(f"Error checking database update needed: {e}")
            return True, {'can_incremental': False, 'needs_full_refresh': True}
    
    def _perform_incremental_update_database(self, ticker: str, existing_info: Dict, 
                                           update_info: Dict) -> pd.DataFrame:
        """Perform incremental update using database."""
        try:
            last_date = update_info['last_date']
            end_date = update_info['end_date']
            
            # Calculate start date for incremental download
            start_date = last_date + timedelta(days=1)
            
            # Skip if start_date is in the future
            if start_date >= end_date:
                self.logger.info(f"No new data needed for {ticker}")
                return self.db_service.get_stock_data(ticker)
            
            self.logger.info(f"Downloading incremental data for {ticker} from {start_date} to {end_date}")
            
            # Download new data
            is_indian = self._is_indian_stock(ticker)
            if is_indian:
                new_data = self._download_incremental_indian(ticker, start_date, end_date)
            else:
                new_data = self._download_incremental_yahoo(ticker, start_date, end_date)
            
            if new_data.empty:
                self.logger.info(f"No new data available for {ticker}")
                return self.db_service.get_stock_data(ticker)
            
            # Store new data in database
            self.db_service.store_stock_data(ticker, new_data, "incremental_update")
            
            # Return combined data
            return self.db_service.get_stock_data(ticker)
            
        except Exception as e:
            self.logger.error(f"Error in database incremental update for {ticker}: {e}")
            raise
    
    def _perform_full_download_database(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Perform full download and store in database."""
        try:
            self.logger.info(f"Performing full download for {ticker} using database")
            
            # Download data
            is_indian = self._is_indian_stock(ticker)
            if is_indian:
                data = self._download_full_indian(ticker, period, interval)
            else:
                data = self._download_full_yahoo(ticker, period, interval)
            
            if data.empty:
                raise Exception(f"No data downloaded for {ticker}")
            
            # Store in database
            self.db_service.store_stock_data(ticker, data, "full_download")
            
            self.logger.info(f"Full download completed for {ticker}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in database full download for {ticker}: {e}")
            raise
    
    def _download_incremental_yahoo(self, ticker: str, start_date, end_date) -> pd.DataFrame:
        """Download incremental data using yfinance."""
        stock = yf.Ticker(ticker)
        return stock.history(start=start_date, end=end_date, interval="1d")
    
    def _download_incremental_indian(self, ticker: str, start_date, end_date) -> pd.DataFrame:
        """Download incremental data for Indian stocks."""
        try:
            from core.angel_one_data_downloader import AngelOneDataDownloader
            from core.angel_one_config import AngelOneConfig
            from core.indian_stock_mapper import get_symbol_info
            
            config = AngelOneConfig()
            downloader = AngelOneDataDownloader(config)
            symbol_info = get_symbol_info(ticker)
            
            if not symbol_info:
                raise Exception(f"Symbol info not found for {ticker}")
            
            return downloader.get_historical_data(
                symbol_name=symbol_info['symbol'],
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d'),
                interval='ONE_DAY'
            )
        except Exception as e:
            self.logger.warning(f"Angel One incremental download failed for {ticker}: {e}")
            # Fallback to yfinance
            return self._download_incremental_yahoo(ticker, start_date, end_date)
    
    def _download_full_yahoo(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Download full data using yfinance."""
        stock = yf.Ticker(ticker)
        return stock.history(period=period, interval=interval)
    
    def _download_full_indian(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Download full data for Indian stocks."""
        try:
            from core.angel_one_data_downloader import AngelOneDataDownloader
            from core.angel_one_config import AngelOneConfig
            from core.indian_stock_mapper import get_symbol_info
            
            config = AngelOneConfig()
            downloader = AngelOneDataDownloader(config)
            symbol_info = get_symbol_info(ticker)
            
            if not symbol_info:
                raise Exception(f"Symbol info not found for {ticker}")
            
            return downloader.download_stock_data(symbol_info['symbol'], period, interval)
        except Exception as e:
            self.logger.warning(f"Angel One full download failed for {ticker}: {e}")
            # Fallback to yfinance
            return self._download_full_yahoo(ticker, period, interval)
    
    def _get_incremental_data_yahoo(self, ticker: str, period: str, 
                                  interval: str, force_refresh: bool) -> pd.DataFrame:
        """Get incremental data using yfinance."""
        try:
            data_file = self.data_dir / f"{ticker}_raw_data.csv"
            
            # Check if we should use existing data
            if not force_refresh and data_file.exists():
                existing_data = self._load_existing_data(data_file)
                if existing_data is not None:
                    # Check if we need incremental update
                    update_needed, update_info = self._check_update_needed(existing_data, period)
                    
                    if not update_needed:
                        self.logger.info(f"Using existing data for {ticker} ({len(existing_data)} records)")
                        return existing_data
                    
                    # Perform incremental update
                    if update_info['can_incremental']:
                        return self._perform_incremental_update_yahoo(ticker, existing_data, update_info)
                    else:
                        self.logger.info(f"Gap too large for {ticker}, performing full refresh")
            
            # Full download
            return self._perform_full_download_yahoo(ticker, period, interval)
            
        except Exception as e:
            self.logger.error(f"Error in incremental yahoo update for {ticker}: {e}")
            return self._fallback_full_download(ticker, period, interval)
    
    def _get_incremental_data_indian(self, ticker: str, period: str, 
                                   interval: str, force_refresh: bool) -> pd.DataFrame:
        """Get incremental data for Indian stocks using Angel One + yfinance fallback."""
        try:
            data_file = self.data_dir / f"{ticker}_raw_data.csv"
            
            # Check if we should use existing data
            if not force_refresh and data_file.exists():
                existing_data = self._load_existing_data(data_file)
                if existing_data is not None:
                    # Check if we need incremental update
                    update_needed, update_info = self._check_update_needed(existing_data, period)
                    
                    if not update_needed:
                        self.logger.info(f"Using existing data for {ticker} ({len(existing_data)} records)")
                        return existing_data
                    
                    # Try Angel One first for incremental update
                    if update_info['can_incremental']:
                        try:
                            return self._perform_incremental_update_angel_one(ticker, existing_data, update_info)
                        except Exception as e:
                            self.logger.warning(f"Angel One incremental update failed for {ticker}: {e}")
                            # Fallback to yfinance
                            return self._perform_incremental_update_yahoo(ticker, existing_data, update_info)
                    else:
                        self.logger.info(f"Gap too large for {ticker}, performing full refresh")
            
            # Full download - try Angel One first, then yfinance
            try:
                return self._perform_full_download_angel_one(ticker, period, interval)
            except Exception as e:
                self.logger.warning(f"Angel One full download failed for {ticker}: {e}")
                return self._perform_full_download_yahoo(ticker, period, interval)
                
        except Exception as e:
            self.logger.error(f"Error in incremental Indian update for {ticker}: {e}")
            return self._fallback_full_download(ticker, period, interval)
    
    def _load_existing_data(self, data_file: Path) -> Optional[pd.DataFrame]:
        """Load existing data file."""
        try:
            df = pd.read_csv(data_file)
            if df.empty:
                return None
            
            # Ensure Date column is datetime (timezone-naive)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df = df.sort_values('Date')
            elif df.index.name == 'Date' or 'Date' in str(df.index.dtype):
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df.sort_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading existing data from {data_file}: {e}")
            return None
    
    def _check_update_needed(self, existing_data: pd.DataFrame, period: str) -> Tuple[bool, Dict]:
        """
        Check if data update is needed and determine update strategy.
        
        Returns:
            (update_needed, update_info)
        """
        try:
            # Get the last date in existing data
            if 'Date' in existing_data.columns:
                last_date = existing_data['Date'].max()
            else:
                last_date = existing_data.index.max()
            
            # Calculate expected end date based on period
            end_date = datetime.now()
            period_days = self._get_period_days(period)
            expected_start = end_date - timedelta(days=period_days)
            
            # Check if we need update
            days_since_last = (end_date - last_date).days
            
            update_info = {
                'last_date': last_date,
                'end_date': end_date,
                'days_since_last': days_since_last,
                'can_incremental': days_since_last <= self.max_gap_days,
                'needs_full_refresh': days_since_last > self.max_gap_days or len(existing_data) < self.min_records
            }
            
            # Determine if update is needed
            update_needed = (
                days_since_last > 0 or  # Data is not current
                last_date < expected_start  # Data doesn't cover full period
            )
            
            return update_needed, update_info
            
        except Exception as e:
            self.logger.error(f"Error checking update needed: {e}")
            return True, {'can_incremental': False, 'needs_full_refresh': True}
    
    def _perform_incremental_update_yahoo(self, ticker: str, existing_data: pd.DataFrame, 
                                        update_info: Dict) -> pd.DataFrame:
        """Perform incremental update using yfinance."""
        try:
            last_date = update_info['last_date']
            end_date = update_info['end_date']
            
            # Calculate start date for incremental download
            start_date = last_date + timedelta(days=1)
            
            # Skip if start_date is in the future
            if start_date >= end_date:
                self.logger.info(f"No new data needed for {ticker}")
                return existing_data
            
            self.logger.info(f"Downloading incremental data for {ticker} from {start_date.date()} to {end_date.date()}")
            
            # Download new data
            stock = yf.Ticker(ticker)
            new_data = stock.history(start=start_date, end=end_date, interval="1d")
            
            if new_data.empty:
                self.logger.info(f"No new data available for {ticker}")
                return existing_data
            
            # Merge with existing data
            merged_data = self._merge_data(existing_data, new_data)
            
            # Save updated data
            self._save_data(merged_data, ticker)
            
            self.logger.info(f"Incremental update completed for {ticker}: {len(new_data)} new records")
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error in incremental yahoo update for {ticker}: {e}")
            raise
    
    def _perform_incremental_update_angel_one(self, ticker: str, existing_data: pd.DataFrame, 
                                            update_info: Dict) -> pd.DataFrame:
        """Perform incremental update using Angel One API."""
        try:
            # Import Angel One components
            from core.angel_one_data_downloader import AngelOneDataDownloader
            from core.angel_one_config import AngelOneConfig
            
            last_date = update_info['last_date']
            end_date = update_info['end_date']
            
            # Calculate start date for incremental download
            start_date = last_date + timedelta(days=1)
            
            # Skip if start_date is in the future
            if start_date >= end_date:
                self.logger.info(f"No new data needed for {ticker}")
                return existing_data
            
            self.logger.info(f"Downloading incremental data from Angel One for {ticker} from {start_date.date()} to {end_date.date()}")
            
            # Download new data using Angel One
            config = AngelOneConfig()
            downloader = AngelOneDataDownloader(config)
            
            # Get symbol info
            from core.indian_stock_mapper import get_symbol_info
            symbol_info = get_symbol_info(ticker)
            
            if not symbol_info:
                raise Exception(f"Symbol info not found for {ticker}")
            
            new_data = downloader.get_historical_data(
                symbol_name=symbol_info['symbol'],
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d'),
                interval='ONE_DAY'
            )
            
            if new_data is None or new_data.empty:
                self.logger.info(f"No new data available from Angel One for {ticker}")
                return existing_data
            
            # Merge with existing data
            merged_data = self._merge_data(existing_data, new_data)
            
            # Save updated data
            self._save_data(merged_data, ticker)
            
            self.logger.info(f"Angel One incremental update completed for {ticker}: {len(new_data)} new records")
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error in incremental Angel One update for {ticker}: {e}")
            raise
    
    def _perform_full_download_yahoo(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Perform full data download using yfinance."""
        try:
            self.logger.info(f"Performing full download for {ticker} using yfinance")
            
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise Exception(f"No data downloaded for {ticker}")
            
            # Preprocess data
            data = self._preprocess_data(data)
            
            # Save data
            self._save_data(data, ticker)
            
            self.logger.info(f"Full download completed for {ticker}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in full yahoo download for {ticker}: {e}")
            raise
    
    def _perform_full_download_angel_one(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Perform full data download using Angel One API."""
        try:
            self.logger.info(f"Performing full download for {ticker} using Angel One")
            
            # Import Angel One components
            from core.angel_one_data_downloader import AngelOneDataDownloader
            from core.angel_one_config import AngelOneConfig
            from core.indian_stock_mapper import get_symbol_info
            
            # Get symbol info
            symbol_info = get_symbol_info(ticker)
            if not symbol_info:
                raise Exception(f"Symbol info not found for {ticker}")
            
            # Download data
            config = AngelOneConfig()
            downloader = AngelOneDataDownloader(config)
            
            data = downloader.download_stock_data(symbol_info['symbol'], period, interval)
            
            if data is None or data.empty:
                raise Exception(f"No data downloaded from Angel One for {ticker}")
            
            # Preprocess data
            data = self._preprocess_data(data)
            
            # Save data
            self._save_data(data, ticker)
            
            self.logger.info(f"Angel One full download completed for {ticker}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error in full Angel One download for {ticker}: {e}")
            raise
    
    def _merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """Merge existing and new data, handling duplicates and gaps."""
        try:
            # Ensure both DataFrames have Date column
            if 'Date' not in existing_data.columns:
                existing_data = existing_data.reset_index()
            if 'Date' not in new_data.columns:
                new_data = new_data.reset_index()
            
            # Convert Date columns to datetime (ensure timezone consistency)
            existing_data['Date'] = pd.to_datetime(existing_data['Date']).dt.tz_localize(None)
            new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None)
            
            # Remove duplicates from new data (in case of overlap)
            new_data = new_data[~new_data['Date'].isin(existing_data['Date'])]
            
            if new_data.empty:
                self.logger.info("No new data to merge (all dates already exist)")
                return existing_data
            
            # Combine data
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Sort by date
            combined_data = combined_data.sort_values('Date')
            
            # Remove any duplicate dates (keep last occurrence)
            combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
            
            # Reset index
            combined_data = combined_data.reset_index(drop=True)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error merging data: {e}")
            raise
    
    def _save_data(self, data: pd.DataFrame, ticker: str):
        """Save data to file with thread safety."""
        try:
            with self.data_lock:
                data_file = self.data_dir / f"{ticker}_raw_data.csv"
                data.to_csv(data_file, index=False)
                
                # Also save to cache
                cache_file = self.cache_dir / f"{ticker}_data.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                
                self.logger.info(f"Data saved for {ticker}: {len(data)} records")
                
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {e}")
            raise
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess downloaded data."""
        try:
            # Reset index to make Date a column
            if data.index.name == 'Date' or 'Date' in str(data.index.dtype):
                data = data.reset_index()
            
            # Ensure Date column is datetime (timezone-naive)
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            
            # Sort by date
            data = data.sort_values('Date')
            
            # Remove duplicates
            data = data.drop_duplicates(subset=['Date'], keep='last')
            
            # Reset index
            data = data.reset_index(drop=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            raise
    
    def _fallback_full_download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fallback to full download when incremental update fails."""
        try:
            self.logger.warning(f"Falling back to full download for {ticker}")
            
            # Try yfinance as fallback
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise Exception(f"Fallback download failed for {ticker}")
            
            # Preprocess and save
            data = self._preprocess_data(data)
            self._save_data(data, ticker)
            
            self.logger.info(f"Fallback download completed for {ticker}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Fallback download failed for {ticker}: {e}")
            raise
    
    def _is_indian_stock(self, ticker: str) -> bool:
        """Check if ticker is an Indian stock."""
        return ticker.endswith('.NS') or ticker.endswith('.BO')
    
    def _get_period_days(self, period: str) -> int:
        """Convert period string to days."""
        period_mapping = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        return period_mapping.get(period, 365)
    
    def get_data_info(self, ticker: str) -> Dict:
        """Get information about existing data for a ticker."""
        try:
            # Try database first if available
            if self.use_database:
                return self.db_service.get_data_info(ticker)
            else:
                # Fallback to file-based system
                data_file = self.data_dir / f"{ticker}_raw_data.csv"
                
                if not data_file.exists():
                    return {
                        'exists': False,
                        'records': 0,
                        'first_date': None,
                        'last_date': None,
                        'days_old': None
                    }
                
                data = self._load_existing_data(data_file)
                if data is None or data.empty:
                    return {
                        'exists': False,
                        'records': 0,
                        'first_date': None,
                        'last_date': None,
                        'days_old': None
                    }
                
                if 'Date' in data.columns:
                    first_date = data['Date'].min()
                    last_date = data['Date'].max()
                else:
                    first_date = data.index.min()
                    last_date = data.index.max()
                
                days_old = (datetime.now() - last_date).days if last_date else None
                
                return {
                    'exists': True,
                    'records': len(data),
                    'first_date': first_date,
                    'last_date': last_date,
                    'days_old': days_old,
                    'needs_update': days_old > 0 if days_old is not None else True
                }
            
        except Exception as e:
            self.logger.error(f"Error getting data info for {ticker}: {e}")
            return {
                'exists': False,
                'records': 0,
                'first_date': None,
                'last_date': None,
                'days_old': None,
                'error': str(e)
            }
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up data files older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_count = 0
            
            for data_file in self.data_dir.glob("*_raw_data.csv"):
                try:
                    # Check file modification time
                    file_time = datetime.fromtimestamp(data_file.stat().st_mtime)
                    
                    if file_time < cutoff_date:
                        data_file.unlink()
                        cleaned_count += 1
                        self.logger.info(f"Cleaned up old data file: {data_file.name}")
                        
                except Exception as e:
                    self.logger.error(f"Error cleaning up {data_file}: {e}")
            
            self.logger.info(f"Cleanup completed: {cleaned_count} files removed")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
            return 0

# Example usage and testing
if __name__ == "__main__":
    # Test the incremental data service
    service = IncrementalDataService()
    
    # Test with a stock
    ticker = "AAPL"
    print(f"Testing incremental data service with {ticker}")
    
    # Get data info
    info = service.get_data_info(ticker)
    print(f"Data info: {info}")
    
    # Get incremental data
    data = service.get_incremental_data(ticker, period="1y")
    print(f"Retrieved {len(data)} records for {ticker}")
    
    # Test again (should use existing data)
    data2 = service.get_incremental_data(ticker, period="1y")
    print(f"Second call retrieved {len(data2)} records for {ticker}")
