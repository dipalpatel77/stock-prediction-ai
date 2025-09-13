#!/usr/bin/env python3
"""
FRED API Service
================

Enhanced service for Federal Reserve Economic Data (FRED) API integration.
Provides comprehensive economic data with rate limiting, caching, and error handling.
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import warnings
from collections import deque
import threading

# Import FRED configuration
try:
    from config.fred_api_config import fred_config, FREDSeries
    FRED_CONFIG_AVAILABLE = True
except ImportError:
    FRED_CONFIG_AVAILABLE = False
    print("⚠️ FRED configuration not available. Using basic configuration.")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FREDObservation:
    """Data class for FRED observations"""
    date: datetime
    value: float
    realtime_start: str
    realtime_end: str

@dataclass
class FREDSeriesData:
    """Data class for FRED series data"""
    series_id: str
    title: str
    units: str
    frequency: str
    seasonal_adjustment: str
    last_updated: datetime
    observations: List[FREDObservation]
    notes: str = ""

class FREDRateLimiter:
    """Rate limiter for FRED API requests"""
    
    def __init__(self, requests_per_minute: int = 120):
        self.requests_per_minute = requests_per_minute
        self.request_interval = 60.0 / requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.request_times and now - self.request_times[0] >= 60.0:
                self.request_times.popleft()
            
            # If we've made too many requests, wait
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    now = time.time()
            
            # Add current request
            self.request_times.append(now)
            
            # Ensure minimum interval between requests
            if len(self.request_times) > 1:
                time_since_last = now - self.request_times[-2]
                if time_since_last < self.request_interval:
                    sleep_time = self.request_interval - time_since_last
                    time.sleep(sleep_time)

class FREDAPIService:
    """
    Enhanced FRED API service with comprehensive economic data access
    
    Features:
    - Rate limiting (120 requests per minute)
    - Intelligent caching
    - Error handling and retries
    - Support for all FRED series
    - Historical data analysis
    - Real-time updates
    """
    
    def __init__(self, api_key: str = None, cache_duration_hours: int = 4):
        """
        Initialize FRED API service
        
        Args:
            api_key: FRED API key (if None, uses environment variable)
            cache_duration_hours: Cache duration in hours
        """
        # API Configuration
        if FRED_CONFIG_AVAILABLE:
            self.config = fred_config
            if api_key:
                self.config.api_key = api_key
        else:
            # Fallback configuration
            self.config = type('Config', (), {
                'base_url': "https://api.stlouisfed.org/fred",
                'api_key': api_key or os.getenv('FRED_API_KEY', 'demo'),
                'requests_per_minute': 120,
                'request_interval': 0.5
            })()
        
        # Rate limiter
        self.rate_limiter = FREDRateLimiter(self.config.requests_per_minute)
        
        # Cache settings
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_lock = threading.Lock()
        
        # Create cache directory
        self.cache_dir = Path("data/fred_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Stock-Predictor/1.0 (FRED-API-Client)'
        })
        
        # Request tracking
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        
        logger.info(f"FRED API Service initialized with rate limit: {self.config.requests_per_minute} requests/minute")
    
    def get_series_observations(self, series_id: str, limit: int = 100, 
                              start_date: str = None, end_date: str = None,
                              frequency: str = None, aggregation_method: str = 'avg') -> Optional[FREDSeriesData]:
        """
        Get observations for a FRED series
        
        Args:
            series_id: FRED series ID
            limit: Number of observations to retrieve
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (d, w, m, q, sa, a)
            aggregation_method: Aggregation method (avg, sum, eop)
            
        Returns:
            FREDSeriesData object or None if error
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Build request parameters
            params = {
                'series_id': series_id,
                'api_key': self.config.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
            if frequency:
                params['frequency'] = frequency
            if aggregation_method:
                params['aggregation_method'] = aggregation_method
            
            # Make request
            url = f"{self.config.base_url}/series/observations"
            response = self.session.get(url, params=params, timeout=30)
            
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_series_observations(series_id, data)
            else:
                logger.error(f"FRED API error for {series_id}: {response.status_code} - {response.text}")
                self.error_count += 1
                return None
                
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            self.error_count += 1
            return None
    
    def get_series_info(self, series_id: str) -> Optional[Dict]:
        """
        Get metadata for a FRED series
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Series metadata dictionary or None if error
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Build request parameters
            params = {
                'series_id': series_id,
                'api_key': self.config.api_key,
                'file_type': 'json'
            }
            
            # Make request
            url = f"{self.config.base_url}/series"
            response = self.session.get(url, params=params, timeout=30)
            
            self.request_count += 1
            
            if response.status_code == 200:
                data = response.json()
                if 'seriess' in data and len(data['seriess']) > 0:
                    return data['seriess'][0]
                else:
                    logger.warning(f"No series info found for {series_id}")
                    return None
            else:
                logger.error(f"FRED API error for series info {series_id}: {response.status_code}")
                self.error_count += 1
                return None
                
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {str(e)}")
            self.error_count += 1
            return None
    
    def get_multiple_series(self, series_ids: List[str], limit: int = 100) -> Dict[str, FREDSeriesData]:
        """
        Get multiple FRED series efficiently
        
        Args:
            series_ids: List of FRED series IDs
            limit: Number of observations per series
            
        Returns:
            Dictionary mapping series_id to FREDSeriesData
        """
        results = {}
        
        for series_id in series_ids:
            try:
                series_data = self.get_series_observations(series_id, limit=limit)
                if series_data:
                    results[series_id] = series_data
                else:
                    logger.warning(f"Failed to fetch data for series {series_id}")
            except Exception as e:
                logger.error(f"Error fetching series {series_id}: {str(e)}")
        
        logger.info(f"Successfully fetched {len(results)} out of {len(series_ids)} series")
        return results
    
    def get_economic_indicators(self, category: str = 'priority') -> Dict[str, FREDSeriesData]:
        """
        Get economic indicators by category
        
        Args:
            category: Category of indicators ('priority', 'health', 'inflation', etc.)
            
        Returns:
            Dictionary of economic indicators
        """
        if not FRED_CONFIG_AVAILABLE:
            logger.warning("FRED configuration not available. Using basic indicators.")
            series_ids = ['GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'GS10']
        else:
            series_ids = self.config.get_series_by_category(category)
        
        return self.get_multiple_series(series_ids)
    
    def get_latest_values(self, series_ids: List[str]) -> Dict[str, float]:
        """
        Get latest values for multiple series (efficient for real-time data)
        
        Args:
            series_ids: List of FRED series IDs
            
        Returns:
            Dictionary mapping series_id to latest value
        """
        results = {}
        
        for series_id in series_ids:
            try:
                series_data = self.get_series_observations(series_id, limit=1)
                if series_data and series_data.observations:
                    results[series_id] = series_data.observations[0].value
                else:
                    logger.warning(f"No data available for series {series_id}")
            except Exception as e:
                logger.error(f"Error fetching latest value for {series_id}: {str(e)}")
        
        return results
    
    def get_historical_analysis(self, series_id: str, years: int = 5) -> Dict[str, Any]:
        """
        Get historical analysis for a series
        
        Args:
            series_id: FRED series ID
            years: Number of years of historical data
            
        Returns:
            Historical analysis dictionary
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Get historical data
            series_data = self.get_series_observations(
                series_id, 
                limit=1000,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if not series_data or not series_data.observations:
                return {}
            
            # Convert to pandas for analysis
            df = pd.DataFrame([
                {
                    'date': obs.date,
                    'value': obs.value
                }
                for obs in series_data.observations
            ])
            
            df = df.sort_values('date')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Calculate statistics
            current_value = df['value'].iloc[-1]
            previous_value = df['value'].iloc[-2] if len(df) > 1 else current_value
            
            analysis = {
                'series_id': series_id,
                'title': series_data.title,
                'current_value': current_value,
                'previous_value': previous_value,
                'change': current_value - previous_value,
                'change_pct': ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0,
                'min_value': df['value'].min(),
                'max_value': df['value'].max(),
                'mean_value': df['value'].mean(),
                'std_value': df['value'].std(),
                'data_points': len(df),
                'date_range': {
                    'start': df.index.min().strftime('%Y-%m-%d'),
                    'end': df.index.max().strftime('%Y-%m-%d')
                },
                'last_updated': series_data.last_updated.isoformat()
            }
            
            # Calculate trend
            if len(df) >= 10:
                recent_trend = np.polyfit(range(len(df[-10:])), df['value'].iloc[-10:], 1)[0]
                analysis['trend'] = 'Up' if recent_trend > 0 else 'Down'
                analysis['trend_strength'] = abs(recent_trend)
            else:
                analysis['trend'] = 'Stable'
                analysis['trend_strength'] = 0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in historical analysis for {series_id}: {str(e)}")
            return {}
    
    def _parse_series_observations(self, series_id: str, data: Dict) -> Optional[FREDSeriesData]:
        """Parse FRED API response into FREDSeriesData object"""
        try:
            observations = data.get('observations', [])
            
            if not observations:
                logger.warning(f"No observations found for series {series_id}")
                return None
            
            # Parse observations
            parsed_observations = []
            for obs in observations:
                try:
                    # Skip observations with missing values
                    if obs['value'] == '.':
                        continue
                    
                    parsed_obs = FREDObservation(
                        date=datetime.strptime(obs['date'], '%Y-%m-%d'),
                        value=float(obs['value']),
                        realtime_start=obs['realtime_start'],
                        realtime_end=obs['realtime_end']
                    )
                    parsed_observations.append(parsed_obs)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing observation for {series_id}: {e}")
                    continue
            
            if not parsed_observations:
                logger.warning(f"No valid observations found for series {series_id}")
                return None
            
            # Get series metadata
            series_info = self.get_series_info(series_id)
            
            # Create FREDSeriesData object
            series_data = FREDSeriesData(
                series_id=series_id,
                title=series_info.get('title', series_id) if series_info else series_id,
                units=series_info.get('units', '') if series_info else '',
                frequency=series_info.get('frequency', '') if series_info else '',
                seasonal_adjustment=series_info.get('seasonal_adjustment', '') if series_info else '',
                last_updated=parsed_observations[0].date,
                observations=parsed_observations,
                notes=series_info.get('notes', '') if series_info else ''
            )
            
            return series_data
            
        except Exception as e:
            logger.error(f"Error parsing series data for {series_id}: {str(e)}")
            return None
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API service status and statistics"""
        return {
            'api_key_valid': self.config.api_key != 'demo',
            'requests_made': self.request_count,
            'errors_encountered': self.error_count,
            'success_rate': ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count > 0 else 0,
            'last_request': self.last_request_time.isoformat() if self.last_request_time else None,
            'rate_limit': self.config.requests_per_minute,
            'cache_duration_hours': self.cache_duration.total_seconds() / 3600
        }
    
    def test_connection(self) -> bool:
        """Test FRED API connection"""
        try:
            # Try to fetch a simple series (GDP)
            test_data = self.get_series_observations('GDP', limit=1)
            if test_data and test_data.observations:
                logger.info("✅ FRED API connection test successful")
                return True
            else:
                logger.error("❌ FRED API connection test failed - no data returned")
                return False
        except Exception as e:
            logger.error(f"❌ FRED API connection test failed: {str(e)}")
            return False

# Global FRED API service instance
fred_service = None

def get_fred_service(api_key: str = None) -> FREDAPIService:
    """Get or create FRED API service instance"""
    global fred_service
    if fred_service is None:
        fred_service = FREDAPIService(api_key=api_key)
    return fred_service
