#!/usr/bin/env python3
"""
Angel One Data Downloader
Download historical stock data from Angel One SmartAPI
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pyotp
from pathlib import Path
import warnings

from .angel_one_config import AngelOneConfig

warnings.filterwarnings('ignore')

class AngelOneDataDownloader:
    """Data downloader for Angel One SmartAPI."""
    
    def __init__(self):
        """Initialize Angel One data downloader."""
        self.config = AngelOneConfig()
        # Use the exact credentials from MyOwnAngleLogin.py
        self.api_key = "3PMAARNa "  # Note the space at the end
        self.client_code = "D54448"
        self.client_pin = "2251"
        self.totp_secret = "NP4SAXOKMTJQZ4KZP2TBTYXRCE"
        
        self.session = requests.Session()
        self.jwt_token = None
        self.refresh_token = None
        self.last_request_time = 0
        
        # Rate limiting
        self.request_count = 0
        self.request_window_start = time.time()
        
        # Cache directory
        self.cache_dir = Path("data/angel_one_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Instruments cache
        self.instruments_cache = None
        self.instruments_cache_time = None
    
    def authenticate(self, totp_code: str = None) -> bool:
        """Authenticate with Angel One API."""
        try:
            print("🔐 Authenticating with Angel One API...")
            
            # Generate TOTP if not provided
            if not totp_code:
                totp_code = pyotp.TOTP(self.totp_secret).now()
                print(f"✅ TOTP generated: {totp_code}")
            
            # Prepare login payload
            login_payload = {
                "clientcode": self.client_code,
                "password": self.client_pin,
                "totp": totp_code
            }
            
            # Use the working headers from MyOwnAngleLogin.py
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "XX:XX:XX:XX:XX:XX",
                "X-PrivateKey": self.api_key
            }
            
            # Use the working login URL
            login_url = "https://apiconnect.angelbroking.com/rest/auth/angelbroking/user/v1/loginByPassword"
            
            response = requests.post(
                login_url,
                json=login_payload,  # Use json parameter instead of data
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == True:  # Use "status" instead of "success"
                    self.jwt_token = data["data"]["jwtToken"]
                    self.refresh_token = data["data"]["refreshToken"]
                    print("✅ Authentication successful!")
                    print(f"   JWT Token: {self.jwt_token[:20]}...")
                    return True
                else:
                    error_code = data.get("errorCode", "Unknown")
                    error_msg = data.get("message", "Unknown error")
                    print(f"❌ Authentication failed: {error_msg} (Code: {error_code})")
                    return False
            else:
                print(f"❌ Authentication failed with status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def get_instruments(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get instruments list with caching."""
        try:
            # Check cache (valid for 1 hour)
            if (not force_refresh and 
                self.instruments_cache is not None and 
                self.instruments_cache_time and 
                (datetime.now() - self.instruments_cache_time).seconds < 3600):
                
                print("📋 Using cached instruments data")
                return self.instruments_cache
            
            print("📥 Downloading instruments file...")
            instruments_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            
            response = requests.get(instruments_url, timeout=30)
            response.raise_for_status()
            
            instruments = response.json()
            df = pd.DataFrame(instruments)
            
            # Cache the data
            self.instruments_cache = df
            self.instruments_cache_time = datetime.now()
            
            print(f"✅ Downloaded {len(df)} instruments")
            return df
            
        except Exception as e:
            print(f"❌ Error downloading instruments: {e}")
            if self.instruments_cache is not None:
                print("📋 Using cached instruments data as fallback")
                return self.instruments_cache
            raise
    
    def get_symbol_token(self, symbol_name: str, exchange: str = "NSE") -> str:
        """Get symbol token for a given symbol and exchange."""
        try:
            df = self.get_instruments()
            
            # Filter by exchange and symbol (using contains like in working code)
            df_filtered = df[
                (df["exch_seg"] == exchange) & 
                (df["symbol"].str.upper().str.contains(symbol_name.upper()))
            ]
            
            if df_filtered.empty:
                raise Exception(f"❌ Symbol {symbol_name} not found in {exchange} instruments list!")
            
            symbol_token = df_filtered.iloc[0]["token"]
            symbol_info = df_filtered.iloc[0]
            
            print(f"✅ Found Symbol: {symbol_name}")
            print(f"   Token: {symbol_token}")
            print(f"   Exchange: {symbol_info['exch_seg']}")
            print(f"   Name: {symbol_info.get('name', 'N/A')}")
            
            return str(symbol_token)
            
        except Exception as e:
            print(f"❌ Error getting symbol token: {e}")
            raise
    
    def refresh_auth_token(self) -> bool:
        """Refresh authentication token."""
        try:
            if not self.refresh_token:
                print("❌ No refresh token available")
                return False
            
            payload = self.config.get_token_refresh_payload(self.refresh_token)
            url = self.config.base_url + self.config.endpoints['generate_tokens']
            headers = self.config.default_headers.copy()
            
            response = self._make_request('POST', url, headers=headers, json=payload)
            
            if response and response.get('status'):
                data = response.get('data', {})
                self.jwt_token = data.get('jwtToken')
                self.refresh_token = data.get('refreshToken')
                
                if self.jwt_token:
                    print("✅ Token refreshed successfully")
                    return True
                else:
                    print("❌ Token refresh failed: Missing JWT token")
                    return False
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                print(f"❌ Token refresh failed: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Token refresh error: {e}")
            return False
    
    def get_ltp_data(self, symbol_name: str, exchange: str = "NSE") -> Dict:
        """Get Last Traded Price (LTP) data for a symbol."""
        try:
            # Ensure authentication
            if not self.jwt_token:
                if not self.authenticate():
                    raise Exception("Authentication failed")
            
            # Get symbol token
            symbol_token = self.get_symbol_token(symbol_name, exchange)
            
            # Prepare LTP payload
            ltp_payload = {
                "exchange": exchange,
                "tradingsymbol": symbol_name,
                "symboltoken": symbol_token
            }
            
            # Prepare headers with JWT token (using working format)
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "XX:XX:XX:XX:XX:XX",
                "X-PrivateKey": self.api_key,
                "Authorization": f"Bearer {self.jwt_token}"
            }
            
            # Make LTP request (using working format)
            ltp_url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/order/v1/getLtpData"
            
            print(f"💰 Fetching LTP Data for {symbol_name}...")
            
            response = requests.post(
                ltp_url,
                data=json.dumps(ltp_payload),
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == True:
                    ltp_data = data.get("data", {})
                    print(f"✅ LTP Data retrieved")
                    return ltp_data
                else:
                    error_msg = data.get("message", "Unknown error")
                    print(f"❌ LTP data failed: {error_msg}")
                    return {}
            else:
                print(f"❌ LTP data failed with status code: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Error getting LTP data: {e}")
            return {}
    
    def save_data(self, df: pd.DataFrame, symbol_name: str, format: str = "both") -> None:
        """Save data to files."""
        try:
            if df.empty:
                print("⚠️ No data to save")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format in ["json", "both"]:
                json_file = f"data/{symbol_name}_historical_{timestamp}.json"
                os.makedirs("data", exist_ok=True)
                
                # Convert DataFrame to JSON format (handle timestamp serialization)
                df_copy = df.copy()
                df_copy["Datetime"] = df_copy["Datetime"].astype(str)  # Convert timestamps to strings
                
                json_data = {
                    "symbol": symbol_name,
                    "timestamp": timestamp,
                    "records": len(df),
                    "data": df_copy.to_dict('records')
                }
                
                with open(json_file, "w") as f:
                    json.dump(json_data, f, indent=4)
                print(f"✅ Data saved to {json_file}")
            
            if format in ["csv", "both"]:
                csv_file = f"data/{symbol_name}_historical_{timestamp}.csv"
                os.makedirs("data", exist_ok=True)
                
                df.to_csv(csv_file, index=False)
                print(f"✅ Data saved to {csv_file}")
                
        except Exception as e:
            print(f"❌ Error saving data: {e}")
    
    def test_connection(self) -> bool:
        """Test the complete connection flow."""
        try:
            print("🧪 Testing Angel One Connection...")
            print("=" * 50)
            
            # Test 1: Authentication
            print("\n📋 Test 1: Authentication")
            if not self.authenticate():
                return False
            
            # Test 2: Get instruments
            print("\n📋 Test 2: Get Instruments")
            instruments_df = self.get_instruments()
            if instruments_df.empty:
                print("❌ No instruments data received")
                return False
            
            # Test 3: Get symbol token
            print("\n📋 Test 3: Get Symbol Token")
            try:
                token = self.get_symbol_token("RELIANCE", "NSE")
                print(f"✅ Symbol token test passed: {token}")
            except Exception as e:
                print(f"❌ Symbol token test failed: {e}")
                return False
            
            # Test 4: Get historical data
            print("\n📋 Test 4: Get Historical Data")
            hist_data = self.get_historical_data("RELIANCE", "NSE", "ONE_DAY", days_back=7)
            if hist_data.empty:
                print("❌ No historical data received")
                return False
            
            # Test 5: Get LTP data
            print("\n📋 Test 5: Get LTP Data")
            ltp_data = self.get_ltp_data("RELIANCE", "NSE")
            if not ltp_data:
                print("❌ No LTP data received")
                return False
            
            print("\n🎉 All tests passed! Angel One connection is working.")
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def get_historical_data(self, symbol_name: str, exchange: str = "NSE",
                          interval: str = "ONE_DAY", from_date: str = None, to_date: str = None,
                          days_back: int = 30) -> pd.DataFrame:
        """Get historical data for a symbol."""
        try:
            # Ensure authentication
            if not self.jwt_token:
                if not self.authenticate():
                    raise Exception("Authentication failed")
            
            # Get symbol token
            symbol_token = self.get_symbol_token(symbol_name, exchange)
            
            # Set default dates if not provided
            if not from_date:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # Prepare historical data payload
            candle_payload = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            # Prepare headers with JWT token (using working format)
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "XX:XX:XX:XX:XX:XX",
                "X-PrivateKey": self.api_key,
                "Authorization": f"Bearer {self.jwt_token}"
            }
            
            # Make historical data request (using working format)
            historical_url = "https://apiconnect.angelbroking.com/rest/secure/angelbroking/historical/v1/getCandleData"
            
            print(f"📊 Fetching Historical Data...")
            print(f"   Symbol: {symbol_name}")
            print(f"   Exchange: {exchange}")
            print(f"   Interval: {interval}")
            print(f"   Period: {from_date} to {to_date}")
            
            response = requests.post(
                historical_url,
                data=json.dumps(candle_payload),
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == True:
                    candles = data.get("data", [])
                    
                    if not candles:
                        print("⚠️ No data received for the specified period")
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    df_candles = pd.DataFrame(candles, columns=[
                        "Datetime", "Open", "High", "Low", "Close", "Volume"
                    ])
                    
                    # Convert data types
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        df_candles[col] = pd.to_numeric(df_candles[col], errors='coerce')
                    
                    # Convert datetime
                    df_candles["Datetime"] = pd.to_datetime(df_candles["Datetime"])
                    
                    print(f"✅ Retrieved {len(df_candles)} records")
                    return df_candles
                else:
                    error_msg = data.get("message", "Unknown error")
                    print(f"❌ Historical data failed: {error_msg}")
                    return pd.DataFrame()
            else:
                print(f"❌ Historical data failed with status code: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error getting historical data: {e}")
            return pd.DataFrame()
    
    def search_symbol(self, search_text: str) -> Optional[List[Dict]]:
        """Search for symbols."""
        try:
            if not self._ensure_authenticated():
                return None
            
            payload = {
                'searchscrip': search_text
            }
            
            url = self.config.base_url + self.config.endpoints['search_scrip']
            headers = self.config.get_auth_headers(self.jwt_token)
            
            response = self._make_request('POST', url, headers=headers, json=payload)
            
            if response and response.get('status'):
                return response.get('data', [])
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                print(f"❌ Symbol search failed: {error_msg}")
                return None
                
        except Exception as e:
            print(f"❌ Symbol search error: {e}")
            return None
    
    def get_quote(self, symbol: str, token: str = None, exchange: str = 'NSE') -> Optional[Dict]:
        """Get quote data for a symbol."""
        try:
            if not self._ensure_authenticated():
                return None
            
            # Get token if not provided
            if not token:
                symbol_info = self.config.get_symbol_info(symbol)
                if symbol_info:
                    token = symbol_info['token']
                    exchange = symbol_info['exchange']
                else:
                    print(f"❌ Token not found for symbol: {symbol}")
                    return None
            
            payload = {
                'exchange': exchange,
                'tradingsymbol': symbol,
                'symboltoken': token
            }
            
            url = self.config.base_url + self.config.endpoints['quote']
            headers = self.config.get_auth_headers(self.jwt_token)
            
            response = self._make_request('POST', url, headers=headers, json=payload)
            
            if response and response.get('status'):
                return response.get('data', {})
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                print(f"❌ Quote fetch failed: {error_msg}")
                return None
                
        except Exception as e:
            print(f"❌ Quote fetch error: {e}")
            return None
    
    def download_stock_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Download stock data with period specification."""
        try:
            # Convert period to date range
            end_date = datetime.now()
            
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '5d':
                start_date = end_date - timedelta(days=5)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=730)
            elif period == '5y':
                start_date = end_date - timedelta(days=1825)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
            
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Map interval to Angel One format
            interval_mapping = self.config.get_interval_mapping()
            angel_interval = interval_mapping.get(interval, 'ONE_DAY')
            
            return self.get_historical_data(symbol_name=symbol, from_date=from_date, to_date=to_date, interval=angel_interval)
            
        except Exception as e:
            print(f"❌ Stock data download error: {e}")
            return None
    
    def _ensure_authenticated(self) -> bool:
        """Ensure authentication is valid."""
        if not self.jwt_token:
            print("❌ Not authenticated. Please authenticate first.")
            return False
        
        # Check if token needs refresh (simple check - in production, check expiry)
        if time.time() - self.last_request_time > 3600:  # 1 hour
            return self.refresh_auth_token()
        
        return True
    
    def _make_request(self, method: str, url: str, endpoint: str = 'historical_data', **kwargs) -> Optional[Dict]:
        """Make API request with rate limiting."""
        try:
            # Rate limiting
            self._check_rate_limit(endpoint)
            
            response = self.session.request(method, url, **kwargs)
            self.last_request_time = time.time()
            self.request_count += 1
            
            print(f"🔍 API Request: {method} {url}")
            print(f"🔍 Response Status: {response.status_code}")
            print(f"🔍 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                try:
                    # Check if response has content
                    if response.content:
                        return response.json()
                    else:
                        print("⚠️ Empty response received")
                        return None
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"🔍 Response content: {response.text[:500]}")
                    return None
            elif response.status_code == 401:
                print("⚠️ Token expired, attempting refresh...")
                if self.refresh_auth_token():
                    # Retry the request
                    return self._make_request(method, url, **kwargs)
                else:
                    print("❌ Token refresh failed")
                    return None
            else:
                print(f"❌ API request failed with status {response.status_code}")
                print(f"🔍 Response content: {response.text[:500]}")
                return None
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None
    
    def _check_rate_limit(self, endpoint: str = 'historical_data'):
        """Check and enforce rate limits for specific endpoints."""
        current_time = time.time()
        
        # Get rate limits for the specific endpoint
        endpoint_limits = self.config.rate_limits.get(endpoint, {})
        per_second = endpoint_limits.get('per_second', 1)
        per_minute = endpoint_limits.get('per_minute', 60)
        
        # Ensure minimum delay between requests (for per-second limits)
        if per_second:
            min_delay = 1.0 / per_second
            time_since_last = current_time - self.last_request_time
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                print(f"⏳ Rate limiting: waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
        
        # Check per-minute limits
        if per_minute:
            # Reset counter if window has passed
            if current_time - self.request_window_start >= 60:  # 1 minute window
                self.request_count = 0
                self.request_window_start = current_time
            
            # Check if we're over the limit
            if self.request_count >= per_minute:
                sleep_time = 60 - (current_time - self.request_window_start)
                if sleep_time > 0:
                    print(f"⏳ Rate limit reached ({per_minute} req/min), waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    self.request_count = 0
                    self.request_window_start = time.time()
        
        # Update last request time
        self.last_request_time = time.time()
    
    def _parse_historical_data(self, data: List[Dict], symbol: str) -> Optional[pd.DataFrame]:
        """Parse historical data response."""
        try:
            if not data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Standardize column names
            column_mapping = {
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            if 'Date' in df.columns:
                df = df.sort_values('Date').reset_index(drop=True)
            
            # Add symbol column
            df['Symbol'] = symbol
            
            return df
            
        except Exception as e:
            print(f"❌ Data parsing error: {e}")
            return None
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached data."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.csv"
            if cache_file.exists():
                # Check if cache is less than 1 hour old
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < 3600:  # 1 hour
                    df = pd.read_csv(cache_file)
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df
        except Exception as e:
            print(f"⚠️ Cache read error: {e}")
        return None
    
    def _cache_data(self, cache_key: str, df: pd.DataFrame):
        """Cache data."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.csv"
            df.to_csv(cache_file, index=False)
        except Exception as e:
            print(f"⚠️ Cache write error: {e}")
    
    def logout(self):
        """Logout from Angel One API."""
        try:
            if self.jwt_token:
                url = self.config.base_url + self.config.endpoints['logout']
                headers = self.config.get_auth_headers(self.jwt_token)
                
                response = self._make_request('POST', url, headers=headers)
                
                if response and response.get('status'):
                    print("✅ Logout successful")
                else:
                    print("⚠️ Logout failed")
            
            # Clear tokens
            self.jwt_token = None
            self.refresh_token = None
            
        except Exception as e:
            print(f"⚠️ Logout error: {e}")
    
    def get_user_profile(self) -> Optional[Dict]:
        """Get user profile information."""
        try:
            if not self._ensure_authenticated():
                return None
            
            url = self.config.base_url + self.config.endpoints['profile']
            headers = self.config.get_auth_headers(self.jwt_token)
            
            response = self._make_request('GET', url, headers=headers)
            
            if response and response.get('status'):
                return response.get('data', {})
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                print(f"❌ Profile fetch failed: {error_msg}")
                return None
                
        except Exception as e:
            print(f"❌ Profile fetch error: {e}")
            return None
    
    def get_rms_limits(self) -> Optional[Dict]:
        """Get RMS limits."""
        try:
            if not self._ensure_authenticated():
                return None
            
            url = self.config.base_url + self.config.endpoints['rms_limits']
            headers = self.config.get_auth_headers(self.jwt_token)
            
            response = self._make_request('GET', url, headers=headers)
            
            if response and response.get('status'):
                return response.get('data', {})
            else:
                error_msg = response.get('message', 'Unknown error') if response else 'No response'
                print(f"❌ RMS limits fetch failed: {error_msg}")
                return None
                
        except Exception as e:
            print(f"❌ RMS limits fetch error: {e}")
            return None
    
    def __del__(self):
        """Cleanup on object destruction."""
        self.logout()

def main():
    """Main function for testing."""
    print("🚀 Angel One Data Downloader Test")
    print("=" * 60)
    
    # Initialize downloader
    downloader = AngelOneDataDownloader()
    
    # Test connection
    if downloader.test_connection():
        print("\n✅ Angel One data downloader is ready to use!")
        
        # Example: Get data for a specific symbol
        print("\n📊 Example: Getting data for INFY")
        try:
            infy_data = downloader.get_historical_data(
                symbol_name="INFY",
                exchange="NSE",
                interval="ONE_DAY",
                days_back=30
            )
            
            if not infy_data.empty:
                downloader.save_data(infy_data, "INFY", "both")
                print(f"📈 Sample data:")
                print(infy_data.head())
            else:
                print("❌ No data received for INFY")
                
        except Exception as e:
            print(f"❌ Error getting INFY data: {e}")
    else:
        print("\n❌ Angel One data downloader test failed")

if __name__ == "__main__":
    main()
