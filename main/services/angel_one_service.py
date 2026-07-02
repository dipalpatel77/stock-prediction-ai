"""
Angel One API Service for Real-Time and Historical Data

This service implements the Angel One API endpoints for fetching:
- Historical candle data
- Open Interest data
- Real-time market data
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import pyotp
import sys
import os

# Add the main directory to the path to import dynamic lookup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamic_stock_lookup import DynamicStockLookup

logger = logging.getLogger(__name__)

class AngelOneService:
    """Angel One API Service for fetching real market data"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Angel One service with API credentials
        
        Args:
            config: Configuration dictionary with API credentials
        """
        self.config = config
        self.base_url = "https://apiconnect.angelone.in"
        self.api_key = config.get('api_key', 'fRBSMrnn')
        self.client_code = config.get('api_secret', 'D54448')
        self.client_pin = config.get('access_token', '2251')
        self.totp_secret = config.get('totp_secret', 'NP4SAXOKMTJQZ4KZP2TBTYXRCE')
        
        # Initialize authentication tokens
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        self.is_authenticated = False
        
        # Initialize dynamic stock lookup
        self.dynamic_lookup = DynamicStockLookup()
        
        # Common headers for all requests - Updated for Angel One API
        self.headers = {
            'X-PrivateKey': self.api_key,
            'Accept': 'application/json',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '192.168.1.1',  # Default local IP
            'X-ClientPublicIP': '192.168.1.1',  # Default public IP
            'X-MACAddress': '00:00:00:00:00:00',  # Default MAC
            'X-UserType': 'USER',
            'Content-Type': 'application/json'
        }
        
        # Stock symbol mappings with tokens and exchanges
        self.symbol_tokens = {
            'RELIANCE': '500325',
            'TCS': '2955',
            'INFY': '4085',
            'HDFC': '1333',
            'ICICIBANK': '4963',
            'WIPRO': '9695',
            'BHARTIARTL': '99926000',
            'HDFCBANK': '1333',
            'KOTAKBANK': '4929',
            'ITC': '4241',
            'LT': '4671',
            'SBIN': '7793',
            'ASIANPAINT': '1660',
            'MARUTI': '10999',
            'NESTLEIND': '4598',
            'POWERGRID': '8985',
            'NTPC': '10749',
            'ONGC': '2475',
            'COALINDIA': '2031',
            'TITAN': '11543',
            'ULTRACEMCO': '11532',
            'AXISBANK': '5900',
            'BAJFINANCE': '811',
            'BAJAJFINSV': '812',
            'DRREDDY': '2254',
            'EICHERMOT': '910',
            'GRASIM': '3152',
            'HCLTECH': '7229',
            'HEROMOTOCO': '3456',
            'HINDALCO': '1363',
            'INDUSINDBK': '5258',
            'JSWSTEEL': '14977',
            'M&M': '11536',
            'GIPCL': '1145'
        }
        
        # Exchange mappings for stocks
        self.stock_exchanges = {
            'ONGC': 'NSE',
            'RELIANCE': 'BSE',
            'TCS': 'BSE',
            'INFY': 'BSE',
            'HDFC': 'BSE',
            'ICICIBANK': 'BSE',
            'WIPRO': 'BSE',
            'BHARTIARTL': 'BSE',
            'HDFCBANK': 'BSE',
            'KOTAKBANK': 'BSE',
            'ITC': 'BSE',
            'LT': 'BSE',
            'SBIN': 'BSE',
            'ASIANPAINT': 'BSE',
            'MARUTI': 'BSE',
            'NESTLEIND': 'BSE',
            'POWERGRID': 'BSE',
            'NTPC': 'NSE',
            'COALINDIA': 'BSE',
            'TITAN': 'BSE',
            'ULTRACEMCO': 'BSE',
            'AXISBANK': 'BSE',
            'BAJFINANCE': 'BSE',
            'GIPCL': 'NSE',
        }
        
        logger.info("Angel One Service initialized with real API credentials")
    
    def get_stock_exchange(self, symbol: str) -> str:
        """Get the correct exchange for a stock symbol"""
        return self.stock_exchanges.get(symbol.upper(), 'BSE')  # Default to BSE
    
    def get_dynamic_token_and_exchange(self, symbol: str) -> tuple:
        """
        Get token and exchange dynamically for any stock symbol
        
        Args:
            symbol: Stock symbol to lookup
            
        Returns:
            Tuple of (token, exchange) or (None, None) if not found
        """
        try:
            token, exchange = self.dynamic_lookup.get_token_and_exchange(symbol)
            if token and exchange:
                logger.info(f"Dynamic lookup found: {symbol} -> Token: {token}, Exchange: {exchange}")
                return token, exchange
            else:
                logger.warning(f"Dynamic lookup failed for: {symbol}")
                return None, None
        except Exception as e:
            logger.error(f"Error in dynamic lookup for {symbol}: {e}")
            return None, None
    
    def generate_totp(self) -> str:
        """
        Generate TOTP code using the TOTP secret
        
        Returns:
            6-digit TOTP code as string
        """
        try:
            # Create TOTP object with the secret
            totp = pyotp.TOTP(self.totp_secret)
            # Generate current TOTP code
            totp_code = totp.now()
            logger.info(f"Generated TOTP code: {totp_code}")
            return totp_code
        except Exception as e:
            logger.error(f"Failed to generate TOTP: {e}")
            # Fallback to a default code if TOTP generation fails
            return "123456"
    
    def authenticate(self) -> bool:
        """
        Authenticate with Angel One API to get JWT token using proper login flow
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Step 1: Call login API with credentials
            login_url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
            
            # Generate TOTP code
            totp_code = self.generate_totp()
            
            # Prepare login payload
            login_payload = {
                "clientcode": self.client_code,
                "password": self.client_pin,
                "totp": totp_code,  # Generated TOTP code
                "state": "live"
            }
            
            # Prepare login headers
            login_headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserType': 'USER',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '192.168.1.1',
                'X-ClientPublicIP': '192.168.1.1',
                'X-MACAddress': '00:00:00:00:00:00',
                'X-PrivateKey': self.api_key
            }
            
            logger.info(f"Attempting login to: {login_url}")
            logger.info(f"Login payload: {login_payload}")
            
            # Make login request
            login_response = requests.post(login_url, headers=login_headers, json=login_payload)
            
            logger.info(f"Login response status: {login_response.status_code}")
            logger.info(f"Login response: {login_response.text[:500]}...")
            
            if login_response.status_code == 200:
                login_data = login_response.json()
                if login_data.get('status') and login_data.get('data'):
                    # Extract tokens from response
                    data = login_data['data']
                    self.jwt_token = data.get('jwtToken')
                    self.refresh_token = data.get('refreshToken')
                    self.feed_token = data.get('feedToken')
                    
                    if self.jwt_token:
                        self.is_authenticated = True
                        logger.info("Authentication successful - JWT token obtained")
                        return True
                    else:
                        logger.error("No JWT token in login response")
                        return False
                else:
                    logger.error(f"Login failed: {login_data.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"Login request failed with status {login_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def refresh_auth_token(self) -> bool:
        """
        Refresh the JWT token using refresh token
        
        Returns:
            True if token refresh successful, False otherwise
        """
        try:
            if not self.refresh_token:
                logger.error("No refresh token available")
                return False
            
            # Call generate token API
            refresh_url = f"{self.base_url}/rest/auth/angelbroking/jwt/v1/generateTokens"
            
            refresh_payload = {
                "refreshToken": self.refresh_token
            }
            
            refresh_headers = {
                'Authorization': f'Bearer {self.jwt_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserType': 'USER',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '192.168.1.1',
                'X-ClientPublicIP': '192.168.1.1',
                'X-MACAddress': '00:00:00:00:00:00',
                'X-PrivateKey': self.api_key
            }
            
            logger.info("Refreshing authentication token")
            refresh_response = requests.post(refresh_url, headers=refresh_headers, json=refresh_payload)
            
            if refresh_response.status_code == 200:
                refresh_data = refresh_response.json()
                if refresh_data.get('status') and refresh_data.get('data'):
                    data = refresh_data['data']
                    self.jwt_token = data.get('jwtToken')
                    self.refresh_token = data.get('refreshToken')
                    self.feed_token = data.get('feedToken')
                    
                    logger.info("Token refresh successful")
                    return True
                else:
                    logger.error(f"Token refresh failed: {refresh_data.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"Token refresh request failed with status {refresh_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    def get_historical_data(self, symbol: str, interval: str = 'ONE_DAY', 
                           from_date: str = None, to_date: str = None, 
                           days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get historical candle data from Angel One API
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            interval: Data interval (ONE_MINUTE, FIVE_MINUTE, ONE_HOUR, ONE_DAY, etc.)
            from_date: Start date in 'YYYY-MM-DD HH:MM' format
            to_date: End date in 'YYYY-MM-DD HH:MM' format
            days: Number of days to fetch (if dates not provided)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Get symbol token - try dynamic lookup first, then fallback to hardcoded
            symbol_token = None
            exchange = None
            
            # Try dynamic lookup first
            symbol_token, exchange = self.get_dynamic_token_and_exchange(symbol)
            
            # Fallback to hardcoded tokens if dynamic lookup fails
            if not symbol_token:
                symbol_token = self.symbol_tokens.get(symbol.upper())
                if symbol_token:
                    exchange = self.get_stock_exchange(symbol)
                    logger.info(f"Using hardcoded token for {symbol}: {symbol_token}")
            
            if not symbol_token:
                logger.error(f"Symbol token not found for {symbol} (tried dynamic and hardcoded lookup)")
                return None
            
            # Set default dates if not provided
            if not from_date or not to_date:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=days)
                from_date = from_date.strftime('%Y-%m-%d 09:15')
                to_date = to_date.strftime('%Y-%m-%d 15:30')
            
            # Use the exchange from dynamic lookup or fallback to hardcoded
            if not exchange:
                exchange = self.get_stock_exchange(symbol)
            
            # Prepare request payload
            payload = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            # Make API request
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            
            # Authenticate if not already authenticated
            if not self.is_authenticated:
                if not self.authenticate():
                    logger.error("Failed to authenticate with Angel One API")
                    return None
            
            # Add authentication headers
            request_headers = self.headers.copy()
            request_headers['X-ClientCode'] = self.client_code
            request_headers['Authorization'] = f'Bearer {self.jwt_token}'
            
            logger.info(f"Making API request to: {url}")
            logger.info(f"Payload: {payload}")
            logger.info(f"Headers: {request_headers}")
            
            response = requests.post(url, headers=request_headers, json=payload)
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response content: {response.text[:500]}...")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data'):
                    # Convert to DataFrame
                    df = self._convert_to_dataframe(data['data'], symbol)
                    logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                    return df
                else:
                    # Check if token expired and try to refresh
                    if data.get('errorcode') == 'AG8001' and self.refresh_token:
                        logger.info("Token expired, attempting to refresh...")
                        if self.refresh_auth_token():
                            # Retry the request with new token
                            request_headers['Authorization'] = f'Bearer {self.jwt_token}'
                            logger.info("Retrying request with refreshed token")
                            response = requests.post(url, headers=request_headers, json=payload)
                            
                            if response.status_code == 200:
                                data = response.json()
                                if data.get('status') and data.get('data'):
                                    df = self._convert_to_dataframe(data['data'], symbol)
                                    logger.info(f"Successfully fetched {len(df)} records for {symbol} after token refresh")
                                    return df
                    
                    logger.error(f"API returned error: {data.get('message', 'Unknown error')}")
                    logger.error(f"Full response: {data}")
                    return None
            else:
                logger.error(f"API request failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return None
    
    def _convert_to_dataframe(self, data: List, symbol: str) -> pd.DataFrame:
        """
        Convert API response data to pandas DataFrame
        
        Args:
            data: Raw data from API response
            symbol: Stock symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # API returns: [timestamp, open, high, low, close, volume]
            df_data = []
            for record in data:
                if len(record) >= 6:
                    df_data.append({
                        'Date': pd.to_datetime(record[0]),
                        'Open': float(record[1]),
                        'High': float(record[2]),
                        'Low': float(record[3]),
                        'Close': float(record[4]),
                        'Volume': int(record[5])
                    })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to convert data to DataFrame: {e}")
            return pd.DataFrame()
    
    def get_multiple_intervals_data(self, symbol: str, intervals: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple intervals
        
        Args:
            symbol: Stock symbol
            intervals: List of intervals to fetch
            
        Returns:
            Dictionary with interval as key and DataFrame as value
        """
        if intervals is None:
            intervals = ['ONE_DAY', 'ONE_HOUR', 'FIFTEEN_MINUTE', 'FIVE_MINUTE']
        
        multi_interval_data = {}
        
        for interval in intervals:
            try:
                # Calculate appropriate date range based on interval
                days = self._get_max_days_for_interval(interval)
                data = self.get_historical_data(symbol, interval, days=days)
                
                if data is not None and not data.empty:
                    multi_interval_data[interval] = data
                    logger.info(f"Fetched {len(data)} records for {symbol} - {interval}")
                else:
                    logger.warning(f"No data received for {symbol} - {interval}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch {interval} data for {symbol}: {e}")
        
        return multi_interval_data
    
    def get_incremental_data(self, symbol: str, interval: str, last_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get incremental data since last update
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            last_date: Last date in the existing data
            
        Returns:
            DataFrame with new data or None if failed
        """
        try:
            # Calculate date range for incremental update
            from_date = (last_date + timedelta(days=1)).strftime('%d-%m-%Y')
            to_date = datetime.now().strftime('%d-%m-%Y')
            
            logger.info(f"Fetching incremental data for {symbol} from {from_date} to {to_date}")
            
            # Get incremental data
            data = self.get_historical_data(
                symbol=symbol,
                interval=interval,
                from_date=from_date,
                to_date=to_date
            )
            
            if data is not None and not data.empty:
                # Filter to only include data after last_date
                data = data[data.index > last_date]
                logger.info(f"Retrieved {len(data)} new records for {symbol}")
                return data
            else:
                logger.info(f"No new data available for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get incremental data for {symbol}: {e}")
            return None
    
    def _get_max_days_for_interval(self, interval: str) -> int:
        """
        Get maximum days for an interval based on Angel One API limits
        
        Args:
            interval: Data interval
            
        Returns:
            Maximum days for the interval
        """
        max_days_map = {
            'ONE_MINUTE': 30,
            'THREE_MINUTE': 60,
            'FIVE_MINUTE': 100,
            'TEN_MINUTE': 100,
            'FIFTEEN_MINUTE': 200,
            'THIRTY_MINUTE': 200,
            'ONE_HOUR': 400,
            'ONE_DAY': 2000
        }
        
        return max_days_map.get(interval, 30)
    
    def get_oi_data(self, symbol: str, interval: str = 'THREE_MINUTE',
                   from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get Open Interest data for F&O contracts
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            from_date: Start date
            to_date: End date
            
        Returns:
            DataFrame with OI data or None if failed
        """
        try:
            # Get symbol token - try dynamic lookup first, then fallback to hardcoded
            symbol_token = None
            exchange = None
            
            # Try dynamic lookup first
            symbol_token, exchange = self.get_dynamic_token_and_exchange(symbol)
            
            # Fallback to hardcoded tokens if dynamic lookup fails
            if not symbol_token:
                symbol_token = self.symbol_tokens.get(symbol.upper())
                if symbol_token:
                    exchange = self.get_stock_exchange(symbol)
                    logger.info(f"Using hardcoded token for {symbol}: {symbol_token}")
            
            if not symbol_token:
                logger.error(f"Symbol token not found for {symbol} (tried dynamic and hardcoded lookup)")
                return None
            
            # Set default dates if not provided
            if not from_date or not to_date:
                to_date = datetime.now()
                from_date = to_date - timedelta(days=1)
                from_date = from_date.strftime('%Y-%m-%d 09:15')
                to_date = to_date.strftime('%Y-%m-%d 15:30')
            
            # Prepare request payload
            payload = {
                "exchange": "NFO",
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            # Make API request
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getOIData"
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data'):
                    # Convert to DataFrame
                    df_data = []
                    for record in data['data']:
                        df_data.append({
                            'Date': pd.to_datetime(record['time']),
                            'OI': record['oi']
                        })
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    logger.info(f"Successfully fetched {len(df)} OI records for {symbol}")
                    return df
                else:
                    logger.error(f"OI API returned error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"OI API request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch OI data for {symbol}: {e}")
            return None
    
    def is_configured(self) -> bool:
        """
        Check if Angel One service is properly configured
        
        Returns:
            True if configured, False otherwise
        """
        return all([
            self.api_key and self.api_key != 'your_api_key_here',
            self.client_code and self.client_code != 'D54448',
            self.client_pin and self.client_pin != '2251'
        ])
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols
        
        Returns:
            List of available symbol names
        """
        return list(self.symbol_tokens.keys())
