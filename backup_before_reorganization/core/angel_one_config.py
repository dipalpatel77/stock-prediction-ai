#!/usr/bin/env python3
"""
Angel One API Configuration
Configuration settings for Angel One SmartAPI integration
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AngelOneConfig:
    """Configuration class for Angel One API integration."""
    
    def __init__(self):
        """Initialize Angel One configuration."""
        self.base_url = "https://apiconnect.angelbroking.com"
        self.api_key = os.getenv('ANGEL_ONE_API_KEY', '3PMAARNa ')
        self.client_code = os.getenv('ANGEL_ONE_CLIENT_CODE', 'D54448')
        self.client_pin = os.getenv('ANGEL_ONE_CLIENT_PIN', '2251')
        self.totp_secret = os.getenv('ANGEL_ONE_TOTP_SECRET', 'NP4SAXOKMTJQZ4KZP2TBTYXRCE')
        
        # For testing with provided tokens
        self.test_jwt_token = os.getenv('ANGEL_ONE_JWT_TOKEN', '')
        self.test_refresh_token = os.getenv('ANGEL_ONE_REFRESH_TOKEN', '')
        
        # Get public IP address
        self.public_ip = self._get_public_ip()
        
        # API endpoints - Updated to match official Angel One API
        self.endpoints = {
            'login': '/rest/auth/angelbroking/user/v1/loginByPassword',
            'generate_tokens': '/rest/auth/angelbroking/jwt/v1/generateTokens',
            'profile': '/rest/secure/angelbroking/user/v1/getProfile',
            'rms_limits': '/rest/secure/angelbroking/user/v1/getRMS',
            'logout': '/rest/secure/angelbroking/user/v1/logout',
            'place_order': '/rest/secure/angelbroking/order/v1/placeOrder',
            'modify_order': '/rest/secure/angelbroking/order/v1/modifyOrder',
            'cancel_order': '/rest/secure/angelbroking/order/v1/cancelOrder',
            'order_book': '/rest/secure/angelbroking/order/v1/getOrderBook',
            'trade_book': '/rest/secure/angelbroking/order/v1/getTradeBook',
            'ltp_data': '/rest/secure/angelbroking/order/v1/getLtpData',
            'historical_data': '/rest/secure/angelbroking/historical/v1/getCandleData',
            'search_scrip': '/rest/secure/angelbroking/order/v1/searchScrip',
            'quote': '/rest/secure/angelbroking/market/v1/quote',
            'position': '/rest/secure/angelbroking/order/v1/getPosition',
            'holding': '/rest/secure/angelbroking/portfolio/v1/getHolding',
            'all_holding': '/rest/secure/angelbroking/portfolio/v1/getAllHolding'
        }
        
        # Default headers - Updated with official Angel One API headers
        self.default_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '127.0.0.1',
            'X-ClientPublicIP': self.public_ip,
            'X-MACAddress': '00:00:00:00:00:00',  # Updated to match official format
            'X-APIKey': self.api_key  # Correct header name for API key
        }
        
        # Trading parameters
        self.trading_params = {
            'default_variety': 'NORMAL',
            'default_exchange': 'NSE',
            'default_product_type': 'DELIVERY',
            'default_duration': 'DAY',
            'default_order_type': 'LIMIT'
        }
        
        # Rate limiting - Updated based on official Angel One API limits
        self.rate_limits = {
            # Per endpoint limits
            'login': {'per_second': 1, 'per_minute': None, 'per_hour': None},
            'generate_tokens': {'per_second': 1, 'per_minute': None, 'per_hour': 1000},
            'profile': {'per_second': 3, 'per_minute': None, 'per_hour': 1000},
            'logout': {'per_second': 1, 'per_minute': None, 'per_hour': None},
            'rms_limits': {'per_second': 2, 'per_minute': None, 'per_hour': None},
            'place_order': {'per_second': 20, 'per_minute': 500, 'per_hour': 1000},
            'modify_order': {'per_second': 20, 'per_minute': 500, 'per_hour': 1000},
            'cancel_order': {'per_second': 20, 'per_minute': 500, 'per_hour': 1000},
            'order_book': {'per_second': 1, 'per_minute': None, 'per_hour': None},
            'ltp_data': {'per_second': 10, 'per_minute': 500, 'per_hour': 5000},
            'position': {'per_second': 1, 'per_minute': None, 'per_hour': None},
            'trade_book': {'per_second': 1, 'per_minute': None, 'per_hour': None},
            'search_scrip': {'per_second': 1, 'per_minute': None, 'per_hour': None},
            'holding': {'per_second': 1, 'per_minute': None, 'per_hour': None},
            'quote': {'per_second': 10, 'per_minute': 500, 'per_hour': 5000},
            'historical_data': {'per_second': 3, 'per_minute': 500, 'per_hour': 5000},
            
            # Global limits
            'max_concurrent_requests': 5,
            'default_delay': 0.1  # 100ms between requests
        }
        
        # Data download settings
        self.data_settings = {
            'default_interval': '1D',
            'max_days_back': 365,
            'batch_size': 100,
            'retry_attempts': 3,
            'retry_delay': 1  # seconds
        }
    
    def get_auth_headers(self, jwt_token: str) -> Dict[str, str]:
        """Get headers with authentication token."""
        headers = self.default_headers.copy()
        headers['Authorization'] = f'Bearer {jwt_token}'
        return headers
    
    def get_login_payload(self, totp_code: str) -> Dict:
        """Get login payload."""
        return {
            'clientcode': self.client_code,
            'password': self.client_pin,
            'totp': totp_code
        }
    
    def get_token_refresh_payload(self, refresh_token: str) -> Dict:
        """Get token refresh payload."""
        return {
            'refreshToken': refresh_token
        }
    
    def get_ltp_payload(self, symbol: str, token: str, exchange: str = 'NSE') -> Dict:
        """Get LTP data payload."""
        return {
            'exchange': exchange,
            'tradingsymbol': symbol,
            'symboltoken': token
        }
    
    def get_historical_data_payload(self, symbol: str, token: str, 
                                  exchange: str = 'NSE', interval: str = '1D',
                                  from_date: str = None, to_date: str = None) -> Dict:
        """Get historical data payload."""
        payload = {
            'exchange': exchange,
            'tradingsymbol': symbol,
            'symboltoken': token,
            'interval': interval
        }
        
        if from_date:
            # Convert date to DD-MM-YYYY format for Angel One API
            try:
                from datetime import datetime
                date_obj = datetime.strptime(from_date, '%Y-%m-%d')
                payload['fromdate'] = date_obj.strftime('%d-%m-%Y')
            except:
                payload['fromdate'] = from_date
                
        if to_date:
            # Convert date to DD-MM-YYYY format for Angel One API
            try:
                from datetime import datetime
                date_obj = datetime.strptime(to_date, '%Y-%m-%d')
                payload['todate'] = date_obj.strftime('%d-%m-%Y')
            except:
                payload['todate'] = to_date
            
        return payload
    
    def get_order_payload(self, symbol: str, token: str, quantity: int,
                         transaction_type: str, price: float = None,
                         exchange: str = 'NSE', order_type: str = 'LIMIT') -> Dict:
        """Get order placement payload."""
        payload = {
            'variety': self.trading_params['default_variety'],
            'tradingsymbol': symbol,
            'symboltoken': token,
            'transactiontype': transaction_type.upper(),
            'exchange': exchange,
            'ordertype': order_type,
            'producttype': self.trading_params['default_product_type'],
            'duration': self.trading_params['default_duration'],
            'quantity': str(quantity),
            'squareoff': '0',
            'stoploss': '0'
        }
        
        if price:
            payload['price'] = str(price)
            
        return payload
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        # If we have test tokens, we don't need TOTP secret
        if self.test_jwt_token and self.test_refresh_token:
            required_fields = ['api_key', 'client_code', 'client_pin']
        else:
            required_fields = ['api_key', 'client_code', 'client_pin', 'totp_secret']
        
        missing_fields = []
        for field in required_fields:
            if not getattr(self, field):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing required configuration fields: {missing_fields}")
            print("Please set the following environment variables:")
            for field in missing_fields:
                env_var = f'ANGEL_ONE_{field.upper()}'
                print(f"   {env_var}")
            return False
        
        return True
    
    def _get_public_ip(self) -> str:
        """Get the public IP address."""
        try:
            import requests
            response = requests.get('https://api.ipify.org', timeout=5)
            if response.status_code == 200:
                return response.text.strip()
        except Exception as e:
            print(f"⚠️ Could not get public IP: {e}")
        
        # Fallback to environment variable or default
        return os.getenv('ANGEL_ONE_PUBLIC_IP', '127.0.0.1')
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information for common Indian stocks."""
        # Common Indian stock symbols and their tokens
        symbol_map = {
            # NSE symbols
            'RELIANCE': {'token': '2885', 'exchange': 'NSE'},
            'TCS': {'token': '11536', 'exchange': 'NSE'},
            'HDFCBANK': {'token': '341', 'exchange': 'NSE'},
            'INFY': {'token': '1594', 'exchange': 'NSE'},
            'ICICIBANK': {'token': '4963', 'exchange': 'NSE'},
            'HINDUNILVR': {'token': '3045', 'exchange': 'NSE'},
            'ITC': {'token': '1660', 'exchange': 'NSE'},
            'SBIN': {'token': '3045', 'exchange': 'NSE'},
            'BHARTIARTL': {'token': '822', 'exchange': 'NSE'},
            'KOTAKBANK': {'token': '1922', 'exchange': 'NSE'},
            'AXISBANK': {'token': '590', 'exchange': 'NSE'},
            'ASIANPAINT': {'token': '22', 'exchange': 'NSE'},
            'MARUTI': {'token': '10999', 'exchange': 'NSE'},
            'SUNPHARMA': {'token': '3351', 'exchange': 'NSE'},
            'TATAMOTORS': {'token': '3456', 'exchange': 'NSE'},
            'WIPRO': {'token': '3787', 'exchange': 'NSE'},
            'ULTRACEMCO': {'token': '2950', 'exchange': 'NSE'},
            'TITAN': {'token': '3506', 'exchange': 'NSE'},
            'BAJFINANCE': {'token': '317', 'exchange': 'NSE'},
            'NESTLEIND': {'token': '2124', 'exchange': 'NSE'},
            
            # New-age Indian tech companies (may not be listed yet)
            # Note: These companies might not be available on Angel One yet
            # as they may be unlisted or have different symbols
            'ZOMATO': {'token': '12345', 'exchange': 'NSE'},  # Placeholder
            'PAYTM': {'token': '12346', 'exchange': 'NSE'},   # Placeholder
            'NYKAA': {'token': '12347', 'exchange': 'NSE'},   # Placeholder
            'DELHIVERY': {'token': '12348', 'exchange': 'NSE'}, # Placeholder
            'POLICYBZR': {'token': '12349', 'exchange': 'NSE'}, # Placeholder
            'CARTRADE': {'token': '12350', 'exchange': 'NSE'},  # Placeholder
            'FINO': {'token': '12351', 'exchange': 'NSE'},      # Placeholder
            'MAPMYINDIA': {'token': '12352', 'exchange': 'NSE'}, # Placeholder
            
            # Add more symbols as needed
        }
        
        # Check if symbol exists in our map
        if symbol.upper() in symbol_map:
            return symbol_map.get(symbol.upper())
        
        # For symbols not in our map, return None
        # This will trigger fallback to yfinance
        print(f"⚠️ Symbol '{symbol}' not found in Angel One symbol database")
        print(f"   This may be because:")
        print(f"   - The stock is not listed on NSE/BSE yet")
        print(f"   - The symbol is different (e.g., SWIGGY might be listed as BUNDL)")
        print(f"   - The stock is unlisted/private")
        print(f"   - Using fallback to yfinance...")
        
        return None
    
    def get_interval_mapping(self) -> Dict[str, str]:
        """Get interval mapping for historical data."""
        return {
            '1m': 'ONE_MINUTE',
            '5m': 'FIVE_MINUTE',
            '15m': 'FIFTEEN_MINUTE',
            '30m': 'THIRTY_MINUTE',
            '1h': 'ONE_HOUR',
            '1d': 'ONE_DAY',
            '1w': 'ONE_WEEK',
            '1M': 'ONE_MONTH'
        }
    
    def get_exchange_mapping(self) -> Dict[str, str]:
        """Get exchange mapping."""
        return {
            'NSE': 'NSE',
            'BSE': 'BSE',
            'NFO': 'NFO',
            'CDS': 'CDS',
            'MCX': 'MCX'
        }
    
    def get_product_type_mapping(self) -> Dict[str, str]:
        """Get product type mapping."""
        return {
            'delivery': 'DELIVERY',
            'intraday': 'INTRADAY',
            'margin': 'MARGIN',
            'cnc': 'CNC',
            'co': 'CO',
            'mis': 'MIS'
        }
    
    def get_order_type_mapping(self) -> Dict[str, str]:
        """Get order type mapping."""
        return {
            'market': 'MARKET',
            'limit': 'LIMIT',
            'stop_loss': 'STOP_LOSS',
            'stop_loss_market': 'STOP_LOSS_MARKET'
        }
    
    def get_transaction_type_mapping(self) -> Dict[str, str]:
        """Get transaction type mapping."""
        return {
            'buy': 'BUY',
            'sell': 'SELL'
        }
    
    def print_config_summary(self):
        """Print configuration summary."""
        print("🔧 Angel One Configuration Summary:")
        print(f"   Base URL: {self.base_url}")
        print(f"   Client Code: {self.client_code}")
        print(f"   API Key: {'✅ Set' if self.api_key else '❌ Not Set'}")
        print(f"   Client PIN: {'✅ Set' if self.client_pin else '❌ Not Set'}")
        print(f"   TOTP Secret: {'✅ Set' if self.totp_secret else '❌ Not Set'}")
        print(f"   Available Endpoints: {len(self.endpoints)}")
        print(f"   Rate Limits: {len(self.rate_limits)} endpoint limits configured")
        print(f"   Data Settings: {self.data_settings['max_days_back']} days max")
