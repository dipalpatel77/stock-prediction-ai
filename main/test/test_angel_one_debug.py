#!/usr/bin/env python3
"""
Debug Angel One API to find the correct symbol tokens and date formats
"""

import sys
import os
import logging
import requests
import json
import pyotp
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AngelOneDebugger:
    """Debug Angel One API to find correct parameters"""
    
    def __init__(self):
        self.config = {
            'api_key': '1TKgQThc',
            'api_secret': 'D54448', 
            'access_token': '2251',
            'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
        }
        
        self.base_url = "https://apiconnect.angelone.in"
        self.jwt_token = None
        self.is_authenticated = False
        
        # Test different symbol tokens for RELIANCE
        self.test_tokens = [
            '2881',      # Current token
            '99926000',  # Alternative token
            '2881',      # NSE token
            '2881',      # BSE token
        ]
        
        logger.info("Angel One Debugger initialized")
    
    def generate_totp(self):
        """Generate TOTP code"""
        try:
            totp = pyotp.TOTP(self.config['totp_secret'])
            totp_code = totp.now()
            logger.info(f"Generated TOTP code: {totp_code}")
            return totp_code
        except Exception as e:
            logger.error(f"Failed to generate TOTP: {e}")
            return "123456"
    
    def authenticate(self):
        """Authenticate with Angel One API"""
        try:
            login_url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
            totp_code = self.generate_totp()
            
            login_payload = {
                "clientcode": self.config['api_secret'],
                "password": self.config['access_token'],
                "totp": totp_code,
                "state": "live"
            }
            
            login_headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserType': 'USER',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '192.168.1.1',
                'X-ClientPublicIP': '192.168.1.1',
                'X-MACAddress': '00:00:00:00:00:00',
                'X-PrivateKey': self.config['api_key']
            }
            
            logger.info("Attempting authentication...")
            response = requests.post(login_url, headers=login_headers, json=login_payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data'):
                    self.jwt_token = data['data'].get('jwtToken')
                    self.is_authenticated = True
                    logger.info("✅ Authentication successful")
                    return True
                else:
                    logger.error(f"❌ Authentication failed: {data.get('message')}")
                    return False
            else:
                logger.error(f"❌ Authentication request failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False
    
    def test_symbol_tokens(self):
        """Test different symbol tokens for RELIANCE"""
        if not self.is_authenticated:
            if not self.authenticate():
                return
        
        logger.info("=" * 60)
        logger.info("TESTING DIFFERENT SYMBOL TOKENS FOR RELIANCE")
        logger.info("=" * 60)
        
        # Test different date ranges
        date_ranges = [
            {
                'name': 'Last 7 days',
                'from_date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M'),
                'to_date': datetime.now().strftime('%Y-%m-%d %H:%M')
            },
            {
                'name': 'Last 30 days',
                'from_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M'),
                'to_date': datetime.now().strftime('%Y-%m-%d %H:%M')
            },
            {
                'name': 'Last 90 days',
                'from_date': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d %H:%M'),
                'to_date': datetime.now().strftime('%Y-%m-%d %H:%M')
            }
        ]
        
        for token in self.test_tokens:
            logger.info(f"\nTesting token: {token}")
            
            for date_range in date_ranges:
                logger.info(f"  Testing {date_range['name']}: {date_range['from_date']} to {date_range['to_date']}")
                
                try:
                    # Test with ONE_DAY interval first
                    result = self.get_historical_data(
                        symbol_token=token,
                        interval='ONE_DAY',
                        from_date=date_range['from_date'],
                        to_date=date_range['to_date']
                    )
                    
                    if result and result.get('data'):
                        logger.info(f"    ✅ SUCCESS: Found {len(result['data'])} records")
                        logger.info(f"    Sample data: {result['data'][:2] if result['data'] else 'No data'}")
                        return token, date_range  # Return successful combination
                    else:
                        logger.info(f"    ❌ No data returned")
                        
                except Exception as e:
                    logger.error(f"    ❌ Error: {e}")
        
        return None, None
    
    def get_historical_data(self, symbol_token, interval, from_date, to_date):
        """Get historical data with specific parameters"""
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            
            payload = {
                "exchange": "NSE",
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            headers = {
                'X-PrivateKey': self.config['api_key'],
                'Accept': 'application/json',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '192.168.1.1',
                'X-ClientPublicIP': '192.168.1.1',
                'X-MACAddress': '00:00:00:00:00:00',
                'X-UserType': 'USER',
                'Content-Type': 'application/json',
                'X-ClientCode': self.config['api_secret'],
                'Authorization': f'Bearer {self.jwt_token}'
            }
            
            logger.info(f"    Request: {payload}")
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"    Response: {data}")
                return data
            else:
                logger.error(f"    Request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"    Request error: {e}")
            return None
    
    def test_different_exchanges(self):
        """Test different exchanges"""
        if not self.is_authenticated:
            if not self.authenticate():
                return
        
        logger.info("\n" + "=" * 60)
        logger.info("TESTING DIFFERENT EXCHANGES")
        logger.info("=" * 60)
        
        exchanges = ['NSE', 'BSE', 'NFO', 'CDS', 'MCX']
        
        for exchange in exchanges:
            logger.info(f"\nTesting exchange: {exchange}")
            
            try:
                result = self.get_historical_data_exchange(
                    symbol_token='2881',
                    exchange=exchange,
                    interval='ONE_DAY',
                    from_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M'),
                    to_date=datetime.now().strftime('%Y-%m-%d %H:%M')
                )
                
                if result and result.get('data'):
                    logger.info(f"  ✅ SUCCESS: Found {len(result['data'])} records")
                else:
                    logger.info(f"  ❌ No data returned")
                    
            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
    
    def get_historical_data_exchange(self, symbol_token, exchange, interval, from_date, to_date):
        """Get historical data with specific exchange"""
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            
            payload = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            headers = {
                'X-PrivateKey': self.config['api_key'],
                'Accept': 'application/json',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '192.168.1.1',
                'X-ClientPublicIP': '192.168.1.1',
                'X-MACAddress': '00:00:00:00:00:00',
                'X-UserType': 'USER',
                'Content-Type': 'application/json',
                'X-ClientCode': self.config['api_secret'],
                'Authorization': f'Bearer {self.jwt_token}'
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"  Response: {data}")
                return data
            else:
                logger.error(f"  Request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"  Request error: {e}")
            return None
    
    def run_debug(self):
        """Run all debug tests"""
        logger.info("🚀 Starting Angel One API Debug")
        logger.info("=" * 60)
        
        try:
            # Test 1: Different symbol tokens
            successful_token, successful_date_range = self.test_symbol_tokens()
            
            if successful_token:
                logger.info(f"\n✅ FOUND WORKING COMBINATION:")
                logger.info(f"   Token: {successful_token}")
                logger.info(f"   Date Range: {successful_date_range}")
            else:
                logger.warning("\n❌ No working combination found with different tokens")
            
            # Test 2: Different exchanges
            self.test_different_exchanges()
            
        except Exception as e:
            logger.error(f"❌ Debug failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main debug execution"""
    print("🚀 Angel One API Debug Tool")
    print("=" * 60)
    
    debugger = AngelOneDebugger()
    debugger.run_debug()

if __name__ == "__main__":
    main()
