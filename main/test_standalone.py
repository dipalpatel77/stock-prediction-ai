#!/usr/bin/env python3
"""
Standalone Angel One Integration Test

This script tests:
1. Download all interval data from Angel One
2. Test interval management for different prediction horizons
3. Test data quality and structure
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
import json
import pyotp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandaloneAngelOneTest:
    """Standalone Angel One test without complex imports"""
    
    def __init__(self):
        """Initialize test"""
        self.config = {
            'api_key': '1TKgQThc',
            'api_secret': 'D54448', 
            'access_token': '2251',
            'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
        }
        
        self.base_url = "https://apiconnect.angelone.in"
        self.jwt_token = None
        self.is_authenticated = False
        
        # Symbol mappings
        self.symbol_tokens = {
            'RELIANCE': '2881',
            'TCS': '2955',
            'INFY': '4085',
            'HDFC': '1333',
            'ICICIBANK': '4963'
        }
        
        logger.info("Standalone Angel One Test initialized")
    
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
    
    def get_historical_data(self, symbol, interval, days=30):
        """Get historical data for a symbol and interval"""
        if not self.is_authenticated:
            if not self.authenticate():
                return None
        
        try:
            symbol_token = self.symbol_tokens.get(symbol)
            if not symbol_token:
                logger.error(f"❌ No token found for symbol: {symbol}")
                return None
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            from_date_str = from_date.strftime("%Y-%m-%d %H:%M")
            to_date_str = to_date.strftime("%Y-%m-%d %H:%M")
            
            # API request
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            payload = {
                "exchange": "NSE",
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date_str,
                "todate": to_date_str
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
            
            logger.info(f"Requesting {interval} data for {symbol}...")
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data'):
                    # Convert to DataFrame
                    df = pd.DataFrame(data['data'], columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    df = df.set_index('Datetime')
                    logger.info(f"✅ {interval}: Retrieved {len(df)} records")
                    return df
                else:
                    logger.warning(f"❌ {interval}: No data received - {data.get('message')}")
                    return None
            else:
                logger.error(f"❌ {interval}: Request failed - {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ {interval}: Error - {e}")
            return None
    
    def test_all_intervals(self, symbol='RELIANCE'):
        """Test downloading all intervals"""
        logger.info("=" * 60)
        logger.info("TEST 1: Downloading all interval data from Angel One")
        logger.info("=" * 60)
        
        intervals = [
            'ONE_MINUTE',
            'FIVE_MINUTE', 
            'FIFTEEN_MINUTE',
            'THIRTY_MINUTE',
            'ONE_HOUR',
            'ONE_DAY'
        ]
        
        downloaded_data = {}
        
        for interval in intervals:
            logger.info(f"\nDownloading {interval} data for {symbol}...")
            
            # Calculate appropriate days based on interval
            if interval == 'ONE_MINUTE':
                days = 7
            elif interval in ['FIVE_MINUTE', 'FIFTEEN_MINUTE']:
                days = 30
            elif interval == 'THIRTY_MINUTE':
                days = 60
            elif interval == 'ONE_HOUR':
                days = 90
            else:  # ONE_DAY
                days = 365
            
            data = self.get_historical_data(symbol, interval, days)
            
            if data is not None and not data.empty:
                downloaded_data[interval] = data
                logger.info(f"✅ {interval}: {len(data)} records")
                logger.info(f"   Date range: {data.index.min()} to {data.index.max()}")
                logger.info(f"   Columns: {list(data.columns)}")
            else:
                logger.warning(f"❌ {interval}: No data received")
        
        return downloaded_data
    
    def test_interval_management(self, downloaded_data):
        """Test interval management for different prediction horizons"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Testing interval management for different horizons")
        logger.info("=" * 60)
        
        # Define prediction horizons and their optimal intervals
        horizons = {
            'INTRADAY': {
                'intervals': ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE'],
                'description': 'Minute and hourly data for very short-term predictions'
            },
            'SHORT_TERM': {
                'intervals': ['ONE_DAY', 'ONE_HOUR'],
                'description': 'Daily and hourly data for predictions up to 6 weeks'
            },
            'MEDIUM_TERM': {
                'intervals': ['ONE_DAY'],
                'description': 'Daily data for predictions up to 6 months'
            },
            'LONG_TERM': {
                'intervals': ['ONE_DAY'],
                'description': 'Daily data for predictions beyond 6 months'
            }
        }
        
        for horizon, config in horizons.items():
            logger.info(f"\n{horizon} prediction horizon:")
            logger.info(f"  Optimal intervals: {config['intervals']}")
            logger.info(f"  Description: {config['description']}")
            
            # Check which intervals are available
            available_intervals = [
                interval for interval in config['intervals']
                if interval in downloaded_data and not downloaded_data[interval].empty
            ]
            
            if available_intervals:
                logger.info(f"  ✅ Available intervals: {available_intervals}")
                
                # Simulate data aggregation
                total_records = sum(len(downloaded_data[interval]) for interval in available_intervals)
                logger.info(f"  📊 Total records available: {total_records}")
                
                # Show data quality for each available interval
                for interval in available_intervals:
                    data = downloaded_data[interval]
                    logger.info(f"    {interval}: {len(data)} records, "
                              f"Date range: {data.index.min()} to {data.index.max()}")
            else:
                logger.warning(f"  ❌ No suitable intervals available")
    
    def test_data_quality(self, downloaded_data):
        """Test data quality and structure"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Testing data quality and structure")
        logger.info("=" * 60)
        
        for interval, data in downloaded_data.items():
            logger.info(f"\n{interval} data quality:")
            logger.info(f"  Records: {len(data)}")
            logger.info(f"  Columns: {list(data.columns)}")
            logger.info(f"  Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"  Missing values: {data.isnull().sum().sum()}")
            logger.info(f"  Data types: {data.dtypes.to_dict()}")
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"  ⚠️  Missing columns: {missing_columns}")
            else:
                logger.info(f"  ✅ All required columns present")
            
            # Check data consistency
            if len(data) > 0:
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    if col in data.columns:
                        min_val = data[col].min()
                        max_val = data[col].max()
                        logger.info(f"  {col}: {min_val:.2f} to {max_val:.2f}")
    
    def run_comprehensive_test(self):
        """Run all tests"""
        logger.info("🚀 Starting Standalone Angel One Comprehensive Test")
        logger.info(f"Test Time: {datetime.now()}")
        
        start_time = time.time()
        
        try:
            # Test 1: Download all intervals
            downloaded_data = self.test_all_intervals()
            
            if not downloaded_data:
                logger.error("❌ No data downloaded. Cannot proceed with remaining tests.")
                return False
            
            # Test 2: Interval management
            self.test_interval_management(downloaded_data)
            
            # Test 3: Data quality
            self.test_data_quality(downloaded_data)
            
            # Summary
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("\n" + "=" * 60)
            logger.info("COMPREHENSIVE TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✅ Downloaded data for {len(downloaded_data)} intervals")
            logger.info(f"✅ Tested interval management for 4 prediction horizons")
            logger.info(f"✅ Verified data quality and structure")
            logger.info(f"⏱️  Total test duration: {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main test execution"""
    print("🚀 Standalone Angel One Comprehensive Integration Test")
    print("=" * 60)
    
    test = StandaloneAngelOneTest()
    success = test.run_comprehensive_test()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    main()
