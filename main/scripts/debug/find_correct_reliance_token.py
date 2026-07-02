#!/usr/bin/env python3
"""
Find the correct RELIANCE symbol token for Angel One API
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

class RelianceTokenFinder:
    """Find the correct RELIANCE symbol token"""
    
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
        
        # Common RELIANCE token candidates
        self.reliance_tokens = [
            '2881',      # Original token
            '99926000',  # Current token (wrong)
            '2881',      # NSE token
            '500325',    # BSE token
            '2881',      # Alternative NSE
            '2881',      # RELIANCE NSE
        ]
        
        logger.info("RELIANCE Token Finder initialized")
    
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
    
    def test_token(self, token, exchange='NSE'):
        """Test a specific token for RELIANCE data"""
        if not self.is_authenticated:
            if not self.authenticate():
                return None
        
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            
            # Test with recent data (last 7 days)
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
            to_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            payload = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": 'ONE_DAY',
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
            
            logger.info(f"Testing token {token} on {exchange}...")
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data') and len(data['data']) > 0:
                    # Check if the price range looks like RELIANCE (around 1000-2000)
                    sample_data = data['data'][0]  # [datetime, open, high, low, close, volume]
                    close_price = sample_data[4]  # Close price
                    
                    logger.info(f"  ✅ Token {token}: Found {len(data['data'])} records")
                    logger.info(f"  📊 Sample data: {sample_data}")
                    logger.info(f"  💰 Close price: ₹{close_price}")
                    
                    # Check if price is in RELIANCE range (1000-2000)
                    if 1000 <= close_price <= 2000:
                        logger.info(f"  🎯 CORRECT RANGE! This looks like RELIANCE!")
                        return {
                            'token': token,
                            'exchange': exchange,
                            'price': close_price,
                            'data': data['data']
                        }
                    else:
                        logger.warning(f"  ⚠️  Price {close_price} is not in RELIANCE range (1000-2000)")
                        return {
                            'token': token,
                            'exchange': exchange,
                            'price': close_price,
                            'data': data['data'],
                            'note': 'Wrong price range'
                        }
                else:
                    logger.info(f"  ❌ Token {token}: No data returned")
                    return None
            else:
                logger.error(f"  ❌ Token {token}: Request failed - {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"  ❌ Token {token}: Error - {e}")
            return None
    
    def find_correct_token(self):
        """Find the correct RELIANCE token"""
        logger.info("🔍 Searching for correct RELIANCE token...")
        logger.info("Expected price range: ₹1,000 - ₹2,000")
        
        results = []
        
        # Test different tokens
        for token in self.reliance_tokens:
            result = self.test_token(token, 'NSE')
            if result:
                results.append(result)
        
        # Test BSE tokens
        for token in ['500325', '2881']:
            result = self.test_token(token, 'BSE')
            if result:
                results.append(result)
        
        # Analyze results
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS RESULTS")
        logger.info("=" * 60)
        
        correct_tokens = []
        for result in results:
            if result.get('note') != 'Wrong price range':
                if 1000 <= result['price'] <= 2000:
                    correct_tokens.append(result)
                    logger.info(f"✅ CORRECT: Token {result['token']} on {result['exchange']} - Price: ₹{result['price']}")
                else:
                    logger.warning(f"❌ WRONG: Token {result['token']} on {result['exchange']} - Price: ₹{result['price']} (not RELIANCE range)")
        
        if correct_tokens:
            logger.info(f"\n🎯 FOUND {len(correct_tokens)} CORRECT TOKEN(S):")
            for token_info in correct_tokens:
                logger.info(f"   Token: {token_info['token']}")
                logger.info(f"   Exchange: {token_info['exchange']}")
                logger.info(f"   Price: ₹{token_info['price']}")
        else:
            logger.warning("\n❌ No correct RELIANCE tokens found!")
            logger.info("All tested tokens returned prices outside RELIANCE range (₹1,000-₹2,000)")
        
        return correct_tokens

def main():
    """Main execution"""
    print("🔍 RELIANCE Token Finder")
    print("=" * 60)
    print("Expected RELIANCE price range: ₹1,000 - ₹2,000")
    print("Current system shows: ₹25,000+ (WRONG!)")
    print("=" * 60)
    
    finder = RelianceTokenFinder()
    correct_tokens = finder.find_correct_token()
    
    if correct_tokens:
        print(f"\n✅ Found {len(correct_tokens)} correct token(s)!")
        for token_info in correct_tokens:
            print(f"   Use token: {token_info['token']} on {token_info['exchange']}")
    else:
        print("\n❌ No correct tokens found. Need to investigate further.")

if __name__ == "__main__":
    main()
