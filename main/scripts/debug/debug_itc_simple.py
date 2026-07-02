#!/usr/bin/env python3
"""
Simple debug script to test ITC token directly with Angel One API
"""

import requests
import json
import pyotp
from datetime import datetime, timedelta

class ITCDebugger:
    def __init__(self):
        self.base_url = "https://apiconnect.angelone.in"
        self.api_key = "1TKgQThc"
        self.client_code = "D54448"
        self.client_pin = "2251"
        self.totp_secret = "NP4SAXOKMTJQZ4KZP2TBTYXRCE"
        self.jwt_token = None
        
    def generate_totp(self):
        """Generate TOTP code"""
        try:
            totp = pyotp.TOTP(self.totp_secret)
            return totp.now()
        except Exception as e:
            print(f"TOTP generation failed: {e}")
            return "123456"
    
    def authenticate(self):
        """Authenticate with Angel One API"""
        try:
            login_url = f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
            totp_code = self.generate_totp()
            
            login_payload = {
                "clientcode": self.client_code,
                "password": self.client_pin,
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
                'X-PrivateKey': self.api_key
            }
            
            print(f"🔐 Authenticating with TOTP: {totp_code}")
            response = requests.post(login_url, headers=login_headers, json=login_payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data'):
                    self.jwt_token = data['data'].get('jwtToken')
                    print(f"✅ Authentication successful")
                    return True
                else:
                    print(f"❌ Authentication failed: {data}")
                    return False
            else:
                print(f"❌ Authentication failed with status {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def test_itc_token(self, token, exchange):
        """Test ITC token on specific exchange"""
        try:
            if not self.jwt_token:
                print("❌ No JWT token available")
                return False
                
            # Prepare date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            from_date = start_date.strftime('%Y-%m-%d 09:15')
            to_date = end_date.strftime('%Y-%m-%d 15:30')
            
            # Make API request
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            
            payload = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": "ONE_DAY",
                "fromdate": from_date,
                "todate": to_date
            }
            
            headers = {
                'Authorization': f'Bearer {self.jwt_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserType': 'USER',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '192.168.1.1',
                'X-ClientPublicIP': '192.168.1.1',
                'X-MACAddress': '00:00:00:00:00:00',
                'X-PrivateKey': self.api_key,
                'X-ClientCode': self.client_code
            }
            
            print(f"🔍 Testing ITC token {token} on {exchange} exchange...")
            print(f"   Date range: {from_date} to {to_date}")
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data'):
                    candle_data = data['data']
                    if len(candle_data) > 0:
                        # Extract latest price
                        latest_candle = candle_data[-1]
                        latest_price = latest_candle[4]  # Close price
                        print(f"✅ SUCCESS: Found {len(candle_data)} records")
                        print(f"   Latest price: ₹{latest_price:.2f}")
                        
                        # Check if price is reasonable for ITC
                        if 300 <= latest_price <= 600:
                            print(f"✅ PRICE LOOKS CORRECT: ₹{latest_price:.2f} is reasonable for ITC")
                            return True
                        else:
                            print(f"⚠️ PRICE SUSPICIOUS: ₹{latest_price:.2f} doesn't look like ITC price")
                            return False
                    else:
                        print(f"❌ No data in response")
                        return False
                else:
                    print(f"❌ API error: {data}")
                    return False
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False
                
        except Exception as e:
            print(f"❌ Error testing {token} on {exchange}: {e}")
            return False
    
    def find_correct_itc_config(self):
        """Find the correct ITC configuration"""
        print("🔍 Searching for correct ITC token and exchange...")
        
        # Authenticate first
        if not self.authenticate():
            print("❌ Authentication failed, cannot proceed")
            return
        
        # Test different token and exchange combinations
        potential_tokens = [
            '4241',  # Current token
            '4240',  # Common variations
            '4242', '4243', '4244', '4245',
            '4246', '4247', '4248', '4249', '4250'
        ]
        
        exchanges = ['NSE', 'BSE', 'NFO', 'CDS', 'MCX']
        
        successful_combinations = []
        
        for token in potential_tokens:
            for exchange in exchanges:
                try:
                    if self.test_itc_token(token, exchange):
                        successful_combinations.append((token, exchange))
                        print(f"🎉 FOUND WORKING COMBINATION: Token {token} on {exchange}")
                except Exception as e:
                    print(f"Error testing {token} on {exchange}: {e}")
                    continue
        
        if successful_combinations:
            print(f"\n✅ Found {len(successful_combinations)} working combinations:")
            for token, exchange in successful_combinations:
                print(f"   Token: {token}, Exchange: {exchange}")
        else:
            print("\n❌ No working combinations found")
            
        return successful_combinations

def main():
    debugger = ITCDebugger()
    combinations = debugger.find_correct_itc_config()
    
    if combinations:
        print(f"\n🎯 RECOMMENDED ITC CONFIGURATION:")
        print(f"   Symbol Token: {combinations[0][0]}")
        print(f"   Exchange: {combinations[0][1]}")
        print(f"\n📝 Update angel_one_service.py with:")
        print(f"   'ITC': '{combinations[0][0]}',")
        print(f"   And set exchange to '{combinations[0][1]}' in get_historical_data method")
    else:
        print("\n❌ No working ITC configuration found")

if __name__ == "__main__":
    main()
