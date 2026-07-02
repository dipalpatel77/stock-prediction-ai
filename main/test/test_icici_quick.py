#!/usr/bin/env python3
"""
Quick test for ICICI Bank data from Angel One API
"""

import requests
import json
import pyotp
from datetime import datetime, timedelta

def test_icici_bank():
    """Test ICICI Bank data from Angel One API"""
    
    # API configuration
    base_url = "https://apiconnect.angelone.in"
    api_key = "1TKgQThc"
    client_code = "D54448"
    client_pin = "2251"
    totp_secret = "NP4SAXOKMTJQZ4KZP2TBTYXRCE"
    
    try:
        # Step 1: Authenticate
        print("🔐 Authenticating with Angel One API...")
        totp = pyotp.TOTP(totp_secret)
        totp_code = totp.now()
        
        login_url = f"{base_url}/rest/auth/angelbroking/user/v1/loginByPassword"
        login_payload = {
            "clientcode": client_code,
            "password": client_pin,
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
            'X-PrivateKey': api_key
        }
        
        print(f"   TOTP Code: {totp_code}")
        login_response = requests.post(login_url, headers=login_headers, json=login_payload)
        
        if login_response.status_code != 200:
            print(f"❌ Authentication failed: {login_response.status_code}")
            return False
            
        login_data = login_response.json()
        if not login_data.get('status') or not login_data.get('data'):
            print(f"❌ Authentication failed: {login_data}")
            return False
            
        jwt_token = login_data['data'].get('jwtToken')
        print("✅ Authentication successful")
        
        # Step 2: Test ICICI Bank data
        print("\n🔍 Testing ICICI Bank data...")
        
        # Test different exchanges
        exchanges = ['NSE', 'BSE']
        icici_token = '4963'
        
        for exchange in exchanges:
            print(f"\n📊 Testing ICICI Bank (token: {icici_token}) on {exchange}...")
            
            # Prepare date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            from_date = start_date.strftime('%Y-%m-%d 09:15')
            to_date = end_date.strftime('%Y-%m-%d 15:30')
            
            # Make API request
            url = f"{base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            payload = {
                "exchange": exchange,
                "symboltoken": icici_token,
                "interval": "ONE_DAY",
                "fromdate": from_date,
                "todate": to_date
            }
            
            headers = {
                'Authorization': f'Bearer {jwt_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserType': 'USER',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '192.168.1.1',
                'X-ClientPublicIP': '192.168.1.1',
                'X-MACAddress': '00:00:00:00:00:00',
                'X-PrivateKey': api_key,
                'X-ClientCode': client_code
            }
            
            print(f"   Date range: {from_date} to {to_date}")
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') and data.get('data'):
                    candle_data = data['data']
                    if len(candle_data) > 0:
                        latest_candle = candle_data[-1]
                        latest_price = latest_candle[4]  # Close price
                        print(f"✅ SUCCESS: Found {len(candle_data)} records")
                        print(f"   Latest price: ₹{latest_price:.2f}")
                        
                        # Check if price is reasonable for ICICI Bank (should be around ₹800-1200)
                        if 600 <= latest_price <= 1500:
                            print(f"✅ PRICE LOOKS CORRECT: ₹{latest_price:.2f} is realistic for ICICI Bank")
                            return True, exchange, latest_price
                        else:
                            print(f"⚠️ PRICE SUSPICIOUS: ₹{latest_price:.2f} doesn't look like ICICI Bank price")
                    else:
                        print(f"❌ No data in response")
                else:
                    print(f"❌ API error: {data}")
            else:
                print(f"❌ Request failed with status {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
        
        print("\n❌ No working ICICI Bank configuration found")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    result = test_icici_bank()
    if result:
        print(f"\n🎯 ICICI Bank is available in Angel One API!")
    else:
        print(f"\n❌ ICICI Bank is not available in Angel One API - will use fallback data")
