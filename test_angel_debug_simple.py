#!/usr/bin/env python3
"""
Simple Angel One API Debug Test
"""

import sys
import os
import time
from datetime import datetime, timedelta
import requests
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_simple_angel_request():
    """Test a simple Angel One API request with detailed debugging"""
    print("🔍 Simple Angel One API Debug Test")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        
        # Initialize downloader
        downloader = AngelOneDataDownloader()
        
        # Authenticate
        print("🔐 Authenticating...")
        auth_success = downloader.authenticate()
        
        if not auth_success:
            print("❌ Authentication failed!")
            return False
        
        print("✅ Authentication successful!")
        
        # Test with a simple request - 1 day of daily data
        print("\n📊 Testing simple request (1 day, daily data)...")
        
        # Use a simple stock and short period
        ticker = 'HDFC'
        exchange = 'NSE'
        interval = 'ONE_DAY'
        
        # Get symbol info
        symbol_info = downloader.get_symbol_info(ticker, exchange)
        if not symbol_info:
            print(f"❌ Symbol info not found for {ticker}")
            return False
        
        print(f"✅ Symbol info: {symbol_info}")
        
        # Calculate date range - just 1 day
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        from_date = start_date.strftime('%Y-%m-%d %H:%M')
        to_date = end_date.strftime('%Y-%m-%d %H:%M')
        
        print(f"📅 Date range: {from_date} to {to_date}")
        
        # Prepare request payload
        candle_payload = {
            "exchange": exchange,
            "symboltoken": symbol_info['token'],
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }
        
        print(f"📤 Request payload: {json.dumps(candle_payload, indent=2)}")
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB, WEB",
            "X-ClientLocalIP": "127.0.0.1",
            "X-ClientPublicIP": "127.0.0.1",
            "X-MACAddress": "XX:XX:XX:XX:XX:XX",
            "X-PrivateKey": downloader.api_key,
            "Authorization": f"Bearer {downloader.jwt_token}"
        }
        
        print(f"📤 Headers: {json.dumps(headers, indent=2)}")
        
        # Make request
        historical_url = "https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData"
        
        print(f"🌐 Making request to: {historical_url}")
        
        response = requests.post(
            historical_url,
            json=candle_payload,
            headers=headers,
            timeout=30
        )
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response data: {json.dumps(data, indent=2)}")
            
            if data.get("status") == True:
                candles = data.get("data", [])
                print(f"✅ Success: {len(candles)} records")
                if candles:
                    print(f"📈 Sample candle: {candles[0]}")
                return True
            else:
                error_code = data.get("errorCode", "Unknown")
                error_msg = data.get("message", "Unknown error")
                print(f"❌ API Error: {error_msg} (Code: {error_code})")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_date_formats():
    """Test different date formats"""
    print("\n📅 Testing Different Date Formats")
    print("=" * 35)
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        
        downloader = AngelOneDataDownloader()
        
        if not downloader.authenticate():
            print("❌ Authentication failed!")
            return False
        
        # Test different date formats
        date_formats = [
            "2025-09-19 09:15",  # Market hours
            "2025-09-19 15:30",  # Market close
            "2025-09-18 09:15",  # Previous day
            "2025-09-18 15:30",  # Previous day close
        ]
        
        ticker = 'HDFC'
        exchange = 'NSE'
        interval = 'ONE_DAY'
        
        symbol_info = downloader.get_symbol_info(ticker, exchange)
        if not symbol_info:
            print(f"❌ Symbol info not found for {ticker}")
            return False
        
        for date_str in date_formats:
            print(f"\n📅 Testing date format: {date_str}")
            
            candle_payload = {
                "exchange": exchange,
                "symboltoken": symbol_info['token'],
                "interval": interval,
                "fromdate": date_str,
                "todate": date_str
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, application/json",
                "X-UserType": "USER",
                "X-SourceID": "WEB, WEB",
                "X-ClientLocalIP": "127.0.0.1",
                "X-ClientPublicIP": "127.0.0.1",
                "X-MACAddress": "XX:XX:XX:XX:XX:XX",
                "X-PrivateKey": downloader.api_key,
                "Authorization": f"Bearer {downloader.jwt_token}"
            }
            
            try:
                response = requests.post(
                    "https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData",
                    json=candle_payload,
                    headers=headers,
                    timeout=30
                )
                
                print(f"   📊 Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == True:
                        candles = data.get("data", [])
                        print(f"   ✅ Success: {len(candles)} records")
                    else:
                        error_msg = data.get("message", "Unknown error")
                        print(f"   ❌ API Error: {error_msg}")
                else:
                    print(f"   ❌ HTTP Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Request failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Date format test failed: {e}")
        return False

def main():
    """Main test function"""
    try:
        print("🧪 Angel One Simple Debug Test")
        print("=" * 40)
        
        # Test simple request
        simple_success = test_simple_angel_request()
        
        # Test different date formats
        date_success = test_different_date_formats()
        
        # Summary
        print("\n" + "=" * 40)
        print("📊 TEST SUMMARY")
        print("=" * 40)
        
        print(f"Simple Request: {'✅ PASS' if simple_success else '❌ FAIL'}")
        print(f"Date Formats: {'✅ PASS' if date_success else '❌ FAIL'}")
        
        overall_success = simple_success or date_success
        print(f"\n🎯 Overall Result: {'✅ DEBUGGING SUCCESSFUL' if overall_success else '❌ DEBUGGING FAILED'}")
        
        if overall_success:
            print("\n🎉 Angel One API debugging completed!")
            print("✅ Issues identified and documented")
            print("✅ Next steps clear")
        else:
            print("\n❌ Angel One API debugging failed!")
            print("❌ Need to investigate further")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
