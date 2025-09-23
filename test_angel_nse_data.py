#!/usr/bin/env python3
"""
Test Angel One NSE Data Fetching
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_nse_data_fetching():
    """Test Angel One NSE data fetching"""
    print("📊 Testing Angel One NSE Data Fetching")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        
        # Initialize downloader
        print("📊 Initializing Angel One Data Downloader...")
        downloader = AngelOneDataDownloader()
        
        # Authenticate
        print("🔐 Authenticating...")
        auth_success = downloader.authenticate()
        
        if not auth_success:
            print("❌ Authentication failed!")
            return False
        
        print("✅ Authentication successful!")
        
        # Test NSE stocks with proper date formatting
        test_cases = [
            ('HDFC', 'NSE', 'ONE_DAY', 7),      # 7 days of daily data
            ('ICICIBANK', 'NSE', 'ONE_DAY', 7), # 7 days of daily data
            ('SBIN', 'NSE', 'ONE_DAY', 7),      # 7 days of daily data
        ]
        
        for ticker, exchange, interval, days in test_cases:
            print(f"\n📈 Testing {ticker} ({interval}) for {days} days...")
            try:
                # Calculate date range with proper formatting
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Format dates as required by API: "yyyy-MM-dd hh:mm"
                from_date = start_date.strftime('%Y-%m-%d %H:%M')
                to_date = end_date.strftime('%Y-%m-%d %H:%M')
                
                print(f"   Period: {from_date} to {to_date}")
                
                # Get historical data
                data = downloader.get_historical_data(
                    symbol_name=ticker,
                    exchange=exchange,
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                    days_back=days
                )
                
                if data is not None and not data.empty:
                    print(f"   ✅ Success: {len(data)} records")
                    print(f"   📊 Columns: {list(data.columns)}")
                    print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
                    
                    # Show sample data
                    print(f"   📈 Sample data:")
                    print(f"      {data.head(2).to_string()}")
                else:
                    print(f"   ⚠️  No data received")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ NSE data fetching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_historical_data_with_debug():
    """Test historical data with detailed debugging"""
    print("\n🔍 Testing Historical Data with Debug Info")
    print("=" * 45)
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        import requests
        
        downloader = AngelOneDataDownloader()
        
        # Authenticate
        if not downloader.authenticate():
            print("❌ Authentication failed!")
            return False
        
        # Test with HDFC (NSE stock)
        ticker = 'HDFC'
        exchange = 'NSE'
        interval = 'ONE_DAY'
        
        # Get symbol info
        symbol_info = downloader.get_symbol_info(ticker, exchange)
        if not symbol_info:
            print(f"❌ Symbol info not found for {ticker}")
            return False
        
        print(f"✅ Symbol info: {symbol_info}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
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
        
        print(f"📤 Request payload: {candle_payload}")
        
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
        
        print(f"📤 Headers: {headers}")
        
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
            print(f"📊 Response data: {data}")
            
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

def main():
    """Main test function"""
    try:
        print("🧪 Testing Angel One NSE Data Fetching")
        print("=" * 50)
        
        # Test NSE data fetching
        nse_success = test_nse_data_fetching()
        
        # Test with debug info
        debug_success = test_historical_data_with_debug()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        print(f"NSE Data Fetching: {'✅ PASS' if nse_success else '❌ FAIL'}")
        print(f"Debug Test: {'✅ PASS' if debug_success else '❌ FAIL'}")
        
        overall_success = nse_success and debug_success
        print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 Angel One NSE data fetching is working perfectly!")
            print("✅ Authentication successful")
            print("✅ NSE data fetching successful")
            print("✅ Debug information available")
            print("\n🚀 Ready for comprehensive interval testing!")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
