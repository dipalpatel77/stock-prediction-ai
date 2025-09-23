#!/usr/bin/env python3
"""
Test Angel One Data Fetching After Authentication
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_angel_data_fetching():
    """Test Angel One data fetching after authentication"""
    print("📊 Testing Angel One Data Fetching")
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
        
        # Test data fetching for different stocks
        test_cases = [
            ('RELIANCE', 'NSE', 'ONE_DAY', 7),      # 7 days of daily data
            ('TATAMOTORS', 'NSE', 'ONE_HOUR', 3),   # 3 days of hourly data
            ('TCS', 'NSE', 'FIFTEEN_MINUTE', 1),   # 1 day of 15-min data
        ]
        
        for ticker, exchange, interval, days in test_cases:
            print(f"\n📈 Testing {ticker} ({interval}) for {days} days...")
            try:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
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
        print(f"❌ Data fetching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_symbol_lookup():
    """Test symbol lookup functionality"""
    print("\n🔍 Testing Symbol Lookup")
    print("=" * 30)
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        
        downloader = AngelOneDataDownloader()
        
        # Test symbol lookup
        test_symbols = ['RELIANCE', 'TATAMOTORS', 'TCS', 'INFY', 'HDFC']
        
        for symbol in test_symbols:
            print(f"\n🔍 Looking up {symbol}...")
            try:
                symbol_info = downloader.get_symbol_info(symbol, 'NSE')
                if symbol_info:
                    print(f"   ✅ Found: {symbol_info}")
                else:
                    print(f"   ❌ Not found")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Symbol lookup test failed: {e}")
        return False

def main():
    """Main test function"""
    try:
        print("🧪 Testing Angel One Data Fetching")
        print("=" * 50)
        
        # Test data fetching
        fetch_success = test_angel_data_fetching()
        
        # Test symbol lookup
        lookup_success = test_symbol_lookup()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        print(f"Data Fetching: {'✅ PASS' if fetch_success else '❌ FAIL'}")
        print(f"Symbol Lookup: {'✅ PASS' if lookup_success else '❌ FAIL'}")
        
        overall_success = fetch_success and lookup_success
        print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 Angel One data fetching is working perfectly!")
            print("✅ Authentication successful")
            print("✅ Data fetching successful")
            print("✅ Symbol lookup working")
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
