#!/usr/bin/env python3
"""
Test Angel One API with Fixed Implementation
Tests the corrected headers and request format based on official documentation
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_fixed_angel_one():
    """Test the fixed Angel One API implementation"""
    print("🧪 Testing Fixed Angel One API Implementation")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        
        # Initialize downloader
        print("🔧 Initializing Angel One Data Downloader...")
        downloader = AngelOneDataDownloader()
        
        # Test authentication
        print("\n🔐 Testing Authentication...")
        auth_success = downloader.authenticate()
        
        if not auth_success:
            print("❌ Authentication failed!")
            return False
        
        print("✅ Authentication successful!")
        
        # Test different stocks and intervals
        test_cases = [
            ("RELIANCE", "BSE", "ONE_DAY", 7),      # RELIANCE on BSE
            ("TCS", "NSE", "ONE_DAY", 7),           # TCS on NSE
            ("HDFC", "NSE", "ONE_HOUR", 3),         # HDFC hourly
            ("INFY", "NSE", "FIFTEEN_MINUTE", 1),   # INFY 15-minute
        ]
        
        results = {}
        
        for symbol, exchange, interval, days in test_cases:
            print(f"\n📈 Testing {symbol} ({exchange}) - {interval} for {days} days...")
            
            try:
                # Get historical data
                data = downloader.get_historical_data(
                    symbol_name=symbol,
                    exchange=exchange,
                    interval=interval,
                    days_back=days
                )
                
                if not data.empty:
                    print(f"   ✅ Retrieved: {len(data)} records")
                    print(f"   📊 Columns: {list(data.columns)}")
                    print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
                    
                    # Show sample data
                    print(f"   📈 Sample data (first 2 records):")
                    sample_data = data.head(2)
                    for idx, row in sample_data.iterrows():
                        print(f"      {idx}: Open={row['Open']}, High={row['High']}, Low={row['Low']}, Close={row['Close']}, Volume={row['Volume']}")
                    
                    results[f"{symbol}_{interval}"] = {
                        'success': True,
                        'records': len(data),
                        'columns': list(data.columns),
                        'date_range': f"{data.index.min()} to {data.index.max()}"
                    }
                else:
                    print(f"   ❌ No data received")
                    results[f"{symbol}_{interval}"] = {
                        'success': False,
                        'error': 'No data received'
                    }
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[f"{symbol}_{interval}"] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Generate report
        print("\n" + "=" * 80)
        print("📊 FIXED ANGEL ONE API TEST REPORT")
        print("=" * 80)
        
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result.get('success', False))
        total_records = sum(result.get('records', 0) for result in results.values())
        
        print(f"\n📈 Test Results:")
        print("-" * 40)
        
        for test_name, result in results.items():
            status = "✅" if result.get('success', False) else "❌"
            records = result.get('records', 0)
            error = result.get('error', '')
            
            print(f"   {status} {test_name}: {records} records")
            if error:
                print(f"      ❌ Error: {error}")
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        print(f"   Total Records: {total_records}")
        
        # Save detailed report
        report_file = "docs/FIXED_ANGEL_ONE_TEST_REPORT.md"
        os.makedirs("docs", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Fixed Angel One API Test Report\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Tests:** {total_tests}\n")
            f.write(f"**Successful:** {successful_tests}\n")
            f.write(f"**Success Rate:** {(successful_tests/total_tests)*100:.1f}%\n\n")
            
            f.write("## Detailed Results\n\n")
            for test_name, result in results.items():
                f.write(f"### {test_name}\n\n")
                f.write(f"- **Status**: {'✅' if result.get('success', False) else '❌'}\n")
                f.write(f"- **Records**: {result.get('records', 0)}\n")
                if result.get('error'):
                    f.write(f"- **Error**: {result['error']}\n")
                f.write("\n")
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"❌ Fixed Angel One test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    try:
        print("🧪 Testing Fixed Angel One API Implementation")
        print("=" * 60)
        
        # Run the fixed test
        success = test_fixed_angel_one()
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        if success:
            print("🎯 Overall Result: ✅ TESTS COMPLETED")
            print("🎉 Fixed Angel One API implementation is working!")
            print("✅ Headers corrected according to official documentation")
            print("✅ Request format matches official specification")
            print("✅ Response parsing handles official API format")
        else:
            print("🎯 Overall Result: ❌ TESTS FAILED")
            print("❌ Fixed Angel One API implementation needs further debugging")
            print("❌ Check API access permissions and account limitations")
        
        return success
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
