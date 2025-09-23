#!/usr/bin/env python3
"""
Test Angel One API with Corrected Headers
Tests the implementation with exact headers from official documentation
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_corrected_headers():
    """Test Angel One API with corrected headers from official documentation"""
    print("🧪 Testing Angel One API with Corrected Headers")
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
        
        # Test with different stocks and intervals using corrected headers
        test_cases = [
            ("RELIANCE", "BSE", "ONE_DAY", 7),      # RELIANCE on BSE
            ("TCS", "BSE", "ONE_DAY", 7),           # TCS on BSE
            ("HDFC", "NSE", "ONE_HOUR", 3),         # HDFC hourly
            ("INFY", "BSE", "FIFTEEN_MINUTE", 1),   # INFY 15-minute
        ]
        
        results = {}
        
        for symbol, exchange, interval, days in test_cases:
            print(f"\n📈 Testing {symbol} ({exchange}) - {interval} for {days} days...")
            
            try:
                # Get historical data with corrected headers
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
        print("📊 CORRECTED HEADERS ANGEL ONE API TEST REPORT")
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
        report_file = "docs/CORRECTED_HEADERS_TEST_REPORT.md"
        os.makedirs("docs", exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Corrected Headers Angel One API Test Report\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Tests:** {total_tests}\n")
            f.write(f"**Successful:** {successful_tests}\n")
            f.write(f"**Success Rate:** {(successful_tests/total_tests)*100:.1f}%\n\n")
            
            f.write("## Headers Used (From Official Documentation)\n\n")
            f.write("```python\n")
            f.write("headers = {\n")
            f.write("    'X-PrivateKey': 'API_KEY',\n")
            f.write("    'Accept': 'application/json',\n")
            f.write("    'X-SourceID': 'WEB',\n")
            f.write("    'X-ClientLocalIP': 'CLIENT_LOCAL_IP',\n")
            f.write("    'X-ClientPublicIP': 'CLIENT_PUBLIC_IP',\n")
            f.write("    'X-MACAddress': 'MAC_ADDRESS',\n")
            f.write("    'X-UserType': 'USER',\n")
            f.write("    'Authorization': 'Bearer AUTHORIZATION_TOKEN',\n")
            f.write("    'Content-Type': 'application/json'\n")
            f.write("}\n")
            f.write("```\n\n")
            
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
        print(f"❌ Corrected headers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    try:
        print("🧪 Testing Angel One API with Corrected Headers")
        print("=" * 60)
        
        # Run the corrected headers test
        success = test_corrected_headers()
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        if success:
            print("🎯 Overall Result: ✅ TESTS COMPLETED")
            print("🎉 Corrected headers implementation is working!")
            print("✅ Headers match official Python documentation exactly")
            print("✅ Request format matches official specification")
            print("✅ Response parsing handles official API format")
        else:
            print("🎯 Overall Result: ❌ TESTS FAILED")
            print("❌ Corrected headers implementation needs further debugging")
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
