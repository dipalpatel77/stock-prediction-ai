#!/usr/bin/env python3
"""
Test Incremental Data Efficiency
Demonstrates the efficiency gains from incremental data updates
"""

import time
import os
import pandas as pd
from datetime import datetime
from core.incremental_data_service import IncrementalDataService
from core.data_service import DataService

def test_incremental_efficiency():
    """Test and compare incremental vs full data downloads."""
    
    print("🚀 TESTING INCREMENTAL DATA EFFICIENCY")
    print("=" * 60)
    
    # Test tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "TCS.NS", "RELIANCE.NS"]
    
    # Initialize services
    incremental_service = IncrementalDataService()
    data_service = DataService()
    
    results = {}
    
    for ticker in test_tickers:
        print(f"\n📊 Testing {ticker}")
        print("-" * 40)
        
        # Test 1: First run (full download)
        print("🔄 First run (full download)...")
        start_time = time.time()
        
        try:
            data1 = incremental_service.get_incremental_data(ticker, period="1y")
            first_run_time = time.time() - start_time
            first_run_records = len(data1)
            
            print(f"✅ First run: {first_run_time:.2f}s, {first_run_records} records")
            
        except Exception as e:
            print(f"❌ First run failed: {e}")
            continue
        
        # Test 2: Second run (should use existing data)
        print("🔄 Second run (should use existing data)...")
        start_time = time.time()
        
        try:
            data2 = incremental_service.get_incremental_data(ticker, period="1y")
            second_run_time = time.time() - start_time
            second_run_records = len(data2)
            
            print(f"✅ Second run: {second_run_time:.2f}s, {second_run_records} records")
            
        except Exception as e:
            print(f"❌ Second run failed: {e}")
            continue
        
        # Test 3: Force refresh (full download)
        print("🔄 Force refresh (full download)...")
        start_time = time.time()
        
        try:
            data3 = incremental_service.get_incremental_data(ticker, period="1y", force_refresh=True)
            force_refresh_time = time.time() - start_time
            force_refresh_records = len(data3)
            
            print(f"✅ Force refresh: {force_refresh_time:.2f}s, {force_refresh_records} records")
            
        except Exception as e:
            print(f"❌ Force refresh failed: {e}")
            continue
        
        # Test 4: Traditional method (for comparison)
        print("🔄 Traditional method (for comparison)...")
        start_time = time.time()
        
        try:
            data4 = data_service.load_stock_data(ticker, period="1y")
            traditional_time = time.time() - start_time
            traditional_records = len(data4)
            
            print(f"✅ Traditional: {traditional_time:.2f}s, {traditional_records} records")
            
        except Exception as e:
            print(f"❌ Traditional method failed: {e}")
            continue
        
        # Calculate efficiency gains
        if second_run_time > 0:
            speedup = first_run_time / second_run_time
            efficiency_gain = ((first_run_time - second_run_time) / first_run_time) * 100
            
            results[ticker] = {
                'first_run_time': first_run_time,
                'second_run_time': second_run_time,
                'force_refresh_time': force_refresh_time,
                'traditional_time': traditional_time,
                'speedup': speedup,
                'efficiency_gain': efficiency_gain,
                'records': first_run_records
            }
            
            print(f"📈 Efficiency Analysis:")
            print(f"   Speedup: {speedup:.1f}x faster")
            print(f"   Time saved: {efficiency_gain:.1f}%")
            print(f"   Records: {first_run_records}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EFFICIENCY SUMMARY")
    print("=" * 60)
    
    if results:
        total_speedup = sum(r['speedup'] for r in results.values()) / len(results)
        total_efficiency = sum(r['efficiency_gain'] for r in results.values()) / len(results)
        
        print(f"Average Speedup: {total_speedup:.1f}x faster")
        print(f"Average Time Saved: {total_efficiency:.1f}%")
        print(f"Tickers Tested: {len(results)}")
        
        print(f"\n📋 Detailed Results:")
        for ticker, result in results.items():
            print(f"  {ticker}: {result['speedup']:.1f}x speedup, {result['efficiency_gain']:.1f}% time saved")
    
    return results

def test_data_info():
    """Test data information functionality."""
    print("\n🔍 TESTING DATA INFORMATION")
    print("=" * 60)
    
    service = IncrementalDataService()
    test_tickers = ["AAPL", "MSFT", "TCS.NS"]
    
    for ticker in test_tickers:
        print(f"\n📊 Data Info for {ticker}:")
        info = service.get_data_info(ticker)
        
        if info['exists']:
            print(f"  ✅ Data exists: {info['records']} records")
            print(f"  📅 First date: {info['first_date']}")
            print(f"  📅 Last date: {info['last_date']}")
            print(f"  ⏰ Days old: {info['days_old']}")
            print(f"  🔄 Needs update: {info['needs_update']}")
        else:
            print(f"  ❌ No data found")

def test_cleanup():
    """Test data cleanup functionality."""
    print("\n🧹 TESTING DATA CLEANUP")
    print("=" * 60)
    
    service = IncrementalDataService()
    
    # Clean up files older than 1 day (for testing)
    cleaned_count = service.cleanup_old_data(days_old=1)
    print(f"Cleaned up {cleaned_count} old data files")

if __name__ == "__main__":
    try:
        # Run efficiency tests
        results = test_incremental_efficiency()
        
        # Test data info
        test_data_info()
        
        # Test cleanup
        test_cleanup()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
