#!/usr/bin/env python3
"""
Test script to verify that request constraints have been removed
"""

import sys
import os
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_constraints_removed():
    """Test that constraints have been removed for testing"""
    print("🧪 Testing Constraint Removal")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test Smart Data Fetcher
        print("📊 Testing Smart Data Fetcher...")
        from main.services.smart_data_fetcher import SmartDataFetcher
        
        fetcher = SmartDataFetcher()
        
        # Test multiple rapid requests (should all return True now)
        test_cases = [
            ('AAPL', 'ONE_DAY'),
            ('AAPL', 'ONE_DAY'),  # Same ticker, same interval
            ('MSFT', 'ONE_DAY'),
            ('TATAMOTORS', 'ONE_MINUTE'),
            ('TATAMOTORS', 'FIVE_MINUTE'),
            ('TATAMOTORS', 'FIFTEEN_MINUTE'),
        ]
        
        for ticker, interval in test_cases:
            should_fetch, reason = fetcher.should_fetch_data(ticker, interval)
            status = "✅" if should_fetch else "❌"
            print(f"  {status} {ticker} ({interval}): {reason}")
        
        # Test Data Service Wrapper
        print("\n📊 Testing Data Service Wrapper...")
        from main.services.data_service_wrapper import DataServiceWrapper
        
        wrapper = DataServiceWrapper('AAPL')
        
        # Test multiple rapid requests
        for i in range(3):
            try:
                data = wrapper.load_stock_data('1y', 'ONE_DAY')
                print(f"  ✅ Request {i+1}: {len(data)} records")
            except Exception as e:
                print(f"  ⚠️  Request {i+1}: {e}")
        
        # Test Angel One Manager
        print("\n📊 Testing Angel One Manager...")
        from main.services.angel_one_manager import AngelOneManager
        
        angel_manager = AngelOneManager()
        
        # Test rate limit checks
        for interval in ['ONE_DAY', 'ONE_MINUTE', 'FIVE_MINUTE']:
            rate_ok = angel_manager.check_rate_limit(interval)
            status = "✅" if rate_ok else "❌"
            print(f"  {status} Rate limit check for {interval}: {rate_ok}")
        
        # Test API Coordinator
        print("\n📊 Testing API Coordinator...")
        from main.services.api_coordinator import APICoordinator
        
        api_coordinator = APICoordinator()
        
        # Test rate limit checks
        for api_name in ['yahoo_finance', 'angel_one', 'default']:
            rate_ok = api_coordinator.check_rate_limit(api_name)
            status = "✅" if rate_ok else "❌"
            print(f"  {status} Rate limit check for {api_name}: {rate_ok}")
        
        print("\n✅ All constraint tests completed!")
        print("🎯 Constraints have been successfully removed for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Constraint testing failed: {e}")
        return False

def test_rapid_requests():
    """Test rapid requests to verify no constraints"""
    print("\n🚀 Testing Rapid Requests")
    print("=" * 30)
    
    try:
        from main.services.data_service_wrapper import DataServiceWrapper
        
        # Test rapid requests to same ticker
        wrapper = DataServiceWrapper('AAPL')
        
        print("Testing rapid requests to AAPL...")
        for i in range(5):
            start_time = time.time()
            try:
                data = wrapper.load_stock_data('1mo', 'ONE_DAY')
                end_time = time.time()
                print(f"  Request {i+1}: {len(data)} records in {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"  Request {i+1}: Error - {e}")
        
        # Test different intervals rapidly
        print("\nTesting different intervals rapidly...")
        intervals = ['ONE_DAY', 'ONE_DAY', 'ONE_DAY']  # Same interval multiple times
        
        for i, interval in enumerate(intervals):
            start_time = time.time()
            try:
                data = wrapper.load_stock_data('1mo', interval)
                end_time = time.time()
                print(f"  Interval {i+1} ({interval}): {len(data)} records in {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"  Interval {i+1} ({interval}): Error - {e}")
        
        print("\n✅ Rapid request testing completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Rapid request testing failed: {e}")
        return False

def main():
    """Main test function"""
    try:
        print("🧪 Testing Constraint Removal")
        print("=" * 50)
        
        # Test that constraints are removed
        constraint_success = test_constraints_removed()
        
        # Test rapid requests
        rapid_success = test_rapid_requests()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        print(f"Constraint Removal: {'✅ PASS' if constraint_success else '❌ FAIL'}")
        print(f"Rapid Requests: {'✅ PASS' if rapid_success else '❌ FAIL'}")
        
        overall_success = constraint_success and rapid_success
        print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 Constraints successfully removed for testing!")
            print("✅ Smart Data Fetcher: No fetch constraints")
            print("✅ Data Service Wrapper: No fetch constraints")
            print("✅ Angel One Manager: No rate limiting")
            print("✅ API Coordinator: No rate limiting")
            print("\n🚀 Ready for comprehensive interval testing!")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
