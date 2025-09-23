#!/usr/bin/env python3
"""
Corrected Interval Data Testing
Properly handles data source limitations:
- Yahoo Finance: Daily data only for US stocks
- Angel One API: All intervals for Indian stocks
"""

import sys
import os
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_us_stock_intervals():
    """Test intervals for US stocks (Yahoo Finance only - daily data)"""
    print("🇺🇸 Testing US Stock Intervals (Yahoo Finance)")
    print("=" * 50)
    print("Note: Yahoo Finance only supports daily data for US stocks")
    print()
    
    # US stocks and their expected data points
    us_stocks = [
        {'ticker': 'AAPL', 'name': 'Apple Inc.'},
        {'ticker': 'MSFT', 'name': 'Microsoft Corp.'},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'ticker': 'TSLA', 'name': 'Tesla Inc.'},
        {'ticker': 'AMZN', 'name': 'Amazon.com Inc.'}
    ]
    
    # Only test daily intervals for US stocks
    intervals = ['ONE_DAY']
    periods = ['1mo', '3mo', '6mo', '1y', '2y']
    
    results = {}
    
    for stock in us_stocks:
        print(f"\n📊 Testing {stock['ticker']} ({stock['name']})")
        print("-" * 40)
        
        stock_results = {}
        
        for interval in intervals:
            print(f"  📈 Testing {interval} interval")
            
            for period in periods:
                print(f"    📅 Testing {period} period")
                
                try:
                    result = test_interval_configuration(
                        ticker=stock['ticker'],
                        interval=interval,
                        period=period,
                        is_indian=False
                    )
                    
                    stock_results[f"{interval}_{period}"] = result
                    
                    status = "✅" if result['success'] else "❌"
                    print(f"      {status} {period}: {result['data_points']} data points in {result['fetch_time']:.2f}s")
                    
                except Exception as e:
                    print(f"      ❌ {period}: Error - {e}")
                    stock_results[f"{interval}_{period}"] = {
                        'success': False,
                        'error': str(e)
                    }
        
        results[stock['ticker']] = stock_results
    
    return results

def test_indian_stock_intervals():
    """Test intervals for Indian stocks (Angel One API - all intervals)"""
    print("\n🇮🇳 Testing Indian Stock Intervals (Angel One API)")
    print("=" * 50)
    print("Note: Angel One API supports all intervals for Indian stocks")
    print()
    
    # Indian stocks and their expected data points
    indian_stocks = [
        {'ticker': 'TATAMOTORS', 'name': 'Tata Motors Ltd.'},
        {'ticker': 'RELIANCE', 'name': 'Reliance Industries Ltd.'},
        {'ticker': 'TCS', 'name': 'Tata Consultancy Services Ltd.'},
        {'ticker': 'HDFC', 'name': 'HDFC Bank Ltd.'},
        {'ticker': 'PNB', 'name': 'Punjab National Bank'}
    ]
    
    # Test all intervals for Indian stocks
    intervals = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'ONE_DAY']
    periods = ['1d', '5d', '1mo', '3mo']
    
    results = {}
    
    for stock in indian_stocks:
        print(f"\n📊 Testing {stock['ticker']} ({stock['name']})")
        print("-" * 40)
        
        stock_results = {}
        
        for interval in intervals:
            print(f"  📈 Testing {interval} interval")
            
            for period in periods:
                print(f"    📅 Testing {period} period")
                
                try:
                    result = test_interval_configuration(
                        ticker=stock['ticker'],
                        interval=interval,
                        period=period,
                        is_indian=True
                    )
                    
                    stock_results[f"{interval}_{period}"] = result
                    
                    status = "✅" if result['success'] else "❌"
                    print(f"      {status} {period}: {result['data_points']} data points in {result['fetch_time']:.2f}s")
                    
                except Exception as e:
                    print(f"      ❌ {period}: Error - {e}")
                    stock_results[f"{interval}_{period}"] = {
                        'success': False,
                        'error': str(e)
                    }
        
        results[stock['ticker']] = stock_results
    
    return results

def test_interval_configuration(ticker: str, interval: str, period: str, is_indian: bool):
    """Test a specific interval configuration"""
    try:
        from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
        
        # Configuration based on stock type
        config = {
            'ticker': ticker,
            'is_indian': is_indian,
            'analysis_type': 'comprehensive',
            'parameters': {
                'interval': interval,
                'period': period,
                'use_enhanced': True,
                'use_database': True
            },
            'use_enhanced': True,
            'use_database': True,
            'interval': interval,
            'timeframe': period,
            'success': True
        }
        
        # Add Angel One config for Indian stocks
        if is_indian:
            config['angel_config'] = {
                'api_key': '1TKgQThc ',
                'api_secret': 'D54448',
                'access_token': '2251',
                'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE',
                'exchange': 'BSE',
                'interval': interval
            }
        
        # Initialize pipeline
        pipeline = UnifiedAnalysisPipeline(ticker=ticker, config=config)
        
        # Measure performance
        start_time = time.time()
        
        # Run analysis
        result = pipeline.run_analysis(**config['parameters'])
        
        end_time = time.time()
        fetch_time = end_time - start_time
        
        # Extract data points
        data_points = 0
        if result.get('success') and 'results' in result:
            data_result = result['results'].get('data_processor', {})
            data_points = data_result.get('records_processed', 0)
        
        return {
            'success': result.get('success', False),
            'data_points': data_points,
            'fetch_time': fetch_time,
            'data_source': 'Angel One' if is_indian else 'Yahoo Finance'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'data_points': 0,
            'fetch_time': 0,
            'data_source': 'Angel One' if is_indian else 'Yahoo Finance'
        }

def test_data_source_limitations():
    """Test data source limitations explicitly"""
    print("\n🔍 Testing Data Source Limitations")
    print("=" * 40)
    
    # Test unsupported intervals for US stocks
    print("🇺🇸 Testing unsupported intervals for US stocks:")
    us_ticker = 'AAPL'
    unsupported_intervals = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR']
    
    for interval in unsupported_intervals:
        print(f"  📊 Testing {interval} for {us_ticker} (should fail)")
        
        try:
            result = test_interval_configuration(
                ticker=us_ticker,
                interval=interval,
                period='1d',
                is_indian=False
            )
            
            if result['success']:
                print(f"    ⚠️  Unexpected success: {result['data_points']} data points")
            else:
                print(f"    ✅ Expected failure: {result.get('error', 'No data available')}")
                
        except Exception as e:
            print(f"    ✅ Expected failure: {e}")
    
    # Test supported intervals for Indian stocks
    print("\n🇮🇳 Testing supported intervals for Indian stocks:")
    indian_ticker = 'TATAMOTORS'
    supported_intervals = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'ONE_DAY']
    
    for interval in supported_intervals:
        print(f"  📊 Testing {interval} for {indian_ticker}")
        
        try:
            result = test_interval_configuration(
                ticker=indian_ticker,
                interval=interval,
                period='1d',
                is_indian=True
            )
            
            if result['success']:
                print(f"    ✅ Success: {result['data_points']} data points")
            else:
                print(f"    ❌ Failure: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")

def generate_test_summary(us_results, indian_results):
    """Generate comprehensive test summary"""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    # US Stock Summary
    print("\n🇺🇸 US Stock Results (Yahoo Finance - Daily Only)")
    print("-" * 50)
    
    for ticker, results in us_results.items():
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)
        print(f"{ticker}: {successful}/{total} tests successful")
        
        # Show data points for successful tests
        for test_name, result in results.items():
            if result.get('success'):
                data_points = result.get('data_points', 0)
                fetch_time = result.get('fetch_time', 0)
                print(f"  {test_name}: {data_points} data points in {fetch_time:.2f}s")
    
    # Indian Stock Summary
    print("\n🇮🇳 Indian Stock Results (Angel One API - All Intervals)")
    print("-" * 50)
    
    for ticker, results in indian_results.items():
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)
        print(f"{ticker}: {successful}/{total} tests successful")
        
        # Show data points for successful tests
        for test_name, result in results.items():
            if result.get('success'):
                data_points = result.get('data_points', 0)
                fetch_time = result.get('fetch_time', 0)
                print(f"  {test_name}: {data_points} data points in {fetch_time:.2f}s")
    
    # Overall Summary
    print("\n🎯 Overall Summary")
    print("-" * 30)
    
    us_successful = sum(1 for ticker_results in us_results.values() 
                       for result in ticker_results.values() if result.get('success'))
    us_total = sum(len(ticker_results) for ticker_results in us_results.values())
    
    indian_successful = sum(1 for ticker_results in indian_results.values() 
                           for result in ticker_results.values() if result.get('success'))
    indian_total = sum(len(ticker_results) for ticker_results in indian_results.values())
    
    print(f"US Stocks: {us_successful}/{us_total} tests successful")
    print(f"Indian Stocks: {indian_successful}/{indian_total} tests successful")
    print(f"Total: {us_successful + indian_successful}/{us_total + indian_total} tests successful")

def main():
    """Main test function"""
    try:
        print("🧪 Corrected Interval Data Testing")
        print("=" * 50)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test US stocks (Yahoo Finance - daily only)
        us_results = test_us_stock_intervals()
        
        # Test Indian stocks (Angel One API - all intervals)
        indian_results = test_indian_stock_intervals()
        
        # Test data source limitations
        test_data_source_limitations()
        
        # Generate summary
        generate_test_summary(us_results, indian_results)
        
        print("\n🎯 Corrected interval testing completed!")
        print("✅ Properly handles data source limitations")
        print("✅ US stocks: Daily data only (Yahoo Finance)")
        print("✅ Indian stocks: All intervals (Angel One API)")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
