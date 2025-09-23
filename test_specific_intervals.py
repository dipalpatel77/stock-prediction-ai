#!/usr/bin/env python3
"""
Specific Interval Testing
Test specific intervals with maximum data points
"""

import sys
import os
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_interval_with_periods(interval: str, periods: list, ticker: str = 'AAPL'):
    """Test a specific interval with multiple periods"""
    print(f"\n🧪 Testing {interval} interval")
    print("-" * 40)
    
    try:
        from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
        
        results = {}
        
        for period in periods:
            print(f"  📊 Testing period: {period}")
            
            try:
                # Configuration
                config = {
                    'ticker': ticker,
                    'is_indian': False,
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
                
                # Initialize pipeline
                pipeline = UnifiedAnalysisPipeline(ticker=ticker, config=config)
                
                # Measure time
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
                
                results[period] = {
                    'success': result.get('success', False),
                    'data_points': data_points,
                    'fetch_time': fetch_time
                }
                
                status = "✅" if result.get('success') else "❌"
                print(f"    {status} {period}: {data_points} data points in {fetch_time:.2f}s")
                
            except Exception as e:
                print(f"    ❌ {period}: Error - {e}")
                results[period] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
        
    except Exception as e:
        print(f"❌ Interval {interval} testing failed: {e}")
        return {}

def test_all_intervals():
    """Test all intervals with their specific periods"""
    print("🚀 Comprehensive Interval Testing")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test cases
    test_cases = [
        {
            'interval': 'ONE_MINUTE',
            'periods': ['1d', '5d', '1mo'],
            'description': '1-minute intraday data'
        },
        {
            'interval': 'FIVE_MINUTE',
            'periods': ['1d', '5d', '1mo', '3mo'],
            'description': '5-minute intraday data'
        },
        {
            'interval': 'FIFTEEN_MINUTE',
            'periods': ['1d', '5d', '1mo', '3mo', '6mo'],
            'description': '15-minute intraday data'
        },
        {
            'interval': 'ONE_HOUR',
            'periods': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
            'description': '1-hour intraday data'
        },
        {
            'interval': 'ONE_DAY',
            'periods': ['1mo', '3mo', '6mo', '1y', '2y'],
            'description': 'Daily data'
        }
    ]
    
    all_results = {}
    
    for test_case in test_cases:
        print(f"\n📊 {test_case['description']}")
        print("=" * 50)
        
        results = test_interval_with_periods(
            test_case['interval'],
            test_case['periods']
        )
        
        all_results[test_case['interval']] = results
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    for interval, results in all_results.items():
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)
        print(f"{interval}: {successful}/{total} periods successful")
        
        # Show data points for successful tests
        for period, result in results.items():
            if result.get('success'):
                data_points = result.get('data_points', 0)
                fetch_time = result.get('fetch_time', 0)
                print(f"  {period}: {data_points} data points in {fetch_time:.2f}s")
    
    return all_results

def test_maximum_data_points():
    """Test maximum data points for each interval"""
    print("\n📈 Testing Maximum Data Points")
    print("=" * 40)
    
    intervals = ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'ONE_DAY']
    
    for interval in intervals:
        print(f"\n🧪 Testing {interval} with maximum data")
        
        try:
            from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
            
            config = {
                'ticker': 'AAPL',
                'is_indian': False,
                'analysis_type': 'comprehensive',
                'parameters': {
                    'interval': interval,
                    'period': 'max',
                    'use_enhanced': True,
                    'use_database': True
                },
                'use_enhanced': True,
                'use_database': True,
                'interval': interval,
                'timeframe': 'max',
                'success': True
            }
            
            pipeline = UnifiedAnalysisPipeline(ticker='AAPL', config=config)
            
            start_time = time.time()
            result = pipeline.run_analysis(**config['parameters'])
            end_time = time.time()
            
            fetch_time = end_time - start_time
            data_points = 0
            
            if result.get('success') and 'results' in result:
                data_result = result['results'].get('data_processor', {})
                data_points = data_result.get('records_processed', 0)
            
            status = "✅" if result.get('success') else "❌"
            print(f"  {status} {interval}: {data_points} data points in {fetch_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ {interval}: Error - {e}")

def main():
    """Main test function"""
    try:
        # Test all intervals with specific periods
        all_results = test_all_intervals()
        
        # Test maximum data points
        test_maximum_data_points()
        
        print("\n🎯 Interval testing completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
