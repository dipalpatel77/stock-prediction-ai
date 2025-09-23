#!/usr/bin/env python3
"""
Comprehensive Interval Data Testing
Test all data intervals (minute, hourly, daily, etc.) and maximum data points
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interval_testing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntervalDataTester:
    """Comprehensive tester for all data intervals and maximum data points"""
    
    def __init__(self):
        """Initialize the interval data tester"""
        self.logger = logging.getLogger(__name__)
        
        # Define all available intervals
        self.intervals = [
            'ONE_MINUTE', 'THREE_MINUTE', 'FIVE_MINUTE', 'TEN_MINUTE',
            'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'ONE_DAY',
            'ONE_WEEK', 'ONE_MONTH'
        ]
        
        # Expected maximum data points per interval
        self.max_data_points = {
            'ONE_MINUTE': 30000,      # ~20 trading days * 6.5 hours * 60 minutes
            'THREE_MINUTE': 10000,    # ~20 trading days * 6.5 hours * 20 periods
            'FIVE_MINUTE': 6000,      # ~20 trading days * 6.5 hours * 12 periods
            'TEN_MINUTE': 3000,       # ~20 trading days * 6.5 hours * 6 periods
            'FIFTEEN_MINUTE': 2000,   # ~20 trading days * 6.5 hours * 4 periods
            'THIRTY_MINUTE': 1000,    # ~20 trading days * 6.5 hours * 2 periods
            'ONE_HOUR': 500,          # ~20 trading days * 6.5 hours
            'ONE_DAY': 2000,          # ~8 years of daily data
            'ONE_WEEK': 400,          # ~8 years of weekly data
            'ONE_MONTH': 100          # ~8 years of monthly data
        }
        
        # Test periods for each interval
        self.test_periods = {
            'ONE_MINUTE': ['1d', '5d', '1mo', '3mo'],
            'THREE_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo'],
            'FIVE_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
            'TEN_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
            'FIFTEEN_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
            'THIRTY_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
            'ONE_HOUR': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
            'ONE_DAY': ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            'ONE_WEEK': ['1y', '2y', '5y', 'max'],
            'ONE_MONTH': ['2y', '5y', 'max']
        }
        
        # Performance benchmarks
        self.performance_benchmarks = {
            'ONE_MINUTE': {'max_fetch_time': 30, 'max_memory_mb': 500},
            'FIVE_MINUTE': {'max_fetch_time': 20, 'max_memory_mb': 300},
            'ONE_HOUR': {'max_fetch_time': 15, 'max_memory_mb': 200},
            'ONE_DAY': {'max_fetch_time': 10, 'max_memory_mb': 100}
        }
        
        # Test results storage
        self.test_results = {}
        
    def test_single_interval(self, interval: str, ticker: str = 'AAPL') -> Dict[str, Any]:
        """Test a single interval with all available periods"""
        self.logger.info(f"🧪 Testing {interval} interval for {ticker}")
        
        try:
            # Import required modules
            from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
            
            # Test configuration
            config = {
                'ticker': ticker,
                'is_indian': False,
                'analysis_type': 'comprehensive',
                'parameters': {
                    'use_enhanced': True,
                    'use_database': True
                },
                'use_enhanced': True,
                'use_database': True,
                'success': True
            }
            
            # Test results for this interval
            interval_results = {
                'interval': interval,
                'ticker': ticker,
                'period_tests': {},
                'max_data_test': {},
                'performance_metrics': {},
                'success': True,
                'errors': []
            }
            
            # Test each period for this interval
            periods = self.test_periods.get(interval, ['1y'])
            for period in periods:
                self.logger.info(f"  📊 Testing period: {period}")
                
                try:
                    # Update config with interval and period
                    config['parameters']['interval'] = interval
                    config['parameters']['period'] = period
                    config['interval'] = interval
                    config['timeframe'] = period
                    
                    # Initialize pipeline
                    pipeline = UnifiedAnalysisPipeline(ticker=ticker, config=config)
                    
                    # Measure performance
                    start_time = time.time()
                    start_memory = self._get_memory_usage()
                    
                    # Run analysis
                    results = pipeline.run_analysis(**config['parameters'])
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Calculate metrics
                    fetch_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    
                    # Store results
                    period_result = {
                        'success': results.get('success', False),
                        'fetch_time': fetch_time,
                        'memory_usage_mb': memory_usage,
                        'data_points': self._count_data_points(results),
                        'performance_acceptable': self._check_performance(interval, fetch_time, memory_usage)
                    }
                    
                    interval_results['period_tests'][period] = period_result
                    
                    self.logger.info(f"    ✅ {period}: {period_result['data_points']} data points in {fetch_time:.2f}s")
                    
                except Exception as e:
                    error_msg = f"Period {period} failed: {e}"
                    self.logger.error(f"    ❌ {error_msg}")
                    interval_results['errors'].append(error_msg)
                    interval_results['period_tests'][period] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Test maximum data points
            self.logger.info(f"  📈 Testing maximum data points for {interval}")
            max_data_result = self._test_maximum_data_points(interval, ticker)
            interval_results['max_data_test'] = max_data_result
            
            # Calculate overall performance metrics
            interval_results['performance_metrics'] = self._calculate_performance_metrics(interval_results)
            
            # Determine overall success
            interval_results['success'] = len(interval_results['errors']) == 0
            
            self.test_results[interval] = interval_results
            return interval_results
            
        except Exception as e:
            error_msg = f"Interval {interval} testing failed: {e}"
            self.logger.error(error_msg)
            return {
                'interval': interval,
                'success': False,
                'error': str(e)
            }
    
    def test_all_intervals(self, ticker: str = 'AAPL') -> Dict[str, Any]:
        """Test all intervals comprehensively"""
        self.logger.info("🚀 Starting comprehensive interval testing")
        self.logger.info("=" * 60)
        
        all_results = {}
        
        for interval in self.intervals:
            self.logger.info(f"\n📊 Testing {interval}")
            self.logger.info("-" * 40)
            
            try:
                result = self.test_single_interval(interval, ticker)
                all_results[interval] = result
                
                if result['success']:
                    self.logger.info(f"✅ {interval}: PASSED")
                else:
                    self.logger.info(f"❌ {interval}: FAILED")
                    
            except Exception as e:
                self.logger.error(f"❌ {interval}: EXCEPTION - {e}")
                all_results[interval] = {
                    'interval': interval,
                    'success': False,
                    'error': str(e)
                }
        
        return all_results
    
    def test_maximum_data_points(self, interval: str, ticker: str = 'AAPL') -> Dict[str, Any]:
        """Test maximum data points for a specific interval"""
        self.logger.info(f"📈 Testing maximum data points for {interval}")
        
        try:
            # Import required modules
            from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
            
            # Configuration for maximum data
            config = {
                'ticker': ticker,
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
            
            # Initialize pipeline
            pipeline = UnifiedAnalysisPipeline(ticker=ticker, config=config)
            
            # Measure performance
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Run analysis
            results = pipeline.run_analysis(**config['parameters'])
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            fetch_time = end_time - start_time
            memory_usage = end_memory - start_memory
            data_points = self._count_data_points(results)
            expected_max = self.max_data_points.get(interval, 1000)
            
            return {
                'success': results.get('success', False),
                'data_points': data_points,
                'expected_max': expected_max,
                'fetch_time': fetch_time,
                'memory_usage_mb': memory_usage,
                'performance_acceptable': self._check_performance(interval, fetch_time, memory_usage),
                'data_quality': self._assess_data_quality(results)
            }
            
        except Exception as e:
            self.logger.error(f"Maximum data test failed for {interval}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_maximum_data_points(self, interval: str, ticker: str) -> Dict[str, Any]:
        """Internal method to test maximum data points"""
        return self.test_maximum_data_points(interval, ticker)
    
    def _count_data_points(self, results: Dict[str, Any]) -> int:
        """Count data points from results"""
        try:
            if 'results' in results and 'data_processor' in results['results']:
                data_result = results['results']['data_processor']
                return data_result.get('records_processed', 0)
            return 0
        except Exception:
            return 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _check_performance(self, interval: str, fetch_time: float, memory_usage: float) -> bool:
        """Check if performance is within acceptable limits"""
        benchmark = self.performance_benchmarks.get(interval, {'max_fetch_time': 30, 'max_memory_mb': 500})
        
        time_ok = fetch_time <= benchmark['max_fetch_time']
        memory_ok = memory_usage <= benchmark['max_memory_mb']
        
        return time_ok and memory_ok
    
    def _assess_data_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality from results"""
        try:
            if 'results' in results and 'data_processor' in results['results']:
                data_result = results['results']['data_processor']
                return {
                    'quality_score': data_result.get('quality_score', 0),
                    'records_processed': data_result.get('records_processed', 0),
                    'data_source': data_result.get('data_source', 'Unknown')
                }
            return {'quality_score': 0, 'records_processed': 0, 'data_source': 'Unknown'}
        except Exception:
            return {'quality_score': 0, 'records_processed': 0, 'data_source': 'Unknown'}
    
    def _calculate_performance_metrics(self, interval_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics for an interval"""
        try:
            period_tests = interval_results.get('period_tests', {})
            if not period_tests:
                return {'avg_fetch_time': 0, 'avg_memory_usage': 0, 'total_data_points': 0}
            
            # Calculate averages
            fetch_times = [test.get('fetch_time', 0) for test in period_tests.values() if test.get('success')]
            memory_usages = [test.get('memory_usage_mb', 0) for test in period_tests.values() if test.get('success')]
            data_points = [test.get('data_points', 0) for test in period_tests.values() if test.get('success')]
            
            return {
                'avg_fetch_time': sum(fetch_times) / len(fetch_times) if fetch_times else 0,
                'avg_memory_usage': sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                'total_data_points': sum(data_points),
                'successful_periods': len([test for test in period_tests.values() if test.get('success')]),
                'total_periods': len(period_tests)
            }
        except Exception:
            return {'avg_fetch_time': 0, 'avg_memory_usage': 0, 'total_data_points': 0}
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("🧪 COMPREHENSIVE INTERVAL DATA TESTING REPORT")
        report.append("=" * 60)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_intervals = len(results)
        successful_intervals = sum(1 for r in results.values() if r.get('success'))
        failed_intervals = total_intervals - successful_intervals
        
        report.append("📊 SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Intervals Tested: {total_intervals}")
        report.append(f"Successful: {successful_intervals}")
        report.append(f"Failed: {failed_intervals}")
        report.append(f"Success Rate: {(successful_intervals/total_intervals)*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 30)
        
        for interval, result in results.items():
            status = "✅ PASS" if result.get('success') else "❌ FAIL"
            report.append(f"{interval}: {status}")
            
            if result.get('success'):
                metrics = result.get('performance_metrics', {})
                report.append(f"  📈 Data Points: {metrics.get('total_data_points', 0)}")
                report.append(f"  ⏱️ Avg Fetch Time: {metrics.get('avg_fetch_time', 0):.2f}s")
                report.append(f"  💾 Avg Memory: {metrics.get('avg_memory_usage', 0):.1f}MB")
                report.append(f"  ✅ Successful Periods: {metrics.get('successful_periods', 0)}/{metrics.get('total_periods', 0)}")
            else:
                errors = result.get('errors', [])
                if errors:
                    report.append(f"  ❌ Errors: {len(errors)}")
                    for error in errors[:3]:  # Show first 3 errors
                        report.append(f"    - {error}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main test function"""
    try:
        print("🧪 Comprehensive Interval Data Testing")
        print("=" * 50)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Initialize tester
        tester = IntervalDataTester()
        
        # Test all intervals
        print("🚀 Testing all intervals...")
        results = tester.test_all_intervals('AAPL')
        
        # Generate and display report
        report = tester.generate_test_report(results)
        print("\n" + report)
        
        # Save report to file
        with open('interval_testing_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Report saved to: interval_testing_report.txt")
        
        # Determine overall success
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)
        
        print(f"\n🎯 Overall Result: {successful}/{total} intervals passed")
        
        return successful == total
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
