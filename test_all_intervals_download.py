#!/usr/bin/env python3
"""
Test and download data for all Angel One API intervals
Comprehensive test of all available timeframes
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
from datetime import datetime, timedelta
import time

class AllIntervalsTester:
    """
    Test all Angel One API intervals
    """
    
    def __init__(self):
        self.angel_one_limits = {
            'ONE_MINUTE': 30,
            'THREE_MINUTE': 60,
            'FIVE_MINUTE': 100,
            'TEN_MINUTE': 100,
            'FIFTEEN_MINUTE': 200,
            'THIRTY_MINUTE': 200,
            'ONE_HOUR': 400,
            'ONE_DAY': 2000
        }
        
        self.interval_descriptions = {
            'ONE_MINUTE': '1 Minute (Intraday)',
            'THREE_MINUTE': '3 Minutes (Intraday)',
            'FIVE_MINUTE': '5 Minutes (Intraday)',
            'TEN_MINUTE': '10 Minutes (Intraday)',
            'FIFTEEN_MINUTE': '15 Minutes (Intraday)',
            'THIRTY_MINUTE': '30 Minutes (Intraday)',
            'ONE_HOUR': '1 Hour (Intraday)',
            'ONE_DAY': '1 Day (Daily)'
        }
        
        self.results = {}
    
    def test_single_interval(self, ticker, interval, max_days):
        """Test a single interval"""
        try:
            print(f"\n🔄 Testing {self.interval_descriptions[interval]}...")
            print(f"   📊 Max days: {max_days:,}")
            print(f"   🎯 Ticker: {ticker}")
            
            from src.core.enhanced_angel_one_service import EnhancedAngelOneService
            service = EnhancedAngelOneService()
            
            # Use maximum days for each interval
            start_time = time.time()
            
            df = service.get_optimal_historical_data(
                ticker,
                "BSE",  # Use BSE for Indian stocks
                interval,
                max_days
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if df is not None and not df.empty:
                # Calculate actual trading days
                if len(df) > 1:
                    date_range = (df.index.max() - df.index.min()).days
                else:
                    date_range = 1
                
                # Calculate data density
                data_density = (len(df) / date_range) * 100 if date_range > 0 else 0
                
                result = {
                    'success': True,
                    'records': len(df),
                    'date_range_days': date_range,
                    'data_density': data_density,
                    'duration_seconds': duration,
                    'first_date': df.index.min(),
                    'last_date': df.index.max(),
                    'latest_price': df['Close'].iloc[-1],
                    'avg_volume': df['Volume'].mean(),
                    'price_range': {
                        'min': df['Close'].min(),
                        'max': df['Close'].max()
                    }
                }
                
                print(f"   ✅ SUCCESS: {len(df):,} records")
                print(f"   📅 Date range: {date_range} days")
                print(f"   📊 Data density: {data_density:.1f}%")
                print(f"   ⏱️ Duration: {duration:.2f} seconds")
                print(f"   💰 Latest price: ₹{df['Close'].iloc[-1]:.2f}")
                print(f"   📈 Price range: ₹{df['Close'].min():.2f} - ₹{df['Close'].max():.2f}")
                
                return result
            else:
                print(f"   ❌ FAILED: No data retrieved")
                return {
                    'success': False,
                    'error': 'No data retrieved',
                    'duration_seconds': duration
                }
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': 0
            }
    
    def test_all_intervals(self, ticker="TCS"):
        """Test all intervals for a given ticker"""
        try:
            print(f"🚀 TESTING ALL ANGEL ONE API INTERVALS")
            print(f"📊 Ticker: {ticker}")
            print("=" * 60)
            
            print(f"📋 Available Intervals:")
            for interval, description in self.interval_descriptions.items():
                max_days = self.angel_one_limits[interval]
                print(f"   • {description}: {max_days:,} days max")
            print()
            
            # Test each interval
            for interval, max_days in self.angel_one_limits.items():
                result = self.test_single_interval(ticker, interval, max_days)
                self.results[interval] = result
                
                # Small delay between requests to be respectful to API
                time.sleep(2)
            
            # Display summary
            self.display_summary()
            
            return self.results
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return None
    
    def display_summary(self):
        """Display comprehensive summary of all tests"""
        print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 50)
        
        successful_tests = 0
        total_records = 0
        total_duration = 0
        
        print(f"{'Interval':<20} {'Status':<8} {'Records':<10} {'Days':<8} {'Duration':<10} {'Latest Price':<12}")
        print("-" * 80)
        
        for interval, result in self.results.items():
            if result['success']:
                successful_tests += 1
                total_records += result['records']
                total_duration += result['duration_seconds']
                
                status = "✅ PASS"
                records = f"{result['records']:,}"
                days = f"{result['date_range_days']}"
                duration = f"{result['duration_seconds']:.1f}s"
                price = f"₹{result['latest_price']:.2f}"
            else:
                status = "❌ FAIL"
                records = "0"
                days = "0"
                duration = f"{result.get('duration_seconds', 0):.1f}s"
                price = "N/A"
            
            description = self.interval_descriptions[interval][:18]
            print(f"{description:<20} {status:<8} {records:<10} {days:<8} {duration:<10} {price:<12}")
        
        print("-" * 80)
        print(f"📊 SUMMARY STATISTICS:")
        print(f"   • Successful tests: {successful_tests}/{len(self.results)}")
        print(f"   • Total records: {total_records:,}")
        print(f"   • Total duration: {total_duration:.2f} seconds")
        print(f"   • Average duration: {total_duration/len(self.results):.2f} seconds per test")
        print()
        
        # Data analysis
        if successful_tests > 0:
            print(f"📈 DATA ANALYSIS:")
            
            # Find best performing intervals
            successful_results = {k: v for k, v in self.results.items() if v['success']}
            
            if successful_results:
                # Most records
                most_records = max(successful_results.items(), key=lambda x: x[1]['records'])
                print(f"   • Most records: {most_records[0]} ({most_records[1]['records']:,} records)")
                
                # Longest date range
                longest_range = max(successful_results.items(), key=lambda x: x[1]['date_range_days'])
                print(f"   • Longest range: {longest_range[0]} ({longest_range[1]['date_range_days']} days)")
                
                # Highest data density
                highest_density = max(successful_results.items(), key=lambda x: x[1]['data_density'])
                print(f"   • Highest density: {highest_density[0]} ({highest_density[1]['data_density']:.1f}%)")
                
                # Fastest download
                fastest = min(successful_results.items(), key=lambda x: x[1]['duration_seconds'])
                print(f"   • Fastest download: {fastest[0]} ({fastest[1]['duration_seconds']:.2f}s)")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   • For intraday trading: Use 1-5 minute intervals")
        print(f"   • For short-term analysis: Use 15-30 minute intervals")
        print(f"   • For medium-term analysis: Use 1-hour intervals")
        print(f"   • For long-term analysis: Use daily intervals")
        print(f"   • For maximum data: Use daily intervals (up to 2000 days)")
    
    def save_results_to_csv(self, filename="angel_one_intervals_test_results.csv"):
        """Save test results to CSV file"""
        try:
            if not self.results:
                print("❌ No results to save")
                return False
            
            # Prepare data for CSV
            csv_data = []
            for interval, result in self.results.items():
                row = {
                    'interval': interval,
                    'description': self.interval_descriptions[interval],
                    'max_days_allowed': self.angel_one_limits[interval],
                    'success': result['success'],
                    'records_retrieved': result.get('records', 0),
                    'date_range_days': result.get('date_range_days', 0),
                    'data_density_percent': result.get('data_density', 0),
                    'duration_seconds': result.get('duration_seconds', 0),
                    'first_date': result.get('first_date', ''),
                    'last_date': result.get('last_date', ''),
                    'latest_price': result.get('latest_price', 0),
                    'avg_volume': result.get('avg_volume', 0),
                    'min_price': result.get('price_range', {}).get('min', 0),
                    'max_price': result.get('price_range', {}).get('max', 0),
                    'error': result.get('error', '')
                }
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
            
            print(f"✅ Results saved to {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False

def main():
    """Main function"""
    try:
        print("🚀 ANGEL ONE API - ALL INTERVALS TEST")
        print("=" * 50)
        
        # Get ticker from user
        ticker = input("📊 Enter ticker symbol (default: TCS): ").strip().upper()
        if not ticker:
            ticker = "TCS"
        
        print(f"🎯 Testing with ticker: {ticker}")
        print()
        
        # Create tester and run tests
        tester = AllIntervalsTester()
        results = tester.test_all_intervals(ticker)
        
        if results:
            # Ask if user wants to save results
            save_choice = input("\n💾 Save results to CSV? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                tester.save_results_to_csv()
            
            print(f"\n🎉 All intervals test completed!")
            print(f"📊 Check the summary above for detailed results")
        else:
            print(f"\n❌ Test failed")
            
    except KeyboardInterrupt:
        print(f"\n\n❌ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
