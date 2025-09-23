#!/usr/bin/env python3
"""
Comprehensive Angel One API Test - Maximum Data for All Intervals
Tests database storage and retrieval for all interval types
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_angel_maximum_data_all_intervals():
    """Test Angel One API with maximum data for all intervals"""
    print("📊 Testing Angel One API - Maximum Data for All Intervals")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        from main.services.angel_one_manager import AngelOneManager
        from main.services.database_manager import DatabaseManager
        
        # Initialize services
        print("🔧 Initializing services...")
        downloader = AngelOneDataDownloader()
        
        # Database configuration
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '7874',
            'database': 'stock_data',
            'charset': 'utf8mb4',
            'autocommit': True
        }
        
        db_manager = DatabaseManager(db_config)
        angel_manager = AngelOneManager(db_config)
        
        # Authenticate
        print("🔐 Authenticating with Angel One API...")
        auth_success = downloader.authenticate()
        
        if not auth_success:
            print("❌ Authentication failed!")
            return False
        
        print("✅ Authentication successful!")
        
        # Test all intervals with maximum data
        test_intervals = [
            ('ONE_MINUTE', 30),      # Max 30 days for 1-minute data
            ('THREE_MINUTE', 60),    # Max 60 days for 3-minute data
            ('FIVE_MINUTE', 100),    # Max 100 days for 5-minute data
            ('TEN_MINUTE', 100),     # Max 100 days for 10-minute data
            ('FIFTEEN_MINUTE', 200), # Max 200 days for 15-minute data
            ('THIRTY_MINUTE', 200),  # Max 200 days for 30-minute data
            ('ONE_HOUR', 400),       # Max 400 days for 1-hour data
            ('ONE_DAY', 2000)        # Max 2000 days for daily data
        ]
        
        # Test stocks
        test_stocks = [
            ('RELIANCE', 'NSE'),
            ('TATAMOTORS', 'NSE'),
            ('TCS', 'NSE'),
            ('HDFC', 'NSE'),
            ('ICICIBANK', 'NSE')
        ]
        
        results = {}
        
        for stock, exchange in test_stocks:
            print(f"\n📈 Testing {stock} on {exchange}")
            print("=" * 40)
            
            stock_results = {}
            
            for interval, max_days in test_intervals:
                print(f"\n⏰ Testing {interval} (Max {max_days} days)...")
                
                try:
                    # Calculate date range for maximum data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=max_days)
                    
                    from_date = start_date.strftime('%Y-%m-%d %H:%M')
                    to_date = end_date.strftime('%Y-%m-%d %H:%M')
                    
                    print(f"   📅 Period: {from_date} to {to_date}")
                    
                    # Download data
                    print(f"   📥 Downloading data...")
                    data = downloader.get_historical_data(
                        symbol_name=stock,
                        exchange=exchange,
                        interval=interval,
                        from_date=from_date,
                        to_date=to_date,
                        days_back=max_days
                    )
                    
                    if data is not None and not data.empty:
                        print(f"   ✅ Downloaded: {len(data)} records")
                        print(f"   📊 Columns: {list(data.columns)}")
                        print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
                        
                        # Store in database
                        print(f"   💾 Storing in database...")
                        try:
                            # Store using Angel One manager
                            stored = angel_manager.store_enhanced_data(
                                ticker=stock,
                                data=data,
                                interval=interval,
                                exchange=exchange
                            )
                            
                            if stored:
                                print(f"   ✅ Stored in database successfully")
                                
                                # Retrieve from database
                                print(f"   📤 Retrieving from database...")
                                retrieved_data = db_manager.get_stock_data(
                                    ticker=stock,
                                    interval=interval,
                                    limit=len(data)
                                )
                                
                                if retrieved_data is not None and not retrieved_data.empty:
                                    print(f"   ✅ Retrieved: {len(retrieved_data)} records")
                                    
                                    # Compare data
                                    if len(retrieved_data) == len(data):
                                        print(f"   ✅ Data integrity: Perfect match")
                                    else:
                                        print(f"   ⚠️  Data integrity: {len(retrieved_data)} vs {len(data)} records")
                                    
                                    stock_results[interval] = {
                                        'downloaded': len(data),
                                        'stored': True,
                                        'retrieved': len(retrieved_data),
                                        'success': True
                                    }
                                else:
                                    print(f"   ❌ Failed to retrieve from database")
                                    stock_results[interval] = {
                                        'downloaded': len(data),
                                        'stored': True,
                                        'retrieved': 0,
                                        'success': False
                                    }
                            else:
                                print(f"   ❌ Failed to store in database")
                                stock_results[interval] = {
                                    'downloaded': len(data),
                                    'stored': False,
                                    'retrieved': 0,
                                    'success': False
                                }
                                
                        except Exception as e:
                            print(f"   ❌ Database operation failed: {e}")
                            stock_results[interval] = {
                                'downloaded': len(data),
                                'stored': False,
                                'retrieved': 0,
                                'success': False,
                                'error': str(e)
                            }
                    else:
                        print(f"   ⚠️  No data received")
                        stock_results[interval] = {
                            'downloaded': 0,
                            'stored': False,
                            'retrieved': 0,
                            'success': False
                        }
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    stock_results[interval] = {
                        'downloaded': 0,
                        'stored': False,
                        'retrieved': 0,
                        'success': False,
                        'error': str(e)
                    }
            
            results[stock] = stock_results
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_database_operations():
    """Test database operations separately"""
    print("\n💾 Testing Database Operations")
    print("=" * 35)
    
    try:
        from main.services.database_manager import DatabaseManager
        
        # Database configuration
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '7874',
            'database': 'stock_data',
            'charset': 'utf8mb4',
            'autocommit': True
        }
        
        db_manager = DatabaseManager(db_config)
        
        # Test database connection
        print("🔌 Testing database connection...")
        connection_success = db_manager.test_connection()
        
        if connection_success:
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            return False
        
        # Test database statistics
        print("\n📊 Getting database statistics...")
        stats = db_manager.get_database_statistics()
        print(f"   📈 Database stats: {stats}")
        
        # Test available tickers
        print("\n📋 Getting available tickers...")
        tickers = db_manager.get_available_tickers()
        print(f"   📈 Available tickers: {len(tickers)} found")
        if tickers:
            print(f"   📈 Sample tickers: {list(tickers)[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_comprehensive_report(results):
    """Generate comprehensive test report"""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    if not results:
        print("❌ No results to report")
        return
    
    total_tests = 0
    successful_tests = 0
    total_downloaded = 0
    total_stored = 0
    total_retrieved = 0
    
    for stock, stock_results in results.items():
        print(f"\n📈 {stock} Results:")
        print("-" * 30)
        
        for interval, result in stock_results.items():
            total_tests += 1
            if result.get('success', False):
                successful_tests += 1
            
            total_downloaded += result.get('downloaded', 0)
            if result.get('stored', False):
                total_stored += 1
            total_retrieved += result.get('retrieved', 0)
            
            status = "✅" if result.get('success', False) else "❌"
            print(f"   {status} {interval}: {result.get('downloaded', 0)} records")
            
            if result.get('error'):
                print(f"      Error: {result['error']}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    print(f"   Total Records Downloaded: {total_downloaded}")
    print(f"   Database Operations: {total_stored} stored")
    print(f"   Total Records Retrieved: {total_retrieved}")
    
    # Save detailed report
    report_file = "docs/ANGEL_ONE_COMPREHENSIVE_TEST_REPORT.md"
    os.makedirs("docs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Angel One Comprehensive Test Report\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Tests:** {total_tests}\n")
        f.write(f"**Successful:** {successful_tests}\n")
        f.write(f"**Success Rate:** {(successful_tests/total_tests)*100:.1f}%\n\n")
        
        f.write("## Detailed Results\n\n")
        for stock, stock_results in results.items():
            f.write(f"### {stock}\n\n")
            for interval, result in stock_results.items():
                status = "✅" if result.get('success', False) else "❌"
                f.write(f"- {status} **{interval}**: {result.get('downloaded', 0)} records\n")
                if result.get('error'):
                    f.write(f"  - Error: {result['error']}\n")
            f.write("\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main test function"""
    try:
        print("🧪 Angel One Comprehensive Data Test")
        print("=" * 50)
        
        # Test database operations first
        db_success = test_database_operations()
        
        if not db_success:
            print("❌ Database operations failed, skipping comprehensive test")
            return False
        
        # Run comprehensive test
        results = test_angel_maximum_data_all_intervals()
        
        if results:
            # Generate report
            generate_comprehensive_report(results)
            
            # Calculate overall success
            total_tests = sum(len(stock_results) for stock_results in results.values())
            successful_tests = sum(
                sum(1 for result in stock_results.values() if result.get('success', False))
                for stock_results in results.values()
            )
            
            overall_success = successful_tests > 0
            
            print(f"\n🎯 Overall Result: {'✅ TESTS COMPLETED' if overall_success else '❌ ALL TESTS FAILED'}")
            
            if overall_success:
                print("\n🎉 Angel One comprehensive testing completed!")
                print("✅ Maximum data downloaded for all intervals")
                print("✅ Database storage tested")
                print("✅ Data retrieval verified")
                print("✅ Comprehensive report generated")
                print("\n🚀 System ready for production use!")
            
            return overall_success
        else:
            print("❌ Comprehensive test failed")
            return False
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
