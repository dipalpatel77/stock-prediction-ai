#!/usr/bin/env python3
"""
Comprehensive Angel One API Test for RELIANCE Stock
Tests maximum data download, database storage, and retrieval for all intervals
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_reliance_maximum_data():
    """Test RELIANCE stock with maximum data for all intervals"""
    print("📊 Testing RELIANCE Stock - Maximum Data for All Intervals")
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
            return None
        
        print("✅ Authentication successful!")
        
        # Test all intervals with maximum data for RELIANCE
        test_intervals = [
            ('ONE_MINUTE', 30, '1-minute data'),      # Max 30 days
            ('THREE_MINUTE', 60, '3-minute data'),    # Max 60 days
            ('FIVE_MINUTE', 100, '5-minute data'),    # Max 100 days
            ('TEN_MINUTE', 100, '10-minute data'),     # Max 100 days
            ('FIFTEEN_MINUTE', 200, '15-minute data'), # Max 200 days
            ('THIRTY_MINUTE', 200, '30-minute data'),  # Max 200 days
            ('ONE_HOUR', 400, '1-hour data'),          # Max 400 days
            ('ONE_DAY', 2000, 'daily data')            # Max 2000 days
        ]
        
        results = {}
        
        print(f"\n📈 Testing RELIANCE Stock")
        print("=" * 40)
        
        for interval, max_days, description in test_intervals:
            print(f"\n⏰ Testing {interval} ({description}) - Max {max_days} days...")
            
            try:
                # Calculate date range for maximum data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=max_days)
                
                from_date = start_date.strftime('%Y-%m-%d %H:%M')
                to_date = end_date.strftime('%Y-%m-%d %H:%M')
                
                print(f"   📅 Period: {from_date} to {to_date}")
                print(f"   📊 Expected max records: {max_days * 24 * 60 // {'ONE_MINUTE': 1, 'THREE_MINUTE': 3, 'FIVE_MINUTE': 5, 'TEN_MINUTE': 10, 'FIFTEEN_MINUTE': 15, 'THIRTY_MINUTE': 30, 'ONE_HOUR': 60, 'ONE_DAY': 1440}[interval]}")
                
                # Download data
                print(f"   📥 Downloading data...")
                data = downloader.get_historical_data(
                    symbol_name='RELIANCE',
                    exchange='NSE',
                    interval=interval,
                    from_date=from_date,
                    to_date=to_date,
                    days_back=max_days
                )
                
                if data is not None and not data.empty:
                    print(f"   ✅ Downloaded: {len(data)} records")
                    print(f"   📊 Columns: {list(data.columns)}")
                    print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
                    
                    # Show sample data
                    print(f"   📈 Sample data (first 2 records):")
                    sample_data = data.head(2)
                    for idx, row in sample_data.iterrows():
                        print(f"      {idx}: {dict(row)}")
                    
                    # Store in database
                    print(f"   💾 Storing in database...")
                    try:
                        # Store using Angel One manager
                        stored = angel_manager.store_enhanced_data(
                            ticker='RELIANCE',
                            data=data,
                            interval=interval,
                            exchange='NSE'
                        )
                        
                        if stored:
                            print(f"   ✅ Stored in database successfully")
                            
                            # Retrieve from database
                            print(f"   📤 Retrieving from database...")
                            retrieved_data = db_manager.get_stock_data(
                                ticker='RELIANCE',
                                interval=interval,
                                limit=len(data)
                            )
                            
                            if retrieved_data is not None and not retrieved_data.empty:
                                print(f"   ✅ Retrieved: {len(retrieved_data)} records")
                                
                                # Compare data integrity
                                if len(retrieved_data) == len(data):
                                    print(f"   ✅ Data integrity: Perfect match ({len(retrieved_data)} records)")
                                else:
                                    print(f"   ⚠️  Data integrity: {len(retrieved_data)} vs {len(data)} records")
                                
                                # Show sample retrieved data
                                print(f"   📈 Retrieved sample data (first 2 records):")
                                sample_retrieved = retrieved_data.head(2)
                                for idx, row in sample_retrieved.iterrows():
                                    print(f"      {idx}: {dict(row)}")
                                
                                results[interval] = {
                                    'downloaded': len(data),
                                    'stored': True,
                                    'retrieved': len(retrieved_data),
                                    'success': True,
                                    'data_integrity': len(retrieved_data) == len(data)
                                }
                            else:
                                print(f"   ❌ Failed to retrieve from database")
                                results[interval] = {
                                    'downloaded': len(data),
                                    'stored': True,
                                    'retrieved': 0,
                                    'success': False,
                                    'error': 'Database retrieval failed'
                                }
                        else:
                            print(f"   ❌ Failed to store in database")
                            results[interval] = {
                                'downloaded': len(data),
                                'stored': False,
                                'retrieved': 0,
                                'success': False,
                                'error': 'Database storage failed'
                            }
                            
                    except Exception as e:
                        print(f"   ❌ Database operation failed: {e}")
                        results[interval] = {
                            'downloaded': len(data),
                            'stored': False,
                            'retrieved': 0,
                            'success': False,
                            'error': str(e)
                        }
                else:
                    print(f"   ⚠️  No data received")
                    results[interval] = {
                        'downloaded': 0,
                        'stored': False,
                        'retrieved': 0,
                        'success': False,
                        'error': 'No data received from API'
                    }
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[interval] = {
                    'downloaded': 0,
                    'stored': False,
                    'retrieved': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results
        
    except Exception as e:
        print(f"❌ RELIANCE comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_database_operations():
    """Test database operations for RELIANCE"""
    print("\n💾 Testing Database Operations for RELIANCE")
    print("=" * 50)
    
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
        try:
            tickers = db_manager.get_available_tickers()
            print(f"   📈 Available tickers: {len(tickers)} found")
            if tickers:
                print(f"   📈 Sample tickers: {list(tickers)[:10]}")
                if 'RELIANCE' in tickers:
                    print("   ✅ RELIANCE found in database")
                else:
                    print("   ⚠️  RELIANCE not found in database")
        except Exception as e:
            print(f"   ⚠️  Could not get tickers: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_reliance_report(results):
    """Generate comprehensive report for RELIANCE testing"""
    print("\n" + "=" * 80)
    print("📊 RELIANCE COMPREHENSIVE TEST REPORT")
    print("=" * 80)
    
    if not results:
        print("❌ No results to report")
        return
    
    total_tests = len(results)
    successful_tests = sum(1 for result in results.values() if result.get('success', False))
    total_downloaded = sum(result.get('downloaded', 0) for result in results.values())
    total_stored = sum(1 for result in results.values() if result.get('stored', False))
    total_retrieved = sum(result.get('retrieved', 0) for result in results.values())
    
    print(f"\n📈 RELIANCE Stock Results:")
    print("-" * 40)
    
    for interval, result in results.items():
        status = "✅" if result.get('success', False) else "❌"
        downloaded = result.get('downloaded', 0)
        stored = "✅" if result.get('stored', False) else "❌"
        retrieved = result.get('retrieved', 0)
        
        print(f"   {status} {interval}: {downloaded} records")
        print(f"      💾 Stored: {stored}")
        print(f"      📤 Retrieved: {retrieved} records")
        
        if result.get('data_integrity'):
            print(f"      ✅ Data integrity: Perfect")
        elif result.get('error'):
            print(f"      ❌ Error: {result['error']}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful: {successful_tests}")
    print(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    print(f"   Total Records Downloaded: {total_downloaded}")
    print(f"   Database Operations: {total_stored} stored")
    print(f"   Total Records Retrieved: {total_retrieved}")
    
    # Save detailed report
    report_file = "docs/RELIANCE_COMPREHENSIVE_TEST_REPORT.md"
    os.makedirs("docs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# RELIANCE Stock Comprehensive Test Report\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Stock:** RELIANCE\n")
        f.write(f"**Total Tests:** {total_tests}\n")
        f.write(f"**Successful:** {successful_tests}\n")
        f.write(f"**Success Rate:** {(successful_tests/total_tests)*100:.1f}%\n\n")
        
        f.write("## Detailed Results\n\n")
        for interval, result in results.items():
            status = "✅" if result.get('success', False) else "❌"
            f.write(f"### {interval}\n\n")
            f.write(f"- **Status**: {status}\n")
            f.write(f"- **Downloaded**: {result.get('downloaded', 0)} records\n")
            f.write(f"- **Stored**: {'✅' if result.get('stored', False) else '❌'}\n")
            f.write(f"- **Retrieved**: {result.get('retrieved', 0)} records\n")
            if result.get('error'):
                f.write(f"- **Error**: {result['error']}\n")
            f.write("\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main test function"""
    try:
        print("🧪 RELIANCE Stock Comprehensive Test")
        print("=" * 50)
        
        # Test database operations first
        db_success = test_database_operations()
        
        if not db_success:
            print("❌ Database operations failed, skipping comprehensive test")
            return False
        
        # Run comprehensive test for RELIANCE
        results = test_reliance_maximum_data()
        
        if results:
            # Generate report
            generate_reliance_report(results)
            
            # Calculate overall success
            total_tests = len(results)
            successful_tests = sum(1 for result in results.values() if result.get('success', False))
            
            overall_success = successful_tests > 0
            
            print(f"\n🎯 Overall Result: {'✅ TESTS COMPLETED' if overall_success else '❌ ALL TESTS FAILED'}")
            
            if overall_success:
                print("\n🎉 RELIANCE comprehensive testing completed!")
                print("✅ Maximum data downloaded for all intervals")
                print("✅ Database storage tested")
                print("✅ Data retrieval verified")
                print("✅ Comprehensive report generated")
                print("\n🚀 RELIANCE stock testing successful!")
            else:
                print("\n❌ RELIANCE comprehensive testing failed!")
                print("❌ No data could be downloaded or stored")
                print("❌ Check API access and database configuration")
            
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
