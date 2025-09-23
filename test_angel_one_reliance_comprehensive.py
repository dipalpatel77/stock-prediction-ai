#!/usr/bin/env python3
"""
Comprehensive Angel One API Test for RELIANCE Stock
Tests all intervals with maximum data, database storage, and retrieval
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_angel_one_reliance_comprehensive():
    """Test Angel One API with RELIANCE stock for all intervals with maximum data"""
    print("🧪 Comprehensive Angel One API Test for RELIANCE Stock")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        from main.services.database_manager import DatabaseManager
        
        # Initialize downloader
        print("🔧 Initializing Angel One Data Downloader...")
        downloader = AngelOneDataDownloader()
        
        # Initialize database manager
        print("🔧 Initializing Database Manager...")
        db_manager = DatabaseManager()
        
        # Test database connection
        print("🔍 Testing database connection...")
        if not db_manager.test_connection():
            print("❌ Database connection failed!")
            return False
        print("✅ Database connection successful!")
        
        # Test authentication
        print("\n🔐 Testing Authentication...")
        auth_success = downloader.authenticate()
        
        if not auth_success:
            print("❌ Authentication failed!")
            return False
        
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
        storage_results = {}
        retrieval_results = {}
        
        print(f"\n📈 Testing RELIANCE Stock - All Intervals with Maximum Data")
        print("=" * 60)
        
        for interval, max_days, description in test_intervals:
            print(f"\n⏰ Testing {interval} ({description}) - Max {max_days} days...")
            
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
                    symbol_name='RELIANCE',
                    exchange='BSE',
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
                        print(f"      {idx}: Open={row['Open']}, High={row['High']}, Low={row['Low']}, Close={row['Close']}, Volume={row['Volume']}")
                    
                    # Store data in database
                    print(f"   💾 Storing data in database...")
                    try:
                        db_manager.store_stock_data(
                            ticker='RELIANCE',
                            data=data,
                            source='angel_one',
                            interval=interval
                        )
                        print(f"   ✅ Data stored successfully in database")
                        storage_results[interval] = {
                            'success': True,
                            'records_stored': len(data),
                            'interval': interval
                        }
                    except Exception as e:
                        print(f"   ❌ Database storage failed: {e}")
                        storage_results[interval] = {
                            'success': False,
                            'error': str(e),
                            'interval': interval
                        }
                    
                    # Retrieve data from database
                    print(f"   🔍 Retrieving data from database...")
                    try:
                        retrieved_data = db_manager.get_stock_data(
                            ticker='RELIANCE',
                            period=f'{max_days}d',
                            source='angel_one',
                            interval=interval
                        )
                        
                        if retrieved_data is not None and not retrieved_data.empty:
                            print(f"   ✅ Retrieved: {len(retrieved_data)} records from database")
                            print(f"   📊 Retrieved columns: {list(retrieved_data.columns)}")
                            print(f"   📅 Retrieved date range: {retrieved_data.index.min()} to {retrieved_data.index.max()}")
                            
                            # Compare original vs retrieved data
                            data_match = len(data) == len(retrieved_data)
                            print(f"   🔍 Data integrity check: {'✅ Match' if data_match else '❌ Mismatch'}")
                            
                            retrieval_results[interval] = {
                                'success': True,
                                'records_retrieved': len(retrieved_data),
                                'data_match': data_match,
                                'interval': interval
                            }
                        else:
                            print(f"   ❌ No data retrieved from database")
                            retrieval_results[interval] = {
                                'success': False,
                                'error': 'No data retrieved',
                                'interval': interval
                            }
                    except Exception as e:
                        print(f"   ❌ Database retrieval failed: {e}")
                        retrieval_results[interval] = {
                            'success': False,
                            'error': str(e),
                            'interval': interval
                        }
                    
                    results[interval] = {
                        'success': True,
                        'records': len(data),
                        'columns': list(data.columns),
                        'date_range': f"{data.index.min()} to {data.index.max()}",
                        'max_days': max_days,
                        'description': description,
                        'storage_success': storage_results[interval]['success'],
                        'retrieval_success': retrieval_results[interval]['success']
                    }
                else:
                    print(f"   ⚠️ No data received")
                    results[interval] = {
                        'success': False,
                        'records': 0,
                        'error': 'No data received from API',
                        'max_days': max_days,
                        'description': description,
                        'storage_success': False,
                        'retrieval_success': False
                    }
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[interval] = {
                    'success': False,
                    'records': 0,
                    'error': str(e),
                    'max_days': max_days,
                    'description': description,
                    'storage_success': False,
                    'retrieval_success': False
                }
        
        return results, storage_results, retrieval_results
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_comprehensive_report(interval_results, storage_results, retrieval_results):
    """Generate comprehensive report for RELIANCE stock testing"""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE ANGEL ONE API TEST REPORT - RELIANCE STOCK")
    print("=" * 80)
    
    if not interval_results:
        print("❌ No interval results to report")
        return
    
    total_interval_tests = len(interval_results)
    successful_interval_tests = sum(1 for result in interval_results.values() if result.get('success', False))
    total_interval_records = sum(result.get('records', 0) for result in interval_results.values())
    
    total_storage_tests = len(storage_results)
    successful_storage_tests = sum(1 for result in storage_results.values() if result.get('success', False))
    
    total_retrieval_tests = len(retrieval_results)
    successful_retrieval_tests = sum(1 for result in retrieval_results.values() if result.get('success', False))
    
    print(f"\n📈 RELIANCE Stock - All Intervals Results:")
    print("-" * 50)
    
    for interval, result in interval_results.items():
        status = "✅" if result.get('success', False) else "❌"
        records = result.get('records', 0)
        max_days = result.get('max_days', 0)
        description = result.get('description', '')
        storage_status = "✅" if result.get('storage_success', False) else "❌"
        retrieval_status = "✅" if result.get('retrieval_success', False) else "❌"
        
        print(f"   {status} {interval}: {records} records (Max: {max_days} days)")
        print(f"      📝 {description}")
        print(f"      💾 Storage: {storage_status}")
        print(f"      🔍 Retrieval: {retrieval_status}")
        
        if result.get('error'):
            print(f"      ❌ Error: {result['error']}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Interval Tests: {total_interval_tests}")
    print(f"   Successful Intervals: {successful_interval_tests}")
    print(f"   Interval Success Rate: {(successful_interval_tests/total_interval_tests)*100:.1f}%")
    print(f"   Total Interval Records: {total_interval_records}")
    print(f"   Storage Tests: {total_storage_tests}")
    print(f"   Successful Storage: {successful_storage_tests}")
    print(f"   Storage Success Rate: {(successful_storage_tests/total_storage_tests)*100:.1f}%")
    print(f"   Retrieval Tests: {total_retrieval_tests}")
    print(f"   Successful Retrieval: {successful_retrieval_tests}")
    print(f"   Retrieval Success Rate: {(successful_retrieval_tests/total_retrieval_tests)*100:.1f}%")
    
    # Save detailed report
    report_file = "docs/RELIANCE_COMPREHENSIVE_TEST_REPORT.md"
    os.makedirs("docs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# RELIANCE Stock Comprehensive Angel One API Test Report\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Stock:** RELIANCE (BSE)\n")
        f.write(f"**Interval Tests:** {total_interval_tests}\n")
        f.write(f"**Successful Intervals:** {successful_interval_tests}\n")
        f.write(f"**Interval Success Rate:** {(successful_interval_tests/total_interval_tests)*100:.1f}%\n")
        f.write(f"**Total Interval Records:** {total_interval_records}\n\n")
        f.write(f"**Storage Tests:** {total_storage_tests}\n")
        f.write(f"**Successful Storage:** {successful_storage_tests}\n")
        f.write(f"**Storage Success Rate:** {(successful_storage_tests/total_storage_tests)*100:.1f}%\n\n")
        f.write(f"**Retrieval Tests:** {total_retrieval_tests}\n")
        f.write(f"**Successful Retrieval:** {successful_retrieval_tests}\n")
        f.write(f"**Retrieval Success Rate:** {(successful_retrieval_tests/total_retrieval_tests)*100:.1f}%\n\n")
        
        f.write("## Interval Test Results\n\n")
        for interval, result in interval_results.items():
            f.write(f"### {interval}\n\n")
            f.write(f"- **Status**: {'✅' if result.get('success', False) else '❌'}\n")
            f.write(f"- **Records**: {result.get('records', 0)}\n")
            f.write(f"- **Max Days**: {result.get('max_days', 0)}\n")
            f.write(f"- **Description**: {result.get('description', '')}\n")
            f.write(f"- **Storage Success**: {'✅' if result.get('storage_success', False) else '❌'}\n")
            f.write(f"- **Retrieval Success**: {'✅' if result.get('retrieval_success', False) else '❌'}\n")
            if result.get('error'):
                f.write(f"- **Error**: {result['error']}\n")
            f.write("\n")
        
        f.write("## Storage Test Results\n\n")
        for interval, result in storage_results.items():
            f.write(f"### {interval} Storage\n\n")
            f.write(f"- **Status**: {'✅' if result.get('success', False) else '❌'}\n")
            f.write(f"- **Records Stored**: {result.get('records_stored', 0)}\n")
            if result.get('error'):
                f.write(f"- **Error**: {result['error']}\n")
            f.write("\n")
        
        f.write("## Retrieval Test Results\n\n")
        for interval, result in retrieval_results.items():
            f.write(f"### {interval} Retrieval\n\n")
            f.write(f"- **Status**: {'✅' if result.get('success', False) else '❌'}\n")
            f.write(f"- **Records Retrieved**: {result.get('records_retrieved', 0)}\n")
            f.write(f"- **Data Match**: {'✅' if result.get('data_match', False) else '❌'}\n")
            if result.get('error'):
                f.write(f"- **Error**: {result['error']}\n")
            f.write("\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main test function"""
    try:
        print("🧪 Comprehensive Angel One API Test for RELIANCE Stock")
        print("=" * 60)
        
        # Run comprehensive test
        interval_results, storage_results, retrieval_results = test_angel_one_reliance_comprehensive()
        
        if interval_results:
            # Generate comprehensive report
            generate_comprehensive_report(interval_results, storage_results, retrieval_results)
            
            # Calculate overall success
            total_interval_tests = len(interval_results)
            successful_interval_tests = sum(1 for result in interval_results.values() if result.get('success', False))
            
            total_storage_tests = len(storage_results)
            successful_storage_tests = sum(1 for result in storage_results.values() if result.get('success', False))
            
            total_retrieval_tests = len(retrieval_results)
            successful_retrieval_tests = sum(1 for result in retrieval_results.values() if result.get('success', False))
            
            overall_success = (successful_interval_tests > 0 and 
                             successful_storage_tests > 0 and 
                             successful_retrieval_tests > 0)
            
            print(f"\n🎯 Overall Result: {'✅ COMPREHENSIVE TEST COMPLETED' if overall_success else '❌ COMPREHENSIVE TEST FAILED'}")
            
            if overall_success:
                print("\n🎉 Comprehensive Angel One API test for RELIANCE successful!")
                print("✅ All intervals tested with maximum data")
                print("✅ Database storage verified")
                print("✅ Database retrieval verified")
                print("✅ Data integrity confirmed")
                print("✅ Comprehensive report generated")
                print("\n🚀 Angel One API is fully functional with database integration!")
            else:
                print("\n❌ Comprehensive Angel One API test for RELIANCE failed!")
                print("❌ Check API access, database connection, and data integrity")
                print("❌ Review detailed report for specific issues")
            
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
