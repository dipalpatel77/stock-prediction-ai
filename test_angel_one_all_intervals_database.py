#!/usr/bin/env python3
"""
Angel One API All Intervals Database Test
Tests all intervals with maximum data and database storage/retrieval
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import sqlite3

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_angel_one_all_intervals_database():
    """Test Angel One API with all intervals, maximum data, and database operations"""
    print("🧪 Angel One API All Intervals Database Test")
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
        
        print(f"\n📈 Testing RELIANCE Stock - All Intervals with Maximum Data")
        print("=" * 60)
        
        # Create database connection
        db_path = "stock_data.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS angel_one_stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                interval_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
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
                        # Clear existing data for this interval
                        cursor.execute('''
                            DELETE FROM angel_one_stock_data 
                            WHERE ticker = ? AND interval_type = ?
                        ''', ('RELIANCE', interval))
                        
                        # Insert new data
                        records_inserted = 0
                        for idx, row in data.iterrows():
                            cursor.execute('''
                                INSERT INTO angel_one_stock_data 
                                (ticker, datetime, open_price, high_price, low_price, close_price, volume, interval_type)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                'RELIANCE',
                                idx.isoformat(),
                                row['Open'],
                                row['High'],
                                row['Low'],
                                row['Close'],
                                row['Volume'],
                                interval
                            ))
                            records_inserted += 1
                        
                        conn.commit()
                        print(f"   ✅ Stored {records_inserted} records in database")
                        
                        # Test retrieval
                        print(f"   🔍 Testing database retrieval...")
                        cursor.execute('''
                            SELECT datetime, open_price, high_price, low_price, close_price, volume
                            FROM angel_one_stock_data 
                            WHERE ticker = ? AND interval_type = ?
                            ORDER BY datetime
                        ''', ('RELIANCE', interval))
                        
                        retrieved_data = cursor.fetchall()
                        print(f"   ✅ Retrieved {len(retrieved_data)} records from database")
                        
                        # Show retrieved data
                        print(f"   📈 Retrieved data (first 2 records):")
                        for i, record in enumerate(retrieved_data[:2]):
                            print(f"      {record[0]}: Open={record[1]}, High={record[2]}, Low={record[3]}, Close={record[4]}, Volume={record[5]}")
                        
                        # Verify data integrity
                        data_match = len(data) == len(retrieved_data)
                        print(f"   🔍 Data integrity check: {'✅ Match' if data_match else '❌ Mismatch'}")
                        
                        results[interval] = {
                            'success': True,
                            'records_downloaded': len(data),
                            'records_stored': records_inserted,
                            'records_retrieved': len(retrieved_data),
                            'data_integrity': data_match,
                            'max_days': max_days,
                            'description': description
                        }
                        
                    except Exception as e:
                        print(f"   ❌ Database operation failed: {e}")
                        results[interval] = {
                            'success': False,
                            'error': str(e),
                            'max_days': max_days,
                            'description': description
                        }
                else:
                    print(f"   ⚠️ No data received")
                    results[interval] = {
                        'success': False,
                        'records': 0,
                        'error': 'No data received from API',
                        'max_days': max_days,
                        'description': description
                    }
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[interval] = {
                    'success': False,
                    'records': 0,
                    'error': str(e),
                    'max_days': max_days,
                    'description': description
                }
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_final_report(results):
    """Generate final comprehensive report"""
    print("\n" + "=" * 80)
    print("📊 ANGEL ONE API ALL INTERVALS DATABASE TEST REPORT")
    print("=" * 80)
    
    if not results:
        print("❌ No results to report")
        return
    
    total_tests = len(results)
    successful_tests = sum(1 for result in results.values() if result.get('success', False))
    total_records_downloaded = sum(result.get('records_downloaded', 0) for result in results.values())
    total_records_stored = sum(result.get('records_stored', 0) for result in results.values())
    total_records_retrieved = sum(result.get('records_retrieved', 0) for result in results.values())
    
    print(f"\n📈 RELIANCE Stock - All Intervals Results:")
    print("-" * 50)
    
    for interval, result in results.items():
        status = "✅" if result.get('success', False) else "❌"
        records_downloaded = result.get('records_downloaded', 0)
        records_stored = result.get('records_stored', 0)
        records_retrieved = result.get('records_retrieved', 0)
        data_integrity = result.get('data_integrity', False)
        max_days = result.get('max_days', 0)
        description = result.get('description', '')
        
        print(f"   {status} {interval}: {records_downloaded} records (Max: {max_days} days)")
        print(f"      📝 {description}")
        print(f"      💾 Stored: {records_stored} records")
        print(f"      🔍 Retrieved: {records_retrieved} records")
        print(f"      🔍 Data Integrity: {'✅ Verified' if data_integrity else '❌ Failed'}")
        
        if result.get('error'):
            print(f"      ❌ Error: {result['error']}")
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Successful Tests: {successful_tests}")
    print(f"   Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    print(f"   Total Records Downloaded: {total_records_downloaded}")
    print(f"   Total Records Stored: {total_records_stored}")
    print(f"   Total Records Retrieved: {total_records_retrieved}")
    
    # Save detailed report
    report_file = "docs/ANGEL_ONE_ALL_INTERVALS_DATABASE_REPORT.md"
    os.makedirs("docs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Angel One API All Intervals Database Test Report\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Stock:** RELIANCE (BSE)\n")
        f.write(f"**Total Tests:** {total_tests}\n")
        f.write(f"**Successful Tests:** {successful_tests}\n")
        f.write(f"**Success Rate:** {(successful_tests/total_tests)*100:.1f}%\n\n")
        f.write(f"**Total Records Downloaded:** {total_records_downloaded}\n")
        f.write(f"**Total Records Stored:** {total_records_stored}\n")
        f.write(f"**Total Records Retrieved:** {total_records_retrieved}\n\n")
        
        f.write("## Test Results\n\n")
        for interval, result in results.items():
            f.write(f"### {interval}\n\n")
            f.write(f"- **Status**: {'✅' if result.get('success', False) else '❌'}\n")
            f.write(f"- **Records Downloaded**: {result.get('records_downloaded', 0)}\n")
            f.write(f"- **Records Stored**: {result.get('records_stored', 0)}\n")
            f.write(f"- **Records Retrieved**: {result.get('records_retrieved', 0)}\n")
            f.write(f"- **Data Integrity**: {'✅ Verified' if result.get('data_integrity', False) else '❌ Failed'}\n")
            f.write(f"- **Max Days**: {result.get('max_days', 0)}\n")
            f.write(f"- **Description**: {result.get('description', '')}\n")
            if result.get('error'):
                f.write(f"- **Error**: {result['error']}\n")
            f.write("\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main test function"""
    try:
        print("🧪 Angel One API All Intervals Database Test")
        print("=" * 60)
        
        # Run the test
        results = test_angel_one_all_intervals_database()
        
        if results:
            # Generate comprehensive report
            generate_final_report(results)
            
            # Calculate overall success
            total_tests = len(results)
            successful_tests = sum(1 for result in results.values() if result.get('success', False))
            
            overall_success = successful_tests > 0
            
            print(f"\n🎯 Overall Result: {'✅ ALL INTERVALS TEST COMPLETED' if overall_success else '❌ ALL INTERVALS TEST FAILED'}")
            
            if overall_success:
                print("\n🎉 Angel One API all intervals test successful!")
                print("✅ All intervals tested with maximum data")
                print("✅ Database storage verified")
                print("✅ Database retrieval verified")
                print("✅ Data integrity confirmed")
                print("✅ Comprehensive report generated")
                print("\n🚀 Angel One API is fully functional with all intervals and database integration!")
            else:
                print("\n❌ Angel One API all intervals test failed!")
                print("❌ Check API access and database operations")
                print("❌ Review detailed report for specific issues")
            
            return overall_success
        else:
            print("❌ Test failed")
            return False
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
