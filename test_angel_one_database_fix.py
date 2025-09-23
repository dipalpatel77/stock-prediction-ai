#!/usr/bin/env python3
"""
Angel One API Database Fix Test
Tests Angel One API with simplified database operations
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_angel_one_database_fix():
    """Test Angel One API with simplified database operations"""
    print("🧪 Angel One API Database Fix Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        import sqlite3
        
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
        
        # Test with ONE_DAY interval for RELIANCE
        print(f"\n📈 Testing RELIANCE Stock - ONE_DAY interval...")
        
        try:
            # Download data
            print(f"   📥 Downloading data...")
            data = downloader.get_historical_data(
                symbol_name='RELIANCE',
                exchange='BSE',
                interval='ONE_DAY',
                days_back=7  # Small dataset for testing
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
                
                # Test direct database storage
                print(f"   💾 Testing direct database storage...")
                try:
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
                    
                    # Insert data
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
                            'ONE_DAY'
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
                    ''', ('RELIANCE', 'ONE_DAY'))
                    
                    retrieved_data = cursor.fetchall()
                    print(f"   ✅ Retrieved {len(retrieved_data)} records from database")
                    
                    # Show retrieved data
                    print(f"   📈 Retrieved data (first 2 records):")
                    for i, record in enumerate(retrieved_data[:2]):
                        print(f"      {record[0]}: Open={record[1]}, High={record[2]}, Low={record[3]}, Close={record[4]}, Volume={record[5]}")
                    
                    # Verify data integrity
                    data_match = len(data) == len(retrieved_data)
                    print(f"   🔍 Data integrity check: {'✅ Match' if data_match else '❌ Mismatch'}")
                    
                    conn.close()
                    
                    return {
                        'success': True,
                        'records_downloaded': len(data),
                        'records_stored': records_inserted,
                        'records_retrieved': len(retrieved_data),
                        'data_integrity': data_match
                    }
                    
                except Exception as e:
                    print(f"   ❌ Database operation failed: {e}")
                    return {
                        'success': False,
                        'error': str(e)
                    }
            else:
                print(f"   ❌ No data received")
                return {
                    'success': False,
                    'error': 'No data received from API'
                }
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    try:
        print("🧪 Angel One API Database Fix Test")
        print("=" * 50)
        
        # Run the test
        result = test_angel_one_database_fix()
        
        if result:
            print(f"\n🎯 Test Result: {'✅ SUCCESS' if result.get('success', False) else '❌ FAILED'}")
            
            if result.get('success', False):
                print("\n🎉 Angel One API database test successful!")
                print(f"✅ Records downloaded: {result.get('records_downloaded', 0)}")
                print(f"✅ Records stored: {result.get('records_stored', 0)}")
                print(f"✅ Records retrieved: {result.get('records_retrieved', 0)}")
                print(f"✅ Data integrity: {'✅ Verified' if result.get('data_integrity', False) else '❌ Failed'}")
                print("\n🚀 Angel One API is working with database storage!")
            else:
                print(f"\n❌ Angel One API database test failed!")
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            return result.get('success', False)
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
