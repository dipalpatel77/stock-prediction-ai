#!/usr/bin/env python3
"""
Simple SQLite Database Testing Script
Test database functionality using direct SQLite commands
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta

def test_sqlite_direct():
    """Test SQLite database directly"""
    print("🔍 Testing SQLite Database Directly...")
    
    # Create test database
    db_file = 'test_direct.db'
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        print("✅ Connected to SQLite database")
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("✅ Table created successfully")
        
        # Insert sample data
        sample_data = [
            ('TCS', '2024-01-01', 100.0, 105.0, 95.0, 102.0, 1000000),
            ('TCS', '2024-01-02', 102.0, 108.0, 98.0, 106.0, 1100000),
            ('TCS', '2024-01-03', 106.0, 110.0, 104.0, 108.0, 1200000),
        ]
        
        cursor.executemany("""
            INSERT INTO stock_data (ticker, date, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, sample_data)
        
        conn.commit()
        print("✅ Sample data inserted successfully")
        
        # Query data
        cursor.execute("SELECT * FROM stock_data WHERE ticker = ?", ('TCS',))
        results = cursor.fetchall()
        
        print(f"✅ Retrieved {len(results)} records")
        for row in results:
            print(f"   {row}")
        
        # Test performance
        print("\n📊 Performance Test:")
        
        # Insert more data for performance test
        start_time = datetime.now()
        
        for i in range(100):
            cursor.execute("""
                INSERT INTO stock_data (ticker, date, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (f'TEST{i}', f'2024-01-{i+1:02d}', 100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1000000 + i))
        
        conn.commit()
        end_time = datetime.now()
        
        print(f"⏱️ Inserted 100 records in {(end_time - start_time).total_seconds():.2f} seconds")
        
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM stock_data")
        count = cursor.fetchone()[0]
        print(f"📊 Total records in database: {count}")
        
        # Test date range query
        cursor.execute("""
            SELECT ticker, date, close_price 
            FROM stock_data 
            WHERE date >= '2024-01-01' 
            ORDER BY date DESC 
            LIMIT 5
        """)
        
        recent_data = cursor.fetchall()
        print(f"📊 Recent records: {len(recent_data)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()
        
        # Cleanup
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"🧹 Cleaned up {db_file}")

def test_database_file_exists():
    """Test if the main database file exists and is accessible"""
    print("\n🔍 Testing Main Database File...")
    
    db_files = [
        'data/stock_data.db',
        'main/data/stock_data.db',
        'stock_data.db'
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            print(f"✅ Found database file: {db_file}")
            
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check if angel_one_data table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='angel_one_data'
                """)
                
                if cursor.fetchone():
                    print(f"✅ angel_one_data table exists in {db_file}")
                    
                    # Count records
                    cursor.execute("SELECT COUNT(*) FROM angel_one_data")
                    count = cursor.fetchone()[0]
                    print(f"📊 Records in angel_one_data: {count}")
                    
                    # Show sample data
                    cursor.execute("SELECT ticker, date, close_price FROM angel_one_data LIMIT 5")
                    sample_data = cursor.fetchall()
                    print(f"📊 Sample data:")
                    for row in sample_data:
                        print(f"   {row}")
                    
                else:
                    print(f"⚠️ angel_one_data table not found in {db_file}")
                
                conn.close()
                
            except Exception as e:
                print(f"❌ Error accessing {db_file}: {e}")
        else:
            print(f"❌ Database file not found: {db_file}")

def main():
    """Run SQLite tests"""
    print("🚀 Starting SQLite Database Tests")
    print("=" * 50)
    
    # Test direct SQLite functionality
    test_sqlite_direct()
    
    # Test main database files
    test_database_file_exists()
    
    print("\n🎯 SQLite tests completed!")

if __name__ == "__main__":
    main()
