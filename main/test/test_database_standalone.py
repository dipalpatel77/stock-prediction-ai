#!/usr/bin/env python3
"""
Standalone Database Testing Script
Test the database functionality independently of the main application
"""

import sqlite3
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_database_connection():
    """Test basic database connection"""
    print("🔍 Testing Database Connection...")
    
    try:
        # Test SQLite connection
        conn = sqlite3.connect('test_standalone.db')
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        
        if result and result[0] == 1:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_table_creation():
    """Test table creation"""
    print("\n🔍 Testing Table Creation...")
    
    try:
        conn = sqlite3.connect('test_standalone.db')
        cursor = conn.cursor()
        
        # Create test table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS test_angel_one_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open_price DECIMAL(10,2),
            high_price DECIMAL(10,2),
            low_price DECIMAL(10,2),
            close_price DECIMAL(10,2),
            volume BIGINT,
            interval_type VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_sql)
        conn.commit()
        
        # Verify table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_angel_one_data'")
        result = cursor.fetchone()
        
        if result:
            print("✅ Table creation successful")
            return True
        else:
            print("❌ Table creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Table creation error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_data_insertion():
    """Test data insertion performance"""
    print("\n🔍 Testing Data Insertion Performance...")
    
    try:
        conn = sqlite3.connect('test_standalone.db')
        cursor = conn.cursor()
        
        # Create sample data
        sample_data = []
        base_date = datetime.now() - timedelta(days=100)
        
        for i in range(100):
            date = base_date + timedelta(days=i)
            sample_data.append((
                'TEST',
                date.strftime('%Y-%m-%d'),
                100.0 + i,
                105.0 + i,
                95.0 + i,
                102.0 + i,
                1000000 + i * 1000,
                '1d',
                datetime.now()
            ))
        
        # Test row-by-row insertion (old method)
        print("📊 Testing row-by-row insertion...")
        start_time = time.time()
        
        insert_sql = """
        INSERT INTO test_angel_one_data 
        (ticker, date, open_price, high_price, low_price, close_price, volume, interval_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for row in sample_data:
            cursor.execute(insert_sql, row)
        
        conn.commit()
        row_time = time.time() - start_time
        print(f"⏱️ Row-by-row insertion: {row_time:.2f} seconds")
        
        # Clear table for batch test
        cursor.execute("DELETE FROM test_angel_one_data")
        conn.commit()
        
        # Test batch insertion (new method)
        print("📊 Testing batch insertion...")
        start_time = time.time()
        
        cursor.executemany(insert_sql, sample_data)
        conn.commit()
        
        batch_time = time.time() - start_time
        print(f"⏱️ Batch insertion: {batch_time:.2f} seconds")
        
        # Calculate performance improvement
        improvement = row_time / batch_time if batch_time > 0 else 0
        print(f"🚀 Performance improvement: {improvement:.1f}x faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Data insertion error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_pandas_optimization():
    """Test pandas to_sql optimization"""
    print("\n🔍 Testing Pandas to_sql Optimization...")
    
    try:
        # Create sample DataFrame
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        sample_df = pd.DataFrame({
            'ticker': ['TEST'] * 100,
            'date': dates.strftime('%Y-%m-%d'),
            'open_price': [100.0 + i for i in range(100)],
            'high_price': [105.0 + i for i in range(100)],
            'low_price': [95.0 + i for i in range(100)],
            'close_price': [102.0 + i for i in range(100)],
            'volume': [1000000 + i * 1000 for i in range(100)],
            'interval_type': ['1d'] * 100,
            'created_at': [datetime.now()] * 100
        })
        
        # Test pandas to_sql
        print("📊 Testing pandas to_sql method...")
        start_time = time.time()
        
        conn = sqlite3.connect('test_standalone.db')
        sample_df.to_sql('test_pandas_data', conn, if_exists='replace', index=False)
        conn.close()
        
        pandas_time = time.time() - start_time
        print(f"⏱️ Pandas to_sql: {pandas_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Pandas optimization error: {e}")
        return False

def test_data_retrieval():
    """Test data retrieval"""
    print("\n🔍 Testing Data Retrieval...")
    
    try:
        conn = sqlite3.connect('test_standalone.db')
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT COUNT(*) FROM test_angel_one_data")
        count = cursor.fetchone()[0]
        print(f"📊 Total records: {count}")
        
        # Test date range query
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT ticker, date, close_price 
            FROM test_angel_one_data 
            WHERE date >= ? 
            ORDER BY date DESC 
            LIMIT 10
        """, (start_date,))
        
        results = cursor.fetchall()
        print(f"📊 Recent records (last 30 days): {len(results)}")
        
        if results:
            print("✅ Data retrieval successful")
            return True
        else:
            print("❌ No data found")
            return False
            
    except Exception as e:
        print(f"❌ Data retrieval error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_database_cleanup():
    """Test database cleanup"""
    print("\n🔍 Testing Database Cleanup...")
    
    try:
        conn = sqlite3.connect('test_standalone.db')
        cursor = conn.cursor()
        
        # Test cleanup with old date
        old_date = datetime.now() - timedelta(days=1)
        cursor.execute("DELETE FROM test_angel_one_data WHERE created_at < ?", (old_date,))
        deleted_rows = cursor.rowcount
        
        conn.commit()
        print(f"✅ Cleaned up {deleted_rows} old records")
        
        return True
        
    except Exception as e:
        print(f"❌ Database cleanup error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_database_manager_integration():
    """Test the actual database manager"""
    print("\n🔍 Testing Database Manager Integration...")
    
    try:
        from main.services.database_manager import DatabaseManager
        
        # Initialize database manager
        config = {
            'database_url': 'sqlite:///test_standalone.db',
            'max_connections': 5,
            'min_connections': 1,
            'connection_timeout': 30,
            'query_timeout': 30,
            'enable_query_cache': True,
            'cache_ttl': 300
        }
        
        db_manager = DatabaseManager(config)
        
        # Test connection
        if db_manager.test_connection():
            print("✅ Database manager connection successful")
        else:
            print("❌ Database manager connection failed")
            return False
        
        # Test storing data
        sample_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [102.0, 103.0, 104.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        success = db_manager.store_stock_data(
            ticker='TEST',
            data=sample_data,
            source='angel_one',
            interval='1d'
        )
        
        if success:
            print("✅ Database manager data storage successful")
        else:
            print("❌ Database manager data storage failed")
            return False
        
        # Test retrieving data
        retrieved_data = db_manager.get_stock_data(
            ticker='TEST',
            period='1y',
            source='angel_one',
            interval='1d'
        )
        
        if retrieved_data is not None and not retrieved_data.empty:
            print(f"✅ Database manager data retrieval successful: {len(retrieved_data)} records")
        else:
            print("❌ Database manager data retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database manager integration error: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    test_files = [
        'test_standalone.db',
        'test_pandas_data.db'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Removed {file}")

def main():
    """Run all database tests"""
    print("🚀 Starting Standalone Database Tests")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Table Creation", test_table_creation),
        ("Data Insertion Performance", test_data_insertion),
        ("Pandas Optimization", test_pandas_optimization),
        ("Data Retrieval", test_data_retrieval),
        ("Database Cleanup", test_database_cleanup),
        ("Database Manager Integration", test_database_manager_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All database tests passed! Database is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    # Cleanup
    cleanup_test_files()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
