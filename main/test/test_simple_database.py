#!/usr/bin/env python3
"""
Simple Database Test
Test basic database functionality without the complex pool
"""

import sqlite3
import pandas as pd
from datetime import datetime

def test_simple_database():
    """Test simple database operations"""
    try:
        # Create a simple SQLite connection
        conn = sqlite3.connect('simple_test.db')
        cursor = conn.cursor()
        
        # Create table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS angel_one_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            open_price DECIMAL(10,2),
            high_price DECIMAL(10,2),
            low_price DECIMAL(10,2),
            close_price DECIMAL(10,2),
            volume BIGINT,
            interval_type VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date, interval_type)
        )
        """
        cursor.execute(create_table_sql)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [95.0, 96.0, 97.0],
            'Close': [102.0, 103.0, 104.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        
        # Insert data
        insert_sql = """
        INSERT OR REPLACE INTO angel_one_data (ticker, date, open_price, high_price, low_price, close_price, volume, interval_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for date, row in sample_data.iterrows():
            cursor.execute(insert_sql, (
                'TEST',
                date.strftime('%Y-%m-%d'),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume']),
                'ONE_DAY'
            ))
        
        conn.commit()
        print("✅ Data inserted successfully")
        
        # Query data
        query_sql = """
        SELECT date, open_price, high_price, low_price, close_price, volume
        FROM angel_one_data
        WHERE ticker = ? AND interval_type = ?
        ORDER BY date ASC
        """
        
        cursor.execute(query_sql, ('TEST', 'ONE_DAY'))
        results = cursor.fetchall()
        
        print(f"✅ Retrieved {len(results)} records")
        for row in results:
            print(f"  Date: {row[0]}, Open: {row[1]}, Close: {row[4]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    success = test_simple_database()
    if success:
        print("🎉 Simple database test passed!")
    else:
        print("💥 Simple database test failed!")
