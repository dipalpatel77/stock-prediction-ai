#!/usr/bin/env python3
"""
Database Browser
Simple command-line tool to browse and test the database
"""

import sqlite3
import os
import sys
from datetime import datetime

class DatabaseBrowser:
    """Simple database browser for testing"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            print("✅ Disconnected from database")
    
    def list_tables(self):
        """List all tables in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            print(f"\n📊 Tables in database:")
            for table in tables:
                print(f"   - {table[0]}")
            
            return [table[0] for table in tables]
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            return []
    
    def describe_table(self, table_name):
        """Describe table structure"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print(f"\n📊 Table structure for '{table_name}':")
            print("   Column Name    | Type    | Not Null | Default")
            print("   " + "-" * 50)
            for col in columns:
                col_id, name, type_, not_null, default, pk = col
                not_null_str = "YES" if not_null else "NO"
                default_str = str(default) if default else "NULL"
                print(f"   {name:<15} | {type_:<7} | {not_null_str:<7} | {default_str}")
            
            return columns
        except Exception as e:
            print(f"❌ Error describing table: {e}")
            return []
    
    def count_records(self, table_name):
        """Count records in table"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"📊 Records in '{table_name}': {count}")
            return count
        except Exception as e:
            print(f"❌ Error counting records: {e}")
            return 0
    
    def show_sample_data(self, table_name, limit=10):
        """Show sample data from table"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            rows = cursor.fetchall()
            
            if rows:
                print(f"\n📊 Sample data from '{table_name}' (first {len(rows)} records):")
                for i, row in enumerate(rows, 1):
                    print(f"   {i}: {row}")
            else:
                print(f"📊 No data found in '{table_name}'")
            
            return rows
        except Exception as e:
            print(f"❌ Error showing sample data: {e}")
            return []
    
    def run_query(self, query):
        """Run custom SQL query"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            print(f"\n📊 Query results ({len(results)} rows):")
            for row in results:
                print(f"   {row}")
            
            return results
        except Exception as e:
            print(f"❌ Query error: {e}")
            return []

def main():
    """Main function"""
    print("🚀 Database Browser")
    print("=" * 50)
    
    # Look for database files
    db_files = [
        'data/stock_data.db',
        'main/data/stock_data.db',
        'stock_data.db',
        'test_stock_data.db'
    ]
    
    found_dbs = []
    for db_file in db_files:
        if os.path.exists(db_file):
            found_dbs.append(db_file)
    
    if not found_dbs:
        print("❌ No database files found!")
        print("Available locations checked:")
        for db_file in db_files:
            print(f"   - {db_file}")
        return
    
    print("📊 Found database files:")
    for i, db_file in enumerate(found_dbs, 1):
        print(f"   {i}. {db_file}")
    
    # Use first database found
    db_path = found_dbs[0]
    print(f"\n🔍 Using database: {db_path}")
    
    # Create browser
    browser = DatabaseBrowser(db_path)
    
    if not browser.connect():
        return
    
    try:
        # List tables
        tables = browser.list_tables()
        
        if not tables:
            print("❌ No tables found in database")
            return
        
        # Focus on angel_one_data table if it exists
        if 'angel_one_data' in tables:
            print(f"\n🎯 Focusing on 'angel_one_data' table:")
            browser.describe_table('angel_one_data')
            browser.count_records('angel_one_data')
            browser.show_sample_data('angel_one_data', 5)
            
            # Show recent data
            print(f"\n📊 Recent data from 'angel_one_data':")
            browser.run_query("""
                SELECT ticker, date, close_price, volume 
                FROM angel_one_data 
                ORDER BY date DESC 
                LIMIT 10
            """)
            
            # Show data by ticker
            print(f"\n📊 Data by ticker:")
            browser.run_query("""
                SELECT ticker, COUNT(*) as record_count, 
                       MIN(date) as earliest_date, 
                       MAX(date) as latest_date
                FROM angel_one_data 
                GROUP BY ticker 
                ORDER BY record_count DESC
            """)
        
        # Show all tables info
        print(f"\n📊 All tables summary:")
        for table in tables:
            browser.count_records(table)
    
    finally:
        browser.disconnect()

if __name__ == "__main__":
    main()
