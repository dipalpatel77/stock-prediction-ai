#!/usr/bin/env python3
"""
MySQL Database Setup Script
Creates the stock_data database and all required tables
"""

import mysql.connector
from mysql.connector import Error
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database_config import get_database_config

def create_database_and_tables():
    """Create MySQL database and all required tables."""
    
    print("🗄️ MYSQL DATABASE SETUP")
    print("=" * 60)
    
    # Database connection parameters
    host = "localhost"
    user = "root"
    password = "7874"
    database = "stock_data"
    
    try:
        # Connect to MySQL server (without specifying database)
        print(f"🔌 Connecting to MySQL server at {host}...")
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        
        if connection.is_connected():
            print("✅ Connected to MySQL server successfully")
            
            cursor = connection.cursor()
            
            # Create database if it doesn't exist
            print(f"📁 Creating database '{database}' if it doesn't exist...")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"✅ Database '{database}' created/verified")
            
            # Use the database
            cursor.execute(f"USE {database}")
            print(f"✅ Using database '{database}'")
            
            # Create stock_data table
            print("📊 Creating stock_data table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open DECIMAL(10,4),
                    high DECIMAL(10,4),
                    low DECIMAL(10,4),
                    close DECIMAL(10,4),
                    volume BIGINT,
                    adj_close DECIMAL(10,4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_ticker_date (ticker, date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            print("✅ stock_data table created")
            
            # Create indexes for stock_data table
            print("🔍 Creating indexes for stock_data table...")
            try:
                cursor.execute("CREATE INDEX idx_ticker_date ON stock_data(ticker, date)")
            except Error as e:
                if "Duplicate key name" not in str(e):
                    print(f"Warning: {e}")
            
            try:
                cursor.execute("CREATE INDEX idx_ticker ON stock_data(ticker)")
            except Error as e:
                if "Duplicate key name" not in str(e):
                    print(f"Warning: {e}")
            
            try:
                cursor.execute("CREATE INDEX idx_date ON stock_data(date)")
            except Error as e:
                if "Duplicate key name" not in str(e):
                    print(f"Warning: {e}")
            
            print("✅ Indexes created")
            
            # Create stock_metadata table
            print("📋 Creating stock_metadata table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_metadata (
                    ticker VARCHAR(20) PRIMARY KEY,
                    name VARCHAR(200),
                    exchange VARCHAR(50),
                    currency VARCHAR(10),
                    country VARCHAR(50),
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    market_cap DECIMAL(20,2),
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    data_source VARCHAR(50),
                    total_records INT DEFAULT 0,
                    first_date DATE,
                    last_date DATE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            print("✅ stock_metadata table created")
            
            # Create data_quality table
            print("🔍 Creating data_quality table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    check_date DATE NOT NULL,
                    quality_score DECIMAL(3,2),
                    missing_ratio DECIMAL(3,2),
                    outlier_count INT,
                    price_change_anomaly BOOLEAN,
                    volume_anomaly BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_ticker_check_date (ticker, check_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            print("✅ data_quality table created")
            
            # Show table information
            print("\n📊 DATABASE STRUCTURE:")
            print("-" * 40)
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"  📋 {table_name}: {count} records")
            
            # Test the database connection with our service
            print("\n🧪 TESTING DATABASE SERVICE:")
            print("-" * 40)
            
            try:
                from core.database_service import DatabaseService
                
                # Test with MySQL configuration
                config = get_database_config("local")
                db_service = DatabaseService(config.db_type, config.connection_string)
                
                # Get database stats
                stats = db_service.get_database_stats()
                print(f"✅ Database service connected successfully")
                print(f"✅ Database type: {stats['database_type']}")
                print(f"✅ Total records: {stats['total_records']}")
                print(f"✅ Unique tickers: {stats['unique_tickers']}")
                
            except Exception as e:
                print(f"❌ Database service test failed: {e}")
            
            print("\n🎉 MYSQL DATABASE SETUP COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
    except Error as e:
        print(f"❌ MySQL Error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False
        
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("🔌 MySQL connection closed")
    
    return True

def show_usage_examples():
    """Show examples of how to use the MySQL database."""
    
    print("\n💡 MYSQL DATABASE USAGE EXAMPLES:")
    print("=" * 60)
    
    print("""
# Basic Database Operations:
from core.database_service import DatabaseService
from config.database_config import get_database_config

config = get_database_config("local")  # Uses MySQL
db_service = DatabaseService(config.db_type, config.connection_string)

# Store data
success = db_service.store_stock_data("AAPL", data, "yfinance")

# Retrieve data
data = db_service.get_stock_data("AAPL", start_date="2024-01-01")

# Get data info
info = db_service.get_data_info("AAPL")

# Incremental Updates:
from core.incremental_data_service import IncrementalDataService

service = IncrementalDataService(use_database=True)
data = service.get_incremental_data("AAPL", period="1y")

# Migration:
python migrate_to_database.py --preset local

# Direct MySQL Connection:
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="7874",
    database="stock_data"
)
""")

if __name__ == "__main__":
    try:
        success = create_database_and_tables()
        if success:
            show_usage_examples()
        else:
            print("\n❌ MySQL database setup failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
