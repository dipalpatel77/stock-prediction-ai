#!/usr/bin/env python3
"""
Database Service
High-performance database operations for stock data storage and retrieval
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import sqlite3
import threading
from pathlib import Path
import logging
import json
from contextlib import contextmanager
import time

# Optional imports for advanced databases
try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pymongo
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

class DatabaseService:
    """
    High-performance database service for stock data.
    Supports SQLite (default), MySQL, PostgreSQL, and MongoDB.
    """
    
    def __init__(self, db_type: str = "sqlite", connection_string: str = None):
        self.db_type = db_type.lower()
        self.connection_string = connection_string
        self.connection_pool = {}
        self.pool_lock = threading.Lock()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            if self.db_type == "sqlite":
                self._init_sqlite()
            elif self.db_type == "mysql":
                self._init_mysql()
            elif self.db_type == "postgresql":
                self._init_postgresql()
            elif self.db_type == "mongodb":
                self._init_mongodb()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            self.logger.info(f"Database initialized: {self.db_type}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _init_sqlite(self):
        """Initialize SQLite database."""
        if not self.connection_string:
            self.connection_string = "data/stock_data.db"
        
        # Ensure directory exists
        Path(self.connection_string).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            self._create_tables_sqlite(conn)
    
    def _init_mysql(self):
        """Initialize MySQL database."""
        if not self.connection_string:
            raise ValueError("MySQL connection string required")
        
        if not MYSQL_AVAILABLE:
            raise ImportError("mysql-connector-python not available. Install with: pip install mysql-connector-python")
        
        with self._get_connection() as conn:
            self._create_tables_mysql(conn)
    
    def _init_postgresql(self):
        """Initialize PostgreSQL database."""
        if not self.connection_string:
            raise ValueError("PostgreSQL connection string required")
        
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")
        
        with self._get_connection() as conn:
            self._create_tables_postgresql(conn)
    
    def _init_mongodb(self):
        """Initialize MongoDB database."""
        if not self.connection_string:
            self.connection_string = "mongodb://localhost:27017/stock_data"
        
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo not available. Install with: pip install pymongo")
        
        self._create_collections_mongodb()
    
    def _create_tables_sqlite(self, conn):
        """Create SQLite tables."""
        cursor = conn.cursor()
        
        # Stock data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)")
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_metadata (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                currency TEXT,
                country TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT,
                total_records INTEGER DEFAULT 0,
                first_date DATE,
                last_date DATE
            )
        """)
        
        # Data quality table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                check_date DATE NOT NULL,
                quality_score REAL,
                missing_ratio REAL,
                outlier_count INTEGER,
                price_change_anomaly BOOLEAN,
                volume_anomaly BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, check_date)
            )
        """)
        
        conn.commit()
    
    def _create_tables_mysql(self, conn):
        """Create MySQL tables."""
        cursor = conn.cursor()
        
        # Stock data table
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
        
        # Create indexes for performance
        try:
            cursor.execute("CREATE INDEX idx_ticker_date ON stock_data(ticker, date)")
        except Error as e:
            if "Duplicate key name" not in str(e):
                self.logger.warning(f"Index creation warning: {e}")
        
        try:
            cursor.execute("CREATE INDEX idx_ticker ON stock_data(ticker)")
        except Error as e:
            if "Duplicate key name" not in str(e):
                self.logger.warning(f"Index creation warning: {e}")
        
        try:
            cursor.execute("CREATE INDEX idx_date ON stock_data(date)")
        except Error as e:
            if "Duplicate key name" not in str(e):
                self.logger.warning(f"Index creation warning: {e}")
        
        # Metadata table
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
        
        # Data quality table
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
        
        conn.commit()
    
    def _create_tables_postgresql(self, conn):
        """Create PostgreSQL tables."""
        cursor = conn.cursor()
        
        # Stock data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(10,4),
                high DECIMAL(10,4),
                low DECIMAL(10,4),
                close DECIMAL(10,4),
                volume BIGINT,
                adj_close DECIMAL(10,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)")
        
        # Metadata table
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
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_source VARCHAR(50),
                total_records INTEGER DEFAULT 0,
                first_date DATE,
                last_date DATE
            )
        """)
        
        # Data quality table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                check_date DATE NOT NULL,
                quality_score DECIMAL(3,2),
                missing_ratio DECIMAL(3,2),
                outlier_count INTEGER,
                price_change_anomaly BOOLEAN,
                volume_anomaly BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, check_date)
            )
        """)
        
        conn.commit()
    
    def _create_collections_mongodb(self):
        """Create MongoDB collections."""
        client = MongoClient(self.connection_string)
        db = client.get_default_database()
        
        # Create collections with indexes
        stock_data = db.stock_data
        stock_data.create_index([("ticker", 1), ("date", 1)], unique=True)
        stock_data.create_index([("ticker", 1)])
        stock_data.create_index([("date", 1)])
        
        stock_metadata = db.stock_metadata
        stock_metadata.create_index([("ticker", 1)], unique=True)
        
        data_quality = db.data_quality
        data_quality.create_index([("ticker", 1), ("check_date", 1)], unique=True)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup."""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(self.connection_string, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        
        elif self.db_type == "mysql":
            # Parse MySQL connection string
            conn_params = self._parse_mysql_connection_string()
            conn = mysql.connector.connect(**conn_params)
            try:
                yield conn
            finally:
                conn.close()
        
        elif self.db_type == "postgresql":
            conn = psycopg2.connect(self.connection_string)
            try:
                yield conn
            finally:
                conn.close()
        
        elif self.db_type == "mongodb":
            client = MongoClient(self.connection_string)
            try:
                yield client.get_default_database()
            finally:
                client.close()
    
    def _parse_mysql_connection_string(self) -> dict:
        """Parse MySQL connection string into connection parameters."""
        # Parse connection string like: mysql://user:pass@host:port/database
        import re
        
        pattern = r'mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)'
        match = re.match(pattern, self.connection_string)
        
        if match:
            username, password, host, port, database = match.groups()
            return {
                'host': host,
                'port': int(port),
                'user': username,
                'password': password,
                'database': database,
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci',
                'autocommit': True
            }
        else:
            raise ValueError(f"Invalid MySQL connection string: {self.connection_string}")
    
    def store_stock_data(self, ticker: str, data: pd.DataFrame, 
                        data_source: str = "yfinance") -> bool:
        """
        Store stock data in database.
        
        Args:
            ticker: Stock ticker symbol
            data: DataFrame with stock data
            data_source: Source of the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if data.empty:
                self.logger.warning(f"No data to store for {ticker}")
                return False
            
            # Prepare data for storage
            data_to_store = self._prepare_data_for_storage(data, ticker)
            
            if self.db_type == "sqlite":
                return self._store_data_sqlite(ticker, data_to_store, data_source)
            elif self.db_type == "mysql":
                return self._store_data_mysql(ticker, data_to_store, data_source)
            elif self.db_type == "postgresql":
                return self._store_data_postgresql(ticker, data_to_store, data_source)
            elif self.db_type == "mongodb":
                return self._store_data_mongodb(ticker, data_to_store, data_source)
            
        except Exception as e:
            self.logger.error(f"Error storing data for {ticker}: {e}")
            return False
    
    def _prepare_data_for_storage(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare data for database storage."""
        # Reset index to make Date a column
        if data.index.name == 'Date' or 'Date' in str(data.index.dtype):
            data = data.reset_index()
        
        # Ensure Date column is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.date
        else:
            raise ValueError("Date column not found in data")
        
        # Add ticker column
        data['ticker'] = ticker
        
        # Rename columns to match database schema
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Select only required columns
        required_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        data = data[[col for col in required_columns if col in data.columns]]
        
        return data
    
    def _store_data_sqlite(self, ticker: str, data: pd.DataFrame, data_source: str) -> bool:
        """Store data in SQLite database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert or replace data
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO stock_data 
                    (ticker, date, open, high, low, close, volume, adj_close, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    row['ticker'], row['date'], row.get('open'), row.get('high'),
                    row.get('low'), row.get('close'), row.get('volume'), row.get('adj_close')
                ))
            
            # Update metadata
            self._update_metadata_sqlite(cursor, ticker, data, data_source)
            
            conn.commit()
            
        self.logger.info(f"Stored {len(data)} records for {ticker} in SQLite")
        return True
    
    def _store_data_mysql(self, ticker: str, data: pd.DataFrame, data_source: str) -> bool:
        """Store data in MySQL database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert or replace data
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT INTO stock_data 
                    (ticker, date, open, high, low, close, volume, adj_close, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON DUPLICATE KEY UPDATE
                        open = VALUES(open),
                        high = VALUES(high),
                        low = VALUES(low),
                        close = VALUES(close),
                        volume = VALUES(volume),
                        adj_close = VALUES(adj_close),
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    row['ticker'], row['date'], row.get('open'), row.get('high'),
                    row.get('low'), row.get('close'), row.get('volume'), row.get('adj_close')
                ))
            
            # Update metadata
            self._update_metadata_mysql(cursor, ticker, data, data_source)
            
            conn.commit()
            
        self.logger.info(f"Stored {len(data)} records for {ticker} in MySQL")
        return True
    
    def _store_data_postgresql(self, ticker: str, data: pd.DataFrame, data_source: str) -> bool:
        """Store data in PostgreSQL database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert or replace data
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT INTO stock_data 
                    (ticker, date, open, high, low, close, volume, adj_close, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (ticker, date) 
                    DO UPDATE SET 
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        adj_close = EXCLUDED.adj_close,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    row['ticker'], row['date'], row.get('open'), row.get('high'),
                    row.get('low'), row.get('close'), row.get('volume'), row.get('adj_close')
                ))
            
            # Update metadata
            self._update_metadata_postgresql(cursor, ticker, data, data_source)
            
            conn.commit()
            
        self.logger.info(f"Stored {len(data)} records for {ticker} in PostgreSQL")
        return True
    
    def _store_data_mongodb(self, ticker: str, data: pd.DataFrame, data_source: str) -> bool:
        """Store data in MongoDB."""
        with self._get_connection() as db:
            collection = db.stock_data
            
            # Convert DataFrame to documents
            documents = []
            for _, row in data.iterrows():
                doc = {
                    'ticker': row['ticker'],
                    'date': row['date'],
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume'),
                    'adj_close': row.get('adj_close'),
                    'updated_at': datetime.now()
                }
                documents.append(doc)
            
            # Bulk upsert
            for doc in documents:
                collection.replace_one(
                    {'ticker': doc['ticker'], 'date': doc['date']},
                    doc,
                    upsert=True
                )
            
            # Update metadata
            self._update_metadata_mongodb(db, ticker, data, data_source)
            
        self.logger.info(f"Stored {len(data)} records for {ticker} in MongoDB")
        return True
    
    def _update_metadata_sqlite(self, cursor, ticker: str, data: pd.DataFrame, data_source: str):
        """Update metadata in SQLite."""
        first_date = data['date'].min()
        last_date = data['date'].max()
        total_records = len(data)
        
        cursor.execute("""
            INSERT OR REPLACE INTO stock_metadata 
            (ticker, data_source, total_records, first_date, last_date, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (ticker, data_source, total_records, first_date, last_date))
    
    def _update_metadata_mysql(self, cursor, ticker: str, data: pd.DataFrame, data_source: str):
        """Update metadata in MySQL."""
        first_date = data['date'].min()
        last_date = data['date'].max()
        total_records = len(data)
        
        cursor.execute("""
            INSERT INTO stock_metadata 
            (ticker, data_source, total_records, first_date, last_date, last_updated)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE 
                data_source = VALUES(data_source),
                total_records = VALUES(total_records),
                first_date = VALUES(first_date),
                last_date = VALUES(last_date),
                last_updated = CURRENT_TIMESTAMP
        """, (ticker, data_source, total_records, first_date, last_date))
    
    def _update_metadata_postgresql(self, cursor, ticker: str, data: pd.DataFrame, data_source: str):
        """Update metadata in PostgreSQL."""
        first_date = data['date'].min()
        last_date = data['date'].max()
        total_records = len(data)
        
        cursor.execute("""
            INSERT INTO stock_metadata 
            (ticker, data_source, total_records, first_date, last_date, last_updated)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (ticker) 
            DO UPDATE SET 
                data_source = EXCLUDED.data_source,
                total_records = EXCLUDED.total_records,
                first_date = EXCLUDED.first_date,
                last_date = EXCLUDED.last_date,
                last_updated = CURRENT_TIMESTAMP
        """, (ticker, data_source, total_records, first_date, last_date))
    
    def _update_metadata_mongodb(self, db, ticker: str, data: pd.DataFrame, data_source: str):
        """Update metadata in MongoDB."""
        first_date = data['date'].min()
        last_date = data['date'].max()
        total_records = len(data)
        
        metadata = {
            'ticker': ticker,
            'data_source': data_source,
            'total_records': total_records,
            'first_date': first_date,
            'last_date': last_date,
            'last_updated': datetime.now()
        }
        
        db.stock_metadata.replace_one(
            {'ticker': ticker},
            metadata,
            upsert=True
        )
    
    def get_stock_data(self, ticker: str, start_date: str = None, 
                      end_date: str = None, limit: int = None) -> pd.DataFrame:
        """
        Retrieve stock data from database.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of records
            
        Returns:
            DataFrame with stock data
        """
        try:
            if self.db_type == "sqlite":
                return self._get_data_sqlite(ticker, start_date, end_date, limit)
            elif self.db_type == "mysql":
                return self._get_data_mysql(ticker, start_date, end_date, limit)
            elif self.db_type == "postgresql":
                return self._get_data_postgresql(ticker, start_date, end_date, limit)
            elif self.db_type == "mongodb":
                return self._get_data_mongodb(ticker, start_date, end_date, limit)
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _get_data_sqlite(self, ticker: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
        """Retrieve data from SQLite."""
        with self._get_connection() as conn:
            query = "SELECT * FROM stock_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.drop(['id', 'ticker', 'created_at', 'updated_at'], axis=1, errors='ignore')
            
            return df
    
    def _get_data_mysql(self, ticker: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
        """Retrieve data from MySQL database."""
        with self._get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            query = "SELECT * FROM stock_data WHERE ticker = %s"
            params = [ticker]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            
            query += " ORDER BY date DESC"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Convert decimal.Decimal columns to float
            df = self._convert_decimal_columns(df)
            
            # Remove database-specific columns
            columns_to_drop = ['id', 'created_at', 'updated_at']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
        self.logger.info(f"Retrieved {len(df)} records for {ticker} from MySQL")
        return df
    
    def _convert_decimal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert decimal.Decimal columns to float for compatibility."""
        try:
            from decimal import Decimal
            
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Check if column contains Decimal objects
                    sample_values = df[column].dropna().head(5)
                    if sample_values.any() and isinstance(sample_values.iloc[0], Decimal):
                        df[column] = df[column].astype(float)
                        self.logger.debug(f"Converted {column} from Decimal to float")
            
            return df
        except Exception as e:
            self.logger.warning(f"Error converting decimal columns: {e}")
            return df
    
    def _get_data_postgresql(self, ticker: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
        """Retrieve data from PostgreSQL."""
        with self._get_connection() as conn:
            query = "SELECT * FROM stock_data WHERE ticker = %s"
            params = [ticker]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.drop(['id', 'ticker', 'created_at', 'updated_at'], axis=1, errors='ignore')
                # Convert decimal.Decimal columns to float
                df = self._convert_decimal_columns(df)
            
            return df
    
    def _get_data_mongodb(self, ticker: str, start_date: str, end_date: str, limit: int) -> pd.DataFrame:
        """Retrieve data from MongoDB."""
        with self._get_connection() as db:
            collection = db.stock_data
            
            # Build query
            query = {'ticker': ticker}
            
            if start_date or end_date:
                date_query = {}
                if start_date:
                    date_query['$gte'] = datetime.strptime(start_date, '%Y-%m-%d')
                if end_date:
                    date_query['$lte'] = datetime.strptime(end_date, '%Y-%m-%d')
                query['date'] = date_query
            
            # Execute query
            cursor = collection.find(query).sort('date')
            
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to DataFrame
            data = list(cursor)
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.drop(['_id', 'ticker', 'updated_at'], axis=1, errors='ignore')
                return df
            else:
                return pd.DataFrame()
    
    def get_data_info(self, ticker: str) -> Dict:
        """Get information about stored data for a ticker."""
        try:
            if self.db_type == "sqlite":
                return self._get_info_sqlite(ticker)
            elif self.db_type == "mysql":
                return self._get_info_mysql(ticker)
            elif self.db_type == "postgresql":
                return self._get_info_postgresql(ticker)
            elif self.db_type == "mongodb":
                return self._get_info_mongodb(ticker)
            
        except Exception as e:
            self.logger.error(f"Error getting info for {ticker}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def _get_info_sqlite(self, ticker: str) -> Dict:
        """Get info from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get metadata
            cursor.execute("SELECT * FROM stock_metadata WHERE ticker = ?", (ticker,))
            metadata = cursor.fetchone()
            
            if metadata:
                return {
                    'exists': True,
                    'records': metadata['total_records'],
                    'first_date': metadata['first_date'],
                    'last_date': metadata['last_date'],
                    'data_source': metadata['data_source'],
                    'last_updated': metadata['last_updated']
                }
            else:
                return {'exists': False}
    
    def _get_info_mysql(self, ticker: str) -> Dict:
        """Get info from MySQL."""
        with self._get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # Check if data exists
            cursor.execute("SELECT COUNT(*) as count FROM stock_data WHERE ticker = %s", (ticker,))
            count_result = cursor.fetchone()
            count = count_result['count'] if count_result else 0
            
            if count == 0:
                return {'exists': False, 'records': 0}
            
            # Get metadata
            cursor.execute("""
                SELECT * FROM stock_metadata WHERE ticker = %s
            """, (ticker,))
            metadata = cursor.fetchone()
            
            if metadata:
                return {
                    'exists': True,
                    'records': metadata['total_records'],
                    'first_date': str(metadata['first_date']) if metadata['first_date'] else None,
                    'last_date': str(metadata['last_date']) if metadata['last_date'] else None,
                    'data_source': metadata['data_source'],
                    'last_updated': str(metadata['last_updated']) if metadata['last_updated'] else None
                }
            else:
                # Fallback: get info from stock_data table
                cursor.execute("""
                    SELECT 
                        COUNT(*) as records,
                        MIN(date) as first_date,
                        MAX(date) as last_date
                    FROM stock_data 
                    WHERE ticker = %s
                """, (ticker,))
                result = cursor.fetchone()
                
                return {
                    'exists': True,
                    'records': result['records'],
                    'first_date': str(result['first_date']) if result['first_date'] else None,
                    'last_date': str(result['last_date']) if result['last_date'] else None,
                    'data_source': 'unknown',
                    'last_updated': None
                }
    
    def _get_info_postgresql(self, ticker: str) -> Dict:
        """Get info from PostgreSQL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get metadata
            cursor.execute("SELECT * FROM stock_metadata WHERE ticker = %s", (ticker,))
            metadata = cursor.fetchone()
            
            if metadata:
                return {
                    'exists': True,
                    'records': metadata[2],  # total_records
                    'first_date': metadata[3],  # first_date
                    'last_date': metadata[4],  # last_date
                    'data_source': metadata[1],  # data_source
                    'last_updated': metadata[5]  # last_updated
                }
            else:
                return {'exists': False}
    
    def _get_info_mongodb(self, ticker: str) -> Dict:
        """Get info from MongoDB."""
        with self._get_connection() as db:
            metadata = db.stock_metadata.find_one({'ticker': ticker})
            
            if metadata:
                return {
                    'exists': True,
                    'records': metadata['total_records'],
                    'first_date': metadata['first_date'],
                    'last_date': metadata['last_date'],
                    'data_source': metadata['data_source'],
                    'last_updated': metadata['last_updated']
                }
            else:
                return {'exists': False}
    
    def cleanup_old_data(self, days_old: int = 365) -> int:
        """Clean up old data records."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.strftime('%Y-%m-%d')
            
            if self.db_type == "sqlite":
                return self._cleanup_sqlite(cutoff_str)
            elif self.db_type == "postgresql":
                return self._cleanup_postgresql(cutoff_str)
            elif self.db_type == "mongodb":
                return self._cleanup_mongodb(cutoff_date)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def _cleanup_sqlite(self, cutoff_date: str) -> int:
        """Clean up old data in SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stock_data WHERE date < ?", (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
    
    def _cleanup_postgresql(self, cutoff_date: str) -> int:
        """Clean up old data in PostgreSQL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stock_data WHERE date < %s", (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
    
    def _cleanup_mongodb(self, cutoff_date: datetime) -> int:
        """Clean up old data in MongoDB."""
        with self._get_connection() as db:
            result = db.stock_data.delete_many({'date': {'$lt': cutoff_date}})
            return result.deleted_count
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            if self.db_type == "sqlite":
                return self._get_stats_sqlite()
            elif self.db_type == "mysql":
                return self._get_stats_mysql()
            elif self.db_type == "postgresql":
                return self._get_stats_postgresql()
            elif self.db_type == "mongodb":
                return self._get_stats_mongodb()
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def _get_stats_sqlite(self) -> Dict:
        """Get SQLite statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM stock_data")
            total_records = cursor.fetchone()[0]
            
            # Get unique tickers
            cursor.execute("SELECT COUNT(DISTINCT ticker) FROM stock_data")
            unique_tickers = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM stock_data")
            date_range = cursor.fetchone()
            
            return {
                'database_type': 'SQLite',
                'total_records': total_records,
                'unique_tickers': unique_tickers,
                'date_range': {
                    'start': date_range[0],
                    'end': date_range[1]
                }
            }
    
    def _get_stats_mysql(self) -> Dict:
        """Get MySQL statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # Get total records
            cursor.execute("SELECT COUNT(*) as total FROM stock_data")
            total_result = cursor.fetchone()
            total_records = total_result['total'] if total_result else 0
            
            # Get unique tickers
            cursor.execute("SELECT COUNT(DISTINCT ticker) as unique_tickers FROM stock_data")
            tickers_result = cursor.fetchone()
            unique_tickers = tickers_result['unique_tickers'] if tickers_result else 0
            
            # Get date range
            cursor.execute("SELECT MIN(date) as start_date, MAX(date) as end_date FROM stock_data")
            date_result = cursor.fetchone()
            start_date = str(date_result['start_date']) if date_result and date_result['start_date'] else None
            end_date = str(date_result['end_date']) if date_result and date_result['end_date'] else None
            
            return {
                'database_type': 'MySQL',
                'total_records': total_records,
                'unique_tickers': unique_tickers,
                'date_range': {
                    'start': start_date,
                    'end': end_date
                }
            }
    
    def _get_stats_postgresql(self) -> Dict:
        """Get PostgreSQL statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM stock_data")
            total_records = cursor.fetchone()[0]
            
            # Get unique tickers
            cursor.execute("SELECT COUNT(DISTINCT ticker) FROM stock_data")
            unique_tickers = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM stock_data")
            date_range = cursor.fetchone()
            
            return {
                'database_type': 'PostgreSQL',
                'total_records': total_records,
                'unique_tickers': unique_tickers,
                'date_range': {
                    'start': date_range[0],
                    'end': date_range[1]
                }
            }
    
    def _get_stats_mongodb(self) -> Dict:
        """Get MongoDB statistics."""
        with self._get_connection() as db:
            # Get total records
            total_records = db.stock_data.count_documents({})
            
            # Get unique tickers
            unique_tickers = len(db.stock_data.distinct('ticker'))
            
            # Get date range
            pipeline = [
                {'$group': {
                    '_id': None,
                    'min_date': {'$min': '$date'},
                    'max_date': {'$max': '$date'}
                }}
            ]
            result = list(db.stock_data.aggregate(pipeline))
            
            date_range = result[0] if result else {'min_date': None, 'max_date': None}
            
            return {
                'database_type': 'MongoDB',
                'total_records': total_records,
                'unique_tickers': unique_tickers,
                'date_range': {
                    'start': date_range['min_date'],
                    'end': date_range['max_date']
                }
            }

# Example usage and testing
if __name__ == "__main__":
    # Test the database service
    print("Testing Database Service...")
    
    # Test SQLite (default)
    db_service = DatabaseService("sqlite")
    
    # Test with sample data
    import yfinance as yf
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    data = stock.history(period="1mo")
    
    if not data.empty:
        # Store data
        success = db_service.store_stock_data(ticker, data, "yfinance")
        print(f"Data storage: {'Success' if success else 'Failed'}")
        
        # Retrieve data
        retrieved_data = db_service.get_stock_data(ticker)
        print(f"Retrieved {len(retrieved_data)} records")
        
        # Get info
        info = db_service.get_data_info(ticker)
        print(f"Data info: {info}")
        
        # Get stats
        stats = db_service.get_database_stats()
        print(f"Database stats: {stats}")
    
    print("Database service test completed!")
