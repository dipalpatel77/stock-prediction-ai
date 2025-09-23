#!/usr/bin/env python3
"""
Enhanced database schema for Angel One data storage
Optimized for maximum efficiency and prediction accuracy
"""

import mysql.connector
from datetime import datetime
from typing import Dict, List, Optional
import logging
import pandas as pd

class AngelOneDatabaseSchema:
    """
    Enhanced database schema for Angel One data storage.
    Optimized for maximum efficiency and prediction accuracy.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
    
    def create_enhanced_schema(self):
        """Create enhanced database schema for Angel One data."""
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="7874",
                database="stock_data"
            )
            cursor = conn.cursor()
            
            # 1. Enhanced stock_data table with Angel One specific fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS angel_one_stock_data (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    exchange VARCHAR(10) NOT NULL,
                    symbol_token VARCHAR(20) NOT NULL,
                    date DATETIME NOT NULL,
                    open DECIMAL(15,4) NOT NULL,
                    high DECIMAL(15,4) NOT NULL,
                    low DECIMAL(15,4) NOT NULL,
                    close DECIMAL(15,4) NOT NULL,
                    volume BIGINT NOT NULL,
                    interval_type VARCHAR(20) NOT NULL DEFAULT 'ONE_DAY',
                    data_source VARCHAR(20) NOT NULL DEFAULT 'angel_one',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    
                    UNIQUE KEY unique_data (ticker, exchange, date, interval_type),
                    INDEX idx_ticker_date (ticker, date),
                    INDEX idx_exchange_date (exchange, date),
                    INDEX idx_symbol_token (symbol_token),
                    INDEX idx_interval (interval_type),
                    INDEX idx_data_source (data_source)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 2. Open Interest data table for F&O contracts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS angel_one_oi_data (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    exchange VARCHAR(10) NOT NULL,
                    symbol_token VARCHAR(20) NOT NULL,
                    date DATETIME NOT NULL,
                    open_interest BIGINT NOT NULL,
                    interval_type VARCHAR(20) NOT NULL DEFAULT 'THREE_MINUTE',
                    data_source VARCHAR(20) NOT NULL DEFAULT 'angel_one',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    
                    UNIQUE KEY unique_oi_data (ticker, exchange, date, interval_type),
                    INDEX idx_ticker_date (ticker, date),
                    INDEX idx_exchange_date (exchange, date),
                    INDEX idx_symbol_token (symbol_token),
                    INDEX idx_interval (interval_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 3. Enhanced metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS angel_one_metadata (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    exchange VARCHAR(10) NOT NULL,
                    symbol_token VARCHAR(20) NOT NULL,
                    symbol_name VARCHAR(100) NOT NULL,
                    instrument_type VARCHAR(20) NOT NULL,
                    segment VARCHAR(20) NOT NULL,
                    lot_size INT NOT NULL,
                    tick_size DECIMAL(10,4) NOT NULL,
                    expiry_date DATE NULL,
                    strike_price DECIMAL(15,4) NULL,
                    option_type VARCHAR(5) NULL,
                    data_source VARCHAR(20) NOT NULL DEFAULT 'angel_one',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    
                    UNIQUE KEY unique_metadata (ticker, exchange, symbol_token),
                    INDEX idx_ticker (ticker),
                    INDEX idx_exchange (exchange),
                    INDEX idx_symbol_token (symbol_token),
                    INDEX idx_instrument_type (instrument_type),
                    INDEX idx_segment (segment)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 4. Data quality metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS angel_one_data_quality (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    exchange VARCHAR(10) NOT NULL,
                    date DATE NOT NULL,
                    total_records INT NOT NULL,
                    missing_records INT NOT NULL,
                    data_completeness DECIMAL(5,2) NOT NULL,
                    price_volatility DECIMAL(10,4) NOT NULL,
                    volume_anomaly_score DECIMAL(5,2) NOT NULL,
                    data_quality_score DECIMAL(5,2) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE KEY unique_quality (ticker, exchange, date),
                    INDEX idx_ticker_date (ticker, date),
                    INDEX idx_quality_score (data_quality_score),
                    INDEX idx_completeness (data_completeness)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # 5. Prediction accuracy tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS angel_one_prediction_accuracy (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    exchange VARCHAR(10) NOT NULL,
                    prediction_date DATE NOT NULL,
                    actual_date DATE NOT NULL,
                    predicted_price DECIMAL(15,4) NOT NULL,
                    actual_price DECIMAL(15,4) NOT NULL,
                    prediction_error DECIMAL(10,4) NOT NULL,
                    prediction_accuracy DECIMAL(5,2) NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    time_horizon VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_ticker_prediction (ticker, prediction_date),
                    INDEX idx_accuracy (prediction_accuracy),
                    INDEX idx_model (model_name),
                    INDEX idx_time_horizon (time_horizon)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            conn.commit()
            self.logger.info("✅ Enhanced Angel One database schema created successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating enhanced schema: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_optimized_data_query(self, ticker: str, exchange: str, 
                                start_date: str, end_date: str, 
                                interval: str = 'ONE_DAY') -> str:
        """Get optimized query for fetching Angel One data."""
        return f"""
            SELECT 
                date,
                open,
                high,
                low,
                close,
                volume,
                interval_type,
                data_source
            FROM angel_one_stock_data 
            WHERE ticker = '{ticker}'
            AND exchange = '{exchange}'
            AND interval_type = '{interval}'
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date ASC
        """
    
    def get_data_quality_metrics(self, ticker: str, exchange: str) -> Dict:
        """Get data quality metrics for a ticker."""
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="7874",
                database="stock_data"
            )
            cursor = conn.cursor(dictionary=True)
            
            # Get data completeness
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_records,
                    AVG(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL THEN 1 ELSE 0 END) * 100 as missing_percentage,
                    MIN(date) as first_date,
                    MAX(date) as last_date
                FROM angel_one_stock_data 
                WHERE ticker = '{ticker}' AND exchange = '{exchange}'
            """)
            
            quality_metrics = cursor.fetchone()
            
            # Get price volatility
            cursor.execute(f"""
                SELECT 
                    STDDEV(close) as price_volatility,
                    AVG(volume) as avg_volume,
                    STDDEV(volume) as volume_volatility
                FROM angel_one_stock_data 
                WHERE ticker = '{ticker}' AND exchange = '{exchange}'
                AND date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            
            volatility_metrics = cursor.fetchone()
            
            return {
                'total_records': quality_metrics['total_records'],
                'missing_percentage': self._safe_float_conversion(quality_metrics['missing_percentage']),
                'first_date': quality_metrics['first_date'],
                'last_date': quality_metrics['last_date'],
                'price_volatility': self._safe_float_conversion(volatility_metrics['price_volatility']),
                'avg_volume': self._safe_float_conversion(volatility_metrics['avg_volume']),
                'volume_volatility': self._safe_float_conversion(volatility_metrics['volume_volatility'])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting data quality metrics: {e}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()
    
    def store_stock_data(self, ticker: str, data: pd.DataFrame, interval: str = 'ONE_DAY', exchange: str = 'NSE') -> bool:
        """
        Store stock data in Angel One database schema
        
        Args:
            ticker: Stock ticker symbol
            data: Stock data DataFrame
            interval: Data interval
            exchange: Exchange (NSE/BSE)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if data.empty:
                self.logger.warning(f"No data to store for {ticker}")
                return False
            
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="7874",
                database="stock_data"
            )
            cursor = conn.cursor()
            
            # Prepare data for insertion
            data_to_insert = []
            for _, row in data.iterrows():
                data_to_insert.append((
                    ticker,
                    exchange,
                    row.get('symbol_token', ''),
                    row.name,  # date index
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']),
                    interval,
                    'angel_one'
                ))
            
            # Insert data with conflict resolution
            insert_query = """
                INSERT INTO angel_one_stock_data 
                (ticker, exchange, symbol_token, date, open, high, low, close, volume, interval_type, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume),
                updated_at = CURRENT_TIMESTAMP
            """
            
            cursor.executemany(insert_query, data_to_insert)
            conn.commit()
            
            self.logger.info(f"✅ Stored {len(data_to_insert)} records for {ticker} in Angel One schema")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing stock data for {ticker}: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_stock_data(self, ticker: str, interval: str = 'ONE_DAY', days: int = 30, exchange: str = 'NSE') -> Optional[pd.DataFrame]:
        """
        Get stock data from Angel One database schema
        
        Args:
            ticker: Stock ticker symbol
            interval: Data interval
            days: Number of days to retrieve
            exchange: Exchange (NSE/BSE)
            
        Returns:
            DataFrame with stock data or None if not found
        """
        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="7874",
                database="stock_data"
            )
            cursor = conn.cursor()
            
            # Get data from database
            query = """
                SELECT date, open, high, low, close, volume, symbol_token, interval_type, data_source
                FROM angel_one_stock_data 
                WHERE ticker = %s AND exchange = %s AND interval_type = %s
                AND date >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY date ASC
            """
            
            cursor.execute(query, (ticker, exchange, interval, days))
            results = cursor.fetchall()
            
            if not results:
                self.logger.warning(f"No data found for {ticker} in Angel One schema")
                return None
            
            # Convert to DataFrame
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol_token', 'interval_type', 'data_source']
            df = pd.DataFrame(results, columns=columns)
            df.set_index('date', inplace=True)
            
            self.logger.info(f"✅ Retrieved {len(df)} records for {ticker} from Angel One schema")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting stock data for {ticker}: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _safe_float_conversion(self, value) -> float:
        """Safely convert various numeric types to float."""
        try:
            if value is None:
                return 0.0
            
            # Handle decimal.Decimal
            from decimal import Decimal
            if isinstance(value, Decimal):
                return float(value)
            
            # Handle other numeric types
            return float(value)
        except (ValueError, TypeError):
            return 0.0

def main():
    """Create enhanced database schema."""
    try:
        print("🚀 Creating enhanced Angel One database schema...")
        
        schema = AngelOneDatabaseSchema("mysql://root:7874@localhost/stock_data")
        schema.create_enhanced_schema()
        
        print("✅ Enhanced Angel One database schema created successfully!")
        print("\n📊 Schema includes:")
        print("   • angel_one_stock_data - Optimized stock data storage")
        print("   • angel_one_oi_data - Open Interest data for F&O")
        print("   • angel_one_metadata - Enhanced symbol metadata")
        print("   • angel_one_data_quality - Data quality metrics")
        print("   • angel_one_prediction_accuracy - Prediction tracking")
        
    except Exception as e:
        print(f"❌ Error creating schema: {e}")

if __name__ == "__main__":
    main()
