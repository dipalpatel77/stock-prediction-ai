#!/usr/bin/env python3
"""
Enhanced Angel One Data Service
Optimized for maximum efficiency and prediction accuracy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import mysql.connector
from src.utils.angel_one_data_downloader import AngelOneDataDownloader
from src.utils.angel_one_config import AngelOneConfig
from src.utils.indian_stock_mapper import get_symbol_info

class EnhancedAngelOneService:
    """
    Enhanced Angel One data service optimized for maximum efficiency.
    Uses official API documentation for optimal data fetching.
    """
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.config = AngelOneConfig()
        self.downloader = AngelOneDataDownloader()
        
        # Database connection
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '7874',
            'database': 'stock_data'
        }
        
        # Interval to max days mapping (from official documentation)
        self.max_days_by_interval = {
            'ONE_MINUTE': 30,
            'THREE_MINUTE': 60,
            'FIVE_MINUTE': 100,
            'TEN_MINUTE': 100,
            'FIFTEEN_MINUTE': 200,
            'THIRTY_MINUTE': 200,
            'ONE_HOUR': 400,
            'ONE_DAY': 2000
        }
    
    def get_optimal_historical_data(self, ticker: str, exchange: str = "NSE", 
                                  interval: str = "ONE_DAY", 
                                  days_back: int = None) -> pd.DataFrame:
        """
        Get optimal historical data using Angel One API.
        Automatically optimizes for maximum data retrieval.
        """
        try:
            # Get symbol info
            symbol_info = get_symbol_info(ticker)
            if not symbol_info:
                raise Exception(f"Symbol info not found for {ticker}")
            
            # Determine optimal days_back
            if days_back is None:
                days_back = self.max_days_by_interval.get(interval, 30)
            
            # Ensure we don't exceed API limits
            max_days = self.max_days_by_interval.get(interval, 30)
            if days_back > max_days:
                self.logger.warning(f"Requested {days_back} days, but max for {interval} is {max_days}")
                days_back = max_days
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates as required by API
            from_date = start_date.strftime('%Y-%m-%d %H:%M')
            to_date = end_date.strftime('%Y-%m-%d %H:%M')
            
            self.logger.info(f"📊 Fetching optimal data for {ticker}")
            self.logger.info(f"   Exchange: {exchange}")
            self.logger.info(f"   Interval: {interval}")
            self.logger.info(f"   Period: {from_date} to {to_date}")
            self.logger.info(f"   Max days: {max_days}")
            
            # Get data from Angel One
            df = self.downloader.get_historical_data(
                symbol_name=ticker,
                exchange=exchange,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                days_back=days_back
            )
            
            self.logger.info(f"🔍 Data received: {type(df)}, Empty: {df.empty if df is not None else 'None'}")
            if df is not None:
                self.logger.info(f"🔍 DataFrame shape: {df.shape}")
                self.logger.info(f"🔍 DataFrame columns: {list(df.columns)}")
            
            if df is not None and not df.empty:
                # Add metadata
                df['ticker'] = ticker
                df['exchange'] = exchange
                df['symbol_token'] = symbol_info['token']
                df['interval_type'] = interval
                df['data_source'] = 'angel_one'
                
                self.logger.info(f"✅ Retrieved {len(df)} records for {ticker}")
                return df
            else:
                self.logger.error(f"❌ No data retrieved for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error getting optimal data for {ticker}: {e}")
            return pd.DataFrame()
    
    def store_enhanced_data(self, df: pd.DataFrame, ticker: str, exchange: str) -> bool:
        """
        Store data in enhanced database schema.
        """
        try:
            if df.empty:
                return False
            
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Prepare data for insertion
            data_to_insert = []
            for _, row in df.iterrows():
                data_to_insert.append((
                    ticker,
                    exchange,
                    row.get('symbol_token', ''),
                    row.name,  # date index
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    row.get('interval_type', 'ONE_DAY'),
                    row.get('data_source', 'angel_one')
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
            
            self.logger.info(f"✅ Stored {len(data_to_insert)} records for {ticker} in enhanced schema")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing enhanced data for {ticker}: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_enhanced_data(self, ticker: str, exchange: str, 
                         start_date: str, end_date: str,
                         interval: str = 'ONE_DAY') -> pd.DataFrame:
        """
        Get data from enhanced database schema.
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            
            query = f"""
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
            
            df = pd.read_sql(query, conn)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                self.logger.info(f"✅ Retrieved {len(df)} records for {ticker} from enhanced schema")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting enhanced data for {ticker}: {e}")
            return pd.DataFrame()
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_data_quality_metrics(self, ticker: str, exchange: str) -> Dict:
        """
        Get comprehensive data quality metrics.
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            # Get basic metrics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    AVG(close) as avg_price,
                    STDDEV(close) as price_volatility,
                    AVG(volume) as avg_volume,
                    STDDEV(volume) as volume_volatility
                FROM angel_one_stock_data 
                WHERE ticker = '{ticker}' AND exchange = '{exchange}'
            """)
            
            metrics = cursor.fetchone()
            
            # Calculate data completeness
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_records,
                    SUM(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL THEN 1 ELSE 0 END) as missing_records
                FROM angel_one_stock_data 
                WHERE ticker = '{ticker}' AND exchange = '{exchange}'
            """)
            
            completeness = cursor.fetchone()
            
            if completeness['total_records'] > 0:
                data_completeness = ((completeness['total_records'] - completeness['missing_records']) / completeness['total_records']) * 100
            else:
                data_completeness = 0
            
            return {
                'total_records': metrics['total_records'],
                'first_date': metrics['first_date'],
                'last_date': metrics['last_date'],
                'avg_price': self._safe_float_conversion(metrics['avg_price']),
                'price_volatility': self._safe_float_conversion(metrics['price_volatility']),
                'avg_volume': self._safe_float_conversion(metrics['avg_volume']),
                'volume_volatility': self._safe_float_conversion(metrics['volume_volatility']),
                'data_completeness': data_completeness,
                'data_quality_score': min(100, data_completeness * 0.8 + (100 - (self._safe_float_conversion(metrics['price_volatility']) * 10)))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting data quality metrics: {e}")
            return {}
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
    
    def batch_fetch_multiple_stocks(self, tickers: List[str], exchange: str = "NSE",
                                  interval: str = "ONE_DAY", days_back: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch data for multiple stocks efficiently.
        """
        results = {}
        
        for ticker in tickers:
            try:
                self.logger.info(f"📊 Fetching data for {ticker}...")
                df = self.get_optimal_historical_data(ticker, exchange, interval, days_back)
                
                if not df.empty:
                    # Store in enhanced database
                    self.store_enhanced_data(df, ticker, exchange)
                    results[ticker] = df
                    self.logger.info(f"✅ {ticker}: {len(df)} records")
                else:
                    self.logger.warning(f"⚠️ {ticker}: No data retrieved")
                    
            except Exception as e:
                self.logger.error(f"❌ {ticker}: Error - {e}")
                continue
        
        return results
    
    def get_prediction_ready_data(self, ticker: str, exchange: str = "NSE",
                                days_back: int = 365) -> pd.DataFrame:
        """
        Get data optimized for prediction models.
        """
        try:
            # Get comprehensive historical data
            df = self.get_optimal_historical_data(ticker, exchange, "ONE_DAY", days_back)
            
            if df.empty:
                return df
            
            # Add technical indicators for better predictions
            df = self._add_technical_indicators(df)
            
            # Add market sentiment features
            df = self._add_sentiment_features(df)
            
            # Clean and validate data
            df = self._clean_data_for_prediction(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing prediction data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for better predictions."""
        try:
            # Moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error adding technical indicators: {e}")
            return df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market sentiment features."""
        try:
            # Price momentum
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5'] = df['Close'].pct_change(5)
            df['Price_Change_10'] = df['Close'].pct_change(10)
            
            # Volatility
            df['Volatility'] = df['Price_Change'].rolling(window=20).std()
            
            # High-Low ratio
            df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
            
            # Close position in daily range
            df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error adding sentiment features: {e}")
            return df
    
    def _clean_data_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data for prediction models."""
        try:
            # Remove rows with NaN values
            df = df.dropna()
            
            # Remove outliers (beyond 3 standard deviations)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    mean = df[col].mean()
                    std = df[col].std()
                    df = df[abs(df[col] - mean) <= 3 * std]
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning data: {e}")
            return df
    
    def test_connection(self) -> bool:
        """
        Test Angel One API connection
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Test with a simple API call
            result = self.downloader.get_historical_data("RELIANCE", "ONE_DAY", 1, "NSE")
            return result is not None and not result.empty
        except Exception as e:
            self.logger.error(f"Angel One connection test failed: {e}")
            return False
    
    def get_historical_data(self, symbol: str, interval: str = "ONE_DAY", 
                          days: int = None, exchange: str = "NSE") -> Optional[pd.DataFrame]:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            days: Number of days
            exchange: Exchange name
            
        Returns:
            DataFrame with historical data
        """
        try:
            return self.get_optimal_historical_data(symbol, exchange, interval, days)
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None

def main():
    """Test the enhanced Angel One service."""
    try:
        print("🚀 Testing Enhanced Angel One Service...")
        
        service = EnhancedAngelOneService()
        
        # Test with RELIANCE
        print("\n📊 Testing with RELIANCE...")
        df = service.get_optimal_historical_data("RELIANCE", "BSE", "ONE_DAY", 30)
        
        if not df.empty:
            print(f"✅ Retrieved {len(df)} records")
            print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
            print(f"💰 Latest price: ₹{df['Close'].iloc[-1]:.2f}")
            
            # Store in enhanced database
            success = service.store_enhanced_data(df, "RELIANCE", "BSE")
            if success:
                print("✅ Data stored in enhanced database")
            
            # Get data quality metrics
            metrics = service.get_data_quality_metrics("RELIANCE", "BSE")
            print(f"📊 Data quality score: {metrics.get('data_quality_score', 0):.2f}%")
            print(f"📈 Data completeness: {metrics.get('data_completeness', 0):.2f}%")
        else:
            print("❌ No data retrieved")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
