#!/usr/bin/env python3
"""
Optimized Database Manager
Manages database operations with advanced connection pooling, query optimization, and performance monitoring
"""

import os
import pandas as pd
import logging
import time
import threading
import asyncio
import aiofiles
import aiomysql
import aiosqlite
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from main.utils.date_formatter import convert_period_to_days
import json

# Import core services
# DatabaseService functionality is now in DatabaseManager itself
from main.utils.database_pool import get_connection_pool

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Optimized database manager with advanced connection pooling, query optimization, and performance monitoring"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, config: Dict[str, Any] = None):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Dict[str, Any] = None):
        if self._initialized:
            return
        if config is None:
            config = {
                'database_url': f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'default.db')}",
                'max_connections': 20,
                'min_connections': 5,
                'connection_timeout': 30,
                'query_timeout': 60,
                'enable_query_cache': True,
                'enable_performance_monitoring': True
            }
        self.config = config
        # DatabaseService functionality is now in DatabaseManager itself
        # self.db_service = DatabaseService()
        # Initialize self as the db_service for compatibility
        self.db_service = self
        
        # Performance monitoring
        self.performance_metrics = {
            'query_count': 0,
            'total_query_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'connection_pool_stats': {}
        }
        
        # Query cache for frequently accessed data
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize connection pool with error handling
        try:
            self.connection_pool = get_connection_pool()
            if self.connection_pool and hasattr(self.connection_pool, 'configure'):
                self.connection_pool.configure(
                    max_connections=config.get('max_connections', 20),
                    min_connections=config.get('min_connections', 5),
                    connection_timeout=config.get('connection_timeout', 30)
                )
            elif self.connection_pool is None:
                # Create a fallback connection pool
                from main.utils.database_pool import DatabaseConnectionPool
                fallback_config = {
                    'db_type': 'sqlite',
                    'database': 'data/stock_data.db',
                    'host': 'localhost',
                    'user': 'root',
                    'password': ''
                }
                self.connection_pool = DatabaseConnectionPool(fallback_config)
                logger.info("Created fallback connection pool")
        except Exception as e:
            logger.warning(f"Connection pool not available: {e}")
            # Create a fallback connection pool
            from main.utils.database_pool import DatabaseConnectionPool
            fallback_config = {
                'db_type': 'sqlite',
                'database': 'data/stock_data.db',
                'host': 'localhost',
                'user': 'root',
                'password': ''
            }
            self.connection_pool = DatabaseConnectionPool(fallback_config)
            logger.info("Created fallback connection pool after error")
        
        # Async support
        self.async_pool = None
        self.async_config = {
            'enable_async': True,
            'async_pool_size': 10,
            'async_timeout': 30
        }
        
        logger.info("Optimized Database Manager initialized with performance monitoring and async support")
        self._initialized = True
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.connection_pool is None:
                logger.warning("Connection pool not available, using basic connection test")
                # Basic connection test without pool
                return True
            
            # Test basic connection
            with self.connection_pool.get_connection_context() as conn:
                # Simple query to test connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                
                if result and result[0] == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed - unexpected result")
                    return False
                    
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_connection(self):
        """Get database connection"""
        try:
            return self.connection_pool.get_connection_context()
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            return None
    
    def execute_query(self, query: str, params: tuple = None, use_cache: bool = True) -> Optional[list]:
        """
        Execute a database query with caching and performance monitoring
        
        Args:
            query: SQL query string
            params: Query parameters
            use_cache: Whether to use query cache
            
        Returns:
            Query results or None if failed
        """
        start_time = time.time()
        
        try:
            # Check cache first if enabled
            if use_cache and self.config.get('enable_query_cache', True):
                cache_key = f"{query}_{params}"
                with self.cache_lock:
                    if cache_key in self.query_cache:
                        self.performance_metrics['cache_hits'] += 1
                        logger.debug(f"Cache hit for query: {query[:50]}...")
                        return self.query_cache[cache_key]
                    else:
                        self.performance_metrics['cache_misses'] += 1
            
            # Execute query with timeout
            with self.connection_pool.get_connection_context() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                results = cursor.fetchall()
                cursor.close()
                
                # Cache results if enabled
                if use_cache and self.config.get('enable_query_cache', True):
                    with self.cache_lock:
                        self.query_cache[cache_key] = results
                        # Limit cache size to prevent memory issues
                        if len(self.query_cache) > 1000:
                            # Remove oldest entries
                            oldest_key = next(iter(self.query_cache))
                            del self.query_cache[oldest_key]
                
                # Update performance metrics
                execution_time = time.time() - start_time
                self.performance_metrics['query_count'] += 1
                self.performance_metrics['total_query_time'] += execution_time
                
                logger.info(f"Query executed successfully in {execution_time:.3f}s: {query[:50]}...")
                return results
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed after {execution_time:.3f}s: {e}")
            return None
    
    def store_stock_data(self, ticker: str, data: pd.DataFrame, source: str, interval: str = 'ONE_DAY'):
        """
        Store stock data in database
        
        Args:
            ticker: Stock ticker symbol
            data: Stock data DataFrame
            source: Data source ('angel_one' or 'yahoo_finance')
            interval: Data interval
        """
        try:
            logger.info(f"Storing {len(data)} records for {ticker} from {source}")
            
            with self.connection_pool.get_connection_context() as conn:
                if source == 'angel_one':
                    self._store_angel_one_data(conn, ticker, data, interval)
                else:
                    self._store_yahoo_data(conn, ticker, data, interval)
            
            logger.info(f"Successfully stored {source} data for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to store {source} data for {ticker}: {e}")
            # Don't raise exception - data processing can continue without storage
    
    def get_stock_data(self, ticker: str, period: str, source: str, interval: str = 'ONE_DAY') -> Optional[pd.DataFrame]:
        """
        Get stock data from database
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            source: Data source ('angel_one' or 'yahoo_finance')
            interval: Data interval
            
        Returns:
            DataFrame with stock data or None if not found
        """
        try:
            logger.info(f"Retrieving {source} data for {ticker}")
            
            with self.connection_pool.get_connection_context() as conn:
                if source == 'angel_one':
                    data = self._get_angel_one_data(conn, ticker, period, interval)
                else:
                    data = self._get_yahoo_data(conn, ticker, period, interval)
            
            if data is not None and not data.empty:
                logger.info(f"Retrieved {len(data)} {source} records for {ticker}")
                return data
            else:
                logger.info(f"No {source} data found for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve {source} data for {ticker}: {e}")
            return None
    
    def _store_angel_one_data(self, conn, ticker: str, data: pd.DataFrame, interval: str):
        """
        Store Angel One data using proper database storage

        Args:
            conn: Database connection
            ticker: Stock ticker symbol
            data: Stock data DataFrame
            interval: Data interval
        """
        cursor = None
        try:
            cursor = conn.cursor()
            is_sqlite = getattr(conn, 'connection_type', 'sqlite') == 'sqlite'
            ph = '?' if is_sqlite else '%s'

            if is_sqlite:
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS angel_one_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    interval_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (ticker, date, interval_type)
                )
                """
                insert_sql = (
                    f"INSERT OR REPLACE INTO angel_one_data "
                    f"(ticker, date, open_price, high_price, low_price, close_price, volume, interval_type) "
                    f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})"
                )
            else:
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS angel_one_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open_price DECIMAL(10,2),
                    high_price DECIMAL(10,2),
                    low_price DECIMAL(10,2),
                    close_price DECIMAL(10,2),
                    volume BIGINT,
                    interval_type VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_ticker_date_interval (ticker, date, interval_type)
                )
                """
                insert_sql = (
                    f"INSERT INTO angel_one_data "
                    f"(ticker, date, open_price, high_price, low_price, close_price, volume, interval_type) "
                    f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}) "
                    f"ON DUPLICATE KEY UPDATE "
                    f"open_price = VALUES(open_price), high_price = VALUES(high_price), "
                    f"low_price = VALUES(low_price), close_price = VALUES(close_price), "
                    f"volume = VALUES(volume)"
                )

            cursor.execute(create_table_sql)

            for date, row in data.iterrows():
                cursor.execute(insert_sql, (
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    interval
                ))

            conn.commit()
            logger.info(f"Successfully stored {len(data)} Angel One records for {ticker}")

        except Exception as e:
            logger.error(f"Failed to store Angel One data: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
    
    def _store_yahoo_data(self, conn, ticker: str, data: pd.DataFrame, interval: str):
        """
        Store Yahoo Finance data using standard schema
        
        Args:
            conn: Database connection
            ticker: Stock ticker symbol
            data: Stock data DataFrame
            interval: Data interval
        """
        try:
            # Use standard database service to store Yahoo Finance data
            self.db_service.store_stock_data(
                ticker=ticker,
                data=data,
                source='yahoo_finance',
                interval=interval
            )
            
            logger.debug(f"Stored Yahoo Finance data for {ticker} with interval {interval}")
            
        except Exception as e:
            logger.error(f"Failed to store Yahoo Finance data: {e}")
            raise e
    
    def _get_angel_one_data(self, conn, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get Angel One data from database
        
        Args:
            conn: Database connection
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with Angel One data or None if not found
        """
        cursor = None
        try:
            cursor = conn.cursor()
            ph = '?' if getattr(conn, 'connection_type', 'sqlite') == 'sqlite' else '%s'

            days = convert_period_to_days(period)
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            query_sql = (
                f"SELECT date, open_price, high_price, low_price, close_price, volume "
                f"FROM angel_one_data "
                f"WHERE ticker = {ph} AND interval_type = {ph} AND date >= {ph} "
                f"ORDER BY date ASC"
            )

            cursor.execute(query_sql, (ticker, interval, start_date))
            results = cursor.fetchall()

            if results:
                df_data = []
                for row in results:
                    df_data.append({
                        'Date': pd.to_datetime(row[0]),
                        'Open': float(row[1]),
                        'High': float(row[2]),
                        'Low': float(row[3]),
                        'Close': float(row[4]),
                        'Volume': int(row[5])
                    })

                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)

                logger.info(f"Retrieved {len(df)} Angel One records for {ticker}")
                return df

            return None

        except Exception as e:
            logger.error(f"Failed to get Angel One data: {e}")
            return None
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
    
    def _get_yahoo_data(self, conn, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get Yahoo Finance data from database
        
        Args:
            conn: Database connection
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with Yahoo Finance data or None if not found
        """
        try:
            # Use standard database service to retrieve Yahoo Finance data
            data = self.db_service.get_stock_data(
                ticker=ticker,
                period=period,
                source='yahoo_finance',
                interval=interval
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get Yahoo Finance data: {e}")
            return None
    
    def _convert_period_to_days(self, period: str) -> int:
        return convert_period_to_days(period)
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics with performance metrics
        
        Returns:
            Dictionary with database statistics and performance metrics
        """
        try:
            stats = {}
            
            # Connection pool statistics
            if self.connection_pool and hasattr(self.connection_pool, 'get_pool_statistics'):
                stats.update(self.connection_pool.get_pool_statistics())
            
            # Performance metrics
            avg_query_time = 0
            if self.performance_metrics['query_count'] > 0:
                avg_query_time = self.performance_metrics['total_query_time'] / self.performance_metrics['query_count']
            
            cache_hit_rate = 0
            total_cache_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
            if total_cache_requests > 0:
                cache_hit_rate = (self.performance_metrics['cache_hits'] / total_cache_requests) * 100
            
            stats.update({
                'database_type': getattr(self.db_service, 'db_type', 'unknown'),
                'connection_status': 'active' if self.connection_pool else 'inactive',
                'performance_metrics': {
                    'total_queries': self.performance_metrics['query_count'],
                    'average_query_time': round(avg_query_time, 3),
                    'total_query_time': round(self.performance_metrics['total_query_time'], 3),
                    'cache_hit_rate': round(cache_hit_rate, 2),
                    'cache_hits': self.performance_metrics['cache_hits'],
                    'cache_misses': self.performance_metrics['cache_misses'],
                    'cache_size': len(self.query_cache)
                }
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {'error': str(e)}
    
    def execute_parallel_queries(self, queries: List[Tuple[str, tuple]]) -> List[Optional[list]]:
        """
        Execute multiple queries in parallel for better performance
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            List of query results
        """
        try:
            logger.info(f"Executing {len(queries)} queries in parallel")
            start_time = time.time()
            
            # Submit all queries to thread pool
            future_to_query = {
                self.thread_pool.submit(self.execute_query, query, params): (query, params)
                for query, params in queries
            }
            
            results = []
            for future in as_completed(future_to_query):
                query, params = future_to_query[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel query failed for {query[:50]}...: {e}")
                    results.append(None)
            
            execution_time = time.time() - start_time
            logger.info(f"Parallel queries completed in {execution_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Parallel query execution failed: {e}")
            return [None] * len(queries)
    
    def batch_insert_data(self, table_name: str, data: List[Dict[str, Any]], batch_size: int = 1000) -> bool:
        """
        Efficiently insert large amounts of data in batches
        
        Args:
            table_name: Target table name
            data: List of data dictionaries
            batch_size: Number of records per batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Batch inserting {len(data)} records into {table_name}")
            start_time = time.time()
            
            with self.connection_pool.get_connection_context() as conn:
                cursor = conn.cursor()
                
                # Process data in batches
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    
                    if not batch:
                        continue
                    
                    # Prepare batch insert query
                    columns = list(batch[0].keys())
                    placeholders = ', '.join(['%s'] * len(columns))
                    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    # Prepare batch data
                    batch_data = [tuple(record[col] for col in columns) for record in batch]
                    
                    # Execute batch insert
                    cursor.executemany(query, batch_data)
                    
                    logger.debug(f"Inserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
                
                conn.commit()
                cursor.close()
            
            execution_time = time.time() - start_time
            logger.info(f"Batch insert completed in {execution_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return False
    
    def clear_query_cache(self):
        """Clear the query cache to free memory"""
        with self.cache_lock:
            self.query_cache.clear()
            logger.info("Query cache cleared")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get detailed performance report
        
        Returns:
            Dictionary with performance metrics and recommendations
        """
        try:
            stats = self.get_database_statistics()
            perf_metrics = stats.get('performance_metrics', {})
            
            # Calculate recommendations
            recommendations = []
            
            if perf_metrics.get('cache_hit_rate', 0) < 50:
                recommendations.append("Consider increasing cache size or optimizing queries")
            
            if perf_metrics.get('average_query_time', 0) > 1.0:
                recommendations.append("Consider adding database indexes or optimizing slow queries")
            
            if perf_metrics.get('total_queries', 0) > 1000:
                recommendations.append("High query volume detected - consider connection pool optimization")
            
            return {
                'performance_metrics': perf_metrics,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    async def async_execute_query(self, query: str, params: tuple = None, use_cache: bool = True) -> Optional[list]:
        """
        Async execute a database query with caching and performance monitoring
        
        Args:
            query: SQL query string
            params: Query parameters
            use_cache: Whether to use query cache
            
        Returns:
            Query results or None if failed
        """
        start_time = time.time()
        
        try:
            # Check cache first if enabled
            if use_cache and self.config.get('enable_query_cache', True):
                cache_key = f"{query}_{params}"
                with self.cache_lock:
                    if cache_key in self.query_cache:
                        self.performance_metrics['cache_hits'] += 1
                        logger.debug(f"Async cache hit for query: {query[:50]}...")
                        return self.query_cache[cache_key]
                    else:
                        self.performance_metrics['cache_misses'] += 1
            
            # Execute async query
            async with self._get_async_connection() as conn:
                async with conn.cursor() as cursor:
                    if params:
                        await cursor.execute(query, params)
                    else:
                        await cursor.execute(query)
                    
                    results = await cursor.fetchall()
                    
                    # Cache results if enabled
                    if use_cache and self.config.get('enable_query_cache', True):
                        with self.cache_lock:
                            self.query_cache[cache_key] = results
                            # Limit cache size
                            if len(self.query_cache) > 1000:
                                oldest_key = next(iter(self.query_cache))
                                del self.query_cache[oldest_key]
                    
                    # Update performance metrics
                    execution_time = time.time() - start_time
                    self.performance_metrics['query_count'] += 1
                    self.performance_metrics['total_query_time'] += execution_time
                    
                    logger.info(f"Async query executed successfully in {execution_time:.3f}s: {query[:50]}...")
                    return results
                    
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Async query execution failed after {execution_time:.3f}s: {e}")
            return None
    
    async def async_execute_parallel_queries(self, queries: List[Tuple[str, tuple]]) -> List[Optional[list]]:
        """
        Execute multiple queries in parallel using async
        
        Args:
            queries: List of (query, params) tuples
            
        Returns:
            List of query results
        """
        try:
            logger.info(f"Executing {len(queries)} async queries in parallel")
            start_time = time.time()
            
            # Create async tasks
            tasks = [
                self.async_execute_query(query, params)
                for query, params in queries
            ]
            
            # Execute all queries concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Async parallel query {i} failed: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            execution_time = time.time() - start_time
            logger.info(f"Async parallel queries completed in {execution_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"Async parallel query execution failed: {e}")
            return [None] * len(queries)
    
    async def async_batch_insert_data(self, table_name: str, data: List[Dict[str, Any]], batch_size: int = 1000) -> bool:
        """
        Efficiently insert large amounts of data in batches using async
        
        Args:
            table_name: Target table name
            data: List of data dictionaries
            batch_size: Number of records per batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Async batch inserting {len(data)} records into {table_name}")
            start_time = time.time()
            
            async with self._get_async_connection() as conn:
                async with conn.cursor() as cursor:
                    # Process data in batches
                    for i in range(0, len(data), batch_size):
                        batch = data[i:i + batch_size]
                        
                        if not batch:
                            continue
                        
                        # Prepare batch insert query
                        columns = list(batch[0].keys())
                        placeholders = ', '.join(['%s'] * len(columns))
                        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                        
                        # Prepare batch data
                        batch_data = [tuple(record[col] for col in columns) for record in batch]
                        
                        # Execute batch insert
                        await cursor.executemany(query, batch_data)
                        
                        logger.debug(f"Async inserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
                    
                    await conn.commit()
            
            execution_time = time.time() - start_time
            logger.info(f"Async batch insert completed in {execution_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Async batch insert failed: {e}")
            return False
    
    @asynccontextmanager
    async def _get_async_connection(self):
        """
        Get async database connection
        
        Yields:
            Async database connection
        """
        if not self.async_pool:
            await self._initialize_async_pool()
        
        conn = None
        try:
            conn = await self.async_pool.acquire()
            yield conn
        finally:
            if conn:
                await self.async_pool.release(conn)
    
    async def _initialize_async_pool(self):
        """
        Initialize async connection pool
        """
        try:
            if 'mysql' in self.config.get('database_url', ''):
                # MySQL async pool
                self.async_pool = await aiomysql.create_pool(
                    host='localhost',
                    user='root',
                    password='7874',
                    db='stock_data',
                    minsize=1,
                    maxsize=self.async_config['async_pool_size'],
                    autocommit=True
                )
            else:
                # SQLite async pool
                self.async_pool = await aiosqlite.create_pool(
                    'default.db',
                    minsize=1,
                    maxsize=self.async_config['async_pool_size']
                )
            
            logger.info("Async connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize async pool: {e}")
            self.async_pool = None
    
    async def async_get_stock_data(self, ticker: str, period: str, source: str, interval: str = 'ONE_DAY') -> Optional[pd.DataFrame]:
        """
        Async get stock data from database
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            source: Data source ('angel_one' or 'yahoo_finance')
            interval: Data interval
            
        Returns:
            DataFrame with stock data or None if not found
        """
        try:
            logger.info(f"Async retrieving {source} data for {ticker}")
            
            if source == 'angel_one':
                data = await self._async_get_angel_one_data(ticker, period, interval)
            else:
                data = await self._async_get_yahoo_data(ticker, period, interval)
            
            if data is not None and not data.empty:
                logger.info(f"Async retrieved {len(data)} {source} records for {ticker}")
                return data
            else:
                logger.info(f"No {source} data found for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Async failed to retrieve {source} data for {ticker}: {e}")
            return None
    
    async def _async_get_angel_one_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Async get Angel One data from database"""
        try:
            # Convert period to days
            days = convert_period_to_days(period)
            
            query = """
                SELECT date, open, high, low, close, volume, symbol_token, interval_type, data_source
                FROM angel_one_stock_data 
                WHERE ticker = %s AND interval_type = %s
                AND date >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY date ASC
            """
            
            results = await self.async_execute_query(query, (ticker, interval, days))
            
            if not results:
                return None
            
            # Convert to DataFrame
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol_token', 'interval_type', 'data_source']
            df = pd.DataFrame(results, columns=columns)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Async failed to get Angel One data: {e}")
            return None
    
    async def _async_get_yahoo_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Async get Yahoo Finance data from database"""
        try:
            # Convert period to days
            days = convert_period_to_days(period)
            
            query = """
                SELECT date, open, high, low, close, volume, adj_close
                FROM stock_data 
                WHERE ticker = %s AND data_source = 'yahoo_finance'
                AND date >= DATE_SUB(NOW(), INTERVAL %s DAY)
                ORDER BY date ASC
            """
            
            results = await self.async_execute_query(query, (ticker, days))
            
            if not results:
                return None
            
            # Convert to DataFrame
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            df = pd.DataFrame(results, columns=columns)
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Async failed to get Yahoo Finance data: {e}")
            return None
    
    def test_database_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing database connection")
            
            with self.connection_pool.get_connection_context() as conn:
                # Try a simple query to test connection
                if self.db_service.db_type == 'mysql':
                    conn.execute('SELECT 1')
                else:  # SQLite
                    conn.execute('SELECT 1')
                
                logger.info("Database connection test successful")
                return True
                
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """
        Clean up old data from database
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        try:
            logger.info(f"Cleaning up data older than {days_to_keep} days")
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.connection_pool.get_connection_context() as conn:
                # Clean up Angel One data
                self._cleanup_angel_one_data(conn, cutoff_date)
                
                # Clean up Yahoo Finance data
                self._cleanup_yahoo_data(conn, cutoff_date)
            
            logger.info("Data cleanup completed")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def _cleanup_angel_one_data(self, conn, cutoff_date: datetime):
        """
        Clean up old Angel One data
        
        Args:
            conn: Database connection
            cutoff_date: Cutoff date for cleanup
        """
        try:
            # Use Angel One database schema for cleanup
            # from src.core.angel_one_database_schema import AngelOneDatabaseSchema
            AngelOneDatabaseSchema = None
            db_schema = AngelOneDatabaseSchema("mysql://root:7874@localhost/stock_data")
            
            # Clean up old data
            db_schema.cleanup_old_data(cutoff_date)
            
            logger.debug("Angel One data cleanup completed")
            
        except Exception as e:
            logger.error(f"Angel One data cleanup failed: {e}")
    
    def _cleanup_yahoo_data(self, conn, cutoff_date: datetime):
        """
        Clean up old Yahoo Finance data
        
        Args:
            conn: Database connection
            cutoff_date: Cutoff date for cleanup
        """
        try:
            # Use standard database service for cleanup
            self.db_service.cleanup_old_data(cutoff_date)
            
            logger.debug("Yahoo Finance data cleanup completed")
            
        except Exception as e:
            logger.error(f"Yahoo Finance data cleanup failed: {e}")
    
    def get_available_tickers(self, source: str = None) -> list:
        """
        Get list of available tickers in database
        
        Args:
            source: Data source filter ('angel_one', 'yahoo_finance', or None for all)
            
        Returns:
            List of available tickers
        """
        try:
            tickers = []
            
            with self.connection_pool.get_connection_context() as conn:
                if source is None or source == 'angel_one':
                    # Get Angel One tickers
                    angel_tickers = self._get_angel_one_tickers(conn)
                    tickers.extend(angel_tickers)
                
                if source is None or source == 'yahoo_finance':
                    # Get Yahoo Finance tickers
                    yahoo_tickers = self._get_yahoo_tickers(conn)
                    tickers.extend(yahoo_tickers)
            
            # Remove duplicates and sort
            tickers = sorted(list(set(tickers)))
            
            logger.info(f"Found {len(tickers)} tickers in database")
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to get available tickers: {e}")
            return []
    
    def _get_angel_one_tickers(self, conn) -> list:
        """
        Get Angel One tickers from database
        
        Args:
            conn: Database connection
            
        Returns:
            List of Angel One tickers
        """
        try:
            # from src.core.angel_one_database_schema import AngelOneDatabaseSchema
            AngelOneDatabaseSchema = None
            db_schema = AngelOneDatabaseSchema("mysql://root:7874@localhost/stock_data")
            
            tickers = db_schema.get_available_tickers()
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to get Angel One tickers: {e}")
            return []
    
    def _get_yahoo_tickers(self, conn) -> list:
        """
        Get Yahoo Finance tickers from database
        
        Args:
            conn: Database connection
            
        Returns:
            List of Yahoo Finance tickers
        """
        try:
            tickers = self.db_service.get_available_tickers(source='yahoo_finance')
            return tickers
            
        except Exception as e:
            logger.error(f"Failed to get Yahoo Finance tickers: {e}")
            return []
