#!/usr/bin/env python3
"""
Database Connection Pool
Provides efficient database connection management with pooling
"""

import os
import threading
import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from queue import Queue, Empty
import mysql.connector
from mysql.connector import Error as MySQLError
import sqlite3

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Wrapper for database connections with metadata"""
    
    def __init__(self, connection, connection_type: str, created_at: float):
        self.connection = connection
        self.connection_type = connection_type
        self.created_at = created_at
        self.last_used = created_at
        self.is_active = True
        self.query_count = 0
    
    def execute_query(self, query: str, params: tuple = None) -> Any:
        """Execute a query and track usage"""
        self.last_used = time.time()
        self.query_count += 1
        
        try:
            cursor = self.connection.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return cursor.rowcount
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self.connection.rollback()
            raise
    
    def is_healthy(self) -> bool:
        """Check if connection is still healthy"""
        try:
            if self.connection_type == 'mysql':
                self.connection.ping(reconnect=False)
            elif self.connection_type == 'sqlite':
                self.connection.execute('SELECT 1')
            return True
        except Exception:
            return False
    
    def cursor(self):
        """Return a cursor on the underlying connection."""
        return self.connection.cursor()

    def commit(self):
        """Commit the current transaction on the underlying connection."""
        return self.connection.commit()

    def rollback(self):
        """Roll back the current transaction on the underlying connection."""
        return self.connection.rollback()

    def close(self):
        """Close the connection"""
        try:
            self.connection.close()
            self.is_active = False
        except Exception as e:
            logger.error(f"Error closing connection: {e}")

class DatabaseConnectionPool:
    """Thread-safe database connection pool"""
    
    def __init__(self, connection_config: Dict[str, Any], 
                 min_connections: int = 2, 
                 max_connections: int = 10,
                 connection_timeout: int = 30,
                 idle_timeout: int = 300):
        self.connection_config = connection_config
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        
        # Thread-safe queue for available connections
        self.available_connections = Queue(maxsize=max_connections)
        self.all_connections: List[DatabaseConnection] = []
        self.lock = threading.Lock()
        
        # Statistics
        self.total_connections_created = 0
        self.total_queries_executed = 0
        self.connection_errors = 0
        
        # Initialize minimum connections
        self._initialize_pool()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_idle_connections, daemon=True)
        self.cleanup_thread.start()
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections"""
        for _ in range(self.min_connections):
            try:
                conn = self._create_connection()
                if conn:
                    self.available_connections.put(conn)
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
                self.connection_errors += 1
    
    def _create_connection(self) -> Optional[DatabaseConnection]:
        """Create a new database connection"""
        try:
            if self.connection_config.get('db_type') == 'mysql':
                connection = mysql.connector.connect(
                    host=self.connection_config.get('host', 'localhost'),
                    user=self.connection_config.get('user', 'root'),
                    password=self.connection_config.get('password', ''),
                    database=self.connection_config.get('database', 'mysql'),
                    autocommit=False,
                    charset='utf8mb4',
                    use_unicode=True
                )
                connection_type = 'mysql'
            else:  # SQLite
                _default_db = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data', 'stock_data.db'
                )
                db_path = self.connection_config.get('database', _default_db)
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                connection = sqlite3.connect(db_path, check_same_thread=False)
                connection_type = 'sqlite'
            
            conn_wrapper = DatabaseConnection(connection, connection_type, time.time())
            self.all_connections.append(conn_wrapper)
            self.total_connections_created += 1
            
            logger.debug(f"Created new {connection_type} connection")
            return conn_wrapper
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            self.connection_errors += 1
            return None
    
    def get_connection(self) -> Optional[DatabaseConnection]:
        """Get a connection from the pool"""
        try:
            # Try to get an existing connection
            connection = self.available_connections.get(timeout=self.connection_timeout)
            
            # Check if connection is still healthy
            if connection.is_healthy():
                return connection
            else:
                # Connection is dead, create a new one
                connection.close()
                self.all_connections.remove(connection)
                return self._create_connection()
                
        except Empty:
            # No available connections, try to create a new one
            with self.lock:
                if len(self.all_connections) < self.max_connections:
                    return self._create_connection()
                else:
                    logger.warning("Connection pool exhausted")
                    return None
        except Exception as e:
            logger.error(f"Error getting connection: {e}")
            return None
    
    def return_connection(self, connection: DatabaseConnection):
        """Return a connection to the pool"""
        if connection and connection.is_active and connection.is_healthy():
            try:
                self.available_connections.put(connection, timeout=1)
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")
                connection.close()
        else:
            if connection:
                connection.close()
                if connection in self.all_connections:
                    self.all_connections.remove(connection)
    
    @contextmanager
    def get_connection_context(self):
        """Context manager for database connections"""
        connection = self.get_connection()
        if not connection:
            raise Exception("Failed to get database connection")
        
        try:
            yield connection
        finally:
            self.return_connection(connection)
    
    def execute_query(self, query: str, params: tuple = None) -> Any:
        """Execute a query using a connection from the pool"""
        with self.get_connection_context() as conn:
            result = conn.execute_query(query, params)
            self.total_queries_executed += 1
            return result
    
    def _cleanup_idle_connections(self):
        """Clean up idle connections (runs in background thread)"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                current_time = time.time()
                connections_to_remove = []
                
                with self.lock:
                    for conn in self.all_connections:
                        if (current_time - conn.last_used > self.idle_timeout and 
                            len(self.all_connections) > self.min_connections):
                            connections_to_remove.append(conn)
                
                # Remove idle connections
                for conn in connections_to_remove:
                    conn.close()
                    self.all_connections.remove(conn)
                    logger.debug("Removed idle connection")
                    
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self.lock:
            active_connections = len([c for c in self.all_connections if c.is_active])
            available_connections = self.available_connections.qsize()
            
            return {
                'total_connections': len(self.all_connections),
                'active_connections': active_connections,
                'available_connections': available_connections,
                'total_created': self.total_connections_created,
                'total_queries': self.total_queries_executed,
                'connection_errors': self.connection_errors,
                'pool_utilization': f"{(active_connections / self.max_connections) * 100:.1f}%"
            }
    
    def close_all_connections(self):
        """Close all connections in the pool"""
        with self.lock:
            for conn in self.all_connections:
                conn.close()
            self.all_connections.clear()
            
            # Clear the queue
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except Empty:
                    break
        
        logger.info("All database connections closed")

# Global connection pool instance
_connection_pool: Optional[DatabaseConnectionPool] = None

def initialize_connection_pool(connection_config: Dict[str, Any], **kwargs) -> DatabaseConnectionPool:
    """Initialize the global connection pool"""
    global _connection_pool
    _connection_pool = DatabaseConnectionPool(connection_config, **kwargs)
    logger.info("Database connection pool initialized")
    return _connection_pool

def get_connection_pool() -> Optional[DatabaseConnectionPool]:
    """Get the global connection pool instance"""
    return _connection_pool

def close_connection_pool():
    """Close the global connection pool"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.close_all_connections()
        _connection_pool = None
        logger.info("Database connection pool closed")
