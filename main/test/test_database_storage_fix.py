#!/usr/bin/env python3
"""
Test Database Storage Fix
Tests the fixed database storage functionality for Angel One data
"""

import unittest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from main.services.database_manager import DatabaseManager


class TestDatabaseStorageFix(unittest.TestCase):
    """Test the fixed database storage functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'database_url': 'sqlite:///test_stock_data.db',
            'max_connections': 5,
            'min_connections': 1,
            'connection_timeout': 10,
            'query_timeout': 30,
            'enable_query_cache': True,
            'enable_performance_monitoring': True
        }
        self.db_manager = DatabaseManager(self.config)
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=pd.date_range('2024-01-01', periods=5, freq='D'))
        
        self.ticker = "TEST"
        self.interval = "ONE_DAY"
    
    def test_database_connection(self):
        """Test database connection"""
        try:
            result = self.db_manager.test_connection()
            self.assertTrue(result, "Database connection should be successful")
        except Exception as e:
            self.fail(f"Database connection test failed: {e}")
    
    def test_angel_one_data_storage(self):
        """Test Angel One data storage functionality"""
        try:
            # Test storing Angel One data
            self.db_manager.store_stock_data(
                ticker=self.ticker,
                data=self.sample_data,
                source='angel_one',
                interval=self.interval
            )
            
            # Test retrieving the stored data
            retrieved_data = self.db_manager.get_stock_data(
                ticker=self.ticker,
                period='1y',
                source='angel_one',
                interval=self.interval
            )
            
            self.assertIsNotNone(retrieved_data, "Retrieved data should not be None")
            self.assertFalse(retrieved_data.empty, "Retrieved data should not be empty")
            self.assertEqual(len(retrieved_data), len(self.sample_data), 
                           "Retrieved data should have same number of records")
            
        except Exception as e:
            self.fail(f"Angel One data storage test failed: {e}")
    
    def test_angel_one_tickers_retrieval(self):
        """Test Angel One tickers retrieval"""
        try:
            # First store some data
            self.db_manager.store_stock_data(
                ticker=self.ticker,
                data=self.sample_data,
                source='angel_one',
                interval=self.interval
            )
            
            # Test getting available tickers
            tickers = self.db_manager.get_available_tickers(source='angel_one')
            
            self.assertIsInstance(tickers, list, "Tickers should be a list")
            self.assertIn(self.ticker, tickers, f"Ticker {self.ticker} should be in the list")
            
        except Exception as e:
            self.fail(f"Angel One tickers retrieval test failed: {e}")
    
    def test_data_cleanup(self):
        """Test data cleanup functionality"""
        try:
            # Store some data first
            self.db_manager.store_stock_data(
                ticker=self.ticker,
                data=self.sample_data,
                source='angel_one',
                interval=self.interval
            )
            
            # Test cleanup with very recent date (should clean up all data)
            # Use 0 days to keep to clean up all data
            self.db_manager.cleanup_old_data(days_to_keep=0)
            
            # Verify data is cleaned up
            retrieved_data = self.db_manager.get_stock_data(
                ticker=self.ticker,
                period='1y',
                source='angel_one',
                interval=self.interval
            )
            
            # Data should be None or empty after cleanup
            self.assertTrue(retrieved_data is None or retrieved_data.empty, 
                           "Data should be cleaned up")
            
        except Exception as e:
            self.fail(f"Data cleanup test failed: {e}")
    
    def test_database_statistics(self):
        """Test database statistics functionality"""
        try:
            stats = self.db_manager.get_database_statistics()
            
            self.assertIsInstance(stats, dict, "Statistics should be a dictionary")
            self.assertIn('connection_status', stats, "Should have connection status")
            self.assertIn('performance_metrics', stats, "Should have performance metrics")
            
        except Exception as e:
            self.fail(f"Database statistics test failed: {e}")
    
    def test_performance_report(self):
        """Test performance report functionality"""
        try:
            report = self.db_manager.get_performance_report()
            
            self.assertIsInstance(report, dict, "Report should be a dictionary")
            self.assertIn('performance_metrics', report, "Should have performance metrics")
            self.assertIn('recommendations', report, "Should have recommendations")
            self.assertIn('timestamp', report, "Should have timestamp")
            
        except Exception as e:
            self.fail(f"Performance report test failed: {e}")
    
    def tearDown(self):
        """Clean up test data"""
        try:
            # Clean up test database file if it exists
            if os.path.exists('test_stock_data.db'):
                os.remove('test_stock_data.db')
        except Exception as e:
            print(f"Warning: Could not clean up test database: {e}")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
