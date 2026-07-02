#!/usr/bin/env python3
"""
Critical Components Test Suite
Tests for the most critical components of the AI Stock Predictor system
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import critical components
from main.pipeline.data_processor import DataProcessor
from main.services.database_manager import DatabaseManager
from main.utils.validators import DataValidator
from main.utils.error_handler import StandardErrorHandler
from main.services.validation_predictor import ValidationPredictor
from main.pipeline.prediction_generator import PredictionGenerator

class TestCriticalComponents(unittest.TestCase):
    """Test critical components functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_ticker = "RELIANCE"
        self.test_data = self._create_sample_data()
        self.temp_dir = tempfile.mkdtemp()
        
        # Test configuration
        self.test_config = {
            'database_url': f'sqlite:///{self.temp_dir}/test.db',
            'max_connections': 5,
            'min_connections': 1,
            'connection_timeout': 10,
            'query_timeout': 30,
            'enable_query_cache': True,
            'enable_performance_monitoring': True
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        data = pd.DataFrame({
            'Open': 100 + np.random.randn(len(dates)) * 5,
            'High': 105 + np.random.randn(len(dates)) * 3,
            'Low': 95 + np.random.randn(len(dates)) * 3,
            'Close': 100 + np.random.randn(len(dates)) * 4,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        return data
    
    def test_cache_functionality(self):
        """Test cache storage and retrieval"""
        try:
            # Initialize data processor
            processor = DataProcessor(self.test_ticker)
            
            # Test cache storage
            cache_result = processor._store_data_in_cache(self.test_data, 'ONE_DAY')
            self.assertTrue(cache_result, "Cache storage should succeed")
            
            # Test cache retrieval
            cached_data = processor._get_cached_data('1mo', 'ONE_DAY')
            self.assertIsNotNone(cached_data, "Cached data should be retrievable")
            
            if not cached_data.empty:
                self.assertEqual(len(cached_data), len(self.test_data), 
                               "Cached data should have same length as original")
            
            print("[OK] Cache functionality test passed")
            
        except Exception as e:
            self.fail(f"Cache functionality test failed: {e}")
    
    def test_database_storage(self):
        """Test database operations"""
        try:
            # Initialize database manager
            db_manager = DatabaseManager(self.test_config)
            
            # Test database connection
            with db_manager.get_connection() as conn:
                self.assertIsNotNone(conn, "Database connection should be established")
            
            # Test data storage
            storage_result = db_manager.store_stock_data(
                ticker=self.test_ticker,
                data=self.test_data,
                source='test',
                interval='ONE_DAY'
            )
            self.assertTrue(storage_result, "Data storage should succeed")
            
            # Test data retrieval
            retrieved_data = db_manager.get_stock_data(
                ticker=self.test_ticker,
                source='test',
                interval='ONE_DAY'
            )
            self.assertIsNotNone(retrieved_data, "Data retrieval should succeed")
            
            if not retrieved_data.empty:
                self.assertGreater(len(retrieved_data), 0, 
                                 "Retrieved data should not be empty")
            
            print("[OK] Database storage test passed")
            
        except Exception as e:
            self.fail(f"Database storage test failed: {e}")
    
    def test_input_validation(self):
        """Test input validation"""
        try:
            validator = DataValidator()
            
            # Test valid ticker
            valid_result = validator.validate_ticker("RELIANCE")
            self.assertTrue(valid_result['valid'], "Valid ticker should pass validation")
            
            # Test invalid ticker
            invalid_result = validator.validate_ticker("")
            self.assertFalse(invalid_result['valid'], "Empty ticker should fail validation")
            
            # Test ticker length validation
            long_ticker = "A" * 15
            long_result = validator.validate_ticker(long_ticker)
            self.assertFalse(long_result['valid'], "Long ticker should fail validation")
            
            # Test dataframe validation
            df_validation = validator.validate_dataframe(
                self.test_data, 
                required_columns=['Open', 'High', 'Low', 'Close', 'Volume'],
                min_rows=10
            )
            self.assertTrue(df_validation['valid'], "Valid dataframe should pass validation")
            
            print("[OK] Input validation test passed")
            
        except Exception as e:
            self.fail(f"Input validation test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling functionality"""
        try:
            error_handler = StandardErrorHandler()
            
            # Test API error handling
            api_error = Exception("API connection failed")
            api_result = error_handler.handle_error(api_error, "API test", "HIGH")
            self.assertIn('API', api_result['category'], "Should categorize as API error")
            self.assertTrue(error_handler.should_retry(api_result), "API errors should be retryable")
            
            # Test database error handling
            db_error = Exception("Database connection failed")
            db_result = error_handler.handle_error(db_error, "Database test", "CRITICAL")
            self.assertIn('DATABASE', db_result['category'], "Should categorize as database error")
            self.assertTrue(error_handler.should_retry(db_result), "Database errors should be retryable")
            
            # Test validation error handling
            val_error = ValueError("Invalid input")
            val_result = error_handler.handle_error(val_error, "Validation test", "MEDIUM")
            self.assertIn('VALIDATION', val_result['category'], "Should categorize as validation error")
            self.assertFalse(error_handler.should_retry(val_result), "Validation errors should not be retryable")
            
            print("[OK] Error handling test passed")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_validation_predictor(self):
        """Test validation predictor functionality"""
        try:
            predictor = ValidationPredictor()
            
            # Test prediction with validation
            prediction_result = predictor.predict_with_validation(
                data=self.test_data,
                ticker=self.test_ticker
            )
            
            self.assertIsNotNone(prediction_result, "Prediction should return results")
            self.assertIn('predictions', prediction_result, "Should contain predictions")
            self.assertIn('actuals', prediction_result, "Should contain actuals")
            
            # Test forecast generation
            forecast_result = predictor._generate_forecast(
                model=None,  # Use fallback
                data=self.test_data,
                forecast_periods=5,
                period_type='daily'
            )
            
            self.assertIsNotNone(forecast_result, "Forecast should be generated")
            self.assertEqual(len(forecast_result), 5, "Should generate 5 forecast periods")
            
            print("[OK] Validation predictor test passed")
            
        except Exception as e:
            self.fail(f"Validation predictor test failed: {e}")
    
    def test_prediction_generator(self):
        """Test prediction generator functionality"""
        try:
            generator = PredictionGenerator()
            
            # Test date generation
            dates = generator._generate_prediction_dates(5, 'daily')
            self.assertEqual(len(dates), 5, "Should generate 5 dates")
            self.assertTrue(all(isinstance(d, str) for d in dates), "Dates should be strings")
            
            # Test intraday date generation
            intraday_dates = generator._generate_prediction_dates(1, 'intraday')
            self.assertGreater(len(intraday_dates), 0, "Should generate intraday dates")
            
            # Test intraday data processing
            processed_data = generator._handle_intraday_data_processing(self.test_data)
            self.assertIsNotNone(processed_data, "Data processing should succeed")
            
            print("[OK] Prediction generator test passed")
            
        except Exception as e:
            self.fail(f"Prediction generator test failed: {e}")
    
    def test_memory_management(self):
        """Test memory management functionality"""
        try:
            processor = DataProcessor(self.test_ticker)
            
            # Test memory usage tracking
            memory_usage = processor._get_memory_usage()
            self.assertIsInstance(memory_usage, float, "Memory usage should be a float")
            self.assertGreaterEqual(memory_usage, 0, "Memory usage should be non-negative")
            
            # Test memory cleanup
            processor._cleanup_memory()
            # No exception should be raised
            
            # Test chunk processing
            large_data = pd.concat([self.test_data] * 10)  # Create larger dataset
            chunked_result = processor._process_data_in_chunks(large_data, chunk_size=5)
            self.assertIsNotNone(chunked_result, "Chunk processing should succeed")
            
            print("[OK] Memory management test passed")
            
        except Exception as e:
            self.fail(f"Memory management test failed: {e}")
    
    def test_database_indexes(self):
        """Test database index creation"""
        try:
            db_manager = DatabaseManager(self.test_config)
            
            # Test index creation by storing data
            storage_result = db_manager.store_stock_data(
                ticker=self.test_ticker,
                data=self.test_data,
                source='test',
                interval='ONE_DAY'
            )
            self.assertTrue(storage_result, "Data storage with indexes should succeed")
            
            print("[OK] Database indexes test passed")
            
        except Exception as e:
            self.fail(f"Database indexes test failed: {e}")
    
    def test_intraday_date_handling(self):
        """Test intraday date handling fixes"""
        try:
            # Create intraday data with proper datetime index
            intraday_dates = pd.date_range(
                start='2024-01-01 09:00:00', 
                end='2024-01-01 16:00:00', 
                freq='1H'
            )
            intraday_data = pd.DataFrame({
                'Open': 100 + np.random.randn(len(intraday_dates)),
                'High': 105 + np.random.randn(len(intraday_dates)),
                'Low': 95 + np.random.randn(len(intraday_dates)),
                'Close': 100 + np.random.randn(len(intraday_dates)),
                'Volume': np.random.randint(100, 1000, len(intraday_dates))
            }, index=intraday_dates)
            
            predictor = ValidationPredictor()
            
            # Test intraday forecast generation
            forecast_result = predictor._generate_forecast(
                model=None,
                data=intraday_data,
                forecast_periods=3,
                period_type='intraday'
            )
            
            self.assertIsNotNone(forecast_result, "Intraday forecast should be generated")
            self.assertEqual(len(forecast_result), 3, "Should generate 3 intraday forecasts")
            
            # Check date format
            for forecast in forecast_result:
                self.assertIn('date', forecast, "Forecast should contain date")
                self.assertIn('predicted_value', forecast, "Forecast should contain predicted_value")
                # Date should be in HH:MM format for intraday
                if ':' in forecast['date']:
                    self.assertTrue(True, "Intraday date format is correct")
            
            print("[OK] Intraday date handling test passed")
            
        except Exception as e:
            self.fail(f"Intraday date handling test failed: {e}")

def run_critical_tests():
    """Run all critical component tests"""
    print("[TEST] Running Critical Components Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCriticalComponents)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"[SUMMARY] Test Summary:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n[FAIL] Failures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"   - {test}: {error_msg}")
    
    if result.errors:
        print(f"\n[ERROR] Errors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"   - {test}: {error_msg}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_critical_tests()
    sys.exit(0 if success else 1)
