#!/usr/bin/env python3
"""
Validation Test Suite
Tests validation, error handling, and edge cases
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import warnings

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import main components
from main.pipeline.data_processor import DataProcessor
from main.pipeline.model_trainer import ModelTrainer
from main.pipeline.enhanced_model_trainer import EnhancedModelTrainer
from main.pipeline.strategy_analyzer import StrategyAnalyzer
from main.pipeline.prediction_generator import PredictionGenerator
from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
from main.services.database_manager import DatabaseManager
from main.services.angel_one_manager import AngelOneManager
from main.services.api_coordinator import APICoordinator
from main.utils.error_handler import ErrorHandler, ErrorSeverity, ErrorCategory
from main.utils.validators import DataValidator, ConfigValidator
from main.utils.pipeline_logger import PipelineLogger


class TestInputValidation(unittest.TestCase):
    """Test input validation across components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
        self.validator = DataValidator()
        self.config_validator = ConfigValidator()
    
    def test_ticker_validation(self):
        """Test ticker symbol validation"""
        # Valid tickers
        valid_tickers = ["AAPL", "MSFT", "GOOGL", "RELIANCE", "TCS.NS", "INFY.BO"]
        for ticker in valid_tickers:
            result = self.validator.validate_ticker(ticker)
            self.assertTrue(result['valid'], f"Ticker {ticker} should be valid")
        
        # Invalid tickers
        invalid_tickers = ["", "123", "N/A", "NULL", "NONE", "A" * 100]
        for ticker in invalid_tickers:
            result = self.validator.validate_ticker(ticker)
            self.assertFalse(result['valid'], f"Ticker {ticker} should be invalid")
    
    def test_price_validation(self):
        """Test price validation"""
        # Valid prices
        valid_prices = [100.50, 0.01, 1000, "100.50", Decimal("99.99")]
        for price in valid_prices:
            result = self.validator.validate_price(price)
            self.assertTrue(result['valid'], f"Price {price} should be valid")
        
        # Invalid prices
        invalid_prices = [-100, "invalid", None, float('inf'), float('-inf')]
        for price in invalid_prices:
            result = self.validator.validate_price(price)
            self.assertFalse(result['valid'], f"Price {price} should be invalid")
    
    def test_volume_validation(self):
        """Test volume validation"""
        # Valid volumes
        valid_volumes = [1000000, 0, 1000000000, "1000000"]
        for volume in valid_volumes:
            result = self.validator.validate_volume(volume)
            self.assertTrue(result['valid'], f"Volume {volume} should be valid")
        
        # Invalid volumes
        invalid_volumes = [-1000, "invalid", None, float('inf')]
        for volume in invalid_volumes:
            result = self.validator.validate_volume(volume)
            self.assertFalse(result['valid'], f"Volume {volume} should be invalid")
    
    def test_date_validation(self):
        """Test date validation"""
        # Valid dates
        valid_dates = ["2023-01-01", datetime.now(), datetime.now().date()]
        for date in valid_dates:
            result = self.validator.validate_date(date)
            self.assertTrue(result['valid'], f"Date {date} should be valid")
        
        # Invalid dates
        invalid_dates = ["invalid", "2023-13-01", "2023-01-32", None]
        for date in invalid_dates:
            result = self.validator.validate_date(date)
            self.assertFalse(result['valid'], f"Date {date} should be invalid")
    
    def test_dataframe_validation(self):
        """Test DataFrame validation"""
        # Valid DataFrame
        valid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = self.validator.validate_dataframe(valid_df)
        self.assertTrue(result['valid'])
        
        # Invalid DataFrame (empty)
        invalid_df = pd.DataFrame()
        result = self.validator.validate_dataframe(invalid_df)
        self.assertFalse(result['valid'])
        
        # Invalid DataFrame (missing required columns)
        result = self.validator.validate_dataframe(valid_df, required_columns=['C'])
        self.assertFalse(result['valid'])
    
    def test_json_validation(self):
        """Test JSON validation"""
        # Valid JSON
        valid_json = {"key": "value", "number": 123}
        result = self.validator.validate_json(valid_json)
        self.assertTrue(result['valid'])
        
        # Invalid JSON
        invalid_json = "invalid json"
        result = self.validator.validate_json(invalid_json)
        self.assertFalse(result['valid'])


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_validator = ConfigValidator()
    
    def test_base_config_validation(self):
        """Test base configuration validation"""
        # Valid config
        valid_config = {
            'use_enhanced': True,
            'database_url': 'sqlite:///test.db',
            'max_workers': 4
        }
        result = self.config_validator.validate_config(valid_config)
        self.assertTrue(result['valid'])
        
        # Invalid config (missing required field)
        invalid_config = {'use_enhanced': True}
        result = self.config_validator.validate_config(
            invalid_config,
            required_fields=['database_url']
        )
        self.assertFalse(result['valid'])
        
        # Invalid config (wrong type)
        invalid_config = {'max_workers': 'invalid'}
        result = self.config_validator.validate_config(
            invalid_config,
            field_types={'max_workers': int}
        )
        self.assertFalse(result['valid'])
    
    def test_angel_one_config_validation(self):
        """Test Angel One configuration validation"""
        # Valid Angel One config
        valid_config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'exchange': 'NSE',
            'interval': 'ONE_DAY'
        }
        result = self.config_validator.validate_angel_one_config(valid_config)
        self.assertTrue(result['valid'])
        
        # Invalid Angel One config (missing fields)
        invalid_config = {'api_key': 'test_key'}
        result = self.config_validator.validate_angel_one_config(invalid_config)
        self.assertFalse(result['valid'])
        
        # Invalid Angel One config (wrong exchange)
        invalid_config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'exchange': 'INVALID',
            'interval': 'ONE_DAY'
        }
        result = self.config_validator.validate_angel_one_config(invalid_config)
        self.assertFalse(result['valid'])
    
    def test_analysis_config_validation(self):
        """Test analysis configuration validation"""
        # Valid analysis config
        valid_config = {
            'analysis_type': 'comprehensive',
            'timeframe': '1y',
            'use_enhanced': True
        }
        result = self.config_validator.validate_analysis_config(valid_config)
        self.assertTrue(result['valid'])
        
        # Invalid analysis config (wrong analysis type)
        invalid_config = {
            'analysis_type': 'invalid',
            'timeframe': '1y',
            'use_enhanced': True
        }
        result = self.config_validator.validate_analysis_config(invalid_config)
        self.assertFalse(result['valid'])
    
    def test_data_config_validation(self):
        """Test data configuration validation"""
        # Valid data config
        valid_config = {
            'data_source': 'yahoo_finance',
            'period': '1y',
            'interval': '1d'
        }
        result = self.config_validator.validate_data_config(valid_config)
        self.assertTrue(result['valid'])
        
        # Invalid data config (wrong data source)
        invalid_config = {
            'data_source': 'invalid',
            'period': '1y',
            'interval': '1d'
        }
        result = self.config_validator.validate_data_config(invalid_config)
        self.assertFalse(result['valid'])


class TestErrorHandling(unittest.TestCase):
    """Test error handling across components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.error_handler = ErrorHandler()
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
    
    def test_error_categorization(self):
        """Test error categorization"""
        # Test different error types
        data_error = ValueError("Data error")
        api_error = ConnectionError("API error")
        db_error = Exception("Database error")
        network_error = TimeoutError("Network error")
        
        data_category = self.error_handler._detect_error_category(data_error)
        api_category = self.error_handler._detect_error_category(api_error)
        db_category = self.error_handler._detect_error_category(db_error)
        network_category = self.error_handler._detect_error_category(network_error)
        
        self.assertIsInstance(data_category, ErrorCategory)
        self.assertIsInstance(api_category, ErrorCategory)
        self.assertIsInstance(db_category, ErrorCategory)
        self.assertIsInstance(network_category, ErrorCategory)
    
    def test_error_severity_assessment(self):
        """Test error severity assessment"""
        # Test different error severities
        low_severity_error = ValueError("Minor data issue")
        high_severity_error = Exception("Critical system failure")
        
        low_severity = self.error_handler._assess_error_severity(low_severity_error, ErrorCategory.DATA)
        high_severity = self.error_handler._assess_error_severity(high_severity_error, ErrorCategory.DATABASE)
        
        self.assertIsInstance(low_severity, ErrorSeverity)
        self.assertIsInstance(high_severity, ErrorSeverity)
    
    def test_error_handling_workflow(self):
        """Test complete error handling workflow"""
        # Test error handling with context
        test_error = ValueError("Test error")
        context = {'component': 'test', 'operation': 'test_operation'}
        
        result = self.error_handler.handle_error(
            error=test_error,
            context=context,
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.MEDIUM
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('recovery_strategy', result)
        self.assertIn('error_id', result)
        
        # Check that error was recorded
        self.assertGreater(len(self.error_handler.error_history), 0)
    
    def test_error_recovery_strategies(self):
        """Test error recovery strategies"""
        # Test different error types and their recovery strategies
        error_types = [
            (ValueError("Data error"), ErrorCategory.DATA),
            (ConnectionError("API error"), ErrorCategory.API),
            (Exception("Database error"), ErrorCategory.DATABASE)
        ]
        
        for error, category in error_types:
            result = self.error_handler.handle_error(error, category=category)
            self.assertIsInstance(result, dict)
            self.assertIn('recovery_strategy', result)
    
    def test_error_summary(self):
        """Test error summary generation"""
        # Add some errors first
        self.error_handler.handle_error(ValueError("Test error 1"), {'test': 'context1'})
        self.error_handler.handle_error(ConnectionError("Test error 2"), {'test': 'context2'})
        
        summary = self.error_handler.get_error_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_errors', summary)
        self.assertIn('errors_by_category', summary)
        self.assertIn('errors_by_severity', summary)
        self.assertGreater(summary['total_errors'], 0)
    
    def test_error_export(self):
        """Test error report export"""
        # Add some errors first
        self.error_handler.handle_error(ValueError("Test error"), {'test': 'context'})
        
        # Export error report
        report_path = self.error_handler.export_error_report()
        
        self.assertIsInstance(report_path, str)
        self.assertTrue(os.path.exists(report_path))
        
        # Clean up
        if os.path.exists(report_path):
            os.remove(report_path)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        result = data_processor.execute(data=empty_data, period='1y', interval='1d')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        # Should handle empty data gracefully
        self.assertFalse(result['success'])
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Create data with missing values
        data_with_nulls = pd.DataFrame({
            'Close': [100, np.nan, 102, np.nan, 104],
            'Volume': [1000, 2000, np.nan, 4000, 5000]
        })
        
        result = data_processor.execute(data=data_with_nulls, period='1y', interval='1d')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
    
    def test_extreme_values_handling(self):
        """Test handling of extreme values"""
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Create data with extreme values
        extreme_data = pd.DataFrame({
            'Close': [0.01, 1000000, -100, float('inf'), float('-inf')],
            'Volume': [0, 1000000000, -1000, float('inf'), float('-inf')]
        })
        
        result = data_processor.execute(data=extreme_data, period='1y', interval='1d')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
    
    def test_single_row_data_handling(self):
        """Test handling of single row data"""
        model_trainer = ModelTrainer(ticker=self.ticker, config=self.config)
        
        # Create single row data
        single_row_data = pd.DataFrame({
            'Close': [100],
            'Volume': [1000]
        })
        
        result = model_trainer.execute(data=single_row_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
    
    def test_large_data_handling(self):
        """Test handling of large data"""
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Create large dataset
        large_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 10000),
            'Volume': np.random.uniform(1000000, 10000000, 10000)
        })
        
        result = data_processor.execute(data=large_data, period='1y', interval='1d')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
    
    def test_invalid_ticker_handling(self):
        """Test handling of invalid tickers"""
        # Test with invalid ticker
        invalid_ticker = "INVALID_TICKER_12345"
        data_processor = DataProcessor(ticker=invalid_ticker, config=self.config)
        
        # Should not crash
        self.assertEqual(data_processor.ticker, invalid_ticker)
    
    def test_malformed_config_handling(self):
        """Test handling of malformed configuration"""
        # Test with malformed config
        malformed_config = {
            'use_enhanced': 'invalid_boolean',
            'max_workers': 'not_a_number',
            'database_url': None
        }
        
        data_processor = DataProcessor(ticker=self.ticker, config=malformed_config)
        
        # Should not crash
        self.assertEqual(data_processor.config, malformed_config)
    
    def test_concurrent_access_handling(self):
        """Test handling of concurrent access"""
        import threading
        import time
        
        # Test concurrent access to shared resources
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                data_processor = DataProcessor(ticker=f"TEST{worker_id}", config=self.config)
                result = data_processor.get_memory_report()
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should not have errors
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)


class TestWarningHandling(unittest.TestCase):
    """Test warning handling and suppression"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
    
    def test_warning_suppression(self):
        """Test warning suppression"""
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should not generate warnings
            data_processor = DataProcessor(ticker=self.ticker, config=self.config)
            
            # Check that no warnings were generated
            self.assertEqual(len(w), 0)
    
    def test_deprecation_warnings(self):
        """Test deprecation warning handling"""
        # Test that deprecation warnings are handled gracefully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Simulate deprecation warning
            warnings.warn("This is a deprecation warning", DeprecationWarning)
            
            # Should not crash
            data_processor = DataProcessor(ticker=self.ticker, config=self.config)
            
            # Check that warning was captured
            self.assertGreater(len(w), 0)
    
    def test_future_warnings(self):
        """Test future warning handling"""
        # Test that future warnings are handled gracefully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Simulate future warning
            warnings.warn("This is a future warning", FutureWarning)
            
            # Should not crash
            data_processor = DataProcessor(ticker=self.ticker, config=self.config)
            
            # Check that warning was captured
            self.assertGreater(len(w), 0)


class TestResourceCleanup(unittest.TestCase):
    """Test resource cleanup and memory management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
    
    def test_memory_cleanup(self):
        """Test memory cleanup after operations"""
        import gc
        
        # Get initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create and use components
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        model_trainer = ModelTrainer(ticker=self.ticker, config=self.config)
        
        # Perform operations
        sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        
        data_processor.execute(data=sample_data, period='1y', interval='1d')
        model_trainer.execute(data=sample_data)
        
        # Delete components
        del data_processor
        del model_trainer
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Memory should not increase significantly
        memory_increase = final_memory - initial_memory
        self.assertLess(memory_increase, 100.0)  # Should not increase by more than 100MB
    
    def test_connection_cleanup(self):
        """Test connection cleanup"""
        # Test database connection cleanup
        db_manager = DatabaseManager(config={'database_url': 'sqlite:///test.db'})
        
        # Should not crash
        self.assertIsNotNone(db_manager)
        
        # Test cleanup
        try:
            db_manager.close_all_connections()
        except Exception:
            # Expected if database is not available
            pass
    
    def test_cache_cleanup(self):
        """Test cache cleanup"""
        from main.utils.model_cache import ModelCache
        
        cache = ModelCache(max_size=10, max_memory_mb=100)
        
        # Add some items to cache
        for i in range(5):
            cache.cache_model(f"model_{i}.pkl", Mock(), "test_type")
        
        # Clear cache
        cache.clear_cache()
        
        # Check that cache is empty
        self.assertEqual(len(cache.cache), 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestInputValidation,
        TestConfigurationValidation,
        TestErrorHandling,
        TestEdgeCases,
        TestWarningHandling,
        TestResourceCleanup
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"VALIDATION TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    sys.exit(0 if result.wasSuccessful() else 1)
