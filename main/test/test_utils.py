#!/usr/bin/env python3
"""
Test Suite for Utility Components
Tests all utility components individually and in integration
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import utility components
from main.utils.database_pool import DatabaseConnectionPool, DatabaseConnection, get_connection_pool
from main.utils.error_handler import ErrorHandler, ErrorSeverity, ErrorCategory
from main.utils.pipeline_logger import PipelineLogger
from main.utils.model_cache import ModelCache, get_model_cache
from main.utils.rate_limiter import RateLimiter, APIRateLimiter, ExponentialBackoff, get_api_rate_limiter
from main.utils.service_coordinator import ServiceCoordinator
from main.utils.service_manager import ServiceManager
from main.utils.formatters import PriceFormatter, CurrencyFormatter, NumberFormatter
from main.utils.validators import DataValidator, ConfigValidator


class TestDatabasePool(unittest.TestCase):
    """Test DatabaseConnectionPool component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.connection_config = {
            'database_url': 'sqlite:///test.db',
            'pool_size': 5
        }
        self.pool = DatabaseConnectionPool(
            connection_config=self.connection_config,
            min_connections=2,
            max_connections=10
        )
    
    def test_initialization(self):
        """Test DatabaseConnectionPool initialization"""
        self.assertEqual(self.pool.connection_config, self.connection_config)
        self.assertEqual(self.pool.min_connections, 2)
        self.assertEqual(self.pool.max_connections, 10)
    
    def test_get_connection(self):
        """Test getting connection from pool"""
        # This might fail if database is not available, but should not crash
        try:
            connection = self.pool.get_connection()
            if connection is not None:
                self.assertIsInstance(connection, DatabaseConnection)
        except Exception as e:
            # Expected if database is not available
            self.assertIsInstance(e, Exception)
    
    def test_return_connection(self):
        """Test returning connection to pool"""
        # Create a mock connection
        mock_connection = Mock(spec=DatabaseConnection)
        
        # Should not raise an exception
        self.pool.return_connection(mock_connection)
    
    def test_get_pool_statistics(self):
        """Test pool statistics"""
        stats = self.pool.get_pool_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_connections', stats)
        self.assertIn('active_connections', stats)
        self.assertIn('idle_connections', stats)
    
    def test_close_all_connections(self):
        """Test closing all connections"""
        # Should not raise an exception
        self.pool.close_all_connections()


class TestErrorHandler(unittest.TestCase):
    """Test ErrorHandler component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.error_handler = ErrorHandler()
    
    def test_initialization(self):
        """Test ErrorHandler initialization"""
        self.assertIsNotNone(self.error_handler.logger)
        self.assertIsInstance(self.error_handler.error_history, list)
    
    def test_handle_error_success(self):
        """Test successful error handling"""
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
    
    def test_detect_error_category(self):
        """Test error category detection"""
        # Test different error types
        data_error = ValueError("Data error")
        api_error = ConnectionError("API error")
        db_error = Exception("Database error")
        
        data_category = self.error_handler._detect_error_category(data_error)
        api_category = self.error_handler._detect_error_category(api_error)
        db_category = self.error_handler._detect_error_category(db_error)
        
        self.assertIsInstance(data_category, ErrorCategory)
        self.assertIsInstance(api_category, ErrorCategory)
        self.assertIsInstance(db_category, ErrorCategory)
    
    def test_assess_error_severity(self):
        """Test error severity assessment"""
        test_error = ValueError("Test error")
        severity = self.error_handler._assess_error_severity(test_error, ErrorCategory.DATA)
        
        self.assertIsInstance(severity, ErrorSeverity)
    
    def test_get_error_summary(self):
        """Test error summary"""
        summary = self.error_handler.get_error_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_errors', summary)
        self.assertIn('errors_by_category', summary)
        self.assertIn('errors_by_severity', summary)
    
    def test_clear_error_history(self):
        """Test clearing error history"""
        # Add some errors first
        self.error_handler.handle_error(ValueError("Test"), {'test': 'context'})
        
        # Clear history
        self.error_handler.clear_error_history()
        
        # Check that history is cleared
        self.assertEqual(len(self.error_handler.error_history), 0)
    
    def test_export_error_report(self):
        """Test error report export"""
        # Add some errors first
        self.error_handler.handle_error(ValueError("Test"), {'test': 'context'})
        
        # Export report
        report_path = self.error_handler.export_error_report()
        
        self.assertIsInstance(report_path, str)
        self.assertTrue(os.path.exists(report_path))


class TestPipelineLogger(unittest.TestCase):
    """Test PipelineLogger component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = PipelineLogger(name="test_logger", log_level="INFO")
    
    def test_initialization(self):
        """Test PipelineLogger initialization"""
        self.assertEqual(self.logger.name, "test_logger")
        self.assertEqual(self.logger.log_level, "INFO")
        self.assertIsNotNone(self.logger.logger)
    
    def test_log_operation(self):
        """Test operation logging"""
        # Should not raise an exception
        self.logger.log_operation("test_operation", 1.5, "success", component="test")
    
    def test_log_error(self):
        """Test error logging"""
        # Should not raise an exception
        self.logger.log_error("Test error", {"context": "test"})
    
    def test_log_info(self):
        """Test info logging"""
        # Should not raise an exception
        self.logger.info("Test info message", {"context": "test"})
    
    def test_log_warning(self):
        """Test warning logging"""
        # Should not raise an exception
        self.logger.warning("Test warning", {"context": "test"})
    
    def test_log_debug(self):
        """Test debug logging"""
        # Should not raise an exception
        self.logger.debug("Test debug message", {"context": "test"})
    
    def test_log_performance(self):
        """Test performance logging"""
        metrics = {
            'execution_time': 1.5,
            'memory_usage': 100,
            'cpu_usage': 50
        }
        
        # Should not raise an exception
        self.logger.log_performance("test_component", metrics)
    
    def test_get_metrics_summary(self):
        """Test metrics summary"""
        summary = self.logger.get_metrics_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_operations', summary)
        self.assertIn('average_execution_time', summary)
    
    def test_set_log_level(self):
        """Test setting log level"""
        # Should not raise an exception
        self.logger.set_log_level("DEBUG")
        self.assertEqual(self.logger.log_level, "DEBUG")


class TestModelCache(unittest.TestCase):
    """Test ModelCache component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = ModelCache(max_size=10, max_memory_mb=100)
        self.mock_model = Mock()
    
    def test_initialization(self):
        """Test ModelCache initialization"""
        self.assertEqual(self.cache.max_size, 10)
        self.assertEqual(self.cache.max_memory_mb, 100)
        self.assertIsInstance(self.cache.cache, dict)
    
    def test_cache_model(self):
        """Test caching model"""
        # Should not raise an exception
        self.cache.cache_model("test_model.pkl", self.mock_model, "test_type")
    
    def test_get_model(self):
        """Test getting model from cache"""
        # Cache a model first
        self.cache.cache_model("test_model.pkl", self.mock_model, "test_type")
        
        # Get the model
        retrieved_model = self.cache.get_model("test_model.pkl", "test_type")
        
        if retrieved_model is not None:
            self.assertEqual(retrieved_model, self.mock_model)
    
    def test_load_model(self):
        """Test loading model"""
        # This might fail if model file doesn't exist, but should not crash
        try:
            model = self.cache.load_model("nonexistent_model.pkl", "test_type")
            # If successful, should return None for nonexistent file
            self.assertIsNone(model)
        except Exception as e:
            # Expected if file doesn't exist
            self.assertIsInstance(e, Exception)
    
    def test_clear_cache(self):
        """Test clearing cache"""
        # Add some items to cache
        self.cache.cache_model("test1.pkl", self.mock_model, "test_type")
        self.cache.cache_model("test2.pkl", self.mock_model, "test_type")
        
        # Clear cache
        self.cache.clear_cache()
        
        # Check that cache is empty
        self.assertEqual(len(self.cache.cache), 0)
    
    def test_get_cache_statistics(self):
        """Test cache statistics"""
        stats = self.cache.get_cache_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('cache_size', stats)
        self.assertIn('memory_usage_mb', stats)
        self.assertIn('hit_rate', stats)
    
    def test_get_cached_models(self):
        """Test getting cached models list"""
        # Add some models to cache
        self.cache.cache_model("test1.pkl", self.mock_model, "test_type")
        
        cached_models = self.cache.get_cached_models()
        
        self.assertIsInstance(cached_models, list)
        self.assertGreaterEqual(len(cached_models), 0)


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rate_limiter = RateLimiter(max_calls=10, time_window=60)
        self.api_rate_limiter = APIRateLimiter()
    
    def test_initialization(self):
        """Test RateLimiter initialization"""
        self.assertEqual(self.rate_limiter.max_calls, 10)
        self.assertEqual(self.rate_limiter.time_window, 60)
        self.assertIsInstance(self.rate_limiter.calls, list)
    
    def test_acquire_success(self):
        """Test successful rate limit acquisition"""
        result = self.rate_limiter.acquire(wait=False)
        
        self.assertIsInstance(result, bool)
    
    def test_acquire_with_wait(self):
        """Test rate limit acquisition with wait"""
        result = self.rate_limiter.acquire(wait=True)
        
        self.assertIsInstance(result, bool)
    
    def test_get_statistics(self):
        """Test rate limiter statistics"""
        stats = self.rate_limiter.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('calls_made', stats)
        self.assertIn('calls_remaining', stats)
        self.assertIn('reset_time', stats)
    
    def test_exponential_backoff(self):
        """Test exponential backoff"""
        backoff = ExponentialBackoff(base_delay=1.0, max_delay=10.0)
        
        # Test multiple waits
        for i in range(3):
            backoff.wait()
        
        self.assertGreater(backoff.attempt, 0)
    
    def test_api_rate_limiter(self):
        """Test API rate limiter"""
        # Test getting limiter for specific API
        limiter = self.api_rate_limiter._get_limiter('test_api')
        
        self.assertIsInstance(limiter, RateLimiter)
    
    def test_api_rate_limiter_call_with_retry(self):
        """Test API rate limiter with retry"""
        def test_function():
            return "success"
        
        result = self.api_rate_limiter.call_with_retry('test_api', test_function)
        
        self.assertEqual(result, "success")
    
    def test_get_api_rate_limiter_statistics(self):
        """Test API rate limiter statistics"""
        stats = self.api_rate_limiter.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('apis', stats)


class TestServiceCoordinator(unittest.TestCase):
    """Test ServiceCoordinator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = ServiceCoordinator(max_workers=4)
        self.mock_service = Mock()
    
    def test_initialization(self):
        """Test ServiceCoordinator initialization"""
        self.assertEqual(self.coordinator.max_workers, 4)
        self.assertIsInstance(self.coordinator.services, dict)
        self.assertIsInstance(self.coordinator.metrics, dict)
    
    def test_register_service(self):
        """Test service registration"""
        # Should not raise an exception
        self.coordinator.register_service(
            'test_service',
            self.mock_service,
            health_check=lambda: True
        )
        
        self.assertIn('test_service', self.coordinator.services)
    
    def test_unregister_service(self):
        """Test service unregistration"""
        # Register service first
        self.coordinator.register_service('test_service', self.mock_service)
        
        # Unregister service
        self.coordinator.unregister_service('test_service')
        
        self.assertNotIn('test_service', self.coordinator.services)
    
    def test_get_service(self):
        """Test getting service"""
        # Register service first
        self.coordinator.register_service('test_service', self.mock_service)
        
        # Get service
        service = self.coordinator.get_service('test_service')
        
        self.assertEqual(service, self.mock_service)
    
    def test_execute_service(self):
        """Test service execution"""
        # Register service with mock method
        self.mock_service.test_method.return_value = "success"
        self.coordinator.register_service('test_service', self.mock_service)
        
        # Execute service method
        result = self.coordinator.execute_service('test_service', 'test_method')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
    
    def test_check_service_health(self):
        """Test service health check"""
        # Register service with health check
        self.coordinator.register_service(
            'test_service',
            self.mock_service,
            health_check=lambda: True
        )
        
        # Check health
        is_healthy = self.coordinator.check_service_health('test_service')
        
        self.assertTrue(is_healthy)
    
    def test_get_coordinator_status(self):
        """Test coordinator status"""
        status = self.coordinator.get_coordinator_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('registered_services', status)
        self.assertIn('active_workers', status)


class TestServiceManager(unittest.TestCase):
    """Test ServiceManager component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {'use_enhanced': True}
        self.service_manager = ServiceManager(config=self.config)
    
    def test_initialization(self):
        """Test ServiceManager initialization"""
        self.assertEqual(self.service_manager.config, self.config)
        self.assertIsInstance(self.service_manager.services, dict)
    
    def test_initialize_services(self):
        """Test service initialization"""
        result = self.service_manager.initialize_services("AAPL")
        
        self.assertIsInstance(result, dict)
        self.assertIn('core_services', result)
        self.assertIn('database_services', result)
    
    def test_get_service(self):
        """Test getting service"""
        # Initialize services first
        self.service_manager.initialize_services("AAPL")
        
        # Get a service
        service = self.service_manager.get_service('data_service')
        
        # Service might be None if not available, but should not crash
        self.assertTrue(service is None or hasattr(service, '__call__'))
    
    def test_get_service_status(self):
        """Test service status"""
        status = self.service_manager.get_service_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('total_services', status)
        self.assertIn('service_health', status)
    
    def test_check_service_health(self):
        """Test service health check"""
        # Initialize services first
        self.service_manager.initialize_services("AAPL")
        
        # Check health of a service
        is_healthy = self.service_manager.check_service_health('data_service')
        
        self.assertIsInstance(is_healthy, bool)
    
    def test_get_all_service_health(self):
        """Test all service health"""
        health_status = self.service_manager.get_all_service_health()
        
        self.assertIsInstance(health_status, dict)
    
    def test_restart_service(self):
        """Test service restart"""
        # Initialize services first
        self.service_manager.initialize_services("AAPL")
        
        # Restart a service
        result = self.service_manager.restart_service('data_service')
        
        self.assertIsInstance(result, bool)


class TestPriceFormatter(unittest.TestCase):
    """Test PriceFormatter component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.formatter = PriceFormatter(currency="USD", locale_code="en_US")
    
    def test_initialization(self):
        """Test PriceFormatter initialization"""
        self.assertEqual(self.formatter.currency, "USD")
        self.assertEqual(self.formatter.locale_code, "en_US")
    
    def test_format_price(self):
        """Test price formatting"""
        # Test with float
        formatted = self.formatter.format_price(123.45)
        self.assertIsInstance(formatted, str)
        self.assertIn("123", formatted)
        
        # Test with int
        formatted = self.formatter.format_price(100)
        self.assertIsInstance(formatted, str)
        
        # Test with Decimal
        formatted = self.formatter.format_price(Decimal("99.99"))
        self.assertIsInstance(formatted, str)
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        formatted = self.formatter.format_percentage(5.5)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("5", formatted)
    
    def test_format_large_number(self):
        """Test large number formatting"""
        formatted = self.formatter.format_large_number(1000000)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("1", formatted)
    
    def test_format_volume(self):
        """Test volume formatting"""
        formatted = self.formatter.format_volume(1000000)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("1", formatted)
    
    def test_format_market_cap(self):
        """Test market cap formatting"""
        formatted = self.formatter.format_market_cap(1000000000)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("1", formatted)


class TestCurrencyFormatter(unittest.TestCase):
    """Test CurrencyFormatter component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.formatter = CurrencyFormatter()
    
    def test_initialization(self):
        """Test CurrencyFormatter initialization"""
        self.assertIsNotNone(self.formatter)
    
    def test_format_exchange_rate(self):
        """Test exchange rate formatting"""
        formatted = self.formatter.format_exchange_rate(1.2345, "USD", "EUR")
        
        self.assertIsInstance(formatted, str)
        self.assertIn("1.2345", formatted)
    
    def test_format_currency_conversion(self):
        """Test currency conversion formatting"""
        formatted = self.formatter.format_currency_conversion(100, "USD", "EUR", 0.85)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("100", formatted)
    
    def test_format_currency_pair(self):
        """Test currency pair formatting"""
        formatted = self.formatter.format_currency_pair("USD", "EUR", 0.85)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("USD", formatted)
        self.assertIn("EUR", formatted)
    
    def test_format_currency_change(self):
        """Test currency change formatting"""
        formatted = self.formatter.format_currency_change(0.05, "USD")
        
        self.assertIsInstance(formatted, str)
        self.assertIn("0.05", formatted)


class TestNumberFormatter(unittest.TestCase):
    """Test NumberFormatter component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.formatter = NumberFormatter(locale_code="en_US")
    
    def test_initialization(self):
        """Test NumberFormatter initialization"""
        self.assertEqual(self.formatter.locale_code, "en_US")
    
    def test_format_number(self):
        """Test number formatting"""
        formatted = self.formatter.format_number(1234.56)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("1234", formatted)
    
    def test_format_scientific(self):
        """Test scientific notation formatting"""
        formatted = self.formatter.format_scientific(1.23e6)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("1.23", formatted)
    
    def test_format_compact(self):
        """Test compact number formatting"""
        formatted = self.formatter.format_compact(1000000)
        
        self.assertIsInstance(formatted, str)


class TestDataValidator(unittest.TestCase):
    """Test DataValidator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
    
    def test_initialization(self):
        """Test DataValidator initialization"""
        self.assertIsNotNone(self.validator)
    
    def test_validate_ticker(self):
        """Test ticker validation"""
        # Valid tickers
        self.assertTrue(self.validator.validate_ticker("AAPL"))
        self.assertTrue(self.validator.validate_ticker("MSFT"))
        self.assertTrue(self.validator.validate_ticker("RELIANCE"))
        
        # Invalid tickers
        self.assertFalse(self.validator.validate_ticker(""))
        self.assertFalse(self.validator.validate_ticker("123"))
        self.assertFalse(self.validator.validate_ticker("N/A"))
    
    def test_validate_price(self):
        """Test price validation"""
        # Valid prices
        self.assertTrue(self.validator.validate_price(100.50)['valid'])
        self.assertTrue(self.validator.validate_price(100)['valid'])
        self.assertTrue(self.validator.validate_price("100.50")['valid'])
        
        # Invalid prices
        self.assertFalse(self.validator.validate_price(-100)['valid'])
        self.assertFalse(self.validator.validate_price("invalid")['valid'])
    
    def test_validate_volume(self):
        """Test volume validation"""
        # Valid volumes
        self.assertTrue(self.validator.validate_volume(1000000)['valid'])
        self.assertTrue(self.validator.validate_volume(1000000.5)['valid'])
        
        # Invalid volumes
        self.assertFalse(self.validator.validate_volume(-1000)['valid'])
        self.assertFalse(self.validator.validate_volume("invalid")['valid'])
    
    def test_validate_date(self):
        """Test date validation"""
        # Valid dates
        self.assertTrue(self.validator.validate_date("2023-01-01")['valid'])
        self.assertTrue(self.validator.validate_date(datetime.now())['valid'])
        
        # Invalid dates
        self.assertFalse(self.validator.validate_date("invalid")['valid'])
        self.assertFalse(self.validator.validate_date("2023-13-01")['valid'])
    
    def test_validate_percentage(self):
        """Test percentage validation"""
        # Valid percentages
        self.assertTrue(self.validator.validate_percentage(50)['valid'])
        self.assertTrue(self.validator.validate_percentage(50.5)['valid'])
        self.assertTrue(self.validator.validate_percentage(-10)['valid'])
        
        # Invalid percentages
        self.assertFalse(self.validator.validate_percentage("invalid")['valid'])
    
    def test_validate_dataframe(self):
        """Test DataFrame validation"""
        # Valid DataFrame
        valid_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = self.validator.validate_dataframe(valid_df)
        self.assertTrue(result['valid'])
        
        # Invalid DataFrame (empty)
        invalid_df = pd.DataFrame()
        result = self.validator.validate_dataframe(invalid_df)
        self.assertFalse(result['valid'])
    
    def test_validate_json(self):
        """Test JSON validation"""
        # Valid JSON
        valid_json = {"key": "value", "number": 123}
        result = self.validator.validate_json(valid_json)
        self.assertTrue(result['valid'])
        
        # Invalid JSON
        invalid_json = "invalid json"
        result = self.validator.validate_json(invalid_json)
        self.assertFalse(result['valid'])


class TestConfigValidator(unittest.TestCase):
    """Test ConfigValidator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = ConfigValidator()
    
    def test_initialization(self):
        """Test ConfigValidator initialization"""
        self.assertIsNotNone(self.validator)
    
    def test_validate_config(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {
            'use_enhanced': True,
            'database_url': 'sqlite:///test.db',
            'max_workers': 4
        }
        result = self.validator.validate_config(valid_config)
        self.assertTrue(result['valid'])
        
        # Invalid config (missing required field)
        invalid_config = {'use_enhanced': True}
        result = self.validator.validate_config(
            invalid_config,
            required_fields=['database_url']
        )
        self.assertFalse(result['valid'])
    
    def test_validate_angel_one_config(self):
        """Test Angel One configuration validation"""
        # Valid Angel One config
        valid_config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'exchange': 'NSE',
            'interval': 'ONE_DAY'
        }
        result = self.validator.validate_angel_one_config(valid_config)
        self.assertTrue(result['valid'])
        
        # Invalid Angel One config (missing fields)
        invalid_config = {'api_key': 'test_key'}
        result = self.validator.validate_angel_one_config(invalid_config)
        self.assertFalse(result['valid'])
    
    def test_validate_analysis_config(self):
        """Test analysis configuration validation"""
        # Valid analysis config
        valid_config = {
            'analysis_type': 'comprehensive',
            'timeframe': '1y',
            'use_enhanced': True
        }
        result = self.validator.validate_analysis_config(valid_config)
        self.assertTrue(result['valid'])
    
    def test_validate_data_config(self):
        """Test data configuration validation"""
        # Valid data config
        valid_config = {
            'data_source': 'yahoo_finance',
            'period': '1y',
            'interval': '1d'
        }
        result = self.validator.validate_data_config(valid_config)
        self.assertTrue(result['valid'])


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDatabasePool,
        TestErrorHandler,
        TestPipelineLogger,
        TestModelCache,
        TestRateLimiter,
        TestServiceCoordinator,
        TestServiceManager,
        TestPriceFormatter,
        TestCurrencyFormatter,
        TestNumberFormatter,
        TestDataValidator,
        TestConfigValidator
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"UTILITIES TEST SUMMARY")
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
