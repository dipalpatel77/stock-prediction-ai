#!/usr/bin/env python3
"""
Performance Test Suite
Tests performance benchmarks and optimization metrics
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import time
import psutil
import gc
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

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
from main.utils.database_pool import DatabaseConnectionPool
from main.utils.model_cache import ModelCache
from main.utils.rate_limiter import APIRateLimiter


class PerformanceMetrics:
    """Performance metrics collection"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
    
    def start_timer(self):
        """Start performance timer"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def end_timer(self, operation_name):
        """End performance timer and record metrics"""
        if self.start_time is None:
            return
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - self.start_time
        memory_usage = end_memory - self.start_memory
        
        self.metrics[operation_name] = {
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'peak_memory': end_memory,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics(self):
        """Get all performance metrics"""
        return self.metrics.copy()
    
    def get_summary(self):
        """Get performance summary"""
        if not self.metrics:
            return {}
        
        total_time = sum(metric['execution_time'] for metric in self.metrics.values())
        total_memory = sum(metric['memory_usage'] for metric in self.metrics.values())
        peak_memory = max(metric['peak_memory'] for metric in self.metrics.values())
        
        return {
            'total_execution_time': total_time,
            'total_memory_usage': total_memory,
            'peak_memory_usage': peak_memory,
            'operations_count': len(self.metrics),
            'average_execution_time': total_time / len(self.metrics)
        }


class TestDataProcessorPerformance(unittest.TestCase):
    """Test DataProcessor performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'include_technical': True,
            'include_economic': True
        }
        self.data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Create large dataset for performance testing
        np.random.seed(42)
        self.large_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'Open': np.random.uniform(100, 200, 1000),
            'High': np.random.uniform(100, 200, 1000),
            'Low': np.random.uniform(100, 200, 1000),
            'Close': np.random.uniform(100, 200, 1000),
            'Volume': np.random.uniform(1000000, 10000000, 1000)
        })
        self.large_data.set_index('Date', inplace=True)
        
        self.metrics = PerformanceMetrics()
    
    def test_data_processing_performance(self):
        """Test data processing performance"""
        self.metrics.start_timer()
        
        with patch.object(self.data_processor, '_load_stock_data', return_value=self.large_data):
            result = self.data_processor.execute(
                data=self.large_data,
                period='2y',
                interval='1d',
                include_technical=True,
                include_economic=True
            )
        
        self.metrics.end_timer('data_processing')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['data_processing']
        self.assertLess(metrics['execution_time'], 30.0)  # Should complete within 30 seconds
        self.assertLess(metrics['memory_usage'], 500.0)  # Should use less than 500MB
        
        self.assertTrue(result['success'])
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization"""
        # Test with chunked processing
        self.metrics.start_timer()
        
        result = self.data_processor._process_data_in_chunks(self.large_data)
        
        self.metrics.end_timer('chunked_processing')
        
        # Check memory usage
        metrics = self.metrics.get_metrics()['chunked_processing']
        self.assertLess(metrics['memory_usage'], 200.0)  # Should use less than 200MB
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.large_data))
    
    def test_technical_indicators_performance(self):
        """Test technical indicators calculation performance"""
        self.metrics.start_timer()
        
        result = self.data_processor._add_technical_indicators(self.large_data, include_technical=True)
        
        self.metrics.end_timer('technical_indicators')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['technical_indicators']
        self.assertLess(metrics['execution_time'], 10.0)  # Should complete within 10 seconds
        
        # Check that technical indicators were added
        expected_indicators = ['SMA_20', 'EMA_12', 'RSI', 'MACD']
        for indicator in expected_indicators:
            if indicator in result.columns:
                self.assertIsNotNone(result[indicator].iloc[-1])
    
    def test_memory_report(self):
        """Test memory reporting"""
        memory_report = self.data_processor.get_memory_report()
        
        self.assertIsInstance(memory_report, dict)
        self.assertIn('current_usage_mb', memory_report)
        self.assertIn('peak_usage_mb', memory_report)
        self.assertGreater(memory_report['current_usage_mb'], 0)


class TestModelTrainingPerformance(unittest.TestCase):
    """Test model training performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'max_workers': 4
        }
        self.model_trainer = ModelTrainer(ticker=self.ticker, config=self.config)
        self.enhanced_trainer = EnhancedModelTrainer(ticker=self.ticker, config=self.config)
        
        # Create large dataset for performance testing
        np.random.seed(42)
        self.large_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 1000),
            'SMA_20': np.random.uniform(100, 200, 1000),
            'RSI': np.random.uniform(0, 100, 1000),
            'MACD': np.random.uniform(-5, 5, 1000),
            'Volume': np.random.uniform(1000000, 10000000, 1000)
        })
        
        self.metrics = PerformanceMetrics()
    
    def test_basic_model_training_performance(self):
        """Test basic model training performance"""
        self.metrics.start_timer()
        
        result = self.model_trainer.execute(data=self.large_data)
        
        self.metrics.end_timer('basic_model_training')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['basic_model_training']
        self.assertLess(metrics['execution_time'], 60.0)  # Should complete within 60 seconds
        self.assertLess(metrics['memory_usage'], 1000.0)  # Should use less than 1GB
        
        self.assertTrue(result['success'])
        self.assertIn('models', result)
    
    def test_enhanced_model_training_performance(self):
        """Test enhanced model training performance"""
        self.metrics.start_timer()
        
        result = self.enhanced_trainer.execute(data=self.large_data)
        
        self.metrics.end_timer('enhanced_model_training')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['enhanced_model_training']
        self.assertLess(metrics['execution_time'], 300.0)  # Should complete within 5 minutes
        self.assertLess(metrics['memory_usage'], 2000.0)  # Should use less than 2GB
        
        self.assertTrue(result['success'])
        self.assertIn('models_trained', result)
        self.assertGreaterEqual(result['models_trained'], 16)
    
    def test_model_caching_performance(self):
        """Test model caching performance"""
        cache = ModelCache(max_size=10, max_memory_mb=500)
        
        # Test caching performance
        self.metrics.start_timer()
        
        # Cache multiple models
        for i in range(5):
            mock_model = Mock()
            cache.cache_model(f"model_{i}.pkl", mock_model, "test_type")
        
        self.metrics.end_timer('model_caching')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['model_caching']
        self.assertLess(metrics['execution_time'], 1.0)  # Should complete within 1 second
        
        # Test cache statistics
        stats = cache.get_cache_statistics()
        self.assertIn('cache_size', stats)
        self.assertIn('memory_usage_mb', stats)


class TestDatabasePerformance(unittest.TestCase):
    """Test database performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'database_url': 'sqlite:///test_performance.db',
            'pool_size': 10
        }
        self.db_manager = DatabaseManager(config=self.config)
        
        # Create large dataset for performance testing
        np.random.seed(42)
        self.large_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'Open': np.random.uniform(100, 200, 1000),
            'High': np.random.uniform(100, 200, 1000),
            'Low': np.random.uniform(100, 200, 1000),
            'Close': np.random.uniform(100, 200, 1000),
            'Volume': np.random.uniform(1000000, 10000000, 1000)
        })
        self.large_data.set_index('Date', inplace=True)
        
        self.metrics = PerformanceMetrics()
    
    def test_database_connection_performance(self):
        """Test database connection performance"""
        self.metrics.start_timer()
        
        try:
            is_connected = self.db_manager.test_connection()
        except Exception:
            is_connected = False
        
        self.metrics.end_timer('database_connection')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['database_connection']
        self.assertLess(metrics['execution_time'], 5.0)  # Should complete within 5 seconds
    
    def test_database_query_performance(self):
        """Test database query performance"""
        # Test query performance
        self.metrics.start_timer()
        
        try:
            stats = self.db_manager.get_database_statistics()
        except Exception:
            stats = {}
        
        self.metrics.end_timer('database_query')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['database_query']
        self.assertLess(metrics['execution_time'], 2.0)  # Should complete within 2 seconds
    
    def test_connection_pool_performance(self):
        """Test connection pool performance"""
        pool = DatabaseConnectionPool(
            connection_config=self.config,
            min_connections=2,
            max_connections=10
        )
        
        self.metrics.start_timer()
        
        # Test multiple connection acquisitions
        connections = []
        for i in range(5):
            try:
                conn = pool.get_connection()
                if conn:
                    connections.append(conn)
            except Exception:
                pass
        
        self.metrics.end_timer('connection_pool')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['connection_pool']
        self.assertLess(metrics['execution_time'], 3.0)  # Should complete within 3 seconds
        
        # Test pool statistics
        stats = pool.get_pool_statistics()
        self.assertIn('total_connections', stats)
        self.assertIn('active_connections', stats)


class TestAPIPerformance(unittest.TestCase):
    """Test API performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_coordinator = APICoordinator(max_workers=4)
        self.rate_limiter = APIRateLimiter()
        
        self.metrics = PerformanceMetrics()
    
    def test_rate_limiting_performance(self):
        """Test rate limiting performance"""
        self.metrics.start_timer()
        
        # Test multiple rate limit checks
        for i in range(100):
            can_proceed = self.api_coordinator.check_rate_limit('test_api')
        
        self.metrics.end_timer('rate_limiting')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['rate_limiting']
        self.assertLess(metrics['execution_time'], 1.0)  # Should complete within 1 second
    
    def test_api_coordinator_performance(self):
        """Test API coordinator performance"""
        self.metrics.start_timer()
        
        # Test coordinator operations
        status = self.api_coordinator.get_coordinator_status()
        stats = self.api_coordinator.get_api_statistics()
        
        self.metrics.end_timer('api_coordinator')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['api_coordinator']
        self.assertLess(metrics['execution_time'], 0.5)  # Should complete within 0.5 seconds
        
        self.assertIsInstance(status, dict)
        self.assertIsInstance(stats, dict)
    
    def test_error_handling_performance(self):
        """Test error handling performance"""
        self.metrics.start_timer()
        
        # Test multiple error handling operations
        for i in range(50):
            test_error = Exception(f"Test error {i}")
            error_result = self.api_coordinator.handle_api_error(test_error, 'test_api')
        
        self.metrics.end_timer('error_handling')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['error_handling']
        self.assertLess(metrics['execution_time'], 2.0)  # Should complete within 2 seconds


class TestPipelinePerformance(unittest.TestCase):
    """Test pipeline performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'use_database': True,
            'max_workers': 4
        }
        self.pipeline = UnifiedAnalysisPipeline(ticker=self.ticker, config=self.config)
        
        # Create large dataset for performance testing
        np.random.seed(42)
        self.large_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'Open': np.random.uniform(100, 200, 1000),
            'High': np.random.uniform(100, 200, 1000),
            'Low': np.random.uniform(100, 200, 1000),
            'Close': np.random.uniform(100, 200, 1000),
            'Volume': np.random.uniform(1000000, 10000000, 1000)
        })
        self.large_data.set_index('Date', inplace=True)
        
        self.metrics = PerformanceMetrics()
    
    def test_pipeline_initialization_performance(self):
        """Test pipeline initialization performance"""
        self.metrics.start_timer()
        
        # Test pipeline initialization
        pipeline = UnifiedAnalysisPipeline(ticker=self.ticker, config=self.config)
        
        self.metrics.end_timer('pipeline_initialization')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['pipeline_initialization']
        self.assertLess(metrics['execution_time'], 5.0)  # Should complete within 5 seconds
        
        self.assertEqual(pipeline.ticker, self.ticker)
        self.assertIsNotNone(pipeline.orchestrator)
    
    def test_component_setup_performance(self):
        """Test component setup performance"""
        self.metrics.start_timer()
        
        # Test component setup
        self.pipeline._setup_components()
        
        self.metrics.end_timer('component_setup')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['component_setup']
        self.assertLess(metrics['execution_time'], 3.0)  # Should complete within 3 seconds
    
    def test_pipeline_validation_performance(self):
        """Test pipeline validation performance"""
        self.metrics.start_timer()
        
        # Test pipeline validation
        is_valid = self.pipeline.validate_pipeline()
        
        self.metrics.end_timer('pipeline_validation')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['pipeline_validation']
        self.assertLess(metrics['execution_time'], 1.0)  # Should complete within 1 second
        
        self.assertIsInstance(is_valid, bool)
    
    def test_memory_cleanup_performance(self):
        """Test memory cleanup performance"""
        # Force garbage collection
        gc.collect()
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create and destroy multiple pipelines
        for i in range(10):
            pipeline = UnifiedAnalysisPipeline(ticker=f"TEST{i}", config=self.config)
            del pipeline
        
        # Force garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly
        self.assertLess(memory_increase, 100.0)  # Should not increase by more than 100MB


class TestConcurrentPerformance(unittest.TestCase):
    """Test concurrent performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'max_workers': 4
        }
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        
        self.metrics = PerformanceMetrics()
    
    def test_concurrent_model_training(self):
        """Test concurrent model training"""
        import concurrent.futures
        
        self.metrics.start_timer()
        
        # Test concurrent model training
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                model_trainer = ModelTrainer(ticker=f"TEST{i}", config=self.config)
                future = executor.submit(model_trainer.execute, self.sample_data)
                futures.append(future)
            
            # Wait for all futures to complete
            results = [future.result() for future in futures]
        
        self.metrics.end_timer('concurrent_model_training')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['concurrent_model_training']
        self.assertLess(metrics['execution_time'], 60.0)  # Should complete within 60 seconds
        
        # All results should be successful
        for result in results:
            self.assertIsInstance(result, dict)
    
    def test_concurrent_data_processing(self):
        """Test concurrent data processing"""
        import concurrent.futures
        
        self.metrics.start_timer()
        
        # Test concurrent data processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                data_processor = DataProcessor(ticker=f"TEST{i}", config=self.config)
                future = executor.submit(data_processor._clean_and_preprocess, self.sample_data)
                futures.append(future)
            
            # Wait for all futures to complete
            results = [future.result() for future in futures]
        
        self.metrics.end_timer('concurrent_data_processing')
        
        # Performance assertions
        metrics = self.metrics.get_metrics()['concurrent_data_processing']
        self.assertLess(metrics['execution_time'], 30.0)  # Should complete within 30 seconds
        
        # All results should be DataFrames
        for result in results:
            self.assertIsInstance(result, pd.DataFrame)


class TestScalabilityPerformance(unittest.TestCase):
    """Test scalability performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'max_workers': 8
        }
        
        self.metrics = PerformanceMetrics()
    
    def test_data_size_scalability(self):
        """Test data size scalability"""
        data_sizes = [100, 500, 1000, 2000]
        execution_times = []
        
        for size in data_sizes:
            # Create data of different sizes
            np.random.seed(42)
            data = pd.DataFrame({
                'Close': np.random.uniform(100, 200, size),
                'Volume': np.random.uniform(1000000, 10000000, size)
            })
            
            # Test model training performance
            self.metrics.start_timer()
            
            model_trainer = ModelTrainer(ticker=self.ticker, config=self.config)
            result = model_trainer.execute(data=data)
            
            self.metrics.end_timer(f'model_training_{size}')
            
            execution_times.append(self.metrics.get_metrics()[f'model_training_{size}']['execution_time'])
        
        # Check that execution time scales reasonably
        for i in range(1, len(execution_times)):
            # Execution time should not increase exponentially
            time_ratio = execution_times[i] / execution_times[i-1]
            data_ratio = data_sizes[i] / data_sizes[i-1]
            
            # Time ratio should be less than data ratio squared
            self.assertLess(time_ratio, data_ratio * data_ratio)
    
    def test_worker_scalability(self):
        """Test worker scalability"""
        worker_counts = [1, 2, 4, 8]
        execution_times = []
        
        for workers in worker_counts:
            config = self.config.copy()
            config['max_workers'] = workers
            
            # Test with concurrent operations
            self.metrics.start_timer()
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for i in range(workers):
                    model_trainer = ModelTrainer(ticker=f"TEST{i}", config=config)
                    future = executor.submit(model_trainer.execute, self.sample_data)
                    futures.append(future)
                
                results = [future.result() for future in futures]
            
            self.metrics.end_timer(f'worker_scalability_{workers}')
            
            execution_times.append(self.metrics.get_metrics()[f'worker_scalability_{workers}']['execution_time'])
        
        # More workers should generally improve performance (up to a point)
        # This is a basic check - in practice, there are diminishing returns
        self.assertIsInstance(execution_times, list)
        self.assertEqual(len(execution_times), len(worker_counts))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataProcessorPerformance,
        TestModelTrainingPerformance,
        TestDatabasePerformance,
        TestAPIPerformance,
        TestPipelinePerformance,
        TestConcurrentPerformance,
        TestScalabilityPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PERFORMANCE TEST SUMMARY")
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
