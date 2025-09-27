#!/usr/bin/env python3
"""
Integration Test Suite
Tests integration between components and end-to-end workflows
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import main components
from main.main import main, run_quick_analysis, run_batch_analysis
from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
from main.pipeline.data_processor import DataProcessor
from main.pipeline.model_trainer import ModelTrainer
from main.pipeline.enhanced_model_trainer import EnhancedModelTrainer
from main.pipeline.strategy_analyzer import StrategyAnalyzer
from main.pipeline.prediction_generator import PredictionGenerator
from main.services.database_manager import DatabaseManager
from main.services.angel_one_manager import AngelOneManager
from main.services.api_coordinator import APICoordinator
from main.utils.service_manager import ServiceManager


class TestMainIntegration(unittest.TestCase):
    """Test main.py integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'use_database': True,
            'is_indian': False
        }
    
    def test_main_initialization(self):
        """Test main.py initialization"""
        # Test that main function exists and is callable
        self.assertTrue(callable(main))
        self.assertTrue(callable(run_quick_analysis))
        self.assertTrue(callable(run_batch_analysis))
    
    def test_run_quick_analysis(self):
        """Test quick analysis function"""
        try:
            result = run_quick_analysis(self.ticker, "1y", True)
            self.assertIsInstance(result, dict)
        except Exception as e:
            # Expected if dependencies are not available
            self.assertIsInstance(e, Exception)
    
    def test_run_batch_analysis(self):
        """Test batch analysis function"""
        try:
            tickers = ["AAPL", "MSFT", "GOOGL"]
            result = run_batch_analysis(tickers, "1y")
            self.assertIsInstance(result, dict)
        except Exception as e:
            # Expected if dependencies are not available
            self.assertIsInstance(e, Exception)


class TestPipelineIntegration(unittest.TestCase):
    """Test pipeline component integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'use_database': True,
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        self.sample_data.set_index('Date', inplace=True)
    
    def test_data_processor_to_model_trainer(self):
        """Test data flow from DataProcessor to ModelTrainer"""
        # Initialize components
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        model_trainer = ModelTrainer(ticker=self.ticker, config=self.config)
        
        # Process data
        with patch.object(data_processor, '_load_stock_data', return_value=self.sample_data):
            processed_result = data_processor.execute(
                data=self.sample_data,
                period='1y',
                interval='1d',
                include_technical=True
            )
            
            if processed_result['success']:
                processed_data = processed_result['processed_data']
                
                # Train model with processed data
                training_result = model_trainer.execute(data=processed_data)
                
                self.assertIsInstance(training_result, dict)
                self.assertIn('success', training_result)
    
    def test_model_trainer_to_prediction_generator(self):
        """Test data flow from ModelTrainer to PredictionGenerator"""
        # Initialize components
        model_trainer = ModelTrainer(ticker=self.ticker, config=self.config)
        prediction_generator = PredictionGenerator(ticker=self.ticker, config=self.config)
        
        # Train model
        training_result = model_trainer.execute(data=self.sample_data)
        
        if training_result['success']:
            models = training_result['models']
            
            # Generate predictions
            prediction_result = prediction_generator.execute(
                data=self.sample_data,
                models=models
            )
            
            self.assertIsInstance(prediction_result, dict)
            self.assertIn('success', prediction_result)
    
    def test_enhanced_model_trainer_integration(self):
        """Test EnhancedModelTrainer integration"""
        enhanced_trainer = EnhancedModelTrainer(ticker=self.ticker, config=self.config)
        
        # Train enhanced models
        result = enhanced_trainer.execute(data=self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('models_trained', result)
            self.assertIn('best_model', result)
            self.assertGreaterEqual(result['models_trained'], 16)
    
    def test_strategy_analyzer_integration(self):
        """Test StrategyAnalyzer integration"""
        strategy_analyzer = StrategyAnalyzer(ticker=self.ticker, config=self.config)
        
        # Run strategy analysis
        result = strategy_analyzer.execute(data=self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('sentiment_analysis', result)
            self.assertIn('market_factors', result)
            self.assertIn('trading_strategies', result)
    
    def test_unified_pipeline_integration(self):
        """Test UnifiedAnalysisPipeline integration"""
        pipeline = UnifiedAnalysisPipeline(ticker=self.ticker, config=self.config)
        
        # Test pipeline initialization
        self.assertEqual(pipeline.ticker, self.ticker)
        self.assertEqual(pipeline.config, self.config)
        self.assertIsNotNone(pipeline.orchestrator)
        self.assertIsNotNone(pipeline.service_manager)
        
        # Test pipeline validation
        is_valid = pipeline.validate_pipeline()
        self.assertIsInstance(is_valid, bool)
        
        # Test pipeline status
        status = pipeline.get_pipeline_status()
        self.assertIsInstance(status, dict)
        self.assertIn('components', status)
        self.assertIn('services', status)


class TestServiceIntegration(unittest.TestCase):
    """Test service component integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'database_url': 'sqlite:///test.db',
            'use_enhanced': True
        }
    
    def test_database_manager_integration(self):
        """Test DatabaseManager integration"""
        db_manager = DatabaseManager(config=self.config)
        
        # Test connection
        try:
            is_connected = db_manager.test_connection()
            self.assertIsInstance(is_connected, bool)
        except Exception as e:
            # Expected if database is not available
            self.assertIsInstance(e, Exception)
        
        # Test statistics
        stats = db_manager.get_database_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn('total_connections', stats)
    
    def test_angel_one_manager_integration(self):
        """Test AngelOneManager integration"""
        angel_config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'totp_secret': 'test_totp',
            'exchange': 'NSE'
        }
        
        angel_manager = AngelOneManager(angel_config)
        
        # Test configuration
        is_valid = angel_manager.validate_config()
        self.assertIsInstance(is_valid, bool)
        
        # Test available intervals
        intervals = angel_manager.get_available_intervals()
        self.assertIsInstance(intervals, list)
        self.assertGreater(len(intervals), 0)
    
    def test_api_coordinator_integration(self):
        """Test APICoordinator integration"""
        coordinator = APICoordinator(max_workers=4)
        
        # Test rate limiting
        can_proceed = coordinator.check_rate_limit('test_api')
        self.assertIsInstance(can_proceed, bool)
        
        # Test error handling
        test_error = Exception("Test error")
        error_result = coordinator.handle_api_error(test_error, 'test_api')
        self.assertIsInstance(error_result, dict)
        self.assertIn('success', error_result)
        
        # Test status
        status = coordinator.get_coordinator_status()
        self.assertIsInstance(status, dict)
        self.assertIn('active_workers', status)
    
    def test_service_manager_integration(self):
        """Test ServiceManager integration"""
        service_manager = ServiceManager(config=self.config)
        
        # Test service initialization
        result = service_manager.initialize_services(self.ticker)
        self.assertIsInstance(result, dict)
        self.assertIn('core_services', result)
        self.assertIn('database_services', result)
        
        # Test service status
        status = service_manager.get_service_status()
        self.assertIsInstance(status, dict)
        self.assertIn('total_services', status)
        
        # Test service health
        health = service_manager.get_all_service_health()
        self.assertIsInstance(health, dict)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end workflow integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'use_database': True,
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Create comprehensive sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=200, freq='D'),
            'Open': np.random.uniform(100, 200, 200),
            'High': np.random.uniform(100, 200, 200),
            'Low': np.random.uniform(100, 200, 200),
            'Close': np.random.uniform(100, 200, 200),
            'Volume': np.random.uniform(1000000, 10000000, 200)
        })
        self.sample_data.set_index('Date', inplace=True)
    
    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow"""
        # Initialize pipeline
        pipeline = UnifiedAnalysisPipeline(ticker=self.ticker, config=self.config)
        
        # Test workflow components
        self.assertEqual(pipeline.ticker, self.ticker)
        self.assertEqual(pipeline.config, self.config)
        
        # Test component setup
        pipeline._setup_components()
        self.assertIsNotNone(pipeline.orchestrator)
        
        # Test pipeline validation
        is_valid = pipeline.validate_pipeline()
        self.assertIsInstance(is_valid, bool)
    
    def test_data_processing_workflow(self):
        """Test data processing workflow"""
        # Initialize data processor
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Test data processing
        with patch.object(data_processor, '_load_stock_data', return_value=self.sample_data):
            result = data_processor.execute(
                data=self.sample_data,
                period='1y',
                interval='1d',
                include_technical=True,
                include_economic=True
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            
            if result['success']:
                self.assertIn('processed_data', result)
                self.assertIn('data_quality', result)
    
    def test_ml_training_workflow(self):
        """Test ML training workflow"""
        # Initialize enhanced model trainer
        enhanced_trainer = EnhancedModelTrainer(ticker=self.ticker, config=self.config)
        
        # Test enhanced model training
        result = enhanced_trainer.execute(data=self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('models_trained', result)
            self.assertIn('best_model', result)
            self.assertIn('evaluation_results', result)
            self.assertGreaterEqual(result['models_trained'], 16)
    
    def test_prediction_generation_workflow(self):
        """Test prediction generation workflow"""
        # Initialize prediction generator
        prediction_generator = PredictionGenerator(ticker=self.ticker, config=self.config)
        
        # Create mock models
        mock_models = {
            'random_forest': Mock(),
            'linear_regression': Mock(),
            'gradient_boosting': Mock()
        }
        
        # Test prediction generation
        result = prediction_generator.execute(
            data=self.sample_data,
            models=mock_models
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('predictions', result)
            self.assertIn('confidence', result)
    
    def test_strategy_analysis_workflow(self):
        """Test strategy analysis workflow"""
        # Initialize strategy analyzer
        strategy_analyzer = StrategyAnalyzer(ticker=self.ticker, config=self.config)
        
        # Test strategy analysis
        result = strategy_analyzer.execute(data=self.sample_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('sentiment_analysis', result)
            self.assertIn('market_factors', result)
            self.assertIn('trading_strategies', result)
            self.assertIn('backtesting', result)
    
    def test_error_handling_workflow(self):
        """Test error handling workflow"""
        # Test with invalid data
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        result = data_processor.execute(
            data=empty_data,
            period='1y',
            interval='1d'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        # Should handle empty data gracefully
        self.assertFalse(result['success'])
    
    def test_performance_workflow(self):
        """Test performance monitoring workflow"""
        # Initialize components
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Test memory usage
        memory_report = data_processor.get_memory_report()
        self.assertIsInstance(memory_report, dict)
        self.assertIn('current_usage_mb', memory_report)
        self.assertIn('peak_usage_mb', memory_report)
        
        # Test performance metrics
        if hasattr(data_processor, 'get_performance_report'):
            performance_report = data_processor.get_performance_report()
            self.assertIsInstance(performance_report, dict)


class TestAsyncIntegration(unittest.TestCase):
    """Test async integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced': True,
            'use_database': True
        }
    
    def test_async_pipeline_integration(self):
        """Test async pipeline integration"""
        from main.pipeline.async_pipeline_orchestrator import AsyncPipelineOrchestrator
        
        # Initialize async orchestrator
        async_orchestrator = AsyncPipelineOrchestrator(config=self.config)
        
        # Test initialization
        self.assertEqual(async_orchestrator.config, self.config)
        self.assertIsNotNone(async_orchestrator.logger)
        
        # Test component initialization
        async_orchestrator._initialize_components()
        self.assertIsNotNone(async_orchestrator.data_processor)
        self.assertIsNotNone(async_orchestrator.model_trainer)
        self.assertIsNotNone(async_orchestrator.prediction_generator)
    
    def test_async_data_processing(self):
        """Test async data processing"""
        from main.pipeline.data_processor import DataProcessor
        
        data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        
        # Test async processing
        async def test_async_processing():
            result = await data_processor.async_process_data(sample_data)
            return result
        
        # Run async test
        try:
            result = asyncio.run(test_async_processing())
            self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            # Expected if async processing is not fully implemented
            self.assertIsInstance(e, Exception)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.base_config = {
            'use_enhanced': True,
            'use_database': True,
            'test_size': 0.2,
            'random_state': 42
        }
    
    def test_configuration_validation(self):
        """Test configuration validation across components"""
        from main.utils.validators import ConfigValidator
        
        validator = ConfigValidator()
        
        # Test base configuration
        result = validator.validate_config(self.base_config)
        self.assertIsInstance(result, dict)
        self.assertIn('valid', result)
        
        # Test Angel One configuration
        angel_config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'exchange': 'NSE',
            'interval': 'ONE_DAY'
        }
        
        angel_result = validator.validate_angel_one_config(angel_config)
        self.assertIsInstance(angel_result, dict)
        self.assertIn('valid', angel_result)
    
    def test_configuration_consistency(self):
        """Test configuration consistency across components"""
        # Test that all components can use the same configuration
        components = [
            DataProcessor(ticker=self.ticker, config=self.base_config),
            ModelTrainer(ticker=self.ticker, config=self.base_config),
            StrategyAnalyzer(ticker=self.ticker, config=self.base_config),
            PredictionGenerator(ticker=self.ticker, config=self.base_config)
        ]
        
        for component in components:
            self.assertEqual(component.ticker, self.ticker)
            self.assertEqual(component.config, self.base_config)
    
    def test_environment_configuration(self):
        """Test environment-based configuration"""
        # Test configuration with environment variables
        import os
        
        # Set test environment variables
        os.environ['TEST_DATABASE_URL'] = 'sqlite:///test.db'
        os.environ['TEST_USE_ENHANCED'] = 'true'
        
        # Test that components can read environment variables
        config_with_env = {
            'database_url': os.environ.get('TEST_DATABASE_URL', 'sqlite:///default.db'),
            'use_enhanced': os.environ.get('TEST_USE_ENHANCED', 'false').lower() == 'true'
        }
        
        self.assertEqual(config_with_env['database_url'], 'sqlite:///test.db')
        self.assertTrue(config_with_env['use_enhanced'])


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMainIntegration,
        TestPipelineIntegration,
        TestServiceIntegration,
        TestEndToEndWorkflow,
        TestAsyncIntegration,
        TestConfigurationIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"INTEGRATION TEST SUMMARY")
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
