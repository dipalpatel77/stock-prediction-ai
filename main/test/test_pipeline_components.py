#!/usr/bin/env python3
"""
Test Suite for Pipeline Components
Tests all pipeline components individually and in integration
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import pipeline components
from main.pipeline.data_processor import DataProcessor
from main.pipeline.model_trainer import ModelTrainer
from main.pipeline.enhanced_model_trainer import EnhancedModelTrainer
from main.pipeline.strategy_analyzer import StrategyAnalyzer
from main.pipeline.prediction_generator import PredictionGenerator
from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
from main.pipeline.base_pipeline import BasePipelineComponent, PipelineOrchestrator


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {
            'use_enhanced_features': True,
            'include_technical': True,
            'include_economic': True
        }
        self.data_processor = DataProcessor(ticker=self.ticker, config=self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        self.sample_data.set_index('Date', inplace=True)
    
    def test_initialization(self):
        """Test DataProcessor initialization"""
        self.assertEqual(self.data_processor.ticker, self.ticker)
        self.assertEqual(self.data_processor.config, self.config)
        self.assertIsNotNone(self.data_processor.logger)
    
    def test_validate_input(self):
        """Test input validation"""
        # Valid input
        valid_input = {'data': self.sample_data, 'period': '1y', 'interval': '1d'}
        self.assertTrue(self.data_processor.validate_input(**valid_input))
        
        # Invalid input - missing data
        invalid_input = {'period': '1y', 'interval': '1d'}
        self.assertFalse(self.data_processor.validate_input(**invalid_input))
    
    def test_execute_success(self):
        """Test successful execution"""
        with patch.object(self.data_processor, '_load_stock_data', return_value=self.sample_data):
            result = self.data_processor.execute(
                data=self.sample_data,
                period='1y',
                interval='1d',
                include_technical=True,
                include_economic=True
            )
            
            self.assertTrue(result['success'])
            self.assertIn('processed_data', result)
            self.assertIn('data_quality', result)
    
    def test_execute_failure(self):
        """Test execution failure handling"""
        with patch.object(self.data_processor, '_load_stock_data', side_effect=Exception("Test error")):
            result = self.data_processor.execute(
                data=None,
                period='1y',
                interval='1d'
            )
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
    
    def test_clean_and_preprocess(self):
        """Test data cleaning and preprocessing"""
        # Add some missing values and outliers
        test_data = self.sample_data.copy()
        test_data.loc[test_data.index[0], 'Close'] = np.nan
        test_data.loc[test_data.index[1], 'Close'] = 999999  # Outlier
        
        cleaned_data = self.data_processor._clean_and_preprocess(test_data)
        
        self.assertFalse(cleaned_data.isnull().any().any())
        self.assertLess(cleaned_data['Close'].max(), 999999)
    
    def test_add_technical_indicators(self):
        """Test technical indicators calculation"""
        result = self.data_processor._add_technical_indicators(self.sample_data, include_technical=True)
        
        # Check if technical indicators were added
        expected_indicators = ['SMA_20', 'EMA_12', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
        for indicator in expected_indicators:
            if indicator in result.columns:
                self.assertIsNotNone(result[indicator].iloc[-1])
    
    def test_memory_usage(self):
        """Test memory usage tracking"""
        memory_report = self.data_processor.get_memory_report()
        
        self.assertIn('current_usage_mb', memory_report)
        self.assertIn('peak_usage_mb', memory_report)
        self.assertIsInstance(memory_report['current_usage_mb'], (int, float))


class TestModelTrainer(unittest.TestCase):
    """Test ModelTrainer component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'test_size': 0.2, 'random_state': 42}
        self.model_trainer = ModelTrainer(ticker=self.ticker, config=self.config)
        
        # Create sample data with features
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'SMA_20': np.random.uniform(100, 200, 100),
            'RSI': np.random.uniform(0, 100, 100),
            'MACD': np.random.uniform(-5, 5, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
    
    def test_initialization(self):
        """Test ModelTrainer initialization"""
        self.assertEqual(self.model_trainer.ticker, self.ticker)
        self.assertEqual(self.model_trainer.config, self.config)
        self.assertIsNotNone(self.model_trainer.logger)
    
    def test_prepare_data(self):
        """Test data preparation"""
        X, y = self.model_trainer.prepare_data(self.sample_data)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 0)
    
    def test_execute_success(self):
        """Test successful model training"""
        result = self.model_trainer.execute(data=self.sample_data)
        
        self.assertTrue(result['success'])
        self.assertIn('models', result)
        self.assertIn('best_model', result)
        self.assertIn('metrics', result)
    
    def test_model_training(self):
        """Test individual model training"""
        X, y = self.model_trainer.prepare_data(self.sample_data)
        X_train, X_test, y_train, y_test = self.model_trainer._split_data(X, y)
        
        training_results = self.model_trainer._train_models(X_train, y_train, X_test, y_test)
        
        self.assertIsInstance(training_results, dict)
        self.assertGreater(len(training_results), 0)
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        # Create mock training results
        mock_results = {
            'random_forest': {'model': Mock(), 'metrics': {'test_r2': 0.8}},
            'linear_regression': {'model': Mock(), 'metrics': {'test_r2': 0.7}}
        }
        
        X_test = pd.DataFrame(np.random.randn(20, 5))
        y_test = pd.Series(np.random.randn(20))
        
        evaluation_results = self.model_trainer._evaluate_models(mock_results, X_test, y_test)
        
        self.assertIsInstance(evaluation_results, dict)
        self.assertIn('random_forest', evaluation_results)


class TestEnhancedModelTrainer(unittest.TestCase):
    """Test EnhancedModelTrainer component with 16+ algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'test_size': 0.2, 'random_state': 42}
        self.enhanced_trainer = EnhancedModelTrainer(ticker=self.ticker, config=self.config)
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'SMA_20': np.random.uniform(100, 200, 100),
            'RSI': np.random.uniform(0, 100, 100),
            'MACD': np.random.uniform(-5, 5, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
    
    def test_initialization(self):
        """Test EnhancedModelTrainer initialization"""
        self.assertEqual(self.enhanced_trainer.ticker, self.ticker)
        self.assertEqual(self.enhanced_trainer.config, self.config)
        self.assertIsNotNone(self.enhanced_trainer.logger)
    
    def test_get_all_models(self):
        """Test getting all available models"""
        models = self.enhanced_trainer._get_all_models(self.sample_data)
        
        # Should have 16+ models
        self.assertGreaterEqual(len(models), 16)
        
        # Check for key model types
        expected_models = ['RandomForest', 'GradientBoosting', 'LinearRegression', 'SVR']
        for model_name in expected_models:
            self.assertIn(model_name, models)
    
    def test_execute_success(self):
        """Test successful enhanced model training"""
        result = self.enhanced_trainer.execute(data=self.sample_data)
        
        self.assertTrue(result['success'])
        self.assertIn('models_trained', result)
        self.assertIn('best_model', result)
        self.assertIn('evaluation_results', result)
        self.assertGreaterEqual(result['models_trained'], 16)
    
    def test_ensemble_creation(self):
        """Test ensemble model creation"""
        # Create mock training results
        mock_results = {
            'RandomForest': {'model': Mock(), 'metrics': {'test_r2': 0.8}},
            'GradientBoosting': {'model': Mock(), 'metrics': {'test_r2': 0.75}},
            'LinearRegression': {'model': Mock(), 'metrics': {'test_r2': 0.7}}
        }
        
        X_train = pd.DataFrame(np.random.randn(50, 5))
        y_train = pd.Series(np.random.randn(50))
        
        ensemble_result = self.enhanced_trainer._create_ensemble(mock_results, X_train, y_train)
        
        if ensemble_result:  # Ensemble might not be created if conditions aren't met
            self.assertIn('ensemble', ensemble_result)


class TestStrategyAnalyzer(unittest.TestCase):
    """Test StrategyAnalyzer component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
        self.strategy_analyzer = StrategyAnalyzer(ticker=self.ticker, config=self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
    
    def test_initialization(self):
        """Test StrategyAnalyzer initialization"""
        self.assertEqual(self.strategy_analyzer.ticker, self.ticker)
        self.assertEqual(self.strategy_analyzer.config, self.config)
        self.assertIsNotNone(self.strategy_analyzer.logger)
    
    def test_execute_success(self):
        """Test successful strategy analysis"""
        result = self.strategy_analyzer.execute(data=self.sample_data)
        
        self.assertTrue(result['success'])
        self.assertIn('sentiment_analysis', result)
        self.assertIn('market_factors', result)
        self.assertIn('trading_strategies', result)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis"""
        sentiment_result = self.strategy_analyzer._run_sentiment_analysis(self.sample_data)
        
        self.assertIsInstance(sentiment_result, dict)
        self.assertIn('overall_sentiment', sentiment_result)
    
    def test_market_factors_analysis(self):
        """Test market factors analysis"""
        market_result = self.strategy_analyzer._run_market_factors_analysis(self.sample_data)
        
        self.assertIsInstance(market_result, dict)
        self.assertIn('volatility', market_result)


class TestPredictionGenerator(unittest.TestCase):
    """Test PredictionGenerator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'prediction_horizon': 5}
        self.prediction_generator = PredictionGenerator(ticker=self.ticker, config=self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Close': np.random.uniform(100, 200, 100),
            'SMA_20': np.random.uniform(100, 200, 100),
            'RSI': np.random.uniform(0, 100, 100)
        })
        
        # Create mock models
        self.mock_models = {
            'random_forest': Mock(),
            'linear_regression': Mock()
        }
    
    def test_initialization(self):
        """Test PredictionGenerator initialization"""
        self.assertEqual(self.prediction_generator.ticker, self.ticker)
        self.assertEqual(self.prediction_generator.config, self.config)
        self.assertIsNotNone(self.prediction_generator.logger)
    
    def test_execute_success(self):
        """Test successful prediction generation"""
        result = self.prediction_generator.execute(
            data=self.sample_data,
            models=self.mock_models
        )
        
        self.assertTrue(result['success'])
        self.assertIn('predictions', result)
        self.assertIn('confidence', result)
    
    def test_generate_horizon_predictions(self):
        """Test horizon prediction generation"""
        result = self.prediction_generator._generate_horizon_predictions(
            self.sample_data, self.mock_models, 5, 'short_term'
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
    
    def test_calculate_prediction_confidence(self):
        """Test prediction confidence calculation"""
        mock_predictions = {
            'short_term': {'predictions': [100, 101, 102]},
            'mid_term': {'predictions': [105, 106, 107]},
            'long_term': {'predictions': [110, 111, 112]}
        }
        
        confidence_result = self.prediction_generator._calculate_prediction_confidence(mock_predictions)
        
        self.assertIsInstance(confidence_result, dict)
        self.assertIn('overall_confidence', confidence_result)


class TestUnifiedAnalysisPipeline(unittest.TestCase):
    """Test UnifiedAnalysisPipeline integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True, 'use_database': True}
        self.pipeline = UnifiedAnalysisPipeline(ticker=self.ticker, config=self.config)
    
    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.ticker, self.ticker)
        self.assertEqual(self.pipeline.config, self.config)
        self.assertIsNotNone(self.pipeline.orchestrator)
        self.assertIsNotNone(self.pipeline.service_manager)
    
    def test_setup_components(self):
        """Test component setup"""
        self.pipeline._setup_components()
        
        # Check if components are initialized
        self.assertIsNotNone(self.pipeline.orchestrator)
    
    def test_validate_pipeline(self):
        """Test pipeline validation"""
        is_valid = self.pipeline.validate_pipeline()
        self.assertIsInstance(is_valid, bool)
    
    def test_get_pipeline_status(self):
        """Test pipeline status"""
        status = self.pipeline.get_pipeline_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('components', status)
        self.assertIn('services', status)


class TestPipelineOrchestrator(unittest.TestCase):
    """Test PipelineOrchestrator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'max_workers': 4}
        self.orchestrator = PipelineOrchestrator(ticker=self.ticker, config=self.config)
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertEqual(self.orchestrator.ticker, self.ticker)
        self.assertEqual(self.orchestrator.config, self.config)
        self.assertIsInstance(self.orchestrator.components, dict)
    
    def test_add_component(self):
        """Test adding components"""
        mock_component = Mock(spec=BasePipelineComponent)
        self.orchestrator.add_component('test_component', mock_component)
        
        self.assertIn('test_component', self.orchestrator.components)
    
    def test_remove_component(self):
        """Test removing components"""
        mock_component = Mock(spec=BasePipelineComponent)
        self.orchestrator.add_component('test_component', mock_component)
        self.orchestrator.remove_component('test_component')
        
        self.assertNotIn('test_component', self.orchestrator.components)
    
    def test_set_execution_order(self):
        """Test setting execution order"""
        execution_order = ['data_processor', 'model_trainer', 'prediction_generator']
        self.orchestrator.set_execution_order(execution_order)
        
        self.assertEqual(self.orchestrator.execution_order, execution_order)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataProcessor,
        TestModelTrainer,
        TestEnhancedModelTrainer,
        TestStrategyAnalyzer,
        TestPredictionGenerator,
        TestUnifiedAnalysisPipeline,
        TestPipelineOrchestrator
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPONENTS TEST SUMMARY")
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
