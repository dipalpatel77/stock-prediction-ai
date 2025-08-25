#!/usr/bin/env python3
"""
Test helper utilities for the AI Stock Predictor test suite
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestDataGenerator:
    """Helper class for generating test data"""
    
    @staticmethod
    def create_sample_stock_data(days=100, start_date=None):
        """Create sample stock data for testing"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
            
        dates = pd.date_range(start=start_date, periods=days, freq='D')
        
        # Generate realistic stock data
        base_price = 1000
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns with volatility
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
            
        data = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, days),
            'Adj Close': prices
        })
        
        return data
    
    @staticmethod
    def create_sample_features_data(samples=100, features=20):
        """Create sample features data for testing"""
        X = np.random.rand(samples, features)
        y = np.random.rand(samples)
        return X, y
    
    @staticmethod
    def create_sample_predictions_data(days=5):
        """Create sample predictions data for testing"""
        predictions = {
            'RandomForest': np.random.rand(days) * 100 + 1000,
            'LinearRegression': np.random.rand(days) * 100 + 1000,
            'LSTM': np.random.rand(days) * 100 + 1000
        }
        return predictions


class TestEnvironmentManager:
    """Helper class for managing test environment"""
    
    def __init__(self):
        self.temp_dirs = []
        self.original_env = {}
        
    def create_temp_directory(self, prefix="test_"):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_dirs.append(temp_dir)
        return temp_dir
        
    def cleanup_temp_directories(self):
        """Clean up all temporary directories"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
        
    def backup_environment_variable(self, var_name):
        """Backup an environment variable"""
        self.original_env[var_name] = os.environ.get(var_name)
        
    def restore_environment_variables(self):
        """Restore all backed up environment variables"""
        for var_name, value in self.original_env.items():
            if value is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = value
        self.original_env.clear()


class TestAssertions:
    """Custom assertions for testing"""
    
    @staticmethod
    def assert_dataframe_not_empty(df, msg=None):
        """Assert that a DataFrame is not empty"""
        if df is None:
            raise AssertionError(msg or "DataFrame is None")
        if len(df) == 0:
            raise AssertionError(msg or "DataFrame is empty")
            
    @staticmethod
    def assert_dataframe_has_columns(df, required_columns, msg=None):
        """Assert that a DataFrame has required columns"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise AssertionError(msg or f"DataFrame missing columns: {missing_columns}")
            
    @staticmethod
    def assert_predictions_valid(predictions, msg=None):
        """Assert that predictions are valid"""
        if predictions is None:
            raise AssertionError(msg or "Predictions are None")
        if not isinstance(predictions, dict):
            raise AssertionError(msg or "Predictions should be a dictionary")
        if len(predictions) == 0:
            raise AssertionError(msg or "Predictions dictionary is empty")
            
    @staticmethod
    def assert_performance_within_threshold(actual_time, threshold, operation_name, msg=None):
        """Assert that performance is within threshold"""
        if actual_time > threshold:
            raise AssertionError(
                msg or f"{operation_name} took {actual_time:.2f}s, "
                f"exceeded threshold of {threshold:.2f}s"
            )


class MockDataProvider:
    """Mock data provider for testing"""
    
    def __init__(self):
        self.data_generator = TestDataGenerator()
        
    def get_stock_data(self, ticker, days=100):
        """Mock stock data retrieval"""
        return self.data_generator.create_sample_stock_data(days)
        
    def get_features_data(self, samples=100, features=20):
        """Mock features data retrieval"""
        return self.data_generator.create_sample_features_data(samples, features)
        
    def get_predictions_data(self, days=5):
        """Mock predictions data retrieval"""
        return self.data_generator.create_sample_predictions_data(days)


def run_test_with_timeout(test_func, timeout_seconds=300):
    """Run a test function with timeout"""
    import signal
    import functools
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {timeout_seconds} seconds")
        
    # Set up signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = test_func()
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        raise
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        raise


def create_test_config():
    """Create a test configuration"""
    return {
        'test_mode': True,
        'data_source': 'mock',
        'model_cache_enabled': False,
        'log_level': 'DEBUG',
        'max_workers': 2,
        'timeout_seconds': 30
    }


def setup_test_logging():
    """Set up test logging"""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tests/results/test.log')
        ]
    )
    
    return logging.getLogger(__name__)


# Global test utilities instance
test_env_manager = TestEnvironmentManager()
test_data_generator = TestDataGenerator()
mock_data_provider = MockDataProvider()
