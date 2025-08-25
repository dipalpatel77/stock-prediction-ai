#!/usr/bin/env python3
"""
Unit tests for model builder functionality
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from partB_model.enhanced_model_builder import EnhancedModelBuilder
from partB_model.enhanced_training import EnhancedTraining


class TestEnhancedModelBuilder(unittest.TestCase):
    """Test cases for EnhancedModelBuilder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_builder = EnhancedModelBuilder()
        self.test_X = np.random.rand(100, 10)
        self.test_y = np.random.rand(100)
        
    def test_model_builder_initialization(self):
        """Test EnhancedModelBuilder initialization"""
        self.assertIsNotNone(self.model_builder)
        
    def test_build_linear_regression(self):
        """Test building linear regression model"""
        model = self.model_builder.build_linear_regression()
        self.assertIsNotNone(model)
        
    def test_build_random_forest(self):
        """Test building random forest model"""
        model = self.model_builder.build_random_forest()
        self.assertIsNotNone(model)
        
    def test_build_lstm_model(self):
        """Test building LSTM model"""
        # Reshape data for LSTM
        X_lstm = self.test_X.reshape(self.test_X.shape[0], self.test_X.shape[1], 1)
        model = self.model_builder.build_lstm_model(X_lstm.shape[1:])
        self.assertIsNotNone(model)
        
    def test_evaluate_models(self):
        """Test model evaluation"""
        # Create test data
        X_test = np.random.rand(20, 10)
        y_test = np.random.rand(20)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Build a simple model for testing
        model = self.model_builder.build_linear_regression()
        model.fit(X_test_flat, y_test)
        
        # Test evaluation
        self.model_builder.evaluate_models(X_test, y_test, X_test_flat)


class TestEnhancedTraining(unittest.TestCase):
    """Test cases for EnhancedTraining class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.training = EnhancedTraining()
        self.test_X = np.random.rand(100, 10)
        self.test_y = np.random.rand(100)
        
    def test_training_initialization(self):
        """Test EnhancedTraining initialization"""
        self.assertIsNotNone(self.training)
        
    def test_train_linear_regression(self):
        """Test training linear regression model"""
        model = self.training.train_linear_regression(self.test_X, self.test_y)
        self.assertIsNotNone(model)
        
    def test_train_random_forest(self):
        """Test training random forest model"""
        model = self.training.train_random_forest(self.test_X, self.test_y)
        self.assertIsNotNone(model)
        
    def test_train_lstm_model(self):
        """Test training LSTM model"""
        # Reshape data for LSTM
        X_lstm = self.test_X.reshape(self.test_X.shape[0], self.test_X.shape[1], 1)
        model = self.training.train_lstm_model(X_lstm, self.test_y)
        self.assertIsNotNone(model)
        
    def test_evaluate_model(self):
        """Test model evaluation"""
        # Train a model first
        model = self.training.train_linear_regression(self.test_X, self.test_y)
        
        # Test evaluation
        metrics = self.training.evaluate_model(model, self.test_X, self.test_y)
        self.assertIsInstance(metrics, dict)
        self.assertIn('test_rmse', metrics)
        self.assertIn('test_mae', metrics)
        self.assertIn('test_r2', metrics)


if __name__ == '__main__':
    unittest.main()
