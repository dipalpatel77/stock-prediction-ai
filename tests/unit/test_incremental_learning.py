#!/usr/bin/env python3
"""
Unit tests for incremental learning functionality
"""

import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from partB_model.incremental_learning import (
    ModelVersion,
    IncrementalLearningManager,
    IncrementalTrainingPipeline,
    ContinuousLearningScheduler
)


class TestModelVersion(unittest.TestCase):
    """Test cases for ModelVersion class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metadata = {
            'ticker': 'TEST',
            'mode': 'simple',
            'created_at': '2024-01-01T12:00:00',
            'performance_metrics': {'rmse': 0.1, 'mae': 0.05},
            'feature_columns': ['feature1', 'feature2'],
            'training_samples': 1000,
            'validation_samples': 200
        }
        self.version = ModelVersion('TEST_simple_20240101_120000', '/path/to/model.h5', self.metadata)
    
    def test_version_initialization(self):
        """Test ModelVersion initialization"""
        self.assertEqual(self.version.version_id, 'TEST_simple_20240101_120000')
        self.assertEqual(self.version.model_path, '/path/to/model.h5')
        self.assertEqual(self.version.metadata, self.metadata)
        self.assertEqual(self.version.created_at, '2024-01-01T12:00:00')
        self.assertEqual(self.version.performance_metrics, {'rmse': 0.1, 'mae': 0.05})
        self.assertEqual(self.version.feature_columns, ['feature1', 'feature2'])
        self.assertEqual(self.version.training_samples, 1000)
        self.assertEqual(self.version.validation_samples, 200)
    
    def test_to_dict(self):
        """Test ModelVersion to_dict method"""
        version_dict = self.version.to_dict()
        
        self.assertEqual(version_dict['version_id'], 'TEST_simple_20240101_120000')
        self.assertEqual(version_dict['model_path'], '/path/to/model.h5')
        self.assertEqual(version_dict['metadata'], self.metadata)
        self.assertEqual(version_dict['created_at'], '2024-01-01T12:00:00')
        self.assertEqual(version_dict['performance_metrics'], {'rmse': 0.1, 'mae': 0.05})
        self.assertEqual(version_dict['feature_columns'], ['feature1', 'feature2'])
        self.assertEqual(version_dict['training_samples'], 1000)
        self.assertEqual(version_dict['validation_samples'], 200)


class TestIncrementalLearningManager(unittest.TestCase):
    """Test cases for IncrementalLearningManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = IncrementalLearningManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test IncrementalLearningManager initialization"""
        self.assertTrue(os.path.exists(self.manager.base_path))
        self.assertTrue(os.path.exists(self.manager.versions_path))
        self.assertTrue(os.path.exists(self.manager.backup_path))
        self.assertTrue(os.path.exists(self.manager.metadata_path))
        self.assertTrue(os.path.exists(self.manager.performance_path))
        self.assertEqual(self.manager.version_registry, {})
    
    def test_create_version_id(self):
        """Test version ID creation"""
        version_id = self.manager.create_version_id('TEST', 'simple')
        
        self.assertIsInstance(version_id, str)
        self.assertTrue(version_id.startswith('TEST_simple_'))
        self.assertTrue(len(version_id) > len('TEST_simple_'))
    
    def test_register_and_get_version(self):
        """Test version registration and retrieval"""
        metadata = {
            'ticker': 'TEST',
            'mode': 'simple',
            'created_at': datetime.now().isoformat(),
            'performance_metrics': {'rmse': 0.1}
        }
        
        version = ModelVersion('TEST_simple_20240101_120000', '/path/to/model.h5', metadata)
        
        # Register version
        self.manager.register_version('TEST', version)
        
        # Get latest version
        latest_version = self.manager.get_latest_version('TEST', 'simple')
        
        self.assertIsNotNone(latest_version)
        self.assertEqual(latest_version.version_id, 'TEST_simple_20240101_120000')
    
    def test_get_all_versions(self):
        """Test getting all versions for a ticker"""
        metadata1 = {
            'ticker': 'TEST',
            'mode': 'simple',
            'created_at': '2024-01-01T12:00:00',
            'performance_metrics': {'rmse': 0.1}
        }
        metadata2 = {
            'ticker': 'TEST',
            'mode': 'simple',
            'created_at': '2024-01-02T12:00:00',
            'performance_metrics': {'rmse': 0.08}
        }
        
        version1 = ModelVersion('TEST_simple_20240101_120000', '/path/to/model1.h5', metadata1)
        version2 = ModelVersion('TEST_simple_20240102_120000', '/path/to/model2.h5', metadata2)
        
        self.manager.register_version('TEST', version1)
        self.manager.register_version('TEST', version2)
        
        all_versions = self.manager.get_all_versions('TEST', 'simple')
        
        self.assertEqual(len(all_versions), 2)
        # Should be sorted by creation date (newest first)
        self.assertEqual(all_versions[0].version_id, 'TEST_simple_20240102_120000')
        self.assertEqual(all_versions[1].version_id, 'TEST_simple_20240101_120000')
    
    @patch('os.path.exists')
    @patch('shutil.copy2')
    @patch('builtins.open')
    @patch('json.dump')
    def test_backup_current_model(self, mock_json_dump, mock_open, mock_copy2, mock_exists):
        """Test backing up current model"""
        mock_exists.return_value = True
        
        backup_path = self.manager.backup_current_model('TEST', 'simple')
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(backup_path.startswith(str(self.manager.backup_path)))
        self.assertTrue('TEST_simple_' in backup_path)
    
    @patch('os.path.exists')
    @patch('shutil.copy2')
    def test_rollback_to_version(self, mock_copy2, mock_exists):
        """Test rolling back to a specific version"""
        mock_exists.return_value = True
        
        # Create a test version
        metadata = {
            'ticker': 'TEST',
            'mode': 'simple',
            'created_at': datetime.now().isoformat(),
            'performance_metrics': {'rmse': 0.1}
        }
        version = ModelVersion('TEST_simple_20240101_120000', '/path/to/model.h5', metadata)
        self.manager.register_version('TEST', version)
        
        # Test rollback
        success = self.manager.rollback_to_version('TEST', 'TEST_simple_20240101_120000')
        
        self.assertTrue(success)


class TestIncrementalTrainingPipeline(unittest.TestCase):
    """Test cases for IncrementalTrainingPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.learning_manager = IncrementalLearningManager(self.temp_dir)
        self.pipeline = IncrementalTrainingPipeline(self.learning_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_incremental_data(self):
        """Test preparing data for incremental training"""
        # Create sample data
        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100)
        })
        
        feature_columns = ['feature1', 'feature2']
        
        X, y = self.pipeline.prepare_incremental_data(data, feature_columns, 'target')
        
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100,))
        self.assertEqual(X.shape[1], len(feature_columns))
    
    def test_prepare_incremental_data_missing_columns(self):
        """Test preparing data with missing columns"""
        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'target': np.random.rand(100)
        })
        
        feature_columns = ['feature1', 'feature2']  # feature2 is missing
        
        with self.assertRaises(ValueError):
            self.pipeline.prepare_incremental_data(data, feature_columns, 'target')
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation"""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])
        
        X_test = np.array([[1], [2], [3]])
        y_test = np.array([1.1, 2.1, 3.1])
        
        performance = self.pipeline.evaluate_model_performance(mock_model, X_test, y_test)
        
        self.assertIn('mse', performance)
        self.assertIn('mae', performance)
        self.assertIn('rmse', performance)
        self.assertIsInstance(performance['mse'], float)
        self.assertIsInstance(performance['mae'], float)
        self.assertIsInstance(performance['rmse'], float)
    
    def test_should_update_model(self):
        """Test model update decision logic"""
        current_performance = {'rmse': 0.1}
        new_performance_better = {'rmse': 0.08}  # 20% improvement
        new_performance_worse = {'rmse': 0.095}  # 5% improvement (threshold)
        new_performance_no_improvement = {'rmse': 0.099}  # 1% improvement
        
        # Should update with significant improvement
        should_update = self.pipeline.should_update_model(current_performance, new_performance_better)
        self.assertTrue(should_update)
        
        # Should update with threshold improvement
        should_update = self.pipeline.should_update_model(current_performance, new_performance_worse)
        self.assertTrue(should_update)
        
        # Should not update with minimal improvement
        should_update = self.pipeline.should_update_model(current_performance, new_performance_no_improvement)
        self.assertFalse(should_update)


class TestContinuousLearningScheduler(unittest.TestCase):
    """Test cases for ContinuousLearningScheduler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.learning_manager = IncrementalLearningManager(self.temp_dir)
        self.scheduler = ContinuousLearningScheduler(self.learning_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_schedule_update(self):
        """Test scheduling updates"""
        self.scheduler.schedule_update('TEST', 'simple', update_frequency_days=7)
        
        key = 'TEST_simple'
        self.assertIn(key, self.scheduler.update_schedule)
        
        next_update = self.scheduler.update_schedule[key]
        self.assertIsInstance(next_update, datetime)
        
        # Should be approximately 7 days from now
        expected_date = datetime.now() + timedelta(days=7)
        time_diff = abs((next_update - expected_date).total_seconds())
        self.assertLess(time_diff, 60)  # Within 1 minute
    
    def test_get_due_updates(self):
        """Test getting due updates"""
        # Schedule an update in the past
        past_date = datetime.now() - timedelta(days=1)
        self.scheduler.update_schedule['TEST_simple'] = past_date
        
        # Schedule an update in the future
        future_date = datetime.now() + timedelta(days=1)
        self.scheduler.update_schedule['TEST_advanced'] = future_date
        
        due_updates = self.scheduler.get_due_updates()
        
        self.assertEqual(len(due_updates), 1)
        self.assertEqual(due_updates[0], ('TEST', 'simple'))
    
    def test_mark_update_completed(self):
        """Test marking updates as completed"""
        self.scheduler.schedule_update('TEST', 'simple', update_frequency_days=7)
        
        key = 'TEST_simple'
        self.assertIn(key, self.scheduler.update_schedule)
        
        self.scheduler.mark_update_completed('TEST', 'simple')
        
        self.assertNotIn(key, self.scheduler.update_schedule)


if __name__ == '__main__':
    unittest.main()
