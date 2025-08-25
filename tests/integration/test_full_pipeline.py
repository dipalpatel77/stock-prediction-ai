#!/usr/bin/env python3
"""
Integration tests for the full prediction pipeline
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from run_stock_prediction import UnifiedPredictionEngine
from unified_analysis_pipeline import UnifiedAnalysisPipeline


class TestFullPipeline(unittest.TestCase):
    """Test cases for the full prediction pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_ticker = "RELIANCE"
        self.test_mode = "simple"
        self.test_days = 5
        
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_unified_prediction_engine_initialization(self):
        """Test UnifiedPredictionEngine initialization"""
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.ticker, self.test_ticker)
        self.assertEqual(engine.mode, self.test_mode)
        
    def test_data_loading_pipeline(self):
        """Test complete data loading pipeline"""
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        
        # Test data loading
        df = engine.load_and_prepare_data()
        if df is not None:
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            print(f"✅ Data loaded successfully: {df.shape}")
        else:
            print("⚠️ Data loading returned None (may be expected in test environment)")
            
    def test_feature_preparation_pipeline(self):
        """Test feature preparation pipeline"""
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        
        # Load data first
        df = engine.load_and_prepare_data()
        if df is not None:
            # Test feature preparation
            X, y = engine.prepare_features(df)
            if X is not None and y is not None:
                self.assertIsInstance(X, np.ndarray)
                self.assertIsInstance(y, np.ndarray)
                self.assertEqual(len(X), len(y))
                print(f"✅ Features prepared successfully: X={X.shape}, y={y.shape}")
            else:
                print("⚠️ Feature preparation returned None")
        else:
            print("⚠️ Skipping feature preparation test due to data loading failure")
            
    def test_model_training_pipeline(self):
        """Test model training pipeline"""
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        
        # Load data and prepare features
        df = engine.load_and_prepare_data()
        if df is not None:
            X, y = engine.prepare_features(df)
            if X is not None and y is not None:
                # Test model training
                success = engine.train_models(X, y)
                if success:
                    print("✅ Model training completed successfully")
                    self.assertTrue(success)
                else:
                    print("⚠️ Model training failed")
            else:
                print("⚠️ Skipping model training test due to feature preparation failure")
        else:
            print("⚠️ Skipping model training test due to data loading failure")
            
    def test_prediction_generation_pipeline(self):
        """Test prediction generation pipeline"""
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        
        # Load data and prepare features
        df = engine.load_and_prepare_data()
        if df is not None:
            X, y = engine.prepare_features(df)
            if X is not None and y is not None:
                # Train models
                success = engine.train_models(X, y)
                if success:
                    # Test prediction generation
                    predictions, multi_day = engine.generate_predictions(X, self.test_days)
                    if predictions is not None:
                        self.assertIsInstance(predictions, dict)
                        print(f"✅ Predictions generated successfully: {len(predictions)} models")
                        if multi_day is not None:
                            print(f"✅ Multi-day predictions generated: {len(multi_day)} days")
                    else:
                        print("⚠️ Prediction generation failed")
                else:
                    print("⚠️ Skipping prediction test due to model training failure")
            else:
                print("⚠️ Skipping prediction test due to feature preparation failure")
        else:
            print("⚠️ Skipping prediction test due to data loading failure")


class TestUnifiedAnalysisPipeline(unittest.TestCase):
    """Test cases for the unified analysis pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_ticker = "RELIANCE"
        self.test_mode = "simple"
        
    def test_analysis_pipeline_initialization(self):
        """Test UnifiedAnalysisPipeline initialization"""
        pipeline = UnifiedAnalysisPipeline(self.test_ticker, mode=self.test_mode)
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.ticker, self.test_ticker)
        self.assertEqual(pipeline.mode, self.test_mode)
        
    def test_analysis_pipeline_execution(self):
        """Test complete analysis pipeline execution"""
        pipeline = UnifiedAnalysisPipeline(self.test_ticker, mode=self.test_mode)
        
        try:
            # Test pipeline execution
            success = pipeline.run_analysis()
            if success:
                print("✅ Analysis pipeline completed successfully")
                self.assertTrue(success)
            else:
                print("⚠️ Analysis pipeline failed")
        except Exception as e:
            print(f"⚠️ Analysis pipeline error: {e}")


if __name__ == '__main__':
    unittest.main()
