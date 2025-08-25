#!/usr/bin/env python3
"""
Performance tests for the AI Stock Predictor system
"""

import unittest
import time
import psutil
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from run_stock_prediction import UnifiedPredictionEngine
from partB_model.enhanced_model_builder import EnhancedModelBuilder
from partB_model.enhanced_training import EnhancedTraining


class TestPerformance(unittest.TestCase):
    """Performance test cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_ticker = "RELIANCE"
        self.test_mode = "simple"
        self.performance_thresholds = {
            'data_loading_time': 30.0,  # seconds
            'feature_preparation_time': 10.0,  # seconds
            'model_training_time': 60.0,  # seconds
            'prediction_time': 5.0,  # seconds
            'memory_usage_mb': 500.0,  # MB
        }
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
        
    def test_data_loading_performance(self):
        """Test data loading performance"""
        print("\n🧪 Testing data loading performance...")
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        df = engine.load_and_prepare_data()
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        loading_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"⏱️ Data loading time: {loading_time:.2f}s")
        print(f"💾 Memory used: {memory_used:.2f}MB")
        
        if df is not None:
            print(f"✅ Data loaded: {df.shape}")
            self.assertLess(loading_time, self.performance_thresholds['data_loading_time'])
            self.assertLess(memory_used, self.performance_thresholds['memory_usage_mb'])
        else:
            print("⚠️ Data loading returned None")
            
    def test_feature_preparation_performance(self):
        """Test feature preparation performance"""
        print("\n🧪 Testing feature preparation performance...")
        
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        df = engine.load_and_prepare_data()
        
        if df is not None:
            start_time = time.time()
            start_memory = self.get_memory_usage()
            
            X, y = engine.prepare_features(df)
            
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            prep_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            print(f"⏱️ Feature preparation time: {prep_time:.2f}s")
            print(f"💾 Memory used: {memory_used:.2f}MB")
            
            if X is not None and y is not None:
                print(f"✅ Features prepared: X={X.shape}, y={y.shape}")
                self.assertLess(prep_time, self.performance_thresholds['feature_preparation_time'])
                self.assertLess(memory_used, self.performance_thresholds['memory_usage_mb'])
            else:
                print("⚠️ Feature preparation returned None")
        else:
            print("⚠️ Skipping feature preparation test due to data loading failure")
            
    def test_model_training_performance(self):
        """Test model training performance"""
        print("\n🧪 Testing model training performance...")
        
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        df = engine.load_and_prepare_data()
        
        if df is not None:
            X, y = engine.prepare_features(df)
            
            if X is not None and y is not None:
                start_time = time.time()
                start_memory = self.get_memory_usage()
                
                success = engine.train_models(X, y)
                
                end_time = time.time()
                end_memory = self.get_memory_usage()
                
                training_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                print(f"⏱️ Model training time: {training_time:.2f}s")
                print(f"💾 Memory used: {memory_used:.2f}MB")
                
                if success:
                    print("✅ Model training completed")
                    self.assertLess(training_time, self.performance_thresholds['model_training_time'])
                    self.assertLess(memory_used, self.performance_thresholds['memory_usage_mb'])
                else:
                    print("⚠️ Model training failed")
            else:
                print("⚠️ Skipping model training test due to feature preparation failure")
        else:
            print("⚠️ Skipping model training test due to data loading failure")
            
    def test_prediction_performance(self):
        """Test prediction generation performance"""
        print("\n🧪 Testing prediction performance...")
        
        engine = UnifiedPredictionEngine(self.test_ticker, mode=self.test_mode, interactive=False)
        df = engine.load_and_prepare_data()
        
        if df is not None:
            X, y = engine.prepare_features(df)
            
            if X is not None and y is not None:
                success = engine.train_models(X, y)
                
                if success:
                    start_time = time.time()
                    start_memory = self.get_memory_usage()
                    
                    predictions, multi_day = engine.generate_predictions(X, 5)
                    
                    end_time = time.time()
                    end_memory = self.get_memory_usage()
                    
                    prediction_time = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    print(f"⏱️ Prediction time: {prediction_time:.2f}s")
                    print(f"💾 Memory used: {memory_used:.2f}MB")
                    
                    if predictions is not None:
                        print(f"✅ Predictions generated: {len(predictions)} models")
                        self.assertLess(prediction_time, self.performance_thresholds['prediction_time'])
                        self.assertLess(memory_used, self.performance_thresholds['memory_usage_mb'])
                    else:
                        print("⚠️ Prediction generation failed")
                else:
                    print("⚠️ Skipping prediction test due to model training failure")
            else:
                print("⚠️ Skipping prediction test due to feature preparation failure")
        else:
            print("⚠️ Skipping prediction test due to data loading failure")
            
    def test_concurrent_processing(self):
        """Test concurrent processing performance"""
        print("\n🧪 Testing concurrent processing...")
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def process_ticker(ticker):
            """Process a single ticker"""
            try:
                engine = UnifiedPredictionEngine(ticker, mode="simple", interactive=False)
                df = engine.load_and_prepare_data()
                
                if df is not None:
                    X, y = engine.prepare_features(df)
                    if X is not None and y is not None:
                        success = engine.train_models(X, y)
                        if success:
                            predictions, _ = engine.generate_predictions(X, 3)
                            results_queue.put((ticker, True, predictions is not None))
                            return
                
                results_queue.put((ticker, False, False))
            except Exception as e:
                results_queue.put((ticker, False, str(e)))
        
        # Test with multiple tickers
        test_tickers = ["RELIANCE", "AAPL", "TCS"]
        threads = []
        
        start_time = time.time()
        
        for ticker in test_tickers:
            thread = threading.Thread(target=process_ticker, args=(ticker,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        print(f"⏱️ Concurrent processing time: {concurrent_time:.2f}s")
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        successful = sum(1 for _, success, _ in results if success)
        print(f"✅ Successful concurrent processes: {successful}/{len(test_tickers)}")
        
        self.assertGreater(successful, 0)  # At least one should succeed


if __name__ == '__main__':
    unittest.main()
