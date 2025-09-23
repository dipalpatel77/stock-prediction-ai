#!/usr/bin/env python3
"""
Simple test for parallel model training functionality
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add the main directory to the path
sys.path.append('main')

def create_sample_data(n_samples=1000, n_features=20):
    """Create sample data for testing"""
    np.random.seed(42)
    
    # Generate synthetic stock data
    data = {
        'Close': np.random.randn(n_samples).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, n_samples),
        'SMA_20': np.random.randn(n_samples).cumsum() + 100,
        'RSI': np.random.uniform(20, 80, n_samples),
        'MACD': np.random.randn(n_samples),
        'BB_upper': np.random.randn(n_samples).cumsum() + 110,
        'BB_lower': np.random.randn(n_samples).cumsum() + 90,
        'EMA_12': np.random.randn(n_samples).cumsum() + 100,
        'EMA_26': np.random.randn(n_samples).cumsum() + 100,
        'Stochastic': np.random.uniform(0, 100, n_samples),
        'Williams_R': np.random.uniform(-100, 0, n_samples),
        'CCI': np.random.uniform(-200, 200, n_samples),
        'ATR': np.random.uniform(0.5, 5.0, n_samples),
        'ADX': np.random.uniform(0, 50, n_samples),
        'OBV': np.random.randint(-1000, 1000, n_samples),
        'MFI': np.random.uniform(0, 100, n_samples),
        'ROC': np.random.uniform(-10, 10, n_samples),
        'MOM': np.random.uniform(-5, 5, n_samples),
        'TRIX': np.random.uniform(-2, 2, n_samples),
        'UO': np.random.uniform(0, 100, n_samples)
    }
    
    return pd.DataFrame(data)

def test_parallel_training():
    """Test parallel model training"""
    print("🧪 Testing Parallel Model Training")
    print("=" * 50)
    
    try:
        from main.pipeline.model_trainer import ModelTrainer
        
        # Create sample data
        print("📊 Creating sample data...")
        data = create_sample_data(n_samples=500, n_features=20)
        print(f"   Data shape: {data.shape}")
        
        # Test configuration
        config = {
            'random_state': 42,
            'cv_folds': 3,
            'test_size': 0.2,
            'enable_parallel_training': True,
            'max_workers': 4,
            'use_process_pool': True,
            'enable_cross_validation': True,
            'enable_ensemble': True
        }
        
        # Initialize model trainer
        print("🤖 Initializing Model Trainer...")
        trainer = ModelTrainer("TEST", config)
        
        # Test parallel training
        print("\n🚀 Testing Parallel Training...")
        start_time = time.time()
        
        results = trainer.execute(data)
        
        if results['success']:
            print(f"✅ Parallel training completed successfully!")
            print(f"   Models trained: {results['models_trained']}")
            print(f"   Best model: {results['best_model_name']}")
            print(f"   Best score: {results['best_model_score']:.4f}")
            print(f"   Total time: {time.time() - start_time:.2f}s")
            
            # Display model performance
            print("\n📈 Model Performance:")
            for name, model_data in results['models'].items():
                metrics = model_data['metrics']
                print(f"   {name}: R²={metrics['test_r2']:.4f}, Time={metrics['training_time']:.2f}s")
                
        else:
            print(f"❌ Parallel training failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception during parallel training: {e}")
        import traceback
        traceback.print_exc()

def test_sequential_vs_parallel():
    """Test sequential vs parallel performance"""
    print("\n🏁 Benchmarking Sequential vs Parallel Training")
    print("=" * 50)
    
    try:
        from main.pipeline.model_trainer import ModelTrainer
        
        # Create sample data
        data = create_sample_data(n_samples=300, n_features=15)
        
        # Configuration for parallel
        parallel_config = {
            'random_state': 42,
            'cv_folds': 3,
            'test_size': 0.2,
            'enable_parallel_training': True,
            'max_workers': 4,
            'use_process_pool': True,
            'enable_cross_validation': False,  # Disable for faster testing
            'enable_ensemble': False
        }
        
        # Configuration for sequential
        sequential_config = {
            'random_state': 42,
            'cv_folds': 3,
            'test_size': 0.2,
            'enable_parallel_training': False,
            'max_workers': 1,
            'use_process_pool': False,
            'enable_cross_validation': False,
            'enable_ensemble': False
        }
        
        # Test sequential
        print("🔄 Testing Sequential Training...")
        sequential_trainer = ModelTrainer("TEST_SEQ", sequential_config)
        start_time = time.time()
        sequential_results = sequential_trainer.execute(data)
        sequential_time = time.time() - start_time
        
        if sequential_results['success']:
            print(f"   ✅ Sequential: {sequential_time:.2f}s ({sequential_results['models_trained']} models)")
        else:
            print(f"   ❌ Sequential failed: {sequential_results.get('error')}")
            return
        
        # Test parallel
        print("🚀 Testing Parallel Training...")
        parallel_trainer = ModelTrainer("TEST_PAR", parallel_config)
        start_time = time.time()
        parallel_results = parallel_trainer.execute(data)
        parallel_time = time.time() - start_time
        
        if parallel_results['success']:
            print(f"   ✅ Parallel: {parallel_time:.2f}s ({parallel_results['models_trained']} models)")
            
            # Calculate performance metrics
            time_saved = sequential_time - parallel_time
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            efficiency = speedup / parallel_config['max_workers']
            
            print(f"\n📊 Performance Comparison:")
            print(f"   Sequential Time: {sequential_time:.2f}s")
            print(f"   Parallel Time: {parallel_time:.2f}s")
            print(f"   Time Saved: {time_saved:.2f}s ({time_saved/sequential_time*100:.1f}%)")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Efficiency: {efficiency:.2f}")
            
        else:
            print(f"   ❌ Parallel failed: {parallel_results.get('error')}")
            
    except Exception as e:
        print(f"❌ Exception during benchmark: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🚀 Parallel Model Training Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Basic parallel training
        test_parallel_training()
        
        # Test 2: Sequential vs Parallel benchmark
        test_sequential_vs_parallel()
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
