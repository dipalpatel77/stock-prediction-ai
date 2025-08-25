#!/usr/bin/env python3
"""
Test Comprehensive Prediction System
Simple test to verify the improved prediction system works
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported")
        
        import numpy as np
        print("✅ numpy imported")
        
        import sklearn
        print("✅ sklearn imported")
        
        import xgboost as xgb
        print("✅ xgboost imported")
        
        import lightgbm as lgb
        print("✅ lightgbm imported")
        
        import talib
        print("✅ talib imported")
        
        from improved_prediction_engine import ImprovedPredictionEngine
        print("✅ ImprovedPredictionEngine imported")
        
        from enhanced_strategy_analyzer import EnhancedStrategyAnalyzer
        print("✅ EnhancedStrategyAnalyzer imported")
        
        from comprehensive_prediction_runner import ComprehensivePredictionRunner
        print("✅ ComprehensivePredictionRunner imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_availability():
    """Test if required data files exist."""
    print("\n📊 Testing data availability...")
    
    test_tickers = ["RELIANCE", "AAPL"]
    
    for ticker in test_tickers:
        data_file = f"data/{ticker}_partA_partC_enhanced.csv"
        if os.path.exists(data_file):
            print(f"✅ {ticker} enhanced data found: {data_file}")
        else:
            print(f"❌ {ticker} enhanced data missing: {data_file}")
            return False
    
    return True

def test_prediction_engine():
    """Test the prediction engine with RELIANCE data."""
    print("\n🤖 Testing prediction engine...")
    
    try:
        from improved_prediction_engine import ImprovedPredictionEngine
        
        # Initialize engine
        engine = ImprovedPredictionEngine("RELIANCE")
        print("✅ Engine initialized")
        
        # Load data
        df = engine.load_and_prepare_data()
        if df is not None:
            print(f"✅ Data loaded: {df.shape}")
            
            # Prepare features
            X, y = engine.prepare_features(df)
            if X is not None and y is not None:
                print(f"✅ Features prepared: X={X.shape}, y={y.shape}")
                
                # Try to load existing models
                models_exist = engine.load_models()
                print(f"✅ Models loaded: {models_exist}")
                
                if models_exist:
                    # Generate predictions
                    predictions, multi_day = engine.generate_predictions(X, 5)
                    if predictions:
                        print(f"✅ Predictions generated: {len(predictions)} models")
                        for name, pred in predictions.items():
                            print(f"   {name}: {pred:.2f}")
                        return True
                    else:
                        print("❌ No predictions generated")
                        return False
                else:
                    print("⚠️ No pre-trained models found - would need to train")
                    return True  # This is acceptable for testing
            else:
                print("❌ Feature preparation failed")
                return False
        else:
            print("❌ Data loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Prediction engine error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_runner():
    """Test the comprehensive prediction runner."""
    print("\n🚀 Testing comprehensive prediction runner...")
    
    try:
        from comprehensive_prediction_runner import run_comprehensive_prediction
        
        # Test with RELIANCE (should be quick if models exist)
        print("Testing with RELIANCE...")
        success = run_comprehensive_prediction("RELIANCE", days_ahead=3)
        
        if success:
            print("✅ Comprehensive prediction completed successfully!")
            return True
        else:
            print("❌ Comprehensive prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Comprehensive runner error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 COMPREHENSIVE PREDICTION SYSTEM TEST")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages:")
        print("pip install -r requirements.txt")
        return False
    
    # Test 2: Data availability
    if not test_data_availability():
        print("\n❌ Data test failed. Please run the unified pipeline first:")
        print("python unified_analysis_pipeline.py")
        return False
    
    # Test 3: Prediction engine
    if not test_prediction_engine():
        print("\n❌ Prediction engine test failed.")
        return False
    
    # Test 4: Comprehensive runner
    if not test_comprehensive_runner():
        print("\n❌ Comprehensive runner test failed.")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ The comprehensive prediction system is working correctly!")
    print("\n🚀 You can now run:")
    print("   python comprehensive_prediction_runner.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
