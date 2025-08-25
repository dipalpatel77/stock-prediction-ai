#!/usr/bin/env python3
"""
Simple test script for the stock prediction system
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def test_data_loading():
    """Test if we can load existing data."""
    print("🧪 Testing data loading...")
    
    # Check for existing data files
    data_files = [
        "data/RELIANCE_partA_partC_enhanced.csv",
        "data/AAPL_partA_partC_enhanced.csv",
        "data/TSLA_partA_partC_enhanced.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
            try:
                df = pd.read_csv(file_path)
                print(f"   📊 Shape: {df.shape}")
                print(f"   📅 Date range: {df.iloc[0, 0]} to {df.iloc[-1, 0]}")
            except Exception as e:
                print(f"   ❌ Error reading: {e}")
        else:
            print(f"❌ Missing: {file_path}")
    
    return True

def test_basic_prediction():
    """Test basic prediction functionality."""
    print("\n🧪 Testing basic prediction...")
    
    try:
        # Import the prediction engine
        from run_stock_prediction import UnifiedPredictionEngine
        
        # Test with RELIANCE
        print("📊 Testing with RELIANCE...")
        engine = UnifiedPredictionEngine('RELIANCE', mode='simple')
        
        # Test data loading
        df = engine.load_and_prepare_data()
        if df is not None:
            print(f"✅ Data loaded successfully: {df.shape}")
            
            # Test basic prediction
            X, y = engine.prepare_features(df)
            predictions = engine.generate_predictions(X, 3)
            if predictions:
                print("✅ Basic prediction successful!")
                return True
            else:
                print("❌ Basic prediction failed")
                return False
        else:
            print("❌ Data loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in basic prediction: {e}")
        return False

def test_advanced_prediction():
    """Test advanced prediction functionality."""
    print("\n🧪 Testing advanced prediction...")
    
    try:
        from run_stock_prediction import UnifiedPredictionEngine
        
        # Test with RELIANCE in advanced mode
        print("📊 Testing with RELIANCE (Advanced)...")
        engine = UnifiedPredictionEngine('RELIANCE', mode='advanced')
        
        # Test data loading
        df = engine.load_and_prepare_data()
        if df is not None:
            print(f"✅ Data loaded successfully: {df.shape}")
            
            # Test advanced prediction
            X, y = engine.prepare_features(df)
            predictions = engine.generate_predictions(X, 3)
            if predictions:
                print("✅ Advanced prediction successful!")
                return True
            else:
                print("❌ Advanced prediction failed")
                return False
        else:
            print("❌ Data loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in advanced prediction: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Stock Prediction System Test")
    print("=" * 50)
    
    # Test 1: Data loading
    data_ok = test_data_loading()
    
    # Test 2: Basic prediction
    basic_ok = test_basic_prediction()
    
    # Test 3: Advanced prediction
    advanced_ok = test_advanced_prediction()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    print(f"Data Loading: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"Basic Prediction: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Advanced Prediction: {'✅ PASS' if advanced_ok else '❌ FAIL'}")
    
    if data_ok and basic_ok and advanced_ok:
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
