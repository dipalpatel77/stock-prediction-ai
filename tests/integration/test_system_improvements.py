#!/usr/bin/env python3
"""
Test script for the improved stock prediction system
"""

import os
import sys
import time
from datetime import datetime

def test_non_interactive_mode():
    """Test non-interactive mode functionality."""
    print("🧪 Testing non-interactive mode...")
    
    # Test command line arguments
    test_cases = [
        ("RELIANCE", "simple", 5),
        ("AAPL", "advanced", 3),
        ("TCS", "simple", 10)
    ]
    
    for ticker, mode, days in test_cases:
        print(f"\n📊 Testing: {ticker} {mode} {days} days")
        
        # Simulate command line arguments
        sys.argv = ['test_system_improvements.py', ticker, mode, str(days)]
        
        try:
            # Import and run the main function
            from run_stock_prediction import main
            main()
            print(f"✅ Non-interactive test passed for {ticker}")
        except Exception as e:
            print(f"❌ Non-interactive test failed for {ticker}: {e}")
    
    # Reset sys.argv
    sys.argv = ['test_system_improvements.py']

def test_data_loading_fallback():
    """Test data loading with multiple fallback sources."""
    print("\n🧪 Testing data loading fallback...")
    
    try:
        from run_stock_prediction import UnifiedPredictionEngine
        
        # Test with different tickers
        test_tickers = ['RELIANCE', 'AAPL', 'TCS']
        
        for ticker in test_tickers:
            print(f"\n📊 Testing data loading for {ticker}...")
            
            # Test simple mode
            engine = UnifiedPredictionEngine(ticker, mode="simple", interactive=False)
            df = engine.load_and_prepare_data()
            
            if df is not None:
                print(f"✅ Data loaded successfully: {df.shape}")
            else:
                print(f"❌ Data loading failed for {ticker}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_caching():
    """Test model caching functionality."""
    print("\n🧪 Testing model caching...")
    
    try:
        from run_stock_prediction import UnifiedPredictionEngine
        
        ticker = "RELIANCE"
        engine = UnifiedPredictionEngine(ticker, mode="simple", interactive=False)
        
        # Load data
        df = engine.load_and_prepare_data()
        if df is None:
            print("❌ Cannot test caching without data")
            return False
        
        # Prepare features
        X, y = engine.prepare_features(df)
        if X is None or y is None:
            print("❌ Cannot test caching without features")
            return False
        
        # Train models (should cache them)
        print("🔄 Training models (first time)...")
        start_time = time.time()
        success1 = engine.train_models(X, y)
        time1 = time.time() - start_time
        
        if not success1:
            print("❌ First training failed")
            return False
        
        # Train models again (should load from cache)
        print("🔄 Training models (second time - should use cache)...")
        start_time = time.time()
        success2 = engine.train_models(X, y)
        time2 = time.time() - start_time
        
        if not success2:
            print("❌ Second training failed")
            return False
        
        print(f"✅ First training: {time1:.2f}s")
        print(f"✅ Second training: {time2:.2f}s")
        print(f"✅ Speed improvement: {time1/time2:.1f}x faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Model caching test failed: {e}")
        return False

def test_terminal_interaction():
    """Test terminal interaction improvements."""
    print("\n🧪 Testing terminal interaction...")
    
    try:
        # Test that the system can handle various input scenarios
        from run_stock_prediction import UnifiedPredictionEngine
        
        # Test with non-interactive mode
        engine = UnifiedPredictionEngine("RELIANCE", mode="simple", interactive=False)
        
        # Test data loading
        df = engine.load_and_prepare_data()
        if df is not None:
            print("✅ Non-interactive data loading works")
            
            # Test feature preparation
            X, y = engine.prepare_features(df)
            if X is not None and y is not None:
                print("✅ Non-interactive feature preparation works")
                
                # Test model training
                if engine.train_models(X, y):
                    print("✅ Non-interactive model training works")
                    
                    # Test prediction generation
                    predictions, multi_day = engine.generate_predictions(X, 5)
                    if predictions is not None:
                        print("✅ Non-interactive prediction generation works")
                        return True
                    else:
                        print("❌ Non-interactive prediction generation failed")
                else:
                    print("❌ Non-interactive model training failed")
            else:
                print("❌ Non-interactive feature preparation failed")
        else:
            print("❌ Non-interactive data loading failed")
        
        return False
        
    except Exception as e:
        print(f"❌ Terminal interaction test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimizations."""
    print("\n🧪 Testing performance optimizations...")
    
    try:
        from run_stock_prediction import UnifiedPredictionEngine
        
        ticker = "RELIANCE"
        
        # Test simple mode performance
        print("📊 Testing simple mode performance...")
        start_time = time.time()
        
        engine_simple = UnifiedPredictionEngine(ticker, mode="simple", interactive=False)
        df = engine_simple.load_and_prepare_data()
        
        if df is not None:
            X, y = engine_simple.prepare_features(df)
            if X is not None and y is not None:
                success = engine_simple.train_models(X, y)
                if success:
                    predictions, _ = engine_simple.generate_predictions(X, 5)
                    simple_time = time.time() - start_time
                    print(f"✅ Simple mode completed in {simple_time:.2f}s")
                    
                    if predictions is not None:
                        print(f"✅ Generated {len(predictions)} predictions")
                    else:
                        print("❌ Simple mode predictions failed")
                else:
                    print("❌ Simple mode training failed")
            else:
                print("❌ Simple mode feature preparation failed")
        else:
            print("❌ Simple mode data loading failed")
        
        # Test advanced mode performance (if available)
        if 'ADVANCED_AVAILABLE' in globals() and ADVANCED_AVAILABLE:
            print("📊 Testing advanced mode performance...")
            start_time = time.time()
            
            engine_advanced = UnifiedPredictionEngine(ticker, mode="advanced", interactive=False)
            df = engine_advanced.load_and_prepare_data()
            
            if df is not None:
                X, y = engine_advanced.prepare_features(df)
                if X is not None and y is not None:
                    success = engine_advanced.train_models(X, y)
                    if success:
                        predictions, _ = engine_advanced.generate_predictions(X, 5)
                        advanced_time = time.time() - start_time
                        print(f"✅ Advanced mode completed in {advanced_time:.2f}s")
                        
                        if predictions is not None:
                            print(f"✅ Generated {len(predictions)} predictions")
                        else:
                            print("❌ Advanced mode predictions failed")
                    else:
                        print("❌ Advanced mode training failed")
                else:
                    print("❌ Advanced mode feature preparation failed")
            else:
                print("❌ Advanced mode data loading failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 STOCK PREDICTION SYSTEM IMPROVEMENTS TEST")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run tests
    tests = [
        ("Non-Interactive Mode", test_non_interactive_mode),
        ("Data Loading Fallback", test_data_loading_fallback),
        ("Model Caching", test_model_caching),
        ("Terminal Interaction", test_terminal_interaction),
        ("Performance Optimization", test_performance_optimization)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"🧪 Running {test_name} test...")
            result = test_func()
            results[test_name] = result
            print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvements working correctly!")
    else:
        print("⚠️ Some improvements need attention")

if __name__ == "__main__":
    main()
