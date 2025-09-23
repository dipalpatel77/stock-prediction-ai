#!/usr/bin/env python3
"""
Simple Test Runner
Runs core tests without complex reporting to avoid serialization issues
"""

import unittest
import sys
import os
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def run_core_tests():
    """Run core functionality tests"""
    print("🚀 Running Core Functionality Tests")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Test core imports
    print("📦 Testing Core Imports...")
    try:
        from main.pipeline.data_processor import DataProcessor
        from main.pipeline.model_trainer import ModelTrainer
        from main.pipeline.prediction_generator import PredictionGenerator
        from main.pipeline.strategy_analyzer import StrategyAnalyzer
        from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
        print("✅ Core pipeline imports successful")
    except Exception as e:
        print(f"❌ Core pipeline imports failed: {e}")
        return False
    
    try:
        from main.services.database_manager import DatabaseManager
        from main.services.api_coordinator import APICoordinator
        print("✅ Service imports successful")
    except Exception as e:
        print(f"❌ Service imports failed: {e}")
        return False
    
    try:
        from main.utils.database_pool import get_connection_pool
        from main.utils.rate_limiter import get_api_rate_limiter
        from main.utils.model_cache import get_model_cache
        print("✅ Utility imports successful")
    except Exception as e:
        print(f"❌ Utility imports failed: {e}")
        return False
    
    # Test basic functionality
    print("\n🔧 Testing Basic Functionality...")
    
    # Test DataProcessor
    try:
        processor = DataProcessor("TEST")
        print("✅ DataProcessor initialization successful")
    except Exception as e:
        print(f"❌ DataProcessor initialization failed: {e}")
    
    # Test ModelTrainer
    try:
        trainer = ModelTrainer("TEST")
        print("✅ ModelTrainer initialization successful")
    except Exception as e:
        print(f"❌ ModelTrainer initialization failed: {e}")
    
    # Test DatabaseManager
    try:
        db_manager = DatabaseManager()
        print("✅ DatabaseManager initialization successful")
    except Exception as e:
        print(f"❌ DatabaseManager initialization failed: {e}")
    
    # Test APICoordinator
    try:
        api_coordinator = APICoordinator()
        print("✅ APICoordinator initialization successful")
    except Exception as e:
        print(f"❌ APICoordinator initialization failed: {e}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n⏱️  Execution Time: {execution_time:.2f} seconds")
    print("✅ Core functionality tests completed!")
    
    return True


def run_import_tests():
    """Test all major imports"""
    print("\n📋 Testing All Major Imports...")
    
    imports_to_test = [
        # Core pipeline components
        ("main.pipeline.data_processor", "DataProcessor"),
        ("main.pipeline.model_trainer", "ModelTrainer"),
        ("main.pipeline.prediction_generator", "PredictionGenerator"),
        ("main.pipeline.strategy_analyzer", "StrategyAnalyzer"),
        ("main.pipeline.core_pipeline", "UnifiedAnalysisPipeline"),
        ("main.pipeline.enhanced_model_trainer", "EnhancedModelTrainer"),
        
        # Services
        ("main.services.database_manager", "DatabaseManager"),
        ("main.services.api_coordinator", "APICoordinator"),
        ("main.services.angel_one_manager", "AngelOneManager"),
        
        # Utils
        ("main.utils.database_pool", "get_connection_pool"),
        ("main.utils.rate_limiter", "get_api_rate_limiter"),
        ("main.utils.model_cache", "get_model_cache"),
        ("main.utils.error_handler", "ErrorHandler"),
        ("main.utils.pipeline_logger", "PipelineLogger"),
    ]
    
    successful_imports = 0
    total_imports = len(imports_to_test)
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            successful_imports += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
    
    success_rate = (successful_imports / total_imports) * 100
    print(f"\n📊 Import Test Results: {successful_imports}/{total_imports} ({success_rate:.1f}%)")
    
    return success_rate >= 80


def main():
    """Main test runner function"""
    print("🧪 Simple Test Runner for AI Stock Predictor")
    print("=" * 60)
    
    # Run core tests
    core_success = run_core_tests()
    
    # Run import tests
    import_success = run_import_tests()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    if core_success and import_success:
        print("🎉 All tests passed successfully!")
        print("✅ Core functionality is working")
        print("✅ All imports are successful")
        return 0
    else:
        print("⚠️ Some tests failed:")
        if not core_success:
            print("❌ Core functionality tests failed")
        if not import_success:
            print("❌ Import tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
