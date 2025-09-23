#!/usr/bin/env python3
"""
Content Validation Test
Test the content and functionality of key files
"""

import os
import re
from datetime import datetime
from pathlib import Path

def test_content_validation():
    """Test content validation of key files"""
    print("🚀 CONTENT VALIDATION TESTING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {}
    
    # Test main.py content
    print("📄 Testing main.py Content")
    print("-" * 40)
    
    main_py_path = "main/main.py"
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for key functions
            if "def main():" in content:
                print("✅ main() function found")
                test_results["main_function"] = True
            else:
                print("❌ main() function not found")
                test_results["main_function"] = False
            
            # Check for argument parsing
            if "argparse" in content:
                print("✅ argparse import found")
                test_results["argparse"] = True
            else:
                print("❌ argparse import not found")
                test_results["argparse"] = False
            
            # Check for Angel One support
            if "angel_one" in content.lower():
                print("✅ Angel One support found")
                test_results["angel_one_support"] = True
            else:
                print("❌ Angel One support not found")
                test_results["angel_one_support"] = False
    
    # Test data_processor.py content
    print("\n🔧 Testing data_processor.py Content")
    print("-" * 40)
    
    data_processor_path = "main/pipeline/data_processor.py"
    if os.path.exists(data_processor_path):
        with open(data_processor_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class DataProcessor" in content:
                print("✅ DataProcessor class found")
                test_results["data_processor_class"] = True
            else:
                print("❌ DataProcessor class not found")
                test_results["data_processor_class"] = False
            
            # Check for async methods
            if "async def" in content:
                print("✅ Async methods found")
                test_results["data_processor_async"] = True
            else:
                print("❌ Async methods not found")
                test_results["data_processor_async"] = False
            
            # Check for memory optimization
            if "memory" in content.lower():
                print("✅ Memory optimization found")
                test_results["data_processor_memory"] = True
            else:
                print("❌ Memory optimization not found")
                test_results["data_processor_memory"] = False
    
    # Test database_manager.py content
    print("\n🗄️ Testing database_manager.py Content")
    print("-" * 40)
    
    db_manager_path = "main/services/database_manager.py"
    if os.path.exists(db_manager_path):
        with open(db_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class DatabaseManager" in content:
                print("✅ DatabaseManager class found")
                test_results["db_manager_class"] = True
            else:
                print("❌ DatabaseManager class not found")
                test_results["db_manager_class"] = False
            
            # Check for async methods
            if "async def" in content:
                print("✅ Async methods found")
                test_results["db_manager_async"] = True
            else:
                print("❌ Async methods not found")
                test_results["db_manager_async"] = False
            
            # Check for connection pooling
            if "connection_pool" in content.lower():
                print("✅ Connection pooling found")
                test_results["db_manager_pooling"] = True
            else:
                print("❌ Connection pooling not found")
                test_results["db_manager_pooling"] = False
    
    # Test api_coordinator.py content
    print("\n🌐 Testing api_coordinator.py Content")
    print("-" * 40)
    
    api_coordinator_path = "main/services/api_coordinator.py"
    if os.path.exists(api_coordinator_path):
        with open(api_coordinator_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class APICoordinator" in content:
                print("✅ APICoordinator class found")
                test_results["api_coordinator_class"] = True
            else:
                print("❌ APICoordinator class not found")
                test_results["api_coordinator_class"] = False
            
            # Check for rate limiting
            if "rate_limit" in content.lower():
                print("✅ Rate limiting found")
                test_results["api_coordinator_rate_limit"] = True
            else:
                print("❌ Rate limiting not found")
                test_results["api_coordinator_rate_limit"] = False
            
            # Check for caching
            if "cache" in content.lower():
                print("✅ Caching found")
                test_results["api_coordinator_cache"] = True
            else:
                print("❌ Caching not found")
                test_results["api_coordinator_cache"] = False
    
    # Test advanced_cache_manager.py content
    print("\n🗄️ Testing advanced_cache_manager.py Content")
    print("-" * 40)
    
    cache_manager_path = "main/services/advanced_cache_manager.py"
    if os.path.exists(cache_manager_path):
        with open(cache_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class AdvancedCacheManager" in content:
                print("✅ AdvancedCacheManager class found")
                test_results["cache_manager_class"] = True
            else:
                print("❌ AdvancedCacheManager class not found")
                test_results["cache_manager_class"] = False
            
            # Check for multi-level caching
            if "multi_level" in content.lower():
                print("✅ Multi-level caching found")
                test_results["cache_manager_multi_level"] = True
            else:
                print("❌ Multi-level caching not found")
                test_results["cache_manager_multi_level"] = False
            
            # Check for Redis support
            if "redis" in content.lower():
                print("✅ Redis support found")
                test_results["cache_manager_redis"] = True
            else:
                print("❌ Redis support not found")
                test_results["cache_manager_redis"] = False
    
    # Test ml_optimizer.py content
    print("\n🤖 Testing ml_optimizer.py Content")
    print("-" * 40)
    
    ml_optimizer_path = "main/services/ml_optimizer.py"
    if os.path.exists(ml_optimizer_path):
        with open(ml_optimizer_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class MLOptimizer" in content:
                print("✅ MLOptimizer class found")
                test_results["ml_optimizer_class"] = True
            else:
                print("❌ MLOptimizer class not found")
                test_results["ml_optimizer_class"] = False
            
            # Check for hyperparameter tuning
            if "hyperparameter" in content.lower():
                print("✅ Hyperparameter tuning found")
                test_results["ml_optimizer_hyperparameter"] = True
            else:
                print("❌ Hyperparameter tuning not found")
                test_results["ml_optimizer_hyperparameter"] = False
            
            # Check for ensemble methods
            if "ensemble" in content.lower():
                print("✅ Ensemble methods found")
                test_results["ml_optimizer_ensemble"] = True
            else:
                print("❌ Ensemble methods not found")
                test_results["ml_optimizer_ensemble"] = False
    
    # Test monitoring_dashboard.py content
    print("\n📊 Testing monitoring_dashboard.py Content")
    print("-" * 40)
    
    dashboard_path = "main/services/monitoring_dashboard.py"
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class MonitoringDashboard" in content:
                print("✅ MonitoringDashboard class found")
                test_results["dashboard_class"] = True
            else:
                print("❌ MonitoringDashboard class not found")
                test_results["dashboard_class"] = False
            
            # Check for WebSocket support
            if "websocket" in content.lower():
                print("✅ WebSocket support found")
                test_results["dashboard_websocket"] = True
            else:
                print("❌ WebSocket support not found")
                test_results["dashboard_websocket"] = False
            
            # Check for real-time monitoring
            if "real_time" in content.lower() or "realtime" in content.lower():
                print("✅ Real-time monitoring found")
                test_results["dashboard_realtime"] = True
            else:
                print("❌ Real-time monitoring not found")
                test_results["dashboard_realtime"] = False
    
    # Test auto_scaler.py content
    print("\n📈 Testing auto_scaler.py Content")
    print("-" * 40)
    
    auto_scaler_path = "main/services/auto_scaler.py"
    if os.path.exists(auto_scaler_path):
        with open(auto_scaler_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class AutoScaler" in content:
                print("✅ AutoScaler class found")
                test_results["auto_scaler_class"] = True
            else:
                print("❌ AutoScaler class not found")
                test_results["auto_scaler_class"] = False
            
            # Check for scaling logic
            if "scaling" in content.lower():
                print("✅ Scaling logic found")
                test_results["auto_scaler_scaling"] = True
            else:
                print("❌ Scaling logic not found")
                test_results["auto_scaler_scaling"] = False
            
            # Check for cost optimization
            if "cost" in content.lower():
                print("✅ Cost optimization found")
                test_results["auto_scaler_cost"] = True
            else:
                print("❌ Cost optimization not found")
                test_results["auto_scaler_cost"] = False
    
    # Test unified_analysis_pipeline.py content
    print("\n🔗 Testing unified_analysis_pipeline.py Content")
    print("-" * 40)
    
    pipeline_path = "main/unified_analysis_pipeline.py"
    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for main class
            if "class UnifiedAnalysisPipeline" in content:
                print("✅ UnifiedAnalysisPipeline class found")
                test_results["pipeline_class"] = True
            else:
                print("❌ UnifiedAnalysisPipeline class not found")
                test_results["pipeline_class"] = False
            
            # Check for Angel One integration
            if "angel_one" in content.lower():
                print("✅ Angel One integration found")
                test_results["pipeline_angel_one"] = True
            else:
                print("❌ Angel One integration not found")
                test_results["pipeline_angel_one"] = False
            
            # Check for async support
            if "async def" in content:
                print("✅ Async support found")
                test_results["pipeline_async"] = True
            else:
                print("❌ Async support not found")
                test_results["pipeline_async"] = False
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 CONTENT VALIDATION TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n📋 DETAILED RESULTS:")
    print("-" * 30)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print("\n🎯 SYSTEM STATUS:")
    if passed_tests == total_tests:
        print("   🎉 ALL CONTENT VALIDATION PASSED!")
    elif passed_tests >= total_tests * 0.8:
        print("   ⚠️ MOSTLY VALID - MINOR ISSUES")
    else:
        print("   🚨 SIGNIFICANT CONTENT ISSUES DETECTED")
    
    print(f"\n🚀 CONTENT VALIDATION TESTING COMPLETED!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return test_results

if __name__ == "__main__":
    test_content_validation()
