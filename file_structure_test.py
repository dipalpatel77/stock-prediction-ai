#!/usr/bin/env python3
"""
File Structure Test
Test file structure and basic content validation
"""

import os
import sys
from datetime import datetime
from pathlib import Path

def test_file_structure():
    """Test file structure and content"""
    print("🚀 FILE STRUCTURE TESTING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {}
    
    # Test main directory structure
    print("📁 Testing Main Directory Structure")
    print("-" * 40)
    
    main_dir = "main"
    if not os.path.exists(main_dir):
        print("❌ Main directory not found")
        return False
    
    print("✅ Main directory exists")
    
    # Test subdirectories
    subdirs = ["pipeline", "services", "interfaces", "utils"]
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        if os.path.exists(subdir_path):
            print(f"✅ {subdir} directory exists")
            test_results[f"{subdir}_dir"] = True
        else:
            print(f"❌ {subdir} directory not found")
            test_results[f"{subdir}_dir"] = False
    
    # Test main files
    print("\n📄 Testing Main Files")
    print("-" * 40)
    
    main_files = [
        "main.py",
        "unified_analysis_pipeline.py",
        "unified_analysis_pipeline_backup.py"
    ]
    
    for file in main_files:
        file_path = os.path.join(main_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} exists ({file_size:,} bytes)")
            test_results[f"{file}_file"] = True
        else:
            print(f"❌ {file} not found")
            test_results[f"{file}_file"] = False
    
    # Test pipeline files
    print("\n🔧 Testing Pipeline Files")
    print("-" * 40)
    
    pipeline_files = [
        "base_pipeline.py",
        "core_pipeline.py",
        "data_processor.py",
        "model_trainer.py",
        "prediction_generator.py",
        "strategy_analyzer.py",
        "async_pipeline_orchestrator.py"
    ]
    
    for file in pipeline_files:
        file_path = os.path.join(main_dir, "pipeline", file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} exists ({file_size:,} bytes)")
            test_results[f"pipeline_{file}"] = True
        else:
            print(f"❌ {file} not found")
            test_results[f"pipeline_{file}"] = False
    
    # Test services files
    print("\n🛠️ Testing Services Files")
    print("-" * 40)
    
    services_files = [
        "database_manager.py",
        "api_coordinator.py",
        "angel_one_manager.py",
        "data_service_wrapper.py",
        "smart_data_fetcher.py",
        "advanced_cache_manager.py",
        "ml_optimizer.py",
        "monitoring_dashboard.py",
        "auto_scaler.py"
    ]
    
    for file in services_files:
        file_path = os.path.join(main_dir, "services", file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} exists ({file_size:,} bytes)")
            test_results[f"services_{file}"] = True
        else:
            print(f"❌ {file} not found")
            test_results[f"services_{file}"] = False
    
    # Test interfaces files
    print("\n🖥️ Testing Interfaces Files")
    print("-" * 40)
    
    interfaces_files = [
        "user_interface.py",
        "angel_one_interface.py",
        "input_validator.py",
        "interactive_selector.py"
    ]
    
    for file in interfaces_files:
        file_path = os.path.join(main_dir, "interfaces", file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} exists ({file_size:,} bytes)")
            test_results[f"interfaces_{file}"] = True
        else:
            print(f"❌ {file} not found")
            test_results[f"interfaces_{file}"] = False
    
    # Test utils files
    print("\n🔧 Testing Utils Files")
    print("-" * 40)
    
    utils_files = [
        "service_manager.py",
        "service_coordinator.py",
        "error_handler.py",
        "validators.py",
        "formatters.py",
        "pipeline_logger.py"
    ]
    
    for file in utils_files:
        file_path = os.path.join(main_dir, "utils", file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file} exists ({file_size:,} bytes)")
            test_results[f"utils_{file}"] = True
        else:
            print(f"❌ {file} not found")
            test_results[f"utils_{file}"] = False
    
    # Test file content validation
    print("\n📝 Testing File Content")
    print("-" * 40)
    
    # Test main.py content
    main_py_path = os.path.join(main_dir, "main.py")
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "def main():" in content:
                print("✅ main.py contains main function")
                test_results["main_py_content"] = True
            else:
                print("❌ main.py missing main function")
                test_results["main_py_content"] = False
    
    # Test unified_analysis_pipeline.py content
    pipeline_py_path = os.path.join(main_dir, "unified_analysis_pipeline.py")
    if os.path.exists(pipeline_py_path):
        with open(pipeline_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class UnifiedAnalysisPipeline" in content:
                print("✅ unified_analysis_pipeline.py contains main class")
                test_results["pipeline_py_content"] = True
            else:
                print("❌ unified_analysis_pipeline.py missing main class")
                test_results["pipeline_py_content"] = False
    
    # Test data_processor.py content
    data_processor_path = os.path.join(main_dir, "pipeline", "data_processor.py")
    if os.path.exists(data_processor_path):
        with open(data_processor_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class DataProcessor" in content:
                print("✅ data_processor.py contains main class")
                test_results["data_processor_content"] = True
            else:
                print("❌ data_processor.py missing main class")
                test_results["data_processor_content"] = False
    
    # Test database_manager.py content
    db_manager_path = os.path.join(main_dir, "services", "database_manager.py")
    if os.path.exists(db_manager_path):
        with open(db_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class DatabaseManager" in content:
                print("✅ database_manager.py contains main class")
                test_results["db_manager_content"] = True
            else:
                print("❌ database_manager.py missing main class")
                test_results["db_manager_content"] = False
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 FILE STRUCTURE TEST SUMMARY")
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
        print("   🎉 ALL FILES PRESENT AND VALID!")
    elif passed_tests >= total_tests * 0.8:
        print("   ⚠️ MOSTLY COMPLETE - MINOR ISSUES")
    else:
        print("   🚨 SIGNIFICANT ISSUES DETECTED")
    
    print(f"\n🚀 FILE STRUCTURE TESTING COMPLETED!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return test_results

if __name__ == "__main__":
    test_file_structure()
