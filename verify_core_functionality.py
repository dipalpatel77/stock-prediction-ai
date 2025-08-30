#!/usr/bin/env python3
"""
AI Stock Predictor - Core Functionality Verification
Tests essential components after cleanup
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing critical imports...")
    
    tests = [
        ("Core Data Service", "from core.data_service import DataService"),
        ("Core Model Service", "from core.model_service import ModelService"),
        ("Core Reporting Service", "from core.reporting_service import ReportingService"),
        ("Core Strategy Service", "from core.strategy_service import StrategyService"),
        ("Core Incremental Service", "from core.incremental_service import IncrementalService"),
        ("Angel One Config", "from core.angel_one_config import AngelOneConfig"),
        ("Angel One Downloader", "from core.angel_one_data_downloader import AngelOneDataDownloader"),
        ("Indian Stock Mapper", "from data_downloaders.indian_stock_mapper import load_angel_master"),
        ("Analysis Config", "from config.analysis_config import AnalysisConfig"),
        ("Unified Pipeline", "from unified_analysis_pipeline import UnifiedAnalysisPipeline"),
        ("Enhanced Runner", "from unified_analysis_pipeline import UnifiedAnalysisPipeline"),
        ("Stock Prediction", "from run_stock_prediction import StockPredictionEngine")
    ]
    
    failed_imports = []
    
    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            failed_imports.append((name, e))
    
    return failed_imports

def test_core_services():
    """Test core service instantiation."""
    print("\n🔍 Testing core service instantiation...")
    
    tests = [
        ("DataService", "DataService()"),
        ("ModelService", "ModelService()"),
        ("ReportingService", "ReportingService()"),
        ("StrategyService", "StrategyService()"),
        ("IncrementalService", "IncrementalService()"),
        ("AnalysisConfig", "AnalysisConfig()")
    ]
    
    failed_services = []
    
    for name, instantiation in tests:
        try:
            exec(f"service = {instantiation}")
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            failed_services.append((name, e))
    
    return failed_services

def test_angel_one_integration():
    """Test Angel One integration."""
    print("\n🔍 Testing Angel One integration...")
    
    try:
        from core.angel_one_config import AngelOneConfig
        from core.angel_one_data_downloader import AngelOneDataDownloader
        
        # Test config
        config = AngelOneConfig()
        print("   ✅ AngelOneConfig")
        
        # Test downloader (without authentication)
        downloader = AngelOneDataDownloader()
        print("   ✅ AngelOneDataDownloader")
        
        return []
    except Exception as e:
        print(f"   ❌ Angel One integration: {e}")
        return [("Angel One Integration", e)]

def test_indian_stock_mapper():
    """Test Indian Stock Mapper."""
    print("\n🔍 Testing Indian Stock Mapper...")
    
    try:
        from data_downloaders.indian_stock_mapper import load_angel_master, get_symbol_info
        
        # Test function availability
        print("   ✅ Indian Stock Mapper functions available")
        
        return []
    except Exception as e:
        print(f"   ❌ Indian Stock Mapper: {e}")
        return [("Indian Stock Mapper", e)]

def test_pipeline_creation():
    """Test pipeline creation."""
    print("\n🔍 Testing pipeline creation...")
    
    try:
        from unified_analysis_pipeline import UnifiedAnalysisPipeline
        
        # Test pipeline creation (without running)
        pipeline = UnifiedAnalysisPipeline("RELIANCE", max_workers=1)
        print("   ✅ UnifiedAnalysisPipeline")
        
        return []
    except Exception as e:
        print(f"   ❌ Pipeline creation: {e}")
        return [("Pipeline Creation", e)]

def test_enhanced_runner():
    """Test enhanced analysis runner (now part of unified pipeline)."""
    print("\n🔍 Testing enhanced analysis runner (unified pipeline)...")
    
    try:
        from unified_analysis_pipeline import UnifiedAnalysisPipeline
        
        # Test pipeline creation with enhanced features
        pipeline = UnifiedAnalysisPipeline("AAPL")
        print("   ✅ UnifiedAnalysisPipeline (with enhanced features)")
        
        return []
    except Exception as e:
        print(f"   ❌ Enhanced runner: {e}")
        return [("Enhanced Runner", e)]

def test_stock_prediction():
    """Test stock prediction engine."""
    print("\n🔍 Testing stock prediction engine...")
    
    try:
        from run_stock_prediction import StockPredictionEngine
        
        # Test engine creation (without running)
        engine = StockPredictionEngine()
        print("   ✅ StockPredictionEngine")
        
        return []
    except Exception as e:
        print(f"   ❌ Stock prediction: {e}")
        return [("Stock Prediction", e)]

def test_critical_files():
    """Test if critical files exist."""
    print("\n🔍 Testing critical file existence...")
    
    critical_files = [
        "unified_analysis_pipeline.py",
        "run_stock_prediction.py",
        # "enhanced_analysis_runner.py",  # Merged into unified_analysis_pipeline.py
        "requirements.txt",
        "README.md",
        ".gitignore",
        "LICENSE",
        "core/data_service.py",
        "core/model_service.py",
        "core/reporting_service.py",
        "core/strategy_service.py",
        "core/incremental_service.py",
        "core/angel_one_config.py",
        "core/angel_one_data_downloader.py",
        "data_downloaders/indian_stock_mapper.py",
        "config/analysis_config.py"
    ]
    
    missing_files = []
    
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (MISSING)")
            missing_files.append(file_path)
    
    return missing_files

def main():
    """Main verification function."""
    print("🚀 AI Stock Predictor - Core Functionality Verification")
    print("=" * 60)
    print("Testing essential components after cleanup...")
    print()
    
    all_errors = []
    
    # Test critical files
    missing_files = test_critical_files()
    if missing_files:
        all_errors.extend([("Missing File", file) for file in missing_files])
    
    # Test imports
    failed_imports = test_imports()
    all_errors.extend(failed_imports)
    
    # Test core services
    failed_services = test_core_services()
    all_errors.extend(failed_services)
    
    # Test Angel One integration
    angel_errors = test_angel_one_integration()
    all_errors.extend(angel_errors)
    
    # Test Indian Stock Mapper
    mapper_errors = test_indian_stock_mapper()
    all_errors.extend(mapper_errors)
    
    # Test pipeline creation
    pipeline_errors = test_pipeline_creation()
    all_errors.extend(pipeline_errors)
    
    # Test enhanced runner
    runner_errors = test_enhanced_runner()
    all_errors.extend(runner_errors)
    
    # Test stock prediction
    prediction_errors = test_stock_prediction()
    all_errors.extend(prediction_errors)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    if not all_errors:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Core functionality is intact after cleanup.")
        print("🚀 The project is ready for use.")
    else:
        print(f"⚠️ {len(all_errors)} issues found:")
        for error_type, error_detail in all_errors:
            print(f"   ❌ {error_type}: {error_detail}")
        
        print("\n🔧 RECOMMENDATIONS:")
        print("   1. Check if all critical files are present")
        print("   2. Verify Python environment and dependencies")
        print("   3. Ensure all imports are working correctly")
        print("   4. Test with a simple stock analysis")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
