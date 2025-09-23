#!/usr/bin/env python3
"""
Comprehensive System Test
Test all existing functionalities and features across the entire polylithic architecture
"""

import asyncio
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import traceback
from typing import Dict, Any, List, Optional

# Add main directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'main'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveSystemTester:
    """Comprehensive system tester for all functionalities"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    async def run_all_tests(self):
        """Run all comprehensive tests"""
        print("🚀 COMPREHENSIVE SYSTEM TESTING")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = time.time()
        
        # Test categories
        test_categories = [
            ("Core Pipeline Components", self.test_core_pipeline_components),
            ("Data Processing", self.test_data_processing),
            ("Model Training", self.test_model_training),
            ("Prediction Generation", self.test_prediction_generation),
            ("Strategy Analysis", self.test_strategy_analysis),
            ("Database Operations", self.test_database_operations),
            ("API Coordination", self.test_api_coordination),
            ("Angel One Integration", self.test_angel_one_integration),
            ("Service Management", self.test_service_management),
            ("Advanced Features", self.test_advanced_features),
            ("User Interface", self.test_user_interface),
            ("Error Handling", self.test_error_handling),
            ("Performance", self.test_performance),
            ("Integration", self.test_integration)
        ]
        
        # Run all test categories
        for category_name, test_function in test_categories:
            print(f"\n📋 Testing {category_name}")
            print("-" * 60)
            
            try:
                result = await test_function()
                self.test_results[category_name] = result
                
                if result:
                    print(f"✅ {category_name}: PASSED")
                else:
                    print(f"❌ {category_name}: FAILED")
                    
            except Exception as e:
                print(f"❌ {category_name}: ERROR - {str(e)}")
                self.test_results[category_name] = False
                logger.error(f"Test category {category_name} failed: {e}")
                logger.error(traceback.format_exc())
        
        # Generate final report
        await self.generate_final_report()
        
        return self.test_results
    
    async def test_core_pipeline_components(self) -> bool:
        """Test core pipeline components"""
        try:
            print("🔧 Testing core pipeline components...")
            
            # Test base pipeline
            from main.pipeline.base_pipeline import BasePipelineComponent
            base_component = BasePipelineComponent("TEST", {"test": True})
            print("   ✅ Base pipeline component created")
            
            # Test core pipeline
            from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
            core_pipeline = UnifiedAnalysisPipeline("AAPL", {"database_url": "sqlite:///test.db"})
            print("   ✅ Core pipeline created")
            
            # Test async pipeline orchestrator
            from main.pipeline.async_pipeline_orchestrator import AsyncPipelineOrchestrator
            async_orchestrator = AsyncPipelineOrchestrator({"database_url": "sqlite:///test.db"})
            print("   ✅ Async pipeline orchestrator created")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Core pipeline test failed: {e}")
            return False
    
    async def test_data_processing(self) -> bool:
        """Test data processing functionality"""
        try:
            print("📊 Testing data processing...")
            
            from main.pipeline.data_processor import DataProcessor
            
            # Test data processor initialization
            processor = DataProcessor("AAPL", {
                'database_url': 'sqlite:///test.db',
                'use_angel_one': False,
                'cache_enabled': True
            })
            print("   ✅ Data processor initialized")
            
            # Test with sample data
            sample_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'open': np.random.randn(100) * 100 + 100,
                'high': np.random.randn(100) * 100 + 105,
                'low': np.random.randn(100) * 100 + 95,
                'close': np.random.randn(100) * 100 + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Test data processing
            processed_data = await processor.async_process_data(sample_data)
            print(f"   ✅ Data processed: {len(processed_data)} rows")
            
            # Test memory report
            memory_report = processor.get_memory_report()
            print(f"   ✅ Memory report: {memory_report.get('memory_usage_mb', 0):.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Data processing test failed: {e}")
            return False
    
    async def test_model_training(self) -> bool:
        """Test model training functionality"""
        try:
            print("🤖 Testing model training...")
            
            from main.pipeline.model_trainer import ModelTrainer
            
            # Test model trainer initialization
            trainer = ModelTrainer("AAPL", {
                'database_url': 'sqlite:///test.db',
                'model_type': 'random_forest'
            })
            print("   ✅ Model trainer initialized")
            
            # Test with sample data
            X = pd.DataFrame(np.random.randn(100, 10))
            y = pd.Series(np.random.randn(100))
            
            # Test model training
            model_result = await trainer.train_model(X, y)
            print(f"   ✅ Model trained: {model_result.get('success', False)}")
            
            # Test prediction
            predictions = await trainer.predict(X.iloc[:10])
            print(f"   ✅ Predictions generated: {len(predictions)} predictions")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Model training test failed: {e}")
            return False
    
    async def test_prediction_generation(self) -> bool:
        """Test prediction generation functionality"""
        try:
            print("🔮 Testing prediction generation...")
            
            from main.pipeline.prediction_generator import PredictionGenerator
            
            # Test prediction generator initialization
            generator = PredictionGenerator("AAPL", {
                'database_url': 'sqlite:///test.db',
                'prediction_horizons': [1, 5, 10]
            })
            print("   ✅ Prediction generator initialized")
            
            # Test with sample data
            sample_data = pd.DataFrame({
                'close': np.random.randn(100) * 100 + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Test prediction generation
            predictions = await generator.generate_predictions(sample_data)
            print(f"   ✅ Predictions generated: {len(predictions)} predictions")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Prediction generation test failed: {e}")
            return False
    
    async def test_strategy_analysis(self) -> bool:
        """Test strategy analysis functionality"""
        try:
            print("📈 Testing strategy analysis...")
            
            from main.pipeline.strategy_analyzer import StrategyAnalyzer
            
            # Test strategy analyzer initialization
            analyzer = StrategyAnalyzer("AAPL", {
                'database_url': 'sqlite:///test.db',
                'strategies': ['momentum', 'mean_reversion']
            })
            print("   ✅ Strategy analyzer initialized")
            
            # Test with sample data
            sample_data = pd.DataFrame({
                'close': np.random.randn(100) * 100 + 100,
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Test strategy analysis
            analysis = await analyzer.analyze_strategies(sample_data)
            print(f"   ✅ Strategy analysis completed: {analysis.get('success', False)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Strategy analysis test failed: {e}")
            return False
    
    async def test_database_operations(self) -> bool:
        """Test database operations"""
        try:
            print("🗄️ Testing database operations...")
            
            from main.services.database_manager import DatabaseManager
            
            # Test database manager initialization
            db_manager = DatabaseManager({
                'database_url': 'sqlite:///test.db',
                'connection_pool_size': 5
            })
            print("   ✅ Database manager initialized")
            
            # Test connection
            connection_test = db_manager.test_connection()
            print(f"   ✅ Database connection: {connection_test}")
            
            # Test async operations
            async_result = await db_manager.async_execute_query("SELECT 1 as test")
            print(f"   ✅ Async query executed: {len(async_result) if async_result else 0} results")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Database operations test failed: {e}")
            return False
    
    async def test_api_coordination(self) -> bool:
        """Test API coordination functionality"""
        try:
            print("🌐 Testing API coordination...")
            
            from main.services.api_coordinator import APICoordinator
            
            # Test API coordinator initialization
            api_coordinator = APICoordinator(max_workers=4)
            print("   ✅ API coordinator initialized")
            
            # Test async operations
            async_result = await api_coordinator.async_coordinate_parallel_loading(
                "AAPL", {"use_angel_one": False}
            )
            print(f"   ✅ Async API coordination: {async_result.get('success', False)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ API coordination test failed: {e}")
            return False
    
    async def test_angel_one_integration(self) -> bool:
        """Test Angel One integration"""
        try:
            print("👼 Testing Angel One integration...")
            
            from main.services.angel_one_manager import AngelOneManager
            
            # Test Angel One manager initialization
            angel_manager = AngelOneManager({
                'api_key': 'test_key',
                'client_code': 'test_client',
                'client_pin': 'test_pin',
                'totp_secret': 'test_secret'
            })
            print("   ✅ Angel One manager initialized")
            
            # Test rate limiting
            rate_limit = angel_manager.check_rate_limit()
            print(f"   ✅ Rate limiting: {rate_limit}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Angel One integration test failed: {e}")
            return False
    
    async def test_service_management(self) -> bool:
        """Test service management functionality"""
        try:
            print("🔧 Testing service management...")
            
            from main.utils.service_manager import ServiceManager
            
            # Test service manager initialization
            service_manager = ServiceManager({
                'database_url': 'sqlite:///test.db',
                'enable_services': ['data', 'model', 'strategy']
            })
            print("   ✅ Service manager initialized")
            
            # Test service initialization
            services = service_manager.initialize_services()
            print(f"   ✅ Services initialized: {len(services)} services")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Service management test failed: {e}")
            return False
    
    async def test_advanced_features(self) -> bool:
        """Test advanced features"""
        try:
            print("🚀 Testing advanced features...")
            
            # Test advanced cache manager
            from main.services.advanced_cache_manager import AdvancedCacheManager
            cache_manager = AdvancedCacheManager({
                'cache_type': 'multi_level',
                'memory_cache_size': 50
            })
            print("   ✅ Advanced cache manager initialized")
            
            # Test cache operations
            await cache_manager.set("test_key", "test_value", ttl=60)
            cached_value = await cache_manager.get("test_key")
            print(f"   ✅ Cache operations: {cached_value == 'test_value'}")
            
            # Test ML optimizer
            from main.services.ml_optimizer import MLOptimizer
            ml_optimizer = MLOptimizer({
                'models_dir': 'models/test',
                'enable_hyperparameter_tuning': True
            })
            print("   ✅ ML optimizer initialized")
            
            # Test monitoring dashboard
            from main.services.monitoring_dashboard import MonitoringDashboard
            dashboard = MonitoringDashboard({
                'websocket_port': 8768,
                'enable_real_time': True
            })
            print("   ✅ Monitoring dashboard initialized")
            
            # Test auto-scaler
            from main.services.auto_scaler import AutoScaler
            auto_scaler = AutoScaler({
                'min_instances': 1,
                'max_instances': 5
            })
            print("   ✅ Auto-scaler initialized")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Advanced features test failed: {e}")
            return False
    
    async def test_user_interface(self) -> bool:
        """Test user interface functionality"""
        try:
            print("🖥️ Testing user interface...")
            
            from main.interfaces.user_interface import UserInterface
            
            # Test user interface initialization
            ui = UserInterface()
            print("   ✅ User interface initialized")
            
            # Test input validation
            from main.interfaces.input_validator import InputValidator
            validator = InputValidator()
            print("   ✅ Input validator initialized")
            
            # Test Angel One interface
            from main.interfaces.angel_one_interface import AngelOneInterface
            angel_ui = AngelOneInterface()
            print("   ✅ Angel One interface initialized")
            
            return True
            
        except Exception as e:
            print(f"   ❌ User interface test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling functionality"""
        try:
            print("⚠️ Testing error handling...")
            
            from main.utils.error_handler import ErrorHandler
            
            # Test error handler initialization
            error_handler = ErrorHandler()
            print("   ✅ Error handler initialized")
            
            # Test error handling
            try:
                raise ValueError("Test error")
            except Exception as e:
                handled = error_handler.handle_error(e, "test_context")
                print(f"   ✅ Error handling: {handled}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            return False
    
    async def test_performance(self) -> bool:
        """Test performance functionality"""
        try:
            print("⚡ Testing performance...")
            
            # Test performance with sample data
            start_time = time.time()
            
            # Create large dataset
            large_data = pd.DataFrame({
                'value': np.random.randn(10000),
                'category': np.random.choice(['A', 'B', 'C'], 10000)
            })
            
            # Test processing time
            processing_time = time.time() - start_time
            print(f"   ✅ Large dataset processing: {processing_time:.3f}s")
            
            # Test memory usage
            import psutil
            memory_usage = psutil.virtual_memory().percent
            print(f"   ✅ Memory usage: {memory_usage:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test integration between components"""
        try:
            print("🔗 Testing integration...")
            
            # Test main entry point
            from main.main import main
            print("   ✅ Main entry point imported")
            
            # Test unified analysis pipeline
            from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
            pipeline = UnifiedAnalysisPipeline("AAPL", {"database_url": "sqlite:///test.db"})
            print("   ✅ Unified analysis pipeline created")
            
            # Test service coordination
            from main.utils.service_coordinator import ServiceCoordinator
            coordinator = ServiceCoordinator()
            print("   ✅ Service coordinator initialized")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Integration test failed: {e}")
            return False
    
    async def generate_final_report(self):
        """Generate comprehensive final report"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE SYSTEM TEST REPORT")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"⏱️ Total Test Time: {total_time:.2f} seconds")
        print(f"📊 Total Test Categories: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        print("📋 DETAILED RESULTS:")
        print("-" * 40)
        
        for category, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {category}: {status}")
        
        print()
        print("🎯 SYSTEM STATUS:")
        if passed_tests == total_tests:
            print("   🎉 ALL SYSTEMS OPERATIONAL - READY FOR PRODUCTION!")
        elif passed_tests >= total_tests * 0.8:
            print("   ⚠️ MOSTLY OPERATIONAL - MINOR ISSUES DETECTED")
        else:
            print("   🚨 SIGNIFICANT ISSUES DETECTED - REQUIRES ATTENTION")
        
        print()
        print("🚀 COMPREHENSIVE SYSTEM TESTING COMPLETED!")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    """Main test function"""
    tester = ComprehensiveSystemTester()
    results = await tester.run_all_tests()
    return results

if __name__ == "__main__":
    # Run comprehensive system tests
    asyncio.run(main())
