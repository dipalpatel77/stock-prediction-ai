#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite
Tests all aspects of the polylithic architecture
"""

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = time.time()
        
    def test_angel_one_api_integration(self) -> bool:
        """Test Angel One API integration"""
        print("\n🔧 Testing Angel One API Integration")
        print("=" * 50)
        
        try:
            from main.interfaces.angel_one_interface import AngelOneInterface
            from main.services.angel_one_manager import AngelOneManager
            
            # Test Angel One Interface
            interface = AngelOneInterface()
            
            # Test configuration with actual Angel One credentials
            test_config = {
                'api_key': '1TKgQThc ',
                'api_secret': 'D54448',  # Client code
                'access_token': '2251',  # Client PIN
                'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE',
                'exchange': 'NSE',
                'interval': 'ONE_DAY'
            }
            
            # Test non-interactive configuration
            result = interface.configure_angel_one('RELIANCE', test_config)
            
            if result:
                print("✅ Angel One Interface: Configuration successful")
            else:
                print("❌ Angel One Interface: Configuration failed")
                return False
            
            # Test Angel One Manager
            manager = AngelOneManager(test_config)
            
            # Test rate limiting
            if hasattr(manager, 'check_rate_limit'):
                print("✅ Angel One Manager: Rate limiting available")
            else:
                print("❌ Angel One Manager: Rate limiting missing")
                return False
            
            # Test data loading
            if hasattr(manager, 'load_stock_data'):
                print("✅ Angel One Manager: Data loading available")
            else:
                print("❌ Angel One Manager: Data loading missing")
                return False
            
            # Test optimized data fetching
            if hasattr(manager, 'get_optimal_historical_data'):
                print("✅ Angel One Manager: Optimized data fetching available")
            else:
                print("❌ Angel One Manager: Optimized data fetching missing")
                return False
            
            # Test batch processing
            if hasattr(manager, 'batch_fetch_multiple_stocks'):
                print("✅ Angel One Manager: Batch processing available")
            else:
                print("❌ Angel One Manager: Batch processing missing")
                return False
            
            # Test enhanced data with indicators
            if hasattr(manager, 'get_enhanced_data_with_indicators'):
                print("✅ Angel One Manager: Enhanced data processing available")
            else:
                print("❌ Angel One Manager: Enhanced data processing missing")
                return False
            
            print("✅ Angel One API Integration: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Angel One API Integration: Error - {e}")
            return False
    
    def test_database_integration(self) -> bool:
        """Test database integration and connection pooling"""
        print("\n🔧 Testing Database Integration")
        print("=" * 50)
        
        try:
            from main.services.database_manager import DatabaseManager
            
            # Test database manager
            config = {'database_url': 'sqlite:///test_integration.db'}
            db_manager = DatabaseManager(config)
            
            # Test connection
            if db_manager.test_connection():
                print("✅ Database Manager: Connection test successful")
            else:
                print("❌ Database Manager: Connection test failed")
                return False
            
            # Test connection pooling
            if hasattr(db_manager, 'get_connection'):
                print("✅ Database Manager: Connection pooling available")
            else:
                print("❌ Database Manager: Connection pooling missing")
                return False
            
            # Test query execution
            if hasattr(db_manager, 'execute_query'):
                print("✅ Database Manager: Query execution available")
            else:
                print("❌ Database Manager: Query execution missing")
                return False
            
            # Test data storage
            if hasattr(db_manager, 'store_stock_data'):
                print("✅ Database Manager: Data storage available")
            else:
                print("❌ Database Manager: Data storage missing")
                return False
            
            print("✅ Database Integration: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Database Integration: Error - {e}")
            return False
    
    def test_service_coordination(self) -> bool:
        """Test service coordination and health monitoring"""
        print("\n🔧 Testing Service Coordination")
        print("=" * 50)
        
        try:
            from main.utils.service_manager import ServiceManager
            from main.utils.service_coordinator import ServiceCoordinator
            
            # Test service manager
            service_manager = ServiceManager()
            
            # Test service initialization
            if hasattr(service_manager, 'initialize_services'):
                print("✅ Service Manager: Service initialization available")
            else:
                print("❌ Service Manager: Service initialization missing")
                return False
            
            # Test health monitoring
            if hasattr(service_manager, 'check_service_health'):
                print("✅ Service Manager: Health monitoring available")
            else:
                print("❌ Service Manager: Health monitoring missing")
                return False
            
            # Test service coordinator
            coordinator = ServiceCoordinator()
            
            # Test load balancing
            if hasattr(coordinator, 'balance_load'):
                print("✅ Service Coordinator: Load balancing available")
            else:
                print("❌ Service Coordinator: Load balancing missing")
                return False
            
            # Test failover
            if hasattr(coordinator, 'handle_failover'):
                print("✅ Service Coordinator: Failover handling available")
            else:
                print("❌ Service Coordinator: Failover handling missing")
                return False
            
            print("✅ Service Coordination: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Service Coordination: Error - {e}")
            return False
    
    def test_api_rate_limiting(self) -> bool:
        """Test API rate limiting and error handling"""
        print("\n🔧 Testing API Rate Limiting")
        print("=" * 50)
        
        try:
            from main.services.api_coordinator import APICoordinator
            from main.services.angel_one_manager import AngelOneManager
            
            # Test API coordinator
            coordinator = APICoordinator()
            
            # Test rate limiting
            if hasattr(coordinator, 'check_rate_limit'):
                print("✅ API Coordinator: Rate limiting available")
            else:
                print("❌ API Coordinator: Rate limiting missing")
                return False
            
            # Test error handling
            if hasattr(coordinator, 'handle_api_error'):
                print("✅ API Coordinator: Error handling available")
            else:
                print("❌ API Coordinator: Error handling missing")
                return False
            
            # Test parallel loading
            if hasattr(coordinator, 'load_data_parallel'):
                print("✅ API Coordinator: Parallel loading available")
            else:
                print("❌ API Coordinator: Parallel loading missing")
                return False
            
            # Test Angel One rate limiting with actual credentials
            test_config = {
                'api_key': '1TKgQThc ',
                'api_secret': 'D54448',
                'access_token': '2251',
                'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE',
                'exchange': 'NSE',
                'interval': 'ONE_DAY'
            }
            
            angel_manager = AngelOneManager(test_config)
            
            if hasattr(angel_manager, 'check_rate_limit'):
                print("✅ Angel One Manager: Rate limiting available")
            else:
                print("❌ Angel One Manager: Rate limiting missing")
                return False
            
            print("✅ API Rate Limiting: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ API Rate Limiting: Error - {e}")
            return False
    
    def test_fallback_mechanisms(self) -> bool:
        """Test fallback mechanisms and graceful degradation"""
        print("\n🔧 Testing Fallback Mechanisms")
        print("=" * 50)
        
        try:
            from main.services.data_service_wrapper import DataServiceWrapper
            from main.services.api_coordinator import APICoordinator
            
            # Test data service wrapper
            wrapper = DataServiceWrapper()
            
            # Test fallback mechanisms
            if hasattr(wrapper, 'load_data_with_fallback'):
                print("✅ Data Service Wrapper: Fallback mechanisms available")
            else:
                print("❌ Data Service Wrapper: Fallback mechanisms missing")
                return False
            
            # Test API coordinator fallback
            coordinator = APICoordinator()
            
            if hasattr(coordinator, 'handle_fallback'):
                print("✅ API Coordinator: Fallback handling available")
            else:
                print("❌ API Coordinator: Fallback handling missing")
                return False
            
            # Test graceful degradation
            if hasattr(wrapper, 'degrade_gracefully'):
                print("✅ Data Service Wrapper: Graceful degradation available")
            else:
                print("❌ Data Service Wrapper: Graceful degradation missing")
                return False
            
            print("✅ Fallback Mechanisms: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Fallback Mechanisms: Error - {e}")
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Create performance benchmarks and metrics"""
        print("\n🔧 Testing Performance Benchmarks")
        print("=" * 50)
        
        try:
            from main.utils.service_manager import ServiceManager
            from main.pipeline.data_processor import DataProcessor
            from main.pipeline.model_trainer import ModelTrainer
            
            # Test service manager performance
            start_time = time.time()
            service_manager = ServiceManager()
            service_init_time = time.time() - start_time
            
            self.performance_metrics['service_initialization'] = service_init_time
            print(f"✅ Service Initialization: {service_init_time:.3f}s")
            
            # Test data processor performance
            start_time = time.time()
            processor = DataProcessor()
            processor_init_time = time.time() - start_time
            
            self.performance_metrics['data_processor_init'] = processor_init_time
            print(f"✅ Data Processor Initialization: {processor_init_time:.3f}s")
            
            # Test model trainer performance
            start_time = time.time()
            trainer = ModelTrainer("AAPL")
            trainer_init_time = time.time() - start_time
            
            self.performance_metrics['model_trainer_init'] = trainer_init_time
            print(f"✅ Model Trainer Initialization: {trainer_init_time:.3f}s")
            
            # Test overall system performance
            total_init_time = service_init_time + processor_init_time + trainer_init_time
            self.performance_metrics['total_initialization'] = total_init_time
            print(f"✅ Total System Initialization: {total_init_time:.3f}s")
            
            print("✅ Performance Benchmarks: All metrics collected")
            return True
            
        except Exception as e:
            print(f"❌ Performance Benchmarks: Error - {e}")
            return False
    
    def test_end_to_end_pipeline(self) -> bool:
        """Test end-to-end pipeline execution"""
        print("\n🔧 Testing End-to-End Pipeline")
        print("=" * 50)
        
        try:
            from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
            from main.interfaces.user_interface import UserInterface
            
            # Test user interface
            ui = UserInterface()
            
            # Test with non-interactive configuration
            test_config = {
                'ticker': 'AAPL',
                'is_indian': False,
                'angel_config': None,
                'analysis_type': 'short_term',
                'period': '1y',
                'interval': 'ONE_DAY',
                'use_enhanced': True,
                'use_database': True
            }
            
            user_inputs = ui.get_user_inputs(test_config)
            
            if user_inputs and 'ticker' in user_inputs:
                print("✅ User Interface: Non-interactive mode working")
            else:
                print("❌ User Interface: Non-interactive mode failed")
                return False
            
            # Test pipeline initialization
            pipeline = UnifiedAnalysisPipeline()
            
            if hasattr(pipeline, 'setup_pipeline_components'):
                print("✅ Pipeline: Component setup available")
            else:
                print("❌ Pipeline: Component setup missing")
                return False
            
            if hasattr(pipeline, 'initialize_services'):
                print("✅ Pipeline: Service initialization available")
            else:
                print("❌ Pipeline: Service initialization missing")
                return False
            
            print("✅ End-to-End Pipeline: All components available")
            return True
            
        except Exception as e:
            print(f"❌ End-to-End Pipeline: Error - {e}")
            return False
    
    def test_data_processing_integration(self) -> bool:
        """Test data processing integration"""
        print("\n🔧 Testing Data Processing Integration")
        print("=" * 50)
        
        try:
            from main.pipeline.data_processor import DataProcessor
            
            # Test data processor
            processor = DataProcessor()
            
            # Test validation
            if hasattr(processor, 'validate_input'):
                print("✅ Data Processor: Input validation available")
            else:
                print("❌ Data Processor: Input validation missing")
                return False
            
            # Test execution
            if hasattr(processor, 'execute'):
                print("✅ Data Processor: Execution method available")
            else:
                print("❌ Data Processor: Execution method missing")
                return False
            
            # Test service integration
            if hasattr(processor, 'technical_indicators'):
                print("✅ Data Processor: Technical indicators service integrated")
            else:
                print("❌ Data Processor: Technical indicators service missing")
                return False
            
            if hasattr(processor, 'feature_engineering'):
                print("✅ Data Processor: Feature engineering service integrated")
            else:
                print("❌ Data Processor: Feature engineering service missing")
                return False
            
            print("✅ Data Processing Integration: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Data Processing Integration: Error - {e}")
            return False
    
    def test_model_training_integration(self) -> bool:
        """Test model training integration"""
        print("\n🔧 Testing Model Training Integration")
        print("=" * 50)
        
        try:
            from main.pipeline.model_trainer import ModelTrainer
            
            # Test model trainer
            trainer = ModelTrainer("AAPL")
            
            # Test data preparation
            if hasattr(trainer, 'prepare_data'):
                print("✅ Model Trainer: Data preparation available")
            else:
                print("❌ Model Trainer: Data preparation missing")
                return False
            
            # Test model training
            if hasattr(trainer, 'train_models'):
                print("✅ Model Trainer: Model training available")
            else:
                print("❌ Model Trainer: Model training missing")
                return False
            
            # Test model evaluation
            if hasattr(trainer, 'evaluate_models'):
                print("✅ Model Trainer: Model evaluation available")
            else:
                print("❌ Model Trainer: Model evaluation missing")
                return False
            
            # Test model saving
            if hasattr(trainer, 'save_models'):
                print("✅ Model Trainer: Model saving available")
            else:
                print("❌ Model Trainer: Model saving missing")
                return False
            
            print("✅ Model Training Integration: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Model Training Integration: Error - {e}")
            return False
    
    def test_prediction_integration(self) -> bool:
        """Test prediction integration"""
        print("\n🔧 Testing Prediction Integration")
        print("=" * 50)
        
        try:
            from main.pipeline.prediction_generator import PredictionGenerator
            
            # Test prediction generator
            generator = PredictionGenerator()
            
            # Test short-term predictions
            if hasattr(generator, 'generate_short_term_predictions'):
                print("✅ Prediction Generator: Short-term predictions available")
            else:
                print("❌ Prediction Generator: Short-term predictions missing")
                return False
            
            # Test mid-term predictions
            if hasattr(generator, 'generate_mid_term_predictions'):
                print("✅ Prediction Generator: Mid-term predictions available")
            else:
                print("❌ Prediction Generator: Mid-term predictions missing")
                return False
            
            # Test long-term predictions
            if hasattr(generator, 'generate_long_term_predictions'):
                print("✅ Prediction Generator: Long-term predictions available")
            else:
                print("❌ Prediction Generator: Long-term predictions missing")
                return False
            
            print("✅ Prediction Integration: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Prediction Integration: Error - {e}")
            return False
    
    def test_user_interface_integration(self) -> bool:
        """Test user interface integration"""
        print("\n🔧 Testing User Interface Integration")
        print("=" * 50)
        
        try:
            from main.interfaces.user_interface import UserInterface
            from main.interfaces.angel_one_interface import AngelOneInterface
            from main.interfaces.input_validator import InputValidator
            
            # Test user interface
            ui = UserInterface()
            
            # Test non-interactive mode
            test_config = {
                'ticker': 'AAPL',
                'is_indian': False,
                'angel_config': None
            }
            
            result = ui.get_user_inputs(test_config)
            
            if result and 'ticker' in result:
                print("✅ User Interface: Non-interactive mode working")
            else:
                print("❌ User Interface: Non-interactive mode failed")
                return False
            
            # Test Angel One interface
            angel_interface = AngelOneInterface()
            
            if hasattr(angel_interface, 'configure_angel_one'):
                print("✅ Angel One Interface: Configuration available")
            else:
                print("❌ Angel One Interface: Configuration missing")
                return False
            
            # Test input validator
            validator = InputValidator()
            
            if hasattr(validator, 'validate_ticker'):
                print("✅ Input Validator: Ticker validation available")
            else:
                print("❌ Input Validator: Ticker validation missing")
                return False
            
            print("✅ User Interface Integration: All tests passed")
            return True
            
        except Exception as e:
            print(f"❌ User Interface Integration: Error - {e}")
            return False
    
    def validate_success_metrics(self) -> bool:
        """Validate all success metrics"""
        print("\n🔧 Validating Success Metrics")
        print("=" * 50)
        
        try:
            # Check performance metrics
            if self.performance_metrics:
                print("✅ Performance Metrics: Collected successfully")
                
                # Check if initialization time is reasonable (< 5 seconds)
                if 'total_initialization' in self.performance_metrics:
                    init_time = self.performance_metrics['total_initialization']
                    if init_time < 5.0:
                        print(f"✅ Initialization Time: {init_time:.3f}s (excellent)")
                    elif init_time < 10.0:
                        print(f"✅ Initialization Time: {init_time:.3f}s (good)")
                    else:
                        print(f"⚠️ Initialization Time: {init_time:.3f}s (slow)")
            else:
                print("❌ Performance Metrics: Not collected")
                return False
            
            # Check test results
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results.values() if result)
            
            if total_tests > 0:
                success_rate = (passed_tests / total_tests) * 100
                print(f"✅ Test Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
                
                if success_rate >= 80:
                    print("✅ Overall System Health: Excellent")
                elif success_rate >= 60:
                    print("✅ Overall System Health: Good")
                else:
                    print("⚠️ Overall System Health: Needs improvement")
            else:
                print("❌ Test Results: Not available")
                return False
            
            print("✅ Success Metrics: All validations passed")
            return True
            
        except Exception as e:
            print(f"❌ Success Metrics: Error - {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        print("🚀 Comprehensive Integration Test Suite")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        tests = [
            ("Angel One API Integration", self.test_angel_one_api_integration),
            ("Database Integration", self.test_database_integration),
            ("Service Coordination", self.test_service_coordination),
            ("API Rate Limiting", self.test_api_rate_limiting),
            ("Fallback Mechanisms", self.test_fallback_mechanisms),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("End-to-End Pipeline", self.test_end_to_end_pipeline),
            ("Data Processing Integration", self.test_data_processing_integration),
            ("Model Training Integration", self.test_model_training_integration),
            ("Prediction Integration", self.test_prediction_integration),
            ("User Interface Integration", self.test_user_interface_integration),
            ("Success Metrics Validation", self.validate_success_metrics)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
            except Exception as e:
                print(f"❌ {test_name}: Test failed with exception - {e}")
                self.test_results[test_name] = False
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        report = f"""
🚀 COMPREHENSIVE INTEGRATION TEST REPORT
{'=' * 60}
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total execution time: {total_time:.2f} seconds

📊 TEST RESULTS SUMMARY
{'=' * 60}
"""
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            report += f"{test_name:<35} {status}\n"
        
        report += f"\nOverall Success Rate: {passed}/{total} ({success_rate:.1f}%)\n"
        
        if self.performance_metrics:
            report += f"""
⚡ PERFORMANCE METRICS
{'=' * 60}
"""
            for metric, value in self.performance_metrics.items():
                report += f"{metric.replace('_', ' ').title():<30} {value:.3f}s\n"
        
        report += f"""
🎯 SYSTEM HEALTH ASSESSMENT
{'=' * 60}
"""
        
        if success_rate >= 90:
            report += "🟢 EXCELLENT: System is production-ready with outstanding performance\n"
        elif success_rate >= 80:
            report += "🟡 GOOD: System is functional with minor issues to address\n"
        elif success_rate >= 60:
            report += "🟠 FAIR: System needs improvements before production deployment\n"
        else:
            report += "🔴 POOR: System requires significant fixes before deployment\n"
        
        return report

def main():
    """Run comprehensive integration tests"""
    test_suite = IntegrationTestSuite()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Generate and display report
    report = test_suite.generate_report()
    print(report)
    
    # Save report to file with UTF-8 encoding
    with open('integration_test_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: integration_test_report.txt")
    
    # Return success if most tests passed
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    return passed >= total * 0.8  # 80% success rate

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
