#!/usr/bin/env python3
"""
Comprehensive Test Runner
Runs all test suites and generates comprehensive reports
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Test modules will be imported dynamically as needed


class ComprehensiveTestRunner:
    """Comprehensive test runner with reporting"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.test_suites = {
            'pipeline_components': 'Pipeline Components',
            'services': 'Services',
            'utils': 'Utilities',
            'integration': 'Integration',
            'performance': 'Performance',
            'validation': 'Validation'
        }
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = time.time()
        
        # Run each test suite
        for suite_name, suite_display in self.test_suites.items():
            print(f"🧪 Running {suite_display} Tests...")
            print("-" * 40)
            
            suite_result = self._run_test_suite(suite_name)
            self.results[suite_name] = suite_result
            
            print(f"✅ {suite_display} Tests: {suite_result['summary']}")
            print()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        return self.results
    
    def _run_test_suite(self, suite_name):
        """Run a specific test suite"""
        try:
            # Import and run test suite
            if suite_name == 'pipeline_components':
                import test_pipeline_components
                test_classes = [
                    test_pipeline_components.TestDataProcessor, 
                    test_pipeline_components.TestModelTrainer, 
                    test_pipeline_components.TestEnhancedModelTrainer,
                    test_pipeline_components.TestStrategyAnalyzer, 
                    test_pipeline_components.TestPredictionGenerator, 
                    test_pipeline_components.TestUnifiedAnalysisPipeline,
                    test_pipeline_components.TestPipelineOrchestrator
                ]
            elif suite_name == 'services':
                import test_services
                test_classes = [
                    test_services.TestDatabaseManager, 
                    test_services.TestAngelOneManager, 
                    test_services.TestAPICoordinator,
                    test_services.TestDataServiceWrapper, 
                    test_services.TestEconomicDataService, 
                    test_services.TestFeatureEngineeringService,
                    test_services.TestTechnicalIndicatorsService, 
                    test_services.TestSmartDataFetcher, 
                    test_services.TestAdvancedCacheManager,
                    test_services.TestMLOptimizer, 
                    test_services.TestAutoScaler, 
                    test_services.TestMonitoringDashboard
                ]
            elif suite_name == 'utils':
                import test_utils
                test_classes = [
                    test_utils.TestDatabasePool, 
                    test_utils.TestErrorHandler, 
                    test_utils.TestPipelineLogger, 
                    test_utils.TestModelCache,
                    test_utils.TestRateLimiter, 
                    test_utils.TestServiceCoordinator, 
                    test_utils.TestServiceManager,
                    test_utils.TestPriceFormatter, 
                    test_utils.TestCurrencyFormatter, 
                    test_utils.TestNumberFormatter,
                    test_utils.TestDataValidator, 
                    test_utils.TestConfigValidator
                ]
            elif suite_name == 'integration':
                import test_integration
                test_classes = [
                    test_integration.TestMainIntegration, 
                    test_integration.TestPipelineIntegration, 
                    test_integration.TestServiceIntegration,
                    test_integration.TestEndToEndWorkflow, 
                    test_integration.TestAsyncIntegration, 
                    test_integration.TestConfigurationIntegration
                ]
            elif suite_name == 'performance':
                import test_performance
                test_classes = [
                    test_performance.TestDataProcessorPerformance, 
                    test_performance.TestModelTrainingPerformance,
                    test_performance.TestDatabasePerformance, 
                    test_performance.TestAPIPerformance, 
                    test_performance.TestPipelinePerformance,
                    test_performance.TestConcurrentPerformance, 
                    test_performance.TestScalabilityPerformance
                ]
            elif suite_name == 'validation':
                import test_validation
                test_classes = [
                    test_validation.TestInputValidation, 
                    test_validation.TestConfigurationValidation, 
                    test_validation.TestErrorHandling,
                    test_validation.TestEdgeCases, 
                    test_validation.TestWarningHandling, 
                    test_validation.TestResourceCleanup
                ]
            else:
                raise ValueError(f"Unknown test suite: {suite_name}")
            
            # Create test suite
            test_suite = unittest.TestSuite()
            for test_class in test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                test_suite.addTests(tests)
            
            # Run tests with proper encoding
            import io
            stream = io.StringIO()
            runner = unittest.TextTestRunner(verbosity=1, stream=stream)
            result = runner.run(test_suite)
            
            # Calculate summary
            total_tests = result.testsRun
            failures = len(result.failures)
            errors = len(result.errors)
            success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
            
            return {
                'total_tests': total_tests,
                'failures': failures,
                'errors': errors,
                'success_rate': success_rate,
                'summary': f"{total_tests} tests, {success_rate:.1f}% success",
                'details': {
                    'failures': result.failures,
                    'errors': result.errors
                }
            }
            
        except Exception as e:
            return {
                'total_tests': 0,
                'failures': 0,
                'errors': 1,
                'success_rate': 0,
                'summary': f"Error running tests: {str(e)}",
                'details': {'errors': [str(e)]}
            }
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        # Calculate overall statistics
        total_tests = sum(result['total_tests'] for result in self.results.values())
        total_failures = sum(result['failures'] for result in self.results.values())
        total_errors = sum(result['errors'] for result in self.results.values())
        overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        execution_time = self.end_time - self.start_time
        
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Successful: {total_tests - total_failures - total_errors}")
        print(f"❌ Failures: {total_failures}")
        print(f"🚨 Errors: {total_errors}")
        print(f"🎯 Overall Success Rate: {overall_success_rate:.1f}%")
        print()
        
        # Detailed results by suite
        print("📋 DETAILED RESULTS BY SUITE:")
        print("-" * 40)
        
        for suite_name, suite_display in self.test_suites.items():
            result = self.results[suite_name]
            status = "✅ PASS" if result['success_rate'] >= 80 else "⚠️ PARTIAL" if result['success_rate'] >= 50 else "❌ FAIL"
            
            print(f"{status} {suite_display}: {result['summary']}")
            
            if result['failures'] > 0 or result['errors'] > 0:
                print(f"   Failures: {result['failures']}, Errors: {result['errors']}")
        
        print()
        
        # Generate JSON report
        self._generate_json_report()
        
        # Generate HTML report
        self._generate_html_report()
        
        print("📄 Reports generated:")
        print("   - docs/COMPREHENSIVE_TEST_REPORT.json")
        print("   - docs/COMPREHENSIVE_TEST_REPORT.html")
        print()
        
        # Final status
        if overall_success_rate >= 90:
            print("🎉 EXCELLENT: Test suite passed with high success rate!")
        elif overall_success_rate >= 80:
            print("✅ GOOD: Test suite passed with acceptable success rate!")
        elif overall_success_rate >= 60:
            print("⚠️ WARNING: Test suite has some issues that need attention!")
        else:
            print("❌ CRITICAL: Test suite has significant issues that need immediate attention!")
    
    def _generate_json_report(self):
        """Generate JSON test report"""
        # Clean results for JSON serialization
        cleaned_results = {}
        for suite_name, result in self.results.items():
            cleaned_results[suite_name] = {
                'total_tests': result['total_tests'],
                'failures': result['failures'],
                'errors': result['errors'],
                'success_rate': result['success_rate'],
                'summary': result['summary']
            }
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': self.end_time - self.start_time,
            'overall_statistics': {
                'total_tests': sum(result['total_tests'] for result in self.results.values()),
                'total_failures': sum(result['failures'] for result in self.results.values()),
                'total_errors': sum(result['errors'] for result in self.results.values()),
                'overall_success_rate': self._calculate_overall_success_rate()
            },
            'suite_results': cleaned_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Ensure docs directory exists
        os.makedirs('docs', exist_ok=True)
        
        # Write JSON report with proper encoding
        with open('docs/COMPREHENSIVE_TEST_REPORT.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    def _generate_html_report(self):
        """Generate HTML test report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .suite {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .pass {{ background-color: #d4edda; }}
        .partial {{ background-color: #fff3cd; }}
        .fail {{ background-color: #f8d7da; }}
        .details {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Comprehensive Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Execution Time: {self.end_time - self.start_time:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>📊 Overall Statistics</h2>
        <p>Total Tests: {sum(result['total_tests'] for result in self.results.values())}</p>
        <p>Success Rate: {self._calculate_overall_success_rate():.1f}%</p>
    </div>
    
    <h2>📋 Test Suite Results</h2>
"""
        
        for suite_name, suite_display in self.test_suites.items():
            result = self.results[suite_name]
            status_class = "pass" if result['success_rate'] >= 80 else "partial" if result['success_rate'] >= 50 else "fail"
            
            html_content += f"""
    <div class="suite {status_class}">
        <h3>{suite_display}</h3>
        <p>Tests: {result['total_tests']}, Success Rate: {result['success_rate']:.1f}%</p>
        <p>Failures: {result['failures']}, Errors: {result['errors']}</p>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Write HTML report with proper encoding
        with open('docs/COMPREHENSIVE_TEST_REPORT.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _calculate_overall_success_rate(self):
        """Calculate overall success rate"""
        total_tests = sum(result['total_tests'] for result in self.results.values())
        total_failures = sum(result['failures'] for result in self.results.values())
        total_errors = sum(result['errors'] for result in self.results.values())
        
        if total_tests == 0:
            return 0
        
        return ((total_tests - total_failures - total_errors) / total_tests * 100)
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        for suite_name, result in self.results.items():
            if result['success_rate'] < 80:
                recommendations.append(f"Improve {suite_name} test suite - current success rate: {result['success_rate']:.1f}%")
            
            if result['errors'] > 0:
                recommendations.append(f"Fix {result['errors']} errors in {suite_name} test suite")
            
            if result['failures'] > 0:
                recommendations.append(f"Address {result['failures']} failures in {suite_name} test suite")
        
        if not recommendations:
            recommendations.append("All test suites are performing well!")
        
        return recommendations


def main():
    """Main test runner function"""
    print("🚀 Starting Comprehensive Test Suite...")
    print()
    
    # Create test runner
    test_runner = ComprehensiveTestRunner()
    
    # Run all tests
    results = test_runner.run_all_tests()
    
    # Exit with appropriate code
    overall_success_rate = test_runner._calculate_overall_success_rate()
    
    if overall_success_rate >= 80:
        print("✅ Test suite completed successfully!")
        sys.exit(0)
    else:
        print("❌ Test suite has issues that need attention!")
        sys.exit(1)


if __name__ == '__main__':
    main()
