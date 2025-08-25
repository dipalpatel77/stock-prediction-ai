#!/usr/bin/env python3
"""
Main test runner for the AI Stock Predictor test suite
"""

import os
import sys
import unittest
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.test_helpers import setup_test_logging, create_test_config


def discover_and_run_tests(test_type=None, pattern="test_*.py", verbose=False):
    """Discover and run tests based on type"""
    
    # Set up test logging
    logger = setup_test_logging()
    
    # Create test results directory
    results_dir = Path("tests/results")
    results_dir.mkdir(exist_ok=True)
    
    # Determine test directory based on type
    if test_type == "unit":
        test_dir = "tests/unit"
        test_suite_name = "Unit Tests"
    elif test_type == "integration":
        test_dir = "tests/integration"
        test_suite_name = "Integration Tests"
    elif test_type == "performance":
        test_dir = "tests/performance"
        test_suite_name = "Performance Tests"
    else:
        test_dir = "tests"
        test_suite_name = "All Tests"
    
    print(f"\n🧪 Running {test_suite_name}")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Test Directory: {test_dir}")
    print(f"🔍 Test Pattern: {pattern}")
    print()
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    # Run tests
    start_time = time.time()
    
    if verbose:
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    else:
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    
    result = runner.run(suite)
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    # Print summary
    print("\n📋 TEST SUMMARY")
    print("=" * 30)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Duration: {test_duration:.2f}s")
    
    # Save detailed results
    save_test_results(result, test_suite_name, test_duration, results_dir)
    
    return result


def save_test_results(result, test_suite_name, duration, results_dir):
    """Save detailed test results to file"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"test_results_{test_suite_name.lower().replace(' ', '_')}_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"Test Suite: {test_suite_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Total Tests: {result.testsRun}\n")
        f.write(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Failed: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n\n")
        
        if result.failures:
            f.write("FAILURES:\n")
            f.write("-" * 20 + "\n")
            for test, traceback in result.failures:
                f.write(f"Test: {test}\n")
                f.write(f"Traceback:\n{traceback}\n\n")
        
        if result.errors:
            f.write("ERRORS:\n")
            f.write("-" * 20 + "\n")
            for test, traceback in result.errors:
                f.write(f"Test: {test}\n")
                f.write(f"Traceback:\n{traceback}\n\n")
    
    print(f"📄 Detailed results saved to: {results_file}")


def run_specific_test(test_file, verbose=False):
    """Run a specific test file"""
    
    print(f"\n🧪 Running specific test: {test_file}")
    print("=" * 60)
    
    # Load and run the specific test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_file)
    
    start_time = time.time()
    
    if verbose:
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    else:
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    
    result = runner.run(suite)
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    print(f"\n⏱️ Test duration: {test_duration:.2f}s")
    
    return result


def main():
    """Main function to run tests"""
    
    parser = argparse.ArgumentParser(description="Run AI Stock Predictor tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Test file pattern to match"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--file",
        help="Run a specific test file"
    )
    parser.add_argument(
        "--config",
        help="Test configuration file"
    )
    
    args = parser.parse_args()
    
    # Set up test configuration
    config = create_test_config()
    if args.config and os.path.exists(args.config):
        # Load custom config if provided
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    print("🚀 AI STOCK PREDICTOR TEST SUITE")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ Configuration: {config}")
    print()
    
    try:
        if args.file:
            # Run specific test file
            result = run_specific_test(args.file, args.verbose)
        else:
            # Run tests by type
            if args.type == "all":
                # Run all test types
                results = {}
                for test_type in ["unit", "integration", "performance"]:
                    print(f"\n{'='*20} {test_type.upper()} TESTS {'='*20}")
                    results[test_type] = discover_and_run_tests(test_type, args.pattern, args.verbose)
                
                # Overall summary
                print(f"\n{'='*20} OVERALL SUMMARY {'='*20}")
                total_tests = sum(r.testsRun for r in results.values())
                total_failures = sum(len(r.failures) for r in results.values())
                total_errors = sum(len(r.errors) for r in results.values())
                total_passed = total_tests - total_failures - total_errors
                
                print(f"Total Tests: {total_tests}")
                print(f"Passed: {total_passed}")
                print(f"Failed: {total_failures}")
                print(f"Errors: {total_errors}")
                print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
                
                # Return overall result
                result = unittest.TestResult()
                result.testsRun = total_tests
                result.failures = [(f"Overall", f"Failures: {total_failures}, Errors: {total_errors}")]
                result.errors = []
                
            else:
                # Run specific test type
                result = discover_and_run_tests(args.type, args.pattern, args.verbose)
        
        # Final status
        if result.wasSuccessful():
            print("\n🎉 All tests passed successfully!")
            return 0
        else:
            print(f"\n⚠️ {len(result.failures) + len(result.errors)} test(s) failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
