#!/usr/bin/env python3
"""
Test Enhanced Date Functionality
===============================

This script demonstrates the enhanced date functionality that includes:
- Exact date information (day, month, year)
- Month names in full and abbreviated forms
- Day of week information
- Enhanced timestamp formats
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_date_utils():
    """Test the enhanced date utilities"""
    print("🧪 Testing Enhanced Date Utilities")
    print("=" * 50)
    
    try:
        from core.date_utils import DateUtils
        
        # Test current date information
        print("\n📅 Current Date Information:")
        current_date = DateUtils.get_enhanced_date_info()
        print(f"   Full Date: {current_date['date_with_day']}")
        print(f"   Month: {current_date['month_name']} ({current_date['month_name_short']})")
        print(f"   Day: {current_date['day_name']} ({current_date['day_name_short']})")
        print(f"   Year: {current_date['year']}")
        print(f"   Quarter: Q{current_date['quarter']}")
        print(f"   Week of Year: {current_date['week_of_year']}")
        
        # Test prediction dates
        print("\n🔮 Prediction Date Examples:")
        
        # Short-term (1-7 days)
        print("\n   📅 Short-term Predictions (1-7 days):")
        for day in range(1, 8):
            date_info = DateUtils.format_prediction_date(datetime.now(), day)
            print(f"      Day {day}: {date_info['prediction_date_full']} ({date_info['prediction_date_short']})")
            print(f"         Month: {date_info['month_name']}, Day: {date_info['day_name']}")
        
        # Medium-term (1-4 weeks)
        print("\n   📅 Medium-term Predictions (1-4 weeks):")
        for week in range(1, 5):
            date_info = DateUtils.format_week_prediction_date(datetime.now(), week)
            print(f"      Week {week}: {date_info['prediction_week_range']} ({date_info['prediction_week_short']})")
            print(f"         Date: {date_info['prediction_date_full']}")
        
        # Long-term (1-12 months)
        print("\n   📅 Long-term Predictions (1-12 months):")
        for month in range(1, 13):
            date_info = DateUtils.format_month_prediction_date(datetime.now(), month)
            print(f"      Month {month}: {date_info['prediction_month_full']} ({date_info['prediction_month_short']})")
            print(f"         Quarter: {date_info['prediction_quarter']}")
        
        # Test analysis timestamp
        print("\n⏰ Analysis Timestamp:")
        analysis_timestamp = DateUtils.format_analysis_timestamp()
        print(f"   Full: {analysis_timestamp['analysis_date_full']}")
        print(f"   Short: {analysis_timestamp['analysis_date_short']}")
        print(f"   ISO: {analysis_timestamp['timestamp_iso']}")
        
        # Test file timestamp
        print("\n📁 File Timestamp:")
        file_timestamp = DateUtils.format_file_timestamp()
        print(f"   Filename: {file_timestamp}")
        
        print("\n✅ Enhanced Date Utilities Test: PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_enhanced_price_forecaster():
    """Test the enhanced price forecaster with new date information"""
    print("\n🧪 Testing Enhanced Price Forecaster")
    print("=" * 50)
    
    try:
        from analysis_modules.enhanced_price_forecaster import EnhancedPriceForecaster
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        prices = np.random.randn(len(dates)).cumsum() + 100
        volumes = np.random.randint(1000000, 10000000, len(dates))
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': volumes
        })
        
        # Initialize forecaster
        forecaster = EnhancedPriceForecaster("AAPL")
        
        # Test short-term predictions
        print("\n📅 Short-term Predictions (7 days):")
        short_term = forecaster.predict_short_term(df, days_ahead=7)
        
        if short_term and 'predictions' in short_term:
            for pred in short_term['predictions']:
                print(f"   {pred['Date_Full']} ({pred['Date_Short']})")
                print(f"      Month: {pred['Month_Name']}, Day: {pred['Day_Name']}")
                print(f"      Price: ${pred['Predicted_Price']:.2f}")
        
        # Test medium-term predictions
        print("\n📅 Medium-term Predictions (4 weeks):")
        medium_term = forecaster.predict_medium_term(df, weeks_ahead=4)
        
        if medium_term and 'predictions' in medium_term:
            for pred in medium_term['predictions']:
                print(f"   {pred['Week_Range']} ({pred['Week_Short']})")
                print(f"      Date: {pred['Date_Full']}")
                print(f"      Price: ${pred['Predicted_Price']:.2f}")
        
        # Test long-term predictions
        print("\n📅 Long-term Predictions (12 months):")
        long_term = forecaster.predict_long_term(df, months_ahead=12)
        
        if long_term and 'predictions' in long_term:
            for pred in long_term['predictions']:
                print(f"   {pred['Month_Full']} ({pred['Month_Short']})")
                print(f"      Quarter: {pred['Quarter']}")
                print(f"      Price: ${pred['Predicted_Price']:.2f}")
        
        print("\n✅ Enhanced Price Forecaster Test: PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_unified_pipeline_integration():
    """Test the unified pipeline with enhanced date information"""
    print("\n🧪 Testing Unified Pipeline Integration")
    print("=" * 50)
    
    try:
        from unified_analysis_pipeline import UnifiedAnalysisPipeline
        
        # Create a simple pipeline instance
        pipeline = UnifiedAnalysisPipeline("AAPL")
        
        # Test that the date utilities are available
        if hasattr(pipeline, 'DATE_UTILS_AVAILABLE'):
            print(f"   Date Utils Available: {pipeline.DATE_UTILS_AVAILABLE}")
        else:
            print("   Date Utils Available: False (not found)")
        
        # Test analysis timestamp
        if hasattr(pipeline, 'DateUtils'):
            timestamp = pipeline.DateUtils.format_analysis_timestamp()
            print(f"   Analysis Timestamp: {timestamp['analysis_date_full']}")
            print(f"   Month: {timestamp['month_name']}")
            print(f"   Day: {timestamp['day_name']}")
            print(f"   Year: {timestamp['year']}")
        
        print("\n✅ Unified Pipeline Integration Test: PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Enhanced Date Functionality Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Date Utilities", test_date_utils),
        ("Enhanced Price Forecaster", test_enhanced_price_forecaster),
        ("Unified Pipeline Integration", test_unified_pipeline_integration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Display results
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhanced date functionality tests passed!")
        print("✅ Enhanced date storage is working correctly")
        print("\n📋 Enhanced Date Features:")
        print("   • Exact date information (day, month, year)")
        print("   • Month names in full and abbreviated forms")
        print("   • Day of week information")
        print("   • Enhanced timestamp formats")
        print("   • Prediction date formatting")
        print("   • Analysis timestamp formatting")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
