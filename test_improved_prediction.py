#!/usr/bin/env python3
"""
Test Improved Prediction System
Run the improved prediction system with a specific ticker
"""

from comprehensive_prediction_runner import run_comprehensive_prediction

def test_improved_prediction():
    """Test the improved prediction system."""
    print("🧪 Testing Improved Prediction System")
    print("=" * 50)
    
    # Test with RELIANCE
    ticker = "RELIANCE"
    days_ahead = 5
    
    print(f"🎯 Testing with ticker: {ticker}")
    print(f"📅 Prediction horizon: {days_ahead} days")
    print()
    
    # Run comprehensive prediction
    success = run_comprehensive_prediction(ticker, days_ahead)
    
    if success:
        print(f"\n✅ {ticker} improved prediction test completed successfully!")
    else:
        print(f"\n❌ {ticker} improved prediction test failed!")

if __name__ == "__main__":
    test_improved_prediction()
