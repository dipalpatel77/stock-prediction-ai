#!/usr/bin/env python3
"""
Quick Analysis Runner
Simple script to run stock analysis
"""

import sys
import os

# Add main directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main'))

def main():
    """Main function to run analysis"""
    print("🚀 AI Stock Predictor - Quick Runner")
    print("=" * 50)
    
    # Check if ticker is provided
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        period = sys.argv[2] if len(sys.argv) > 2 else '1y'
        
        print(f"📊 Analyzing {ticker} for {period}")
        
        try:
            from main.main import run_quick_analysis
            result = run_quick_analysis(ticker, period)
            
            if result['success']:
                print("✅ Analysis completed successfully!")
                return True
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    else:
        print("Usage: python run_analysis.py <TICKER> [PERIOD]")
        print("Examples:")
        print("  python run_analysis.py AAPL")
        print("  python run_analysis.py AAPL 1y")
        print("  python run_analysis.py RELIANCE")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)