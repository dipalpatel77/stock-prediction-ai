#!/usr/bin/env python3
"""
Custom Analysis Runner for AI Stock Predictor
Run specific analyses with custom parameters
"""

import sys
import os
sys.path.insert(0, '.')

from main.pipeline.core_pipeline import UnifiedAnalysisPipeline

def run_quick_analysis(ticker, period='1mo', days_ahead=5):
    """Run a quick analysis for testing"""
    print(f"🚀 Quick Analysis: {ticker}")
    print("=" * 50)
    
    pipeline = UnifiedAnalysisPipeline(ticker, max_workers=2)
    result = pipeline.run_unified_analysis(
        period=period, 
        days_ahead=days_ahead, 
        use_enhanced=True
    )
    
    if result:
        print(f"✅ Quick analysis completed for {ticker}")
    else:
        print(f"⚠️ Quick analysis completed with issues for {ticker}")
    
    return result

def run_comprehensive_analysis(ticker, period='1y', days_ahead=30):
    """Run a comprehensive analysis"""
    print(f"🎯 Comprehensive Analysis: {ticker}")
    print("=" * 50)
    
    pipeline = UnifiedAnalysisPipeline(ticker, max_workers=4)
    result = pipeline.run_unified_analysis(
        period=period, 
        days_ahead=days_ahead, 
        use_enhanced=True
    )
    
    if result:
        print(f"✅ Comprehensive analysis completed for {ticker}")
    else:
        print(f"⚠️ Comprehensive analysis completed with issues for {ticker}")
    
    return result

def run_batch_analysis(tickers, period='3mo', days_ahead=10):
    """Run analysis for multiple tickers"""
    print(f"📊 Batch Analysis: {', '.join(tickers)}")
    print("=" * 50)
    
    results = {}
    for ticker in tickers:
        print(f"\n🔄 Analyzing {ticker}...")
        try:
            pipeline = UnifiedAnalysisPipeline(ticker, max_workers=2)
            result = pipeline.run_unified_analysis(
                period=period, 
                days_ahead=days_ahead, 
                use_enhanced=True
            )
            results[ticker] = result
            print(f"✅ {ticker}: {'Success' if result else 'Issues'}")
        except Exception as e:
            print(f"❌ {ticker}: Error - {e}")
            results[ticker] = False
    
    return results

if __name__ == "__main__":
    print("🎯 AI Stock Predictor - Custom Analysis Runner")
    print("=" * 60)
    
    # Example 1: Quick analysis
    print("\n📊 Example 1: Quick Analysis")
    run_quick_analysis('AAPL', period='1mo', days_ahead=5)
    
    # Example 2: Comprehensive analysis
    print("\n📊 Example 2: Comprehensive Analysis")
    run_comprehensive_analysis('MSFT', period='1y', days_ahead=30)
    
    # Example 3: Batch analysis
    print("\n📊 Example 3: Batch Analysis")
    popular_stocks = ['GOOGL', 'AMZN', 'NVDA']
    batch_results = run_batch_analysis(popular_stocks, period='3mo', days_ahead=10)
    
    print("\n🎯 Batch Analysis Summary:")
    for ticker, result in batch_results.items():
        status = "✅ Success" if result else "❌ Failed"
        print(f"   {ticker}: {status}")
    
    print("\n✅ Custom analysis runner completed!")
