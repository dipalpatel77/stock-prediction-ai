#!/usr/bin/env python3
"""
Predict RELIANCE Stock Price
Run the unified analysis pipeline for RELIANCE stock prediction
"""

from unified_analysis_pipeline import UnifiedAnalysisPipeline

def predict_reliance():
    """Predict RELIANCE stock price."""
    print("🎯 RELIANCE Stock Price Prediction")
    print("=" * 50)
    
    # Create unified pipeline for RELIANCE
    ticker = "RELIANCE"
    pipeline = UnifiedAnalysisPipeline(ticker, max_workers=8)
    
    # Run analysis with default parameters
    period = '2y'
    days_ahead = 5
    use_enhanced = True
    
    print(f"📊 Analyzing {ticker} stock...")
    print(f"📅 Period: {period}")
    print(f"⏰ Prediction Days: {days_ahead}")
    print(f"⚡ Enhanced Features: {use_enhanced}")
    print()
    
    # Run the unified analysis
    success = pipeline.run_unified_analysis(period, days_ahead, use_enhanced)
    
    if success:
        print(f"\n✅ RELIANCE analysis completed successfully!")
    else:
        print(f"\n❌ RELIANCE analysis failed!")

if __name__ == "__main__":
    predict_reliance()
