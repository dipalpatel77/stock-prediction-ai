#!/usr/bin/env python3
"""
Command Line Analysis Runner for AI Stock Predictor
Usage: python run_analysis.py <ticker> [period] [days_ahead] [workers]
"""

import sys
import os
import argparse
sys.path.insert(0, '.')

from main.unified_analysis_pipeline import UnifiedAnalysisPipeline

def main():
    parser = argparse.ArgumentParser(description='AI Stock Predictor Analysis Runner')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL, MSFT, TSLA)')
    parser.add_argument('--period', default='1y', 
                       choices=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                       help='Data period (default: 1y)')
    parser.add_argument('--days-ahead', type=int, default=30,
                       help='Prediction days ahead (default: 30)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker threads (default: 4)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (1mo data, 5 days prediction)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive analysis (2y data, 60 days prediction)')
    
    args = parser.parse_args()
    
    # Override settings based on flags
    if args.quick:
        args.period = '1mo'
        args.days_ahead = 5
        args.workers = 2
    elif args.comprehensive:
        args.period = '2y'
        args.days_ahead = 60
        args.workers = 6
    
    print("🎯 AI Stock Predictor - Command Line Runner")
    print("=" * 60)
    print(f"📊 Ticker: {args.ticker}")
    print(f"📅 Period: {args.period}")
    print(f"⏰ Prediction Days: {args.days_ahead}")
    print(f"🔧 Workers: {args.workers}")
    print("=" * 60)
    
    try:
        # Create pipeline
        pipeline = UnifiedAnalysisPipeline(args.ticker, max_workers=args.workers)
        
        # Run analysis
        result = pipeline.run_unified_analysis(
            period=args.period,
            days_ahead=args.days_ahead,
            use_enhanced=True
        )
        
        if result:
            print(f"\n✅ Analysis completed successfully for {args.ticker}")
            print(f"📁 Check data/ folder for generated files")
        else:
            print(f"\n⚠️ Analysis completed with issues for {args.ticker}")
            
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
