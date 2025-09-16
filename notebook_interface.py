#!/usr/bin/env python3
"""
Jupyter Notebook Interface for AI Stock Predictor
Interactive analysis with visualization capabilities
"""

import sys
import os
sys.path.insert(0, '.')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main.unified_analysis_pipeline import UnifiedAnalysisPipeline

class StockAnalysisNotebook:
    """Interactive notebook interface for stock analysis"""
    
    def __init__(self, ticker, max_workers=4):
        self.ticker = ticker
        self.pipeline = UnifiedAnalysisPipeline(ticker, max_workers=max_workers)
        self.results = None
        
    def run_analysis(self, period='1y', days_ahead=30, use_enhanced=True):
        """Run the complete analysis"""
        print(f"🚀 Running analysis for {self.ticker}")
        print("=" * 50)
        
        self.results = self.pipeline.run_unified_analysis(
            period=period,
            days_ahead=days_ahead,
            use_enhanced=use_enhanced
        )
        
        return self.results
    
    def get_predictions(self):
        """Get prediction data"""
        if not self.results:
            print("❌ No analysis results available. Run analysis first.")
            return None
            
        try:
            # Load prediction files
            predictions_file = f"data/{self.ticker}_advanced_predictions.csv"
            if os.path.exists(predictions_file):
                return pd.read_csv(predictions_file)
            else:
                print(f"❌ Prediction file not found: {predictions_file}")
                return None
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return None
    
    def get_confidence_analysis(self):
        """Get confidence analysis data"""
        if not self.results:
            print("❌ No analysis results available. Run analysis first.")
            return None
            
        try:
            confidence_file = f"data/{self.ticker}_confidence_analysis.csv"
            if os.path.exists(confidence_file):
                return pd.read_csv(confidence_file)
            else:
                print(f"❌ Confidence file not found: {confidence_file}")
                return None
        except Exception as e:
            print(f"❌ Error loading confidence analysis: {e}")
            return None
    
    def plot_predictions(self, days=30):
        """Plot prediction results"""
        predictions = self.get_predictions()
        if predictions is None:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot individual model predictions
        for column in predictions.columns:
            if column != 'Date' and 'prediction' in column.lower():
                plt.plot(predictions.index[:days], predictions[column][:days], 
                        label=column, alpha=0.7)
        
        plt.title(f'{self.ticker} - Model Predictions (Next {days} Days)')
        plt.xlabel('Days Ahead')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_confidence_intervals(self):
        """Plot confidence intervals"""
        confidence = self.get_confidence_analysis()
        if confidence is None:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Plot confidence intervals
        plt.fill_between(confidence.index, 
                        confidence['lower_68'], 
                        confidence['upper_68'], 
                        alpha=0.3, label='68% Confidence')
        plt.fill_between(confidence.index, 
                        confidence['lower_95'], 
                        confidence['upper_95'], 
                        alpha=0.2, label='95% Confidence')
        
        plt.plot(confidence.index, confidence['mean_prediction'], 
                'r-', label='Mean Prediction', linewidth=2)
        
        plt.title(f'{self.ticker} - Prediction Confidence Intervals')
        plt.xlabel('Days Ahead')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_summary(self):
        """Get analysis summary"""
        if not self.results:
            print("❌ No analysis results available. Run analysis first.")
            return None
            
        try:
            summary_file = f"data/{self.ticker}_unified_summary.csv"
            if os.path.exists(summary_file):
                return pd.read_csv(summary_file)
            else:
                print(f"❌ Summary file not found: {summary_file}")
                return None
        except Exception as e:
            print(f"❌ Error loading summary: {e}")
            return None
    
    def display_results(self):
        """Display comprehensive results"""
        print(f"\n🎯 Analysis Results for {self.ticker}")
        print("=" * 60)
        
        # Display predictions
        predictions = self.get_predictions()
        if predictions is not None:
            print("\n📊 Predictions:")
            print(predictions.head(10))
        
        # Display confidence analysis
        confidence = self.get_confidence_analysis()
        if confidence is not None:
            print("\n📈 Confidence Analysis:")
            print(confidence.head(10))
        
        # Display summary
        summary = self.get_summary()
        if summary is not None:
            print("\n📋 Summary:")
            print(summary)

# Example usage functions
def analyze_stock(ticker, period='1y', days_ahead=30):
    """Quick analysis function"""
    analyzer = StockAnalysisNotebook(ticker)
    result = analyzer.run_analysis(period=period, days_ahead=days_ahead)
    
    if result:
        analyzer.display_results()
        return analyzer
    else:
        print(f"❌ Analysis failed for {ticker}")
        return None

def compare_stocks(tickers, period='1y', days_ahead=30):
    """Compare multiple stocks"""
    results = {}
    
    for ticker in tickers:
        print(f"\n🔄 Analyzing {ticker}...")
        analyzer = StockAnalysisNotebook(ticker)
        result = analyzer.run_analysis(period=period, days_ahead=days_ahead)
        results[ticker] = analyzer if result else None
    
    return results

# Example usage
if __name__ == "__main__":
    print("🎯 AI Stock Predictor - Notebook Interface")
    print("=" * 50)
    
    # Example 1: Single stock analysis
    print("\n📊 Example 1: Single Stock Analysis")
    analyzer = analyze_stock('AAPL', period='1y', days_ahead=30)
    
    if analyzer:
        print("\n📈 Plotting predictions...")
        analyzer.plot_predictions(days=30)
        
        print("\n📊 Plotting confidence intervals...")
        analyzer.plot_confidence_intervals()
    
    # Example 2: Compare multiple stocks
    print("\n📊 Example 2: Compare Multiple Stocks")
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    comparison_results = compare_stocks(stocks, period='6mo', days_ahead=15)
    
    print("\n🎯 Comparison Summary:")
    for ticker, analyzer in comparison_results.items():
        if analyzer:
            predictions = analyzer.get_predictions()
            if predictions is not None:
                avg_pred = predictions.iloc[0].mean()
                print(f"   {ticker}: Average Prediction = ${avg_pred:.2f}")
    
    print("\n✅ Notebook interface demo completed!")
