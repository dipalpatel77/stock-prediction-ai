#!/usr/bin/env python3
"""
Enhanced Analysis Runner
Integrates Short-term, Mid-term, and Long-term analysis modules
with improved price forecasting and comprehensive reporting
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
from typing import Dict, List, Tuple, Optional

# Import analysis modules
from analysis_modules import ShortTermAnalyzer, MidTermAnalyzer, LongTermAnalyzer

class EnhancedAnalysisRunner:
    """Enhanced analysis runner with comprehensive timeframe analysis."""
    
    def __init__(self, ticker: str, max_workers: int = 4):
        self.ticker = ticker
        self.max_workers = max_workers
        self.data_dir = "data"
        self.models_dir = "models"
        
        # Initialize analyzers
        self.short_term_analyzer = ShortTermAnalyzer(ticker, max_workers)
        self.mid_term_analyzer = MidTermAnalyzer(ticker, max_workers)
        self.long_term_analyzer = LongTermAnalyzer(ticker, max_workers)
        
        # Analysis results storage
        self.analysis_results = {
            'short_term': {},
            'mid_term': {},
            'long_term': {}
        }
    
    def run_comprehensive_analysis(self, use_enhanced: bool = True) -> Dict:
        """Run comprehensive analysis across all timeframes."""
        print(f"\n{'='*80}")
        print(f"🚀 STARTING COMPREHENSIVE ANALYSIS FOR {self.ticker}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Run all analyses in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_short_term_analysis, use_enhanced): 'short_term',
                executor.submit(self._run_mid_term_analysis, use_enhanced): 'mid_term',
                executor.submit(self._run_long_term_analysis, use_enhanced): 'long_term'
            }
            
            for future in concurrent.futures.as_completed(futures):
                timeframe = futures[future]
                try:
                    result = future.result()
                    self.analysis_results[timeframe] = result
                    print(f"✅ {timeframe.replace('_', ' ').title()} analysis completed")
                except Exception as e:
                    print(f"❌ {timeframe.replace('_', ' ').title()} analysis failed: {e}")
                    self.analysis_results[timeframe] = {'success': False, 'error': str(e)}
        
        execution_time = time.time() - start_time
        
        # Generate comprehensive report
        self._generate_comprehensive_report(execution_time, use_enhanced)
        
        # Display results
        self._display_comprehensive_results(execution_time, use_enhanced)
        
        return self.analysis_results
    
    def _run_short_term_analysis(self, use_enhanced: bool) -> Dict:
        """Run short-term analysis (1-7 days)."""
        print(f"\n📊 Running Short-term Analysis for {self.ticker}...")
        
        start_time = time.time()
        
        # Data processing
        data_success = self.short_term_analyzer.run_short_term_data_processing()
        
        # Model preparation
        model_success = self.short_term_analyzer.prepare_short_term_model()
        
        # Enhanced analysis
        enhanced_success = True
        if use_enhanced:
            enhanced_success = self.short_term_analyzer.run_short_term_enhanced_analysis()
        
        # Strategy analysis (7 days ahead)
        strategy_success = self.short_term_analyzer.run_short_term_strategy_analysis(7)
        
        # Report generation
        report_success = self.short_term_analyzer.run_short_term_report_generation(7, use_enhanced)
        
        execution_time = time.time() - start_time
        
        return {
            'success': all([data_success, model_success, enhanced_success, strategy_success, report_success]),
            'execution_time': execution_time,
            'timeframe': '1-7 days',
            'components': {
                'data_processing': data_success,
                'model_preparation': model_success,
                'enhanced_analysis': enhanced_success,
                'strategy_analysis': strategy_success,
                'report_generation': report_success
            }
        }
    
    def _run_mid_term_analysis(self, use_enhanced: bool) -> Dict:
        """Run mid-term analysis (1-4 weeks)."""
        print(f"\n📈 Running Mid-term Analysis for {self.ticker}...")
        
        start_time = time.time()
        
        # Data processing
        data_success = self.mid_term_analyzer.run_mid_term_data_processing()
        
        # Model preparation
        model_success = self.mid_term_analyzer.prepare_mid_term_model()
        
        # Enhanced analysis
        enhanced_success = True
        if use_enhanced:
            enhanced_success = self.mid_term_analyzer.run_mid_term_enhanced_analysis()
        
        # Strategy analysis (4 weeks ahead)
        strategy_success = self.mid_term_analyzer.run_mid_term_strategy_analysis(4)
        
        # Report generation
        report_success = self.mid_term_analyzer.run_mid_term_report_generation(4, use_enhanced)
        
        execution_time = time.time() - start_time
        
        return {
            'success': all([data_success, model_success, enhanced_success, strategy_success, report_success]),
            'execution_time': execution_time,
            'timeframe': '1-4 weeks',
            'components': {
                'data_processing': data_success,
                'model_preparation': model_success,
                'enhanced_analysis': enhanced_success,
                'strategy_analysis': strategy_success,
                'report_generation': report_success
            }
        }
    
    def _run_long_term_analysis(self, use_enhanced: bool) -> Dict:
        """Run long-term analysis (1-12 months)."""
        print(f"\n📊 Running Long-term Analysis for {self.ticker}...")
        
        start_time = time.time()
        
        # Data processing
        data_success = self.long_term_analyzer.run_long_term_data_processing()
        
        # Model preparation
        model_success = self.long_term_analyzer.prepare_long_term_model()
        
        # Enhanced analysis
        enhanced_success = True
        if use_enhanced:
            enhanced_success = self.long_term_analyzer.run_long_term_enhanced_analysis()
        
        # Strategy analysis (12 months ahead)
        strategy_success = self.long_term_analyzer.run_long_term_strategy_analysis(12)
        
        # Report generation
        report_success = self.long_term_analyzer.run_long_term_report_generation(12, use_enhanced)
        
        execution_time = time.time() - start_time
        
        return {
            'success': all([data_success, model_success, enhanced_success, strategy_success, report_success]),
            'execution_time': execution_time,
            'timeframe': '1-12 months',
            'components': {
                'data_processing': data_success,
                'model_preparation': model_success,
                'enhanced_analysis': enhanced_success,
                'strategy_analysis': strategy_success,
                'report_generation': report_success
            }
        }
    
    def _generate_comprehensive_report(self, execution_time: float, use_enhanced: bool):
        """Generate comprehensive analysis report."""
        print(f"\n📋 Generating comprehensive report...")
        
        # Load current price
        current_price = self._get_current_price()
        
        # Load predictions from all timeframes
        predictions = self._load_all_predictions()
        
        # Generate comprehensive summary
        summary = {
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ticker': self.ticker,
            'Current_Price': current_price,
            'Total_Execution_Time': f"{execution_time:.2f} seconds",
            'Enhanced_Features': use_enhanced,
            'Analysis_Status': 'Completed',
            'Timeframes_Analyzed': list(self.analysis_results.keys()),
            'Success_Rate': self._calculate_success_rate()
        }
        
        # Add timeframe-specific results
        for timeframe, result in self.analysis_results.items():
            if result.get('success'):
                summary[f'{timeframe}_status'] = 'Success'
                summary[f'{timeframe}_execution_time'] = f"{result.get('execution_time', 0):.2f}s"
            else:
                summary[f'{timeframe}_status'] = 'Failed'
                summary[f'{timeframe}_error'] = result.get('error', 'Unknown error')
        
        # Add price predictions
        if predictions:
            summary.update(predictions)
        
        # Save comprehensive report
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"{self.data_dir}/{self.ticker}_comprehensive_analysis.csv", index=False)
        
        # Generate price forecast summary
        self._generate_price_forecast_summary(current_price, predictions)
    
    def _get_current_price(self) -> float:
        """Get current stock price."""
        try:
            import yfinance as yf
            stock = yf.Ticker(self.ticker)
            current_price = stock.info.get('regularMarketPrice', 0)
            return current_price if current_price else 0
        except:
            return 0
    
    def _load_all_predictions(self) -> Dict:
        """Load predictions from all timeframes."""
        predictions = {}
        
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            pred_file = f"{self.data_dir}/{self.ticker}_{timeframe}_predictions.csv"
            if os.path.exists(pred_file):
                try:
                    df = pd.read_csv(pred_file)
                    if not df.empty:
                        # Get the latest prediction
                        latest_pred = df.iloc[-1]
                        predictions[f'{timeframe}_predicted_price'] = latest_pred.get('Predicted_Price', 0)
                        predictions[f'{timeframe}_confidence'] = latest_pred.get('Confidence', 0)
                        predictions[f'{timeframe}_prediction_date'] = latest_pred.get('Date', '')
                except Exception as e:
                    print(f"Warning: Could not load {timeframe} predictions: {e}")
        
        return predictions
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        successful = sum(1 for result in self.analysis_results.values() if result.get('success'))
        total = len(self.analysis_results)
        return (successful / total) * 100 if total > 0 else 0
    
    def _generate_price_forecast_summary(self, current_price: float, predictions: Dict):
        """Generate price forecast summary."""
        if not predictions:
            return
        
        forecast_summary = {
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ticker': self.ticker,
            'Current_Price': current_price,
            'Short_Term_Price': predictions.get('short_term_predicted_price', 0),
            'Short_Term_Change': self._calculate_change(current_price, predictions.get('short_term_predicted_price', 0)),
            'Short_Term_Confidence': predictions.get('short_term_confidence', 0),
            'Mid_Term_Price': predictions.get('mid_term_predicted_price', 0),
            'Mid_Term_Change': self._calculate_change(current_price, predictions.get('mid_term_predicted_price', 0)),
            'Mid_Term_Confidence': predictions.get('mid_term_confidence', 0),
            'Long_Term_Price': predictions.get('long_term_predicted_price', 0),
            'Long_Term_Change': self._calculate_change(current_price, predictions.get('long_term_predicted_price', 0)),
            'Long_Term_Confidence': predictions.get('long_term_confidence', 0),
            'Overall_Sentiment': self._calculate_overall_sentiment(predictions),
            'Risk_Level': self._calculate_risk_level(predictions),
            'Recommended_Action': self._get_recommended_action(predictions)
        }
        
        # Save forecast summary
        forecast_df = pd.DataFrame([forecast_summary])
        forecast_df.to_csv(f"{self.data_dir}/{self.ticker}_price_forecast_summary.csv", index=False)
    
    def _calculate_change(self, current: float, predicted: float) -> float:
        """Calculate percentage change."""
        if current > 0 and predicted > 0:
            return ((predicted - current) / current) * 100
        return 0
    
    def _calculate_overall_sentiment(self, predictions: Dict) -> str:
        """Calculate overall sentiment based on predictions."""
        changes = []
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            change = predictions.get(f'{timeframe}_predicted_price', 0) - predictions.get('current_price', 0)
            if change != 0:
                changes.append(change)
        
        if not changes:
            return 'Neutral'
        
        avg_change = sum(changes) / len(changes)
        if avg_change > 0.05:  # 5% threshold
            return 'Bullish'
        elif avg_change < -0.05:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _calculate_risk_level(self, predictions: Dict) -> str:
        """Calculate risk level based on prediction confidence."""
        confidences = []
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            conf = predictions.get(f'{timeframe}_confidence', 0)
            if conf > 0:
                confidences.append(conf)
        
        if not confidences:
            return 'Unknown'
        
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence >= 0.8:
            return 'Low'
        elif avg_confidence >= 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def _get_recommended_action(self, predictions: Dict) -> str:
        """Get recommended trading action."""
        sentiment = self._calculate_overall_sentiment(predictions)
        risk_level = self._calculate_risk_level(predictions)
        
        if sentiment == 'Bullish' and risk_level in ['Low', 'Medium']:
            return 'BUY'
        elif sentiment == 'Bearish' and risk_level in ['Low', 'Medium']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _display_comprehensive_results(self, execution_time: float, use_enhanced: bool):
        """Display comprehensive analysis results."""
        print(f"\n{'='*80}")
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"⏱️ Total Execution Time: {execution_time:.2f} seconds")
        print(f"📊 Ticker: {self.ticker}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print(f"🔧 Threads Used: {self.max_workers}")
        print()
        
        # Display timeframe results
        for timeframe, result in self.analysis_results.items():
            status = "✅ Success" if result.get('success') else "❌ Failed"
            exec_time = result.get('execution_time', 0)
            print(f"📈 {timeframe.replace('_', ' ').title()}: {status} ({exec_time:.2f}s)")
        
        print(f"\n📁 Generated Files:")
        data_files = [f for f in os.listdir('data') if f.startswith(self.ticker)]
        for file in sorted(data_files):
            print(f"    📄 {file}")
        
        # Display price forecast if available
        forecast_file = f"{self.data_dir}/{self.ticker}_price_forecast_summary.csv"
        if os.path.exists(forecast_file):
            print(f"\n💰 PRICE FORECAST SUMMARY:")
            try:
                forecast_df = pd.read_csv(forecast_file)
                if not forecast_df.empty:
                    forecast = forecast_df.iloc[0]
                    current_price = forecast.get('Current_Price', 0)
                    
                    print(f"   💵 Current Price: ${current_price:.2f}")
                    
                    for timeframe in ['Short_Term', 'Mid_Term', 'Long_Term']:
                        pred_price = forecast.get(f'{timeframe}_Price', 0)
                        change = forecast.get(f'{timeframe}_Change', 0)
                        confidence = forecast.get(f'{timeframe}_Confidence', 0)
                        
                        if pred_price > 0:
                            change_symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                            print(f"   {change_symbol} {timeframe}: ${pred_price:.2f} ({change:+.2f}%) [Confidence: {confidence:.1%}]")
                    
                    sentiment = forecast.get('Overall_Sentiment', 'Unknown')
                    risk_level = forecast.get('Risk_Level', 'Unknown')
                    action = forecast.get('Recommended_Action', 'Unknown')
                    
                    print(f"   🎯 Overall Sentiment: {sentiment}")
                    print(f"   ⚠️ Risk Level: {risk_level}")
                    print(f"   💡 Recommended Action: {action}")
            except Exception as e:
                print(f"   ⚠️ Could not load forecast: {e}")
        
        print(f"\n💡 Trading Recommendations:")
        print(f"   • Short-term: Use for day trading and swing trading")
        print(f"   • Mid-term: Use for position trading and trend following")
        print(f"   • Long-term: Use for investment decisions and portfolio allocation")
        print(f"   • Always consider risk management and diversification")
        print(f"   • Past performance doesn't guarantee future results")

def main():
    """Main function for running enhanced analysis."""
    if len(sys.argv) < 2:
        print("Usage: python enhanced_analysis_runner.py <TICKER> [enhanced]")
        print("Example: python enhanced_analysis_runner.py AAPL enhanced")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    use_enhanced = len(sys.argv) > 2 and sys.argv[2].lower() == 'enhanced'
    
    # Create and run analysis
    runner = EnhancedAnalysisRunner(ticker)
    results = runner.run_comprehensive_analysis(use_enhanced)
    
    # Return success status
    success_count = sum(1 for result in results.values() if result.get('success'))
    if success_count >= 2:  # At least 2 out of 3 timeframes successful
        print(f"\n✅ Analysis completed successfully ({success_count}/3 timeframes)")
        sys.exit(0)
    else:
        print(f"\n❌ Analysis partially failed ({success_count}/3 timeframes)")
        sys.exit(1)

if __name__ == "__main__":
    main()
