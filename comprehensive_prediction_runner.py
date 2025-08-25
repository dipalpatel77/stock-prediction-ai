#!/usr/bin/env python3
"""
Comprehensive Prediction Runner
Combines improved prediction engine with enhanced strategy analysis
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our improved modules
from improved_prediction_engine import ImprovedPredictionEngine
from enhanced_strategy_analyzer import run_enhanced_strategy_analysis

class ComprehensivePredictionRunner:
    """
    Comprehensive prediction runner with advanced algorithms and strategy analysis
    """
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.prediction_engine = ImprovedPredictionEngine(ticker)
        self.results = {}
        
    def run_comprehensive_analysis(self, days_ahead=5):
        """Run comprehensive prediction analysis."""
        try:
            print(f"🚀 Comprehensive Prediction Analysis for {self.ticker}")
            print("=" * 70)
            
            # Step 1: Load and prepare data
            print("📊 Step 1: Loading and preparing data...")
            df = self.prediction_engine.load_and_prepare_data()
            if df is None:
                return False
            
            # Step 2: Prepare features
            print("🔧 Step 2: Preparing advanced features...")
            X, y = self.prediction_engine.prepare_features(df)
            if X is None or y is None:
                return False
            
            # Step 3: Train or load models
            print("🤖 Step 3: Training/loading advanced models...")
            models_exist = self.prediction_engine.load_models()
            
            if not models_exist:
                print("🔄 Training new advanced models...")
                if not self.prediction_engine.train_advanced_models(X, y):
                    return False
                self.prediction_engine.save_models()
            else:
                print("✅ Using pre-trained advanced models")
            
            # Step 4: Generate predictions
            print("🔮 Step 4: Generating advanced predictions...")
            predictions, multi_day_predictions = self.prediction_engine.generate_predictions(X, days_ahead)
            
            if predictions is None:
                return False
            
            # Step 5: Calculate confidence and signals
            print("📈 Step 5: Calculating confidence and signals...")
            current_price = float(df['Close'].iloc[-1])
            confidence = self.prediction_engine.calculate_prediction_confidence(predictions)
            trading_signals = self.prediction_engine.generate_trading_signals(predictions, current_price, confidence)
            
            # Step 6: Enhanced strategy analysis
            print("🔍 Step 6: Running enhanced strategy analysis...")
            ensemble_prediction = predictions.get('Ensemble', current_price)
            strategy_success = run_enhanced_strategy_analysis(self.ticker, df, ensemble_prediction)
            
            # Step 7: Display comprehensive results
            print("📋 Step 7: Displaying comprehensive results...")
            self.display_comprehensive_results(
                df, predictions, multi_day_predictions, 
                confidence, trading_signals, current_price
            )
            
            # Step 8: Save comprehensive report
            print("💾 Step 8: Saving comprehensive report...")
            self.save_comprehensive_report(
                predictions, multi_day_predictions, confidence, 
                trading_signals, current_price
            )
            
            print("✅ Comprehensive analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            return False
    
    def display_comprehensive_results(self, df, predictions, multi_day_predictions, confidence, trading_signals, current_price):
        """Display comprehensive prediction results."""
        print("\n🎯 COMPREHENSIVE PREDICTION RESULTS")
        print("=" * 80)
        print(f"📊 Stock: {self.ticker}")
        print(f"💰 CURRENT PRICE: ₹{current_price:.2f}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        print()
        
        # Display individual model predictions
        print("🤖 ADVANCED MODEL PREDICTIONS:")
        for model_name, pred in predictions.items():
            change = pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   • {model_name:15}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        print()
        
        # Display ensemble prediction with confidence
        if 'Ensemble' in predictions:
            ensemble_pred = predictions['Ensemble']
            change = ensemble_pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            
            print("🎯 ENSEMBLE PREDICTION (PRIMARY):")
            print(f"   Predicted Price: ₹{ensemble_pred:.2f} ({direction} {change_pct:+.2f}%)")
            print(f"   Confidence 68%: ₹{confidence['confidence_68'][0]:.2f} - ₹{confidence['confidence_68'][1]:.2f}")
            print(f"   Confidence 95%: ₹{confidence['confidence_95'][0]:.2f} - ₹{confidence['confidence_95'][1]:.2f}")
            print(f"   Model Agreement: {confidence['agreement_score']:.2%}")
        
        print()
        
        # Display multi-day forecast
        if multi_day_predictions:
            print("📅 MULTI-DAY FORECAST:")
            for i, pred in enumerate(multi_day_predictions, 1):
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   • Day {i}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        print()
        
        # Display trading signals
        if trading_signals:
            print("💡 TRADING SIGNALS:")
            print(f"   Signal: {trading_signals['signal']}")
            print(f"   Confidence: {trading_signals['confidence_level']}")
            print(f"   Expected Return: {trading_signals['expected_return']:+.2f}%")
            print(f"   Signal Strength: {trading_signals['signal_strength']:.2f}")
            print(f"   Model Agreement: {trading_signals['agreement_score']:.2%}")
        
        print()
        
        # Display feature importance (top 10)
        if hasattr(self.prediction_engine, 'feature_importance') and self.prediction_engine.feature_importance:
            print("🔍 TOP FEATURE IMPORTANCE:")
            # Get feature importance from ensemble model
            if 'Ensemble' in self.prediction_engine.feature_importance:
                importance = self.prediction_engine.feature_importance['Ensemble']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                for feature, importance_score in sorted_features:
                    print(f"   • {feature:25}: {importance_score:.4f}")
        
        print()
        
        # Display risk metrics
        print("⚠️ RISK METRICS:")
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * 100
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            max_drawdown = (df['Close'] / df['Close'].expanding().max() - 1).min() * 100
            
            print(f"   Volatility (20d): {volatility:.2f}%")
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {max_drawdown:.2f}%")
        
        print()
        
        # Display comprehensive recommendation
        print("📋 COMPREHENSIVE RECOMMENDATION:")
        if trading_signals:
            signal = trading_signals['signal']
            if signal == "STRONG_BUY":
                print("   🟢 STRONG BUY - High confidence bullish signal")
                print("   📈 Expected significant upward movement")
                print("   💰 Consider aggressive position sizing")
            elif signal == "BUY":
                print("   🟡 BUY - Moderate confidence bullish signal")
                print("   📈 Expected moderate upward movement")
                print("   💰 Consider moderate position sizing")
            elif signal == "STRONG_SELL":
                print("   🔴 STRONG SELL - High confidence bearish signal")
                print("   📉 Expected significant downward movement")
                print("   💰 Consider short position or exit long positions")
            elif signal == "SELL":
                print("   🟠 SELL - Moderate confidence bearish signal")
                print("   📉 Expected moderate downward movement")
                print("   💰 Consider reducing position size")
            else:
                print("   ⚪ HOLD - Neutral signal, wait for better opportunities")
                print("   ➡️ No clear directional bias")
                print("   💰 Maintain current position or wait")
        
        print()
        print("=" * 80)
    
    def save_comprehensive_report(self, predictions, multi_day_predictions, confidence, trading_signals, current_price):
        """Save comprehensive prediction report."""
        try:
            # Prepare report data
            report_data = {
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Ticker': [self.ticker],
                'Current_Price': [current_price],
                'Ensemble_Prediction': [predictions.get('Ensemble', current_price)],
                'Confidence_68_Lower': [confidence['confidence_68'][0]],
                'Confidence_68_Upper': [confidence['confidence_68'][1]],
                'Confidence_95_Lower': [confidence['confidence_95'][0]],
                'Confidence_95_Upper': [confidence['confidence_95'][1]],
                'Model_Agreement': [confidence['agreement_score']],
                'Trading_Signal': [trading_signals['signal'] if trading_signals else 'HOLD'],
                'Signal_Confidence': [trading_signals['confidence_level'] if trading_signals else 'LOW'],
                'Expected_Return_Pct': [trading_signals['expected_return'] if trading_signals else 0.0],
                'Signal_Strength': [trading_signals['signal_strength'] if trading_signals else 0.0]
            }
            
            # Add individual model predictions
            for model_name, pred in predictions.items():
                report_data[f'{model_name}_Prediction'] = [pred]
            
            # Add multi-day predictions
            if multi_day_predictions:
                for i, pred in enumerate(multi_day_predictions, 1):
                    report_data[f'Day_{i}_Prediction'] = [pred]
            
            # Create DataFrame and save
            report_df = pd.DataFrame(report_data)
            report_path = f"data/{self.ticker}_comprehensive_prediction_report.csv"
            report_df.to_csv(report_path, index=False)
            
            print(f"💾 Comprehensive report saved: {report_path}")
            
            # Also save detailed results
            detailed_results = {
                'ticker': self.ticker,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predictions': predictions,
                'multi_day_predictions': multi_day_predictions,
                'confidence': confidence,
                'trading_signals': trading_signals,
                'current_price': current_price
            }
            
            detailed_path = f"data/{self.ticker}_detailed_results.pkl"
            joblib.dump(detailed_results, detailed_path)
            print(f"💾 Detailed results saved: {detailed_path}")
            
            return True
            
        except Exception as e:
            print(f"Warning: Error saving comprehensive report: {e}")
            return False
    
    def get_prediction_summary(self):
        """Get a summary of the latest prediction."""
        try:
            report_path = f"data/{self.ticker}_comprehensive_prediction_report.csv"
            if os.path.exists(report_path):
                df = pd.read_csv(report_path)
                if not df.empty:
                    latest = df.iloc[-1]
                    return {
                        'ticker': latest['Ticker'],
                        'timestamp': latest['Timestamp'],
                        'current_price': latest['Current_Price'],
                        'predicted_price': latest['Ensemble_Prediction'],
                        'signal': latest['Trading_Signal'],
                        'confidence': latest['Signal_Confidence'],
                        'expected_return': latest['Expected_Return_Pct']
                    }
            return None
            
        except Exception as e:
            print(f"Warning: Error getting prediction summary: {e}")
            return None

def run_comprehensive_prediction(ticker="AAPL", days_ahead=5):
    """Run comprehensive prediction for a given ticker."""
    try:
        # Initialize runner
        runner = ComprehensivePredictionRunner(ticker)
        
        # Run comprehensive analysis
        success = runner.run_comprehensive_analysis(days_ahead)
        
        if success:
            # Get summary
            summary = runner.get_prediction_summary()
            if summary:
                print(f"\n📊 PREDICTION SUMMARY:")
                print(f"   Ticker: {summary['ticker']}")
                print(f"   Current Price: ₹{summary['current_price']:.2f}")
                print(f"   Predicted Price: ₹{summary['predicted_price']:.2f}")
                print(f"   Signal: {summary['signal']}")
                print(f"   Confidence: {summary['confidence']}")
                print(f"   Expected Return: {summary['expected_return']:+.2f}%")
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error in comprehensive prediction: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Comprehensive Stock Prediction System")
    print("=" * 60)
    print("Advanced prediction with enhanced strategy analysis")
    print("=" * 60)
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
    
    days_ahead = input("Enter prediction days (default: 5): ").strip()
    if not days_ahead:
        days_ahead = 5
    else:
        try:
            days_ahead = int(days_ahead)
        except ValueError:
            days_ahead = 5
    
    print(f"\n🎯 Running comprehensive prediction for {ticker}...")
    print(f"📅 Prediction horizon: {days_ahead} days")
    print()
    
    # Run comprehensive prediction
    success = run_comprehensive_prediction(ticker, days_ahead)
    
    if success:
        print(f"\n✅ {ticker} comprehensive prediction completed successfully!")
        print(f"📁 Check the 'data' folder for detailed reports")
    else:
        print(f"\n❌ {ticker} comprehensive prediction failed!")

if __name__ == "__main__":
    main()
