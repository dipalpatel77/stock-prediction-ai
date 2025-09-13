#!/usr/bin/env python3
"""
Prediction Validation Dashboard
Comprehensive dashboard for validating prediction accuracy
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ValidationDashboard:
    """Comprehensive validation dashboard for stock predictions."""
    
    def __init__(self, ticker: str, data_dir: str = "data"):
        self.ticker = ticker
        self.data_dir = data_dir
        self.results = {}
        
    def run_full_validation(self):
        """Run comprehensive validation analysis."""
        print(f"🎯 COMPREHENSIVE VALIDATION DASHBOARD FOR {self.ticker}")
        print("=" * 70)
        
        # 1. Current Prediction Accuracy
        self._validate_current_predictions()
        
        # 2. Historical Performance
        self._analyze_historical_performance()
        
        # 3. Model Reliability
        self._assess_model_reliability()
        
        # 4. Market Context
        self._analyze_market_context()
        
        # 5. Generate Dashboard
        self._generate_dashboard()
        
        return self.results
    
    def _validate_current_predictions(self):
        """Validate current predictions against actual price."""
        print("\n📊 1. CURRENT PREDICTION ACCURACY")
        print("-" * 50)
        
        try:
            # Load predictions
            prediction_file = f"{self.data_dir}/{self.ticker}_timeframe_predictions.csv"
            if not os.path.exists(prediction_file):
                self.results['current_accuracy'] = {"error": "No predictions found"}
                return
            
            df_predictions = pd.read_csv(prediction_file)
            
            # Get current price
            stock = yf.Ticker(self.ticker)
            current_data = stock.history(period="1d")
            current_price = current_data['Close'].iloc[-1]
            
            # Calculate accuracy for each timeframe
            accuracy_results = {}
            
            for timeframe in ['Short-Term', 'Medium-Term', 'Long-Term']:
                tf_data = df_predictions[df_predictions['Timeframe'] == timeframe]
                if not tf_data.empty:
                    errors = []
                    accuracies = []
                    
                    for _, row in tf_data.iterrows():
                        predicted = row['Predicted_Price']
                        error = abs(predicted - current_price)
                        error_percent = (error / current_price) * 100
                        accuracy = max(0, 1 - error_percent/100)
                        
                        errors.append(error_percent)
                        accuracies.append(accuracy)
                    
                    accuracy_results[timeframe] = {
                        'mean_error': np.mean(errors),
                        'mean_accuracy': np.mean(accuracies),
                        'min_accuracy': np.min(accuracies),
                        'max_accuracy': np.max(accuracies),
                        'std_accuracy': np.std(accuracies)
                    }
                    
                    print(f"  {timeframe}:")
                    print(f"    Mean Accuracy: {np.mean(accuracies)*100:.1f}%")
                    print(f"    Mean Error: {np.mean(errors):.1f}%")
                    print(f"    Accuracy Range: {np.min(accuracies)*100:.1f}% - {np.max(accuracies)*100:.1f}%")
            
            self.results['current_accuracy'] = {
                'current_price': current_price,
                'timeframes': accuracy_results
            }
            
        except Exception as e:
            self.results['current_accuracy'] = {"error": str(e)}
    
    def _analyze_historical_performance(self):
        """Analyze historical performance of the stock."""
        print("\n📈 2. HISTORICAL PERFORMANCE ANALYSIS")
        print("-" * 50)
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get different time periods
            periods = {
                '1M': '1mo',
                '3M': '3mo', 
                '6M': '6mo',
                '1Y': '1y',
                '2Y': '2y'
            }
            
            historical_analysis = {}
            
            for period_name, period_code in periods.items():
                try:
                    data = stock.history(period=period_code)
                    if not data.empty:
                        # Calculate metrics
                        returns = data['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)  # Annualized
                        trend = returns.mean() * 252  # Annualized
                        sharpe = trend / volatility if volatility > 0 else 0
                        
                        # Price performance
                        total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                        
                        historical_analysis[period_name] = {
                            'total_return': total_return,
                            'volatility': volatility * 100,
                            'trend': trend * 100,
                            'sharpe_ratio': sharpe,
                            'max_drawdown': self._calculate_max_drawdown(data['Close'])
                        }
                        
                        print(f"  {period_name}:")
                        print(f"    Total Return: {total_return:.1f}%")
                        print(f"    Volatility: {volatility*100:.1f}%")
                        print(f"    Trend: {trend*100:.1f}%")
                        print(f"    Sharpe Ratio: {sharpe:.2f}")
                        
                except Exception as e:
                    print(f"  {period_name}: Error - {e}")
            
            self.results['historical_performance'] = historical_analysis
            
        except Exception as e:
            self.results['historical_performance'] = {"error": str(e)}
    
    def _assess_model_reliability(self):
        """Assess the reliability of the prediction models."""
        print("\n🎯 3. MODEL RELIABILITY ASSESSMENT")
        print("-" * 50)
        
        try:
            # Check if we have model files
            model_files = []
            for file in os.listdir(f"{self.data_dir}/models"):
                if self.ticker in file and file.endswith('.pkl'):
                    model_files.append(file)
            
            if not model_files:
                self.results['model_reliability'] = {"error": "No model files found"}
                return
            
            # Analyze model consistency
            reliability_metrics = {
                'models_found': len(model_files),
                'model_types': list(set([f.split('_')[-1].replace('.pkl', '') for f in model_files])),
                'data_quality': self._assess_data_quality(),
                'prediction_consistency': self._assess_prediction_consistency()
            }
            
            print(f"  Models Found: {len(model_files)}")
            print(f"  Model Types: {', '.join(reliability_metrics['model_types'])}")
            print(f"  Data Quality: {reliability_metrics['data_quality']}")
            print(f"  Prediction Consistency: {reliability_metrics['prediction_consistency']}")
            
            self.results['model_reliability'] = reliability_metrics
            
        except Exception as e:
            self.results['model_reliability'] = {"error": str(e)}
    
    def _analyze_market_context(self):
        """Analyze current market context and conditions."""
        print("\n🌍 4. MARKET CONTEXT ANALYSIS")
        print("-" * 50)
        
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get recent data
            recent_data = stock.history(period="30d")
            if recent_data.empty:
                self.results['market_context'] = {"error": "No recent data available"}
                return
            
            # Calculate market indicators
            current_price = recent_data['Close'].iloc[-1]
            sma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = recent_data['Close'].rolling(50).mean().iloc[-1] if len(recent_data) >= 50 else None
            
            # Volatility analysis
            daily_returns = recent_data['Close'].pct_change().dropna()
            current_volatility = daily_returns.std() * np.sqrt(252)
            
            # Trend analysis
            price_trend = "Bullish" if current_price > sma_20 else "Bearish"
            if sma_50:
                long_trend = "Bullish" if sma_20 > sma_50 else "Bearish"
            else:
                long_trend = "Insufficient Data"
            
            # Support and resistance
            recent_high = recent_data['High'].max()
            recent_low = recent_data['Low'].min()
            support_level = recent_low
            resistance_level = recent_high
            
            market_context = {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'price_trend': price_trend,
                'long_trend': long_trend,
                'volatility': current_volatility * 100,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'price_position': ((current_price - support_level) / (resistance_level - support_level)) * 100
            }
            
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Price Trend: {price_trend}")
            print(f"  Long Trend: {long_trend}")
            print(f"  Volatility: {current_volatility*100:.1f}%")
            print(f"  Support: ${support_level:.2f}")
            print(f"  Resistance: ${resistance_level:.2f}")
            print(f"  Price Position: {market_context['price_position']:.1f}%")
            
            self.results['market_context'] = market_context
            
        except Exception as e:
            self.results['market_context'] = {"error": str(e)}
    
    def _generate_dashboard(self):
        """Generate a comprehensive validation dashboard."""
        print("\n📋 5. VALIDATION DASHBOARD SUMMARY")
        print("=" * 70)
        
        # Overall assessment
        overall_score = self._calculate_overall_score()
        
        print(f"🎯 OVERALL VALIDATION SCORE: {overall_score:.1f}/10")
        print()
        
        # Recommendations
        self._generate_recommendations()
        
        # Save results
        self._save_validation_results()
    
    def _calculate_overall_score(self):
        """Calculate overall validation score."""
        score = 0
        max_score = 10
        
        # Current accuracy (40% weight)
        if 'current_accuracy' in self.results and 'error' not in self.results['current_accuracy']:
            acc = self.results['current_accuracy']
            if 'timeframes' in acc:
                short_term_acc = acc['timeframes'].get('Short-Term', {}).get('mean_accuracy', 0)
                score += short_term_acc * 4  # 40% weight
        
        # Historical performance (30% weight)
        if 'historical_performance' in self.results and 'error' not in self.results['historical_performance']:
            perf = self.results['historical_performance']
            if '1M' in perf:
                sharpe = perf['1M'].get('sharpe_ratio', 0)
                score += min(1, max(0, sharpe + 1)) * 3  # 30% weight
        
        # Model reliability (20% weight)
        if 'model_reliability' in self.results and 'error' not in self.results['model_reliability']:
            rel = self.results['model_reliability']
            models_count = rel.get('models_found', 0)
            score += min(1, models_count / 5) * 2  # 20% weight
        
        # Market context (10% weight)
        if 'market_context' in self.results and 'error' not in self.results['market_context']:
            score += 1  # 10% weight for having market context
        
        return min(10, score)
    
    def _generate_recommendations(self):
        """Generate validation recommendations."""
        print("💡 VALIDATION RECOMMENDATIONS:")
        print("-" * 50)
        
        recommendations = []
        
        # Accuracy-based recommendations
        if 'current_accuracy' in self.results and 'error' not in self.results['current_accuracy']:
            acc = self.results['current_accuracy']
            if 'timeframes' in acc:
                short_acc = acc['timeframes'].get('Short-Term', {}).get('mean_accuracy', 0)
                if short_acc > 0.95:
                    recommendations.append("✅ Excellent short-term accuracy - predictions are highly reliable")
                elif short_acc > 0.90:
                    recommendations.append("✅ Good short-term accuracy - predictions are reliable")
                elif short_acc > 0.80:
                    recommendations.append("⚠️ Moderate accuracy - consider model improvements")
                else:
                    recommendations.append("❌ Low accuracy - significant model improvements needed")
        
        # Volatility-based recommendations
        if 'market_context' in self.results and 'error' not in self.results['market_context']:
            vol = self.results['market_context'].get('volatility', 0)
            if vol > 30:
                recommendations.append("⚠️ High volatility detected - predictions may be less reliable")
            elif vol < 15:
                recommendations.append("✅ Low volatility - predictions should be more stable")
        
        # Model-based recommendations
        if 'model_reliability' in self.results and 'error' not in self.results['model_reliability']:
            models = self.results['model_reliability'].get('models_found', 0)
            if models >= 5:
                recommendations.append("✅ Multiple models available - good ensemble approach")
            elif models >= 3:
                recommendations.append("⚠️ Limited models - consider training more algorithms")
            else:
                recommendations.append("❌ Insufficient models - need more training")
        
        # General recommendations
        recommendations.extend([
            "📊 Monitor predictions daily for accuracy tracking",
            "🔄 Retrain models weekly with new data",
            "📈 Use multiple timeframes for better validation",
            "🎯 Consider market conditions when interpreting results",
            "📋 Keep validation logs for performance tracking"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def _save_validation_results(self):
        """Save validation results to file."""
        try:
            import json
            results_file = f"{self.data_dir}/{self.ticker}_validation_dashboard.json"
            
            # Add metadata
            self.results['metadata'] = {
                'ticker': self.ticker,
                'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'overall_score': self._calculate_overall_score()
            }
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\n💾 Validation results saved: {results_file}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min() * 100
    
    def _assess_data_quality(self):
        """Assess the quality of training data."""
        try:
            data_file = f"{self.data_dir}/{self.ticker}_short_term_data.csv"
            if os.path.exists(data_file):
                df = pd.read_csv(data_file)
                missing_data = df.isnull().sum().sum()
                total_data = df.size
                quality_score = 1 - (missing_data / total_data)
                
                if quality_score > 0.95:
                    return "Excellent"
                elif quality_score > 0.90:
                    return "Good"
                elif quality_score > 0.80:
                    return "Fair"
                else:
                    return "Poor"
            return "Unknown"
        except:
            return "Unknown"
    
    def _assess_prediction_consistency(self):
        """Assess consistency of predictions across timeframes."""
        try:
            prediction_file = f"{self.data_dir}/{self.ticker}_timeframe_predictions.csv"
            if os.path.exists(prediction_file):
                df = pd.read_csv(prediction_file)
                
                # Check if predictions follow logical progression
                short_term = df[df['Timeframe'] == 'Short-Term']['Predicted_Price'].values
                medium_term = df[df['Timeframe'] == 'Medium-Term']['Predicted_Price'].values
                long_term = df[df['Timeframe'] == 'Long-Term']['Predicted_Price'].values
                
                if len(short_term) > 0 and len(medium_term) > 0 and len(long_term) > 0:
                    # Check if predictions are generally increasing (bullish) or decreasing (bearish)
                    short_avg = np.mean(short_term)
                    medium_avg = np.mean(medium_term)
                    long_avg = np.mean(long_term)
                    
                    # Calculate consistency score
                    if (medium_avg > short_avg and long_avg > medium_avg) or \
                       (medium_avg < short_avg and long_avg < medium_avg):
                        return "Consistent"
                    else:
                        return "Mixed"
                return "Insufficient Data"
            return "No Predictions"
        except:
            return "Error"

def run_validation_dashboard(ticker: str):
    """Run the comprehensive validation dashboard."""
    dashboard = ValidationDashboard(ticker)
    return dashboard.run_full_validation()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("Enter stock ticker for validation dashboard: ").upper()
    
    run_validation_dashboard(ticker)
