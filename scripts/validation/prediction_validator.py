#!/usr/bin/env python3
"""
Prediction Validation Framework
Validates the accuracy of stock price predictions using multiple methods
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PredictionValidator:
    """Validates prediction accuracy using multiple methods."""
    
    def __init__(self, ticker: str, data_dir: str = "data"):
        self.ticker = ticker
        self.data_dir = data_dir
        self.validation_results = {}
        
    def validate_predictions(self) -> Dict:
        """Main validation method that runs all validation checks."""
        print(f"🔍 VALIDATING PREDICTIONS FOR {self.ticker}")
        print("=" * 60)
        
        results = {}
        
        # 1. Historical Backtesting
        print("\n📊 1. HISTORICAL BACKTESTING")
        print("-" * 40)
        results['backtesting'] = self._historical_backtesting()
        
        # 2. Cross-Validation
        print("\n🔄 2. CROSS-VALIDATION")
        print("-" * 40)
        results['cross_validation'] = self._cross_validation()
        
        # 3. Prediction vs Actual (if available)
        print("\n📈 3. PREDICTION vs ACTUAL COMPARISON")
        print("-" * 40)
        results['prediction_accuracy'] = self._prediction_vs_actual()
        
        # 4. Model Performance Metrics
        print("\n🎯 4. MODEL PERFORMANCE METRICS")
        print("-" * 40)
        results['model_metrics'] = self._calculate_model_metrics()
        
        # 5. Generate Validation Report
        print("\n📋 5. GENERATING VALIDATION REPORT")
        print("-" * 40)
        self._generate_validation_report(results)
        
        return results
    
    def _historical_backtesting(self) -> Dict:
        """Perform historical backtesting on past predictions."""
        try:
            # Load historical data
            data_file = f"{self.data_dir}/{self.ticker}_raw_data.csv"
            if not os.path.exists(data_file):
                return {"error": "Historical data not found"}
            
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Simulate predictions for different time periods
            backtest_results = {}
            
            # Test different prediction horizons
            horizons = [1, 3, 7, 14, 30]  # days
            
            for horizon in horizons:
                mae_scores = []
                rmse_scores = []
                accuracy_scores = []
                
                # Use rolling window for backtesting
                window_size = 252  # 1 year of trading days
                
                for i in range(window_size, len(df) - horizon):
                    # Training data
                    train_data = df.iloc[i-window_size:i]
                    
                    # Actual future price
                    actual_price = df.iloc[i + horizon - 1]['Close']
                    
                    # Simple prediction (using trend)
                    recent_trend = train_data['Close'].pct_change().mean()
                    predicted_price = train_data['Close'].iloc[-1] * (1 + recent_trend * horizon)
                    
                    # Calculate errors
                    mae = abs(predicted_price - actual_price)
                    rmse = (predicted_price - actual_price) ** 2
                    accuracy = 1 - abs(predicted_price - actual_price) / actual_price
                    
                    mae_scores.append(mae)
                    rmse_scores.append(rmse)
                    accuracy_scores.append(max(0, accuracy))
                
                backtest_results[f'{horizon}_day'] = {
                    'mae': np.mean(mae_scores),
                    'rmse': np.sqrt(np.mean(rmse_scores)),
                    'accuracy': np.mean(accuracy_scores),
                    'samples': len(mae_scores)
                }
                
                print(f"  {horizon}-day horizon: MAE={np.mean(mae_scores):.2f}, "
                      f"RMSE={np.sqrt(np.mean(rmse_scores)):.2f}, "
                      f"Accuracy={np.mean(accuracy_scores)*100:.1f}%")
            
            return backtest_results
            
        except Exception as e:
            return {"error": f"Backtesting failed: {e}"}
    
    def _cross_validation(self) -> Dict:
        """Perform cross-validation on the model."""
        try:
            # Load model data
            data_file = f"{self.data_dir}/{self.ticker}_short_term_data.csv"
            if not os.path.exists(data_file):
                return {"error": "Model data not found"}
            
            df = pd.read_csv(data_file)
            
            # Prepare features and target
            feature_cols = [col for col in df.columns if col != 'Close']
            X = df[feature_cols].values
            y = df['Close'].values
            
            # Simple cross-validation
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Calculate R² score
                r2 = r2_score(y_test, y_pred)
                scores.append(r2)
            
            cv_results = {
                'mean_r2': np.mean(scores),
                'std_r2': np.std(scores),
                'scores': scores
            }
            
            print(f"  Cross-validation R²: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            return cv_results
            
        except Exception as e:
            return {"error": f"Cross-validation failed: {e}"}
    
    def _prediction_vs_actual(self) -> Dict:
        """Compare predictions with actual prices (if available)."""
        try:
            # Check if we have recent predictions
            prediction_file = f"{self.data_dir}/{self.ticker}_timeframe_predictions.csv"
            if not os.path.exists(prediction_file):
                return {"error": "Prediction file not found"}
            
            df_predictions = pd.read_csv(prediction_file)
            
            # Get current actual price
            try:
                stock = yf.Ticker(self.ticker)
                current_data = stock.history(period="1d")
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    
                    # Compare with short-term predictions
                    short_term = df_predictions[df_predictions['Timeframe'] == 'Short-Term']
                    if not short_term.empty:
                        # Get 1-day prediction
                        day1_pred = short_term[short_term['Period'] == 'Day 1']
                        if not day1_pred.empty:
                            predicted_price = day1_pred['Predicted_Price'].iloc[0]
                            error = abs(predicted_price - current_price)
                            error_percent = (error / current_price) * 100
                            
                            result = {
                                'current_price': current_price,
                                'predicted_price': predicted_price,
                                'error': error,
                                'error_percent': error_percent,
                                'accuracy': max(0, 1 - error_percent/100)
                            }
                            
                            print(f"  Current Price: ${current_price:.2f}")
                            print(f"  Predicted Price: ${predicted_price:.2f}")
                            print(f"  Error: ${error:.2f} ({error_percent:.2f}%)")
                            print(f"  Accuracy: {result['accuracy']*100:.1f}%")
                            
                            return result
                
                return {"error": "Could not fetch current price"}
                
            except Exception as e:
                return {"error": f"Price comparison failed: {e}"}
                
        except Exception as e:
            return {"error": f"Prediction comparison failed: {e}"}
    
    def _calculate_model_metrics(self) -> Dict:
        """Calculate various model performance metrics."""
        try:
            # Load training data
            data_file = f"{self.data_dir}/{self.ticker}_short_term_data.csv"
            if not os.path.exists(data_file):
                return {"error": "Training data not found"}
            
            df = pd.read_csv(data_file)
            
            # Calculate basic statistics
            price_stats = {
                'mean_price': df['Close'].mean(),
                'std_price': df['Close'].std(),
                'min_price': df['Close'].min(),
                'max_price': df['Close'].max(),
                'volatility': df['Close'].pct_change().std() * np.sqrt(252),  # Annualized
                'trend': df['Close'].pct_change().mean() * 252  # Annualized
            }
            
            print(f"  Mean Price: ${price_stats['mean_price']:.2f}")
            print(f"  Volatility: {price_stats['volatility']*100:.1f}%")
            print(f"  Annual Trend: {price_stats['trend']*100:.1f}%")
            
            return price_stats
            
        except Exception as e:
            return {"error": f"Metrics calculation failed: {e}"}
    
    def _generate_validation_report(self, results: Dict):
        """Generate a comprehensive validation report."""
        try:
            report = {
                'ticker': self.ticker,
                'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'results': results
            }
            
            # Save report
            report_file = f"{self.data_dir}/{self.ticker}_validation_report.json"
            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"  ✅ Validation report saved: {report_file}")
            
            # Generate summary
            self._print_validation_summary(results)
            
        except Exception as e:
            print(f"  ❌ Report generation failed: {e}")
    
    def _print_validation_summary(self, results: Dict):
        """Print a summary of validation results."""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        # Backtesting summary
        if 'backtesting' in results and 'error' not in results['backtesting']:
            print("\n🔍 HISTORICAL ACCURACY:")
            for horizon, metrics in results['backtesting'].items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    print(f"  {horizon.replace('_', '-')}: {metrics['accuracy']*100:.1f}% accuracy")
        
        # Cross-validation summary
        if 'cross_validation' in results and 'error' not in results['cross_validation']:
            cv = results['cross_validation']
            print(f"\n🔄 MODEL RELIABILITY:")
            print(f"  R² Score: {cv['mean_r2']:.3f} ± {cv['std_r2']:.3f}")
        
        # Prediction accuracy
        if 'prediction_accuracy' in results and 'error' not in results['prediction_accuracy']:
            acc = results['prediction_accuracy']
            print(f"\n📈 CURRENT PREDICTION ACCURACY:")
            print(f"  Accuracy: {acc['accuracy']*100:.1f}%")
            print(f"  Error: {acc['error_percent']:.2f}%")
        
        # Model metrics
        if 'model_metrics' in results and 'error' not in results['model_metrics']:
            metrics = results['model_metrics']
            print(f"\n📊 MARKET CHARACTERISTICS:")
            print(f"  Volatility: {metrics['volatility']*100:.1f}%")
            print(f"  Trend: {metrics['trend']*100:.1f}%")
        
        print("\n💡 VALIDATION RECOMMENDATIONS:")
        print("  • Monitor predictions daily for accuracy tracking")
        print("  • Use multiple timeframes for better validation")
        print("  • Consider market conditions when interpreting results")
        print("  • Regularly retrain models with new data")
        print("=" * 60)

def validate_stock_predictions(ticker: str):
    """Main function to validate predictions for a stock."""
    validator = PredictionValidator(ticker)
    return validator.validate_predictions()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("Enter stock ticker to validate: ").upper()
    
    validate_stock_predictions(ticker)
