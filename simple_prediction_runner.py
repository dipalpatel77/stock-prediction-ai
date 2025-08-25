#!/usr/bin/env python3
"""
Simple Prediction Runner
A simplified version that uses only basic, reliable models
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Basic ML imports only
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import talib

class SimplePredictionEngine:
    """
    Simple prediction engine with basic, reliable models
    """
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.models = {}
        self.scalers = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data with basic features only."""
        try:
            # Load enhanced data
            data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
            if not os.path.exists(data_file):
                print(f"❌ Enhanced data file not found: {data_file}")
                return None
            
            print(f"📊 Loading enhanced data from: {data_file}")
            df = pd.read_csv(data_file)
            print(f"📈 Original data shape: {df.shape}")
            
            # Clean and prepare data
            df = self._clean_data(df)
            df = self._add_basic_features(df)
            
            print(f"📈 Enhanced data shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _clean_data(self, df):
        """Clean and prepare the dataset."""
        # Remove unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(unnamed_cols, axis=1)
        
        # Handle date column
        if len(df.columns) > 1 and df.columns[1] == 'Price':
            df['Date'] = df['Price']
            df = df.drop('Price', axis=1)
        elif len(df.columns) > 1 and df.columns[1] == 'Date':
            df['Date'] = df['Date']
            df = df.drop('Date', axis=1)
        
        # Set date as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Forward fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _add_basic_features(self, df):
        """Add only basic, reliable technical indicators."""
        try:
            # Basic price features
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Range'] = df['High'] - df['Low']
            
            # Simple moving averages
            df['SMA_5'] = df['Close'].rolling(5).mean()
            df['SMA_10'] = df['Close'].rolling(10).mean()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            
            # Volume features
            df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
            
            # Basic RSI
            df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
            
            # Basic MACD
            df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding basic features: {e}")
            return df
    
    def prepare_features(self, df):
        """Prepare features for prediction."""
        try:
            # Select only basic, reliable features
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Price_Change', 'Price_Range',
                'SMA_5', 'SMA_10', 'SMA_20',
                'Volume_Ratio',
                'RSI_14',
                'MACD', 'MACD_Signal',
                'BB_Upper', 'BB_Middle', 'BB_Lower'
            ]
            
            # Get available features
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) < 8:
                print("❌ Insufficient features available")
                return None, None
            
            # Prepare feature matrix
            X = df[available_features].dropna()
            
            if len(X) < 50:
                print("❌ Insufficient data points")
                return None, None
            
            # Clean the data
            X = self._clean_feature_data(X)
            
            if X is None or len(X) < 50:
                print("❌ Insufficient data after cleaning")
                return None, None
            
            # Create target variable (next day's close price)
            y = X['Close'].shift(-1).dropna()
            X = X[:-1]  # Remove last row since we don't have target
            
            # Ensure X and y have the same length
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            
            print(f"✅ Prepared features: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None, None
    
    def _clean_feature_data(self, X):
        """Clean feature data by removing infinite and extreme values."""
        try:
            print("🧹 Cleaning feature data...")
            
            # Replace infinite values with NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Remove rows with any NaN values
            X_clean = X.dropna()
            
            if len(X_clean) < len(X) * 0.5:  # If we lost more than 50% of data
                print(f"⚠️ Warning: Lost {(len(X) - len(X_clean))} rows due to NaN values")
            
            # Cap extreme values for each column
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use percentile-based capping
                    lower_bound = X_clean[col].quantile(0.001)
                    upper_bound = X_clean[col].quantile(0.999)
                    X_clean[col] = X_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            print(f"✅ Data cleaned: {X_clean.shape}")
            return X_clean
            
        except Exception as e:
            print(f"❌ Error cleaning feature data: {e}")
            return None
    
    def train_simple_models(self, X, y):
        """Train simple, reliable models."""
        try:
            print("🤖 Training simple prediction models...")
            
            # Initialize simple models
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    min_samples_leaf=2, random_state=42, n_jobs=-1
                ),
                'LinearRegression': LinearRegression()
            }
            
            # Train models
            for name, model in models.items():
                print(f"   Training {name}...")
                
                # Scale features for linear regression
                if name == 'LinearRegression':
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[name] = scaler
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                self.models[name] = model
            
            # Create simple ensemble (average of predictions)
            print("   Creating simple ensemble...")
            self.models['Ensemble'] = 'average'  # We'll implement this in prediction
            
            print("✅ All models trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False
    
    def generate_predictions(self, X, days_ahead=5):
        """Generate predictions using trained models."""
        try:
            predictions = {}
            
            # Get last data point
            last_data = X.iloc[-1:].values
            
            # Generate predictions for each model
            for name, model in self.models.items():
                if name == 'Ensemble':
                    # Calculate ensemble as average of other models
                    other_preds = []
                    for other_name, other_model in self.models.items():
                        if other_name != 'Ensemble':
                            if other_name in self.scalers:
                                pred = other_model.predict(self.scalers[other_name].transform(last_data))[0]
                            else:
                                pred = other_model.predict(last_data)[0]
                            other_preds.append(pred)
                    predictions[name] = np.mean(other_preds)
                else:
                    if name in self.scalers:
                        pred = model.predict(self.scalers[name].transform(last_data))[0]
                    else:
                        pred = model.predict(last_data)[0]
                    predictions[name] = pred
            
            # Generate simple multi-day predictions
            multi_day_predictions = self._generate_simple_multi_day_predictions(X, days_ahead)
            
            return predictions, multi_day_predictions
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            return None, None
    
    def _generate_simple_multi_day_predictions(self, X, days_ahead):
        """Generate simple multi-day predictions."""
        try:
            predictions = []
            current_price = X['Close'].iloc[-1]
            
            # Simple trend-based prediction
            recent_trend = (X['Close'].iloc[-5:].mean() - X['Close'].iloc[-10:-5].mean()) / X['Close'].iloc[-10:-5].mean()
            
            for day in range(days_ahead):
                # Simple linear extrapolation with trend decay
                trend_factor = recent_trend * (0.9 ** day)  # Trend decays over time
                pred = current_price * (1 + trend_factor)
                predictions.append(pred)
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Error generating multi-day predictions: {e}")
            return None
    
    def calculate_prediction_confidence(self, predictions):
        """Calculate simple confidence intervals."""
        try:
            # Get all model predictions
            model_predictions = list(predictions.values())
            
            # Calculate statistics
            mean_pred = np.mean(model_predictions)
            std_pred = np.std(model_predictions)
            
            # Simple confidence intervals
            confidence_68 = (mean_pred - std_pred, mean_pred + std_pred)
            confidence_95 = (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
            
            # Simple agreement score
            agreement_score = 1 / (1 + std_pred/mean_pred) if mean_pred != 0 else 0
            
            return {
                'mean': mean_pred,
                'std': std_pred,
                'confidence_68': confidence_68,
                'confidence_95': confidence_95,
                'agreement_score': agreement_score
            }
            
        except Exception as e:
            print(f"Warning: Error calculating confidence: {e}")
            return None
    
    def save_models(self):
        """Save trained models."""
        try:
            os.makedirs("models", exist_ok=True)
            
            for name, model in self.models.items():
                if name != 'Ensemble':  # Don't save ensemble as it's just a string
                    model_path = f"models/{self.ticker}_simple_{name.lower()}_model.pkl"
                    joblib.dump(model, model_path)
                    print(f"💾 Saved {name} model: {model_path}")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                scaler_path = f"models/{self.ticker}_simple_{name.lower()}_scaler.pkl"
                joblib.dump(scaler, scaler_path)
                print(f"💾 Saved {name} scaler: {scaler_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load pre-trained models."""
        try:
            model_names = ['randomforest', 'linearregression']
            
            for name in model_names:
                model_path = f"models/{self.ticker}_simple_{name}_model.pkl"
                if os.path.exists(model_path):
                    self.models[name.title()] = joblib.load(model_path)
                    print(f"📂 Loaded {name} model")
            
            # Load scalers
            scaler_path = f"models/{self.ticker}_simple_linearregression_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scalers['LinearRegression'] = joblib.load(scaler_path)
                print(f"📂 Loaded LinearRegression scaler")
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def run_simple_prediction(ticker="AAPL", days_ahead=5):
    """Run simple prediction for a given ticker."""
    try:
        print(f"🚀 Simple Prediction Engine for {ticker}")
        print("=" * 60)
        
        # Initialize engine
        engine = SimplePredictionEngine(ticker)
        
        # Load and prepare data
        df = engine.load_and_prepare_data()
        if df is None:
            return False
        
        # Prepare features
        X, y = engine.prepare_features(df)
        if X is None or y is None:
            return False
        
        # Check if models exist
        models_exist = engine.load_models()
        
        # Train models if they don't exist
        if not models_exist:
            print("🔄 Training new models...")
            if not engine.train_simple_models(X, y):
                return False
            engine.save_models()
        else:
            print("✅ Using pre-trained models")
        
        # Generate predictions
        print("🔮 Generating predictions...")
        predictions, multi_day_predictions = engine.generate_predictions(X, days_ahead)
        
        if predictions is None:
            return False
        
        # Get current price
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate confidence
        confidence = engine.calculate_prediction_confidence(predictions)
        
        # Display results
        display_simple_results(ticker, predictions, multi_day_predictions, current_price, confidence)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in simple prediction: {e}")
        return False

def display_simple_results(ticker, predictions, multi_day_predictions, current_price, confidence):
    """Display simple prediction results."""
    print("\n🎯 SIMPLE PREDICTION RESULTS")
    print("=" * 60)
    print(f"📊 Stock: {ticker}")
    print(f"💰 CURRENT PRICE: ₹{current_price:.2f}")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    print()
    
    # Display individual model predictions
    print("🤖 Model Predictions:")
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
        
        print("🎯 ENSEMBLE PREDICTION:")
        print(f"   Predicted Price: ₹{ensemble_pred:.2f} ({direction} {change_pct:+.2f}%)")
        if confidence:
            print(f"   Confidence 68%: ₹{confidence['confidence_68'][0]:.2f} - ₹{confidence['confidence_68'][1]:.2f}")
            print(f"   Model Agreement: {confidence['agreement_score']:.2%}")
    
    print()
    
    # Display multi-day predictions
    if multi_day_predictions:
        print("📅 MULTI-DAY FORECAST:")
        for i, pred in enumerate(multi_day_predictions, 1):
            change = pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   • Day {i}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
    
    print()
    
    # Trading recommendation
    if 'Ensemble' in predictions:
        ensemble_pred = predictions['Ensemble']
        change_pct = (ensemble_pred - current_price) / current_price * 100
        
        if change_pct > 2:
            recommendation = "🟢 BUY - Strong upward momentum expected"
        elif change_pct > 0.5:
            recommendation = "🟡 BUY - Moderate upward potential"
        elif change_pct < -2:
            recommendation = "🔴 SELL - Strong downward pressure expected"
        elif change_pct < -0.5:
            recommendation = "🟠 SELL - Moderate downward potential"
        else:
            recommendation = "⚪ HOLD - Stable price movement expected"
        
        print(f"💡 Trading Recommendation: {recommendation}")
    
    print()
    print("=" * 60)

def main():
    """Main function."""
    print("🚀 Simple Stock Prediction System")
    print("=" * 50)
    
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
    
    print(f"\n🎯 Running simple prediction for {ticker}...")
    print(f"📅 Prediction horizon: {days_ahead} days")
    print()
    
    # Run simple prediction
    success = run_simple_prediction(ticker, days_ahead)
    
    if success:
        print(f"\n✅ {ticker} simple prediction completed successfully!")
    else:
        print(f"\n❌ {ticker} simple prediction failed!")

if __name__ == "__main__":
    main()
