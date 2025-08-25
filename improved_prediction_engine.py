#!/usr/bin/env python3
"""
Improved Prediction Engine
Advanced stock prediction with sophisticated algorithms and feature engineering
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Technical Analysis
import talib

class ImprovedPredictionEngine:
    """
    Advanced prediction engine with sophisticated algorithms
    """
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = []
        
    def load_and_prepare_data(self):
        """Load and prepare data with advanced feature engineering."""
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
            df = self._add_advanced_features(df)
            df = self._add_market_regime_features(df)
            df = self._add_volatility_features(df)
            df = self._add_momentum_features(df)
            df = self._add_support_resistance_features(df)
            
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
    
    def _add_advanced_features(self, df):
        """Add advanced technical indicators and features."""
        try:
            # Price-based features
            df['Price_Range'] = df['High'] - df['Low']
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_2d'] = df['Close'].pct_change(2)
            df['Price_Change_5d'] = df['Close'].pct_change(5)
            df['Price_Change_10d'] = df['Close'].pct_change(10)
            
            # Volume features
            df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
            df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
            df['Volume_Price_Trend'] = (df['Volume'] * df['Price_Change']).cumsum()
            
            # Advanced moving averages
            df['EMA_5'] = talib.EMA(df['Close'], timeperiod=5)
            df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
            df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
            df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
            
            # Moving average crossovers
            df['MA_Cross_5_20'] = df['EMA_5'] - df['EMA_20']
            df['MA_Cross_10_50'] = df['EMA_10'] - df['EMA_50']
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # RSI variations
            df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
            df['RSI_5'] = talib.RSI(df['Close'], timeperiod=5)
            df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
            
            # Stochastic
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
            
            # Williams %R
            df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'])
            
            # CCI
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
            
            # ATR
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
            
            # ADX
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
            
            # OBV
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            
            # MFI
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding advanced features: {e}")
            return df
    
    def _add_market_regime_features(self, df):
        """Add market regime detection features."""
        try:
            # Trend strength
            df['Trend_Strength'] = abs(df['EMA_10'] - df['EMA_50']) / df['EMA_50']
            
            # Volatility regime
            df['Volatility_20d'] = df['Close'].rolling(20).std()
            df['Volatility_Regime'] = df['Volatility_20d'].rolling(50).mean()
            
            # Market momentum
            df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
            df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
            
            # Market efficiency ratio
            df['Market_Efficiency'] = abs(df['Close'] - df['Close'].shift(20)) / df['Close'].rolling(20).apply(lambda x: sum(abs(x.diff().dropna())))
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding market regime features: {e}")
            return df
    
    def _add_volatility_features(self, df):
        """Add volatility-based features."""
        try:
            # GARCH-like volatility
            returns = df['Close'].pct_change().dropna()
            df['Volatility_GARCH'] = returns.rolling(20).std()
            
            # Parkinson volatility
            df['Volatility_Parkinson'] = np.sqrt(1/(4*np.log(2)) * (np.log(df['High']/df['Low'])**2).rolling(20).mean())
            
            # Garman-Klass volatility
            df['Volatility_GK'] = np.sqrt((0.5 * (np.log(df['High']/df['Low'])**2) - 
                                          (2*np.log(2)-1) * (np.log(df['Close']/df['Open'])**2)).rolling(20).mean())
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding volatility features: {e}")
            return df
    
    def _add_momentum_features(self, df):
        """Add momentum-based features."""
        try:
            # Price momentum
            for period in [3, 5, 10, 15, 20]:
                df[f'Momentum_{period}d'] = df['Close'].pct_change(period)
            
            # Volume momentum
            df['Volume_Momentum_5d'] = df['Volume'].pct_change(5)
            df['Volume_Momentum_10d'] = df['Volume'].pct_change(10)
            
            # RSI momentum
            df['RSI_Momentum'] = df['RSI_14'].diff()
            
            # MACD momentum
            df['MACD_Momentum'] = df['MACD'].diff()
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding momentum features: {e}")
            return df
    
    def _add_support_resistance_features(self, df):
        """Add support and resistance level features."""
        try:
            # Pivot points
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low']
            df['S1'] = 2 * df['Pivot'] - df['High']
            df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
            df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
            
            # Distance from support/resistance
            df['Distance_R1'] = (df['R1'] - df['Close']) / df['Close']
            df['Distance_S1'] = (df['Close'] - df['S1']) / df['Close']
            
            # Fibonacci retracement levels
            high_20d = df['High'].rolling(20).max()
            low_20d = df['Low'].rolling(20).min()
            range_20d = high_20d - low_20d
            
            df['Fib_38'] = high_20d - 0.382 * range_20d
            df['Fib_50'] = high_20d - 0.5 * range_20d
            df['Fib_61'] = high_20d - 0.618 * range_20d
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding support/resistance features: {e}")
            return df
    
    def prepare_features(self, df):
        """Prepare features for prediction."""
        try:
            # Select relevant features
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Price_Range', 'Price_Change', 'Price_Change_2d', 'Price_Change_5d', 'Price_Change_10d',
                'Volume_MA_5', 'Volume_MA_10', 'Volume_Ratio', 'Volume_Price_Trend',
                'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
                'MA_Cross_5_20', 'MA_Cross_10_50',
                'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
                'RSI_14', 'RSI_5', 'RSI_21',
                'MACD', 'MACD_Signal', 'MACD_Hist',
                'Stoch_K', 'Stoch_D',
                'Williams_R', 'CCI', 'ATR', 'ADX', 'OBV', 'MFI',
                'Trend_Strength', 'Volatility_20d', 'Volatility_Regime',
                'Momentum_5d', 'Momentum_10d', 'Momentum_20d',
                'Market_Efficiency',
                'Volatility_GARCH', 'Volatility_Parkinson', 'Volatility_GK',
                'Momentum_3d', 'Momentum_15d',
                'Volume_Momentum_5d', 'Volume_Momentum_10d',
                'RSI_Momentum', 'MACD_Momentum',
                'Pivot', 'R1', 'S1', 'R2', 'S2',
                'Distance_R1', 'Distance_S1',
                'Fib_38', 'Fib_50', 'Fib_61'
            ]
            
            # Get available features
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) < 10:
                print("❌ Insufficient features available")
                return None, None
            
            # Prepare feature matrix
            X = df[available_features].dropna()
            
            if len(X) < 50:
                print("❌ Insufficient data points")
                return None, None
            
            # Clean the data to remove infinite and extreme values
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
            
            # Remove extreme outliers (beyond 3 standard deviations)
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    Q1 = X_clean[col].quantile(0.01)
                    Q3 = X_clean[col].quantile(0.99)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap extreme values instead of removing rows
                    X_clean[col] = X_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            print(f"✅ Data cleaned: {X_clean.shape}")
            return X_clean
            
        except Exception as e:
            print(f"❌ Error cleaning feature data: {e}")
            return None
    
    def train_advanced_models(self, X, y):
        """Train advanced ensemble models."""
        try:
            print("🤖 Training advanced prediction models...")
            
            # Split data for time series
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Initialize models
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=5,
                    min_samples_leaf=2, random_state=42, n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=8,
                    min_samples_split=5, random_state=42
                ),
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=8,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                ),
                'LightGBM': lgb.LGBMRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=8,
                    subsample=0.8, colsample_bytree=0.8, random_state=42
                ),
                'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1),
                'MLP': MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500,
                    random_state=42, early_stopping=True
                )
            }
            
            # Train individual models
            for name, model in models.items():
                print(f"   Training {name}...")
                
                # Scale features for certain models
                if name in ['SVR', 'Ridge', 'Lasso', 'MLP']:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[name] = scaler
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                # Cross-validation score
                if name not in ['SVR', 'MLP']:  # Skip slow models for CV
                    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                    rmse = np.sqrt(-cv_scores.mean())
                    print(f"     CV RMSE: {rmse:.4f}")
                
                self.models[name] = model
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            
            # Create ensemble model
            print("   Creating ensemble model...")
            ensemble_models = [
                ('rf', self.models['RandomForest']),
                ('gb', self.models['GradientBoosting']),
                ('xgb', self.models['XGBoost']),
                ('lgb', self.models['LightGBM'])
            ]
            
            self.models['Ensemble'] = VotingRegressor(
                estimators=ensemble_models,
                weights=[0.25, 0.25, 0.25, 0.25]
            )
            
            # Train ensemble
            self.models['Ensemble'].fit(X, y)
            
            print("✅ All models trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False
    
    def generate_predictions(self, X, days_ahead=5):
        """Generate predictions using all trained models."""
        try:
            predictions = {}
            
            # Get last data point
            last_data = X.iloc[-1:].values
            last_data_scaled = {}
            
            # Scale data for models that need it
            for name, scaler in self.scalers.items():
                last_data_scaled[name] = scaler.transform(last_data)
            
            # Generate predictions for each model
            for name, model in self.models.items():
                if name in self.scalers:
                    pred = model.predict(last_data_scaled[name])[0]
                else:
                    pred = model.predict(last_data)[0]
                
                predictions[name] = pred
            
            # Generate multi-day predictions using ensemble
            multi_day_predictions = self._generate_multi_day_predictions(X, days_ahead)
            
            return predictions, multi_day_predictions
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            return None, None
    
    def _generate_multi_day_predictions(self, X, days_ahead):
        """Generate multi-day predictions using recursive approach."""
        try:
            predictions = []
            current_data = X.iloc[-1:].copy()
            
            for day in range(days_ahead):
                # Generate prediction for next day
                if 'Ensemble' in self.models:
                    pred = self.models['Ensemble'].predict(current_data.values)[0]
                else:
                    # Use average of available models
                    preds = []
                    for name, model in self.models.items():
                        if name in self.scalers:
                            preds.append(model.predict(self.scalers[name].transform(current_data.values))[0])
                        else:
                            preds.append(model.predict(current_data.values)[0])
                    pred = np.mean(preds)
                
                predictions.append(pred)
                
                # Update features for next prediction (simplified approach)
                if day < days_ahead - 1:
                    # Update price-based features
                    current_data['Close'] = pred
                    current_data['Price_Change'] = (pred - X.iloc[-1]['Close']) / X.iloc[-1]['Close']
                    
                    # Update other features based on prediction
                    current_data['Price_Range'] = current_data['Price_Range'] * 0.95  # Slight decay
                    current_data['Volume_Ratio'] = current_data['Volume_Ratio'] * 0.98  # Slight decay
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Error generating multi-day predictions: {e}")
            return None
    
    def calculate_prediction_confidence(self, predictions):
        """Calculate confidence intervals for predictions."""
        try:
            # Get all model predictions
            model_predictions = list(predictions.values())
            
            # Calculate statistics
            mean_pred = np.mean(model_predictions)
            std_pred = np.std(model_predictions)
            
            # Confidence intervals
            confidence_68 = (mean_pred - std_pred, mean_pred + std_pred)
            confidence_95 = (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
            
            # Model agreement (lower std = higher agreement)
            agreement_score = 1 / (1 + std_pred/mean_pred)
            
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
    
    def generate_trading_signals(self, predictions, current_price, confidence):
        """Generate trading signals based on predictions."""
        try:
            mean_pred = confidence['mean']
            agreement_score = confidence['agreement_score']
            
            # Calculate expected return
            expected_return = (mean_pred - current_price) / current_price * 100
            
            # Signal strength based on agreement and return magnitude
            signal_strength = abs(expected_return) * agreement_score
            
            # Generate signal
            if expected_return > 2 and signal_strength > 1.5:
                signal = "STRONG_BUY"
                confidence_level = "HIGH"
            elif expected_return > 1 and signal_strength > 1.0:
                signal = "BUY"
                confidence_level = "MEDIUM"
            elif expected_return < -2 and signal_strength > 1.5:
                signal = "STRONG_SELL"
                confidence_level = "HIGH"
            elif expected_return < -1 and signal_strength > 1.0:
                signal = "SELL"
                confidence_level = "MEDIUM"
            else:
                signal = "HOLD"
                confidence_level = "LOW"
            
            return {
                'signal': signal,
                'confidence_level': confidence_level,
                'expected_return': expected_return,
                'signal_strength': signal_strength,
                'agreement_score': agreement_score
            }
            
        except Exception as e:
            print(f"Warning: Error generating trading signals: {e}")
            return None
    
    def save_models(self):
        """Save trained models."""
        try:
            os.makedirs("models", exist_ok=True)
            
            for name, model in self.models.items():
                model_path = f"models/{self.ticker}_improved_{name.lower()}_model.pkl"
                joblib.dump(model, model_path)
                print(f"💾 Saved {name} model: {model_path}")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                scaler_path = f"models/{self.ticker}_improved_{name.lower()}_scaler.pkl"
                joblib.dump(scaler, scaler_path)
                print(f"💾 Saved {name} scaler: {scaler_path}")
            
            # Save feature importance
            importance_path = f"models/{self.ticker}_improved_feature_importance.pkl"
            joblib.dump(self.feature_importance, importance_path)
            print(f"💾 Saved feature importance: {importance_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load pre-trained models."""
        try:
            model_names = ['randomforest', 'gradientboosting', 'xgboost', 'lightgbm', 'ensemble']
            
            for name in model_names:
                model_path = f"models/{self.ticker}_improved_{name}_model.pkl"
                if os.path.exists(model_path):
                    self.models[name.title()] = joblib.load(model_path)
                    print(f"📂 Loaded {name} model")
            
            # Load scalers
            scaler_names = ['svr', 'ridge', 'lasso', 'mlp']
            for name in scaler_names:
                scaler_path = f"models/{self.ticker}_improved_{name}_scaler.pkl"
                if os.path.exists(scaler_path):
                    self.scalers[name.upper()] = joblib.load(scaler_path)
                    print(f"📂 Loaded {name} scaler")
            
            # Load feature importance
            importance_path = f"models/{self.ticker}_improved_feature_importance.pkl"
            if os.path.exists(importance_path):
                self.feature_importance = joblib.load(importance_path)
                print(f"📂 Loaded feature importance")
            
            return len(self.models) > 0
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def run_improved_prediction(ticker="AAPL"):
    """Run improved prediction for a given ticker."""
    print(f"🚀 Improved Prediction Engine for {ticker}")
    print("=" * 60)
    
    # Initialize engine
    engine = ImprovedPredictionEngine(ticker)
    
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
        if not engine.train_advanced_models(X, y):
            return False
        engine.save_models()
    else:
        print("✅ Using pre-trained models")
    
    # Generate predictions
    print("🔮 Generating predictions...")
    predictions, multi_day_predictions = engine.generate_predictions(X, days_ahead=5)
    
    if predictions is None:
        return False
    
    # Get current price
    current_price = float(df['Close'].iloc[-1])
    
    # Calculate confidence
    confidence = engine.calculate_prediction_confidence(predictions)
    
    # Generate trading signals
    signals = engine.generate_trading_signals(predictions, current_price, confidence)
    
    # Display results
    display_improved_results(ticker, predictions, multi_day_predictions, current_price, confidence, signals)
    
    return True

def display_improved_results(ticker, predictions, multi_day_predictions, current_price, confidence, signals):
    """Display improved prediction results."""
    print("\n🎯 IMPROVED PREDICTION RESULTS")
    print("=" * 70)
    print(f"📊 Stock: {ticker}")
    print(f"💰 CURRENT PRICE: ₹{current_price:.2f}")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print()
    
    # Display individual model predictions
    print("🤖 Individual Model Predictions:")
    for model_name, pred in predictions.items():
        change = pred - current_price
        change_pct = (change / current_price) * 100
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"   • {model_name:12}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
    
    print()
    
    # Display ensemble prediction with confidence
    if 'Ensemble' in predictions:
        ensemble_pred = predictions['Ensemble']
        change = ensemble_pred - current_price
        change_pct = (change / current_price) * 100
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print("🎯 ENSEMBLE PREDICTION:")
        print(f"   Predicted Price: ₹{ensemble_pred:.2f} ({direction} {change_pct:+.2f}%)")
        print(f"   Confidence 68%: ₹{confidence['confidence_68'][0]:.2f} - ₹{confidence['confidence_68'][1]:.2f}")
        print(f"   Confidence 95%: ₹{confidence['confidence_95'][0]:.2f} - ₹{confidence['confidence_95'][1]:.2f}")
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
    
    # Display trading signals
    if signals:
        print("💡 TRADING SIGNALS:")
        print(f"   Signal: {signals['signal']}")
        print(f"   Confidence: {signals['confidence_level']}")
        print(f"   Expected Return: {signals['expected_return']:+.2f}%")
        print(f"   Signal Strength: {signals['signal_strength']:.2f}")
        print(f"   Model Agreement: {signals['agreement_score']:.2%}")
    
    print()
    
    # Price summary
    if 'Ensemble' in predictions:
        print("📋 PRICE SUMMARY:")
        print(f"   Current Price: ₹{current_price:.2f}")
        print(f"   Predicted Price: ₹{ensemble_pred:.2f}")
        print(f"   Expected Change: {change_pct:+.2f}%")
        print(f"   Prediction Range: ₹{confidence['confidence_68'][0]:.2f} - ₹{confidence['confidence_68'][1]:.2f}")
    
    print("=" * 70)

def main():
    """Main function."""
    print("🚀 Improved Stock Prediction Engine")
    print("=" * 50)
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
    
    print(f"\n🎯 Running improved prediction for {ticker}...")
    
    # Run improved prediction
    success = run_improved_prediction(ticker)
    
    if success:
        print(f"\n✅ {ticker} improved prediction completed successfully!")
    else:
        print(f"\n❌ {ticker} improved prediction failed!")

if __name__ == "__main__":
    main()
