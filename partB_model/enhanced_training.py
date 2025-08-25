import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from partC_strategy.enhanced_market_factors import integrate_enhanced_factors
from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
from partC_strategy.economic_indicators import integrate_economic_factors
from partC_strategy.optimized_technical_indicators import OptimizedTechnicalIndicators

class EnhancedStockPredictor:
    """
    Enhanced stock predictor with comprehensive factor integration.
    """
    
    def __init__(self, ticker, lookback_days=60):
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.scaler = RobustScaler()
        self.model = None
        self.history = None
        
    def prepare_enhanced_data(self, df):
        """Prepare data with all enhanced factors."""
        print(f"�� Preparing enhanced data for {self.ticker}...")
        
        # 1. Add advanced technical indicators
        technical_analyzer = OptimizedTechnicalIndicators()
        df = technical_analyzer.add_all_indicators(df)
        
        # 2. Add market factors
        try:
            df, market_factors = integrate_enhanced_factors(df, self.ticker)
        except Exception as e:
            print(f"Warning: Market factors integration failed: {e}")
            market_factors = {}
        
        # 3. Add sentiment factors
        try:
            sentiment_analyzer = OptimizedSentimentAnalyzer()
            sentiment_df = sentiment_analyzer.analyze_stock_sentiment(self.ticker, days_back=30)
            
            # Improved sentiment data validation
            sentiment_factors = {}
            if sentiment_df is not None:
                # Check if it's a DataFrame and has data
                if hasattr(sentiment_df, 'empty'):
                    if not sentiment_df.empty and len(sentiment_df) > 0:
                        # Add sentiment columns to the main dataframe
                        for col in sentiment_df.columns:
                            if col not in df.columns:
                                try:
                                    # Get the first value safely
                                    if len(sentiment_df) > 0 and col in sentiment_df.columns:
                                        value = sentiment_df[col].iloc[0]
                                        if pd.notna(value):  # Check if value is not NaN
                                            df[f'sentiment_{col}'] = value
                                        else:
                                            df[f'sentiment_{col}'] = 0
                                    else:
                                        df[f'sentiment_{col}'] = 0
                                except Exception as col_error:
                                    print(f"Warning: Could not add sentiment column {col}: {col_error}")
                                    df[f'sentiment_{col}'] = 0
                        
                        # Create sentiment factors dictionary
                        try:
                            sentiment_factors = sentiment_df.to_dict('records')[0] if len(sentiment_df) > 0 else {}
                        except:
                            sentiment_factors = {}
                else:
                    # If it's not a DataFrame, try to handle it as a dict or other format
                    if isinstance(sentiment_df, dict):
                        sentiment_factors = sentiment_df
                        for key, value in sentiment_df.items():
                            if key not in df.columns:
                                df[f'sentiment_{key}'] = value
        except Exception as e:
            print(f"Warning: Sentiment analysis failed: {e}")
            sentiment_factors = {}
        
        # 4. Add economic factors
        try:
            df, economic_factors = integrate_economic_factors(df, self.ticker)
        except Exception as e:
            print(f"Warning: Economic factors integration failed: {e}")
            economic_factors = {}
        
        # 5. Create target variable (next day's price)
        df['Target'] = df['Close'].shift(-1)
        
        # 6. Remove rows with NaN values
        df = df.dropna()
        
        print(f"✅ Enhanced data prepared with {len(df.columns)} features")
        return df
    
    def create_enhanced_model(self, input_shape):
        """Create enhanced LSTM model with multiple layers."""
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, data, target_col='Target'):
        """Prepare sequences for LSTM."""
        X, y = [], []
        
        for i in range(self.lookback_days, len(data)):
            X.append(data.iloc[i-self.lookback_days:i].values)
            y.append(data.iloc[i][target_col])
        
        return np.array(X), np.array(y)
    
    def train_enhanced_model(self, df):
        """Train the enhanced model."""
        print(f"🚀 Training enhanced model for {self.ticker}...")
        
        # Prepare data
        df = self.prepare_enhanced_data(df)
        
        # Select features (exclude date and target)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Target']]
        data = df[feature_cols].values
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data)
        
        # Prepare sequences
        X, y = self.prepare_sequences(pd.DataFrame(data_scaled, columns=feature_cols))
        
        # Split data (time series split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and compile model
        self.model = self.create_enhanced_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5),
            ModelCheckpoint(f'models/{self.ticker}_enhanced.h5', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"📊 Enhanced Model Performance for {self.ticker}:")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Train MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def predict_enhanced(self, df, days_ahead=5):
        """Make enhanced predictions."""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        # Prepare data
        df = self.prepare_enhanced_data(df)
        
        # Select features
        feature_cols = [col for col in df.columns if col not in ['Date', 'Target']]
        data = df[feature_cols].values
        
        # Scale data
        data_scaled = self.scaler.transform(data)
        
        # Get last sequence
        last_sequence = data_scaled[-self.lookback_days:].reshape(1, self.lookback_days, len(feature_cols))
        
        # Make predictions
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Predict next value
            pred = self.model.predict(current_sequence)[0][0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = pred  # Update price (assuming Close is first column)
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
        
        return predictions
    
    def save_model(self):
        """Save the trained model."""
        try:
            if self.model is not None:
                self.model.save(f'models/{self.ticker}_enhanced.h5')
                print(f"✅ Enhanced model saved to models/{self.ticker}_enhanced.h5")
            else:
                print("Warning: No model to save")
        except Exception as e:
            print(f"Error saving model: {e}")

# Usage example
def run_enhanced_prediction(ticker='AAPL'):
    """Run enhanced prediction pipeline."""
    print(f"🎯 Starting enhanced prediction for {ticker}")
    
    # Load data
    import yfinance as yf
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y')
    df.reset_index(inplace=True)
    
    # Create predictor
    predictor = EnhancedStockPredictor(ticker)
    
    # Train model
    metrics = predictor.train_enhanced_model(df)
    
    # Make predictions
    predictions = predictor.predict_enhanced(df, days_ahead=5)
    
    print(f"🔮 Enhanced Predictions for {ticker}:")
    for i, pred in enumerate(predictions, 1):
        print(f"Day {i}: ${pred:.2f}")
    
    return predictor, metrics, predictions

if __name__ == "__main__":
    # Run enhanced prediction
    predictor, metrics, predictions = run_enhanced_prediction('AAPL')
