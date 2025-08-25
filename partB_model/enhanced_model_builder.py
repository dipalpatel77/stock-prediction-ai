import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
import joblib
import os

class EnhancedModelBuilder:
    """
    Enhanced model builder with multiple architectures and ensemble methods
    for improved prediction success rate.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_advanced_lstm(self, input_shape, name="advanced_lstm"):
        """Create advanced LSTM model with attention mechanism."""
        model = Sequential([
            # Bidirectional LSTM layers
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers
            metrics=['mae']
        )
        return model
    
    def create_conv_lstm(self, input_shape, name="conv_lstm"):
        """Create CNN-LSTM hybrid model."""
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            # LSTM layers
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dropout(0.2),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def create_ensemble_ml(self, X_train, y_train):
        """Create ensemble of traditional ML models."""
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        
        return models
    
    def create_sequences_advanced(self, data, seq_length, target_col_idx=-1):
        """Create sequences with multiple targets and features."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, target_col_idx])
        return np.array(X), np.array(y)
    
    def train_enhanced_models(self, df, features, seq_length=60, epochs=50, batch_size=32):
        """
        Train multiple enhanced models for ensemble prediction.
        """
        print("🚀 Training Enhanced Models for Better Predictions...")
        
        # Prepare data
        scaler = RobustScaler()  # More robust to outliers
        scaled_data = scaler.fit_transform(df[features])
        
        # Create sequences
        X, y = self.create_sequences_advanced(scaled_data, seq_length)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(f'models/best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train LSTM models
        print("📊 Training Advanced LSTM...")
        lstm_model = self.create_advanced_lstm((X.shape[1], X.shape[2]))
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.models['advanced_lstm'] = lstm_model
        
        print("📊 Training CNN-LSTM Hybrid...")
        conv_lstm_model = self.create_conv_lstm((X.shape[1], X.shape[2]))
        conv_lstm_history = conv_lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.models['conv_lstm'] = conv_lstm_model
        
        # Train traditional ML models
        print("📊 Training Traditional ML Models...")
        X_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        self.create_ensemble_ml(X_flat, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test, X_test_flat)
        
        self.scalers['robust'] = scaler
        return self.models, self.scalers
    
    def evaluate_models(self, X_test, y_test, X_test_flat):
        """Evaluate all models and print metrics."""
        print("\n📈 Model Performance Evaluation:")
        print("=" * 50)
        
        for name, model in self.models.items():
            if 'lstm' in name:
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test_flat)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name.upper()}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²:  {r2:.4f}")
            print()
    
    def ensemble_predict(self, X, weights=None):
        """
        Make ensemble prediction using all trained models.
        """
        if weights is None:
            weights = {
                'advanced_lstm': 0.4,
                'conv_lstm': 0.3,
                'random_forest': 0.2,
                'gradient_boost': 0.1
            }
        
        predictions = {}
        for name, model in self.models.items():
            if 'lstm' in name:
                pred = model.predict(X)
            else:
                X_flat = X.reshape(X.shape[0], -1)
                pred = model.predict(X_flat)
            predictions[name] = pred.flatten()
        
        # Weighted ensemble
        ensemble_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        return ensemble_pred, predictions
    
    def save_models(self, ticker):
        """Save all trained models."""
        os.makedirs('models', exist_ok=True)
        
        for name, model in self.models.items():
            if hasattr(model, 'save'):
                model.save(f'models/{ticker}_{name}.h5')
            else:
                joblib.dump(model, f'models/{ticker}_{name}.pkl')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'models/{ticker}_{name}_scaler.pkl')
        
        # Save feature importance
        if self.feature_importance:
            joblib.dump(self.feature_importance, f'models/{ticker}_feature_importance.pkl')
        
        print(f"✅ All models saved for {ticker}")
    
    def load_models(self, ticker):
        """Load all trained models."""
        from tensorflow.keras.models import load_model
        
        for name in ['advanced_lstm', 'conv_lstm']:
            model_path = f'models/{ticker}_{name}.h5'
            if os.path.exists(model_path):
                self.models[name] = load_model(model_path)
        
        for name in ['random_forest', 'gradient_boost', 'svr']:
            model_path = f'models/{ticker}_{name}.pkl'
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
        
        # Load scalers
        for name in ['robust']:
            scaler_path = f'models/{ticker}_{name}_scaler.pkl'
            if os.path.exists(scaler_path):
                self.scalers[name] = joblib.load(scaler_path)
        
        print(f"✅ All models loaded for {ticker}")
