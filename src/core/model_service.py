#!/usr/bin/env python3
"""
Model Service
Shared model management and caching service for all analysis tools
"""

import os
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import threading
import time
import json

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Advanced ML imports (optional)
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("⚠️ Advanced models not available. Install xgboost, lightgbm, and catboost for full functionality.")

warnings.filterwarnings('ignore')

class ModelService:
    """Shared model management and caching service."""
    
    def __init__(self, cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache = {}
        self.scaler_cache = {}
        self.cache_lock = threading.Lock()
        
        # Model configurations
        self.model_configs = self._get_model_configs()
        
    def _get_model_configs(self) -> Dict:
        """Get model configurations."""
        configs = {
            'simple': {
                'models': ['random_forest', 'linear_regression'],
                'scalers': ['standard', 'minmax'],
                'ensemble': False
            },
            'advanced': {
                'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost'],
                'scalers': ['standard', 'robust'],
                'ensemble': True
            },
            'full': {
                'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost', 
                          'svr', 'ridge', 'lasso', 'elastic_net', 'mlp', 'gaussian_process'],
                'scalers': ['standard', 'minmax', 'robust'],
                'ensemble': True
            }
        }
        return configs
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        """Create a model instance."""
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )
        
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif model_type == 'linear_regression':
            return LinearRegression()
        
        elif model_type == 'ridge':
            return Ridge(alpha=kwargs.get('alpha', 1.0))
        
        elif model_type == 'lasso':
            return Lasso(alpha=kwargs.get('alpha', 1.0))
        
        elif model_type == 'elastic_net':
            return ElasticNet(alpha=kwargs.get('alpha', 1.0), l1_ratio=kwargs.get('l1_ratio', 0.5))
        
        elif model_type == 'svr':
            return SVR(kernel=kwargs.get('kernel', 'rbf'), C=kwargs.get('C', 1.0))
        
        elif model_type == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                max_iter=kwargs.get('max_iter', 500),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif model_type == 'gaussian_process':
            kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0], (1e-2, 1e2))
            return GaussianProcessRegressor(kernel=kernel, random_state=kwargs.get('random_state', 42))
        
        elif model_type == 'xgboost' and ADVANCED_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )
        
        elif model_type == 'lightgbm' and ADVANCED_AVAILABLE:
            return lgb.LGBMRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3),
                random_state=kwargs.get('random_state', 42),
                n_jobs=kwargs.get('n_jobs', -1)
            )
        
        elif model_type == 'catboost' and ADVANCED_AVAILABLE:
            return CatBoostRegressor(
                iterations=kwargs.get('iterations', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                depth=kwargs.get('depth', 3),
                random_state=kwargs.get('random_state', 42),
                verbose=False
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_scaler(self, scaler_type: str) -> Any:
        """Create a scaler instance."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def train_model(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                   scaler_type: str = 'standard', **kwargs) -> Dict:
        """
        Train a model with caching.
        
        Args:
            model_type: Type of model to train
            X: Feature matrix
            y: Target variable
            scaler_type: Type of scaler to use
            **kwargs: Additional model parameters
            
        Returns:
            Dictionary with model, scaler, and performance metrics
        """
        # Create cache key
        cache_key = f"{model_type}_{scaler_type}_{hash(str(X.shape))}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache first
        if cache_file.exists():
            try:
                with self.cache_lock:
                    if cache_key in self.model_cache:
                        return self.model_cache[cache_key].copy()
                    
                    with open(cache_file, 'rb') as f:
                        cached_result = pickle.load(f)
                    
                    # Validate cached model
                    if self._validate_cached_model(cached_result, X, y):
                        self.model_cache[cache_key] = cached_result
                        print(f"📊 Loaded cached model: {model_type}")
                        return cached_result.copy()
                    else:
                        print(f"⚠️ Cached model {model_type} is invalid, retraining...")
            except Exception as e:
                print(f"⚠️ Cache loading error for {model_type}: {e}")
        
        # Train new model
        print(f"🏋️ Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and fit scaler
        scaler = self.create_scaler(scaler_type)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        model = self.create_model(model_type, **kwargs)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        # Create result dictionary
        result = {
            'model': model,
            'scaler': scaler,
            'model_type': model_type,
            'scaler_type': scaler_type,
            'metrics': metrics,
            'feature_names': list(X.columns),
            'training_date': datetime.now().isoformat(),
            'data_shape': X.shape
        }
        
        # Cache the result
        try:
            with self.cache_lock:
                self.model_cache[cache_key] = result
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            print(f"✅ Model cached: {model_type}")
        except Exception as e:
            print(f"⚠️ Caching error for {model_type}: {e}")
        
        return result
    
    def train_ensemble(self, models: List[str], X: pd.DataFrame, y: pd.Series,
                      scaler_type: str = 'standard', weights: List[float] = None) -> Dict:
        """Train an ensemble model."""
        print(f"🏋️ Training ensemble model with {len(models)} models...")
        
        # Train individual models
        trained_models = []
        for model_type in models:
            try:
                result = self.train_model(model_type, X, y, scaler_type)
                trained_models.append((model_type, result['model'], result['scaler']))
            except Exception as e:
                print(f"⚠️ Failed to train {model_type}: {e}")
        
        if not trained_models:
            raise ValueError("No models were successfully trained")
        
        # Create ensemble
        estimators = [(name, model) for name, model, _ in trained_models]
        ensemble = VotingRegressor(estimators=estimators, weights=weights)
        
        # Use the first model's scaler for consistency
        scaler = trained_models[0][2]
        X_scaled = scaler.transform(X)
        
        # Train ensemble
        ensemble.fit(X_scaled, y)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(X_scaled)
        metrics = self._calculate_ensemble_metrics(y, y_pred)
        
        result = {
            'model': ensemble,
            'scaler': scaler,
            'model_type': 'ensemble',
            'scaler_type': scaler_type,
            'metrics': metrics,
            'feature_names': list(X.columns),
            'training_date': datetime.now().isoformat(),
            'data_shape': X.shape,
            'ensemble_models': [name for name, _, _ in trained_models]
        }
        
        return result
    
    def predict(self, model_result: Dict, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        model = model_result['model']
        scaler = model_result['scaler']
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def predict_with_confidence(self, model_result: Dict, X: pd.DataFrame, 
                              confidence_level: float = 0.95) -> Dict:
        """Make predictions with confidence intervals."""
        predictions = self.predict(model_result, X)
        
        # Calculate confidence intervals (simplified)
        # In practice, you might use bootstrapping or other methods
        std_dev = np.std(predictions)
        z_score = 1.96  # 95% confidence interval
        
        confidence_interval = z_score * std_dev
        
        return {
            'predictions': predictions,
            'confidence_interval': confidence_interval,
            'lower_bound': predictions - confidence_interval,
            'upper_bound': predictions + confidence_interval,
            'confidence_level': confidence_level
        }
    
    def evaluate_model(self, model_result: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate a trained model on new data."""
        predictions = self.predict(model_result, X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        return metrics
    
    def get_feature_importance(self, model_result: Dict) -> pd.DataFrame:
        """Get feature importance from a trained model."""
        model = model_result['model']
        feature_names = model_result['feature_names']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, model_result: Dict, filepath: str):
        """Save a trained model to disk."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_result, f)
            print(f"✅ Model saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self, filepath: str) -> Dict:
        """Load a trained model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_result = pickle.load(f)
            print(f"✅ Model loaded from {filepath}")
            return model_result
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def _calculate_metrics(self, y_train: pd.Series, y_pred_train: np.ndarray,
                          y_test: pd.Series, y_pred_test: np.ndarray) -> Dict:
        """Calculate performance metrics."""
        metrics = {
            'train': {
                'mse': mean_squared_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
        }
        
        return metrics
    
    def _calculate_ensemble_metrics(self, y: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate metrics for ensemble model."""
        return {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
    
    def _validate_cached_model(self, cached_result: Dict, X: pd.DataFrame, y: pd.Series) -> bool:
        """Validate cached model compatibility."""
        try:
            # Check if model exists
            if 'model' not in cached_result or 'scaler' not in cached_result:
                return False
            
            # Check feature compatibility
            if 'feature_names' in cached_result:
                expected_features = cached_result['feature_names']
                if list(X.columns) != expected_features:
                    return False
            
            # Check data shape compatibility
            if 'data_shape' in cached_result:
                expected_shape = cached_result['data_shape']
                if X.shape != expected_shape:
                    return False
            
            # Test prediction
            test_pred = self.predict(cached_result, X.head(1))
            if len(test_pred) != 1:
                return False
            
            return True
            
        except Exception:
            return False
    
    def clear_cache(self, model_type: str = None):
        """Clear model cache."""
        with self.cache_lock:
            if model_type:
                # Clear specific model type cache
                keys_to_remove = [k for k in self.model_cache.keys() if model_type in k]
                for key in keys_to_remove:
                    del self.model_cache[key]
                
                # Remove cache files
                cache_files = list(self.cache_dir.glob(f"*{model_type}*.pkl"))
                for file in cache_files:
                    file.unlink()
                
                print(f"🗑️ Cleared cache for {model_type}")
            else:
                # Clear all cache
                self.model_cache.clear()
                cache_files = list(self.cache_dir.glob("*.pkl"))
                for file in cache_files:
                    file.unlink()
                print("🗑️ Cleared all model cache")
    
    def get_model_info(self, model_result: Dict) -> Dict:
        """Get information about a trained model."""
        info = {
            'model_type': model_result.get('model_type', 'unknown'),
            'scaler_type': model_result.get('scaler_type', 'unknown'),
            'training_date': model_result.get('training_date', 'unknown'),
            'data_shape': model_result.get('data_shape', 'unknown'),
            'feature_count': len(model_result.get('feature_names', [])),
            'metrics': model_result.get('metrics', {})
        }
        
        return info
    
    def compare_models(self, model_results: List[Dict]) -> pd.DataFrame:
        """Compare multiple models."""
        comparison_data = []
        
        for result in model_results:
            info = self.get_model_info(result)
            metrics = info['metrics']
            
            row = {
                'Model': info['model_type'],
                'Scaler': info['scaler_type'],
                'Features': info['feature_count'],
                'Train_RMSE': metrics.get('train', {}).get('rmse', np.nan),
                'Test_RMSE': metrics.get('test', {}).get('rmse', np.nan),
                'Train_R2': metrics.get('train', {}).get('r2', np.nan),
                'Test_R2': metrics.get('test', {}).get('r2', np.nan)
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
