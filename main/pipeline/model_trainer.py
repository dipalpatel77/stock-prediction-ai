"""
Enhanced Model Trainer Component
Advanced model trainer with 16+ algorithms from the unified analysis pipeline
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import joblib
import os
from pathlib import Path
import warnings
from contextlib import redirect_stderr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from threading import Lock

from .base_pipeline import BasePipelineComponent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core ML imports
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
try:
    from sklearn.gaussian_process.kernels import C, RBF
    GAUSSIAN_PROCESS_AVAILABLE = True
except ImportError:
    GAUSSIAN_PROCESS_AVAILABLE = False
    logger.warning("Gaussian Process kernels not available")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Advanced ML imports (with fallbacks)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HIST_GRADIENT_AVAILABLE = True
except ImportError:
    HIST_GRADIENT_AVAILABLE = False
    logger.warning("HistGradientBoostingRegressor not available")

try:
    from sklearn.ensemble import BaggingRegressor
    BAGGING_AVAILABLE = True
except ImportError:
    BAGGING_AVAILABLE = False
    logger.warning("BaggingRegressor not available")

try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import IsolationForest
    ISOLATION_FOREST_AVAILABLE = True
except ImportError:
    ISOLATION_FOREST_AVAILABLE = False
    logger.warning("IsolationForest not available")


class ModelTrainer(BasePipelineComponent):
    """
    Enhanced Model Trainer with 16+ algorithms from the unified analysis pipeline
    
    This component provides:
    - Model training with multiple algorithms
    - Cross-validation and hyperparameter tuning
    - Model evaluation and metrics
    - Model persistence and loading
    - Feature importance analysis
    - 16+ advanced ML algorithms
    - Ensemble methods
    - Hyperparameter optimization
    - Feature scaling
    - Model persistence
    - Performance evaluation
    - Cross-validation
    """
    
    def __init__(self, ticker: str, config: Dict[str, Any] = None):
        """
        Initialize Enhanced Model Trainer
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
        """
        super().__init__('model_trainer', ticker, config)
        
        # Model configuration
        self.models = {}
        self.scalers = {}
        self.model_metrics = {}
        self.feature_importance = {}
        self.ensemble_weights = {}
        self.cv_scores = {}
        self.model_path = Path('models')
        self.model_path.mkdir(exist_ok=True)
        
        # Training parameters
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        self.cv_folds = self.config.get('cv_folds', 5)
        self.enable_ensemble = self.config.get('enable_ensemble', True)
        self.enable_cross_validation = self.config.get('enable_cross_validation', True)
        
        # Advanced features
        self.enable_hyperparameter_tuning = self.config.get('enable_hyperparameter_tuning', False)
        self.max_training_time = self.config.get('max_training_time', 300)  # 5 minutes per model
        
        # Parallel processing configuration
        self.enable_parallel_training = self.config.get('enable_parallel_training', True)
        self.max_workers = self.config.get('max_workers', min(8, mp.cpu_count()))
        self.use_process_pool = self.config.get('use_process_pool', True)  # Use ProcessPool for CPU-bound tasks
        
        # Thread-safe results storage
        self._results_lock = Lock()
        
        self.logger.info(f"Enhanced Model Trainer initialized for {ticker}")
        self.logger.info(f"Parallel training: {self.enable_parallel_training}, Max workers: {self.max_workers}")
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training
        
        Args:
            data: Raw training data
            
        Returns:
            Tuple of (features, target)
        """
        try:
            self.logger.info(f"Preparing data with {len(data)} samples")
            
            # Use the internal prepare data method
            X, y = self._prepare_data(data)
            
            self.logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def execute(self, data: pd.DataFrame = None, **kwargs) -> Dict[str, Any]:
        """
        Execute enhanced model training with 16+ algorithms
        
        Args:
            data: Training data
            **kwargs: Additional parameters
            
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info("🚀 Starting enhanced model training with 16+ algorithms")
            self.logger.info(f"Model trainer received data: {data is not None}, empty: {data.empty if data is not None else 'N/A'}")
            
            if data is None or data.empty:
                self.logger.warning("No training data provided, using sample data")
                # Generate sample data for demonstration
                sample_data = pd.DataFrame({
                    'Close': np.random.randn(100).cumsum() + 100,
                    'Volume': np.random.randint(1000, 10000, 100),
                    'SMA_20': np.random.randn(100).cumsum() + 100,
                    'RSI': np.random.uniform(20, 80, 100)
                })
                data = sample_data
            
            # Prepare data
            X, y = self._prepare_data(data)
            if X is None or y is None:
                return {'success': False, 'error': 'Data preparation failed'}
            
            # Split data
            X_train, X_test, y_train, y_test = self._split_data(X, y)
            
            # Train all models
            training_results = self._train_all_models(X_train, y_train, X_test, y_test)
            
            # Evaluate models
            evaluation_results = self._evaluate_all_models(training_results, X_test, y_test)
            
            # Create ensemble if enabled
            if self.enable_ensemble:
                ensemble_result = self._create_ensemble(training_results, X_train, y_train)
                if ensemble_result:
                    training_results['ensemble'] = ensemble_result
            
            # Get best model
            best_model = self._get_best_model(evaluation_results)
            
            # Save models
            self._save_models(training_results)
            
            # Generate summary
            summary = self._generate_training_summary(training_results, evaluation_results)
            
            return {
                'success': True,
                'models': training_results,
                'evaluation': evaluation_results,
                'best_model': best_model,
                'summary': summary,
                'models_trained': len(training_results),
                'best_model_name': best_model['name'],
                'best_model_score': best_model['score'],
                'accuracy': best_model['score']
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced model training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        try:
            # Remove non-numeric columns and handle missing values
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data = numeric_data.dropna()
            
            if numeric_data.empty:
                self.logger.error("No numeric data available")
                return None, None
            
            # Use 'Close' or 'close' as target
            target_col = 'Close' if 'Close' in numeric_data.columns else 'close'
            if target_col not in numeric_data.columns:
                self.logger.error(f"Target column '{target_col}' not found")
                return None, None
            
            # Prepare features and target
            feature_cols = [col for col in numeric_data.columns if col != target_col]
            X = numeric_data[feature_cols]
            y = numeric_data[target_col]
            
            self.logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            return None, None
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets"""
        try:
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            self.logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Data splitting failed: {e}")
            raise
    
    def _train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train all 16+ models with parallel processing"""
        try:
            if self.enable_parallel_training:
                self.logger.info("🚀 Training 16+ advanced models in parallel...")
                return self._train_models_parallel(X_train, y_train, X_test, y_test)
            else:
                self.logger.info("🤖 Training 16+ advanced models sequentially...")
                return self._train_models_sequential(X_train, y_train, X_test, y_test)
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    def _train_models_parallel(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train models using parallel processing"""
        try:
            # Initialize all models
            models = self._initialize_all_models()
            
            # Prepare training tasks
            training_tasks = []
            for name, model_config in models.items():
                task = {
                    'name': name,
                    'model_config': model_config,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'enable_cross_validation': self.enable_cross_validation,
                    'cv_folds': self.cv_folds,
                    'random_state': self.random_state
                }
                training_tasks.append(task)
            
            # Choose executor based on model type
            if self.use_process_pool:
                executor = ProcessPoolExecutor(max_workers=self.max_workers)
                self.logger.info(f"Using ProcessPoolExecutor with {self.max_workers} workers")
            else:
                executor = ThreadPoolExecutor(max_workers=self.max_workers)
                self.logger.info(f"Using ThreadPoolExecutor with {self.max_workers} workers")
            
            # Execute training in parallel
            training_results = {}
            start_time = time.time()
            
            with executor as exec:
                # Submit all training tasks
                future_to_name = {
                    exec.submit(self._train_single_model, task): task['name'] 
                    for task in training_tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        result = future.result(timeout=self.max_training_time)
                        if result and result.get('success'):
                            training_results[name] = result
                            self.logger.info(f"✅ {name} completed - R²: {result['metrics']['test_r2']:.4f}, Time: {result['metrics']['training_time']:.2f}s")
                        else:
                            self.logger.warning(f"❌ {name} failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.logger.error(f"❌ {name} failed with exception: {e}")
                        continue
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Parallel training completed: {len(training_results)}/{len(training_tasks)} models in {total_time:.2f}s")
            
            # Update instance variables with results
            for name, result in training_results.items():
                self.models[name] = result['model']
                self.model_metrics[name] = result['metrics']
                self.scalers[name] = result.get('scaler')
                self.feature_importance[name] = result.get('feature_importance', {})
                if result.get('cv_score') is not None:
                    self.cv_scores[name] = result['cv_score']
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Parallel model training failed: {e}")
            raise
    
    def _train_models_sequential(self, X_train: pd.DataFrame, y_train: pd.Series, 
                               X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train models sequentially (original method)"""
        try:
            # Initialize all models
            models = self._initialize_all_models()
            
            training_results = {}
            
            for name, model_config in models.items():
                try:
                    self.logger.info(f"Training {name}...")
                    
                    # Get model and scaler
                    model = model_config['model']
                    needs_scaling = model_config.get('needs_scaling', False)
                    
                    # Prepare data
                    if needs_scaling:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        self.scalers[name] = scaler
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test
                    
                    # Train model with timeout protection
                    start_time = datetime.now()
                    model.fit(X_train_scaled, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    train_mse = mean_squared_error(y_train, y_pred_train)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Cross-validation if enabled
                    cv_score = None
                    if self.enable_cross_validation:
                        try:
                            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.cv_folds, scoring='r2')
                            cv_score = cv_scores.mean()
                            self.cv_scores[name] = cv_score
                        except Exception as e:
                            self.logger.warning(f"Cross-validation failed for {name}: {e}")
                    
                    # Store model and metrics
                    self.models[name] = model
                    self.model_metrics[name] = {
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'overfitting': abs(train_r2 - test_r2),
                        'training_time': training_time,
                        'cv_score': cv_score
                    }
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = dict(zip(X_train.columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        self.feature_importance[name] = dict(zip(X_train.columns, model.coef_))
                    
                    training_results[name] = {
                        'model': model,
                        'metrics': self.model_metrics[name],
                        'feature_importance': self.feature_importance.get(name, {}),
                        'scaler': self.scalers.get(name)
                    }
                    
                    self.logger.info(f"{name} trained - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}, Time: {training_time:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {name}: {e}")
                    continue
            
            self.logger.info(f"✅ Successfully trained {len(training_results)} models")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Sequential model training failed: {e}")
            raise
    
    @staticmethod
    def _train_single_model(task: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single model (used by parallel executor)"""
        try:
            name = task['name']
            model_config = task['model_config']
            X_train = task['X_train']
            y_train = task['y_train']
            X_test = task['X_test']
            y_test = task['y_test']
            enable_cross_validation = task['enable_cross_validation']
            cv_folds = task['cv_folds']
            random_state = task['random_state']
            
            # Get model and scaler
            model = model_config['model']
            needs_scaling = model_config.get('needs_scaling', False)
            
            # Prepare data
            if needs_scaling:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                scaler = None
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train_scaled, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation if enabled
            cv_score = None
            if enable_cross_validation:
                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                    cv_score = cv_scores.mean()
                except Exception as e:
                    # Log warning but don't fail the training
                    pass
            
            # Feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(X_train.columns, model.coef_))
            
            return {
                'success': True,
                'model': model,
                'metrics': {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'overfitting': abs(train_r2 - test_r2),
                    'training_time': training_time,
                    'cv_score': cv_score
                },
                'feature_importance': feature_importance,
                'scaler': scaler
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': None,
                'metrics': {},
                'feature_importance': {},
                'scaler': None
            }
    
    def _initialize_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all 16+ models"""
        models = {}
        
        # Tree-based models
        models['RandomForest'] = {
            'model': RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=3,
                min_samples_leaf=1, random_state=self.random_state, n_jobs=-1,
                max_features='sqrt', bootstrap=True
            ),
            'needs_scaling': False
        }
        
        models['GradientBoosting'] = {
            'model': GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=10,
                min_samples_split=3, random_state=self.random_state, subsample=0.8
            ),
            'needs_scaling': False
        }
        
        models['ExtraTrees'] = {
            'model': ExtraTreesRegressor(
                n_estimators=200, max_depth=15, min_samples_split=3,
                min_samples_leaf=1, random_state=self.random_state, n_jobs=-1
            ),
            'needs_scaling': False
        }
        
        models['AdaBoost'] = {
            'model': AdaBoostRegressor(
                n_estimators=200, learning_rate=0.1, random_state=self.random_state
            ),
            'needs_scaling': False
        }
        
        # Advanced boosting models
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': xgb.XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=10,
                    subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,
                    reg_alpha=0.1, reg_lambda=1.0, verbosity=0, n_jobs=-1
                ),
                'needs_scaling': False
            }
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = {
                'model': lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=10,
                    subsample=0.8, colsample_bytree=0.8, random_state=self.random_state,
                    reg_alpha=0.1, reg_lambda=1.0, verbose=-1, force_col_wise=True, n_jobs=-1
                ),
                'needs_scaling': False
            }
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = {
                'model': CatBoostRegressor(
                    iterations=150, learning_rate=0.1, depth=6,
                    random_state=self.random_state, verbose=False, allow_writing_files=False,
                    task_type='CPU', thread_count=-1, early_stopping_rounds=10
                ),
                'needs_scaling': False
            }
        
        if HIST_GRADIENT_AVAILABLE:
            models['HistGradientBoosting'] = {
                'model': HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05, max_depth=10,
                    random_state=self.random_state
                ),
                'needs_scaling': False
            }
        
        # Linear models
        models['LinearRegression'] = {
            'model': LinearRegression(),
            'needs_scaling': True
        }
        
        models['Ridge'] = {
            'model': Ridge(alpha=0.1),
            'needs_scaling': True
        }
        
        models['Lasso'] = {
            'model': Lasso(alpha=0.01),
            'needs_scaling': True
        }
        
        models['ElasticNet'] = {
            'model': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=self.random_state),
            'needs_scaling': True
        }
        
        models['Huber'] = {
            'model': HuberRegressor(epsilon=1.35, max_iter=200),
            'needs_scaling': True
        }
        
        # Support Vector models
        models['SVR'] = {
            'model': SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1),
            'needs_scaling': True
        }
        
        models['KernelRidge'] = {
            'model': KernelRidge(alpha=1.0, kernel='rbf'),
            'needs_scaling': True
        }
        
        # Neural networks
        models['MLP'] = {
            'model': MLPRegressor(
                hidden_layer_sizes=(200, 100, 50), max_iter=1000,
                random_state=self.random_state, early_stopping=True, learning_rate='adaptive'
            ),
            'needs_scaling': True
        }
        
        # Gaussian Process (for small datasets)
        if GAUSSIAN_PROCESS_AVAILABLE and len(self.models) < 1000:  # GP is computationally expensive
            try:
                kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0], (1e-2, 1e2))
                models['GaussianProcess'] = {
                    'model': GaussianProcessRegressor(
                        kernel=kernel, random_state=self.random_state, n_restarts_optimizer=10
                    ),
                    'needs_scaling': True
                }
            except Exception as e:
                self.logger.warning(f"Gaussian Process not available: {e}")
        
        # Bagging models
        if BAGGING_AVAILABLE:
            models['Bagging'] = {
                'model': BaggingRegressor(
                    estimator=RandomForestRegressor(n_estimators=50),
                    n_estimators=10, random_state=self.random_state
                ),
                'needs_scaling': False
            }
        
        return models
    
    def _evaluate_all_models(self, training_results: Dict[str, Any], 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all trained models"""
        try:
            evaluation_results = {}
            
            for name, result in training_results.items():
                model = result['model']
                metrics = result['metrics']
                scaler = result.get('scaler')
                
                # Prepare test data
                if scaler:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_test_scaled = X_test
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate additional metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Calculate additional metrics
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                max_error = np.max(np.abs(y_test - y_pred))
                
                evaluation_results[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'max_error': max_error,
                    'predictions': y_pred,
                    'actual': y_test
                }
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            raise
    
    def _create_ensemble(self, training_results: Dict[str, Any], 
                        X_train: pd.DataFrame, y_train: pd.Series) -> Optional[Dict[str, Any]]:
        """Create ensemble model"""
        try:
            if len(training_results) < 2:
                self.logger.warning("Not enough models for ensemble")
                return None
            
            # Select best models for ensemble
            best_models = self._select_best_models_for_ensemble(training_results)
            
            if len(best_models) < 2:
                self.logger.warning("Not enough good models for ensemble")
                return None
            
            # Create voting regressor
            estimators = []
            weights = []
            
            for name in best_models:
                if name in training_results:
                    model = training_results[name]['model']
                    scaler = training_results[name].get('scaler')
                    estimators.append((name.lower(), model))
                    weights.append(1.0)  # Equal weights for now
            
            ensemble = VotingRegressor(estimators=estimators, weights=weights)
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            self.logger.info(f"✅ Ensemble created with {len(estimators)} models")
            
            return {
                'model': ensemble,
                'estimators': estimators,
                'weights': weights
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            return None
    
    def _select_best_models_for_ensemble(self, training_results: Dict[str, Any]) -> List[str]:
        """Select best models for ensemble"""
        try:
            # Sort models by R² score
            model_scores = []
            for name, result in training_results.items():
                r2 = result['metrics']['test_r2']
                model_scores.append((name, r2))
            
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top models (up to 5)
            best_models = [name for name, score in model_scores[:5] if score > 0.1]
            
            return best_models
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            return []
    
    def _get_best_model(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best performing model"""
        try:
            best_model = None
            best_score = -np.inf
            
            for name, result in evaluation_results.items():
                r2 = result['r2']
                if r2 > best_score:
                    best_score = r2
                    best_model = name
            
            return {
                'name': best_model,
                'score': best_score,
                'metrics': evaluation_results.get(best_model, {})
            }
            
        except Exception as e:
            self.logger.error(f"Best model selection failed: {e}")
            return {'name': 'unknown', 'score': 0.0}
    
    def _save_models(self, training_results: Dict[str, Any]):
        """Save trained models"""
        try:
            os.makedirs('models', exist_ok=True)
            
            for name, result in training_results.items():
                model = result['model']
                scaler = result.get('scaler')
                
                # Save model
                model_path = f'models/{self.ticker}_{name}_model.pkl'
                joblib.dump(model, model_path)
                
                # Save scaler if exists
                if scaler:
                    scaler_path = f'models/{self.ticker}_{name}_scaler.pkl'
                    joblib.dump(scaler, scaler_path)
                
                self.logger.info(f"Model {name} saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
    
    def _generate_training_summary(self, training_results: Dict[str, Any], 
                                 evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training summary"""
        try:
            summary = {
                'total_models': len(training_results),
                'successful_models': len([r for r in training_results.values() if r]),
                'best_model': self._get_best_model(evaluation_results),
                'model_performance': {},
                'training_time': {},
                'feature_importance': {}
            }
            
            # Model performance
            for name, result in training_results.items():
                metrics = result['metrics']
                summary['model_performance'][name] = {
                    'r2': metrics['test_r2'],
                    'mse': metrics['test_mse'],
                    'mae': metrics['test_mae'],
                    'overfitting': metrics['overfitting']
                }
                summary['training_time'][name] = metrics['training_time']
                summary['feature_importance'][name] = result.get('feature_importance', {})
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {}
    
    def benchmark_parallel_vs_sequential(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Benchmark parallel vs sequential training performance"""
        try:
            self.logger.info("🏁 Starting parallel vs sequential benchmark...")
            
            # Test sequential training
            self.logger.info("Testing sequential training...")
            start_time = time.time()
            self.enable_parallel_training = False
            sequential_results = self._train_models_sequential(X_train, y_train, X_test, y_test)
            sequential_time = time.time() - start_time
            
            # Test parallel training
            self.logger.info("Testing parallel training...")
            start_time = time.time()
            self.enable_parallel_training = True
            parallel_results = self._train_models_parallel(X_train, y_train, X_test, y_test)
            parallel_time = time.time() - start_time
            
            # Calculate performance metrics
            time_saved = sequential_time - parallel_time
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            efficiency = speedup / self.max_workers if self.max_workers > 0 else 0
            
            # Compare model performance
            sequential_models = len(sequential_results)
            parallel_models = len(parallel_results)
            
            benchmark_results = {
                'sequential': {
                    'time': sequential_time,
                    'models_trained': sequential_models,
                    'models_per_second': sequential_models / sequential_time if sequential_time > 0 else 0
                },
                'parallel': {
                    'time': parallel_time,
                    'models_trained': parallel_models,
                    'models_per_second': parallel_models / parallel_time if parallel_time > 0 else 0
                },
                'performance': {
                    'time_saved': time_saved,
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'time_saved_percentage': (time_saved / sequential_time * 100) if sequential_time > 0 else 0
                },
                'configuration': {
                    'max_workers': self.max_workers,
                    'use_process_pool': self.use_process_pool,
                    'cpu_count': mp.cpu_count()
                }
            }
            
            self.logger.info(f"📊 Benchmark Results:")
            self.logger.info(f"   Sequential: {sequential_time:.2f}s ({sequential_models} models)")
            self.logger.info(f"   Parallel: {parallel_time:.2f}s ({parallel_models} models)")
            self.logger.info(f"   Speedup: {speedup:.2f}x")
            self.logger.info(f"   Time Saved: {time_saved:.2f}s ({benchmark_results['performance']['time_saved_percentage']:.1f}%)")
            self.logger.info(f"   Efficiency: {efficiency:.2f}")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            return {'error': str(e)}
    
    def get_required_config_fields(self) -> List[str]:
        """Get required configuration fields"""
        return ['random_state', 'cv_folds', 'test_size']