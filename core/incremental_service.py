#!/usr/bin/env python3
"""
Incremental Learning Service for AI Stock Predictor

This core service provides incremental learning capabilities including:
- Model update pipeline
- Model versioning
- Continuous learning
- Performance tracking
- Rollback capabilities
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a version of a trained model"""
    
    def __init__(self, version_id: str, model_path: str, metadata: Dict[str, Any]):
        self.version_id = version_id
        self.model_path = model_path
        self.metadata = metadata
        self.created_at = metadata.get('created_at', datetime.now().isoformat())
        self.performance_metrics = metadata.get('performance_metrics', {})
        self.feature_columns = metadata.get('feature_columns', [])
        self.training_samples = metadata.get('training_samples', 0)
        self.validation_samples = metadata.get('validation_samples', 0)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary"""
        return {
            'version_id': self.version_id,
            'model_path': self.model_path,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'performance_metrics': self.performance_metrics,
            'feature_columns': self.feature_columns,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples
        }


class IncrementalService:
    """Core service for incremental learning management"""
    
    def __init__(self, base_path: str = "models/incremental"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.versions_path = self.base_path / "versions"
        self.versions_path.mkdir(exist_ok=True)
        
        self.backup_path = self.base_path / "backups"
        self.backup_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
        
        self.performance_path = self.base_path / "performance"
        self.performance_path.mkdir(exist_ok=True)
        
        # Load version registry
        self.version_registry = self._load_version_registry()
        
        # Performance threshold for model updates
        self.performance_threshold = 0.05  # 5% performance degradation threshold
        
    def _load_version_registry(self) -> Dict[str, List[ModelVersion]]:
        """Load version registry from disk"""
        registry_file = self.metadata_path / "version_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                    registry = {}
                    for ticker, versions in data.items():
                        registry[ticker] = [ModelVersion(**v) for v in versions]
                    return registry
            except Exception as e:
                logger.warning(f"Failed to load version registry: {e}")
        return {}
    
    def _save_version_registry(self):
        """Save version registry to disk"""
        registry_file = self.metadata_path / "version_registry.json"
        try:
            data = {}
            for ticker, versions in self.version_registry.items():
                data[ticker] = [v.to_dict() for v in versions]
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version registry: {e}")
    
    def create_version_id(self, ticker: str, mode: str) -> str:
        """Create a unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ticker}_{mode}_{timestamp}"
    
    def get_latest_version(self, ticker: str, mode: str) -> Optional[ModelVersion]:
        """Get the latest version for a ticker and mode"""
        if ticker not in self.version_registry:
            return None
        
        versions = [v for v in self.version_registry[ticker] 
                   if v.metadata.get('mode') == mode]
        
        if not versions:
            return None
        
        # Sort by creation date and return latest
        versions.sort(key=lambda x: x.created_at, reverse=True)
        return versions[0]
    
    def get_all_versions(self, ticker: str, mode: str = None) -> List[ModelVersion]:
        """Get all versions for a ticker and optionally filter by mode"""
        if ticker not in self.version_registry:
            return []
        
        versions = self.version_registry[ticker]
        if mode:
            versions = [v for v in versions if v.metadata.get('mode') == mode]
        
        # Sort by creation date
        versions.sort(key=lambda x: x.created_at, reverse=True)
        return versions
    
    def register_version(self, ticker: str, version: ModelVersion):
        """Register a new model version"""
        if ticker not in self.version_registry:
            self.version_registry[ticker] = []
        
        self.version_registry[ticker].append(version)
        self._save_version_registry()
        logger.info(f"Registered version {version.version_id} for {ticker}")
    
    def backup_current_model(self, ticker: str, mode: str) -> Optional[str]:
        """Backup the current model before updating"""
        current_model_path = f"models/{ticker}_{mode}.h5"
        current_scaler_path = f"models/{ticker}_{mode}_scaler.pkl"
        
        if not os.path.exists(current_model_path):
            logger.warning(f"No current model found at {current_model_path}")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_path / f"{ticker}_{mode}_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        try:
            # Copy model files
            import shutil
            shutil.copy2(current_model_path, backup_dir / f"{ticker}_{mode}.h5")
            if os.path.exists(current_scaler_path):
                shutil.copy2(current_scaler_path, backup_dir / f"{ticker}_{mode}_scaler.pkl")
            
            # Save backup metadata
            backup_metadata = {
                'ticker': ticker,
                'mode': mode,
                'backup_time': timestamp,
                'original_model_path': current_model_path,
                'original_scaler_path': current_scaler_path
            }
            
            with open(backup_dir / "backup_metadata.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            logger.info(f"Backed up current model to {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"Failed to backup model: {e}")
            return None
    
    def rollback_to_version(self, ticker: str, version_id: str) -> bool:
        """Rollback to a specific version"""
        if ticker not in self.version_registry:
            logger.error(f"No versions found for {ticker}")
            return False
        
        target_version = None
        for version in self.version_registry[ticker]:
            if version.version_id == version_id:
                target_version = version
                break
        
        if not target_version:
            logger.error(f"Version {version_id} not found for {ticker}")
            return False
        
        try:
            # Backup current model
            mode = target_version.metadata.get('mode', 'simple')
            self.backup_current_model(ticker, mode)
            
            # Copy version files to main model location
            import shutil
            current_model_path = f"models/{ticker}_{mode}.h5"
            current_scaler_path = f"models/{ticker}_{mode}_scaler.pkl"
            
            shutil.copy2(target_version.model_path, current_model_path)
            
            scaler_path = target_version.model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                shutil.copy2(scaler_path, current_scaler_path)
            
            logger.info(f"Successfully rolled back to version {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback to version {version_id}: {e}")
            return False
    
    def prepare_incremental_data(self, new_data: pd.DataFrame, 
                                feature_columns: List[str],
                                target_column: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for incremental training"""
        # Ensure we have the required columns
        required_columns = feature_columns + [target_column]
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing columns in new data: {missing_columns}")
        
        # Prepare features and target
        X = new_data[feature_columns].values
        y = new_data[target_column].values
        
        return X, y
    
    def evaluate_model_performance(self, model, X_test: np.ndarray, 
                                 y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        }
    
    def should_update_model(self, current_performance: Dict[str, float],
                          new_performance: Dict[str, float]) -> bool:
        """Determine if model should be updated based on performance"""
        # Compare RMSE (lower is better)
        current_rmse = current_performance.get('rmse', float('inf'))
        new_rmse = new_performance.get('rmse', float('inf'))
        
        # Calculate improvement
        improvement = (current_rmse - new_rmse) / current_rmse
        
        return improvement > self.performance_threshold
    
    def update_model_incrementally(self, ticker: str, mode: str, 
                                 new_data: pd.DataFrame,
                                 feature_columns: List[str],
                                 target_column: str = 'target',
                                 validation_split: float = 0.2,
                                 epochs: int = 50,
                                 batch_size: int = 32) -> Dict[str, Any]:
        """Update model with new data incrementally"""
        
        logger.info(f"Starting incremental training for {ticker} ({mode})")
        
        # Get current model
        current_model_path = f"models/{ticker}_{mode}.h5"
        current_scaler_path = f"models/{ticker}_{mode}_scaler.pkl"
        
        if not os.path.exists(current_model_path):
            raise FileNotFoundError(f"Current model not found: {current_model_path}")
        
        # Load current model and scaler
        current_model = load_model(current_model_path)
        current_scaler = joblib.load(current_scaler_path)
        
        # Prepare new data
        X_new, y_new = self.prepare_incremental_data(new_data, feature_columns, target_column)
        
        # Scale new data
        X_new_scaled = current_scaler.transform(X_new)
        
        # Split new data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_new_scaled, y_new, test_size=validation_split, random_state=42
        )
        
        # Evaluate current model on new data
        current_performance = self.evaluate_model_performance(current_model, X_val, y_val)
        logger.info(f"Current model performance on new data: {current_performance}")
        
        # Create new model for incremental training
        new_model = tf.keras.models.clone_model(current_model)
        new_model.set_weights(current_model.get_weights())
        
        # Compile new model
        new_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train on new data
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f"models/temp_{ticker}_{mode}.h5",
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        history = new_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate new model
        new_performance = self.evaluate_model_performance(new_model, X_val, y_val)
        logger.info(f"New model performance: {new_performance}")
        
        # Check if we should update
        should_update = self.should_update_model(current_performance, new_performance)
        
        if should_update:
            logger.info("Performance improvement detected. Updating model...")
            
            # Backup current model
            backup_path = self.backup_current_model(ticker, mode)
            
            # Create new version
            version_id = self.create_version_id(ticker, mode)
            version_path = self.versions_path / f"{version_id}.h5"
            
            # Save new model
            new_model.save(str(version_path))
            joblib.dump(current_scaler, str(version_path).replace('.h5', '_scaler.pkl'))
            
            # Create version metadata
            metadata = {
                'ticker': ticker,
                'mode': mode,
                'created_at': datetime.now().isoformat(),
                'performance_metrics': new_performance,
                'feature_columns': feature_columns,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'epochs_trained': len(history.history['loss']),
                'backup_path': backup_path,
                'improvement': (current_performance['rmse'] - new_performance['rmse']) / current_performance['rmse']
            }
            
            # Register version
            version = ModelVersion(version_id, str(version_path), metadata)
            self.register_version(ticker, version)
            
            # Update main model files
            new_model.save(current_model_path)
            joblib.dump(current_scaler, current_scaler_path)
            
            # Clean up temp file
            temp_path = f"models/temp_{ticker}_{mode}.h5"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.info(f"Model updated successfully. Version: {version_id}")
            
            return {
                'success': True,
                'version_id': version_id,
                'performance_improvement': metadata['improvement'],
                'new_performance': new_performance,
                'backup_path': backup_path
            }
        
        else:
            logger.info("No significant performance improvement. Keeping current model.")
            
            # Clean up temp file
            temp_path = f"models/temp_{ticker}_{mode}.h5"
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'success': False,
                'reason': 'No significant performance improvement',
                'current_performance': current_performance,
                'new_performance': new_performance
            }
    
    def get_model_history(self, ticker: str, mode: str = None) -> List[Dict[str, Any]]:
        """Get training history for a model"""
        versions = self.get_all_versions(ticker, mode)
        history = []
        
        for version in versions:
            history.append({
                'version_id': version.version_id,
                'created_at': version.created_at,
                'performance_metrics': version.performance_metrics,
                'training_samples': version.training_samples,
                'validation_samples': version.validation_samples,
                'feature_columns': version.feature_columns
            })
        
        return history
    
    def cleanup_old_versions(self, ticker: str, keep_versions: int = 5) -> int:
        """Clean up old versions, keeping only the most recent ones"""
        versions = self.get_all_versions(ticker)
        
        if len(versions) <= keep_versions:
            return 0
        
        # Sort by creation date and keep only the most recent
        versions.sort(key=lambda x: x.created_at, reverse=True)
        versions_to_keep = versions[:keep_versions]
        versions_to_remove = versions[keep_versions:]
        
        removed_count = 0
        for version in versions_to_remove:
            try:
                # Remove version file
                if os.path.exists(version.model_path):
                    os.remove(version.model_path)
                
                # Remove scaler file
                scaler_path = version.model_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    os.remove(scaler_path)
                
                # Remove from registry
                self.version_registry[ticker].remove(version)
                removed_count += 1
                
                logger.info(f"Removed old version: {version.version_id}")
                
            except Exception as e:
                logger.error(f"Failed to remove version {version.version_id}: {e}")
        
        # Save updated registry
        self._save_version_registry()
        
        return removed_count
