"""
Model Service
Handles model management, versioning, and deployment
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import pickle
import joblib
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Model type enumeration"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    name: str
    model_type: ModelType
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, float]
    features: List[str]
    target: str
    training_data_size: int
    model_size: float
    description: str

@dataclass
class ModelVersion:
    """Model version structure"""
    version: str
    model_metadata: ModelMetadata
    model_path: str
    config_path: str
    performance_report: str
    created_at: datetime
    is_active: bool

class ModelService:
    """Service for model management"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.models_registry = {}
        self.model_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        # Model storage settings
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
        # Model versioning settings
        self.max_versions = self.config.get('max_versions', 10)
        self.auto_cleanup = self.config.get('auto_cleanup', True)
        
        # Performance thresholds
        self.performance_thresholds = self.config.get('performance_thresholds', {
            'r2_score': 0.7,
            'mae': 0.1,
            'rmse': 0.15
        })
        
        self.logger.info("Model Service initialized")

    def register_model(self, model, name: str, model_type: ModelType, 
                      features: List[str], target: str, description: str = "") -> str:
        """Register a new model"""
        try:
            model_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                model_type=model_type,
                version="1.0.0",
                status=ModelStatus.TRAINING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                performance_metrics={},
                features=features,
                target=target,
                training_data_size=0,
                model_size=0.0,
                description=description
            )
            
            # Register model
            self.models_registry[model_id] = metadata
            
            self.logger.info(f"Registered model: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            return None

    def save_model(self, model_id: str, model, performance_metrics: Dict[str, float], 
                   training_data_size: int) -> bool:
        """Save a trained model"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return False
            
            # Update metadata
            metadata = self.models_registry[model_id]
            metadata.status = ModelStatus.TRAINED
            metadata.performance_metrics = performance_metrics
            metadata.training_data_size = training_data_size
            metadata.updated_at = datetime.now()
            
            # Create model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Calculate model size
            model_size = model_path.stat().st_size / (1024 * 1024)  # MB
            metadata.model_size = model_size
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self._metadata_to_dict(metadata), f, indent=2, default=str)
            
            # Save performance report
            performance_path = model_dir / "performance.json"
            with open(performance_path, 'w') as f:
                json.dump(performance_metrics, f, indent=2)
            
            self.logger.info(f"Saved model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_id}: {e}")
            return False

    def load_model(self, model_id: str) -> Optional[Any]:
        """Load a trained model"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return None
            
            # Check cache first
            if model_id in self.model_cache:
                cached_data = self.model_cache[model_id]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['model']
            
            # Load model
            model_dir = self.models_dir / model_id
            model_path = model_dir / "model.pkl"
            
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            
            # Cache model
            self.model_cache[model_id] = {
                'model': model,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Loaded model: {model_id}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            return None

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return None
            
            return self.models_registry[model_id]
            
        except Exception as e:
            self.logger.error(f"Error getting model metadata: {e}")
            return None

    def list_models(self, status: ModelStatus = None, model_type: ModelType = None) -> List[ModelMetadata]:
        """List all models with optional filtering"""
        try:
            models = list(self.models_registry.values())
            
            # Filter by status
            if status:
                models = [m for m in models if m.status == status]
            
            # Filter by model type
            if model_type:
                models = [m for m in models if m.model_type == model_type]
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x.created_at, reverse=True)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []

    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return False
            
            metadata = self.models_registry[model_id]
            metadata.status = status
            metadata.updated_at = datetime.now()
            
            self.logger.info(f"Updated model {model_id} status to {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model status: {e}")
            return False

    def validate_model(self, model_id: str, validation_data: pd.DataFrame, 
                      target_column: str) -> Dict[str, Any]:
        """Validate a model"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return {}
            
            # Load model
            model = self.load_model(model_id)
            if model is None:
                return {}
            
            # Prepare validation data
            X = validation_data.drop(columns=[target_column])
            y = validation_data[target_column]
            
            # Make predictions
            y_pred = model.predict(X)
            
            # Calculate validation metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            validation_metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2
            }
            
            # Check if model meets performance thresholds
            meets_thresholds = self._check_performance_thresholds(validation_metrics)
            
            # Update model status
            if meets_thresholds:
                self.update_model_status(model_id, ModelStatus.VALIDATED)
            else:
                self.update_model_status(model_id, ModelStatus.FAILED)
            
            return {
                'validation_metrics': validation_metrics,
                'meets_thresholds': meets_thresholds,
                'validation_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error validating model {model_id}: {e}")
            return {}

    def _check_performance_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets performance thresholds"""
        try:
            for metric, threshold in self.performance_thresholds.items():
                if metric in metrics:
                    if metric == 'r2_score':
                        if metrics[metric] < threshold:
                            return False
                    else:  # For mae, rmse - lower is better
                        if metrics[metric] > threshold:
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking performance thresholds: {e}")
            return False

    def deploy_model(self, model_id: str) -> bool:
        """Deploy a model"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return False
            
            metadata = self.models_registry[model_id]
            
            # Check if model is validated
            if metadata.status != ModelStatus.VALIDATED:
                self.logger.error(f"Model {model_id} is not validated")
                return False
            
            # Update status to deployed
            self.update_model_status(model_id, ModelStatus.DEPLOYED)
            
            # Create deployment record
            deployment_path = self.models_dir / model_id / "deployment.json"
            deployment_info = {
                'model_id': model_id,
                'deployed_at': datetime.now().isoformat(),
                'status': 'deployed'
            }
            
            with open(deployment_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            self.logger.info(f"Deployed model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying model {model_id}: {e}")
            return False

    def retire_model(self, model_id: str) -> bool:
        """Retire a model"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return False
            
            # Update status to retired
            self.update_model_status(model_id, ModelStatus.RETIRED)
            
            # Create retirement record
            retirement_path = self.models_dir / model_id / "retirement.json"
            retirement_info = {
                'model_id': model_id,
                'retired_at': datetime.now().isoformat(),
                'status': 'retired'
            }
            
            with open(retirement_path, 'w') as f:
                json.dump(retirement_info, f, indent=2)
            
            self.logger.info(f"Retired model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error retiring model {model_id}: {e}")
            return False

    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if model_id not in self.models_registry:
                self.logger.error(f"Model {model_id} not found in registry")
                return {}
            
            metadata = self.models_registry[model_id]
            
            # Load performance report
            performance_path = self.models_dir / model_id / "performance.json"
            if performance_path.exists():
                with open(performance_path, 'r') as f:
                    performance_metrics = json.load(f)
            else:
                performance_metrics = metadata.performance_metrics
            
            return {
                'model_id': model_id,
                'performance_metrics': performance_metrics,
                'status': metadata.status.value,
                'created_at': metadata.created_at,
                'updated_at': metadata.updated_at
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return {}

    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        try:
            if len(model_ids) < 2:
                return {'error': 'Need at least 2 models to compare'}
            
            comparison_data = {}
            
            for model_id in model_ids:
                if model_id in self.models_registry:
                    metadata = self.models_registry[model_id]
                    performance = self.get_model_performance(model_id)
                    
                    comparison_data[model_id] = {
                        'name': metadata.name,
                        'model_type': metadata.model_type.value,
                        'status': metadata.status.value,
                        'performance_metrics': performance.get('performance_metrics', {}),
                        'created_at': metadata.created_at,
                        'model_size': metadata.model_size
                    }
            
            # Find best performing model
            best_model = self._find_best_model(comparison_data)
            
            return {
                'models': comparison_data,
                'best_model': best_model,
                'comparison_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            return {}

    def _find_best_model(self, comparison_data: Dict[str, Any]) -> str:
        """Find the best performing model"""
        try:
            best_model = None
            best_score = -float('inf')
            
            for model_id, data in comparison_data.items():
                performance_metrics = data.get('performance_metrics', {})
                
                # Use R² score as primary metric
                r2_score = performance_metrics.get('r2_score', 0)
                
                if r2_score > best_score:
                    best_score = r2_score
                    best_model = model_id
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error finding best model: {e}")
            return None

    def cleanup_old_models(self) -> int:
        """Clean up old model versions"""
        try:
            if not self.auto_cleanup:
                return 0
            
            cleaned_count = 0
            
            # Group models by name
            model_groups = {}
            for model_id, metadata in self.models_registry.items():
                name = metadata.name
                if name not in model_groups:
                    model_groups[name] = []
                model_groups[name].append((model_id, metadata))
            
            # Clean up each group
            for name, models in model_groups.items():
                if len(models) > self.max_versions:
                    # Sort by creation date (oldest first)
                    models.sort(key=lambda x: x[1].created_at)
                    
                    # Remove oldest models
                    models_to_remove = models[:-self.max_versions]
                    
                    for model_id, metadata in models_to_remove:
                        if metadata.status == ModelStatus.RETIRED:
                            # Remove model files
                            model_dir = self.models_dir / model_id
                            if model_dir.exists():
                                import shutil
                                shutil.rmtree(model_dir)
                            
                            # Remove from registry
                            del self.models_registry[model_id]
                            cleaned_count += 1
            
            self.logger.info(f"Cleaned up {cleaned_count} old models")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old models: {e}")
            return 0

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        try:
            # Get model statistics
            total_models = len(self.models_registry)
            
            # Count by status
            status_counts = {}
            for metadata in self.models_registry.values():
                status = metadata.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by type
            type_counts = {}
            for metadata in self.models_registry.values():
                model_type = metadata.model_type.value
                type_counts[model_type] = type_counts.get(model_type, 0) + 1
            
            # Get recent models
            recent_models = sorted(self.models_registry.values(), 
                                key=lambda x: x.created_at, reverse=True)[:5]
            
            # Calculate average performance
            avg_performance = self._calculate_average_performance()
            
            return {
                'total_models': total_models,
                'status_counts': status_counts,
                'type_counts': type_counts,
                'recent_models': [self._metadata_to_dict(m) for m in recent_models],
                'avg_performance': avg_performance,
                'summary_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model summary: {e}")
            return {}

    def _calculate_average_performance(self) -> Dict[str, float]:
        """Calculate average performance across all models"""
        try:
            performance_metrics = {}
            model_count = 0
            
            for metadata in self.models_registry.values():
                if metadata.performance_metrics:
                    for metric, value in metadata.performance_metrics.items():
                        if metric not in performance_metrics:
                            performance_metrics[metric] = []
                        performance_metrics[metric].append(value)
                    model_count += 1
            
            if model_count == 0:
                return {}
            
            avg_performance = {}
            for metric, values in performance_metrics.items():
                avg_performance[metric] = np.mean(values)
            
            return avg_performance
            
        except Exception as e:
            self.logger.error(f"Error calculating average performance: {e}")
            return {}

    def _metadata_to_dict(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        try:
            return {
                'model_id': metadata.model_id,
                'name': metadata.name,
                'model_type': metadata.model_type.value,
                'version': metadata.version,
                'status': metadata.status.value,
                'created_at': metadata.created_at.isoformat(),
                'updated_at': metadata.updated_at.isoformat(),
                'performance_metrics': metadata.performance_metrics,
                'features': metadata.features,
                'target': metadata.target,
                'training_data_size': metadata.training_data_size,
                'model_size': metadata.model_size,
                'description': metadata.description
            }
            
        except Exception as e:
            self.logger.error(f"Error converting metadata to dict: {e}")
            return {}
