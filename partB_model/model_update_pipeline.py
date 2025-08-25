#!/usr/bin/env python3
"""
Model Update Pipeline for AI Stock Predictor

This module provides a comprehensive pipeline for updating models with new data,
integrating incremental learning with the existing model training system.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from partB_model.incremental_learning import (
    IncrementalLearningManager, 
    IncrementalTrainingPipeline,
    ContinuousLearningScheduler
)
from partA_preprocessing.data_loader import load_data
from partA_preprocessing.preprocess import clean_data, add_technical_indicators
from partB_model.enhanced_model_builder import EnhancedModelBuilder
from partB_model.enhanced_training import EnhancedStockPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelUpdatePipeline:
    """Comprehensive pipeline for model updates and incremental learning"""
    
    def __init__(self, base_path: str = "models/incremental"):
        self.learning_manager = IncrementalLearningManager(base_path)
        self.training_pipeline = IncrementalTrainingPipeline(self.learning_manager)
        self.scheduler = ContinuousLearningScheduler(self.learning_manager)
        
        # Initialize components
        # Note: Using functions instead of classes for data loading and preprocessing
        self.model_builder = EnhancedModelBuilder()
        self.trainer = None  # Will be initialized per ticker
        
        # Configuration
        self.update_config = {
            'min_data_points': 100,  # Minimum new data points required for update
            'performance_threshold': 0.05,  # 5% improvement threshold
            'max_versions_kept': 10,  # Maximum versions to keep per model
            'backup_before_update': True,
            'validate_after_update': True
        }
    
    def check_for_updates(self, ticker: str, mode: str = "simple") -> Dict[str, Any]:
        """Check if a model needs updating based on data freshness and performance"""
        
        logger.info(f"Checking for updates for {ticker} ({mode})")
        
        # Check if model exists
        model_path = f"models/{ticker}_{mode}.h5"
        if not os.path.exists(model_path):
            logger.info(f"No existing model found for {ticker} ({mode})")
            return {
                'needs_update': True,
                'reason': 'No existing model found',
                'update_type': 'full_training'
            }
        
        # Get latest version info
        latest_version = self.learning_manager.get_latest_version(ticker, mode)
        
        # Check data freshness
        data_freshness = self._check_data_freshness(ticker)
        
        # Check if scheduled update is due
        due_updates = self.scheduler.get_due_updates()
        is_scheduled = (ticker, mode) in due_updates
        
        # Determine update type
        if data_freshness['days_since_last_update'] > 7:
            update_type = 'incremental_training'
        elif is_scheduled:
            update_type = 'scheduled_update'
        else:
            update_type = 'none'
        
        return {
            'needs_update': update_type != 'none',
            'reason': f"Data freshness: {data_freshness['days_since_last_update']} days, Scheduled: {is_scheduled}",
            'update_type': update_type,
            'data_freshness': data_freshness,
            'latest_version': latest_version.version_id if latest_version else None
        }
    
    def _check_data_freshness(self, ticker: str) -> Dict[str, Any]:
        """Check how fresh the training data is"""
        try:
            # Get latest data (simplified for now)
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            latest_data = load_data(ticker, start_date, end_date)
            
            if latest_data is None or latest_data.empty:
                return {
                    'days_since_last_update': float('inf'),
                    'latest_data_date': None,
                    'data_available': False
                }
            
            latest_date = pd.to_datetime(latest_data.index[-1])
            days_since_update = (datetime.now() - latest_date).days
            
            return {
                'days_since_last_update': days_since_update,
                'latest_data_date': latest_date.isoformat(),
                'data_available': True
            }
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return {
                'days_since_last_update': float('inf'),
                'latest_data_date': None,
                'data_available': False,
                'error': str(e)
            }
    
    def prepare_update_data(self, ticker: str, mode: str, 
                          days_back: int = 30) -> Optional[pd.DataFrame]:
        """Prepare new data for model update"""
        
        logger.info(f"Preparing update data for {ticker} ({mode})")
        
        try:
            # Get recent data
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            recent_data = load_data(ticker, start_date, end_date)
            
            if recent_data is None or recent_data.empty:
                logger.warning(f"No recent data available for {ticker}")
                return None
            
            # Preprocess data
            processed_data = clean_data(recent_data)
            processed_data = add_technical_indicators(processed_data)
            
            if processed_data is None or processed_data.empty:
                logger.warning(f"Failed to preprocess data for {ticker}")
                return None
            
            # Check if we have enough data
            if len(processed_data) < self.update_config['min_data_points']:
                logger.warning(f"Insufficient data for update: {len(processed_data)} < {self.update_config['min_data_points']}")
                return None
            
            logger.info(f"Prepared {len(processed_data)} data points for update")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preparing update data: {e}")
            return None
    
    def perform_incremental_update(self, ticker: str, mode: str, 
                                 new_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform incremental update of the model"""
        
        logger.info(f"Performing incremental update for {ticker} ({mode})")
        
        try:
            # Get feature columns from existing model
            feature_columns = self._get_model_feature_columns(ticker, mode)
            
            if not feature_columns:
                logger.error(f"Could not determine feature columns for {ticker} ({mode})")
                return {
                    'success': False,
                    'error': 'Could not determine feature columns'
                }
            
            # Perform incremental training
            result = self.training_pipeline.update_model_incrementally(
                ticker=ticker,
                mode=mode,
                new_data=new_data,
                feature_columns=feature_columns,
                target_column='target'
            )
            
            if result['success']:
                # Mark update as completed in scheduler
                self.scheduler.mark_update_completed(ticker, mode)
                
                # Schedule next update
                self.scheduler.schedule_update(ticker, mode, update_frequency_days=7)
                
                logger.info(f"Incremental update completed successfully for {ticker} ({mode})")
            else:
                logger.info(f"Incremental update skipped for {ticker} ({mode}): {result['reason']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during incremental update: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def perform_full_retraining(self, ticker: str, mode: str) -> Dict[str, Any]:
        """Perform full retraining of the model"""
        
        logger.info(f"Performing full retraining for {ticker} ({mode})")
        
        try:
            # Backup existing model if it exists
            if self.update_config['backup_before_update']:
                self.learning_manager.backup_current_model(ticker, mode)
            
            # Get historical data
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            historical_data = load_data(ticker, start_date, end_date)
            
            if historical_data is None or historical_data.empty:
                return {
                    'success': False,
                    'error': 'No historical data available'
                }
            
            # Preprocess data
            processed_data = clean_data(historical_data)
            processed_data = add_technical_indicators(processed_data)
            
            if processed_data is None or processed_data.empty:
                return {
                    'success': False,
                    'error': 'Failed to preprocess data'
                }
            
            # Build and train model
            trainer = EnhancedStockPredictor(ticker, lookback_days=60)
            training_result = trainer.train_enhanced_model(processed_data)
            
            if training_result is not None:
                # Create version for the new model
                version_id = self.learning_manager.create_version_id(ticker, mode)
                version_path = self.learning_manager.versions_path / f"{version_id}.h5"
                
                # Save model to version location
                model.save(str(version_path))
                
                # Create metadata
                metadata = {
                    'ticker': ticker,
                    'mode': mode,
                    'created_at': datetime.now().isoformat(),
                    'training_type': 'full_retraining',
                    'training_samples': len(processed_data),
                    'feature_columns': list(processed_data.columns[:-1]),  # Exclude target
                    'performance_metrics': training_result.get('metrics', {})
                }
                
                # Register version
                from partB_model.incremental_learning import ModelVersion
                version = ModelVersion(version_id, str(version_path), metadata)
                self.learning_manager.register_version(ticker, version)
                
                # Schedule next update
                self.scheduler.schedule_update(ticker, mode, update_frequency_days=7)
                
                logger.info(f"Full retraining completed successfully for {ticker} ({mode})")
                
                return {
                    'success': True,
                    'version_id': version_id,
                    'training_result': training_result
                }
            else:
                return {
                    'success': False,
                    'error': 'Training failed'
                }
                
        except Exception as e:
            logger.error(f"Error during full retraining: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_model_feature_columns(self, ticker: str, mode: str) -> List[str]:
        """Get feature columns from existing model"""
        try:
            # Try to get from latest version
            latest_version = self.learning_manager.get_latest_version(ticker, mode)
            if latest_version and latest_version.feature_columns:
                return latest_version.feature_columns
            
            # Fallback: try to infer from preprocessed data
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            sample_data = load_data(ticker, start_date, end_date)
            if sample_data is not None:
                processed_data = clean_data(sample_data)
                processed_data = add_technical_indicators(processed_data)
                if processed_data is not None:
                    return list(processed_data.columns[:-1])  # Exclude target column
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting feature columns: {e}")
            return []
    
    def run_automatic_updates(self, tickers: List[str], modes: List[str] = ["simple", "advanced"]) -> Dict[str, Any]:
        """Run automatic updates for multiple tickers"""
        
        logger.info(f"Running automatic updates for {len(tickers)} tickers")
        
        results = {}
        
        for ticker in tickers:
            results[ticker] = {}
            
            for mode in modes:
                try:
                    # Check if update is needed
                    update_check = self.check_for_updates(ticker, mode)
                    
                    if not update_check['needs_update']:
                        results[ticker][mode] = {
                            'status': 'no_update_needed',
                            'reason': update_check['reason']
                        }
                        continue
                    
                    # Prepare update data
                    update_data = self.prepare_update_data(ticker, mode)
                    
                    if update_data is None:
                        results[ticker][mode] = {
                            'status': 'no_data_available',
                            'reason': 'Could not prepare update data'
                        }
                        continue
                    
                    # Perform update based on type
                    if update_check['update_type'] == 'full_training':
                        update_result = self.perform_full_retraining(ticker, mode)
                    else:
                        update_result = self.perform_incremental_update(ticker, mode, update_data)
                    
                    results[ticker][mode] = {
                        'status': 'completed' if update_result['success'] else 'failed',
                        'result': update_result
                    }
                    
                except Exception as e:
                    logger.error(f"Error updating {ticker} ({mode}): {e}")
                    results[ticker][mode] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return results
    
    def get_model_version_history(self, ticker: str, mode: str = None) -> List[Dict[str, Any]]:
        """Get version history for a model"""
        versions = self.learning_manager.get_all_versions(ticker, mode)
        return [v.to_dict() for v in versions]
    
    def rollback_model(self, ticker: str, version_id: str) -> bool:
        """Rollback model to a specific version"""
        return self.learning_manager.rollback_to_version(ticker, version_id)
    
    def cleanup_old_versions(self, ticker: str, mode: str = None, max_versions: int = None):
        """Clean up old model versions"""
        if max_versions is None:
            max_versions = self.update_config['max_versions_kept']
        
        versions = self.learning_manager.get_all_versions(ticker, mode)
        
        if len(versions) <= max_versions:
            return
        
        # Keep only the latest versions
        versions_to_keep = versions[:max_versions]
        versions_to_remove = versions[max_versions:]
        
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
                self.learning_manager.version_registry[ticker].remove(version)
                
                logger.info(f"Removed old version: {version.version_id}")
                
            except Exception as e:
                logger.error(f"Error removing version {version.version_id}: {e}")
        
        # Save updated registry
        self.learning_manager._save_version_registry()


# Utility functions
def create_model_update_pipeline(base_path: str = "models/incremental") -> ModelUpdatePipeline:
    """Create and return a model update pipeline"""
    return ModelUpdatePipeline(base_path)

def run_scheduled_updates(tickers: List[str], modes: List[str] = ["simple", "advanced"]) -> Dict[str, Any]:
    """Run scheduled updates for multiple tickers"""
    pipeline = create_model_update_pipeline()
    return pipeline.run_automatic_updates(tickers, modes)
