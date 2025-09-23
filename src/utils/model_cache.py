#!/usr/bin/env python3
"""
Model Caching System
Provides efficient model loading and caching with LRU eviction
"""

import os
import time
import threading
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import OrderedDict
import joblib
import pickle
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelCache:
    """Thread-safe model cache with LRU eviction"""
    
    def __init__(self, max_size: int = 50, max_memory_mb: int = 500):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_loaded = 0
        
        # Memory tracking
        self.current_memory_usage = 0
    
    def _calculate_model_size(self, model: Any) -> int:
        """Calculate approximate memory size of a model"""
        try:
            # Try to get size using pickle
            pickled = pickle.dumps(model)
            return len(pickled)
        except Exception:
            # Fallback estimation
            return 1024 * 1024  # 1MB default
    
    def _evict_oldest(self):
        """Evict the oldest (least recently used) model"""
        if not self.cache:
            return
        
        # Remove oldest entry
        oldest_key = next(iter(self.cache))
        oldest_entry = self.cache.pop(oldest_key)
        
        # Update memory usage
        self.current_memory_usage -= oldest_entry['size']
        self.evictions += 1
        
        logger.debug(f"Evicted model: {oldest_key}")
    
    def _check_memory_limit(self):
        """Check if we need to evict models due to memory limit"""
        while (self.current_memory_usage > self.max_memory_bytes and 
               len(self.cache) > 1):
            self._evict_oldest()
    
    def _generate_cache_key(self, model_path: str, model_type: str = None) -> str:
        """Generate a unique cache key for a model"""
        # Include file modification time for cache invalidation
        try:
            mtime = os.path.getmtime(model_path)
            key_data = f"{model_path}:{mtime}:{model_type or ''}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except OSError:
            # File doesn't exist, use path only
            return hashlib.md5(model_path.encode()).hexdigest()
    
    def get_model(self, model_path: str, model_type: str = None) -> Optional[Any]:
        """
        Get a model from cache
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (optional)
            
        Returns:
            Loaded model or None if not in cache
        """
        cache_key = self._generate_cache_key(model_path, model_type)
        
        with self.lock:
            if cache_key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(cache_key)
                self.cache[cache_key] = entry
                self.hits += 1
                
                logger.debug(f"Model cache hit: {model_path}")
                return entry['model']
            else:
                self.misses += 1
                logger.debug(f"Model cache miss: {model_path}")
                return None
    
    def cache_model(self, model_path: str, model: Any, model_type: str = None):
        """
        Cache a model
        
        Args:
            model_path: Path to the model file
            model: The model object to cache
            model_type: Type of model (optional)
        """
        cache_key = self._generate_cache_key(model_path, model_type)
        model_size = self._calculate_model_size(model)
        
        with self.lock:
            # Remove existing entry if it exists
            if cache_key in self.cache:
                old_entry = self.cache.pop(cache_key)
                self.current_memory_usage -= old_entry['size']
            
            # Check if we need to evict due to size limit
            while len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            # Add new entry
            entry = {
                'model': model,
                'path': model_path,
                'type': model_type,
                'size': model_size,
                'cached_at': time.time(),
                'access_count': 0
            }
            
            self.cache[cache_key] = entry
            self.current_memory_usage += model_size
            self.total_loaded += 1
            
            # Check memory limit
            self._check_memory_limit()
            
            logger.debug(f"Cached model: {model_path} (size: {model_size / 1024 / 1024:.1f}MB)")
    
    def load_model(self, model_path: str, model_type: str = None, 
                   force_reload: bool = False) -> Optional[Any]:
        """
        Load a model from file or cache
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (optional)
            force_reload: Force reload from file even if cached
            
        Returns:
            Loaded model or None if loading failed
        """
        if not force_reload:
            # Try to get from cache first
            cached_model = self.get_model(model_path, model_type)
            if cached_model is not None:
                return cached_model
        
        # Load from file
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Determine file type and load accordingly
            file_ext = Path(model_path).suffix.lower()
            
            if file_ext == '.pkl':
                model = joblib.load(model_path)
            elif file_ext == '.h5':
                # For Keras/TensorFlow models
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                except ImportError:
                    logger.error("TensorFlow not available for .h5 files")
                    return None
            else:
                # Try pickle as fallback
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Cache the loaded model
            self.cache_model(model_path, model, model_type)
            
            logger.info(f"Loaded model: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def preload_models(self, model_directory: str, pattern: str = "*.pkl"):
        """
        Preload all models matching pattern from directory
        
        Args:
            model_directory: Directory containing model files
            pattern: File pattern to match
        """
        model_dir = Path(model_directory)
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_directory}")
            return
        
        model_files = list(model_dir.glob(pattern))
        logger.info(f"Preloading {len(model_files)} models from {model_directory}")
        
        for model_file in model_files:
            try:
                self.load_model(str(model_file))
            except Exception as e:
                logger.error(f"Failed to preload {model_file}: {e}")
    
    def clear_cache(self):
        """Clear all cached models"""
        with self.lock:
            self.cache.clear()
            self.current_memory_usage = 0
            logger.info("Model cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory_usage / 1024 / 1024,
                'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'evictions': self.evictions,
                'total_loaded': self.total_loaded
            }
    
    def get_cached_models(self) -> List[Dict[str, Any]]:
        """Get list of cached models with metadata"""
        with self.lock:
            return [
                {
                    'path': entry['path'],
                    'type': entry['type'],
                    'size_mb': entry['size'] / 1024 / 1024,
                    'cached_at': entry['cached_at'],
                    'access_count': entry['access_count']
                }
                for entry in self.cache.values()
            ]

# Global model cache instance
_model_cache: Optional[ModelCache] = None

def initialize_model_cache(max_size: int = 50, max_memory_mb: int = 500) -> ModelCache:
    """Initialize the global model cache"""
    global _model_cache
    _model_cache = ModelCache(max_size, max_memory_mb)
    logger.info("Model cache initialized")
    return _model_cache

def get_model_cache() -> Optional[ModelCache]:
    """Get the global model cache instance"""
    return _model_cache

def load_model_cached(model_path: str, model_type: str = None, 
                     force_reload: bool = False) -> Optional[Any]:
    """Load a model using the global cache"""
    if _model_cache is None:
        _model_cache = initialize_model_cache()
    
    return _model_cache.load_model(model_path, model_type, force_reload)

def clear_model_cache():
    """Clear the global model cache"""
    if _model_cache:
        _model_cache.clear_cache()
