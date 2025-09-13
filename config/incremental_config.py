#!/usr/bin/env python3
"""
Incremental Data Configuration
Configuration settings for incremental data updates
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import timedelta

@dataclass
class IncrementalConfig:
    """Configuration for incremental data updates."""
    
    # Update thresholds
    max_gap_days: int = 7  # Maximum gap to allow before full refresh
    min_records: int = 10  # Minimum records required for incremental update
    cache_expiry_hours: int = 24  # Cache expiry time in hours
    
    # Data sources priority
    indian_stock_sources: List[str] = None  # Priority order for Indian stocks
    international_stock_sources: List[str] = None  # Priority order for international stocks
    
    # Update intervals
    short_term_period: str = "3mo"  # Short-term data period
    mid_term_period: str = "1y"     # Mid-term data period
    long_term_period: str = "5y"    # Long-term data period
    
    # Performance settings
    max_concurrent_downloads: int = 3  # Maximum concurrent downloads
    download_timeout: int = 30  # Download timeout in seconds
    retry_attempts: int = 3  # Number of retry attempts
    
    # Data validation
    min_data_quality_score: float = 0.8  # Minimum data quality score
    max_missing_ratio: float = 0.1  # Maximum allowed missing data ratio
    price_change_threshold: float = 0.5  # Maximum price change threshold for validation
    
    # Logging
    enable_detailed_logging: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.indian_stock_sources is None:
            self.indian_stock_sources = ["angel_one", "yfinance"]
        
        if self.international_stock_sources is None:
            self.international_stock_sources = ["yfinance"]
    
    def get_period_days(self, period: str) -> int:
        """Convert period string to days."""
        period_mapping = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        return period_mapping.get(period, 365)
    
    def get_cache_expiry_timedelta(self) -> timedelta:
        """Get cache expiry as timedelta."""
        return timedelta(hours=self.cache_expiry_hours)
    
    def is_incremental_update_allowed(self, days_since_last: int, record_count: int) -> bool:
        """Check if incremental update is allowed based on configuration."""
        return (
            days_since_last <= self.max_gap_days and
            record_count >= self.min_records
        )
    
    def get_data_source_priority(self, is_indian_stock: bool) -> List[str]:
        """Get data source priority list based on stock type."""
        if is_indian_stock:
            return self.indian_stock_sources
        else:
            return self.international_stock_sources

# Default configuration instance
DEFAULT_CONFIG = IncrementalConfig()

# Configuration presets
CONFIG_PRESETS = {
    "conservative": IncrementalConfig(
        max_gap_days=3,
        min_records=20,
        cache_expiry_hours=12,
        retry_attempts=5,
        min_data_quality_score=0.9
    ),
    
    "balanced": IncrementalConfig(
        max_gap_days=7,
        min_records=10,
        cache_expiry_hours=24,
        retry_attempts=3,
        min_data_quality_score=0.8
    ),
    
    "aggressive": IncrementalConfig(
        max_gap_days=14,
        min_records=5,
        cache_expiry_hours=48,
        retry_attempts=2,
        min_data_quality_score=0.7
    ),
    
    "development": IncrementalConfig(
        max_gap_days=1,
        min_records=1,
        cache_expiry_hours=1,
        retry_attempts=1,
        min_data_quality_score=0.5,
        enable_detailed_logging=True,
        log_level="DEBUG"
    )
}

def get_config(preset: str = "balanced") -> IncrementalConfig:
    """Get configuration by preset name."""
    return CONFIG_PRESETS.get(preset, DEFAULT_CONFIG)

def create_custom_config(**kwargs) -> IncrementalConfig:
    """Create custom configuration with overrides."""
    config = DEFAULT_CONFIG
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
