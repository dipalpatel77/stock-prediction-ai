"""
Incremental Update Service
Handles intelligent incremental data updates and synchronization
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class UpdateStrategy(Enum):
    """Update strategy enumeration"""
    FULL = "full"
    INCREMENTAL = "incremental"
    SMART = "smart"
    SCHEDULED = "scheduled"

@dataclass
class UpdateResult:
    """Update result structure"""
    success: bool
    records_added: int
    records_updated: int
    records_skipped: int
    new_data_size: int
    total_data_size: int
    update_time: datetime
    strategy_used: UpdateStrategy
    error_message: Optional[str] = None

class IncrementalUpdateService:
    """
    Service for handling incremental data updates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Update settings
        self.max_age_hours = self.config.get('max_age_hours', 24)
        self.incremental_threshold_days = self.config.get('incremental_threshold_days', 7)
        self.force_update_threshold_days = self.config.get('force_update_threshold_days', 30)
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 1000)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1)
        
        self.logger.info("Incremental Update Service initialized")
    
    def determine_update_strategy(self, ticker: str, interval: str, 
                                existing_data: pd.DataFrame) -> UpdateStrategy:
        """
        Determine the best update strategy based on data age and size
        
        Args:
            ticker: Stock ticker
            interval: Data interval
            existing_data: Existing cached data
            
        Returns:
            Update strategy to use
        """
        try:
            if existing_data.empty:
                self.logger.info(f"No existing data for {ticker}, using FULL strategy")
                return UpdateStrategy.FULL
            
            # Calculate data age
            last_date = existing_data.index.max()
            days_since_update = (datetime.now() - last_date).days
            
            # Determine strategy based on age
            if days_since_update <= 1:
                self.logger.info(f"Data is fresh for {ticker}, using INCREMENTAL strategy")
                return UpdateStrategy.INCREMENTAL
            elif days_since_update <= self.incremental_threshold_days:
                self.logger.info(f"Data is moderately stale for {ticker}, using SMART strategy")
                return UpdateStrategy.SMART
            else:
                self.logger.info(f"Data is very stale for {ticker}, using FULL strategy")
                return UpdateStrategy.FULL
                
        except Exception as e:
            self.logger.error(f"Failed to determine update strategy: {e}")
            return UpdateStrategy.FULL
    
    def perform_incremental_update(self, ticker: str, interval: str, 
                                 existing_data: pd.DataFrame,
                                 angel_manager) -> UpdateResult:
        """
        Perform incremental update for a specific ticker and interval
        
        Args:
            ticker: Stock ticker
            interval: Data interval
            existing_data: Existing cached data
            angel_manager: Angel One manager instance
            
        Returns:
            Update result with statistics
        """
        try:
            start_time = datetime.now()
            strategy = self.determine_update_strategy(ticker, interval, existing_data)
            
            if strategy == UpdateStrategy.FULL:
                return self._perform_full_update(ticker, interval, angel_manager, start_time)
            elif strategy == UpdateStrategy.INCREMENTAL:
                return self._perform_incremental_update(ticker, interval, existing_data, angel_manager, start_time)
            elif strategy == UpdateStrategy.SMART:
                return self._perform_smart_update(ticker, interval, existing_data, angel_manager, start_time)
            else:
                return self._perform_full_update(ticker, interval, angel_manager, start_time)
                
        except Exception as e:
            self.logger.error(f"Incremental update failed for {ticker}: {e}")
            return UpdateResult(
                success=False,
                records_added=0,
                records_updated=0,
                records_skipped=0,
                new_data_size=0,
                total_data_size=len(existing_data),
                update_time=datetime.now(),
                strategy_used=UpdateStrategy.FULL,
                error_message=str(e)
            )
    
    def _perform_full_update(self, ticker: str, interval: str, 
                           angel_manager, start_time: datetime) -> UpdateResult:
        """Perform full data update"""
        try:
            self.logger.info(f"Performing full update for {ticker} - {interval}")
            
            # Download fresh data
            fresh_data = angel_manager.get_stock_data(ticker, '1y', interval)
            
            if fresh_data is not None and not fresh_data.empty:
                # Store in database
                angel_manager.store_data_in_database(ticker, fresh_data, interval)
                
                return UpdateResult(
                    success=True,
                    records_added=len(fresh_data),
                    records_updated=0,
                    records_skipped=0,
                    new_data_size=len(fresh_data),
                    total_data_size=len(fresh_data),
                    update_time=datetime.now(),
                    strategy_used=UpdateStrategy.FULL
                )
            else:
                return UpdateResult(
                    success=False,
                    records_added=0,
                    records_updated=0,
                    records_skipped=0,
                    new_data_size=0,
                    total_data_size=0,
                    update_time=datetime.now(),
                    strategy_used=UpdateStrategy.FULL,
                    error_message="No data received from API"
                )
                
        except Exception as e:
            self.logger.error(f"Full update failed: {e}")
            return UpdateResult(
                success=False,
                records_added=0,
                records_updated=0,
                records_skipped=0,
                new_data_size=0,
                total_data_size=0,
                update_time=datetime.now(),
                strategy_used=UpdateStrategy.FULL,
                error_message=str(e)
            )
    
    def _perform_incremental_update(self, ticker: str, interval: str, 
                                 existing_data: pd.DataFrame,
                                 angel_manager, start_time: datetime) -> UpdateResult:
        """Perform incremental update"""
        try:
            self.logger.info(f"Performing incremental update for {ticker} - {interval}")
            
            last_date = existing_data.index.max()
            
            # Get incremental data
            new_data = angel_manager.get_incremental_data(ticker, interval, last_date)
            
            if new_data is not None and not new_data.empty:
                # Merge with existing data
                merged_data = self._merge_data_smart(existing_data, new_data)
                
                # Store updated data
                angel_manager.store_data_in_database(ticker, merged_data, interval)
                
                return UpdateResult(
                    success=True,
                    records_added=len(new_data),
                    records_updated=0,
                    records_skipped=0,
                    new_data_size=len(new_data),
                    total_data_size=len(merged_data),
                    update_time=datetime.now(),
                    strategy_used=UpdateStrategy.INCREMENTAL
                )
            else:
                self.logger.info(f"No new data available for {ticker}")
                return UpdateResult(
                    success=True,
                    records_added=0,
                    records_updated=0,
                    records_skipped=len(existing_data),
                    new_data_size=0,
                    total_data_size=len(existing_data),
                    update_time=datetime.now(),
                    strategy_used=UpdateStrategy.INCREMENTAL
                )
                
        except Exception as e:
            self.logger.error(f"Incremental update failed: {e}")
            return UpdateResult(
                success=False,
                records_added=0,
                records_updated=0,
                records_skipped=0,
                new_data_size=0,
                total_data_size=len(existing_data),
                update_time=datetime.now(),
                strategy_used=UpdateStrategy.INCREMENTAL,
                error_message=str(e)
            )
    
    def _perform_smart_update(self, ticker: str, interval: str, 
                            existing_data: pd.DataFrame,
                            angel_manager, start_time: datetime) -> UpdateResult:
        """Perform smart update (hybrid approach)"""
        try:
            self.logger.info(f"Performing smart update for {ticker} - {interval}")
            
            # Try incremental first
            last_date = existing_data.index.max()
            new_data = angel_manager.get_incremental_data(ticker, interval, last_date)
            
            if new_data is not None and not new_data.empty and len(new_data) > 0:
                # Use incremental approach
                merged_data = self._merge_data_smart(existing_data, new_data)
                angel_manager.store_data_in_database(ticker, merged_data, interval)
                
                return UpdateResult(
                    success=True,
                    records_added=len(new_data),
                    records_updated=0,
                    records_skipped=0,
                    new_data_size=len(new_data),
                    total_data_size=len(merged_data),
                    update_time=datetime.now(),
                    strategy_used=UpdateStrategy.SMART
                )
            else:
                # Fall back to full update
                self.logger.info(f"Incremental failed for {ticker}, falling back to full update")
                return self._perform_full_update(ticker, interval, angel_manager, start_time)
                
        except Exception as e:
            self.logger.error(f"Smart update failed: {e}")
            return UpdateResult(
                success=False,
                records_added=0,
                records_updated=0,
                records_skipped=0,
                new_data_size=0,
                total_data_size=len(existing_data),
                update_time=datetime.now(),
                strategy_used=UpdateStrategy.SMART,
                error_message=str(e)
            )
    
    def _merge_data_smart(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Smart merge of existing and new data
        
        Args:
            existing_data: Existing cached data
            new_data: New data to merge
            
        Returns:
            Merged DataFrame
        """
        try:
            if new_data.empty:
                return existing_data
            
            # Combine data and remove duplicates
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            
            self.logger.info(f"Merged data: {len(existing_data)} existing + {len(new_data)} new = {len(combined_data)} total")
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Data merging failed: {e}")
            return existing_data
    
    def get_update_statistics(self, ticker: str, interval: str, 
                            existing_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the data update
        
        Args:
            ticker: Stock ticker
            interval: Data interval
            existing_data: Existing cached data
            
        Returns:
            Dictionary with update statistics
        """
        try:
            if existing_data.empty:
                return {
                    'has_data': False,
                    'data_size': 0,
                    'last_date': None,
                    'days_since_update': None,
                    'needs_update': True,
                    'recommended_strategy': 'FULL'
                }
            
            last_date = existing_data.index.max()
            days_since_update = (datetime.now() - last_date).days
            
            # Determine if update is needed
            needs_update = days_since_update > 0
            
            # Recommend strategy
            if days_since_update <= 1:
                recommended_strategy = 'INCREMENTAL'
            elif days_since_update <= self.incremental_threshold_days:
                recommended_strategy = 'SMART'
            else:
                recommended_strategy = 'FULL'
            
            return {
                'has_data': True,
                'data_size': len(existing_data),
                'last_date': last_date,
                'days_since_update': days_since_update,
                'needs_update': needs_update,
                'recommended_strategy': recommended_strategy,
                'data_quality': self._assess_data_quality(existing_data)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get update statistics: {e}")
            return {
                'has_data': False,
                'data_size': 0,
                'last_date': None,
                'days_since_update': None,
                'needs_update': True,
                'recommended_strategy': 'FULL',
                'error': str(e)
            }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of the data"""
        try:
            if data.empty:
                return {'quality_score': 0, 'issues': ['No data']}
            
            issues = []
            quality_score = 100
            
            # Check for missing values
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            if missing_pct > 5:
                issues.append(f'High missing values: {missing_pct:.1f}%')
                quality_score -= 20
            
            # Check for duplicates
            duplicate_pct = (data.index.duplicated().sum() / len(data)) * 100
            if duplicate_pct > 0:
                issues.append(f'Duplicate dates: {duplicate_pct:.1f}%')
                quality_score -= 10
            
            # Check for data gaps
            date_diff = data.index.to_series().diff()
            expected_diff = pd.Timedelta(days=1) if 'DAY' in str(data.index[0]) else pd.Timedelta(hours=1)
            gap_ratio = (date_diff > expected_diff * 2).sum() / len(data)
            if gap_ratio > 0.1:
                issues.append(f'Data gaps: {gap_ratio:.1f}%')
                quality_score -= 15
            
            return {
                'quality_score': max(0, quality_score),
                'issues': issues,
                'missing_pct': missing_pct,
                'duplicate_pct': duplicate_pct,
                'gap_ratio': gap_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Data quality assessment failed: {e}")
            return {'quality_score': 0, 'issues': ['Assessment failed']}
