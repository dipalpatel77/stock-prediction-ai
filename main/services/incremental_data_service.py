"""
Incremental Data Service
Handles incremental data updates and synchronization
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

class UpdateType(Enum):
    """Update type enumeration"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DELTA = "delta"
    MERGE = "merge"

class DataSource(Enum):
    """Data source enumeration"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    SEC = "sec"
    FRED = "fred"
    NEWS_API = "news_api"

@dataclass
class DataUpdate:
    """Data update structure"""
    update_id: str
    ticker: str
    data_source: DataSource
    update_type: UpdateType
    start_date: datetime
    end_date: datetime
    records_updated: int
    records_added: int
    records_modified: int
    records_deleted: int
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

@dataclass
class DataSyncStatus:
    """Data sync status structure"""
    ticker: str
    last_update: datetime
    next_update: datetime
    update_frequency: str
    data_sources: List[DataSource]
    sync_status: str
    error_count: int
    success_rate: float

class IncrementalDataService:
    """Service for incremental data updates"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.data_cache = {}
        self.cache_duration = timedelta(minutes=30)
        
        # Data storage settings
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.data_dir.mkdir(exist_ok=True)
        
        # Update settings
        self.update_frequency = self.config.get('update_frequency', '1h')  # 1 hour
        self.batch_size = self.config.get('batch_size', 1000)
        self.max_retries = self.config.get('max_retries', 3)
        
        # Data source settings
        self.enabled_sources = self.config.get('enabled_sources', [
            DataSource.YAHOO_FINANCE, DataSource.ALPHA_VANTAGE
        ])
        
        # Sync status tracking
        self.sync_status = {}
        
        self.logger.info("Incremental Data Service initialized")

    def update_ticker_data(self, ticker: str, start_date: datetime = None, 
                          end_date: datetime = None, force_full_update: bool = False) -> DataUpdate:
        """Update data for a ticker"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=1)
            if end_date is None:
                end_date = datetime.now()
            
            update_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create update record
            data_update = DataUpdate(
                update_id=update_id,
                ticker=ticker,
                data_source=DataSource.YAHOO_FINANCE,  # Default
                update_type=UpdateType.FULL if force_full_update else UpdateType.INCREMENTAL,
                start_date=start_date,
                end_date=end_date,
                records_updated=0,
                records_added=0,
                records_modified=0,
                records_deleted=0,
                status='started',
                created_at=datetime.now(),
                completed_at=None,
                error_message=None
            )
            
            # Update data from each source
            for source in self.enabled_sources:
                try:
                    source_update = self._update_from_source(ticker, source, start_date, end_date, force_full_update)
                    
                    # Aggregate update statistics
                    data_update.records_updated += source_update.get('records_updated', 0)
                    data_update.records_added += source_update.get('records_added', 0)
                    data_update.records_modified += source_update.get('records_modified', 0)
                    data_update.records_deleted += source_update.get('records_deleted', 0)
                    
                except Exception as e:
                    self.logger.error(f"Error updating from {source.value}: {e}")
                    continue
            
            # Mark update as completed
            data_update.status = 'completed'
            data_update.completed_at = datetime.now()
            
            # Update sync status
            self._update_sync_status(ticker, data_update)
            
            self.logger.info(f"Updated data for {ticker}: {data_update.records_updated} records")
            return data_update
            
        except Exception as e:
            self.logger.error(f"Error updating ticker data for {ticker}: {e}")
            return DataUpdate(
                update_id=f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                ticker=ticker,
                data_source=DataSource.YAHOO_FINANCE,
                update_type=UpdateType.INCREMENTAL,
                start_date=start_date or datetime.now() - timedelta(days=1),
                end_date=end_date or datetime.now(),
                records_updated=0,
                records_added=0,
                records_modified=0,
                records_deleted=0,
                status='failed',
                created_at=datetime.now(),
                completed_at=datetime.now(),
                error_message=str(e)
            )

    def _update_from_source(self, ticker: str, source: DataSource, start_date: datetime, 
                           end_date: datetime, force_full_update: bool) -> Dict[str, int]:
        """Update data from a specific source"""
        try:
            if source == DataSource.YAHOO_FINANCE:
                return self._update_from_yahoo_finance(ticker, start_date, end_date, force_full_update)
            elif source == DataSource.ALPHA_VANTAGE:
                return self._update_from_alpha_vantage(ticker, start_date, end_date, force_full_update)
            elif source == DataSource.POLYGON:
                return self._update_from_polygon(ticker, start_date, end_date, force_full_update)
            elif source == DataSource.SEC:
                return self._update_from_sec(ticker, start_date, end_date, force_full_update)
            elif source == DataSource.FRED:
                return self._update_from_fred(ticker, start_date, end_date, force_full_update)
            elif source == DataSource.NEWS_API:
                return self._update_from_news_api(ticker, start_date, end_date, force_full_update)
            else:
                return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}
                
        except Exception as e:
            self.logger.error(f"Error updating from {source.value}: {e}")
            return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}

    def _update_from_yahoo_finance(self, ticker: str, start_date: datetime, 
                                 end_date: datetime, force_full_update: bool) -> Dict[str, int]:
        """Update data from Yahoo Finance"""
        try:
            # This would implement actual Yahoo Finance API calls
            # For now, return dummy data
            return {
                'records_updated': 100,
                'records_added': 50,
                'records_modified': 30,
                'records_deleted': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating from Yahoo Finance: {e}")
            return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}

    def _update_from_alpha_vantage(self, ticker: str, start_date: datetime, 
                                 end_date: datetime, force_full_update: bool) -> Dict[str, int]:
        """Update data from Alpha Vantage"""
        try:
            # This would implement actual Alpha Vantage API calls
            # For now, return dummy data
            return {
                'records_updated': 80,
                'records_added': 40,
                'records_modified': 20,
                'records_deleted': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating from Alpha Vantage: {e}")
            return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}

    def _update_from_polygon(self, ticker: str, start_date: datetime, 
                           end_date: datetime, force_full_update: bool) -> Dict[str, int]:
        """Update data from Polygon"""
        try:
            # This would implement actual Polygon API calls
            # For now, return dummy data
            return {
                'records_updated': 60,
                'records_added': 30,
                'records_modified': 15,
                'records_deleted': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating from Polygon: {e}")
            return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}

    def _update_from_sec(self, ticker: str, start_date: datetime, 
                        end_date: datetime, force_full_update: bool) -> Dict[str, int]:
        """Update data from SEC"""
        try:
            # This would implement actual SEC API calls
            # For now, return dummy data
            return {
                'records_updated': 20,
                'records_added': 10,
                'records_modified': 5,
                'records_deleted': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating from SEC: {e}")
            return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}

    def _update_from_fred(self, ticker: str, start_date: datetime, 
                         end_date: datetime, force_full_update: bool) -> Dict[str, int]:
        """Update data from FRED"""
        try:
            # This would implement actual FRED API calls
            # For now, return dummy data
            return {
                'records_updated': 15,
                'records_added': 8,
                'records_modified': 3,
                'records_deleted': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating from FRED: {e}")
            return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}

    def _update_from_news_api(self, ticker: str, start_date: datetime, 
                             end_date: datetime, force_full_update: bool) -> Dict[str, int]:
        """Update data from News API"""
        try:
            # This would implement actual News API calls
            # For now, return dummy data
            return {
                'records_updated': 25,
                'records_added': 12,
                'records_modified': 6,
                'records_deleted': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error updating from News API: {e}")
            return {'records_updated': 0, 'records_added': 0, 'records_modified': 0, 'records_deleted': 0}

    def _update_sync_status(self, ticker: str, data_update: DataUpdate):
        """Update sync status for a ticker"""
        try:
            if ticker not in self.sync_status:
                self.sync_status[ticker] = DataSyncStatus(
                    ticker=ticker,
                    last_update=datetime.now(),
                    next_update=datetime.now() + timedelta(hours=1),
                    update_frequency=self.update_frequency,
                    data_sources=self.enabled_sources,
                    sync_status='active',
                    error_count=0,
                    success_rate=1.0
                )
            
            sync_status = self.sync_status[ticker]
            sync_status.last_update = data_update.completed_at or datetime.now()
            sync_status.next_update = sync_status.last_update + timedelta(hours=1)
            
            if data_update.status == 'completed':
                sync_status.success_rate = min(1.0, sync_status.success_rate + 0.1)
            else:
                sync_status.error_count += 1
                sync_status.success_rate = max(0.0, sync_status.success_rate - 0.1)
                sync_status.sync_status = 'error'
            
        except Exception as e:
            self.logger.error(f"Error updating sync status: {e}")

    def get_sync_status(self, ticker: str) -> Optional[DataSyncStatus]:
        """Get sync status for a ticker"""
        try:
            return self.sync_status.get(ticker)
            
        except Exception as e:
            self.logger.error(f"Error getting sync status: {e}")
            return None

    def get_all_sync_status(self) -> Dict[str, DataSyncStatus]:
        """Get sync status for all tickers"""
        try:
            return self.sync_status.copy()
            
        except Exception as e:
            self.logger.error(f"Error getting all sync status: {e}")
            return {}

    def schedule_updates(self, tickers: List[str], update_frequency: str = None) -> bool:
        """Schedule updates for multiple tickers"""
        try:
            frequency = update_frequency or self.update_frequency
            
            for ticker in tickers:
                if ticker not in self.sync_status:
                    self.sync_status[ticker] = DataSyncStatus(
                        ticker=ticker,
                        last_update=datetime.now(),
                        next_update=datetime.now() + timedelta(hours=1),
                        update_frequency=frequency,
                        data_sources=self.enabled_sources,
                        sync_status='scheduled',
                        error_count=0,
                        success_rate=1.0
                    )
                else:
                    self.sync_status[ticker].update_frequency = frequency
                    self.sync_status[ticker].next_update = datetime.now() + timedelta(hours=1)
                    self.sync_status[ticker].sync_status = 'scheduled'
            
            self.logger.info(f"Scheduled updates for {len(tickers)} tickers")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scheduling updates: {e}")
            return False

    def get_pending_updates(self) -> List[str]:
        """Get list of tickers with pending updates"""
        try:
            pending_tickers = []
            now = datetime.now()
            
            for ticker, sync_status in self.sync_status.items():
                if sync_status.next_update <= now and sync_status.sync_status != 'error':
                    pending_tickers.append(ticker)
            
            return pending_tickers
            
        except Exception as e:
            self.logger.error(f"Error getting pending updates: {e}")
            return []

    def process_pending_updates(self) -> List[DataUpdate]:
        """Process all pending updates"""
        try:
            pending_tickers = self.get_pending_updates()
            updates = []
            
            for ticker in pending_tickers:
                try:
                    update = self.update_ticker_data(ticker)
                    updates.append(update)
                except Exception as e:
                    self.logger.error(f"Error processing update for {ticker}: {e}")
                    continue
            
            self.logger.info(f"Processed {len(updates)} pending updates")
            return updates
            
        except Exception as e:
            self.logger.error(f"Error processing pending updates: {e}")
            return []

    def get_update_history(self, ticker: str, days: int = 30) -> List[DataUpdate]:
        """Get update history for a ticker"""
        try:
            # This would load from persistent storage
            # For now, return dummy data
            history = []
            start_date = datetime.now() - timedelta(days=days)
            
            for i in range(days):
                date = start_date + timedelta(days=i)
                history.append(DataUpdate(
                    update_id=f"{ticker}_{date.strftime('%Y%m%d_%H%M%S')}",
                    ticker=ticker,
                    data_source=DataSource.YAHOO_FINANCE,
                    update_type=UpdateType.INCREMENTAL,
                    start_date=date,
                    end_date=date + timedelta(hours=1),
                    records_updated=np.random.randint(50, 200),
                    records_added=np.random.randint(10, 50),
                    records_modified=np.random.randint(5, 30),
                    records_deleted=0,
                    status='completed',
                    created_at=date,
                    completed_at=date + timedelta(minutes=30),
                    error_message=None
                ))
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting update history: {e}")
            return []

    def get_data_quality_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get data quality metrics for a ticker"""
        try:
            # This would calculate actual quality metrics
            # For now, return dummy data
            return {
                'ticker': ticker,
                'completeness': 0.95,
                'accuracy': 0.98,
                'consistency': 0.92,
                'timeliness': 0.88,
                'validity': 0.96,
                'overall_quality': 0.94,
                'last_quality_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data quality metrics: {e}")
            return {}

    def optimize_data_storage(self) -> Dict[str, Any]:
        """Optimize data storage and cleanup"""
        try:
            # This would implement actual storage optimization
            # For now, return dummy data
            return {
                'files_cleaned': 50,
                'space_freed_mb': 1024,
                'duplicates_removed': 25,
                'compression_ratio': 0.75,
                'optimization_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing data storage: {e}")
            return {}

    def get_incremental_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive incremental data summary"""
        try:
            # Get sync status for all tickers
            all_sync_status = self.get_all_sync_status()
            
            # Calculate statistics
            total_tickers = len(all_sync_status)
            active_tickers = sum(1 for status in all_sync_status.values() if status.sync_status == 'active')
            error_tickers = sum(1 for status in all_sync_status.values() if status.sync_status == 'error')
            scheduled_tickers = sum(1 for status in all_sync_status.values() if status.sync_status == 'scheduled')
            
            # Calculate average success rate
            avg_success_rate = np.mean([status.success_rate for status in all_sync_status.values()]) if all_sync_status else 0
            
            # Get pending updates
            pending_updates = self.get_pending_updates()
            
            return {
                'total_tickers': total_tickers,
                'active_tickers': active_tickers,
                'error_tickers': error_tickers,
                'scheduled_tickers': scheduled_tickers,
                'avg_success_rate': avg_success_rate,
                'pending_updates': len(pending_updates),
                'update_frequency': self.update_frequency,
                'enabled_sources': [source.value for source in self.enabled_sources],
                'summary_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting incremental data summary: {e}")
            return {}

    def export_data(self, ticker: str, start_date: datetime, end_date: datetime, 
                   format: str = 'csv') -> str:
        """Export data for a ticker"""
        try:
            # This would implement actual data export
            # For now, return dummy file path
            export_file = self.data_dir / f"{ticker}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            export_file.touch()
            
            self.logger.info(f"Exported data for {ticker} to {export_file}")
            return str(export_file)
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return ""

    def import_data(self, ticker: str, file_path: str, format: str = 'csv') -> bool:
        """Import data for a ticker"""
        try:
            # This would implement actual data import
            # For now, return success
            self.logger.info(f"Imported data for {ticker} from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing data: {e}")
            return False

    def backup_data(self, ticker: str = None) -> str:
        """Backup data for a ticker or all tickers"""
        try:
            # This would implement actual data backup
            # For now, return dummy backup path
            backup_path = self.data_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path.mkdir(exist_ok=True)
            
            self.logger.info(f"Backed up data to {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Error backing up data: {e}")
            return ""

    def restore_data(self, backup_path: str, ticker: str = None) -> bool:
        """Restore data from backup"""
        try:
            # This would implement actual data restore
            # For now, return success
            self.logger.info(f"Restored data from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring data: {e}")
            return False
