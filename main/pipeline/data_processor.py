#!/usr/bin/env python3
"""
Optimized Data Processor
Enhanced data processing pipeline with memory optimization, streaming, and performance monitoring
"""

import pandas as pd
import numpy as np
import logging
import time
import gc
import psutil
import asyncio
import aiofiles
from typing import Dict, Any, Optional, List, Generator, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

from .base_pipeline import BasePipelineComponent
from ..services.data_service_wrapper import DataServiceWrapper
from ..services.angel_one_manager import AngelOneManager
from ..services.database_manager import DatabaseManager
from ..services.api_coordinator import APICoordinator

logger = logging.getLogger(__name__)

class DataProcessor(BasePipelineComponent):
    """Optimized data processor with memory optimization, streaming, and performance monitoring"""
    
    def __init__(self, ticker: str = "AAPL", config: Dict[str, Any] = None):
        if config is None:
            config = {
                'database_url': 'sqlite:///default.db',
                'use_angel_one': False,
                'cache_enabled': True,
                'memory_limit_mb': 1024,
                'chunk_size': 1000,
                'enable_streaming': True,
                'enable_memory_monitoring': True
            }
        super().__init__(ticker, config)
        
        # Memory optimization settings
        self.memory_limit = config.get('memory_limit_mb', 1024) * 1024 * 1024  # Convert to bytes
        self.chunk_size = config.get('chunk_size', 1000)
        self.enable_streaming = config.get('enable_streaming', True)
        self.enable_memory_monitoring = config.get('enable_memory_monitoring', True)
        
        # Performance monitoring
        self.performance_metrics = {
            'processing_time': 0,
            'memory_usage': 0,
            'chunks_processed': 0,
            'data_points_processed': 0,
            'memory_peak': 0
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.memory_lock = threading.Lock()
        
        # Async support
        self.async_config = {
            'enable_async': True,
            'async_chunk_size': 500,
            'async_timeout': 30,
            'enable_streaming': True
        }
        
        # Initialize service integrations
        self.data_wrapper = DataServiceWrapper(ticker, config)
        # Extract ticker string from config if it's a dictionary
        ticker_str = ticker if isinstance(ticker, str) else config.get('ticker', 'UNKNOWN')
        self.angel_manager = AngelOneManager(config) if self._is_indian_stock(ticker_str) else None
        self.db_manager = DatabaseManager(config)
        self.api_coordinator = APICoordinator()
        
        # Initialize data processing services
        self._initialize_data_services()
        
        logger.info(f"Optimized DataProcessor initialized for {ticker} with memory monitoring and async support")
    
    def _is_indian_stock(self, ticker: str) -> bool:
        """
        Check if ticker is an Indian stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if Indian stock, False otherwise
        """
        try:
            indian_indicators = ['.NS', '.BO', '.NSE', '.BSE']
            return any(ticker.endswith(indicator) for indicator in indian_indicators)
        except Exception as e:
            logger.error(f"Failed to check if Indian stock: {e}")
            return False
    
    def _initialize_data_services(self):
        """Initialize data processing services"""
        try:
            # Import technical indicators service
            from ..services.technical_indicators_service import TechnicalIndicatorsService
            self.technical_indicators = TechnicalIndicatorsService()
            
            # Import feature engineering service
            from ..services.feature_engineering_service import FeatureEngineeringService
            self.feature_engineering = FeatureEngineeringService()
            
            # Import economic data service
            # Economic data service removed
            self.economic_data = None
            
            logger.info("Data processing services initialized")
            
        except Exception as e:
            logger.warning(f"Some data processing services not available: {e}")
            self.technical_indicators = None
            self.feature_engineering = None
            self.economic_data = None
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required parameters
            required_params = ['period', 'interval']
            for param in required_params:
                if param not in kwargs:
                    logger.error(f"Missing required parameter: {param}")
                    return False
            
            # Validate period
            valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
            if kwargs['period'] not in valid_periods:
                logger.error(f"Invalid period: {kwargs['period']}")
                return False
            
            # Validate interval
            valid_intervals = ['ONE_DAY', 'ONE_HOUR', 'THIRTY_MINUTE', 'FIFTEEN_MINUTE', 'FIVE_MINUTE', 'ONE_MINUTE']
            if kwargs['interval'] not in valid_intervals:
                logger.error(f"Invalid interval: {kwargs['interval']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute enhanced data processing pipeline
        
        Args:
            **kwargs: Processing parameters
                - period: Data period (e.g., '1y', '6mo')
                - interval: Data interval (e.g., 'ONE_DAY', 'ONE_HOUR')
                - include_technical: Whether to include technical indicators
                - include_economic: Whether to include economic data
                - include_features: Whether to include feature engineering
                
        Returns:
            Dictionary with processing results
        """
        try:
            self._log_progress("Starting enhanced data processing pipeline")
            
            # Validate input
            if not self.validate_input(**kwargs):
                return self._handle_error(ValueError("Invalid input parameters"), "Input validation")
            
            # Step 1: Load raw data from appropriate source
            period = kwargs.get('period', '1y')
            interval = kwargs.get('interval', 'ONE_DAY')
            
            # For Indian stocks, load comprehensive data for all intervals
            if self._is_indian_stock(self.ticker):
                print(f"\n📊 Loading comprehensive data for {self.ticker} (Indian stock)")
                comprehensive_data = self._load_comprehensive_angel_one_data(period)
                if comprehensive_data:
                    # Use ONE_DAY data as primary for processing
                    raw_data = comprehensive_data.get('ONE_DAY')
                    if raw_data is None or raw_data.empty:
                        # Fallback to any available interval
                        raw_data = next(iter(comprehensive_data.values()), None)
                    print(f"✅ Comprehensive data loaded: {len(comprehensive_data)} intervals")
                else:
                    print("⚠️ Comprehensive data failed, falling back to standard loading")
                    raw_data = self._load_stock_data(period, interval)
            else:
                # For non-Indian stocks, use standard loading
                raw_data = self._load_stock_data(period, interval)
            
            if raw_data is None or raw_data.empty:
                return self._handle_error(ValueError("No data loaded"), "Data loading")
            
            # Step 2: Store data in database
            self._store_data_in_database(raw_data, kwargs.get('interval', 'ONE_DAY'))
            
            # Step 3: Clean and preprocess
            cleaned_data = self._clean_and_preprocess(raw_data)
            
            # Step 4: Add technical indicators
            enhanced_data = self._add_technical_indicators(
                cleaned_data, 
                kwargs.get('include_technical', True)
            )
            
            # Step 5: Add economic and market data
            enriched_data = self._enrich_with_external_data(
                enhanced_data,
                kwargs.get('include_economic', True)
            )
            
            # Step 6: Feature engineering
            features = self._engineer_features(
                enriched_data,
                kwargs.get('include_features', True)
            )
            
            # Step 7: Data quality assessment
            quality_metrics = self._assess_data_quality(enriched_data)
            
            result = {
                'success': True,
                'data': enriched_data,  # Main processed data for other components
                'processed_data': enriched_data,  # Alias for compatibility
                'raw_data': raw_data,
                'cleaned_data': cleaned_data,
                'enhanced_data': enhanced_data,
                'enriched_data': enriched_data,
                'features': features,
                'quality_metrics': quality_metrics,
                'data_source': self._get_data_source(),
                'processing_summary': self._get_processing_summary(raw_data, enriched_data),
                'records_processed': len(enriched_data) if enriched_data is not None else 0,
                'quality_score': quality_metrics.get('overall_score', 0) * 100 if quality_metrics else 0
            }
            
            self.logger.info(f"Data processor returning: data={enriched_data is not None}, shape={enriched_data.shape if enriched_data is not None else 'None'}")
            
            self._log_progress("Enhanced data processing pipeline completed successfully")
            return result
            
        except Exception as e:
            return self._handle_error(e, "Data processing pipeline")
    
    def _log_progress(self, message: str):
        """Log progress message"""
        self.logger.info(f"📊 {message}")
    
    def _handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle errors with proper logging and formatting"""
        error_msg = f"{context} failed: {str(error)}"
        self.logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg,
            'context': context,
            'ticker': self.ticker
        }
    
    def _load_stock_data(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from appropriate source
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with stock data
        """
        try:
            self._log_progress(f"Loading data for {self.ticker} - period: {period}, interval: {interval}")
            
            # Check cache first
            cached_data = self._get_cached_data(period, interval)
            if cached_data is not None and not cached_data.empty:
                self._log_progress("Using cached data")
                return cached_data
            
            # Load from appropriate source
            ticker_str = self.ticker if isinstance(self.ticker, str) else self.config.get('ticker', 'UNKNOWN')
            if self._is_indian_stock(ticker_str) and self.angel_manager:
                data = self._load_angel_one_data(period, interval)
            else:
                data = self._load_yahoo_data(period, interval)
            
            if data is not None and not data.empty:
                self._log_progress(f"Successfully loaded {len(data)} records")
                return data
            else:
                raise Exception("No data received from any source")
                
        except Exception as e:
            self._log_progress(f"Data loading failed: {e}")
            raise e
    
    def _get_cached_data(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get cached data from database
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            Cached DataFrame or None if not found
        """
        try:
            source = self._get_data_source()
            data = self.db_manager.get_stock_data(
                ticker=self.ticker,
                period=period,
                source=source,
                interval=interval
            )
            
            if data is not None and not data.empty:
                self._log_progress(f"Retrieved {len(data)} cached records")
                return data
            
            return None
            
        except Exception as e:
            self._log_progress(f"Cache retrieval failed: {e}")
            return None
    
    def _load_angel_one_data(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load Angel One data
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with Angel One data
        """
        try:
            if not self.angel_manager:
                return None
            
            data = self.angel_manager.get_stock_data(self.ticker, period, interval)
            return data
            
        except Exception as e:
            self._log_progress(f"Angel One data loading failed: {e}")
            return None
    
    def _load_comprehensive_angel_one_data(self, period: str) -> Dict[str, pd.DataFrame]:
        """
        Load comprehensive Angel One data for all intervals
        
        Args:
            period: Data period
            
        Returns:
            Dictionary with interval as key and DataFrame as value
        """
        try:
            if not self.angel_manager:
                self._log_progress("Angel One manager not available")
                return {}
            
            # Use comprehensive data loading
            comprehensive_data = self.angel_manager.get_comprehensive_historical_data(
                self.ticker, 
                self.angel_manager.config.get('exchange', 'NSE'), 
                period
            )
            
            if comprehensive_data:
                self._log_progress(f"Loaded comprehensive data: {len(comprehensive_data)} intervals")
                return comprehensive_data
            else:
                self._log_progress("No comprehensive data received from Angel One")
                return {}
                
        except Exception as e:
            self._log_progress(f"Comprehensive Angel One data loading failed: {e}")
            return {}
    
    def _load_yahoo_data(self, period: str, interval: str) -> pd.DataFrame:
        """
        Load Yahoo Finance data
        
        Args:
            period: Data period
            interval: Data interval
            
        Returns:
            DataFrame with Yahoo Finance data
        """
        try:
            data = self.data_wrapper.load_stock_data(period, interval)
            return data
            
        except Exception as e:
            self._log_progress(f"Yahoo Finance data loading failed: {e}")
            raise e
    
    def _store_data_in_database(self, data: pd.DataFrame, interval: str):
        """
        Store data in database
        
        Args:
            data: Stock data DataFrame
            interval: Data interval
        """
        try:
            source = self._get_data_source()
            self.db_manager.store_stock_data(
                ticker=self.ticker,
                data=data,
                source=source,
                interval=interval
            )
            
            self._log_progress(f"Stored {len(data)} records in database")
            
        except Exception as e:
            self._log_progress(f"Database storage failed: {e}")
            # Don't raise exception - processing can continue without storage
    
    def _clean_and_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized data cleaning and preprocessing with memory management
        
        Args:
            data: Raw stock data
            
        Returns:
            Cleaned DataFrame
        """
        try:
            start_time = time.time()
            self._log_progress("Cleaning and preprocessing data with memory optimization")
            
            # Check memory usage before processing
            initial_memory = self._get_memory_usage()
            
            # Process data in chunks if it's large
            if len(data) > self.chunk_size and self.enable_streaming:
                cleaned_data = self._process_data_in_chunks(data)
            else:
                # Process entire dataset
                cleaned_data = self._process_full_dataset(data)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            
            with self.memory_lock:
                self.performance_metrics['processing_time'] += processing_time
                self.performance_metrics['memory_usage'] = final_memory
                self.performance_metrics['data_points_processed'] += len(cleaned_data)
                self.performance_metrics['memory_peak'] = max(
                    self.performance_metrics['memory_peak'], 
                    final_memory
                )
            
            # Force garbage collection to free memory
            gc.collect()
            
            self._log_progress(f"Data cleaning completed: {len(cleaned_data)} records in {processing_time:.2f}s")
            return cleaned_data
            
        except Exception as e:
            self._log_progress(f"Data cleaning failed: {e}")
            raise e
    
    def _process_data_in_chunks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process large datasets in chunks to manage memory usage
        
        Args:
            data: Raw stock data
            
        Returns:
            Processed DataFrame
        """
        try:
            logger.info(f"Processing {len(data)} records in chunks of {self.chunk_size}")
            processed_chunks = []
            
            for i in range(0, len(data), self.chunk_size):
                chunk = data.iloc[i:i + self.chunk_size].copy()
                
                # Process chunk
                processed_chunk = self._process_chunk(chunk)
                processed_chunks.append(processed_chunk)
                
                # Update metrics
                with self.memory_lock:
                    self.performance_metrics['chunks_processed'] += 1
                
                # Check memory usage
                if self.enable_memory_monitoring:
                    current_memory = self._get_memory_usage()
                    if current_memory > self.memory_limit:
                        logger.warning(f"Memory limit exceeded: {current_memory / (1024*1024):.1f}MB")
                        # Force garbage collection
                        gc.collect()
                
                logger.debug(f"Processed chunk {i//self.chunk_size + 1}/{(len(data)-1)//self.chunk_size + 1}")
            
            # Combine processed chunks
            result = pd.concat(processed_chunks, ignore_index=False)
            result = result.sort_index()
            
            # Clean up chunks to free memory
            del processed_chunks
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            raise e
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single chunk of data
        
        Args:
            chunk: Data chunk to process
            
        Returns:
            Processed chunk
        """
        try:
            # Remove duplicates
            chunk = chunk.drop_duplicates()
            
            # Handle missing values
            chunk = self._handle_missing_values(chunk)
            
            # Remove outliers
            chunk = self._remove_outliers(chunk)
            
            # Ensure proper data types
            chunk = self._ensure_data_types(chunk)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            raise e
    
    def _process_full_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataset at once
        
        Args:
            data: Raw stock data
            
        Returns:
            Processed DataFrame
        """
        try:
            # Make a copy to avoid modifying original
            cleaned_data = data.copy()
            
            # Remove duplicates
            cleaned_data = cleaned_data.drop_duplicates()
            
            # Handle missing values
            cleaned_data = self._handle_missing_values(cleaned_data)
            
            # Remove outliers
            cleaned_data = self._remove_outliers(cleaned_data)
            
            # Ensure proper data types
            cleaned_data = self._ensure_data_types(cleaned_data)
            
            # Sort by date
            cleaned_data = cleaned_data.sort_index()
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Full dataset processing failed: {e}")
            raise e
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in bytes
        
        Returns:
            Memory usage in bytes
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Get detailed memory usage report
        
        Returns:
            Dictionary with memory metrics
        """
        try:
            current_memory = self._get_memory_usage()
            
            return {
                'current_memory_mb': round(current_memory / (1024 * 1024), 2),
                'memory_limit_mb': round(self.memory_limit / (1024 * 1024), 2),
                'memory_usage_percent': round((current_memory / self.memory_limit) * 100, 2),
                'peak_memory_mb': round(self.performance_metrics['memory_peak'] / (1024 * 1024), 2),
                'chunks_processed': self.performance_metrics['chunks_processed'],
                'data_points_processed': self.performance_metrics['data_points_processed'],
                'processing_time': round(self.performance_metrics['processing_time'], 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate memory report: {e}")
            return {'error': str(e)}
    
    async def async_process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Async process data with streaming and memory optimization
        
        Args:
            data: Raw stock data
            
        Returns:
            Processed DataFrame
        """
        try:
            start_time = time.time()
            self._log_progress("Async processing data with streaming optimization")
            
            # Check memory usage before processing
            initial_memory = self._get_memory_usage()
            
            # Process data in async chunks if it's large
            if len(data) > self.async_config['async_chunk_size'] and self.async_config['enable_streaming']:
                processed_data = await self._async_process_data_in_chunks(data)
            else:
                # Process entire dataset
                processed_data = await self._async_process_full_dataset(data)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            
            with self.memory_lock:
                self.performance_metrics['processing_time'] += processing_time
                self.performance_metrics['memory_usage'] = final_memory
                self.performance_metrics['data_points_processed'] += len(processed_data)
                self.performance_metrics['memory_peak'] = max(
                    self.performance_metrics['memory_peak'], 
                    final_memory
                )
            
            # Force garbage collection to free memory
            gc.collect()
            
            self._log_progress(f"Async data processing completed: {len(processed_data)} records in {processing_time:.2f}s")
            return processed_data
            
        except Exception as e:
            self._log_progress(f"Async data processing failed: {e}")
            raise e
    
    async def _async_process_data_in_chunks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Async process large datasets in chunks to manage memory usage
        
        Args:
            data: Raw stock data
            
        Returns:
            Processed DataFrame
        """
        try:
            logger.info(f"Async processing {len(data)} records in chunks of {self.async_config['async_chunk_size']}")
            processed_chunks = []
            
            # Create async tasks for chunk processing
            async_tasks = []
            for i in range(0, len(data), self.async_config['async_chunk_size']):
                chunk = data.iloc[i:i + self.async_config['async_chunk_size']].copy()
                task = asyncio.create_task(self._async_process_chunk(chunk))
                async_tasks.append(task)
            
            # Execute all chunks concurrently
            processed_chunks = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Handle exceptions
            valid_chunks = []
            for i, chunk in enumerate(processed_chunks):
                if isinstance(chunk, Exception):
                    logger.error(f"Async chunk {i} processing failed: {chunk}")
                else:
                    valid_chunks.append(chunk)
                    with self.memory_lock:
                        self.performance_metrics['chunks_processed'] += 1
                
                # Check memory usage
                if self.enable_memory_monitoring:
                    current_memory = self._get_memory_usage()
                    if current_memory > self.memory_limit:
                        logger.warning(f"Memory limit exceeded: {current_memory / (1024*1024):.1f}MB")
                        # Force garbage collection
                        gc.collect()
            
            # Combine processed chunks
            if valid_chunks:
                result = pd.concat(valid_chunks, ignore_index=False)
                result = result.sort_index()
            else:
                result = pd.DataFrame()
            
            # Clean up chunks to free memory
            del processed_chunks
            del valid_chunks
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Async chunk processing failed: {e}")
            raise e
    
    async def _async_process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Async process a single chunk of data
        
        Args:
            chunk: Data chunk to process
            
        Returns:
            Processed chunk
        """
        try:
            # Simulate async processing with small delay
            await asyncio.sleep(0.01)  # Simulate async I/O
            
            # Remove duplicates
            chunk = chunk.drop_duplicates()
            
            # Handle missing values
            chunk = self._handle_missing_values(chunk)
            
            # Remove outliers
            chunk = self._remove_outliers(chunk)
            
            # Ensure proper data types
            chunk = self._ensure_data_types(chunk)
            
            return chunk
            
        except Exception as e:
            logger.error(f"Async chunk processing failed: {e}")
            raise e
    
    async def _async_process_full_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Async process entire dataset at once
        
        Args:
            data: Raw stock data
            
        Returns:
            Processed DataFrame
        """
        try:
            # Simulate async processing
            await asyncio.sleep(0.01)  # Simulate async I/O
            
            # Make a copy to avoid modifying original
            cleaned_data = data.copy()
            
            # Remove duplicates
            cleaned_data = cleaned_data.drop_duplicates()
            
            # Handle missing values
            cleaned_data = self._handle_missing_values(cleaned_data)
            
            # Remove outliers
            cleaned_data = self._remove_outliers(cleaned_data)
            
            # Ensure proper data types
            cleaned_data = self._ensure_data_types(cleaned_data)
            
            # Sort by date
            cleaned_data = cleaned_data.sort_index()
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Async full dataset processing failed: {e}")
            raise e
    
    async def async_stream_data_processing(self, data_source: str, ticker: str, config: Dict[str, Any]) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Async stream data processing for real-time data
        
        Args:
            data_source: Data source name
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Yields:
            Processed data chunks
        """
        try:
            logger.info(f"Starting async data streaming for {ticker} from {data_source}")
            
            # Simulate streaming data processing
            async for chunk in self._async_data_stream(data_source, ticker, config):
                try:
                    # Process chunk
                    processed_chunk = await self._async_process_chunk(chunk)
                    
                    # Update metrics
                    with self.memory_lock:
                        self.performance_metrics['chunks_processed'] += 1
                        self.performance_metrics['data_points_processed'] += len(processed_chunk)
                    
                    yield processed_chunk
                    
                except Exception as e:
                    logger.error(f"Async streaming chunk processing failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Async data streaming failed for {ticker}: {e}")
            yield pd.DataFrame()  # Return empty DataFrame on error
    
    async def _async_data_stream(self, data_source: str, ticker: str, config: Dict[str, Any]) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Async data stream generator
        
        Args:
            data_source: Data source name
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Yields:
            Data chunks
        """
        try:
            # Simulate streaming data
            for i in range(10):  # Simulate 10 chunks
                # Create sample data chunk
                chunk_data = {
                    'Open': np.random.uniform(100, 200),
                    'High': np.random.uniform(100, 200),
                    'Low': np.random.uniform(100, 200),
                    'Close': np.random.uniform(100, 200),
                    'Volume': np.random.randint(1000, 10000)
                }
                
                chunk = pd.DataFrame([chunk_data], index=[datetime.now()])
                
                # Simulate async delay
                await asyncio.sleep(0.1)
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Async data stream failed: {e}")
            yield pd.DataFrame()
    
    async def async_store_data(self, data: pd.DataFrame, source: str, interval: str = 'ONE_DAY'):
        """
        Async store data in database
        
        Args:
            data: Processed data
            source: Data source
            interval: Data interval
        """
        try:
            logger.info(f"Async storing {len(data)} records from {source}")
            
            # Use async database operations
            if hasattr(self.db_manager, 'async_batch_insert_data'):
                # Convert DataFrame to list of dictionaries
                data_list = data.reset_index().to_dict('records')
                
                # Store in appropriate table
                table_name = f"{source}_data" if source != 'angel_one' else 'angel_one_stock_data'
                
                success = await self.db_manager.async_batch_insert_data(
                    table_name=table_name,
                    data=data_list,
                    batch_size=1000
                )
                
                if success:
                    logger.info(f"Async stored {len(data)} records from {source}")
                else:
                    logger.warning(f"Async storage failed for {source}")
            else:
                logger.warning("Async database operations not available")
                
        except Exception as e:
            logger.error(f"Async data storage failed: {e}")
    
    async def async_get_performance_report(self) -> Dict[str, Any]:
        """
        Async get detailed performance report
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Simulate async processing
            await asyncio.sleep(0.01)
            
            current_memory = self._get_memory_usage()
            
            return {
                'current_memory_mb': round(current_memory / (1024 * 1024), 2),
                'memory_limit_mb': round(self.memory_limit / (1024 * 1024), 2),
                'memory_usage_percent': round((current_memory / self.memory_limit) * 100, 2),
                'peak_memory_mb': round(self.performance_metrics['memory_peak'] / (1024 * 1024), 2),
                'chunks_processed': self.performance_metrics['chunks_processed'],
                'data_points_processed': self.performance_metrics['data_points_processed'],
                'processing_time': round(self.performance_metrics['processing_time'], 2),
                'async_enabled': self.async_config['enable_async'],
                'streaming_enabled': self.async_config['enable_streaming']
            }
            
        except Exception as e:
            logger.error(f"Async performance report generation failed: {e}")
            return {'error': str(e)}
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in data
        
        Args:
            data: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        try:
            # Forward fill missing values
            data = data.fillna(method='ffill')
            
            # Backward fill any remaining missing values
            data = data.fillna(method='bfill')
            
            # Drop any rows that still have missing values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            self._log_progress(f"Missing value handling failed: {e}")
            return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from data
        
        Args:
            data: DataFrame with potential outliers
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            # Use IQR method to detect outliers
            for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if column in data.columns:
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define outlier bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Remove outliers
                    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            
            return data
            
        except Exception as e:
            self._log_progress(f"Outlier removal failed: {e}")
            return data
    
    def _ensure_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure proper data types
        
        Args:
            data: DataFrame to process
            
        Returns:
            DataFrame with proper data types
        """
        try:
            # Ensure numeric columns are numeric
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for column in numeric_columns:
                if column in data.columns:
                    data[column] = pd.to_numeric(data[column], errors='coerce')
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            self._log_progress(f"Data type conversion failed: {e}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame, include_technical: bool = True) -> pd.DataFrame:
        """
        Add technical indicators to data
        
        Args:
            data: Cleaned stock data
            include_technical: Whether to include technical indicators
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            if not include_technical or not self.technical_indicators:
                self._log_progress("Skipping technical indicators")
                return data
            
            self._log_progress("Adding technical indicators")
            
            # Add basic technical indicators
            enhanced_data = data.copy()
            
            # Simple Moving Averages
            enhanced_data['SMA_20'] = enhanced_data['Close'].rolling(window=20).mean()
            enhanced_data['SMA_50'] = enhanced_data['Close'].rolling(window=50).mean()
            enhanced_data['SMA_200'] = enhanced_data['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            enhanced_data['EMA_12'] = enhanced_data['Close'].ewm(span=12).mean()
            enhanced_data['EMA_26'] = enhanced_data['Close'].ewm(span=26).mean()
            
            # MACD
            enhanced_data['MACD'] = enhanced_data['EMA_12'] - enhanced_data['EMA_26']
            enhanced_data['MACD_Signal'] = enhanced_data['MACD'].ewm(span=9).mean()
            enhanced_data['MACD_Histogram'] = enhanced_data['MACD'] - enhanced_data['MACD_Signal']
            
            # RSI
            delta = enhanced_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            enhanced_data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            enhanced_data['BB_Middle'] = enhanced_data['Close'].rolling(window=20).mean()
            bb_std = enhanced_data['Close'].rolling(window=20).std()
            enhanced_data['BB_Upper'] = enhanced_data['BB_Middle'] + (bb_std * 2)
            enhanced_data['BB_Lower'] = enhanced_data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            enhanced_data['Volume_SMA'] = enhanced_data['Volume'].rolling(window=20).mean()
            enhanced_data['Volume_Ratio'] = enhanced_data['Volume'] / enhanced_data['Volume_SMA']
            
            self._log_progress(f"Added technical indicators: {len(enhanced_data.columns)} columns")
            return enhanced_data
            
        except Exception as e:
            self._log_progress(f"Technical indicators failed: {e}")
            return data
    
    def _enrich_with_external_data(self, data: pd.DataFrame, include_economic: bool = True) -> pd.DataFrame:
        """
        Enrich data with external economic and market data
        
        Args:
            data: Enhanced stock data
            include_economic: Whether to include economic data
            
        Returns:
            DataFrame with external data
        """
        try:
            if not include_economic:
                self._log_progress("Skipping external data enrichment")
                return data
            
            self._log_progress("Enriching with external data")
            
            enriched_data = data.copy()
            
            # Add economic indicators (placeholder implementation)
            if self.economic_data:
                try:
                    economic_indicators = self.economic_data.get_economic_indicators()
                    
                    # Add economic data as constant values for now
                    # In a real implementation, these would be time-series data
                    enriched_data['GDP_Growth'] = economic_indicators.get('gdp_growth', 0.0)
                    enriched_data['Inflation_Rate'] = economic_indicators.get('inflation_rate', 0.0)
                    enriched_data['Interest_Rate'] = economic_indicators.get('interest_rate', 0.0)
                    
                except Exception as e:
                    self._log_progress(f"Economic data enrichment failed: {e}")
            
            # Add market indicators
            try:
                # VIX (Volatility Index) - placeholder
                enriched_data['VIX'] = 20.0  # Placeholder value
                
                # Market sentiment - placeholder
                enriched_data['Market_Sentiment'] = 0.5  # Placeholder value
                
            except Exception as e:
                self._log_progress(f"Market indicators failed: {e}")
            
            self._log_progress(f"External data enrichment completed: {len(enriched_data.columns)} columns")
            return enriched_data
            
        except Exception as e:
            self._log_progress(f"External data enrichment failed: {e}")
            return data
    
    def _engineer_features(self, data: pd.DataFrame, include_features: bool = True) -> Dict[str, Any]:
        """
        Engineer features from enriched data
        
        Args:
            data: Enriched stock data
            include_features: Whether to include feature engineering
            
        Returns:
            Dictionary with engineered features
        """
        try:
            if not include_features:
                self._log_progress("Skipping feature engineering")
                return {}
            
            self._log_progress("Engineering features")
            
            features = {}
            
            # Price-based features
            features['price_features'] = self._create_price_features(data)
            
            # Technical indicator features
            features['technical_features'] = self._create_technical_features(data)
            
            # Volume features
            features['volume_features'] = self._create_volume_features(data)
            
            # Time-based features
            features['time_features'] = self._create_time_features(data)
            
            # Statistical features
            features['statistical_features'] = self._create_statistical_features(data)
            
            self._log_progress(f"Feature engineering completed: {len(features)} feature groups")
            return features
            
        except Exception as e:
            self._log_progress(f"Feature engineering failed: {e}")
            return {}
    
    def _create_price_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create price-based features"""
        try:
            features = {}
            
            # Price changes
            features['price_change'] = data['Close'].pct_change()
            features['price_change_abs'] = data['Close'].diff()
            
            # Price ranges
            features['daily_range'] = data['High'] - data['Low']
            features['daily_range_pct'] = features['daily_range'] / data['Close']
            
            # Price positions
            features['close_to_high'] = data['Close'] / data['High']
            features['close_to_low'] = data['Close'] / data['Low']
            features['close_to_open'] = data['Close'] / data['Open']
            
            return features
            
        except Exception as e:
            self._log_progress(f"Price features failed: {e}")
            return {}
    
    def _create_technical_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create technical indicator features"""
        try:
            features = {}
            
            # Moving average features
            if 'SMA_20' in data.columns:
                features['sma_20_signal'] = (data['Close'] > data['SMA_20']).astype(int)
            if 'SMA_50' in data.columns:
                features['sma_50_signal'] = (data['Close'] > data['SMA_50']).astype(int)
            
            # MACD features
            if 'MACD' in data.columns:
                features['macd_signal'] = (data['MACD'] > data['MACD_Signal']).astype(int)
                features['macd_histogram'] = data['MACD_Histogram']
            
            # RSI features
            if 'RSI' in data.columns:
                features['rsi_oversold'] = (data['RSI'] < 30).astype(int)
                features['rsi_overbought'] = (data['RSI'] > 70).astype(int)
            
            return features
            
        except Exception as e:
            self._log_progress(f"Technical features failed: {e}")
            return {}
    
    def _create_volume_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create volume-based features"""
        try:
            features = {}
            
            # Volume changes
            features['volume_change'] = data['Volume'].pct_change()
            features['volume_change_abs'] = data['Volume'].diff()
            
            # Volume ratios
            if 'Volume_SMA' in data.columns:
                features['volume_ratio'] = data['Volume'] / data['Volume_SMA']
                features['volume_above_avg'] = (data['Volume'] > data['Volume_SMA']).astype(int)
            
            return features
            
        except Exception as e:
            self._log_progress(f"Volume features failed: {e}")
            return {}
    
    def _create_time_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create time-based features"""
        try:
            features = {}
            
            # Day of week
            features['day_of_week'] = data.index.dayofweek
            features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features['is_friday'] = (data.index.dayofweek == 4).astype(int)
            
            # Month
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Year
            features['year'] = data.index.year
            
            return features
            
        except Exception as e:
            self._log_progress(f"Time features failed: {e}")
            return {}
    
    def _create_statistical_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create statistical features"""
        try:
            features = {}
            
            # Rolling statistics
            window = 20
            features['price_std'] = data['Close'].rolling(window=window).std()
            features['price_mean'] = data['Close'].rolling(window=window).mean()
            features['price_skew'] = data['Close'].rolling(window=window).skew()
            features['price_kurt'] = data['Close'].rolling(window=window).kurt()
            
            # Z-scores
            features['price_zscore'] = (data['Close'] - features['price_mean']) / features['price_std']
            
            return features
            
        except Exception as e:
            self._log_progress(f"Statistical features failed: {e}")
            return {}
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data quality
        
        Args:
            data: Processed data
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = {
                'total_records': len(data),
                'missing_values': data.isnull().sum().sum(),
                'duplicate_records': data.duplicated().sum(),
                'date_range': {
                    'start': data.index.min().isoformat() if not data.empty else None,
                    'end': data.index.max().isoformat() if not data.empty else None
                },
                'columns': list(data.columns),
                'data_types': data.dtypes.to_dict(),
                'completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            }
            
            return quality_metrics
            
        except Exception as e:
            self._log_progress(f"Data quality assessment failed: {e}")
            return {}
    
    def _get_processing_summary(self, raw_data: pd.DataFrame, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get processing summary
        
        Args:
            raw_data: Original raw data
            processed_data: Final processed data
            
        Returns:
            Dictionary with processing summary
        """
        try:
            summary = {
                'raw_records': len(raw_data),
                'processed_records': len(processed_data),
                'raw_columns': len(raw_data.columns),
                'processed_columns': len(processed_data.columns),
                'data_reduction': len(raw_data) - len(processed_data),
                'column_expansion': len(processed_data.columns) - len(raw_data.columns),
                'processing_efficiency': len(processed_data) / len(raw_data) * 100 if len(raw_data) > 0 else 0
            }
            
            return summary
            
        except Exception as e:
            self._log_progress(f"Processing summary failed: {e}")
            return {}
    
    def _get_data_source(self) -> str:
        """
        Get the data source being used
        
        Returns:
            Data source name
        """
        return self.data_wrapper.get_data_source()
    
    def _validate_component_input(self, **kwargs) -> bool:
        """
        Validate component-specific input
        
        Args:
            **kwargs: Input parameters
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required parameters
            if 'period' not in kwargs:
                logger.error("Period parameter is required")
                return False
            
            # Validate period format
            valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            if kwargs['period'] not in valid_periods:
                logger.error(f"Invalid period: {kwargs['period']}. Valid periods: {valid_periods}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Component input validation failed: {e}")
            return False
