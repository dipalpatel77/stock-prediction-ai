#!/usr/bin/env python3
"""
Async Pipeline Orchestrator
Advanced async pipeline orchestration with real-time streaming and WebSocket integration
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import optimized components
from .data_processor import DataProcessor
from ..services.database_manager import DatabaseManager
from ..services.api_coordinator import APICoordinator
from ..services.angel_one_manager import AngelOneManager

logger = logging.getLogger(__name__)

class AsyncPipelineOrchestrator:
    """
    Advanced async pipeline orchestrator with real-time streaming and WebSocket integration
    
    Features:
    - Async/await support for all I/O operations
    - Real-time data streaming
    - WebSocket integration
    - Advanced parallel processing
    - Memory optimization
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {
                'enable_async': True,
                'enable_streaming': True,
                'enable_websocket': True,
                'max_concurrent_tasks': 10,
                'streaming_chunk_size': 100,
                'websocket_timeout': 30,
                'memory_limit_mb': 1024
            }
        
        self.config = config
        self.components = {}
        self.active_streams = {}
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'streaming_sessions': 0,
            'websocket_connections': 0,
            'total_processing_time': 0
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Async Pipeline Orchestrator initialized with advanced features")
    
    def _initialize_components(self):
        """Initialize async components"""
        try:
            # Initialize data processor
            self.components['data_processor'] = DataProcessor(config=self.config)
            
            # Initialize database manager
            self.components['database_manager'] = DatabaseManager(config=self.config)
            
            # Initialize API coordinator
            self.components['api_coordinator'] = APICoordinator(max_workers=4)
            
            # Initialize Angel One manager if needed
            if self.config.get('enable_angel_one', False):
                self.components['angel_one_manager'] = AngelOneManager(self.config)
            
            logger.info("Async components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize async components: {e}")
            raise e
    
    async def async_run_analysis(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async run comprehensive analysis pipeline
        
        Args:
            ticker: Stock ticker symbol
            config: Analysis configuration
            
        Returns:
            Analysis results
        """
        try:
            start_time = time.time()
            logger.info(f"Starting async analysis for {ticker}")
            
            # Create async tasks for parallel execution
            tasks = []
            
            # Data loading task
            data_task = asyncio.create_task(
                self._async_load_data(ticker, config)
            )
            tasks.append(('data_loading', data_task))
            
            # Economic data task
            economic_task = asyncio.create_task(
                self._async_load_economic_data(ticker, config)
            )
            tasks.append(('economic_data', economic_task))
            
            # Market data task
            market_task = asyncio.create_task(
                self._async_load_market_data(ticker, config)
            )
            tasks.append(('market_data', market_task))
            
            # Currency data task
            currency_task = asyncio.create_task(
                self._async_load_currency_data(ticker, config)
            )
            tasks.append(('currency_data', currency_task))
            
            # Execute all tasks concurrently
            results = {}
            for task_name, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=30)
                    results[task_name] = result
                    
                    if result.get('success'):
                        self.performance_metrics['completed_tasks'] += 1
                    else:
                        self.performance_metrics['failed_tasks'] += 1
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task_name} timed out")
                    results[task_name] = {'success': False, 'error': 'Timeout'}
                    self.performance_metrics['failed_tasks'] += 1
                    
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    results[task_name] = {'success': False, 'error': str(e)}
                    self.performance_metrics['failed_tasks'] += 1
                
                self.performance_metrics['total_tasks'] += 1
            
            # Process data if loading was successful
            if results.get('data_loading', {}).get('success'):
                processed_data = await self._async_process_data(
                    results['data_loading']['data'], 
                    ticker, 
                    config
                )
                results['processed_data'] = {
                    'success': True,
                    'data': processed_data,
                    'records': len(processed_data)
                }
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics['total_processing_time'] += execution_time
            
            return {
                'success': True,
                'ticker': ticker,
                'results': results,
                'execution_time': execution_time,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Async analysis failed for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
    
    async def _async_load_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async load stock data"""
        try:
            logger.info(f"Async loading data for {ticker}")
            
            # Use API coordinator for async data loading
            api_coordinator = self.components['api_coordinator']
            result = await api_coordinator.async_coordinate_parallel_loading(ticker, config)
            
            return result
            
        except Exception as e:
            logger.error(f"Async data loading failed for {ticker}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _async_load_economic_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async load economic data"""
        try:
            logger.info(f"Async loading economic data for {ticker}")
            
            # Simulate async economic data loading
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                'success': True,
                'data': f'Economic data for {ticker}',
                'source': 'economic'
            }
            
        except Exception as e:
            logger.error(f"Async economic data loading failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _async_load_market_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async load market data"""
        try:
            logger.info(f"Async loading market data for {ticker}")
            
            # Simulate async market data loading
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                'success': True,
                'data': f'Market data for {ticker}',
                'source': 'market'
            }
            
        except Exception as e:
            logger.error(f"Async market data loading failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _async_load_currency_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async load currency data"""
        try:
            logger.info(f"Async loading currency data for {ticker}")
            
            # Simulate async currency data loading
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                'success': True,
                'data': f'Currency data for {ticker}',
                'source': 'currency'
            }
            
        except Exception as e:
            logger.error(f"Async currency data loading failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _async_process_data(self, data: pd.DataFrame, ticker: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Async process data"""
        try:
            logger.info(f"Async processing data for {ticker}")
            
            # Use data processor for async processing
            data_processor = self.components['data_processor']
            processed_data = await data_processor.async_process_data(data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Async data processing failed for {ticker}: {e}")
            raise e
    
    async def start_realtime_stream(self, ticker: str, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Start real-time data streaming
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Yields:
            Real-time data updates
        """
        try:
            logger.info(f"Starting real-time stream for {ticker}")
            
            # Start WebSocket stream
            api_coordinator = self.components['api_coordinator']
            async for update in api_coordinator.start_websocket_stream(ticker, config):
                try:
                    # Process real-time update
                    processed_update = await self._async_process_realtime_update(update, ticker)
                    
                    # Update metrics
                    self.performance_metrics['streaming_sessions'] += 1
                    
                    yield processed_update
                    
                except Exception as e:
                    logger.error(f"Real-time update processing failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Real-time streaming failed for {ticker}: {e}")
            yield {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
    
    async def _async_process_realtime_update(self, update: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Process real-time update"""
        try:
            # Add processing timestamp
            update['processed_at'] = datetime.now().isoformat()
            update['ticker'] = ticker
            
            # Add any additional processing logic here
            if 'data' in update:
                # Process the data if needed
                pass
            
            return update
            
        except Exception as e:
            logger.error(f"Real-time update processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
    
    async def async_batch_analysis(self, tickers: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async batch analysis for multiple tickers
        
        Args:
            tickers: List of stock ticker symbols
            config: Analysis configuration
            
        Returns:
            Batch analysis results
        """
        try:
            logger.info(f"Starting async batch analysis for {len(tickers)} tickers")
            start_time = time.time()
            
            # Create async tasks for all tickers
            tasks = []
            for ticker in tickers:
                task = asyncio.create_task(
                    self.async_run_analysis(ticker, config)
                )
                tasks.append((ticker, task))
            
            # Execute all analyses concurrently
            results = {}
            for ticker, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=60)  # 1 minute timeout per ticker
                    results[ticker] = result
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Batch analysis timed out for {ticker}")
                    results[ticker] = {'success': False, 'error': 'Timeout'}
                    
                except Exception as e:
                    logger.error(f"Batch analysis failed for {ticker}: {e}")
                    results[ticker] = {'success': False, 'error': str(e)}
            
            # Calculate batch statistics
            successful = sum(1 for r in results.values() if r.get('success'))
            failed = len(tickers) - successful
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'total_tickers': len(tickers),
                'successful': successful,
                'failed': failed,
                'execution_time': execution_time,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Async batch analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def async_get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive async performance report"""
        try:
            # Get performance metrics from all components
            component_metrics = {}
            
            # Database metrics
            if 'database_manager' in self.components:
                db_metrics = self.components['database_manager'].get_performance_report()
                component_metrics['database'] = db_metrics
            
            # API metrics
            if 'api_coordinator' in self.components:
                api_metrics = self.components['api_coordinator'].get_performance_metrics()
                component_metrics['api'] = api_metrics
            
            # Data processor metrics
            if 'data_processor' in self.components:
                data_metrics = await self.components['data_processor'].async_get_performance_report()
                component_metrics['data_processor'] = data_metrics
            
            return {
                'orchestrator_metrics': self.performance_metrics,
                'component_metrics': component_metrics,
                'timestamp': datetime.now().isoformat(),
                'async_enabled': True,
                'streaming_enabled': self.config.get('enable_streaming', False),
                'websocket_enabled': self.config.get('enable_websocket', False)
            }
            
        except Exception as e:
            logger.error(f"Async performance report generation failed: {e}")
            return {'error': str(e)}
    
    async def async_cleanup(self):
        """Cleanup async resources"""
        try:
            logger.info("Cleaning up async resources")
            
            # Close API coordinator resources
            if 'api_coordinator' in self.components:
                await self.components['api_coordinator'].close_async_resources()
            
            # Close active streams
            for ticker, stream in self.active_streams.items():
                try:
                    await stream.aclose()
                except Exception as e:
                    logger.warning(f"Failed to close stream for {ticker}: {e}")
            
            self.active_streams.clear()
            
            logger.info("Async resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Async cleanup failed: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.async_cleanup()
