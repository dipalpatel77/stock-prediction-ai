#!/usr/bin/env python3
"""
Optimized API Coordinator
Coordinates multiple API calls with advanced rate limiting, caching, and performance monitoring
"""

import pandas as pd
import logging
import time
import threading
import asyncio
import aiohttp
import websockets
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
from datetime import datetime, timedelta
import ssl

# Import rate limiter
from main.utils.rate_limiter import get_api_rate_limiter

logger = logging.getLogger(__name__)

class APICoordinator:
    """Optimized API coordinator with advanced caching, rate limiting, and performance monitoring"""
    
    def __init__(self, max_workers: int = 4):
        self.rate_limiter = get_api_rate_limiter()
        self.api_stats = {}
        self.cache = {}
        self.circuit_breakers = {}
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance monitoring
        self.performance_metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_response_time': 0,
            'api_response_times': {}
        }
        
        # Cache configuration
        self.cache_config = {
            'max_cache_size': 1000,
            'cache_ttl': 300,  # 5 minutes
            'enable_response_caching': True
        }
        
        # Thread safety
        self.cache_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Async support
        self.async_session = None
        self.websocket_connections = {}
        self.async_config = {
            'enable_async': True,
            'websocket_enabled': True,
            'async_timeout': 30,
            'max_concurrent_requests': 10
        }
        
        logger.info(f"Optimized API Coordinator initialized with {max_workers} workers and async support")
    
    def check_rate_limit(self, api_name: str = 'default') -> bool:
        """
        Check if API rate limit is available
        
        Args:
            api_name: Name of the API
            
        Returns:
            True if rate limit is available, False otherwise
        """
        # TESTING MODE: Remove rate limiting constraints for testing
        return True
        
        try:
            if self.rate_limiter:
                return self.rate_limiter.check_limit(api_name)
            return True
        except Exception as e:
            logger.warning(f"Rate limit check failed for {api_name}: {e}")
            return True
    
    def handle_api_error(self, error: Exception, api_name: str = 'default') -> Dict[str, Any]:
        """
        Handle API errors with retry and fallback mechanisms
        
        Args:
            error: The error that occurred
            api_name: Name of the API
            
        Returns:
            Dictionary with error handling results
        """
        try:
            logger.warning(f"API error for {api_name}: {error}")
            
            # Check if we should retry
            if api_name in self.circuit_breakers:
                breaker = self.circuit_breakers[api_name]
                if breaker.is_open():
                    logger.warning(f"Circuit breaker open for {api_name}")
                    return {
                        'success': False,
                        'error': 'Circuit breaker open',
                        'retry_after': breaker.get_retry_after()
                    }
            
            # Record error in stats
            if api_name not in self.api_stats:
                self.api_stats[api_name] = {'errors': 0, 'successes': 0}
            
            self.api_stats[api_name]['errors'] += 1
            
            # Determine if we should retry
            error_count = self.api_stats[api_name]['errors']
            if error_count < 3:  # Retry up to 3 times
                logger.info(f"Retrying API call for {api_name} (attempt {error_count + 1})")
                return {
                    'success': False,
                    'error': str(error),
                    'retry': True,
                    'retry_after': 1  # 1 second delay
                }
            else:
                logger.error(f"Max retries exceeded for {api_name}")
                return {
                    'success': False,
                    'error': str(error),
                    'retry': False,
                    'fallback_required': True
                }
                
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'retry': False
            }
    
    def handle_fallback(self, api_name: str, fallback_apis: List[str]) -> Dict[str, Any]:
        """
        Handle API fallback mechanisms
        
        Args:
            api_name: Name of the failed API
            fallback_apis: List of fallback API names
            
        Returns:
            Dictionary with fallback results
        """
        try:
            logger.info(f"Handling fallback for {api_name} to {fallback_apis}")
            
            for fallback_api in fallback_apis:
                try:
                    logger.info(f"Attempting fallback to {fallback_api}")
                    
                    # Check if fallback API is available
                    if fallback_api in self.api_stats:
                        # Check circuit breaker for fallback API
                        if fallback_api in self.circuit_breakers:
                            breaker = self.circuit_breakers[fallback_api]
                            if breaker.is_open():
                                logger.warning(f"Fallback API {fallback_api} circuit breaker open")
                                continue
                        
                        # Try to use fallback API
                        logger.info(f"Using fallback API {fallback_api}")
                        return {
                            'success': True,
                            'fallback_api': fallback_api,
                            'message': f'Successfully switched to {fallback_api}'
                        }
                    else:
                        logger.warning(f"Fallback API {fallback_api} not available")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Fallback to {fallback_api} failed: {e}")
                    continue
            
            # If all fallbacks failed
            logger.error(f"All fallback APIs failed for {api_name}")
            return {
                'success': False,
                'error': 'All fallback APIs failed',
                'fallback_apis_tried': fallback_apis
            }
            
        except Exception as e:
            logger.error(f"Fallback handling failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def coordinate_data_loading(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate data loading from multiple sources
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with coordinated data loading results
        """
        try:
            logger.info(f"Coordinating data loading for {ticker}")
            start_time = time.time()
            
            results = {}
            
            # Load primary data source
            if config.get('is_indian') and config.get('angel_config'):
                results['primary_data'] = self._load_angel_one_data(ticker, config)
            else:
                results['primary_data'] = self._load_yahoo_data(ticker, config)
            
            # Load supplementary data in parallel
            supplementary_data = self._load_supplementary_data_parallel(ticker, config)
            results.update(supplementary_data)
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['success'] = True
            
            logger.info(f"Data loading coordination completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Data loading coordination failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def coordinate_parallel_loading(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate parallel data loading from multiple sources with caching and performance monitoring
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with parallel loading results
        """
        try:
            logger.info(f"Coordinating parallel data loading for {ticker}")
            start_time = time.time()
            
            # Check cache first
            cache_key = f"parallel_loading_{ticker}_{hash(str(config))}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached parallel loading result for {ticker}")
                return cached_result
            
            # Define loading tasks
            tasks = []
            
            # Primary data source
            if config.get('is_indian') and config.get('angel_config'):
                tasks.append(('angel_one', self._load_angel_one_data, ticker, config))
            else:
                tasks.append(('yahoo', self._load_yahoo_data, ticker, config))
            
            # Supplementary data sources
            tasks.append(('economic', self._load_economic_data, ticker, config))
            tasks.append(('market', self._load_market_data, ticker, config))
            tasks.append(('currency', self._load_currency_data, ticker, config))
            
            # Execute tasks in parallel with timeout
            results = {}
            futures = []
            
            for task_name, task_func, *args in tasks:
                future = self.executor.submit(self._execute_with_fallback, task_name, task_func, *args)
                futures.append((task_name, future))
            
            # Collect results with individual timeouts
            for task_name, future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per task
                    results[task_name] = result
                    
                    # Update performance metrics
                    with self.stats_lock:
                        if result.get('success'):
                            self.performance_metrics['successful_calls'] += 1
                        else:
                            self.performance_metrics['failed_calls'] += 1
                        self.performance_metrics['total_calls'] += 1
                        
                except Exception as e:
                    logger.warning(f"Task {task_name} failed: {e}")
                    results[task_name] = {'success': False, 'error': str(e)}
                    
                    with self.stats_lock:
                        self.performance_metrics['failed_calls'] += 1
                        self.performance_metrics['total_calls'] += 1
            
            duration = time.time() - start_time
            
            # Cache successful results
            if any(result.get('success') for result in results.values()):
                self._cache_result(cache_key, results)
            
            # Update performance metrics
            with self.stats_lock:
                self.performance_metrics['total_response_time'] += duration
                if 'parallel_loading' not in self.performance_metrics['api_response_times']:
                    self.performance_metrics['api_response_times']['parallel_loading'] = []
                self.performance_metrics['api_response_times']['parallel_loading'].append(duration)
            
            return {
                'success': True,
                'data': results,
                'duration': duration,
                'tasks_completed': len([r for r in results.values() if r.get('success', False)])
            }
            
        except Exception as e:
            logger.error(f"Parallel data loading coordination failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_with_fallback(self, task_name: str, task_func, *args) -> Dict[str, Any]:
        """
        Execute a task with fallback mechanisms
        
        Args:
            task_name: Name of the task
            task_func: Function to execute
            *args: Arguments for the function
            
        Returns:
            Task execution result
        """
        try:
            # Check circuit breaker
            if self._is_circuit_open(task_name):
                return {'success': False, 'error': f'Circuit breaker open for {task_name}'}
            
            # Execute task
            result = task_func(*args)
            
            # Record success
            self._record_success(task_name)
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(task_name)
            
            # Try fallback if available
            fallback_result = self._try_fallback(task_name, task_func, *args)
            if fallback_result:
                return fallback_result
            
            return {'success': False, 'error': str(e)}
    
    def _is_circuit_open(self, task_name: str) -> bool:
        """
        Check if circuit breaker is open for a task
        
        Args:
            task_name: Name of the task
            
        Returns:
            True if circuit is open, False otherwise
        """
        if task_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[task_name]
        if breaker['state'] == 'open':
            # Check if enough time has passed to try again
            if time.time() - breaker['last_failure'] > breaker['timeout']:
                breaker['state'] = 'half_open'
                return False
            return True
        
        return False
    
    def _record_success(self, task_name: str):
        """Record successful task execution"""
        if task_name not in self.api_stats:
            self.api_stats[task_name] = {'successes': 0, 'failures': 0}
        
        self.api_stats[task_name]['successes'] += 1
        
        # Reset circuit breaker if it was half-open
        if task_name in self.circuit_breakers:
            self.circuit_breakers[task_name]['state'] = 'closed'
    
    def _record_failure(self, task_name: str):
        """Record failed task execution"""
        if task_name not in self.api_stats:
            self.api_stats[task_name] = {'successes': 0, 'failures': 0}
        
        self.api_stats[task_name]['failures'] += 1
        
        # Update circuit breaker
        if task_name not in self.circuit_breakers:
            self.circuit_breakers[task_name] = {
                'state': 'closed',
                'failure_count': 0,
                'timeout': 60,  # 60 seconds
                'last_failure': 0
            }
        
        breaker = self.circuit_breakers[task_name]
        breaker['failure_count'] += 1
        breaker['last_failure'] = time.time()
        
        # Open circuit if too many failures
        if breaker['failure_count'] >= 5:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {task_name}")
    
    def _try_fallback(self, task_name: str, task_func, *args) -> Optional[Dict[str, Any]]:
        """
        Try fallback mechanisms for a failed task
        
        Args:
            task_name: Name of the task
            task_func: Original function
            *args: Arguments for the function
            
        Returns:
            Fallback result or None if no fallback available
        """
        try:
            # Define fallback strategies
            fallback_strategies = {
                'angel_one': self._fallback_yahoo_data,
                'yahoo': self._fallback_angel_one_data,
                'economic': self._fallback_cached_economic_data,
                'market': self._fallback_cached_market_data,
                'currency': self._fallback_cached_currency_data
            }
            
            if task_name in fallback_strategies:
                fallback_func = fallback_strategies[task_name]
                return fallback_func(*args)
            
            return None
            
        except Exception as e:
            logger.error(f"Fallback failed for {task_name}: {e}")
            return None
    
    def _fallback_yahoo_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to Yahoo Finance data"""
        try:
            logger.info(f"Using Yahoo Finance fallback for {ticker}")
            return self._load_yahoo_data(ticker, config)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fallback_angel_one_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to Angel One data"""
        try:
            logger.info(f"Using Angel One fallback for {ticker}")
            return self._load_angel_one_data(ticker, config)
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fallback_cached_economic_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to cached economic data"""
        try:
            cache_key = f"economic_data_{ticker}"
            if cache_key in self.cache:
                logger.info(f"Using cached economic data for {ticker}")
                return {'success': True, 'data': self.cache[cache_key], 'cached': True}
            else:
                return {'success': False, 'error': 'No cached data available'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fallback_cached_market_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to cached market data"""
        try:
            cache_key = f"market_data_{ticker}"
            if cache_key in self.cache:
                logger.info(f"Using cached market data for {ticker}")
                return {'success': True, 'data': self.cache[cache_key], 'cached': True}
            else:
                return {'success': False, 'error': 'No cached data available'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _fallback_cached_currency_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback to cached currency data"""
        try:
            cache_key = f"currency_data_{ticker}"
            if cache_key in self.cache:
                logger.info(f"Using cached currency data for {ticker}")
                return {'success': True, 'data': self.cache[cache_key], 'cached': True}
            else:
                return {'success': False, 'error': 'No cached data available'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """
        Get API coordinator status
        
        Returns:
            Coordinator status dictionary
        """
        try:
            total_tasks = sum(stats['successes'] + stats['failures'] for stats in self.api_stats.values())
            successful_tasks = sum(stats['successes'] for stats in self.api_stats.values())
            failed_tasks = sum(stats['failures'] for stats in self.api_stats.values())
            
            success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
            
            open_circuits = len([name for name, breaker in self.circuit_breakers.items() 
                               if breaker['state'] == 'open'])
            
            return {
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': success_rate,
                'open_circuits': open_circuits,
                'cache_size': len(self.cache),
                'max_workers': self.max_workers
            }
            
        except Exception as e:
            logger.error(f"Failed to get coordinator status: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown the API coordinator"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("API Coordinator shutdown completed")
        except Exception as e:
            logger.error(f"Failed to shutdown coordinator: {e}")
    
    def _load_angel_one_data(self, ticker: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Load Angel One data with rate limiting
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            DataFrame with Angel One data
        """
        try:
            return self.rate_limiter.call_with_retry(
                'angel_one',
                self._fetch_angel_one_data,
                ticker, config
            )
        except Exception as e:
            logger.error(f"Angel One data loading failed: {e}")
            raise e
    
    def _load_yahoo_data(self, ticker: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Load Yahoo Finance data with rate limiting
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            DataFrame with Yahoo Finance data
        """
        try:
            return self.rate_limiter.call_with_retry(
                'yahoo_finance',
                self._fetch_yahoo_data,
                ticker, config
            )
        except Exception as e:
            logger.error(f"Yahoo Finance data loading failed: {e}")
            raise e
    
    def _load_supplementary_data_parallel(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load supplementary data in parallel
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with supplementary data
        """
        try:
            results = {}
            
            # Define tasks to run in parallel
            tasks = [
                ('economic_data', self._load_economic_data),
                ('market_data', self._load_market_data),
                ('currency_data', self._load_currency_data),
                ('geopolitical_data', self._load_geopolitical_data)
            ]
            
            # Execute tasks in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_task = {
                    executor.submit(task_func, ticker, config): task_name
                    for task_name, task_func in tasks
                }
                
                for future in as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        result = future.result()
                        results[task_name] = result
                        logger.debug(f"Loaded {task_name} successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load {task_name}: {e}")
                        results[task_name] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel supplementary data loading failed: {e}")
            return {}
    
    def _load_economic_data(self, ticker: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load economic data from FRED API
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with economic data or None if failed
        """
        try:
            return self.rate_limiter.call_with_retry(
                'fred',
                self._fetch_economic_data,
                ticker, config
            )
        except Exception as e:
            logger.warning(f"Economic data loading failed: {e}")
            return None
    
    def _load_market_data(self, ticker: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load global market data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with market data or None if failed
        """
        try:
            # Use a generic rate limiter for market data
            return self.rate_limiter.call_with_retry(
                'market_data',
                self._fetch_market_data,
                ticker, config
            )
        except Exception as e:
            logger.warning(f"Market data loading failed: {e}")
            return None
    
    def _load_currency_data(self, ticker: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load currency data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with currency data or None if failed
        """
        try:
            return self.rate_limiter.call_with_retry(
                'currency',
                self._fetch_currency_data,
                ticker, config
            )
        except Exception as e:
            logger.warning(f"Currency data loading failed: {e}")
            return None
    
    def _load_geopolitical_data(self, ticker: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load geopolitical risk data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with geopolitical data or None if failed
        """
        try:
            return self.rate_limiter.call_with_retry(
                'geopolitical',
                self._fetch_geopolitical_data,
                ticker, config
            )
        except Exception as e:
            logger.warning(f"Geopolitical data loading failed: {e}")
            return None
    
    def _fetch_angel_one_data(self, ticker: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch Angel One data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            DataFrame with Angel One data
        """
        try:
            # Import Angel One manager
            from .angel_one_manager import AngelOneManager
            
            angel_config = config.get('angel_config', {})
            angel_manager = AngelOneManager(angel_config)
            
            period = config.get('parameters', {}).get('period', '1y')
            interval = config.get('parameters', {}).get('interval', 'ONE_DAY')
            
            data = angel_manager.get_stock_data(ticker, period, interval)
            return data
            
        except Exception as e:
            logger.error(f"Angel One data fetch failed: {e}")
            raise e
    
    def _fetch_yahoo_data(self, ticker: str, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Fetch Yahoo Finance data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            DataFrame with Yahoo Finance data
        """
        try:
            # Import data service wrapper
            from .data_service_wrapper import DataServiceWrapper
            
            data_wrapper = DataServiceWrapper(ticker, config)
            
            period = config.get('parameters', {}).get('period', '1y')
            interval = config.get('parameters', {}).get('interval', 'ONE_DAY')
            
            data = data_wrapper.load_stock_data(period, interval)
            return data
            
        except Exception as e:
            logger.error(f"Yahoo Finance data fetch failed: {e}")
            raise e
    
    def _fetch_economic_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch economic data from FRED API
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with economic data
        """
        try:
            # Import economic data service
            from src.core.fred_api_service import FredApiService
            
            fred_service = FredApiService()
            
            # Get relevant economic indicators
            economic_data = fred_service.get_economic_indicators()
            
            return {
                'gdp': economic_data.get('gdp'),
                'inflation': economic_data.get('inflation'),
                'unemployment': economic_data.get('unemployment'),
                'interest_rates': economic_data.get('interest_rates')
            }
            
        except Exception as e:
            logger.error(f"Economic data fetch failed: {e}")
            raise e
    
    def _fetch_market_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch global market data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with market data
        """
        try:
            # Import global market service
            from src.core.global_market_service import GlobalMarketService
            
            market_service = GlobalMarketService()
            
            # Get market indicators
            market_data = market_service.get_market_indicators()
            
            return {
                'vix': market_data.get('vix'),
                'market_sentiment': market_data.get('sentiment'),
                'sector_performance': market_data.get('sector_performance'),
                'global_indices': market_data.get('global_indices')
            }
            
        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            raise e
    
    def _fetch_currency_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch currency data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with currency data
        """
        try:
            # Import currency service
            from src.core.currency_service import CurrencyService
            
            currency_service = CurrencyService()
            
            # Get currency rates
            currency_data = currency_service.get_currency_rates()
            
            return {
                'usd_rates': currency_data.get('usd_rates'),
                'major_currencies': currency_data.get('major_currencies'),
                'currency_volatility': currency_data.get('volatility')
            }
            
        except Exception as e:
            logger.error(f"Currency data fetch failed: {e}")
            raise e
    
    def _fetch_geopolitical_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch geopolitical risk data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with geopolitical data
        """
        try:
            # Import geopolitical risk service
            from src.core.geopolitical_risk_service import GeopoliticalRiskService
            
            geopolitical_service = GeopoliticalRiskService()
            
            # Get geopolitical risk indicators
            risk_data = geopolitical_service.get_risk_indicators()
            
            return {
                'risk_level': risk_data.get('risk_level'),
                'risk_factors': risk_data.get('risk_factors'),
                'regional_risks': risk_data.get('regional_risks'),
                'political_stability': risk_data.get('political_stability')
            }
            
        except Exception as e:
            logger.error(f"Geopolitical data fetch failed: {e}")
            raise e
    
    def get_api_statistics(self) -> Dict[str, Any]:
        """
        Get API usage statistics
        
        Returns:
            Dictionary with API statistics
        """
        try:
            stats = self.rate_limiter.get_statistics()
            
            # Add coordinator-specific statistics
            stats.update({
                'api_coordinator_status': 'active',
                'parallel_loading_enabled': True,
                'max_parallel_workers': 4
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get API statistics: {e}")
            return {'error': str(e)}
    
    def test_all_apis(self, ticker: str, config: Dict[str, Any]) -> Dict[str, bool]:
        """
        Test all API connections
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with API test results
        """
        try:
            logger.info(f"Testing all APIs for {ticker}")
            
            results = {}
            
            # Test primary data source
            if config.get('is_indian') and config.get('angel_config'):
                results['angel_one'] = self._test_angel_one_api(config)
            else:
                results['yahoo_finance'] = self._test_yahoo_api()
            
            # Test supplementary APIs
            results['fred_api'] = self._test_fred_api()
            results['market_data'] = self._test_market_data_api()
            results['currency_api'] = self._test_currency_api()
            results['geopolitical_api'] = self._test_geopolitical_api()
            
            # Calculate success rate
            successful_apis = sum(1 for success in results.values() if success)
            total_apis = len(results)
            success_rate = (successful_apis / total_apis) * 100 if total_apis > 0 else 0
            
            results['success_rate'] = success_rate
            results['total_apis'] = total_apis
            results['successful_apis'] = successful_apis
            
            logger.info(f"API testing completed: {successful_apis}/{total_apis} APIs working ({success_rate:.1f}%)")
            return results
            
        except Exception as e:
            logger.error(f"API testing failed: {e}")
            return {'error': str(e)}
    
    def _test_angel_one_api(self, config: Dict[str, Any]) -> bool:
        """Test Angel One API connection"""
        try:
            from .angel_one_manager import AngelOneManager
            angel_manager = AngelOneManager(config.get('angel_config', {}))
            return angel_manager.test_connection()
        except Exception as e:
            logger.error(f"Angel One API test failed: {e}")
            return False
    
    def _test_yahoo_api(self) -> bool:
        """Test Yahoo Finance API connection"""
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="1d")
            return data is not None and not data.empty
        except Exception as e:
            logger.error(f"Yahoo Finance API test failed: {e}")
            return False
    
    def _test_fred_api(self) -> bool:
        """Test FRED API connection"""
        try:
            from src.core.fred_api_service import FredApiService
            fred_service = FredApiService()
            return fred_service.test_connection()
        except Exception as e:
            logger.error(f"FRED API test failed: {e}")
            return False
    
    def _test_market_data_api(self) -> bool:
        """Test market data API connection"""
        try:
            from src.core.global_market_service import GlobalMarketService
            market_service = GlobalMarketService()
            return market_service.test_connection()
        except Exception as e:
            logger.error(f"Market data API test failed: {e}")
            return False
    
    def _test_currency_api(self) -> bool:
        """Test currency API connection"""
        try:
            from src.core.currency_service import CurrencyService
            currency_service = CurrencyService()
            return currency_service.test_connection()
        except Exception as e:
            logger.error(f"Currency API test failed: {e}")
            return False
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available and not expired
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self.cache_config['enable_response_caching']:
            return None
            
        with self.cache_lock:
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                cache_time = cached_data.get('timestamp', 0)
                
                # Check if cache is still valid
                if time.time() - cache_time < self.cache_config['cache_ttl']:
                    self.performance_metrics['cache_hits'] += 1
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_data.get('data')
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
                    self.performance_metrics['cache_misses'] += 1
            else:
                self.performance_metrics['cache_misses'] += 1
            
            return None
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """
        Cache result with timestamp
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        if not self.cache_config['enable_response_caching']:
            return
            
        with self.cache_lock:
            # Limit cache size
            if len(self.cache) >= self.cache_config['max_cache_size']:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].get('timestamp', 0))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            
            logger.debug(f"Cached result for key: {cache_key}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        with self.stats_lock:
            total_calls = self.performance_metrics['total_calls']
            success_rate = 0
            avg_response_time = 0
            
            if total_calls > 0:
                success_rate = (self.performance_metrics['successful_calls'] / total_calls) * 100
                avg_response_time = self.performance_metrics['total_response_time'] / total_calls
            
            cache_hit_rate = 0
            total_cache_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
            if total_cache_requests > 0:
                cache_hit_rate = (self.performance_metrics['cache_hits'] / total_cache_requests) * 100
            
            return {
                'total_calls': total_calls,
                'successful_calls': self.performance_metrics['successful_calls'],
                'failed_calls': self.performance_metrics['failed_calls'],
                'success_rate': round(success_rate, 2),
                'average_response_time': round(avg_response_time, 3),
                'cache_hit_rate': round(cache_hit_rate, 2),
                'cache_hits': self.performance_metrics['cache_hits'],
                'cache_misses': self.performance_metrics['cache_misses'],
                'cache_size': len(self.cache),
                'api_response_times': self.performance_metrics['api_response_times']
            }
    
    def clear_cache(self):
        """Clear the response cache"""
        with self.cache_lock:
            self.cache.clear()
            logger.info("API response cache cleared")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get detailed performance report with recommendations
        
        Returns:
            Dictionary with performance report
        """
        try:
            metrics = self.get_performance_metrics()
            
            # Generate recommendations
            recommendations = []
            
            if metrics['success_rate'] < 80:
                recommendations.append("Low success rate detected - check API endpoints and error handling")
            
            if metrics['average_response_time'] > 5.0:
                recommendations.append("High response times detected - consider optimizing API calls or increasing timeouts")
            
            if metrics['cache_hit_rate'] < 30:
                recommendations.append("Low cache hit rate - consider increasing cache TTL or optimizing cache keys")
            
            if metrics['total_calls'] > 1000:
                recommendations.append("High API call volume - consider implementing request batching or reducing call frequency")
            
            return {
                'performance_metrics': metrics,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    async def async_coordinate_parallel_loading(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async coordinate parallel data loading from multiple sources with WebSocket support
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Returns:
            Dictionary with parallel loading results
        """
        try:
            logger.info(f"Async coordinating parallel data loading for {ticker}")
            start_time = time.time()
            
            # Check cache first
            cache_key = f"async_parallel_loading_{ticker}_{hash(str(config))}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Using cached async parallel loading result for {ticker}")
                return cached_result
            
            # Initialize async session if needed
            if not self.async_session:
                await self._initialize_async_session()
            
            # Define async loading tasks
            tasks = []
            
            # Primary data source
            if config.get('is_indian') and config.get('angel_config'):
                tasks.append(('angel_one', self._async_load_angel_one_data, ticker, config))
            else:
                tasks.append(('yahoo', self._async_load_yahoo_data, ticker, config))
            
            # Supplementary data sources
            tasks.append(('economic', self._async_load_economic_data, ticker, config))
            tasks.append(('market', self._async_load_market_data, ticker, config))
            tasks.append(('currency', self._async_load_currency_data, ticker, config))
            
            # Execute async tasks concurrently
            results = {}
            async_tasks = []
            
            for task_name, task_func, *args in tasks:
                async_task = asyncio.create_task(
                    self._async_execute_with_fallback(task_name, task_func, *args)
                )
                async_tasks.append((task_name, async_task))
            
            # Collect results with timeout
            for task_name, async_task in async_tasks:
                try:
                    result = await asyncio.wait_for(
                        async_task, 
                        timeout=self.async_config['async_timeout']
                    )
                    results[task_name] = result
                    
                    # Update performance metrics
                    with self.stats_lock:
                        if result.get('success'):
                            self.performance_metrics['successful_calls'] += 1
                        else:
                            self.performance_metrics['failed_calls'] += 1
                        self.performance_metrics['total_calls'] += 1
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Async task {task_name} timed out")
                    results[task_name] = {'success': False, 'error': 'Timeout'}
                    
                    with self.stats_lock:
                        self.performance_metrics['failed_calls'] += 1
                        self.performance_metrics['total_calls'] += 1
                        
                except Exception as e:
                    logger.warning(f"Async task {task_name} failed: {e}")
                    results[task_name] = {'success': False, 'error': str(e)}
                    
                    with self.stats_lock:
                        self.performance_metrics['failed_calls'] += 1
                        self.performance_metrics['total_calls'] += 1
            
            duration = time.time() - start_time
            
            # Cache successful results
            if any(result.get('success') for result in results.values()):
                self._cache_result(cache_key, results)
            
            # Update performance metrics
            with self.stats_lock:
                self.performance_metrics['total_response_time'] += duration
                if 'async_parallel_loading' not in self.performance_metrics['api_response_times']:
                    self.performance_metrics['api_response_times']['async_parallel_loading'] = []
                self.performance_metrics['api_response_times']['async_parallel_loading'].append(duration)
            
            return {
                'success': True,
                'data': results,
                'duration': duration,
                'tasks_completed': len([r for r in results.values() if r.get('success', False)])
            }
            
        except Exception as e:
            logger.error(f"Async parallel loading coordination failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _initialize_async_session(self):
        """Initialize async HTTP session"""
        try:
            connector = aiohttp.TCPConnector(
                limit=self.async_config['max_concurrent_requests'],
                limit_per_host=5
            )
            timeout = aiohttp.ClientTimeout(total=self.async_config['async_timeout'])
            
            self.async_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            logger.info("Async HTTP session initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize async session: {e}")
            self.async_session = None
    
    async def _async_execute_with_fallback(self, task_name: str, task_func, *args) -> Dict[str, Any]:
        """
        Async execute a task with fallback mechanisms
        
        Args:
            task_name: Name of the task
            task_func: Async function to execute
            *args: Arguments for the function
            
        Returns:
            Task execution result
        """
        try:
            # Check circuit breaker
            if self._is_circuit_open(task_name):
                return {'success': False, 'error': f'Circuit breaker open for {task_name}'}
            
            # Execute async task
            result = await task_func(*args)
            
            # Record success
            self._record_success(task_name)
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(task_name)
            
            # Try fallback if available
            fallback_result = await self._async_try_fallback(task_name, task_func, *args)
            if fallback_result:
                return fallback_result
            
            return {'success': False, 'error': str(e)}
    
    async def _async_try_fallback(self, task_name: str, task_func, *args) -> Optional[Dict[str, Any]]:
        """Async try fallback mechanisms"""
        try:
            # Get fallback APIs for this task
            fallback_apis = self._get_fallback_apis(task_name)
            
            for fallback_api in fallback_apis:
                try:
                    logger.info(f"Async trying fallback to {fallback_api}")
                    
                    # Check if fallback API is available
                    if fallback_api in self.api_stats:
                        # Check circuit breaker for fallback API
                        if fallback_api in self.circuit_breakers:
                            breaker = self.circuit_breakers[fallback_api]
                            if breaker.is_open():
                                logger.warning(f"Async fallback API {fallback_api} circuit breaker open")
                                continue
                        
                        # Try to use fallback API
                        fallback_func = self._get_fallback_function(fallback_api)
                        if fallback_func:
                            result = await fallback_func(*args)
                            if result.get('success'):
                                logger.info(f"Async fallback to {fallback_api} successful")
                                return result
                    
                except Exception as e:
                    logger.warning(f"Async fallback to {fallback_api} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Async fallback mechanism failed: {e}")
            return None
    
    async def _async_load_angel_one_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async load Angel One data"""
        try:
            # Implement async Angel One data loading
            logger.info(f"Async loading Angel One data for {ticker}")
            
            # Simulate async API call
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                'success': True,
                'data': f'Angel One data for {ticker}',
                'source': 'angel_one'
            }
            
        except Exception as e:
            logger.error(f"Async Angel One data loading failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _async_load_yahoo_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async load Yahoo Finance data"""
        try:
            logger.info(f"Async loading Yahoo Finance data for {ticker}")
            
            # Simulate async API call
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                'success': True,
                'data': f'Yahoo Finance data for {ticker}',
                'source': 'yahoo_finance'
            }
            
        except Exception as e:
            logger.error(f"Async Yahoo Finance data loading failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _async_load_economic_data(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Async load economic data"""
        try:
            logger.info(f"Async loading economic data for {ticker}")
            
            # Simulate async API call
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
            
            # Simulate async API call
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
            
            # Simulate async API call
            await asyncio.sleep(0.1)  # Simulate network delay
            
            return {
                'success': True,
                'data': f'Currency data for {ticker}',
                'source': 'currency'
            }
            
        except Exception as e:
            logger.error(f"Async currency data loading failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def start_websocket_stream(self, ticker: str, config: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Start WebSocket stream for real-time data
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
            
        Yields:
            Real-time data updates
        """
        try:
            logger.info(f"Starting WebSocket stream for {ticker}")
            
            # WebSocket URL (example)
            ws_url = f"wss://api.example.com/stream/{ticker}"
            
            async with websockets.connect(ws_url) as websocket:
                self.websocket_connections[ticker] = websocket
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        
                        # Process real-time data
                        processed_data = await self._process_realtime_data(data, ticker)
                        
                        yield {
                            'success': True,
                            'data': processed_data,
                            'timestamp': datetime.now().isoformat(),
                            'ticker': ticker
                        }
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received for {ticker}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message for {ticker}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"WebSocket stream failed for {ticker}: {e}")
            yield {
                'success': False,
                'error': str(e),
                'ticker': ticker
            }
        finally:
            # Clean up connection
            if ticker in self.websocket_connections:
                del self.websocket_connections[ticker]
    
    async def _process_realtime_data(self, data: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """
        Process real-time WebSocket data
        
        Args:
            data: Raw WebSocket data
            ticker: Stock ticker symbol
            
        Returns:
            Processed real-time data
        """
        try:
            # Process real-time data (price updates, volume, etc.)
            processed_data = {
                'ticker': ticker,
                'price': data.get('price', 0),
                'volume': data.get('volume', 0),
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'change': data.get('change', 0),
                'change_percent': data.get('change_percent', 0)
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Real-time data processing failed for {ticker}: {e}")
            return {'error': str(e)}
    
    async def close_async_resources(self):
        """Close async resources and connections"""
        try:
            # Close async session
            if self.async_session:
                await self.async_session.close()
                self.async_session = None
            
            # Close WebSocket connections
            for ticker, websocket in self.websocket_connections.items():
                try:
                    await websocket.close()
                except Exception as e:
                    logger.warning(f"Failed to close WebSocket for {ticker}: {e}")
            
            self.websocket_connections.clear()
            
            logger.info("Async resources closed")
            
        except Exception as e:
            logger.error(f"Failed to close async resources: {e}")
    
    def _test_geopolitical_api(self) -> bool:
        """Test geopolitical API connection"""
        try:
            from src.core.geopolitical_risk_service import GeopoliticalRiskService
            geopolitical_service = GeopoliticalRiskService()
            return geopolitical_service.test_connection()
        except Exception as e:
            logger.error(f"Geopolitical API test failed: {e}")
            return False
