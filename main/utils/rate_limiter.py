#!/usr/bin/env python3
"""
API Rate Limiter
Provides rate limiting for API calls with exponential backoff
"""

import time
import threading
import logging
from typing import Dict, Optional, Callable, Any
from functools import wraps
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter with sliding window"""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60, 
                 burst_limit: int = 10):
        self.max_calls = max_calls
        self.time_window = time_window
        self.burst_limit = burst_limit
        
        # Thread-safe storage for call timestamps
        self.call_times: deque = deque()
        self.lock = threading.Lock()
        
        # Statistics
        self.total_calls = 0
        self.blocked_calls = 0
        self.last_reset = time.time()
    
    def _cleanup_old_calls(self, current_time: float):
        """Remove calls outside the time window"""
        while self.call_times and current_time - self.call_times[0] > self.time_window:
            self.call_times.popleft()
    
    def _check_burst_limit(self, current_time: float) -> bool:
        """Check if we're within burst limit (calls in last second)"""
        recent_calls = sum(1 for call_time in self.call_times 
                          if current_time - call_time < 1.0)
        return recent_calls < self.burst_limit
    
    def acquire(self, wait: bool = True) -> bool:
        """
        Acquire permission to make an API call
        
        Args:
            wait: Whether to wait if rate limit is exceeded
            
        Returns:
            True if permission granted, False otherwise
        """
        current_time = time.time()
        
        with self.lock:
            # Clean up old calls
            self._cleanup_old_calls(current_time)
            
            # Check if we're within limits
            if len(self.call_times) < self.max_calls and self._check_burst_limit(current_time):
                # Permission granted
                self.call_times.append(current_time)
                self.total_calls += 1
                return True
            else:
                # Rate limit exceeded
                self.blocked_calls += 1
                
                if wait:
                    # Calculate wait time
                    if len(self.call_times) >= self.max_calls:
                        # Wait until oldest call expires
                        wait_time = self.time_window - (current_time - self.call_times[0]) + 0.1
                    else:
                        # Burst limit exceeded, wait 1 second
                        wait_time = 1.0
                    
                    logger.debug(f"Rate limit exceeded, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    
                    # Retry after waiting
                    return self.acquire(wait=False)
                else:
                    return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            current_time = time.time()
            self._cleanup_old_calls(current_time)
            
            return {
                'current_calls': len(self.call_times),
                'max_calls': self.max_calls,
                'time_window': self.time_window,
                'total_calls': self.total_calls,
                'blocked_calls': self.blocked_calls,
                'utilization': f"{(len(self.call_times) / self.max_calls) * 100:.1f}%"
            }

class ExponentialBackoff:
    """Exponential backoff for retry logic"""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, 
                 multiplier: float = 2.0, jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.attempt = 0
    
    def wait(self):
        """Wait for the calculated delay"""
        if self.attempt == 0:
            self.attempt += 1
            return
        
        delay = min(self.base_delay * (self.multiplier ** (self.attempt - 1)), self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        logger.debug(f"Exponential backoff: waiting {delay:.2f}s (attempt {self.attempt})")
        time.sleep(delay)
        self.attempt += 1
    
    def reset(self):
        """Reset the backoff counter"""
        self.attempt = 0

class APIRateLimiter:
    """Rate limiter for different APIs with individual limits"""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.backoffs: Dict[str, ExponentialBackoff] = {}
        self.lock = threading.Lock()
    
    def _get_limiter(self, api_name: str) -> RateLimiter:
        """Get or create rate limiter for API"""
        if api_name not in self.limiters:
            # Default limits - can be customized per API
            limits = {
                'yahoo_finance': {'max_calls': 2000, 'time_window': 3600, 'burst_limit': 5},
                'angel_one': {'max_calls': 100, 'time_window': 60, 'burst_limit': 10},
                'fred': {'max_calls': 120, 'time_window': 60, 'burst_limit': 5},
                'default': {'max_calls': 100, 'time_window': 60, 'burst_limit': 10}
            }
            
            config = limits.get(api_name, limits['default'])
            self.limiters[api_name] = RateLimiter(**config)
            self.backoffs[api_name] = ExponentialBackoff()
        
        return self.limiters[api_name]
    
    def _get_backoff(self, api_name: str) -> ExponentialBackoff:
        """Get backoff instance for API"""
        if api_name not in self.backoffs:
            self.backoffs[api_name] = ExponentialBackoff()
        return self.backoffs[api_name]
    
    def call_with_retry(self, api_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Make an API call with rate limiting and retry logic
        
        Args:
            api_name: Name of the API (for rate limiting)
            func: Function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of the function call
        """
        limiter = self._get_limiter(api_name)
        backoff = self._get_backoff(api_name)
        
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Acquire rate limit permission
                if not limiter.acquire(wait=True):
                    raise Exception(f"Rate limit exceeded for {api_name}")
                
                # Make the API call
                logger.debug(f"Making API call to {api_name} (attempt {attempt + 1})")
                result = func(*args, **kwargs)
                
                # Reset backoff on success
                backoff.reset()
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if it's a rate limit error
                if any(keyword in str(e).lower() for keyword in ['rate limit', 'too many requests', '429']):
                    logger.warning(f"Rate limit error for {api_name}: {e}")
                    backoff.wait()
                    continue
                
                # Check if it's a temporary error
                if any(keyword in str(e).lower() for keyword in ['timeout', 'connection', 'network']):
                    logger.warning(f"Temporary error for {api_name}: {e}")
                    if attempt < max_retries:
                        backoff.wait()
                        continue
                
                # For other errors, don't retry
                logger.error(f"API call failed for {api_name}: {e}")
                break
        
        # All retries exhausted
        raise last_exception or Exception(f"API call failed for {api_name} after {max_retries} retries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all APIs"""
        stats = {}
        for api_name, limiter in self.limiters.items():
            stats[api_name] = limiter.get_statistics()
        return stats

# Global API rate limiter instance
_api_rate_limiter: Optional[APIRateLimiter] = None

def get_api_rate_limiter() -> APIRateLimiter:
    """Get the global API rate limiter instance"""
    global _api_rate_limiter
    if _api_rate_limiter is None:
        _api_rate_limiter = APIRateLimiter()
    return _api_rate_limiter

def rate_limited(api_name: str, max_retries: int = 3):
    """
    Decorator for rate-limited API calls
    
    Args:
        api_name: Name of the API
        max_retries: Maximum number of retries
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            rate_limiter = get_api_rate_limiter()
            return rate_limiter.call_with_retry(api_name, func, *args, **kwargs)
        return wrapper
    return decorator

# Convenience functions for common APIs
@rate_limited('yahoo_finance')
def yahoo_finance_call(func, *args, **kwargs):
    """Rate-limited Yahoo Finance API call"""
    return func(*args, **kwargs)

@rate_limited('angel_one')
def angel_one_call(func, *args, **kwargs):
    """Rate-limited Angel One API call"""
    return func(*args, **kwargs)

@rate_limited('fred')
def fred_api_call(func, *args, **kwargs):
    """Rate-limited FRED API call"""
    return func(*args, **kwargs)
