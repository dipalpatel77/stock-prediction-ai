"""
Stock Utilities
Shared helper functions used across pipeline and services.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def is_indian_stock(ticker: str, dynamic_lookup=None) -> bool:
    """
    Return True if ticker is an Indian/NSE stock.

    When dynamic_lookup is provided (a DynamicStockLookup instance with
    is_exact_match()), uses exact symbol matching to avoid false positives
    (e.g. 'AAPL' matching 'HARIAAPL').  Falls back to suffix-based heuristic
    when no lookup is available.
    """
    try:
        if dynamic_lookup is not None:
            if dynamic_lookup.is_exact_match(ticker):
                logger.info(f"Exact match found: {ticker} is an Indian stock")
                return True
            else:
                logger.info(f"No exact match for {ticker} — treating as non-Indian stock")
                return False

        # Suffix-based fallback
        upper = ticker.upper()
        indian_suffixes = ('.NS', '.BO', '.NSE', '.BSE')
        if any(upper.endswith(s) for s in indian_suffixes):
            return True

        # Common NSE large-cap names without suffix
        known_indian = {
            'RELIANCE', 'TCS', 'INFY', 'HDFC', 'HDFCBANK', 'ICICIBANK',
            'KOTAKBANK', 'SBIN', 'BAJFINANCE', 'HINDUNILVR', 'ITC',
            'AXISBANK', 'MARUTI', 'SUNPHARMA', 'TATAMOTORS', 'WIPRO',
            'NESTLEIND', 'TECHM', 'ULTRACEMCO', 'ASIANPAINT',
        }
        return upper in known_indian

    except Exception as e:
        logger.error(f"is_indian_stock check failed for {ticker}: {e}")
        return False


def is_cache_fresh(data: pd.DataFrame, max_age_hours: float = 24.0) -> bool:
    """Return True if the DataFrame's index max timestamp is within max_age_hours."""
    try:
        if data is None or data.empty:
            return False
        last_update = data.index.max()
        age_hours = (datetime.now() - last_update).total_seconds() / 3600
        return age_hours < max_age_hours
    except Exception as e:
        logger.error(f"Cache freshness check failed: {e}")
        return False
