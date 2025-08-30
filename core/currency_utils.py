#!/usr/bin/env python3
"""
Currency Utilities Module
Handles currency symbols and formatting based on stock exchanges and symbols.
"""

import re
from typing import Dict, Optional


class CurrencyUtils:
    """Utility class for handling currency symbols and formatting."""
    
    # Currency mapping for different exchanges
    EXCHANGE_CURRENCIES = {
        'NSE': '₹',  # Indian Rupee
        'BSE': '₹',  # Indian Rupee
        'NASDAQ': '$',  # US Dollar
        'NYSE': '$',  # US Dollar
        'AMEX': '$',  # US Dollar
        'LSE': '£',  # British Pound
        'TSE': '¥',  # Japanese Yen
        'ASX': 'A$',  # Australian Dollar
        'TSX': 'C$',  # Canadian Dollar
        'FRA': '€',  # Euro
        'ETR': '€',  # Euro
        'SWX': 'CHF',  # Swiss Franc
        'HKG': 'HK$',  # Hong Kong Dollar
        'SIN': 'S$',  # Singapore Dollar
    }
    
    # Symbol patterns for different markets
    SYMBOL_PATTERNS = {
        r'\.NS$|\.BO$': '₹',  # Indian stocks (.NS, .BO)
        r'\.L$': '£',  # London stocks (.L)
        r'\.T$|\.TO$': '¥',  # Tokyo stocks (.T, .TO)
        r'\.AX$': 'A$',  # Australian stocks (.AX)
        r'\.V$': 'C$',  # Canadian stocks (.V)
        r'\.F$|\.DE$': '€',  # European stocks (.F, .DE)
        r'\.SW$': 'CHF',  # Swiss stocks (.SW)
        r'\.HK$': 'HK$',  # Hong Kong stocks (.HK)
        r'\.SI$': 'S$',  # Singapore stocks (.SI)
    }
    
    @classmethod
    def get_currency_symbol(cls, symbol: str, exchange: Optional[str] = None) -> str:
        """
        Get the appropriate currency symbol for a given stock symbol and exchange.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'RELIANCE.NS', 'TSLA')
            exchange: Exchange name (e.g., 'NASDAQ', 'NSE', 'NYSE')
            
        Returns:
            Currency symbol (e.g., '$', '₹', '£', '€')
        """
        # First, try to determine currency from symbol patterns
        for pattern, currency in cls.SYMBOL_PATTERNS.items():
            if re.search(pattern, symbol, re.IGNORECASE):
                return currency
        
        # If no pattern match, try exchange-based mapping
        if exchange:
            exchange_upper = exchange.upper()
            if exchange_upper in cls.EXCHANGE_CURRENCIES:
                return cls.EXCHANGE_CURRENCIES[exchange_upper]
        
        # Default to USD for US stocks (no suffix)
        if not re.search(r'\.', symbol):
            return '$'
        
        # Default fallback
        return '$'
    
    @classmethod
    def format_price(cls, price: float, symbol: str, exchange: Optional[str] = None, 
                    decimal_places: int = 2) -> str:
        """
        Format price with appropriate currency symbol.
        
        Args:
            price: Price value
            symbol: Stock symbol
            exchange: Exchange name
            decimal_places: Number of decimal places
            
        Returns:
            Formatted price string with currency symbol
        """
        currency = cls.get_currency_symbol(symbol, exchange)
        
        # Format based on currency
        if currency == '₹':
            # Indian Rupee formatting
            if price >= 10000000:  # 1 crore
                return f"₹{price/10000000:.2f}Cr"
            elif price >= 100000:  # 1 lakh
                return f"₹{price/100000:.2f}L"
            else:
                return f"₹{price:.2f}"
        elif currency in ['$', '£', '€', '¥']:
            # Standard formatting for major currencies
            return f"{currency}{price:.{decimal_places}f}"
        else:
            # Other currencies
            return f"{currency} {price:.{decimal_places}f}"
    
    @classmethod
    def format_change(cls, change: float, symbol: str, exchange: Optional[str] = None) -> str:
        """
        Format price change with appropriate currency symbol.
        
        Args:
            change: Price change value
            symbol: Stock symbol
            exchange: Exchange name
            
        Returns:
            Formatted change string with currency symbol
        """
        currency = cls.get_currency_symbol(symbol, exchange)
        
        if change >= 0:
            sign = "+"
        else:
            sign = ""
        
        if currency == '₹':
            return f"{sign}₹{change:.2f}"
        elif currency in ['$', '£', '€', '¥']:
            return f"{sign}{currency}{change:.2f}"
        else:
            return f"{sign}{currency} {change:.2f}"
    
    @classmethod
    def is_indian_stock(cls, symbol: str, exchange: Optional[str] = None) -> bool:
        """
        Check if the stock is an Indian stock.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name
            
        Returns:
            True if Indian stock, False otherwise
        """
        currency = cls.get_currency_symbol(symbol, exchange)
        return currency == '₹'
    
    @classmethod
    def is_us_stock(cls, symbol: str, exchange: Optional[str] = None) -> bool:
        """
        Check if the stock is a US stock.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name
            
        Returns:
            True if US stock, False otherwise
        """
        currency = cls.get_currency_symbol(symbol, exchange)
        return currency == '$'
    
    @classmethod
    def get_market_info(cls, symbol: str, exchange: Optional[str] = None) -> Dict[str, str]:
        """
        Get comprehensive market information for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name
            
        Returns:
            Dictionary with market information
        """
        currency = cls.get_currency_symbol(symbol, exchange)
        
        info = {
            'currency': currency,
            'currency_name': cls._get_currency_name(currency),
            'market': cls._get_market_name(currency),
            'region': cls._get_region(currency)
        }
        
        return info
    
    @classmethod
    def _get_currency_name(cls, currency: str) -> str:
        """Get currency name from symbol."""
        currency_names = {
            '₹': 'Indian Rupee',
            '$': 'US Dollar',
            '£': 'British Pound',
            '€': 'Euro',
            '¥': 'Japanese Yen',
            'A$': 'Australian Dollar',
            'C$': 'Canadian Dollar',
            'CHF': 'Swiss Franc',
            'HK$': 'Hong Kong Dollar',
            'S$': 'Singapore Dollar'
        }
        return currency_names.get(currency, 'Unknown')
    
    @classmethod
    def _get_market_name(cls, currency: str) -> str:
        """Get market name from currency."""
        market_names = {
            '₹': 'Indian Market',
            '$': 'US Market',
            '£': 'UK Market',
            '€': 'European Market',
            '¥': 'Japanese Market',
            'A$': 'Australian Market',
            'C$': 'Canadian Market',
            'CHF': 'Swiss Market',
            'HK$': 'Hong Kong Market',
            'S$': 'Singapore Market'
        }
        return market_names.get(currency, 'Unknown')
    
    @classmethod
    def _get_region(cls, currency: str) -> str:
        """Get region from currency."""
        regions = {
            '₹': 'Asia',
            '$': 'North America',
            '£': 'Europe',
            '€': 'Europe',
            '¥': 'Asia',
            'A$': 'Oceania',
            'C$': 'North America',
            'CHF': 'Europe',
            'HK$': 'Asia',
            'S$': 'Asia'
        }
        return regions.get(currency, 'Unknown')


# Convenience functions
def get_currency_symbol(symbol: str, exchange: Optional[str] = None) -> str:
    """Get currency symbol for a stock."""
    return CurrencyUtils.get_currency_symbol(symbol, exchange)


def format_price(price: float, symbol: str, exchange: Optional[str] = None, 
                decimal_places: int = 2) -> str:
    """Format price with currency symbol."""
    return CurrencyUtils.format_price(price, symbol, exchange, decimal_places)


def format_change(change: float, symbol: str, exchange: Optional[str] = None) -> str:
    """Format price change with currency symbol."""
    return CurrencyUtils.format_change(change, symbol, exchange)


def is_indian_stock(symbol: str, exchange: Optional[str] = None) -> bool:
    """Check if stock is Indian."""
    return CurrencyUtils.is_indian_stock(symbol, exchange)


def is_us_stock(symbol: str, exchange: Optional[str] = None) -> bool:
    """Check if stock is US."""
    return CurrencyUtils.is_us_stock(symbol, exchange)
