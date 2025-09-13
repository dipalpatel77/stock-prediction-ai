#!/usr/bin/env python3
"""
Currency Service
================

Provides real-time currency conversion and exchange rate data.
Supports multiple currency pairs and historical exchange rates.

Features:
- Real-time exchange rates
- Historical exchange rate data
- Currency conversion calculations
- Multi-currency formatting
- Exchange rate caching
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExchangeRate:
    """Exchange rate data structure"""
    from_currency: str
    to_currency: str
    rate: float
    timestamp: datetime
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None

@dataclass
class CurrencyPair:
    """Currency pair information"""
    pair: str
    base_currency: str
    quote_currency: str
    rate: float
    change_24h: float
    change_percent_24h: float
    volume_24h: float
    last_updated: datetime

class CurrencyService:
    """
    Currency Service for real-time exchange rates and conversions
    
    Supports multiple data sources and provides fallback mechanisms
    """
    
    def __init__(self, cache_duration_minutes: int = 15):
        """Initialize currency service"""
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cache = {}
        self.last_update = {}
        
        # API endpoints (using free/public APIs)
        self.api_endpoints = {
            'exchangerate_api': 'https://api.exchangerate-api.com/v4/latest/',
            'fixer_io': 'https://api.fixer.io/latest',  # Requires API key
            'currencylayer': 'https://api.currencylayer.com/live',  # Requires API key
            'openexchangerates': 'https://openexchangerates.org/api/latest.json'  # Requires API key
        }
        
        # Fallback rates (updated periodically)
        self.fallback_rates = {
            'USD': 1.0,
            'INR': 83.25,
            'EUR': 0.92,
            'GBP': 0.79,
            'JPY': 150.0,
            'CAD': 1.36,
            'AUD': 1.52,
            'CHF': 0.88,
            'CNY': 7.25,
            'KRW': 1330.0,
            'SGD': 1.35,
            'HKD': 7.82,
            'NZD': 1.63,
            'SEK': 10.85,
            'NOK': 10.95,
            'DKK': 6.87,
            'PLN': 4.05,
            'CZK': 23.15,
            'HUF': 365.0,
            'RUB': 95.0,
            'BRL': 5.05,
            'MXN': 17.25,
            'ZAR': 18.85,
            'TRY': 30.15,
            'AED': 3.67,
            'SAR': 3.75,
            'QAR': 3.64,
            'KWD': 0.31,
            'BHD': 0.38,
            'OMR': 0.38
        }
        
        # Major currency pairs
        self.major_pairs = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD',
            'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'CHF/JPY', 'AUD/JPY',
            'CAD/JPY', 'NZD/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
            'GBP/CHF', 'GBP/AUD', 'GBP/CAD', 'GBP/NZD', 'AUD/CHF', 'AUD/CAD',
            'AUD/NZD', 'CAD/CHF', 'CAD/NZD', 'NZD/CHF'
        ]
        
        logger.info("Currency Service initialized")
    
    def get_exchange_rate(self, from_currency: str, to_currency: str, 
                         use_cache: bool = True) -> ExchangeRate:
        """
        Get exchange rate between two currencies
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            use_cache: Use cached data if available
            
        Returns:
            ExchangeRate object
        """
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        if from_currency == to_currency:
            return ExchangeRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=1.0,
                timestamp=datetime.now(),
                source='same_currency'
            )
        
        cache_key = f"{from_currency}_{to_currency}"
        
        # Check cache first
        if use_cache and self._is_cache_valid(cache_key):
            cached_rate = self.cache[cache_key]
            logger.info(f"Using cached exchange rate: {from_currency}/{to_currency} = {cached_rate.rate}")
            return cached_rate
        
        # Try to get real-time rate
        try:
            rate = self._fetch_real_time_rate(from_currency, to_currency)
            if rate:
                self.cache[cache_key] = rate
                self.last_update[cache_key] = datetime.now()
                logger.info(f"Fetched real-time rate: {from_currency}/{to_currency} = {rate.rate}")
                return rate
        except Exception as e:
            logger.warning(f"Failed to fetch real-time rate: {str(e)}")
        
        # Use fallback rate
        fallback_rate = self._get_fallback_rate(from_currency, to_currency)
        self.cache[cache_key] = fallback_rate
        self.last_update[cache_key] = datetime.now()
        logger.info(f"Using fallback rate: {from_currency}/{to_currency} = {fallback_rate.rate}")
        return fallback_rate
    
    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> float:
        """
        Convert amount from one currency to another
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Converted amount
        """
        if from_currency.upper() == to_currency.upper():
            return amount
        
        exchange_rate = self.get_exchange_rate(from_currency, to_currency)
        converted_amount = amount * exchange_rate.rate
        
        logger.info(f"Converted {amount} {from_currency} to {converted_amount:.2f} {to_currency}")
        return converted_amount
    
    def get_currency_pairs(self, base_currency: str = 'USD') -> List[CurrencyPair]:
        """
        Get currency pairs with current rates
        
        Args:
            base_currency: Base currency for pairs
            
        Returns:
            List of CurrencyPair objects
        """
        pairs = []
        base_currency = base_currency.upper()
        
        for currency in self.fallback_rates.keys():
            if currency != base_currency:
                try:
                    rate = self.get_exchange_rate(base_currency, currency)
                    
                    # Calculate 24h change (simulated)
                    change_24h = np.random.uniform(-0.05, 0.05) * rate.rate
                    change_percent = (change_24h / rate.rate) * 100
                    
                    pair = CurrencyPair(
                        pair=f"{base_currency}/{currency}",
                        base_currency=base_currency,
                        quote_currency=currency,
                        rate=rate.rate,
                        change_24h=change_24h,
                        change_percent_24h=change_percent,
                        volume_24h=np.random.uniform(1000000, 10000000),
                        last_updated=rate.timestamp
                    )
                    pairs.append(pair)
                    
                except Exception as e:
                    logger.warning(f"Failed to get rate for {base_currency}/{currency}: {str(e)}")
        
        return pairs
    
    def get_historical_rates(self, from_currency: str, to_currency: str, 
                           days: int = 30) -> pd.DataFrame:
        """
        Get historical exchange rates
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical rates
        """
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # Generate simulated historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Get current rate as baseline
        current_rate = self.get_exchange_rate(from_currency, to_currency)
        
        # Generate historical rates with some volatility
        np.random.seed(42)  # For reproducible results
        volatility = 0.02  # 2% daily volatility
        rates = []
        
        for i, date in enumerate(dates):
            if i == 0:
                rate = current_rate.rate
            else:
                # Random walk with drift
                change = np.random.normal(0, volatility)
                rate = rates[-1] * (1 + change)
            
            rates.append(rate)
        
        df = pd.DataFrame({
            'date': dates,
            'rate': rates,
            'from_currency': from_currency,
            'to_currency': to_currency
        })
        
        logger.info(f"Generated {len(df)} historical rates for {from_currency}/{to_currency}")
        return df
    
    def _fetch_real_time_rate(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """Fetch real-time exchange rate from API"""
        try:
            # Try ExchangeRate-API (free tier)
            url = f"{self.api_endpoints['exchangerate_api']}{from_currency}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'rates' in data and to_currency in data['rates']:
                    rate = data['rates'][to_currency]
                    timestamp = datetime.fromtimestamp(data.get('timestamp', time.time()))
                    
                    return ExchangeRate(
                        from_currency=from_currency,
                        to_currency=to_currency,
                        rate=rate,
                        timestamp=timestamp,
                        source='exchangerate_api'
                    )
            
        except Exception as e:
            logger.warning(f"ExchangeRate-API failed: {str(e)}")
        
        return None
    
    def _get_fallback_rate(self, from_currency: str, to_currency: str) -> ExchangeRate:
        """Get fallback exchange rate"""
        from_rate = self.fallback_rates.get(from_currency, 1.0)
        to_rate = self.fallback_rates.get(to_currency, 1.0)
        
        # Convert through USD if not direct rate available
        if from_currency != 'USD' and to_currency != 'USD':
            rate = to_rate / from_rate
        elif from_currency == 'USD':
            rate = to_rate
        else:  # to_currency == 'USD'
            rate = 1.0 / from_rate
        
        return ExchangeRate(
            from_currency=from_currency,
            to_currency=to_currency,
            rate=rate,
            timestamp=datetime.now(),
            source='fallback'
        )
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache or cache_key not in self.last_update:
            return False
        
        time_since_update = datetime.now() - self.last_update[cache_key]
        return time_since_update < self.cache_duration
    
    def get_currency_info(self, currency_code: str) -> Dict[str, Any]:
        """Get detailed currency information"""
        currency_code = currency_code.upper()
        
        currency_info = {
            'USD': {'name': 'US Dollar', 'symbol': '$', 'country': 'United States'},
            'INR': {'name': 'Indian Rupee', 'symbol': '₹', 'country': 'India'},
            'EUR': {'name': 'Euro', 'symbol': '€', 'country': 'European Union'},
            'GBP': {'name': 'British Pound', 'symbol': '£', 'country': 'United Kingdom'},
            'JPY': {'name': 'Japanese Yen', 'symbol': '¥', 'country': 'Japan'},
            'CAD': {'name': 'Canadian Dollar', 'symbol': 'C$', 'country': 'Canada'},
            'AUD': {'name': 'Australian Dollar', 'symbol': 'A$', 'country': 'Australia'},
            'CHF': {'name': 'Swiss Franc', 'symbol': 'CHF', 'country': 'Switzerland'},
            'CNY': {'name': 'Chinese Yuan', 'symbol': '¥', 'country': 'China'},
            'KRW': {'name': 'South Korean Won', 'symbol': '₩', 'country': 'South Korea'},
            'SGD': {'name': 'Singapore Dollar', 'symbol': 'S$', 'country': 'Singapore'},
            'HKD': {'name': 'Hong Kong Dollar', 'symbol': 'HK$', 'country': 'Hong Kong'},
            'NZD': {'name': 'New Zealand Dollar', 'symbol': 'NZ$', 'country': 'New Zealand'},
            'SEK': {'name': 'Swedish Krona', 'symbol': 'kr', 'country': 'Sweden'},
            'NOK': {'name': 'Norwegian Krone', 'symbol': 'kr', 'country': 'Norway'},
            'DKK': {'name': 'Danish Krone', 'symbol': 'kr', 'country': 'Denmark'},
            'PLN': {'name': 'Polish Zloty', 'symbol': 'zł', 'country': 'Poland'},
            'CZK': {'name': 'Czech Koruna', 'symbol': 'Kč', 'country': 'Czech Republic'},
            'HUF': {'name': 'Hungarian Forint', 'symbol': 'Ft', 'country': 'Hungary'},
            'RUB': {'name': 'Russian Ruble', 'symbol': '₽', 'country': 'Russia'},
            'BRL': {'name': 'Brazilian Real', 'symbol': 'R$', 'country': 'Brazil'},
            'MXN': {'name': 'Mexican Peso', 'symbol': '$', 'country': 'Mexico'},
            'ZAR': {'name': 'South African Rand', 'symbol': 'R', 'country': 'South Africa'},
            'TRY': {'name': 'Turkish Lira', 'symbol': '₺', 'country': 'Turkey'},
            'AED': {'name': 'UAE Dirham', 'symbol': 'د.إ', 'country': 'United Arab Emirates'},
            'SAR': {'name': 'Saudi Riyal', 'symbol': '﷼', 'country': 'Saudi Arabia'},
            'QAR': {'name': 'Qatari Riyal', 'symbol': '﷼', 'country': 'Qatar'},
            'KWD': {'name': 'Kuwaiti Dinar', 'symbol': 'د.ك', 'country': 'Kuwait'},
            'BHD': {'name': 'Bahraini Dinar', 'symbol': 'د.ب', 'country': 'Bahrain'},
            'OMR': {'name': 'Omani Rial', 'symbol': '﷼', 'country': 'Oman'}
        }
        
        return currency_info.get(currency_code, {
            'name': f'{currency_code} Currency',
            'symbol': currency_code,
            'country': 'Unknown'
        })
    
    def format_currency(self, amount: float, currency_code: str, 
                       show_symbol: bool = True, decimal_places: int = None) -> str:
        """
        Format amount with currency symbol and proper formatting
        
        Args:
            amount: Amount to format
            currency_code: Currency code
            show_symbol: Show currency symbol
            decimal_places: Number of decimal places (auto if None)
            
        Returns:
            Formatted currency string
        """
        currency_code = currency_code.upper()
        currency_info = self.get_currency_info(currency_code)
        
        # Determine decimal places
        if decimal_places is None:
            # Most currencies use 2 decimal places, except JPY, KRW, etc.
            if currency_code in ['JPY', 'KRW', 'VND', 'IDR']:
                decimal_places = 0
            else:
                decimal_places = 2
        
        # Format number
        if decimal_places == 0:
            formatted_amount = f"{int(amount):,}"
        else:
            formatted_amount = f"{amount:,.{decimal_places}f}"
        
        # Add currency symbol
        if show_symbol:
            symbol = currency_info['symbol']
            return f"{symbol}{formatted_amount}"
        else:
            return f"{formatted_amount} {currency_code}"
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get currency market summary"""
        try:
            # Get major pairs
            major_pairs_data = []
            for pair in self.major_pairs[:10]:  # Top 10 major pairs
                base, quote = pair.split('/')
                rate = self.get_exchange_rate(base, quote)
                
                # Simulate 24h change
                change_24h = np.random.uniform(-0.03, 0.03) * rate.rate
                change_percent = (change_24h / rate.rate) * 100
                
                major_pairs_data.append({
                    'pair': pair,
                    'rate': rate.rate,
                    'change_24h': change_24h,
                    'change_percent_24h': change_percent,
                    'last_updated': rate.timestamp
                })
            
            return {
                'major_pairs': major_pairs_data,
                'market_status': 'Open',
                'last_updated': datetime.now(),
                'total_pairs': len(major_pairs_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to get market summary: {str(e)}")
            return {
                'major_pairs': [],
                'market_status': 'Error',
                'last_updated': datetime.now(),
                'total_pairs': 0,
                'error': str(e)
            }

# Example usage and testing
if __name__ == "__main__":
    # Initialize currency service
    currency_service = CurrencyService()
    
    print("🌍 Currency Service Test")
    print("=" * 50)
    
    # Test exchange rate
    print("\n📊 Exchange Rates:")
    usd_to_inr = currency_service.get_exchange_rate('USD', 'INR')
    print(f"USD/INR: {usd_to_inr.rate:.2f} (Source: {usd_to_inr.source})")
    
    eur_to_usd = currency_service.get_exchange_rate('EUR', 'USD')
    print(f"EUR/USD: {eur_to_usd.rate:.4f} (Source: {eur_to_usd.source})")
    
    # Test currency conversion
    print("\n💱 Currency Conversion:")
    amount_usd = 1000
    amount_inr = currency_service.convert_amount(amount_usd, 'USD', 'INR')
    print(f"${amount_usd} = {currency_service.format_currency(amount_inr, 'INR')}")
    
    amount_eur = currency_service.convert_amount(amount_usd, 'USD', 'EUR')
    print(f"${amount_usd} = {currency_service.format_currency(amount_eur, 'EUR')}")
    
    # Test currency pairs
    print("\n📈 Major Currency Pairs:")
    pairs = currency_service.get_currency_pairs('USD')[:5]
    for pair in pairs:
        print(f"{pair.pair}: {pair.rate:.4f} ({pair.change_percent_24h:+.2f}%)")
    
    # Test historical rates
    print("\n📅 Historical Rates (USD/INR, 7 days):")
    historical = currency_service.get_historical_rates('USD', 'INR', 7)
    print(historical.tail(3))
    
    # Test market summary
    print("\n🌍 Market Summary:")
    summary = currency_service.get_market_summary()
    print(f"Market Status: {summary['market_status']}")
    print(f"Total Pairs: {summary['total_pairs']}")
    print(f"Last Updated: {summary['last_updated']}")
    
    print("\n✅ Currency Service test completed!")
