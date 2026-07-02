"""
Currency Service
Handles currency conversion, exchange rates, and currency-related analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import requests
import json
from dataclasses import dataclass
from enum import Enum

class CurrencyCode(Enum):
    """Currency codes"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"
    INR = "INR"
    BRL = "BRL"
    KRW = "KRW"
    MXN = "MXN"
    RUB = "RUB"
    ZAR = "ZAR"
    SGD = "SGD"
    HKD = "HKD"
    NOK = "NOK"
    SEK = "SEK"
    DKK = "DKK"
    PLN = "PLN"

@dataclass
class ExchangeRate:
    """Exchange rate data structure"""
    from_currency: str
    to_currency: str
    rate: float
    timestamp: datetime
    source: str

class CurrencyService:
    """Service for handling currency operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.exchange_rates_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        # API configurations
        self.alpha_vantage_key = self.config.get('alpha_vantage_key')
        self.fixer_key = self.config.get('fixer_key')
        self.exchangerate_key = self.config.get('exchangerate_key')
        
        # Default to USD if no base currency specified
        self.base_currency = self.config.get('base_currency', 'USD')
        
        self.logger.info("Currency Service initialized")

    def get_exchange_rate(self, from_currency: str, to_currency: str, 
                         date: datetime = None) -> Optional[float]:
        """Get exchange rate between two currencies"""
        try:
            if date is None:
                date = datetime.now()
            
            # Check cache first
            cache_key = f"{from_currency}_{to_currency}_{date.date()}"
            if cache_key in self.exchange_rates_cache:
                cached_rate = self.exchange_rates_cache[cache_key]
                if datetime.now() - cached_rate.timestamp < self.cache_duration:
                    return cached_rate.rate
            
            # Try multiple data sources
            rate = None
            
            if self.alpha_vantage_key:
                rate = self._get_alpha_vantage_rate(from_currency, to_currency, date)
            
            if rate is None and self.fixer_key:
                rate = self._get_fixer_rate(from_currency, to_currency, date)
            
            if rate is None and self.exchangerate_key:
                rate = self._get_exchangerate_rate(from_currency, to_currency, date)
            
            if rate is None:
                # Fallback to hardcoded rates (for testing)
                rate = self._get_fallback_rate(from_currency, to_currency)
            
            if rate is not None:
                # Cache the rate
                self.exchange_rates_cache[cache_key] = ExchangeRate(
                    from_currency=from_currency,
                    to_currency=to_currency,
                    rate=rate,
                    timestamp=datetime.now(),
                    source='api'
                )
            
            return rate
            
        except Exception as e:
            self.logger.error(f"Error getting exchange rate {from_currency}/{to_currency}: {e}")
            return None

    def _get_alpha_vantage_rate(self, from_currency: str, to_currency: str, 
                               date: datetime) -> Optional[float]:
        """Get exchange rate from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'CURRENCY_EXCHANGE_RATE',
                'from_currency': from_currency,
                'to_currency': to_currency,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'Realtime Currency Exchange Rate' in data:
                rate_data = data['Realtime Currency Exchange Rate']
                return float(rate_data.get('5. Exchange Rate', 0))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage rate: {e}")
            return None

    def _get_fixer_rate(self, from_currency: str, to_currency: str, 
                       date: datetime) -> Optional[float]:
        """Get exchange rate from Fixer.io"""
        try:
            url = "http://data.fixer.io/api/latest"
            params = {
                'access_key': self.fixer_key,
                'base': from_currency,
                'symbols': to_currency
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('success'):
                rates = data.get('rates', {})
                return rates.get(to_currency)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Fixer rate: {e}")
            return None

    def _get_exchangerate_rate(self, from_currency: str, to_currency: str, 
                              date: datetime) -> Optional[float]:
        """Get exchange rate from ExchangeRate-API"""
        try:
            url = f"https://v6.exchangerate-api.com/v6/{self.exchangerate_key}/pair/{from_currency}/{to_currency}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('result') == 'success':
                return data.get('conversion_rate')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting ExchangeRate rate: {e}")
            return None

    def _get_fallback_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get fallback exchange rate (hardcoded for testing)"""
        # This is a simplified fallback with some common rates
        fallback_rates = {
            ('USD', 'EUR'): 0.85,
            ('USD', 'GBP'): 0.73,
            ('USD', 'JPY'): 110.0,
            ('USD', 'CAD'): 1.25,
            ('USD', 'AUD'): 1.35,
            ('USD', 'CHF'): 0.92,
            ('USD', 'CNY'): 6.45,
            ('USD', 'INR'): 74.0,
            ('USD', 'BRL'): 5.2,
            ('USD', 'KRW'): 1180.0,
            ('USD', 'MXN'): 20.0,
            ('USD', 'RUB'): 73.0,
            ('USD', 'ZAR'): 14.5,
            ('USD', 'SGD'): 1.35,
            ('USD', 'HKD'): 7.8,
            ('USD', 'NOK'): 8.5,
            ('USD', 'SEK'): 8.7,
            ('USD', 'DKK'): 6.3,
            ('USD', 'PLN'): 3.9
        }
        
        # Check direct rate
        if (from_currency, to_currency) in fallback_rates:
            return fallback_rates[(from_currency, to_currency)]
        
        # Check reverse rate
        if (to_currency, from_currency) in fallback_rates:
            return 1.0 / fallback_rates[(to_currency, from_currency)]
        
        return None

    def convert_currency(self, amount: float, from_currency: str, to_currency: str, 
                        date: datetime = None) -> Optional[float]:
        """Convert amount from one currency to another"""
        try:
            if from_currency == to_currency:
                return amount
            
            rate = self.get_exchange_rate(from_currency, to_currency, date)
            if rate is None:
                return None
            
            return amount * rate
            
        except Exception as e:
            self.logger.error(f"Error converting currency: {e}")
            return None

    def get_currency_strength(self, currency: str, days: int = 30) -> Dict[str, Any]:
        """Get currency strength analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get rates against major currencies
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']
            rates = {}
            
            for major_currency in major_currencies:
                if major_currency != currency:
                    rate = self.get_exchange_rate(currency, major_currency, end_date)
                    if rate:
                        rates[major_currency] = rate
            
            if not rates:
                return {'strength_score': 0.5, 'trend': 'neutral', 'analysis': 'No data available'}
            
            # Calculate strength score
            strength_score = np.mean(list(rates.values()))
            
            # Determine trend
            if strength_score > 1.0:
                trend = 'strong'
            elif strength_score < 0.5:
                trend = 'weak'
            else:
                trend = 'neutral'
            
            # Generate analysis
            analysis = self._generate_currency_analysis(currency, rates, strength_score, trend)
            
            return {
                'strength_score': strength_score,
                'trend': trend,
                'rates': rates,
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error getting currency strength: {e}")
            return {'strength_score': 0.5, 'trend': 'neutral', 'analysis': 'Error in analysis'}

    def _generate_currency_analysis(self, currency: str, rates: Dict[str, float], 
                                   strength_score: float, trend: str) -> str:
        """Generate currency analysis"""
        try:
            if trend == 'strong':
                return f"{currency} is showing strong performance against major currencies"
            elif trend == 'weak':
                return f"{currency} is showing weakness against major currencies"
            else:
                return f"{currency} is showing neutral performance against major currencies"
                
        except Exception as e:
            self.logger.error(f"Error generating currency analysis: {e}")
            return "Analysis unavailable"

    def get_currency_volatility(self, currency: str, days: int = 30) -> Dict[str, Any]:
        """Get currency volatility analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get historical rates
            rates = []
            for i in range(days):
                date = start_date + timedelta(days=i)
                rate = self.get_exchange_rate(currency, self.base_currency, date)
                if rate:
                    rates.append(rate)
            
            if len(rates) < 2:
                return {'volatility': 0, 'trend': 'stable', 'analysis': 'Insufficient data'}
            
            # Calculate volatility
            rates_array = np.array(rates)
            volatility = np.std(rates_array) / np.mean(rates_array)
            
            # Determine trend
            if rates[-1] > rates[0]:
                trend = 'appreciating'
            elif rates[-1] < rates[0]:
                trend = 'depreciating'
            else:
                trend = 'stable'
            
            # Generate analysis
            analysis = self._generate_volatility_analysis(currency, volatility, trend)
            
            return {
                'volatility': volatility,
                'trend': trend,
                'rates': rates,
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error getting currency volatility: {e}")
            return {'volatility': 0, 'trend': 'stable', 'analysis': 'Error in analysis'}

    def _generate_volatility_analysis(self, currency: str, volatility: float, trend: str) -> str:
        """Generate volatility analysis"""
        try:
            if volatility > 0.1:
                volatility_level = 'high'
            elif volatility > 0.05:
                volatility_level = 'medium'
            else:
                volatility_level = 'low'
            
            return f"{currency} shows {volatility_level} volatility with {trend} trend"
            
        except Exception as e:
            self.logger.error(f"Error generating volatility analysis: {e}")
            return "Analysis unavailable"

    def get_currency_correlation(self, currency1: str, currency2: str, 
                                days: int = 30) -> Dict[str, Any]:
        """Get correlation between two currencies"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get historical rates for both currencies
            rates1 = []
            rates2 = []
            
            for i in range(days):
                date = start_date + timedelta(days=i)
                rate1 = self.get_exchange_rate(currency1, self.base_currency, date)
                rate2 = self.get_exchange_rate(currency2, self.base_currency, date)
                
                if rate1 and rate2:
                    rates1.append(rate1)
                    rates2.append(rate2)
            
            if len(rates1) < 2:
                return {'correlation': 0, 'analysis': 'Insufficient data'}
            
            # Calculate correlation
            correlation = np.corrcoef(rates1, rates2)[0, 1]
            
            # Generate analysis
            analysis = self._generate_correlation_analysis(currency1, currency2, correlation)
            
            return {
                'correlation': correlation,
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error getting currency correlation: {e}")
            return {'correlation': 0, 'analysis': 'Error in analysis'}

    def _generate_correlation_analysis(self, currency1: str, currency2: str, 
                                     correlation: float) -> str:
        """Generate correlation analysis"""
        try:
            if correlation > 0.7:
                return f"{currency1} and {currency2} are highly correlated"
            elif correlation > 0.3:
                return f"{currency1} and {currency2} are moderately correlated"
            elif correlation > -0.3:
                return f"{currency1} and {currency2} are weakly correlated"
            else:
                return f"{currency1} and {currency2} are negatively correlated"
                
        except Exception as e:
            self.logger.error(f"Error generating correlation analysis: {e}")
            return "Analysis unavailable"

    def get_currency_summary(self, currency: str) -> Dict[str, Any]:
        """Get comprehensive currency summary"""
        try:
            # Get strength analysis
            strength = self.get_currency_strength(currency)
            
            # Get volatility analysis
            volatility = self.get_currency_volatility(currency)
            
            # Get correlation with major currencies
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY']
            correlations = {}
            
            for major_currency in major_currencies:
                if major_currency != currency:
                    correlation = self.get_currency_correlation(currency, major_currency)
                    correlations[major_currency] = correlation.get('correlation', 0)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(strength, volatility, correlations)
            
            return {
                'currency': currency,
                'strength': strength,
                'volatility': volatility,
                'correlations': correlations,
                'overall_score': overall_score,
                'recommendations': self._generate_currency_recommendations(strength, volatility)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting currency summary: {e}")
            return {}

    def _calculate_overall_score(self, strength: Dict[str, Any], volatility: Dict[str, Any], 
                                correlations: Dict[str, float]) -> float:
        """Calculate overall currency score"""
        try:
            # Weight the components
            strength_weight = 0.4
            volatility_weight = 0.3
            correlation_weight = 0.3
            
            # Normalize strength score
            strength_score = strength.get('strength_score', 0.5)
            
            # Normalize volatility (lower is better)
            volatility_score = 1.0 - min(volatility.get('volatility', 0.5), 1.0)
            
            # Normalize correlation (average of correlations)
            correlation_score = np.mean(list(correlations.values())) if correlations else 0.5
            correlation_score = (correlation_score + 1) / 2  # Convert from [-1,1] to [0,1]
            
            # Calculate weighted score
            overall_score = (strength_score * strength_weight + 
                           volatility_score * volatility_weight + 
                           correlation_score * correlation_weight)
            
            return min(max(overall_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.5

    def _generate_currency_recommendations(self, strength: Dict[str, Any], 
                                         volatility: Dict[str, Any]) -> List[str]:
        """Generate currency recommendations"""
        try:
            recommendations = []
            
            strength_trend = strength.get('trend', 'neutral')
            volatility_level = volatility.get('volatility', 0)
            
            if strength_trend == 'strong':
                recommendations.append("Currency is showing strength - consider long positions")
            elif strength_trend == 'weak':
                recommendations.append("Currency is showing weakness - consider short positions")
            
            if volatility_level > 0.1:
                recommendations.append("High volatility detected - use appropriate risk management")
            elif volatility_level < 0.05:
                recommendations.append("Low volatility - suitable for conservative strategies")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating currency recommendations: {e}")
            return []
    
    def get_currency_data(self) -> Dict[str, Any]:
        """Get currency data - compatibility method"""
        try:
            # Get major currency rates
            major_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
            currency_rates = {}
            
            for currency in major_currencies:
                if currency != 'USD':
                    rate = self.get_exchange_rate('USD', currency)
                    if rate:
                        currency_rates[f'USD_{currency}'] = rate
            
            # Get currency trends
            trends = self._get_currency_trends()
            
            # Get currency volatility for USD (default base currency)
            volatility = self.get_currency_volatility('USD')
            
            return {
                'currency_rates': currency_rates,
                'trends': trends,
                'volatility': volatility,
                'timestamp': datetime.now().isoformat(),
                'source': 'currency_service'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting currency data: {e}")
            return {
                'currency_rates': {},
                'trends': {},
                'volatility': {},
                'timestamp': datetime.now().isoformat(),
                'source': 'placeholder',
                'error': str(e)
            }
    
    def _get_currency_trends(self) -> Dict[str, Any]:
        """Get currency trends - placeholder method"""
        try:
            return {
                'USD': {'trend': 'stable', 'strength': 0.5},
                'EUR': {'trend': 'stable', 'strength': 0.4},
                'GBP': {'trend': 'stable', 'strength': 0.3},
                'JPY': {'trend': 'stable', 'strength': 0.6}
            }
        except Exception as e:
            self.logger.error(f"Error getting currency trends: {e}")
            return {}
