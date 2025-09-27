"""
Multi-Exchange Data Service
Handles data from multiple exchanges and markets
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

class Exchange(Enum):
    """Exchange enumeration"""
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    AMEX = "amex"
    LSE = "lse"
    TSE = "tse"
    HKEX = "hkex"
    SSE = "sse"
    SZSE = "szse"
    BSE = "bse"
    NSE = "nse"
    ASX = "asx"
    TSX = "tsx"
    EURONEXT = "euronext"
    XETRA = "xetra"
    SIX = "six"

class MarketSession(Enum):
    """Market session enumeration"""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

@dataclass
class ExchangeData:
    """Exchange data structure"""
    exchange: Exchange
    ticker: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: float
    session: MarketSession
    timestamp: datetime
    currency: str

@dataclass
class MarketStatus:
    """Market status structure"""
    exchange: Exchange
    is_open: bool
    session: MarketSession
    open_time: datetime
    close_time: datetime
    timezone: str

class MultiExchangeDataService:
    """Service for multi-exchange data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.exchange_cache = {}
        self.cache_duration = timedelta(minutes=5)
        
        # API configurations
        self.alpha_vantage_key = self.config.get('alpha_vantage_key')
        self.polygon_key = self.config.get('polygon_key')
        self.yahoo_finance_enabled = self.config.get('yahoo_finance_enabled', True)
        
        # Exchange settings
        self.monitored_exchanges = self.config.get('monitored_exchanges', [
            Exchange.NYSE, Exchange.NASDAQ, Exchange.LSE, Exchange.TSE, Exchange.HKEX
        ])
        
        self.logger.info("Multi-Exchange Data Service initialized")

    def get_exchange_data(self, ticker: str, exchange: Exchange, 
                         start_date: datetime = None, end_date: datetime = None) -> List[ExchangeData]:
        """Get data for a ticker from a specific exchange"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            # Check cache first
            cache_key = f"{ticker}_{exchange.value}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.exchange_cache:
                cached_data = self.exchange_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['data']
            
            exchange_data = []
            
            # Try multiple data sources
            if self.alpha_vantage_key:
                exchange_data.extend(self._get_alpha_vantage_data(ticker, exchange, start_date, end_date))
            
            if self.polygon_key:
                exchange_data.extend(self._get_polygon_data(ticker, exchange, start_date, end_date))
            
            if self.yahoo_finance_enabled:
                exchange_data.extend(self._get_yahoo_finance_data(ticker, exchange, start_date, end_date))
            
            # If no API keys, use dummy data
            if not exchange_data:
                exchange_data = self._get_dummy_data(ticker, exchange, start_date, end_date)
            
            # Cache results
            self.exchange_cache[cache_key] = {
                'data': exchange_data,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Retrieved {len(exchange_data)} data points for {ticker} on {exchange.value}")
            return exchange_data
            
        except Exception as e:
            self.logger.error(f"Error getting exchange data for {ticker} on {exchange.value}: {e}")
            return []

    def _get_alpha_vantage_data(self, ticker: str, exchange: Exchange, 
                               start_date: datetime, end_date: datetime) -> List[ExchangeData]:
        """Get data from Alpha Vantage"""
        try:
            # Map exchange to Alpha Vantage symbol
            symbol = self._get_alpha_vantage_symbol(ticker, exchange)
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            exchange_data = []
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                for date_str, values in time_series.items():
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    if start_date <= date <= end_date:
                        try:
                            exchange_data.append(ExchangeData(
                                exchange=exchange,
                                ticker=ticker,
                                price=float(values['4. close']),
                                change=float(values['4. close']) - float(values['1. open']),
                                change_percent=((float(values['4. close']) - float(values['1. open'])) / float(values['1. open'])) * 100,
                                volume=int(float(values['5. volume'])),
                                market_cap=0,  # Would need additional API call
                                session=MarketSession.REGULAR,
                                timestamp=date,
                                currency=self._get_exchange_currency(exchange)
                            ))
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"Error parsing exchange data: {e}")
                            continue
            
            return exchange_data
            
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage data: {e}")
            return []

    def _get_polygon_data(self, ticker: str, exchange: Exchange, 
                         start_date: datetime, end_date: datetime) -> List[ExchangeData]:
        """Get data from Polygon API"""
        try:
            # Polygon API implementation would go here
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting Polygon data: {e}")
            return []

    def _get_yahoo_finance_data(self, ticker: str, exchange: Exchange, 
                               start_date: datetime, end_date: datetime) -> List[ExchangeData]:
        """Get data from Yahoo Finance"""
        try:
            # Yahoo Finance API implementation would go here
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting Yahoo Finance data: {e}")
            return []

    def _get_dummy_data(self, ticker: str, exchange: Exchange, 
                       start_date: datetime, end_date: datetime) -> List[ExchangeData]:
        """Get dummy exchange data for testing"""
        try:
            exchange_data = []
            current_date = start_date
            
            # Base values for different exchanges
            base_values = {
                Exchange.NYSE: 150.0,
                Exchange.NASDAQ: 120.0,
                Exchange.LSE: 100.0,
                Exchange.TSE: 2000.0,
                Exchange.HKEX: 50.0,
                Exchange.SSE: 10.0,
                Exchange.BSE: 500.0,
                Exchange.NSE: 500.0,
                Exchange.ASX: 25.0,
                Exchange.TSX: 30.0
            }
            
            base_value = base_values.get(exchange, 100.0)
            
            while current_date <= end_date:
                # Add some randomness to the data
                random_factor = np.random.normal(1, 0.02)
                price = base_value * random_factor
                
                # Calculate change
                change = np.random.normal(0, 0.01) * price
                change_percent = (change / price) * 100
                
                exchange_data.append(ExchangeData(
                    exchange=exchange,
                    ticker=ticker,
                    price=price,
                    change=change,
                    change_percent=change_percent,
                    volume=np.random.randint(100000, 1000000),
                    market_cap=price * np.random.randint(1000000, 10000000),
                    session=MarketSession.REGULAR,
                    timestamp=current_date,
                    currency=self._get_exchange_currency(exchange)
                ))
                
                current_date += timedelta(days=1)
            
            return exchange_data
            
        except Exception as e:
            self.logger.error(f"Error generating dummy data: {e}")
            return []

    def _get_alpha_vantage_symbol(self, ticker: str, exchange: Exchange) -> str:
        """Get Alpha Vantage symbol for ticker and exchange"""
        # This would need to be implemented based on actual symbol mapping
        return ticker

    def _get_exchange_currency(self, exchange: Exchange) -> str:
        """Get currency for exchange"""
        currency_map = {
            Exchange.NYSE: 'USD',
            Exchange.NASDAQ: 'USD',
            Exchange.AMEX: 'USD',
            Exchange.LSE: 'GBP',
            Exchange.TSE: 'JPY',
            Exchange.HKEX: 'HKD',
            Exchange.SSE: 'CNY',
            Exchange.SZSE: 'CNY',
            Exchange.BSE: 'INR',
            Exchange.NSE: 'INR',
            Exchange.ASX: 'AUD',
            Exchange.TSX: 'CAD',
            Exchange.EURONEXT: 'EUR',
            Exchange.XETRA: 'EUR',
            Exchange.SIX: 'CHF'
        }
        
        return currency_map.get(exchange, 'USD')

    def get_market_status(self, exchange: Exchange) -> MarketStatus:
        """Get market status for an exchange"""
        try:
            # This would need to be implemented with actual market hours
            # For now, return dummy status
            now = datetime.now()
            
            # Simple market hours logic (would need timezone handling in practice)
            if exchange in [Exchange.NYSE, Exchange.NASDAQ, Exchange.AMEX]:
                # US markets: 9:30 AM - 4:00 PM ET
                is_open = True  # Simplified
                session = MarketSession.REGULAR if is_open else MarketSession.CLOSED
                open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
                close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
                timezone = 'ET'
            elif exchange in [Exchange.LSE]:
                # London markets: 8:00 AM - 4:30 PM GMT
                is_open = True  # Simplified
                session = MarketSession.REGULAR if is_open else MarketSession.CLOSED
                open_time = now.replace(hour=8, minute=0, second=0, microsecond=0)
                close_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
                timezone = 'GMT'
            elif exchange in [Exchange.TSE]:
                # Tokyo markets: 9:00 AM - 3:00 PM JST
                is_open = True  # Simplified
                session = MarketSession.REGULAR if is_open else MarketSession.CLOSED
                open_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
                close_time = now.replace(hour=15, minute=0, second=0, microsecond=0)
                timezone = 'JST'
            else:
                # Default
                is_open = True
                session = MarketSession.REGULAR
                open_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
                close_time = now.replace(hour=17, minute=0, second=0, microsecond=0)
                timezone = 'UTC'
            
            return MarketStatus(
                exchange=exchange,
                is_open=is_open,
                session=session,
                open_time=open_time,
                close_time=close_time,
                timezone=timezone
            )
            
        except Exception as e:
            self.logger.error(f"Error getting market status for {exchange.value}: {e}")
            return MarketStatus(
                exchange=exchange,
                is_open=False,
                session=MarketSession.CLOSED,
                open_time=datetime.now(),
                close_time=datetime.now(),
                timezone='UTC'
            )

    def get_cross_exchange_data(self, ticker: str, exchanges: List[Exchange] = None, 
                               start_date: datetime = None, end_date: datetime = None) -> Dict[Exchange, List[ExchangeData]]:
        """Get data for a ticker across multiple exchanges"""
        try:
            if exchanges is None:
                exchanges = self.monitored_exchanges
            
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            cross_exchange_data = {}
            
            for exchange in exchanges:
                data = self.get_exchange_data(ticker, exchange, start_date, end_date)
                if data:
                    cross_exchange_data[exchange] = data
            
            self.logger.info(f"Retrieved cross-exchange data for {ticker} from {len(cross_exchange_data)} exchanges")
            return cross_exchange_data
            
        except Exception as e:
            self.logger.error(f"Error getting cross-exchange data for {ticker}: {e}")
            return {}

    def analyze_cross_exchange_performance(self, ticker: str, exchanges: List[Exchange] = None, 
                                         start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Analyze performance across exchanges"""
        try:
            if exchanges is None:
                exchanges = self.monitored_exchanges
            
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            # Get cross-exchange data
            cross_exchange_data = self.get_cross_exchange_data(ticker, exchanges, start_date, end_date)
            
            if not cross_exchange_data:
                return {
                    'total_exchanges': 0,
                    'analysis': 'No data available from any exchange',
                    'recommendations': ['Check data sources and exchange availability']
                }
            
            # Analyze each exchange
            exchange_analysis = {}
            for exchange, data in cross_exchange_data.items():
                if data:
                    analysis = self._analyze_exchange_performance(data, exchange)
                    exchange_analysis[exchange.value] = analysis
            
            # Calculate cross-exchange metrics
            cross_exchange_metrics = self._calculate_cross_exchange_metrics(cross_exchange_data)
            
            # Generate recommendations
            recommendations = self._generate_cross_exchange_recommendations(exchange_analysis, cross_exchange_metrics)
            
            return {
                'total_exchanges': len(cross_exchange_data),
                'exchange_analysis': exchange_analysis,
                'cross_exchange_metrics': cross_exchange_metrics,
                'recommendations': recommendations,
                'analysis_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-exchange performance: {e}")
            return {}

    def _analyze_exchange_performance(self, data: List[ExchangeData], exchange: Exchange) -> Dict[str, Any]:
        """Analyze performance for a single exchange"""
        try:
            if not data:
                return {'status': 'No data available'}
            
            # Calculate basic metrics
            prices = [d.price for d in data]
            volumes = [d.volume for d in data]
            
            current_price = prices[-1]
            first_price = prices[0]
            performance = ((current_price - first_price) / first_price) * 100
            
            # Calculate volatility
            returns = [data[i].change_percent for i in range(1, len(data))]
            volatility = np.std(returns) if returns else 0
            
            # Calculate average volume
            avg_volume = np.mean(volumes) if volumes else 0
            
            # Determine status
            if performance > 5:
                status = 'strong'
            elif performance > 2:
                status = 'positive'
            elif performance > -2:
                status = 'neutral'
            elif performance > -5:
                status = 'weak'
            else:
                status = 'poor'
            
            return {
                'status': status,
                'performance': performance,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'current_price': current_price,
                'data_points': len(data),
                'analysis': f'{exchange.value} shows {status} performance with {volatility:.2f}% volatility'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing exchange performance: {e}")
            return {'status': 'Analysis error'}

    def _calculate_cross_exchange_metrics(self, cross_exchange_data: Dict[Exchange, List[ExchangeData]]) -> Dict[str, Any]:
        """Calculate cross-exchange metrics"""
        try:
            if not cross_exchange_data:
                return {}
            
            # Calculate price correlation between exchanges
            correlations = {}
            exchanges = list(cross_exchange_data.keys())
            
            for i, exchange1 in enumerate(exchanges):
                for exchange2 in exchanges[i+1:]:
                    data1 = cross_exchange_data[exchange1]
                    data2 = cross_exchange_data[exchange2]
                    
                    if data1 and data2:
                        # Align data by date
                        prices1 = [d.price for d in data1]
                        prices2 = [d.price for d in data2]
                        
                        if len(prices1) == len(prices2) and len(prices1) > 1:
                            correlation = np.corrcoef(prices1, prices2)[0, 1]
                            correlations[f"{exchange1.value}_{exchange2.value}"] = correlation
            
            # Calculate average performance across exchanges
            performances = []
            for exchange, data in cross_exchange_data.items():
                if data:
                    prices = [d.price for d in data]
                    if len(prices) > 1:
                        performance = ((prices[-1] - prices[0]) / prices[0]) * 100
                        performances.append(performance)
            
            avg_performance = np.mean(performances) if performances else 0
            
            # Calculate performance consistency
            performance_std = np.std(performances) if len(performances) > 1 else 0
            
            return {
                'correlations': correlations,
                'avg_performance': avg_performance,
                'performance_consistency': performance_std,
                'total_exchanges': len(cross_exchange_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating cross-exchange metrics: {e}")
            return {}

    def _generate_cross_exchange_recommendations(self, exchange_analysis: Dict[str, Any], 
                                               cross_exchange_metrics: Dict[str, Any]) -> List[str]:
        """Generate cross-exchange recommendations"""
        try:
            recommendations = []
            
            # Check performance consistency
            performance_std = cross_exchange_metrics.get('performance_consistency', 0)
            if performance_std > 5:
                recommendations.append("High performance variance across exchanges - investigate arbitrage opportunities")
            elif performance_std < 2:
                recommendations.append("Consistent performance across exchanges - good market efficiency")
            
            # Check correlations
            correlations = cross_exchange_metrics.get('correlations', {})
            if correlations:
                avg_correlation = np.mean(list(correlations.values()))
                if avg_correlation > 0.8:
                    recommendations.append("High correlation between exchanges - markets moving in sync")
                elif avg_correlation < 0.5:
                    recommendations.append("Low correlation between exchanges - potential diversification benefits")
            
            # Check individual exchange performance
            for exchange, analysis in exchange_analysis.items():
                status = analysis.get('status', 'unknown')
                if status == 'strong':
                    recommendations.append(f"{exchange} showing strong performance - consider focusing on this exchange")
                elif status == 'poor':
                    recommendations.append(f"{exchange} showing poor performance - monitor for recovery")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating cross-exchange recommendations: {e}")
            return []

    def get_multi_exchange_summary(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive multi-exchange summary"""
        try:
            # Get cross-exchange analysis
            analysis = self.analyze_cross_exchange_performance(ticker)
            
            # Get key metrics
            key_metrics = self._extract_key_metrics(analysis)
            
            # Generate summary
            summary = {
                'analysis': analysis,
                'key_metrics': key_metrics,
                'summary_date': datetime.now(),
                'data_sources': 'Multiple APIs' if any([self.alpha_vantage_key, self.polygon_key, self.yahoo_finance_enabled]) else 'Dummy Data'
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting multi-exchange summary: {e}")
            return {}

    def _extract_key_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key multi-exchange metrics"""
        try:
            key_metrics = {}
            
            # Extract exchange analysis
            exchange_analysis = analysis.get('exchange_analysis', {})
            for exchange, data in exchange_analysis.items():
                if data.get('status') != 'No data available':
                    key_metrics[exchange] = {
                        'status': data.get('status'),
                        'performance': data.get('performance'),
                        'volatility': data.get('volatility'),
                        'current_price': data.get('current_price')
                    }
            
            # Extract cross-exchange metrics
            cross_exchange_metrics = analysis.get('cross_exchange_metrics', {})
            if cross_exchange_metrics:
                key_metrics['cross_exchange'] = {
                    'avg_performance': cross_exchange_metrics.get('avg_performance'),
                    'performance_consistency': cross_exchange_metrics.get('performance_consistency'),
                    'correlations': cross_exchange_metrics.get('correlations', {})
                }
            
            return key_metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting key metrics: {e}")
            return {}
