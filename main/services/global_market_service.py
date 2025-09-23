"""
Global Market Service
Handles global market data and analysis
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

class MarketRegion(Enum):
    """Market region enumeration"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    EMERGING_MARKETS = "emerging_markets"
    GLOBAL = "global"

class MarketIndex(Enum):
    """Market index enumeration"""
    SP500 = "SP500"
    NASDAQ = "NASDAQ"
    DOW_JONES = "DOW_JONES"
    FTSE100 = "FTSE100"
    DAX = "DAX"
    CAC40 = "CAC40"
    NIKKEI225 = "NIKKEI225"
    HANG_SENG = "HANG_SENG"
    SHANGHAI_COMPOSITE = "SHANGHAI_COMPOSITE"
    BSE_SENSEX = "BSE_SENSEX"
    ASX200 = "ASX200"
    TSX = "TSX"

@dataclass
class MarketData:
    """Market data structure"""
    index: str
    region: MarketRegion
    value: float
    change: float
    change_percent: float
    volume: float
    timestamp: datetime
    currency: str

@dataclass
class MarketSentiment:
    """Market sentiment data structure"""
    region: MarketRegion
    sentiment_score: float
    fear_greed_index: float
    volatility: float
    trend: str
    timestamp: datetime

class GlobalMarketService:
    """Service for global market analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.market_cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # API configurations
        self.alpha_vantage_key = self.config.get('alpha_vantage_key')
        self.polygon_key = self.config.get('polygon_key')
        self.yahoo_finance_enabled = self.config.get('yahoo_finance_enabled', True)
        
        # Market monitoring settings
        self.monitored_indices = self.config.get('monitored_indices', [
            'SP500', 'NASDAQ', 'DOW_JONES', 'FTSE100', 'DAX', 'NIKKEI225'
        ])
        
        self.logger.info("Global Market Service initialized")

    def get_market_data(self, index: str, start_date: datetime = None, 
                       end_date: datetime = None) -> List[MarketData]:
        """Get market data for an index"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            # Check cache first
            cache_key = f"{index}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.market_cache:
                cached_data = self.market_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['data']
            
            market_data = []
            
            # Try multiple data sources
            if self.alpha_vantage_key:
                market_data.extend(self._get_alpha_vantage_data(index, start_date, end_date))
            
            if self.polygon_key:
                market_data.extend(self._get_polygon_data(index, start_date, end_date))
            
            if self.yahoo_finance_enabled:
                market_data.extend(self._get_yahoo_finance_data(index, start_date, end_date))
            
            # If no API keys, use dummy data
            if not market_data:
                market_data = self._get_dummy_data(index, start_date, end_date)
            
            # Cache results
            self.market_cache[cache_key] = {
                'data': market_data,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Retrieved {len(market_data)} market data points for {index}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {index}: {e}")
            return []

    def _get_alpha_vantage_data(self, index: str, start_date: datetime, 
                               end_date: datetime) -> List[MarketData]:
        """Get market data from Alpha Vantage"""
        try:
            # Map index to Alpha Vantage symbol
            symbol_map = {
                'SP500': 'SPY',
                'NASDAQ': 'QQQ',
                'DOW_JONES': 'DIA',
                'FTSE100': 'EWU',
                'DAX': 'EWG',
                'NIKKEI225': 'EWJ',
                'HANG_SENG': 'EWH',
                'SHANGHAI_COMPOSITE': 'FXI',
                'BSE_SENSEX': 'INDA',
                'ASX200': 'EWA',
                'TSX': 'EWC'
            }
            
            symbol = symbol_map.get(index, index)
            
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
            
            market_data = []
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                for date_str, values in time_series.items():
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    if start_date <= date <= end_date:
                        try:
                            market_data.append(MarketData(
                                index=index,
                                region=self._get_index_region(index),
                                value=float(values['4. close']),
                                change=float(values['4. close']) - float(values['1. open']),
                                change_percent=((float(values['4. close']) - float(values['1. open'])) / float(values['1. open'])) * 100,
                                volume=float(values['5. volume']),
                                timestamp=date,
                                currency=self._get_index_currency(index)
                            ))
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"Error parsing market data: {e}")
                            continue
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage data: {e}")
            return []

    def _get_polygon_data(self, index: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get market data from Polygon API"""
        try:
            # Polygon API implementation would go here
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting Polygon data: {e}")
            return []

    def _get_yahoo_finance_data(self, index: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get market data from Yahoo Finance"""
        try:
            # Yahoo Finance API implementation would go here
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting Yahoo Finance data: {e}")
            return []

    def _get_dummy_data(self, index: str, start_date: datetime, end_date: datetime) -> List[MarketData]:
        """Get dummy market data for testing"""
        try:
            market_data = []
            current_date = start_date
            
            # Base values for different indices
            base_values = {
                'SP500': 4000,
                'NASDAQ': 12000,
                'DOW_JONES': 35000,
                'FTSE100': 7500,
                'DAX': 15000,
                'NIKKEI225': 28000,
                'HANG_SENG': 25000,
                'SHANGHAI_COMPOSITE': 3500,
                'BSE_SENSEX': 60000,
                'ASX200': 7500,
                'TSX': 20000
            }
            
            base_value = base_values.get(index, 1000)
            
            while current_date <= end_date:
                # Add some randomness to the data
                random_factor = np.random.normal(1, 0.02)
                value = base_value * random_factor
                
                # Calculate change
                change = np.random.normal(0, 0.01) * value
                change_percent = (change / value) * 100
                
                market_data.append(MarketData(
                    index=index,
                    region=self._get_index_region(index),
                    value=value,
                    change=change,
                    change_percent=change_percent,
                    volume=np.random.randint(1000000, 10000000),
                    timestamp=current_date,
                    currency=self._get_index_currency(index)
                ))
                
                current_date += timedelta(days=1)
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error generating dummy data: {e}")
            return []

    def _get_index_region(self, index: str) -> MarketRegion:
        """Get region for market index"""
        region_map = {
            'SP500': MarketRegion.NORTH_AMERICA,
            'NASDAQ': MarketRegion.NORTH_AMERICA,
            'DOW_JONES': MarketRegion.NORTH_AMERICA,
            'FTSE100': MarketRegion.EUROPE,
            'DAX': MarketRegion.EUROPE,
            'CAC40': MarketRegion.EUROPE,
            'NIKKEI225': MarketRegion.ASIA_PACIFIC,
            'HANG_SENG': MarketRegion.ASIA_PACIFIC,
            'SHANGHAI_COMPOSITE': MarketRegion.ASIA_PACIFIC,
            'BSE_SENSEX': MarketRegion.EMERGING_MARKETS,
            'ASX200': MarketRegion.ASIA_PACIFIC,
            'TSX': MarketRegion.NORTH_AMERICA
        }
        
        return region_map.get(index, MarketRegion.GLOBAL)

    def _get_index_currency(self, index: str) -> str:
        """Get currency for market index"""
        currency_map = {
            'SP500': 'USD',
            'NASDAQ': 'USD',
            'DOW_JONES': 'USD',
            'FTSE100': 'GBP',
            'DAX': 'EUR',
            'CAC40': 'EUR',
            'NIKKEI225': 'JPY',
            'HANG_SENG': 'HKD',
            'SHANGHAI_COMPOSITE': 'CNY',
            'BSE_SENSEX': 'INR',
            'ASX200': 'AUD',
            'TSX': 'CAD'
        }
        
        return currency_map.get(index, 'USD')

    def get_market_sentiment(self, region: MarketRegion = None) -> MarketSentiment:
        """Get market sentiment for a region"""
        try:
            if region is None:
                region = MarketRegion.GLOBAL
            
            # Get market data for the region
            indices = self._get_region_indices(region)
            all_data = []
            
            for index in indices:
                data = self.get_market_data(index)
                if data:
                    all_data.extend(data)
            
            if not all_data:
                return MarketSentiment(
                    region=region,
                    sentiment_score=0.5,
                    fear_greed_index=0.5,
                    volatility=0.0,
                    trend='neutral',
                    timestamp=datetime.now()
                )
            
            # Calculate sentiment metrics
            sentiment_score = self._calculate_sentiment_score(all_data)
            fear_greed_index = self._calculate_fear_greed_index(all_data)
            volatility = self._calculate_volatility(all_data)
            trend = self._determine_trend(all_data)
            
            return MarketSentiment(
                region=region,
                sentiment_score=sentiment_score,
                fear_greed_index=fear_greed_index,
                volatility=volatility,
                trend=trend,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment: {e}")
            return MarketSentiment(
                region=region or MarketRegion.GLOBAL,
                sentiment_score=0.5,
                fear_greed_index=0.5,
                volatility=0.0,
                trend='neutral',
                timestamp=datetime.now()
            )

    def _get_region_indices(self, region: MarketRegion) -> List[str]:
        """Get indices for a region"""
        region_indices = {
            MarketRegion.NORTH_AMERICA: ['SP500', 'NASDAQ', 'DOW_JONES', 'TSX'],
            MarketRegion.EUROPE: ['FTSE100', 'DAX', 'CAC40'],
            MarketRegion.ASIA_PACIFIC: ['NIKKEI225', 'HANG_SENG', 'SHANGHAI_COMPOSITE', 'ASX200'],
            MarketRegion.EMERGING_MARKETS: ['BSE_SENSEX'],
            MarketRegion.GLOBAL: ['SP500', 'NASDAQ', 'FTSE100', 'DAX', 'NIKKEI225']
        }
        
        return region_indices.get(region, ['SP500'])

    def _calculate_sentiment_score(self, market_data: List[MarketData]) -> float:
        """Calculate market sentiment score"""
        try:
            if not market_data:
                return 0.5
            
            # Calculate based on recent performance
            recent_data = sorted(market_data, key=lambda x: x.timestamp)[-10:]  # Last 10 days
            
            positive_days = sum(1 for data in recent_data if data.change_percent > 0)
            total_days = len(recent_data)
            
            if total_days == 0:
                return 0.5
            
            sentiment_score = positive_days / total_days
            return min(max(sentiment_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment score: {e}")
            return 0.5

    def _calculate_fear_greed_index(self, market_data: List[MarketData]) -> float:
        """Calculate fear and greed index"""
        try:
            if not market_data:
                return 0.5
            
            # Calculate based on volatility and performance
            recent_data = sorted(market_data, key=lambda x: x.timestamp)[-20:]  # Last 20 days
            
            if len(recent_data) < 2:
                return 0.5
            
            # Calculate volatility
            returns = [data.change_percent for data in recent_data]
            volatility = np.std(returns)
            
            # Calculate average return
            avg_return = np.mean(returns)
            
            # Fear and greed calculation
            if volatility > 2.0:  # High volatility
                if avg_return > 0:
                    fear_greed = 0.6  # Greed despite volatility
                else:
                    fear_greed = 0.3  # Fear due to volatility and negative returns
            else:  # Low volatility
                if avg_return > 0:
                    fear_greed = 0.8  # Strong greed
                else:
                    fear_greed = 0.4  # Moderate fear
            
            return min(max(fear_greed, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating fear greed index: {e}")
            return 0.5

    def _calculate_volatility(self, market_data: List[MarketData]) -> float:
        """Calculate market volatility"""
        try:
            if not market_data:
                return 0.0
            
            # Calculate based on recent data
            recent_data = sorted(market_data, key=lambda x: x.timestamp)[-20:]  # Last 20 days
            
            if len(recent_data) < 2:
                return 0.0
            
            returns = [data.change_percent for data in recent_data]
            volatility = np.std(returns)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _determine_trend(self, market_data: List[MarketData]) -> str:
        """Determine market trend"""
        try:
            if not market_data:
                return 'neutral'
            
            # Calculate trend based on recent performance
            recent_data = sorted(market_data, key=lambda x: x.timestamp)[-10:]  # Last 10 days
            
            if len(recent_data) < 2:
                return 'neutral'
            
            # Calculate trend
            first_value = recent_data[0].value
            last_value = recent_data[-1].value
            
            change_percent = ((last_value - first_value) / first_value) * 100
            
            if change_percent > 2:
                return 'bullish'
            elif change_percent < -2:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"Error determining trend: {e}")
            return 'neutral'

    def analyze_global_markets(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Analyze global markets"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            # Get market data for all regions
            regions = [MarketRegion.NORTH_AMERICA, MarketRegion.EUROPE, 
                      MarketRegion.ASIA_PACIFIC, MarketRegion.EMERGING_MARKETS]
            
            market_analysis = {}
            
            for region in regions:
                # Get sentiment for region
                sentiment = self.get_market_sentiment(region)
                
                # Get market data for region
                indices = self._get_region_indices(region)
                region_data = []
                
                for index in indices:
                    data = self.get_market_data(index, start_date, end_date)
                    if data:
                        region_data.extend(data)
                
                # Analyze region
                region_analysis = self._analyze_region(region_data, sentiment)
                market_analysis[region.value] = region_analysis
            
            # Calculate global sentiment
            global_sentiment = self.get_market_sentiment(MarketRegion.GLOBAL)
            
            # Generate recommendations
            recommendations = self._generate_market_recommendations(market_analysis, global_sentiment)
            
            return {
                'regions': market_analysis,
                'global_sentiment': global_sentiment,
                'recommendations': recommendations,
                'analysis_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing global markets: {e}")
            return {}

    def _analyze_region(self, market_data: List[MarketData], sentiment: MarketSentiment) -> Dict[str, Any]:
        """Analyze market data for a region"""
        try:
            if not market_data:
                return {
                    'status': 'No data available',
                    'sentiment': sentiment,
                    'analysis': 'Insufficient data for analysis'
                }
            
            # Calculate basic metrics
            current_values = [data.value for data in market_data if data.timestamp >= datetime.now() - timedelta(days=1)]
            if current_values:
                avg_value = np.mean(current_values)
                max_value = np.max(current_values)
                min_value = np.min(current_values)
            else:
                avg_value = max_value = min_value = 0
            
            # Calculate performance
            recent_data = sorted(market_data, key=lambda x: x.timestamp)[-10:]
            if len(recent_data) >= 2:
                first_value = recent_data[0].value
                last_value = recent_data[-1].value
                performance = ((last_value - first_value) / first_value) * 100
            else:
                performance = 0
            
            # Determine status
            if performance > 2:
                status = 'strong'
            elif performance > 0:
                status = 'positive'
            elif performance > -2:
                status = 'neutral'
            else:
                status = 'weak'
            
            return {
                'status': status,
                'sentiment': sentiment,
                'performance': performance,
                'avg_value': avg_value,
                'max_value': max_value,
                'min_value': min_value,
                'data_points': len(market_data),
                'analysis': f'Region shows {status} performance with {sentiment.trend} trend'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing region: {e}")
            return {'status': 'Analysis error'}

    def _generate_market_recommendations(self, market_analysis: Dict[str, Any], 
                                       global_sentiment: MarketSentiment) -> List[str]:
        """Generate market recommendations"""
        try:
            recommendations = []
            
            # Check global sentiment
            if global_sentiment.sentiment_score > 0.7:
                recommendations.append("Global markets showing strong positive sentiment - consider growth strategies")
            elif global_sentiment.sentiment_score < 0.3:
                recommendations.append("Global markets showing negative sentiment - consider defensive strategies")
            
            # Check volatility
            if global_sentiment.volatility > 2.0:
                recommendations.append("High market volatility detected - use appropriate risk management")
            
            # Check fear and greed
            if global_sentiment.fear_greed_index > 0.8:
                recommendations.append("High greed index - consider taking profits")
            elif global_sentiment.fear_greed_index < 0.3:
                recommendations.append("High fear index - consider buying opportunities")
            
            # Check regional performance
            for region, analysis in market_analysis.items():
                if analysis.get('status') == 'strong':
                    recommendations.append(f"{region.title()} markets showing strong performance")
                elif analysis.get('status') == 'weak':
                    recommendations.append(f"{region.title()} markets showing weakness - monitor closely")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating market recommendations: {e}")
            return []

    def get_global_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive global market summary"""
        try:
            # Get global market analysis
            analysis = self.analyze_global_markets()
            
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
            self.logger.error(f"Error getting global market summary: {e}")
            return {}

    def _extract_key_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key market metrics"""
        try:
            key_metrics = {}
            
            # Extract global sentiment
            global_sentiment = analysis.get('global_sentiment', {})
            if global_sentiment:
                key_metrics['global_sentiment'] = {
                    'sentiment_score': global_sentiment.sentiment_score,
                    'fear_greed_index': global_sentiment.fear_greed_index,
                    'volatility': global_sentiment.volatility,
                    'trend': global_sentiment.trend
                }
            
            # Extract regional metrics
            regions = analysis.get('regions', {})
            for region, data in regions.items():
                if data.get('status') != 'No data available':
                    key_metrics[region] = {
                        'status': data.get('status'),
                        'performance': data.get('performance'),
                        'sentiment_score': data.get('sentiment', {}).sentiment_score
                    }
            
            return key_metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting key metrics: {e}")
            return {}
