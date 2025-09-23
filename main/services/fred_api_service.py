"""
FRED API Service
Handles Federal Reserve Economic Data (FRED) API integration
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

class FREDSeries(Enum):
    """FRED economic data series"""
    GDP = "GDP"
    UNEMPLOYMENT = "UNRATE"
    INFLATION = "CPIAUCSL"
    INTEREST_RATE = "FEDFUNDS"
    CONSUMER_CONFIDENCE = "UMCSENT"
    MANUFACTURING_PMI = "MANEMP"
    RETAIL_SALES = "RSAFS"
    TRADE_BALANCE = "BOPGSTB"
    MONEY_SUPPLY = "M1SL"
    HOUSING_STARTS = "HOUST"
    DURABLE_GOODS = "DGORDER"
    PERSONAL_INCOME = "PI"
    CONSUMER_SPENDING = "PCE"
    BUSINESS_INVENTORIES = "BUSINV"
    CAPACITY_UTILIZATION = "TCU"
    PRODUCTIVITY = "OPHNFB"
    LABOR_FORCE = "CLF16OV"
    WAGE_GROWTH = "AHETPI"
    CREDIT_SPREAD = "T10Y2Y"
    YIELD_CURVE = "T10Y3M"

@dataclass
class FREDDataPoint:
    """FRED data point structure"""
    series_id: str
    date: datetime
    value: float
    units: str
    frequency: str

class FREDAPIService:
    """Service for FRED API integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.fred_api_key = self.config.get('fred_api_key')
        self.base_url = "https://api.stlouisfed.org/fred"
        self.data_cache = {}
        self.cache_duration = timedelta(hours=6)
        
        if not self.fred_api_key:
            self.logger.warning("FRED API key not provided - using dummy data")
        
        self.logger.info("FRED API Service initialized")

    def get_economic_data(self, series_id: str, start_date: datetime = None, 
                         end_date: datetime = None) -> List[FREDDataPoint]:
        """Get economic data from FRED API"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now()
            
            # Check cache first
            cache_key = f"{series_id}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.data_cache:
                cached_data = self.data_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['data']
            
            if self.fred_api_key:
                data = self._get_fred_data(series_id, start_date, end_date)
            else:
                data = self._get_dummy_data(series_id, start_date, end_date)
            
            # Cache results
            self.data_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting FRED data for {series_id}: {e}")
            return []

    def _get_fred_data(self, series_id: str, start_date: datetime, 
                      end_date: datetime) -> List[FREDDataPoint]:
        """Get data from FRED API"""
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'sort_order': 'asc'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            data_points = []
            if 'observations' in data:
                for obs in data['observations']:
                    if obs.get('value') != '.':
                        try:
                            data_points.append(FREDDataPoint(
                                series_id=series_id,
                                date=datetime.strptime(obs['date'], '%Y-%m-%d'),
                                value=float(obs['value']),
                                units=obs.get('units', ''),
                                frequency=obs.get('frequency', '')
                            ))
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"Error parsing observation: {e}")
                            continue
            
            return data_points
            
        except Exception as e:
            self.logger.error(f"Error getting FRED data: {e}")
            return []

    def _get_dummy_data(self, series_id: str, start_date: datetime, 
                       end_date: datetime) -> List[FREDDataPoint]:
        """Get dummy data for testing"""
        try:
            # Generate dummy data based on series type
            data_points = []
            current_date = start_date
            
            # Base values for different series
            base_values = {
                'GDP': 20000,  # Billions
                'UNRATE': 5.0,  # Percent
                'CPIAUCSL': 250,  # Index
                'FEDFUNDS': 2.5,  # Percent
                'UMCSENT': 100,  # Index
                'MANEMP': 12,  # Millions
                'RSAFS': 500,  # Billions
                'BOPGSTB': -50,  # Billions
                'M1SL': 2000,  # Billions
                'HOUST': 1500,  # Thousands
                'DGORDER': 250,  # Billions
                'PI': 20000,  # Billions
                'PCE': 15000,  # Billions
                'BUSINV': 2000,  # Billions
                'TCU': 75,  # Percent
                'OPHNFB': 100,  # Index
                'CLF16OV': 160,  # Millions
                'AHETPI': 25,  # Dollars
                'T10Y2Y': 1.5,  # Percent
                'T10Y3M': 2.0  # Percent
            }
            
            base_value = base_values.get(series_id, 100)
            
            while current_date <= end_date:
                # Add some randomness to the data
                random_factor = np.random.normal(1, 0.05)
                value = base_value * random_factor
                
                data_points.append(FREDDataPoint(
                    series_id=series_id,
                    date=current_date,
                    value=value,
                    units='',
                    frequency='Monthly'
                ))
                
                current_date += timedelta(days=30)  # Monthly data
            
            return data_points
            
        except Exception as e:
            self.logger.error(f"Error generating dummy data: {e}")
            return []

    def get_gdp_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get GDP data"""
        return self.get_economic_data(FREDSeries.GDP.value, start_date, end_date)

    def get_unemployment_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get unemployment rate data"""
        return self.get_economic_data(FREDSeries.UNEMPLOYMENT.value, start_date, end_date)

    def get_inflation_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get inflation data"""
        return self.get_economic_data(FREDSeries.INFLATION.value, start_date, end_date)

    def get_interest_rate_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get interest rate data"""
        return self.get_economic_data(FREDSeries.INTEREST_RATE.value, start_date, end_date)

    def get_consumer_confidence_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get consumer confidence data"""
        return self.get_economic_data(FREDSeries.CONSUMER_CONFIDENCE.value, start_date, end_date)

    def get_manufacturing_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get manufacturing employment data"""
        return self.get_economic_data(FREDSeries.MANUFACTURING_PMI.value, start_date, end_date)

    def get_retail_sales_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get retail sales data"""
        return self.get_economic_data(FREDSeries.RETAIL_SALES.value, start_date, end_date)

    def get_trade_balance_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get trade balance data"""
        return self.get_economic_data(FREDSeries.TRADE_BALANCE.value, start_date, end_date)

    def get_money_supply_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get money supply data"""
        return self.get_economic_data(FREDSeries.MONEY_SUPPLY.value, start_date, end_date)

    def get_housing_starts_data(self, start_date: datetime = None, end_date: datetime = None) -> List[FREDDataPoint]:
        """Get housing starts data"""
        return self.get_economic_data(FREDSeries.HOUSING_STARTS.value, start_date, end_date)

    def analyze_economic_indicators(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Analyze multiple economic indicators"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now()
            
            # Get data for key indicators
            indicators = {
                'gdp': self.get_gdp_data(start_date, end_date),
                'unemployment': self.get_unemployment_data(start_date, end_date),
                'inflation': self.get_inflation_data(start_date, end_date),
                'interest_rate': self.get_interest_rate_data(start_date, end_date),
                'consumer_confidence': self.get_consumer_confidence_data(start_date, end_date),
                'manufacturing': self.get_manufacturing_data(start_date, end_date),
                'retail_sales': self.get_retail_sales_data(start_date, end_date),
                'trade_balance': self.get_trade_balance_data(start_date, end_date)
            }
            
            # Analyze each indicator
            analysis = {}
            for indicator, data in indicators.items():
                if data:
                    analysis[indicator] = self._analyze_indicator(data, indicator)
                else:
                    analysis[indicator] = {'status': 'No data available'}
            
            # Calculate overall economic health score
            overall_score = self._calculate_economic_health_score(analysis)
            
            # Generate recommendations
            recommendations = self._generate_economic_recommendations(analysis)
            
            return {
                'indicators': analysis,
                'overall_score': overall_score,
                'recommendations': recommendations,
                'analysis_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing economic indicators: {e}")
            return {}

    def _analyze_indicator(self, data: List[FREDDataPoint], indicator: str) -> Dict[str, Any]:
        """Analyze a single economic indicator"""
        try:
            if not data:
                return {'status': 'No data available'}
            
            values = [point.value for point in data]
            dates = [point.date for point in data]
            
            # Calculate basic statistics
            current_value = values[-1]
            previous_value = values[-2] if len(values) > 1 else current_value
            change = current_value - previous_value
            change_percent = (change / previous_value * 100) if previous_value != 0 else 0
            
            # Calculate trend
            if len(values) >= 3:
                recent_trend = np.polyfit(range(len(values[-3:])), values[-3:], 1)[0]
            else:
                recent_trend = 0
            
            # Determine status
            status = self._determine_indicator_status(indicator, current_value, change_percent, recent_trend)
            
            return {
                'current_value': current_value,
                'previous_value': previous_value,
                'change': change,
                'change_percent': change_percent,
                'trend': recent_trend,
                'status': status,
                'data_points': len(data),
                'date_range': f"{dates[0].date()} to {dates[-1].date()}"
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing indicator {indicator}: {e}")
            return {'status': 'Analysis error'}

    def _determine_indicator_status(self, indicator: str, current_value: float, 
                                  change_percent: float, trend: float) -> str:
        """Determine the status of an economic indicator"""
        try:
            if indicator == 'gdp':
                if change_percent > 2:
                    return 'strong_growth'
                elif change_percent > 0:
                    return 'moderate_growth'
                elif change_percent > -2:
                    return 'slow_growth'
                else:
                    return 'recession'
            
            elif indicator == 'unemployment':
                if current_value < 4:
                    return 'very_low'
                elif current_value < 6:
                    return 'low'
                elif current_value < 8:
                    return 'moderate'
                else:
                    return 'high'
            
            elif indicator == 'inflation':
                if current_value < 2:
                    return 'low'
                elif current_value < 4:
                    return 'moderate'
                elif current_value < 6:
                    return 'high'
                else:
                    return 'very_high'
            
            elif indicator == 'interest_rate':
                if current_value < 2:
                    return 'very_low'
                elif current_value < 4:
                    return 'low'
                elif current_value < 6:
                    return 'moderate'
                else:
                    return 'high'
            
            elif indicator == 'consumer_confidence':
                if current_value > 100:
                    return 'optimistic'
                elif current_value > 80:
                    return 'positive'
                elif current_value > 60:
                    return 'neutral'
                else:
                    return 'pessimistic'
            
            else:
                # Generic analysis based on trend
                if trend > 0:
                    return 'improving'
                elif trend < 0:
                    return 'declining'
                else:
                    return 'stable'
                    
        except Exception as e:
            self.logger.error(f"Error determining indicator status: {e}")
            return 'unknown'

    def _calculate_economic_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall economic health score"""
        try:
            scores = []
            
            for indicator, data in analysis.items():
                if data.get('status') == 'No data available':
                    continue
                
                status = data.get('status', 'unknown')
                score = self._get_status_score(status)
                scores.append(score)
            
            if not scores:
                return 0.5  # Neutral if no data
            
            return np.mean(scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating economic health score: {e}")
            return 0.5

    def _get_status_score(self, status: str) -> float:
        """Get numerical score for status"""
        status_scores = {
            'strong_growth': 1.0,
            'moderate_growth': 0.8,
            'slow_growth': 0.6,
            'recession': 0.2,
            'very_low': 1.0,
            'low': 0.8,
            'moderate': 0.6,
            'high': 0.4,
            'very_high': 0.2,
            'low': 1.0,
            'moderate': 0.6,
            'high': 0.4,
            'very_high': 0.2,
            'very_low': 0.2,
            'low': 0.4,
            'moderate': 0.6,
            'high': 0.8,
            'optimistic': 1.0,
            'positive': 0.8,
            'neutral': 0.6,
            'pessimistic': 0.4,
            'improving': 0.8,
            'declining': 0.4,
            'stable': 0.6,
            'unknown': 0.5
        }
        
        return status_scores.get(status, 0.5)

    def _generate_economic_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate economic recommendations"""
        try:
            recommendations = []
            
            # Check GDP
            gdp_data = analysis.get('gdp', {})
            if gdp_data.get('status') == 'recession':
                recommendations.append("Economic recession detected - consider defensive strategies")
            elif gdp_data.get('status') == 'strong_growth':
                recommendations.append("Strong economic growth - consider growth strategies")
            
            # Check unemployment
            unemployment_data = analysis.get('unemployment', {})
            if unemployment_data.get('status') == 'high':
                recommendations.append("High unemployment - monitor consumer spending")
            elif unemployment_data.get('status') == 'very_low':
                recommendations.append("Very low unemployment - watch for wage inflation")
            
            # Check inflation
            inflation_data = analysis.get('inflation', {})
            if inflation_data.get('status') == 'very_high':
                recommendations.append("Very high inflation - consider inflation hedges")
            elif inflation_data.get('status') == 'low':
                recommendations.append("Low inflation - favorable for growth")
            
            # Check interest rates
            interest_data = analysis.get('interest_rate', {})
            if interest_data.get('status') == 'high':
                recommendations.append("High interest rates - monitor borrowing costs")
            elif interest_data.get('status') == 'very_low':
                recommendations.append("Very low interest rates - favorable for borrowing")
            
            # Check consumer confidence
            confidence_data = analysis.get('consumer_confidence', {})
            if confidence_data.get('status') == 'pessimistic':
                recommendations.append("Low consumer confidence - monitor retail sales")
            elif confidence_data.get('status') == 'optimistic':
                recommendations.append("High consumer confidence - positive for consumer stocks")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating economic recommendations: {e}")
            return []

    def get_economic_summary(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Get comprehensive economic summary"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now()
            
            # Get analysis
            analysis = self.analyze_economic_indicators(start_date, end_date)
            
            # Get key metrics
            key_metrics = self._extract_key_metrics(analysis)
            
            # Generate summary
            summary = {
                'analysis': analysis,
                'key_metrics': key_metrics,
                'summary_date': datetime.now(),
                'data_sources': 'FRED API' if self.fred_api_key else 'Dummy Data'
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting economic summary: {e}")
            return {}

    def _extract_key_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key economic metrics"""
        try:
            key_metrics = {}
            
            for indicator, data in analysis.get('indicators', {}).items():
                if data.get('status') != 'No data available':
                    key_metrics[indicator] = {
                        'current_value': data.get('current_value'),
                        'change_percent': data.get('change_percent'),
                        'status': data.get('status')
                    }
            
            return key_metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting key metrics: {e}")
            return {}
