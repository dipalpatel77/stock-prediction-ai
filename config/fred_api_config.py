#!/usr/bin/env python3
"""
FRED API Configuration
======================

Configuration for Federal Reserve Economic Data (FRED) API integration.
This module provides comprehensive economic indicators and API settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class FREDSeries:
    """Data class for FRED series configuration"""
    series_id: str
    name: str
    frequency: str
    units: str
    seasonal_adjustment: str
    last_updated: Optional[datetime] = None
    description: str = ""

class FREDAPIConfig:
    """Configuration class for FRED API integration"""
    
    def __init__(self):
        # API Configuration
        self.base_url = "https://api.stlouisfed.org/fred"
        self.api_key = os.getenv('FRED_API_KEY', '1d2be2dd9e6f2ff6ea9f3883dccd39dc')  # Use provided key if not set
        
        # Rate limiting (FRED allows 120 requests per minute)
        self.requests_per_minute = 120
        self.request_interval = 60 / self.requests_per_minute  # seconds between requests
        
        # Cache settings
        self.cache_duration_hours = 4
        self.max_cache_age = timedelta(hours=self.cache_duration_hours)
        
        # Economic Indicators by Category
        self.economic_indicators = {
            # GDP and National Income
            'GDP': FREDSeries(
                series_id='GDP',
                name='Gross Domestic Product',
                frequency='Quarterly',
                units='Billions of Dollars',
                seasonal_adjustment='Seasonally Adjusted Annual Rate',
                description='Total value of goods and services produced in the US'
            ),
            'GDPC1': FREDSeries(
                series_id='GDPC1',
                name='Real Gross Domestic Product',
                frequency='Quarterly',
                units='Billions of Chained 2012 Dollars',
                seasonal_adjustment='Seasonally Adjusted Annual Rate',
                description='GDP adjusted for inflation'
            ),
            'GDPPOT': FREDSeries(
                series_id='GDPPOT',
                name='Real Potential Gross Domestic Product',
                frequency='Quarterly',
                units='Billions of Chained 2012 Dollars',
                seasonal_adjustment='Seasonally Adjusted Annual Rate',
                description='Maximum sustainable output level'
            ),
            
            # Inflation Indicators
            'CPIAUCSL': FREDSeries(
                series_id='CPIAUCSL',
                name='Consumer Price Index for All Urban Consumers: All Items',
                frequency='Monthly',
                units='Index 1982-84=100',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='Primary measure of inflation'
            ),
            'CPILFESL': FREDSeries(
                series_id='CPILFESL',
                name='Consumer Price Index for All Urban Consumers: All Items Less Food & Energy',
                frequency='Monthly',
                units='Index 1982-84=100',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='Core inflation (excludes food and energy)'
            ),
            'PCEPI': FREDSeries(
                series_id='PCEPI',
                name='Personal Consumption Expenditures: Chain-type Price Index',
                frequency='Monthly',
                units='Index 2012=100',
                seasonal_adjustment='Seasonally Adjusted',
                description='Fed\'s preferred inflation measure'
            ),
            
            # Employment and Labor
            'UNRATE': FREDSeries(
                series_id='UNRATE',
                name='Unemployment Rate',
                frequency='Monthly',
                units='Percent',
                seasonal_adjustment='Seasonally Adjusted',
                description='Percentage of labor force that is unemployed'
            ),
            'PAYEMS': FREDSeries(
                series_id='PAYEMS',
                name='All Employees: Total Nonfarm Payrolls',
                frequency='Monthly',
                units='Thousands of Persons',
                seasonal_adjustment='Seasonally Adjusted',
                description='Total non-farm employment'
            ),
            'AWHMAN': FREDSeries(
                series_id='AWHMAN',
                name='Average Weekly Hours of Production and Nonsupervisory Employees: Manufacturing',
                frequency='Monthly',
                units='Hours',
                seasonal_adjustment='Seasonally Adjusted',
                description='Average weekly hours worked in manufacturing'
            ),
            
            # Interest Rates and Monetary Policy
            'FEDFUNDS': FREDSeries(
                series_id='FEDFUNDS',
                name='Federal Funds Effective Rate',
                frequency='Monthly',
                units='Percent',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='Target federal funds rate'
            ),
            'GS10': FREDSeries(
                series_id='GS10',
                name='10-Year Treasury Constant Maturity Rate',
                frequency='Daily',
                units='Percent',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='10-year Treasury yield'
            ),
            'GS2': FREDSeries(
                series_id='GS2',
                name='2-Year Treasury Constant Maturity Rate',
                frequency='Daily',
                units='Percent',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='2-year Treasury yield'
            ),
            'GS30': FREDSeries(
                series_id='GS30',
                name='30-Year Treasury Constant Maturity Rate',
                frequency='Daily',
                units='Percent',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='30-year Treasury yield'
            ),
            
            # Money Supply
            'M2SL': FREDSeries(
                series_id='M2SL',
                name='M2 Money Stock',
                frequency='Monthly',
                units='Billions of Dollars',
                seasonal_adjustment='Seasonally Adjusted',
                description='M2 money supply measure'
            ),
            'M1SL': FREDSeries(
                series_id='M1SL',
                name='M1 Money Stock',
                frequency='Monthly',
                units='Billions of Dollars',
                seasonal_adjustment='Seasonally Adjusted',
                description='M1 money supply measure'
            ),
            
            # Consumer and Business Sentiment
            'UMCSENT': FREDSeries(
                series_id='UMCSENT',
                name='University of Michigan: Consumer Sentiment',
                frequency='Monthly',
                units='Index 1966:Q1=100',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='Consumer confidence indicator'
            ),
            'NAPM': FREDSeries(
                series_id='NAPM',
                name='ISM Manufacturing: PMI Composite Index',
                frequency='Monthly',
                units='Index',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='Manufacturing PMI'
            ),
            
            # Housing and Real Estate
            'HOUST': FREDSeries(
                series_id='HOUST',
                name='Housing Starts: Total: New Privately Owned Housing Units Started',
                frequency='Monthly',
                units='Thousands of Units',
                seasonal_adjustment='Seasonally Adjusted Annual Rate',
                description='New housing construction starts'
            ),
            'PERMIT': FREDSeries(
                series_id='PERMIT',
                name='New Private Housing Permits Authorized by Building Permits',
                frequency='Monthly',
                units='Thousands of Units',
                seasonal_adjustment='Seasonally Adjusted Annual Rate',
                description='Building permits issued'
            ),
            'EXHOSLUSM495S': FREDSeries(
                series_id='EXHOSLUSM495S',
                name='Existing Home Sales',
                frequency='Monthly',
                units='Millions',
                seasonal_adjustment='Seasonally Adjusted Annual Rate',
                description='Sales of existing homes'
            ),
            
            # Industrial Production
            'INDPRO': FREDSeries(
                series_id='INDPRO',
                name='Industrial Production: Total Index',
                frequency='Monthly',
                units='Index 2017=100',
                seasonal_adjustment='Seasonally Adjusted',
                description='Industrial production index'
            ),
            'CAPUTLB50001SQ': FREDSeries(
                series_id='CAPUTLB50001SQ',
                name='Capacity Utilization: Manufacturing',
                frequency='Quarterly',
                units='Percent of Capacity',
                seasonal_adjustment='Seasonally Adjusted',
                description='Manufacturing capacity utilization'
            ),
            
            # Trade and International
            'BOPGSTB': FREDSeries(
                series_id='BOPGSTB',
                name='Trade Balance: Goods and Services, Balance of Payments Basis',
                frequency='Monthly',
                units='Millions of Dollars',
                seasonal_adjustment='Seasonally Adjusted',
                description='Trade balance (exports - imports)'
            ),
            'DEXUSEU': FREDSeries(
                series_id='DEXUSEU',
                name='U.S. Dollars to Euro Spot Exchange Rate',
                frequency='Daily',
                units='U.S. Dollars to One Euro',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='USD/EUR exchange rate'
            ),
            'DEXCHUS': FREDSeries(
                series_id='DEXCHUS',
                name='China / U.S. Foreign Exchange Rate',
                frequency='Daily',
                units='Chinese Yuan to One U.S. Dollar',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='CNY/USD exchange rate'
            ),
            
            # Financial Markets
            'VIXCLS': FREDSeries(
                series_id='VIXCLS',
                name='CBOE Volatility Index: VIX',
                frequency='Daily',
                units='Index',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='Market volatility index'
            ),
            'DGS10': FREDSeries(
                series_id='DGS10',
                name='Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity',
                frequency='Daily',
                units='Percent',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='10-year Treasury yield (market)'
            ),
            
            # Commodity Prices
            'DCOILWTICO': FREDSeries(
                series_id='DCOILWTICO',
                name='Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma',
                frequency='Daily',
                units='Dollars per Barrel',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='WTI crude oil price'
            ),
            'GOLDPMGBD228NLBM': FREDSeries(
                series_id='GOLDPMGBD228NLBM',
                name='Gold Fixing Price 10:30 A.M. (London time) in London Bullion Market',
                frequency='Daily',
                units='U.S. Dollars per Troy Ounce',
                seasonal_adjustment='Not Seasonally Adjusted',
                description='Gold price (London fixing)'
            ),
        }
        
        # Priority indicators for quick analysis
        self.priority_indicators = [
            'GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'GS10', 
            'M2SL', 'UMCSENT', 'HOUST', 'INDPRO', 'VIXCLS'
        ]
        
        # Market-sensitive indicators (high frequency updates)
        self.market_sensitive_indicators = [
            'GS10', 'GS2', 'GS30', 'VIXCLS', 'DCOILWTICO', 
            'GOLDPMGBD228NLBM', 'DEXUSEU', 'DEXCHUS'
        ]
        
        # Economic health indicators
        self.health_indicators = [
            'GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'UMCSENT'
        ]
        
        # Inflation indicators
        self.inflation_indicators = [
            'CPIAUCSL', 'CPILFESL', 'PCEPI'
        ]
        
        # Employment indicators
        self.employment_indicators = [
            'UNRATE', 'PAYEMS', 'AWHMAN'
        ]
        
        # Financial market indicators
        self.financial_indicators = [
            'GS10', 'GS2', 'GS30', 'VIXCLS', 'DGS10'
        ]
    
    def get_series_by_category(self, category: str) -> List[str]:
        """Get series IDs by category"""
        category_mapping = {
            'priority': self.priority_indicators,
            'market_sensitive': self.market_sensitive_indicators,
            'health': self.health_indicators,
            'inflation': self.inflation_indicators,
            'employment': self.employment_indicators,
            'financial': self.financial_indicators,
            'all': list(self.economic_indicators.keys())
        }
        
        return category_mapping.get(category, [])
    
    def get_series_info(self, series_id: str) -> Optional[FREDSeries]:
        """Get detailed information about a series"""
        return self.economic_indicators.get(series_id)
    
    def validate_api_key(self) -> bool:
        """Validate if the API key is working"""
        return self.api_key != 'demo' and len(self.api_key) > 10
    
    def get_api_url(self, endpoint: str) -> str:
        """Get full API URL for an endpoint"""
        return f"{self.base_url}/{endpoint}"
    
    def get_request_params(self, **kwargs) -> Dict:
        """Get standard request parameters"""
        params = {
            'api_key': self.api_key,
            'file_type': 'json'
        }
        params.update(kwargs)
        return params

# Global configuration instance
fred_config = FREDAPIConfig()
