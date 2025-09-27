"""
Interactive Data Selector
Interactive data selection with Angel One support
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveDataSelector:
    """
    Interactive data selector with Angel One support
    
    This selector provides:
    - Interactive data parameter selection
    - Angel One interval selection
    - Data source selection
    - Period selection
    - Enhanced features selection
    """
    
    def __init__(self):
        """Initialize Interactive Data Selector"""
        self.angel_one_limits = {
            'ONE_MINUTE': 30,
            'THREE_MINUTE': 60,
            'FIVE_MINUTE': 100,
            'TEN_MINUTE': 100,
            'FIFTEEN_MINUTE': 200,
            'THIRTY_MINUTE': 200,
            'ONE_HOUR': 400,
            'ONE_DAY': 2000
        }
        
        logger.info("Interactive Data Selector initialized")
    
    def get_data_parameters(self, ticker: str, is_indian: bool = False) -> Dict[str, Any]:
        """
        Get data parameters interactively
        
        Args:
            ticker: Stock ticker symbol
            is_indian: Whether it's an Indian stock
            
        Returns:
            Dictionary with data parameters
        """
        try:
            print(f"\n📊 Data Parameters for {ticker}")
            print("=" * 40)
            
            # Get period selection
            period = self._get_period_selection()
            
            # Get interval selection (different for Indian vs international)
            if is_indian:
                interval = self._get_angel_one_interval()
            else:
                interval = self._get_yahoo_interval()
            
            # Get data source preferences
            data_source = self._get_data_source_preference(is_indian)
            
            # Get enhanced features
            enhanced_features = self._get_enhanced_features()
            
            # Compile parameters
            parameters = {
                'ticker': ticker,
                'period': period,
                'interval': interval,
                'data_source': data_source,
                'enhanced_features': enhanced_features,
                'is_indian': is_indian,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Data parameters selected for {ticker}")
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to get data parameters: {e}")
            return {'error': str(e)}
    
    def _get_period_selection(self) -> str:
        """Get period selection from user"""
        try:
            print("\n📅 Data Period Selection:")
            print("-" * 25)
            print("1. 1 day")
            print("2. 5 days")
            print("3. 1 month")
            print("4. 3 months")
            print("5. 6 months")
            print("6. 1 year")
            print("7. 2 years")
            print("8. 5 years")
            print("9. Maximum available")
            
            period_map = {
                '1': '1d',
                '2': '5d',
                '3': '1mo',
                '4': '3mo',
                '5': '6mo',
                '6': '1y',
                '7': '2y',
                '8': '5y',
                '9': 'max'
            }
            
            while True:
                choice = input("Select period (1-9, default: 6): ").strip()
                
                if not choice:
                    choice = '6'  # Default to 1 year
                
                if choice in period_map:
                    period = period_map[choice]
                    print(f"✅ Selected period: {period}")
                    return period
                else:
                    print("❌ Invalid choice. Please select 1-9.")
                    
        except Exception as e:
            logger.error(f"Failed to get period selection: {e}")
            return '1y'
    
    def _get_angel_one_interval(self) -> str:
        """Get Angel One interval selection"""
        try:
            print("\n⏰ Angel One Interval Selection:")
            print("-" * 30)
            print("1. ONE_DAY (Daily) - 2000 requests")
            print("2. ONE_HOUR (Hourly) - 400 requests")
            print("3. THIRTY_MINUTE (30 min) - 200 requests")
            print("4. FIFTEEN_MINUTE (15 min) - 200 requests")
            print("5. FIVE_MINUTE (5 min) - 100 requests")
            print("6. ONE_MINUTE (1 min) - 30 requests")
            
            interval_map = {
                '1': 'ONE_DAY',
                '2': 'ONE_HOUR',
                '3': 'THIRTY_MINUTE',
                '4': 'FIFTEEN_MINUTE',
                '5': 'FIVE_MINUTE',
                '6': 'ONE_MINUTE'
            }
            
            while True:
                choice = input("Select interval (1-6, default: 1): ").strip()
                
                if not choice:
                    choice = '1'  # Default to daily
                
                if choice in interval_map:
                    interval = interval_map[choice]
                    limit = self.angel_one_limits.get(interval, 2000)
                    print(f"✅ Selected interval: {interval} (Limit: {limit} requests)")
                    return interval
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    
        except Exception as e:
            logger.error(f"Failed to get Angel One interval: {e}")
            return 'ONE_DAY'
    
    def _get_yahoo_interval(self) -> str:
        """Get Yahoo Finance interval selection"""
        try:
            print("\n⏰ Yahoo Finance Interval Selection:")
            print("-" * 35)
            print("1. 1d (Daily)")
            print("2. 1h (Hourly)")
            print("3. 30m (30 minutes)")
            print("4. 15m (15 minutes)")
            print("5. 5m (5 minutes)")
            print("6. 1m (1 minute)")
            
            interval_map = {
                '1': '1d',
                '2': '1h',
                '3': '30m',
                '4': '15m',
                '5': '5m',
                '6': '1m'
            }
            
            while True:
                choice = input("Select interval (1-6, default: 1): ").strip()
                
                if not choice:
                    choice = '1'  # Default to daily
                
                if choice in interval_map:
                    interval = interval_map[choice]
                    print(f"✅ Selected interval: {interval}")
                    return interval
                else:
                    print("❌ Invalid choice. Please select 1-6.")
                    
        except Exception as e:
            logger.error(f"Failed to get Yahoo interval: {e}")
            return '1d'
    
    def _get_data_source_preference(self, is_indian: bool) -> str:
        """Get data source preference"""
        try:
            if is_indian:
                print("\n📡 Data Source Selection:")
                print("-" * 25)
                print("1. Angel One API (Indian stocks)")
                print("2. Yahoo Finance (Fallback)")
                print("3. Both (Primary: Angel One, Fallback: Yahoo)")
                
                while True:
                    choice = input("Select data source (1-3, default: 1): ").strip()
                    
                    if not choice:
                        choice = '1'
                    
                    if choice == '1':
                        return 'angel_one'
                    elif choice == '2':
                        return 'yahoo_finance'
                    elif choice == '3':
                        return 'both'
                    else:
                        print("❌ Invalid choice. Please select 1-3.")
            else:
                print("📊 Using Yahoo Finance for international stocks")
                return 'yahoo_finance'
                
        except Exception as e:
            logger.error(f"Failed to get data source preference: {e}")
            return 'yahoo_finance'
    
    def _get_enhanced_features(self) -> Dict[str, bool]:
        """Get enhanced features selection"""
        try:
            print("\n✨ Enhanced Features Selection:")
            print("-" * 30)
            
            features = {}
            
            # Technical indicators
            tech_indicators = input("Use advanced technical indicators? (y/n, default: y): ").strip().lower()
            features['technical_indicators'] = tech_indicators != 'n'
            
            # Economic data
            economic_data = input("Include economic data? (y/n, default: y): ").strip().lower()
            features['economic_data'] = economic_data != 'n'
            
            # Sentiment analysis
            sentiment = input("Include sentiment analysis? (y/n, default: y): ").strip().lower()
            features['sentiment_analysis'] = sentiment != 'n'
            
            # News analysis
            news = input("Include news analysis? (y/n, default: n): ").strip().lower()
            features['news_analysis'] = news == 'y'
            
            # Social media analysis
            social = input("Include social media analysis? (y/n, default: n): ").strip().lower()
            features['social_analysis'] = social == 'y'
            
            # Machine learning models
            ml_models = input("Use machine learning models? (y/n, default: y): ").strip().lower()
            features['ml_models'] = ml_models != 'n'
            
            # Backtesting
            backtesting = input("Include backtesting? (y/n, default: y): ").strip().lower()
            features['backtesting'] = backtesting != 'n'
            
            # Risk analysis
            risk = input("Include risk analysis? (y/n, default: y): ").strip().lower()
            features['risk_analysis'] = risk != 'n'
            
            print(f"\n✅ Enhanced features configured:")
            for feature, enabled in features.items():
                status = "✅" if enabled else "❌"
                print(f"  {status} {feature.replace('_', ' ').title()}")
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to get enhanced features: {e}")
            return {
                'technical_indicators': True,
                'economic_data': True,
                'sentiment_analysis': True,
                'news_analysis': False,
                'social_analysis': False,
                'ml_models': True,
                'backtesting': True,
                'risk_analysis': True
            }
    
    def estimate_data_requirements(self, period: str, interval: str, is_indian: bool) -> Dict[str, Any]:
        """
        Estimate data requirements
        
        Args:
            period: Data period
            interval: Data interval
            is_indian: Whether it's an Indian stock
            
        Returns:
            Estimation results
        """
        try:
            # Period to days mapping
            period_days = {
                '1d': 1,
                '5d': 5,
                '1mo': 30,
                '3mo': 90,
                '6mo': 180,
                '1y': 365,
                '2y': 730,
                '5y': 1825,
                'max': 3650
            }
            
            # Interval to minutes mapping
            interval_minutes = {
                'ONE_MINUTE': 1,
                'FIVE_MINUTE': 5,
                'FIFTEEN_MINUTE': 15,
                'THIRTY_MINUTE': 30,
                'ONE_HOUR': 60,
                'ONE_DAY': 1440,
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '1d': 1440
            }
            
            days = period_days.get(period, 365)
            minutes_per_interval = interval_minutes.get(interval, 1440)
            
            # Calculate estimated data points
            total_minutes = days * 24 * 60
            estimated_points = total_minutes // minutes_per_interval
            
            # Check against limits
            if is_indian and interval in self.angel_one_limits:
                max_points = self.angel_one_limits[interval]
                within_limits = estimated_points <= max_points
            else:
                within_limits = True
                max_points = None
            
            estimation = {
                'period_days': days,
                'interval_minutes': minutes_per_interval,
                'estimated_points': estimated_points,
                'within_limits': within_limits,
                'max_points': max_points,
                'is_indian': is_indian
            }
            
            return estimation
            
        except Exception as e:
            logger.error(f"Failed to estimate data requirements: {e}")
            return {'error': str(e)}
    
    def get_selector_status(self) -> Dict[str, Any]:
        """
        Get selector status
        
        Returns:
            Status information
        """
        try:
            status = {
                'angel_one_limits': self.angel_one_limits,
                'initialized': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get selector status: {e}")
            return {'error': str(e)}
