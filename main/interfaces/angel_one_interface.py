"""
Angel One Interface
Interface for Angel One API configuration and management
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Angel One manager
from ..services.angel_one_manager import AngelOneManager


class AngelOneInterface:
    """
    Interface for Angel One API configuration and management
    
    This interface provides:
    - Angel One API configuration
    - Connection testing
    - Rate limiting management
    - Exchange selection (NSE/BSE)
    - Interval management
    """
    
    def __init__(self):
        """Initialize Angel One Interface"""
        self.angel_manager = None
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
        
        logger.info("Angel One Interface initialized")
    
    def configure_angel_one(self, ticker: str, config: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Configure Angel One API for Indian stocks
        
        Args:
            ticker: Stock ticker symbol
            config: Optional configuration dictionary for testing
            
        Returns:
            Configuration dictionary or None if configuration failed
        """
        try:
            print(f"\n🇮🇳 Indian stock detected: {ticker}")
            
            # Use provided config for testing, otherwise get user input
            if config:
                print("📡 Angel One API Configuration:")
                print("=" * 50)
                print("✅ Using provided configuration for testing")
                angel_config = config
            else:
                print("📡 Angel One API Configuration:")
                print("=" * 50)
                
                # Ask if user wants to use Angel One
                use_angel_one = input("Use Angel One API for data? (y/n, default: y): ").strip().lower()
                if use_angel_one == 'n':
                    print("📊 Using Yahoo Finance as fallback")
                    return None
            
            if not config:
                print("\n🔧 Angel One Configuration:")
                print("-" * 30)
                
                # Use actual Angel One credentials
                api_key = "1TKgQThc "  # Actual API key
                api_secret = "D54448"  # Client code
                access_token = "2251"  # Client PIN
                totp_secret = "NP4SAXOKMTJQZ4KZP2TBTYXRCE"  # TOTP secret
                
                print(f"✅ Using actual Angel One credentials")
                print(f"   API Key: {api_key[:8]}...")
                print(f"   Client Code: {api_secret}")
                print(f"   Client PIN: {access_token}")
                print(f"   TOTP Secret: {totp_secret[:8]}...")
                
                # Get exchange selection
                print("\n📈 Exchange Selection:")
                print("1. NSE (National Stock Exchange)")
                print("2. BSE (Bombay Stock Exchange)")
                exchange_choice = input("Select exchange (1/2, default: 1): ").strip()
                
                if exchange_choice == '2':
                    exchange = "BSE"
                else:
                    exchange = "NSE"
                
                # Set default interval for comprehensive data download
                print("\n📊 Comprehensive Data Download:")
                print("Will download data for all intervals automatically:")
                print("• ONE_MINUTE (30 days max)")
                print("• THREE_MINUTE (60 days max)")
                print("• FIVE_MINUTE (100 days max)")
                print("• TEN_MINUTE (100 days max)")
                print("• FIFTEEN_MINUTE (200 days max)")
                print("• THIRTY_MINUTE (200 days max)")
                print("• ONE_HOUR (400 days max)")
                print("• ONE_DAY (2000 days max)")
                
                # Use default interval (will download all intervals)
                interval = 'ONE_DAY'
            
            if config:
                # Use provided config
                angel_config = config
            else:
                # Build configuration from user input
                
                angel_config = {
                    'api_key': api_key,
                    'api_secret': api_secret,  # This is actually client_code
                    'access_token': access_token,  # This is actually client_pin
                    'totp_secret': totp_secret,
                    'exchange': exchange,
                    'interval': interval,
                    'ticker': ticker,
                    'configured_at': datetime.now().isoformat()
                }
            
            # Initialize Angel One manager with config
            self.angel_manager = AngelOneManager(angel_config)
            
            print(f"\n✅ Angel One configured for {angel_config['exchange']} exchange")
            print(f"📊 Data interval: {angel_config['interval']}")
            print(f"🔑 API Key: {'*' * 8 + angel_config['api_key'][-4:] if len(angel_config['api_key']) > 4 else '****'}")
            
            return angel_config
            
        except Exception as e:
            logger.error(f"Angel One configuration failed: {e}")
            print(f"❌ Angel One configuration failed: {e}")
            return None
    
    def test_angel_one_connection(self, config: Dict[str, Any]) -> bool:
        """
        Test Angel One API connection
        
        Args:
            config: Angel One configuration
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            print("\n🔍 Testing Angel One API connection...")
            
            if not self.angel_manager:
                self.angel_manager = AngelOneManager(config)
            
            # Test connection with a simple API call
            test_result = self.angel_manager.test_connection()
            
            if test_result:
                print("✅ Angel One API connection successful")
                print(f"📡 Connected to {config.get('exchange', 'NSE')} exchange")
                return True
            else:
                print("❌ Angel One API connection failed")
                print("⚠️ Falling back to Yahoo Finance")
                return False
                
        except Exception as e:
            logger.error(f"Angel One connection test failed: {e}")
            print(f"❌ Angel One connection test failed: {e}")
            return False
    
    def get_angel_one_limits(self) -> Dict[str, int]:
        """
        Get Angel One API rate limits
        
        Returns:
            Dictionary with rate limits for different intervals
        """
        return self.angel_one_limits.copy()
    
    def display_angel_one_info(self, config: Dict[str, Any]):
        """
        Display Angel One configuration information
        
        Args:
            config: Angel One configuration
        """
        try:
            print("\n📊 Angel One Configuration Info:")
            print("=" * 40)
            print(f"Exchange: {config.get('exchange', 'NSE')}")
            print(f"Interval: {config.get('interval', 'ONE_DAY')}")
            print(f"Ticker: {config.get('ticker', 'N/A')}")
            print(f"Configured: {config.get('configured_at', 'N/A')}")
            
            # Display rate limits
            limits = self.get_angel_one_limits()
            print(f"\nRate Limits:")
            for interval, limit in limits.items():
                print(f"  {interval}: {limit} requests")
                
        except Exception as e:
            logger.error(f"Failed to display Angel One info: {e}")
    
    def validate_angel_one_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate Angel One configuration
        
        Args:
            config: Angel One configuration
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            required_fields = ['api_key', 'api_secret', 'access_token', 'exchange', 'interval']
            
            for field in required_fields:
                if field not in config or not config[field]:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate exchange
            if config['exchange'] not in ['NSE', 'BSE']:
                logger.warning(f"Invalid exchange: {config['exchange']}")
                return False
            
            # Validate interval
            if config['interval'] not in self.angel_one_limits:
                logger.warning(f"Invalid interval: {config['interval']}")
                return False
            
            logger.info("Angel One configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_recommended_interval(self, analysis_type: str) -> str:
        """
        Get recommended interval based on analysis type
        
        Args:
            analysis_type: Type of analysis (short_term, mid_term, long_term)
            
        Returns:
            Recommended interval
        """
        try:
            recommendations = {
                'short_term': 'FIFTEEN_MINUTE',
                'mid_term': 'ONE_HOUR',
                'long_term': 'ONE_DAY',
                'intraday': 'FIVE_MINUTE',
                'swing': 'ONE_DAY',
                'position': 'ONE_DAY'
            }
            
            return recommendations.get(analysis_type, 'ONE_DAY')
            
        except Exception as e:
            logger.error(f"Failed to get recommended interval: {e}")
            return 'ONE_DAY'
    
    def estimate_data_points(self, interval: str, period: str) -> int:
        """
        Estimate number of data points for given interval and period
        
        Args:
            interval: Data interval
            period: Time period
            
        Returns:
            Estimated number of data points
        """
        try:
            # Interval to minutes mapping
            interval_minutes = {
                'ONE_MINUTE': 1,
                'FIVE_MINUTE': 5,
                'FIFTEEN_MINUTE': 15,
                'THIRTY_MINUTE': 30,
                'ONE_HOUR': 60,
                'ONE_DAY': 1440
            }
            
            # Period to days mapping
            period_days = {
                '1d': 1,
                '5d': 5,
                '1mo': 30,
                '3mo': 90,
                '6mo': 180,
                '1y': 365,
                '2y': 730,
                '5y': 1825
            }
            
            if interval not in interval_minutes or period not in period_days:
                return 1000  # Default estimate
            
            minutes_per_day = 1440
            total_minutes = period_days[period] * minutes_per_day
            interval_minutes_value = interval_minutes[interval]
            
            estimated_points = total_minutes // interval_minutes_value
            
            # Apply rate limits
            max_points = self.angel_one_limits.get(interval, 2000)
            return min(estimated_points, max_points)
            
        except Exception as e:
            logger.error(f"Failed to estimate data points: {e}")
            return 1000
    
    def get_angel_one_status(self) -> Dict[str, Any]:
        """
        Get Angel One interface status
        
        Returns:
            Status information
        """
        try:
            status = {
                'initialized': self.angel_manager is not None,
                'limits': self.angel_one_limits,
                'manager_available': hasattr(self, 'angel_manager') and self.angel_manager is not None
            }
            
            if self.angel_manager:
                status['manager_status'] = 'active'
            else:
                status['manager_status'] = 'inactive'
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get Angel One status: {e}")
            return {'error': str(e)}
