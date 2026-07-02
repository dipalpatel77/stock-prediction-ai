"""
Enhanced User Interface
User interface with Angel One integration
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import interfaces
from .angel_one_interface import AngelOneInterface
from ..utils.stock_utils import is_indian_stock
from .interactive_selector import InteractiveDataSelector
from .input_validator import InputValidator


class UserInterface:
    """
    Enhanced user interface with Angel One integration
    
    This interface provides:
    - User input collection
    - Angel One configuration
    - Analysis type selection
    - Parameter configuration
    - Enhanced features preference
    - Database preference
    """
    
    def __init__(self):
        """Initialize Enhanced User Interface"""
        self.angel_interface = AngelOneInterface()
        self.interactive_selector = InteractiveDataSelector()
        self.input_validator = InputValidator()
        
        logger.info("Enhanced User Interface initialized")
    
    def get_user_inputs(self, test_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get all user inputs for the pipeline with Angel One support
        
        Args:
            test_config: Optional test configuration for non-interactive testing
            
        Returns:
            Dictionary with all user inputs
        """
        try:
            print("\n🚀 AI Stock Predictor - Enhanced Interface")
            print("=" * 50)
            
            # Use test config if provided, otherwise get user input
            if test_config:
                print("✅ Using test configuration for non-interactive testing")
                ticker = test_config.get('ticker', 'AAPL')
                is_indian = test_config.get('is_indian', False)
                angel_config = test_config.get('angel_config', None)
                timeframe = test_config.get('timeframe', '1y')
                interval = test_config.get('interval', 'ONE_DAY')
            else:
                # Get ticker input
                ticker = self._get_ticker_input()
                
                # Set default values for comprehensive data download
                timeframe = '5y'  # Default to 5 years for comprehensive data
                interval = 'ONE_DAY'  # Default interval (will download all intervals for Indian stocks)
                
                # Check if Indian stock and configure Angel One
                is_indian = self._is_indian_stock(ticker)
                angel_config = None
                
                if is_indian:
                    print(f"\n🇮🇳 Indian stock detected: {ticker}")
                    print("📊 Will download comprehensive data for all intervals automatically")
                    angel_config = self.angel_interface.configure_angel_one(ticker)
                    if angel_config:
                        # Add timeframe and interval to angel config
                        angel_config['timeframe'] = timeframe
                        angel_config['interval'] = interval
                        # Test Angel One connection
                        if not self.angel_interface.test_angel_one_connection(angel_config):
                            print("⚠️ Angel One connection failed, falling back to Yahoo Finance")
                            angel_config = None
                else:
                    print(f"🇺🇸 US/International stock detected: {ticker}")
                    print("📊 Using Yahoo Finance for data")
            
            # Get analysis type and parameters
            if test_config:
                analysis_type = test_config.get('analysis_type', 'comprehensive')
                params = test_config.get('parameters', {
                    'technical_indicators': 'all',
                    'ml_models': 'all',
                    'trading_strategies': 'all'
                })
                use_enhanced = test_config.get('use_enhanced', True)
                use_database = test_config.get('use_database', True)
            else:
                analysis_type = self._get_analysis_type()
                params = self._get_analysis_parameters(analysis_type)
                use_enhanced = self._get_enhanced_features_preference()
                use_database = self._get_database_preference()
            
            # Compile user inputs
            user_inputs = {
                'ticker': ticker,
                'is_indian': is_indian,
                'angel_config': angel_config,
                'analysis_type': analysis_type,
                'parameters': params,
                'use_enhanced': use_enhanced,
                'use_database': use_database,
                'timeframe': timeframe,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            # Validate inputs
            if not self.input_validator.validate_user_inputs(user_inputs):
                print("❌ Input validation failed")
                return {'success': False, 'error': 'Invalid inputs'}
            
            print("\n✅ Configuration completed successfully!")
            return user_inputs
            
        except Exception as e:
            logger.error(f"Failed to get user inputs: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_ticker_input(self) -> str:
        """Get ticker symbol from user"""
        try:
            print("\n📈 Stock Ticker Input:")
            print("-" * 25)
            
            while True:
                ticker = input("Enter stock ticker (e.g., AAPL, RELIANCE.NS, INFY.NS): ").strip().upper()
                
                if not ticker:
                    print("❌ Please enter a valid ticker symbol")
                    continue
                
                # Validate ticker format
                if self.input_validator.validate_ticker(ticker):
                    print(f"✅ Ticker validated: {ticker}")
                    return ticker
                else:
                    print("❌ Invalid ticker format. Please try again.")
                    
        except Exception as e:
            logger.error(f"Failed to get ticker input: {e}")
            return "AAPL"  # Default fallback
    
    def _is_indian_stock(self, ticker: str) -> bool:
        return is_indian_stock(ticker)
    
    def _get_analysis_type(self) -> str:
        """Get analysis type from user"""
        try:
            print("\n🔍 Analysis Type Selection:")
            print("-" * 30)
            print("1. Short-term Analysis (1-30 days)")
            print("2. Mid-term Analysis (1-6 months)")
            print("3. Long-term Analysis (6+ months)")
            print("4. Intraday Analysis (Same day)")
            print("5. Swing Trading (1-4 weeks)")
            print("6. Position Trading (1+ months)")
            print("7. Comprehensive Analysis (All timeframes)")
            
            while True:
                choice = input("Select analysis type (1-7, default: 7): ").strip()
                
                analysis_map = {
                    '1': 'short_term',
                    '2': 'mid_term',
                    '3': 'long_term',
                    '4': 'intraday',
                    '5': 'swing',
                    '6': 'position',
                    '7': 'comprehensive'
                }
                
                if not choice:
                    choice = '7'  # Default to comprehensive
                
                if choice in analysis_map:
                    analysis_type = analysis_map[choice]
                    print(f"✅ Selected: {analysis_type}")
                    return analysis_type
                else:
                    print("❌ Invalid choice. Please select 1-7.")
                    
        except Exception as e:
            logger.error(f"Failed to get analysis type: {e}")
            return 'comprehensive'
    
    def _get_analysis_parameters(self, analysis_type: str) -> Dict[str, Any]:
        """Get analysis parameters based on analysis type"""
        try:
            print(f"\n⚙️ Analysis Parameters for {analysis_type}:")
            print("-" * 40)
            
            params = {
                'analysis_type': analysis_type,
                'timeframe': self._get_timeframe(analysis_type),
                'indicators': self._get_technical_indicators(),
                'models': self._get_ml_models(),
                'strategies': self._get_trading_strategies()
            }
            
            return params
            
        except Exception as e:
            logger.error(f"Failed to get analysis parameters: {e}")
            return {'analysis_type': analysis_type}
    
    def _get_timeframe(self, analysis_type: str) -> str:
        """Get timeframe based on analysis type"""
        try:
            timeframe_map = {
                'short_term': '1mo',
                'mid_term': '6mo',
                'long_term': '2y',
                'intraday': '1d',
                'swing': '3mo',
                'position': '1y',
                'comprehensive': '2y'
            }
            
            return timeframe_map.get(analysis_type, '2y')
            
        except Exception as e:
            logger.error(f"Failed to get timeframe: {e}")
            return '2y'
    
    def _get_technical_indicators(self) -> list:
        """Get technical indicators selection"""
        try:
            print("\n📊 Technical Indicators:")
            print("1. All indicators (recommended)")
            print("2. Basic indicators (SMA, EMA, RSI, MACD)")
            print("3. Advanced indicators (Bollinger Bands, Stochastic, etc.)")
            print("4. Custom selection")
            
            choice = input("Select indicators (1-4, default: 1): ").strip()
            
            if not choice:
                choice = '1'
            
            if choice == '1':
                return ['all']
            elif choice == '2':
                return ['sma', 'ema', 'rsi', 'macd']
            elif choice == '3':
                return ['bollinger', 'stochastic', 'williams_r', 'cci']
            else:
                return ['all']  # Default to all
                
        except Exception as e:
            logger.error(f"Failed to get technical indicators: {e}")
            return ['all']
    
    def _get_ml_models(self) -> list:
        """Get ML models selection"""
        try:
            print("\n🤖 Machine Learning Models:")
            print("1. All models (recommended)")
            print("2. Basic models (Linear Regression, Random Forest)")
            print("3. Advanced models (LSTM, XGBoost, etc.)")
            print("4. Custom selection")
            
            choice = input("Select models (1-4, default: 1): ").strip()
            
            if not choice:
                choice = '1'
            
            if choice == '1':
                return ['all']
            elif choice == '2':
                return ['linear_regression', 'random_forest']
            elif choice == '3':
                return ['lstm', 'xgboost', 'svm']
            else:
                return ['all']  # Default to all
                
        except Exception as e:
            logger.error(f"Failed to get ML models: {e}")
            return ['all']
    
    def _get_trading_strategies(self) -> list:
        """Get trading strategies selection"""
        try:
            print("\n📈 Trading Strategies:")
            print("1. All strategies (recommended)")
            print("2. Technical strategies")
            print("3. Fundamental strategies")
            print("4. Quantitative strategies")
            
            choice = input("Select strategies (1-4, default: 1): ").strip()
            
            if not choice:
                choice = '1'
            
            if choice == '1':
                return ['all']
            elif choice == '2':
                return ['technical']
            elif choice == '3':
                return ['fundamental']
            elif choice == '4':
                return ['quantitative']
            else:
                return ['all']  # Default to all
                
        except Exception as e:
            logger.error(f"Failed to get trading strategies: {e}")
            return ['all']
    
    def _get_enhanced_features_preference(self) -> bool:
        """Get enhanced features preference"""
        try:
            print("\n✨ Enhanced Features:")
            print("1. Yes - Use all enhanced features (recommended)")
            print("2. No - Use basic features only")
            
            choice = input("Use enhanced features? (1/2, default: 1): ").strip()
            
            if not choice:
                choice = '1'
            
            return choice == '1'
            
        except Exception as e:
            logger.error(f"Failed to get enhanced features preference: {e}")
            return True
    
    def _get_database_preference(self) -> bool:
        """Get database preference"""
        try:
            print("\n💾 Database Storage:")
            print("1. Yes - Store data in database (recommended)")
            print("2. No - Use memory only")
            
            choice = input("Use database storage? (1/2, default: 1): ").strip()
            
            if not choice:
                choice = '1'
            
            return choice == '1'
            
        except Exception as e:
            logger.error(f"Failed to get database preference: {e}")
            return True
    
    def display_configuration_summary(self, user_inputs: Dict[str, Any]):
        """
        Display configuration summary
        
        Args:
            user_inputs: User input dictionary
        """
        try:
            print("\n📋 Configuration Summary:")
            print("=" * 30)
            print(f"Ticker: {user_inputs.get('ticker', 'N/A')}")
            print(f"Indian Stock: {user_inputs.get('is_indian', False)}")
            print(f"Analysis Type: {user_inputs.get('analysis_type', 'N/A')}")
            print(f"Enhanced Features: {user_inputs.get('use_enhanced', False)}")
            print(f"Database Storage: {user_inputs.get('use_database', False)}")
            
            if user_inputs.get('angel_config'):
                print(f"Angel One: Configured for {user_inputs['angel_config'].get('exchange', 'NSE')}")
            else:
                print("Data Source: Yahoo Finance")
                
        except Exception as e:
            logger.error(f"Failed to display configuration summary: {e}")
    
    def get_interface_status(self) -> Dict[str, Any]:
        """
        Get interface status
        
        Returns:
            Status information
        """
        try:
            status = {
                'angel_interface': self.angel_interface.get_angel_one_status(),
                'interactive_selector': hasattr(self, 'interactive_selector'),
                'input_validator': hasattr(self, 'input_validator'),
                'initialized': True
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get interface status: {e}")
            return {'error': str(e)}
    
    def _get_timeframe_selection(self) -> str:
        """
        Get timeframe selection from user
        
        Returns:
            Selected timeframe
        """
        try:
            print("\n📅 Timeframe Selection:")
            print("=" * 30)
            print("1. 1 Day (1d)")
            print("2. 5 Days (5d)")
            print("3. 1 Month (1mo)")
            print("4. 3 Months (3mo)")
            print("5. 6 Months (6mo)")
            print("6. 1 Year (1y) - Recommended")
            print("7. 2 Years (2y)")
            print("8. 5 Years (5y)")
            print("9. Maximum (max)")
            
            while True:
                choice = input("Select timeframe (1-9, default: 6): ").strip()
                
                if not choice:
                    return '1y'
                
                timeframe_map = {
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
                
                if choice in timeframe_map:
                    selected = timeframe_map[choice]
                    print(f"✅ Selected timeframe: {selected}")
                    return selected
                else:
                    print("❌ Invalid choice. Please select 1-9.")
                    
        except Exception as e:
            logger.error(f"Timeframe selection failed: {e}")
            return '1y'
    
    def _get_interval_selection(self) -> str:
        """
        Get interval selection for Angel One API
        
        Returns:
            Selected interval
        """
        try:
            print("\n⏰ Data Interval Selection (Angel One API):")
            print("=" * 45)
            print("1. 1 Minute (ONE_MINUTE) - Intraday")
            print("2. 3 Minutes (THREE_MINUTE) - Intraday")
            print("3. 5 Minutes (FIVE_MINUTE) - Intraday")
            print("4. 10 Minutes (TEN_MINUTE) - Intraday")
            print("5. 15 Minutes (FIFTEEN_MINUTE) - Intraday")
            print("6. 30 Minutes (THIRTY_MINUTE) - Intraday")
            print("7. 1 Hour (ONE_HOUR) - Intraday")
            print("8. 1 Day (ONE_DAY) - Daily - Recommended")
            
            while True:
                choice = input("Select interval (1-8, default: 8): ").strip()
                
                if not choice:
                    return 'ONE_DAY'
                
                interval_map = {
                    '1': 'ONE_MINUTE',
                    '2': 'THREE_MINUTE',
                    '3': 'FIVE_MINUTE',
                    '4': 'TEN_MINUTE',
                    '5': 'FIFTEEN_MINUTE',
                    '6': 'THIRTY_MINUTE',
                    '7': 'ONE_HOUR',
                    '8': 'ONE_DAY'
                }
                
                if choice in interval_map:
                    selected = interval_map[choice]
                    print(f"✅ Selected interval: {selected}")
                    return selected
                else:
                    print("❌ Invalid choice. Please select 1-8.")
                    
        except Exception as e:
            logger.error(f"Interval selection failed: {e}")
            return 'ONE_DAY'
