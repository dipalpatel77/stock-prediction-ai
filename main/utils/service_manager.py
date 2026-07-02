"""
Service Manager
Manages and coordinates all services in the polylithic pipeline
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import services
from .stock_utils import is_indian_stock
from ..services.data_service_wrapper import DataServiceWrapper
from ..services.angel_one_manager import AngelOneManager
from ..services.database_manager import DatabaseManager
from ..services.api_coordinator import APICoordinator
from ..services.technical_indicators_service import TechnicalIndicatorsService
from ..services.feature_engineering_service import FeatureEngineeringService

# Import core services
# Use existing services from main.services
try:
    from ..services.model_service import ModelService
except ImportError:
    logger.warning("ModelService not available, using placeholder")
    ModelService = None

try:
    from ..services.report_generator import ReportGenerator as ReportingService
except ImportError:
    logger.warning("ReportingService not available, using placeholder")
    ReportingService = None

try:
    from ..services.fred_api_service import FREDAPIService as FredApiService
except ImportError:
    logger.warning("FredApiService not available, using placeholder")
    FredApiService = None

# Strategy service placeholder
StrategyService = None

# Use existing services from main.services
try:
    from ..services.global_market_service import GlobalMarketService
except ImportError:
    logger.warning("GlobalMarketService not available, using placeholder")
    GlobalMarketService = None

try:
    from ..services.currency_service import CurrencyService
except ImportError:
    logger.warning("CurrencyService not available, using placeholder")
    CurrencyService = None

# Placeholder services (removed during cleanup)
GeopoliticalRiskService = None
CorporateActionService = None
InsiderTradingService = None


class ServiceManager:
    """
    Service Manager for coordinating all services in the polylithic pipeline
    
    This manager provides:
    - Service initialization and coordination
    - Service health monitoring
    - Service fallback mechanisms
    - Service status reporting
    - Service lifecycle management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Service Manager
        
        Args:
            config: Configuration dictionary (optional)
        """
        if config is None:
            config = {
                'database_url': 'sqlite:///default.db',
                'api_rate_limit': 100,
                'cache_enabled': True,
                'log_level': 'INFO'
            }
        self.config = config
        self.services = {}
        self.service_status = {}
        self.initialization_time = None
        
        logger.info("Service Manager initialized")
    
    def initialize_services(self, ticker: str) -> Dict[str, Any]:
        """
        Initialize all required services
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with initialization results
        """
        try:
            start_time = time.time()
            logger.info(f"Initializing services for {ticker}")
            
            # Initialize core services
            self._init_core_services(ticker)
            
            # Initialize API services
            if self._is_indian_stock(ticker):
                self._init_angel_one_services(ticker)
            
            # Initialize database services
            self._init_database_services()
            
            # Initialize external services
            self._init_external_services()
            
            # Initialize data processing services
            self._init_data_processing_services()
            
            self.initialization_time = time.time() - start_time
            
            # Get service status
            status = self.get_service_status()
            
            logger.info(f"Services initialized in {self.initialization_time:.2f} seconds")
            return {
                'success': True,
                'services': list(self.services.keys()),
                'status': status,
                'initialization_time': self.initialization_time
            }
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _init_core_services(self, ticker: str):
        """Initialize core services"""
        try:
            # Data Service
            self.services['data_service'] = DataServiceWrapper(ticker, self.config)
            self.service_status['data_service'] = 'initialized'
            
            # Strategy Service
            if StrategyService:
                self.services['strategy_service'] = StrategyService()
                self.service_status['strategy_service'] = 'initialized'
            else:
                self.services['strategy_service'] = self._create_strategy_service_placeholder()
                self.service_status['strategy_service'] = 'placeholder'
            
            # Model Service
            if ModelService:
                self.services['model_service'] = ModelService()
                self.service_status['model_service'] = 'initialized'
            else:
                self.services['model_service'] = self._create_model_service_placeholder()
                self.service_status['model_service'] = 'placeholder'
            
            # Reporting Service
            if ReportingService:
                self.services['reporting_service'] = ReportingService()
                self.service_status['reporting_service'] = 'initialized'
            else:
                self.services['reporting_service'] = self._create_reporting_service_placeholder()
                self.service_status['reporting_service'] = 'placeholder'
            
            logger.info("Core services initialized")
            
        except Exception as e:
            logger.error(f"Core services initialization failed: {e}")
    
    def _init_angel_one_services(self, ticker: str):
        """Initialize Angel One services"""
        try:
            # Angel One Manager
            self.services['angel_one_manager'] = AngelOneManager(self.config)
            self.service_status['angel_one_manager'] = 'initialized'
            
            logger.info("Angel One services initialized")
            
        except Exception as e:
            logger.error(f"Angel One services initialization failed: {e}")
    
    def _init_database_services(self):
        """Initialize database services"""
        try:
            # Database Manager
            self.services['database_manager'] = DatabaseManager(self.config)
            self.service_status['database_manager'] = 'initialized'
            
            # API Coordinator
            self.services['api_coordinator'] = APICoordinator()
            self.service_status['api_coordinator'] = 'initialized'
            
            logger.info("Database services initialized")
            
        except Exception as e:
            logger.error(f"Database services initialization failed: {e}")
    
    def _init_external_services(self):
        """Initialize external services"""
        try:
            # Economic Data Service - removed, using placeholder
            self.services['economic_data_service'] = None
            self.service_status['economic_data_service'] = 'removed'
            
            # FRED API Service
            if FredApiService:
                self.services['fred_api_service'] = FredApiService()
                self.service_status['fred_api_service'] = 'initialized'
            else:
                self.services['fred_api_service'] = self._create_fred_service_placeholder()
                self.service_status['fred_api_service'] = 'placeholder'
            
            # Geopolitical Risk Service - removed
            self.services['geopolitical_risk_service'] = None
            self.service_status['geopolitical_risk_service'] = 'removed'
            
            # Global Market Service
            if GlobalMarketService:
                self.services['global_market_service'] = GlobalMarketService()
                self.service_status['global_market_service'] = 'initialized'
            else:
                self.services['global_market_service'] = self._create_global_market_service_placeholder()
                self.service_status['global_market_service'] = 'placeholder'
            
            # Corporate Action Service - removed
            self.services['corporate_action_service'] = None
            self.service_status['corporate_action_service'] = 'removed'
            
            # Insider Trading Service - removed
            self.services['insider_trading_service'] = None
            self.service_status['insider_trading_service'] = 'removed'
            
            # Currency Service
            if CurrencyService:
                self.services['currency_service'] = CurrencyService()
                self.service_status['currency_service'] = 'initialized'
            else:
                self.services['currency_service'] = self._create_currency_service_placeholder()
                self.service_status['currency_service'] = 'placeholder'
            
            # Additional services for comprehensive coverage
            self._init_additional_services()
            
            logger.info("External services initialized")
            
        except Exception as e:
            logger.error(f"External services initialization failed: {e}")
    
    def _init_additional_services(self):
        """Initialize additional services for comprehensive coverage"""
        try:
            # News Service
            try:
                from src.core.news_service import NewsService
                self.services['news_service'] = NewsService()
                self.service_status['news_service'] = 'initialized'
            except ImportError:
                self.services['news_service'] = self._create_news_service_placeholder()
                self.service_status['news_service'] = 'placeholder'
            
            # Sentiment Service
            try:
                from src.core.sentiment_service import SentimentService
                self.services['sentiment_service'] = SentimentService()
                self.service_status['sentiment_service'] = 'initialized'
            except ImportError:
                self.services['sentiment_service'] = self._create_sentiment_service_placeholder()
                self.service_status['sentiment_service'] = 'placeholder'
            
            # Technical Analysis Service
            try:
                from src.core.technical_analysis_service import TechnicalAnalysisService
                self.services['technical_analysis_service'] = TechnicalAnalysisService()
                self.service_status['technical_analysis_service'] = 'initialized'
            except ImportError:
                self.services['technical_analysis_service'] = self._create_technical_analysis_service_placeholder()
                self.service_status['technical_analysis_service'] = 'placeholder'
            
            # Risk Management Service
            try:
                from src.core.risk_management_service import RiskManagementService
                self.services['risk_management_service'] = RiskManagementService()
                self.service_status['risk_management_service'] = 'initialized'
            except ImportError:
                self.services['risk_management_service'] = self._create_risk_management_service_placeholder()
                self.service_status['risk_management_service'] = 'placeholder'
            
            # Portfolio Service
            try:
                from src.core.portfolio_service import PortfolioService
                self.services['portfolio_service'] = PortfolioService()
                self.service_status['portfolio_service'] = 'initialized'
            except ImportError:
                self.services['portfolio_service'] = self._create_portfolio_service_placeholder()
                self.service_status['portfolio_service'] = 'placeholder'
            
            # Market Data Service
            try:
                from src.core.market_data_service import MarketDataService
                self.services['market_data_service'] = MarketDataService()
                self.service_status['market_data_service'] = 'initialized'
            except ImportError:
                self.services['market_data_service'] = self._create_market_data_service_placeholder()
                self.service_status['market_data_service'] = 'placeholder'
            
            # Analytics Service
            try:
                from src.core.analytics_service import AnalyticsService
                self.services['analytics_service'] = AnalyticsService()
                self.service_status['analytics_service'] = 'initialized'
            except ImportError:
                self.services['analytics_service'] = self._create_analytics_service_placeholder()
                self.service_status['analytics_service'] = 'placeholder'
            
            # Notification Service
            try:
                from src.core.notification_service import NotificationService
                self.services['notification_service'] = NotificationService()
                self.service_status['notification_service'] = 'initialized'
            except ImportError:
                self.services['notification_service'] = self._create_notification_service_placeholder()
                self.service_status['notification_service'] = 'placeholder'
            
            logger.info("Additional services initialized")
            
        except Exception as e:
            logger.error(f"Additional services initialization failed: {e}")
    
    def _init_data_processing_services(self):
        """Initialize data processing services"""
        try:
            # Technical Indicators Service
            self.services['technical_indicators_service'] = TechnicalIndicatorsService()
            self.service_status['technical_indicators_service'] = 'initialized'
            
            # Feature Engineering Service
            self.services['feature_engineering_service'] = FeatureEngineeringService()
            self.service_status['feature_engineering_service'] = 'initialized'
            
            # Incremental Services
            from ..services.incremental_data_service import IncrementalDataService
            from ..services.incremental_service import IncrementalService
            
            self.services['incremental_data_service'] = IncrementalDataService()
            self.service_status['incremental_data_service'] = 'initialized'
            
            self.services['incremental_service'] = IncrementalService()
            self.service_status['incremental_service'] = 'initialized'
            
            # Interval Specific Storage Service - removed (duplicate functionality)
            self.services['interval_specific_storage'] = None
            self.service_status['interval_specific_storage'] = 'removed'
            
            logger.info("Data processing services initialized")
            
        except Exception as e:
            logger.error(f"Data processing services initialization failed: {e}")
            logger.error(f"Full error details: {repr(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get a specific service
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance or None if not found
        """
        try:
            return self.services.get(service_name)
        except Exception as e:
            logger.error(f"Failed to get service {service_name}: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all services
        
        Returns:
            Dictionary with service status information
        """
        try:
            status = {
                'total_services': len(self.services),
                'initialized_services': len([s for s in self.service_status.values() if s == 'initialized']),
                'placeholder_services': len([s for s in self.service_status.values() if s == 'placeholder']),
                'service_status': self.service_status.copy(),
                'initialization_time': self.initialization_time,
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {'error': str(e)}
    
    def check_service_health(self, service_name: str) -> bool:
        """
        Check health of a specific service
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            if service_name not in self.services:
                return False
            
            service = self.services[service_name]
            
            # Check if service has test_connection method
            if hasattr(service, 'test_connection'):
                return service.test_connection()
            
            # Check if service has get_status method
            if hasattr(service, 'get_status'):
                status = service.get_status()
                return status.get('healthy', True)
            
            # Default to True if no health check method
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return False
    
    def get_all_service_health(self) -> Dict[str, bool]:
        """
        Get health status of all services
        
        Returns:
            Dictionary with service health status
        """
        try:
            health_status = {}
            
            for service_name in self.services.keys():
                health_status[service_name] = self.check_service_health(service_name)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to get all service health: {e}")
            return {}
    
    def restart_service(self, service_name: str) -> bool:
        """
        Restart a specific service
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if restart successful, False otherwise
        """
        try:
            if service_name not in self.services:
                logger.error(f"Service {service_name} not found")
                return False
            
            # Remove service from services dict
            del self.services[service_name]
            del self.service_status[service_name]
            
            # Reinitialize service
            if service_name == 'data_service':
                self.services[service_name] = DataServiceWrapper(self.config.get('ticker', ''), self.config)
            elif service_name == 'economic_data_service':
                # Economic data service removed
                self.services[service_name] = None
            elif service_name == 'technical_indicators_service':
                self.services[service_name] = TechnicalIndicatorsService()
            elif service_name == 'feature_engineering_service':
                self.services[service_name] = FeatureEngineeringService()
            # Add more service reinitialization logic as needed
            
            self.service_status[service_name] = 'restarted'
            
            logger.info(f"Service {service_name} restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            return False
    
    def _is_indian_stock(self, ticker: str) -> bool:
        return is_indian_stock(ticker)
    
    # Placeholder service creation methods
    def _create_strategy_service_placeholder(self):
        """Create placeholder strategy service"""
        class PlaceholderStrategyService:
            def get_strategy_recommendations(self, data): return {'recommendations': []}
            def run_backtesting(self, data): return {'results': {}}
            def test_connection(self): return True
        return PlaceholderStrategyService()
    
    def _create_model_service_placeholder(self):
        """Create placeholder model service"""
        class PlaceholderModelService:
            def train_model(self, data): return {'model': 'placeholder'}
            def predict(self, data): return {'predictions': []}
            def test_connection(self): return True
        return PlaceholderModelService()
    
    def _create_reporting_service_placeholder(self):
        """Create placeholder reporting service"""
        class PlaceholderReportingService:
            def generate_report(self, data): return {'report': 'placeholder'}
            def test_connection(self): return True
        return PlaceholderReportingService()
    
    def _create_fred_service_placeholder(self):
        """Create placeholder FRED service"""
        class PlaceholderFredService:
            def get_economic_data(self): return {'data': {}}
            def test_connection(self): return True
        return PlaceholderFredService()
    
    def _create_geopolitical_service_placeholder(self):
        """Create placeholder geopolitical service"""
        class PlaceholderGeopoliticalService:
            def get_geopolitical_risk(self): return {'risk_score': 0.5}
            def test_connection(self): return True
        return PlaceholderGeopoliticalService()
    
    def _create_global_market_service_placeholder(self):
        """Create placeholder global market service"""
        class PlaceholderGlobalMarketService:
            def get_global_market_data(self): return {'markets': {}}
            def test_connection(self): return True
        return PlaceholderGlobalMarketService()
    
    def _create_corporate_action_service_placeholder(self):
        """Create placeholder corporate action service"""
        class PlaceholderCorporateActionService:
            def get_corporate_actions(self, ticker): return {'actions': []}
            def test_connection(self): return True
        return PlaceholderCorporateActionService()
    
    def _create_insider_trading_service_placeholder(self):
        """Create placeholder insider trading service"""
        class PlaceholderInsiderTradingService:
            def get_insider_trading(self, ticker): return {'trades': []}
            def test_connection(self): return True
        return PlaceholderInsiderTradingService()
    
    def _create_currency_service_placeholder(self):
        """Create placeholder currency service"""
        class PlaceholderCurrencyService:
            def get_currency_data(self): return {'currencies': {}}
            def test_connection(self): return True
        return PlaceholderCurrencyService()
    
    def _create_news_service_placeholder(self):
        """Create placeholder news service"""
        class PlaceholderNewsService:
            def get_news(self, ticker): return {'news': []}
            def test_connection(self): return True
        return PlaceholderNewsService()
    
    def _create_sentiment_service_placeholder(self):
        """Create placeholder sentiment service"""
        class PlaceholderSentimentService:
            def get_sentiment(self, text): return {'sentiment': 0.0}
            def test_connection(self): return True
        return PlaceholderSentimentService()
    
    def _create_technical_analysis_service_placeholder(self):
        """Create placeholder technical analysis service"""
        class PlaceholderTechnicalAnalysisService:
            def analyze(self, data): return {'analysis': {}}
            def test_connection(self): return True
        return PlaceholderTechnicalAnalysisService()
    
    def _create_risk_management_service_placeholder(self):
        """Create placeholder risk management service"""
        class PlaceholderRiskManagementService:
            def assess_risk(self, data): return {'risk_score': 0.5}
            def test_connection(self): return True
        return PlaceholderRiskManagementService()
    
    def _create_portfolio_service_placeholder(self):
        """Create placeholder portfolio service"""
        class PlaceholderPortfolioService:
            def get_portfolio(self): return {'portfolio': {}}
            def test_connection(self): return True
        return PlaceholderPortfolioService()
    
    def _create_market_data_service_placeholder(self):
        """Create placeholder market data service"""
        class PlaceholderMarketDataService:
            def get_market_data(self): return {'market_data': {}}
            def test_connection(self): return True
        return PlaceholderMarketDataService()
    
    def _create_analytics_service_placeholder(self):
        """Create placeholder analytics service"""
        class PlaceholderAnalyticsService:
            def analyze(self, data): return {'analytics': {}}
            def test_connection(self): return True
        return PlaceholderAnalyticsService()
    
    def _create_notification_service_placeholder(self):
        """Create placeholder notification service"""
        class PlaceholderNotificationService:
            def send_notification(self, message): return {'sent': True}
            def test_connection(self): return True
        return PlaceholderNotificationService()
