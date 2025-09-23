# 🗺️ Updated Implementation Roadmap: Including Core Services & Angel API

## Overview

This updated roadmap includes the core services integration, Angel One API, and database systems in the polylithic transformation.

## 📋 **Updated Project Structure**

### **New Directory Structure:**

```
main/
├── pipeline/
│   ├── base_pipeline.py          # Base classes & interfaces
│   ├── core_pipeline.py          # Main orchestration
│   ├── data_processor.py         # Data processing with service integration
│   ├── model_trainer.py          # Model training with service integration
│   ├── strategy_analyzer.py      # Strategy analysis with all services
│   ├── prediction_generator.py   # Prediction logic
│   └── report_generator.py       # Reporting logic
├── interfaces/
│   ├── interactive_selector.py   # Interactive data selection
│   ├── user_interface.py         # Main UI with Angel One support
│   ├── angel_one_interface.py    # Angel One configuration UI
│   └── input_validator.py        # Input validation
├── utils/
│   ├── pipeline_logger.py        # Enhanced logging
│   ├── error_handler.py          # Enhanced error handling
│   ├── formatters.py             # Price/currency formatting
│   ├── validators.py             # Data validation
│   └── service_manager.py        # Service coordination
├── config/
│   ├── pipeline_config.py        # Pipeline configuration
│   ├── analysis_config.py        # Analysis parameters
│   ├── angel_one_config.py       # Angel One configuration
│   └── database_config.py        # Database configuration
├── services/
│   ├── data_service_wrapper.py   # Data service integration
│   ├── model_service_wrapper.py  # Model service integration
│   ├── strategy_service_wrapper.py # Strategy service integration
│   ├── reporting_service_wrapper.py # Reporting service integration
│   ├── angel_one_manager.py      # Angel One API management
│   ├── database_manager.py       # Database management
│   └── api_coordinator.py        # API coordination
└── main.py                       # Entry point
```

## 🚀 **Updated Implementation Steps**

### **Step 1: Service Integration Layer (Day 1-2)**

#### **1.1 Create Service Wrappers**

```python
# main/services/data_service_wrapper.py
from src.core import DataService
from src.core.database_service import DatabaseService
from src.core.enhanced_angel_one_service import EnhancedAngelOneService

class DataServiceWrapper:
    """Wrapper for data service with Angel One and database integration"""

    def __init__(self, ticker: str, config: Dict[str, Any]):
        self.ticker = ticker
        self.config = config
        self.data_service = DataService(use_database=True)
        self.angel_service = None
        self.database_service = DatabaseService()

        # Initialize Angel One service if Indian stock
        if self._is_indian_stock(ticker):
            self.angel_service = EnhancedAngelOneService()

    def load_stock_data(self, period: str, interval: str = 'ONE_DAY') -> pd.DataFrame:
        """Load stock data from appropriate source"""
        if self._is_indian_stock(self.ticker) and self.angel_service:
            return self._load_angel_one_data(period, interval)
        else:
            return self._load_yahoo_finance_data(period)

    def _load_angel_one_data(self, period: str, interval: str) -> pd.DataFrame:
        """Load data from Angel One API"""
        # Implementation using enhanced_angel_one_service
        pass

    def _load_yahoo_finance_data(self, period: str) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        # Implementation using data_service
        pass
```

#### **1.2 Create Angel One Manager**

```python
# main/services/angel_one_manager.py
from src.core.enhanced_angel_one_service import EnhancedAngelOneService
from src.core.angel_one_database_schema import AngelOneDatabaseSchema
from src.utils.rate_limiter import get_api_rate_limiter

class AngelOneManager:
    """Manages Angel One API operations with rate limiting and caching"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.angel_service = EnhancedAngelOneService()
        self.db_schema = AngelOneDatabaseSchema()
        self.rate_limiter = get_api_rate_limiter()

    def get_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Get stock data with rate limiting and caching"""
        return self.rate_limiter.call_with_retry(
            'angel_one',
            self._fetch_stock_data,
            ticker, period, interval
        )

    def store_data_in_database(self, ticker: str, data: pd.DataFrame, interval: str):
        """Store data in database using Angel One schema"""
        # Implementation using angel_one_database_schema
        pass
```

#### **1.3 Create Database Manager**

```python
# main/services/database_manager.py
from src.core.database_service import DatabaseService
from src.utils.database_pool import get_connection_pool

class DatabaseManager:
    """Manages database operations with connection pooling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_service = DatabaseService()
        self.connection_pool = get_connection_pool()

    def store_stock_data(self, ticker: str, data: pd.DataFrame, source: str):
        """Store stock data in database"""
        with self.connection_pool.get_connection_context() as conn:
            if source == 'angel_one':
                self._store_angel_one_data(conn, ticker, data)
            else:
                self._store_yahoo_data(conn, ticker, data)

    def get_stock_data(self, ticker: str, period: str, source: str) -> Optional[pd.DataFrame]:
        """Get stock data from database"""
        with self.connection_pool.get_connection_context() as conn:
            if source == 'angel_one':
                return self._get_angel_one_data(conn, ticker, period)
            else:
                return self._get_yahoo_data(conn, ticker, period)
```

### **Step 2: Enhanced Data Processor (Day 3-4)**

#### **2.1 Updated Data Processor with Service Integration**

```python
# main/pipeline/data_processor.py
from .base_pipeline import BasePipelineComponent
from ..services.data_service_wrapper import DataServiceWrapper
from ..services.angel_one_manager import AngelOneManager
from ..services.database_manager import DatabaseManager

class DataProcessor(BasePipelineComponent):
    """Enhanced data processor with service integration"""

    def __init__(self, ticker: str, config: Dict[str, Any]):
        super().__init__(ticker, config)
        self.data_wrapper = DataServiceWrapper(ticker, config)
        self.angel_manager = AngelOneManager(config) if self._is_indian_stock(ticker) else None
        self.db_manager = DatabaseManager(config)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute enhanced data processing pipeline"""
        try:
            # Step 1: Load raw data from appropriate source
            raw_data = self._load_stock_data(kwargs['period'], kwargs.get('interval', 'ONE_DAY'))

            # Step 2: Store data in database
            self._store_data_in_database(raw_data, kwargs.get('interval', 'ONE_DAY'))

            # Step 3: Clean and preprocess
            cleaned_data = self._clean_and_preprocess(raw_data)

            # Step 4: Add technical indicators
            enhanced_data = self._add_technical_indicators(cleaned_data)

            # Step 5: Add economic and market data
            enriched_data = self._enrich_with_external_data(enhanced_data)

            # Step 6: Feature engineering
            features = self._engineer_features(enriched_data)

            return {
                'raw_data': raw_data,
                'cleaned_data': cleaned_data,
                'enhanced_data': enhanced_data,
                'enriched_data': enriched_data,
                'features': features,
                'data_source': self._get_data_source(),
                'success': True
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### **Step 3: Enhanced Strategy Analyzer (Day 5-6)**

#### **3.1 Updated Strategy Analyzer with All Services**

```python
# main/pipeline/strategy_analyzer.py
from .base_pipeline import BasePipelineComponent
from src.core import StrategyService
from src.core.economic_data_service import EconomicDataService
from src.core.fred_api_service import FredApiService
from src.core.geopolitical_risk_service import GeopoliticalRiskService
from src.core.global_market_service import GlobalMarketService
from src.core.corporate_action_service import CorporateActionService
from src.core.insider_trading_service import InsiderTradingService
from src.core.currency_service import CurrencyService

class StrategyAnalyzer(BasePipelineComponent):
    """Enhanced strategy analyzer with all service integrations"""

    def __init__(self, ticker: str, config: Dict[str, Any]):
        super().__init__(ticker, config)
        self.strategy_service = StrategyService()
        self.economic_service = EconomicDataService()
        self.fred_service = FredApiService()
        self.geopolitical_service = GeopoliticalRiskService()
        self.global_market_service = GlobalMarketService()
        self.corporate_action_service = CorporateActionService()
        self.insider_trading_service = InsiderTradingService()
        self.currency_service = CurrencyService()

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute comprehensive strategy analysis pipeline"""
        try:
            # Run all analysis components
            sentiment_results = self._run_sentiment_analysis()
            market_factors = self._run_market_factors()
            economic_indicators = self._run_economic_indicators()
            geopolitical_risk = self._run_geopolitical_analysis()
            global_market = self._run_global_market_analysis()
            corporate_actions = self._run_corporate_actions_analysis()
            insider_trading = self._run_insider_trading_analysis()
            currency_analysis = self._run_currency_analysis()
            trading_strategy = self._run_trading_strategy(kwargs['enhanced_data'])
            backtest_results = self._run_backtesting(kwargs['enhanced_data'])
            balance_sheet = self._run_balance_sheet_analysis()
            event_impact = self._run_event_impact_analysis()

            return {
                'sentiment': sentiment_results,
                'market_factors': market_factors,
                'economic_indicators': economic_indicators,
                'geopolitical_risk': geopolitical_risk,
                'global_market': global_market,
                'corporate_actions': corporate_actions,
                'insider_trading': insider_trading,
                'currency_analysis': currency_analysis,
                'trading_strategy': trading_strategy,
                'backtest_results': backtest_results,
                'balance_sheet': balance_sheet,
                'event_impact': event_impact,
                'success': True
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### **Step 4: Angel One Interface (Day 7-8)**

#### **4.1 Angel One Configuration Interface**

```python
# main/interfaces/angel_one_interface.py
from ..services.angel_one_manager import AngelOneManager

class AngelOneInterface:
    """Interface for Angel One API configuration and management"""

    def __init__(self):
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

    def configure_angel_one(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Configure Angel One API for Indian stocks"""
        try:
            print(f"\n🇮🇳 Indian stock detected: {ticker}")
            print("📡 Angel One API Configuration:")

            use_angel_one = input("Use Angel One API for data? (y/n, default: y): ").strip().lower()
            if use_angel_one == 'n':
                return None

            print("🔧 Angel One Configuration:")
            api_key = input("Enter API Key (or press Enter for default): ").strip()
            api_secret = input("Enter API Secret (or press Enter for default): ").strip()
            access_token = input("Enter Access Token (or press Enter for default): ").strip()
            exchange = input("Enter Exchange (NSE/BSE, default: NSE): ").strip().upper()
            if not exchange:
                exchange = "NSE"

            config = {
                'api_key': api_key if api_key else 'your_api_key',
                'api_secret': api_secret if api_secret else 'your_api_secret',
                'access_token': access_token if access_token else 'your_access_token',
                'exchange': exchange,
                'interval': 'ONE_DAY'
            }

            # Initialize Angel One manager with config
            self.angel_manager = AngelOneManager(config)

            print(f"✅ Angel One configured for {exchange} exchange")
            return config

        except Exception as e:
            print(f"❌ Angel One configuration failed: {e}")
            return None

    def test_angel_one_connection(self, config: Dict[str, Any]) -> bool:
        """Test Angel One API connection"""
        try:
            if not self.angel_manager:
                self.angel_manager = AngelOneManager(config)

            # Test connection with a simple API call
            test_result = self.angel_manager.test_connection()
            if test_result:
                print("✅ Angel One API connection successful")
                return True
            else:
                print("❌ Angel One API connection failed")
                return False

        except Exception as e:
            print(f"❌ Angel One connection test failed: {e}")
            return False
```

#### **4.2 Enhanced User Interface**

```python
# main/interfaces/user_interface.py
from .angel_one_interface import AngelOneInterface

class UserInterface:
    """Enhanced user interface with Angel One integration"""

    def __init__(self):
        self.angel_interface = AngelOneInterface()

    def get_user_inputs(self) -> Dict[str, Any]:
        """Get all user inputs for the pipeline with Angel One support"""
        try:
            # Get ticker
            ticker = self._get_ticker_input()

            # Check if Indian stock and configure Angel One
            is_indian = self._is_indian_stock(ticker)
            angel_config = None

            if is_indian:
                angel_config = self.angel_interface.configure_angel_one(ticker)
                if angel_config:
                    # Test Angel One connection
                    if not self.angel_interface.test_angel_one_connection(angel_config):
                        print("⚠️ Angel One connection failed, falling back to Yahoo Finance")
                        angel_config = None
            else:
                print(f"🇺🇸 US/International stock detected: {ticker}")
                print("📊 Using Yahoo Finance for data")

            # Get analysis type and parameters
            analysis_type = self._get_analysis_type()
            params = self._get_analysis_parameters(analysis_type)
            use_enhanced = self._get_enhanced_features_preference()
            use_database = self._get_database_preference()

            return {
                'ticker': ticker,
                'is_indian': is_indian,
                'angel_config': angel_config,
                'analysis_type': analysis_type,
                'parameters': params,
                'use_enhanced': use_enhanced,
                'use_database': use_database,
                'success': True
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### **Step 5: Service Manager and API Coordinator (Day 9-10)**

#### **5.1 Service Manager**

```python
# main/utils/service_manager.py
class ServiceManager:
    """Manages and coordinates all services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}

    def initialize_services(self, ticker: str) -> Dict[str, Any]:
        """Initialize all required services"""
        try:
            # Initialize core services
            self.services['data_service'] = self._init_data_service(ticker)
            self.services['model_service'] = self._init_model_service()
            self.services['strategy_service'] = self._init_strategy_service()
            self.services['reporting_service'] = self._init_reporting_service()

            # Initialize API services
            if self._is_indian_stock(ticker):
                self.services['angel_one_service'] = self._init_angel_one_service()

            self.services['database_service'] = self._init_database_service()

            # Initialize external services
            self.services['economic_service'] = self._init_economic_service()
            self.services['fred_service'] = self._init_fred_service()
            self.services['geopolitical_service'] = self._init_geopolitical_service()
            self.services['global_market_service'] = self._init_global_market_service()
            self.services['corporate_action_service'] = self._init_corporate_action_service()
            self.services['insider_trading_service'] = self._init_insider_trading_service()
            self.services['currency_service'] = self._init_currency_service()

            return {'success': True, 'services': list(self.services.keys())}

        except Exception as e:
            return {'success': False, 'error': str(e)}
```

#### **5.2 API Coordinator**

```python
# main/services/api_coordinator.py
from src.utils.rate_limiter import get_api_rate_limiter

class APICoordinator:
    """Coordinates multiple API calls with rate limiting and error handling"""

    def __init__(self):
        self.rate_limiter = get_api_rate_limiter()

    def coordinate_data_loading(self, ticker: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate data loading from multiple sources"""
        try:
            results = {}

            # Load primary data source
            if config.get('is_indian') and config.get('angel_config'):
                results['primary_data'] = self._load_angel_one_data(ticker, config)
            else:
                results['primary_data'] = self._load_yahoo_data(ticker, config)

            # Load supplementary data
            results['economic_data'] = self._load_economic_data()
            results['market_data'] = self._load_market_data()
            results['currency_data'] = self._load_currency_data()

            return {'success': True, 'data': results}

        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### **Step 6: Core Pipeline Integration (Day 11-12)**

#### **6.1 Updated Core Pipeline**

```python
# main/pipeline/core_pipeline.py
from .base_pipeline import PipelineOrchestrator
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .strategy_analyzer import StrategyAnalyzer
from .prediction_generator import PredictionGenerator
from ..interfaces.user_interface import UserInterface
from ..utils.service_manager import ServiceManager

class UnifiedAnalysisPipeline:
    """Main pipeline orchestrator with full service integration"""

    def __init__(self, ticker: str, config: Dict[str, Any] = None):
        self.ticker = ticker
        self.config = config or {}
        self.orchestrator = PipelineOrchestrator(ticker, self.config)
        self.service_manager = ServiceManager(self.config)
        self._setup_components()
        self._initialize_services()

    def _setup_components(self):
        """Setup all pipeline components"""
        self.orchestrator.add_component('data_processor', DataProcessor(self.ticker, self.config))
        self.orchestrator.add_component('model_trainer', ModelTrainer(self.ticker, self.config))
        self.orchestrator.add_component('strategy_analyzer', StrategyAnalyzer(self.ticker, self.config))
        self.orchestrator.add_component('prediction_generator', PredictionGenerator(self.ticker, self.config))

    def _initialize_services(self):
        """Initialize all services"""
        service_result = self.service_manager.initialize_services(self.ticker)
        if not service_result['success']:
            raise Exception(f"Service initialization failed: {service_result['error']}")

    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        try:
            start_time = time.time()
            results = self.orchestrator.execute_pipeline(**kwargs)
            execution_time = time.time() - start_time

            return {
                'success': results['success'],
                'results': results.get('results', {}),
                'execution_time': execution_time,
                'ticker': self.ticker
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### **Step 7: Testing and Validation (Day 13-14)**

#### **7.1 Integration Testing**

```python
# test_integration.py
def test_angel_one_integration():
    """Test Angel One API integration"""
    # Test Angel One configuration
    # Test data loading from Angel One
    # Test database storage with Angel One schema
    pass

def test_database_integration():
    """Test database integration"""
    # Test connection pooling
    # Test data storage and retrieval
    # Test multiple schemas
    pass

def test_service_coordination():
    """Test service coordination"""
    # Test all services initialization
    # Test API rate limiting
    # Test error handling and fallbacks
    pass
```

## 📊 **Updated Success Metrics**

### **Integration Metrics:**

- **Angel One API:** 100% functional for Indian stocks
- **Database Operations:** Connection pooling working
- **Service Coordination:** All 15+ services integrated
- **API Rate Limiting:** Working across all APIs

### **Performance Metrics:**

- **Data Loading:** 30-50% faster with service integration
- **Database Operations:** 40-60% faster with connection pooling
- **API Calls:** 90% fewer failures with rate limiting
- **Memory Usage:** Optimized with service management

### **Reliability Metrics:**

- **Fallback Mechanisms:** Working for all API failures
- **Error Handling:** Comprehensive error handling throughout
- **Service Health:** Monitoring and health checks
- **Graceful Degradation:** System continues with partial failures

## 🎯 **Key Integration Points**

### **Angel One API:**

- ✅ Configuration management for Indian stocks
- ✅ Rate limiting and error handling
- ✅ Database storage using Angel One schema
- ✅ Fallback mechanisms to Yahoo Finance

### **Database Integration:**

- ✅ Connection pooling for performance
- ✅ Multiple schemas (Angel One, Yahoo Finance)
- ✅ Data caching and retrieval
- ✅ Incremental updates

### **Service Coordination:**

- ✅ Unified service management
- ✅ API rate limiting across all services
- ✅ Error handling and fallback mechanisms
- ✅ Performance optimization

This updated roadmap ensures complete integration with all existing core services, Angel One API, and database systems while maintaining the modular polylithic architecture benefits.
