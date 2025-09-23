# 🏗️ Updated Polylithic Transformation Plan (Including Core Services & Angel API)

## Executive Summary

This updated plan transforms the monolithic `main/unified_analysis_pipeline.py` (3999 lines) into a modular polylithic architecture while properly integrating with existing core services, Angel One API, and database systems.

## 📊 **Current State Analysis**

### **Existing Core Services (src/core/):**

- `data_service.py` - Data loading and preprocessing
- `model_service.py` - Model management and caching
- `strategy_service.py` - Strategy analysis
- `reporting_service.py` - Report generation
- `enhanced_angel_one_service.py` - Angel One API integration
- `database_service.py` - Database operations
- `angel_one_database_schema.py` - Angel One database schema
- `economic_data_service.py` - Economic data integration
- `fred_api_service.py` - FRED API integration
- `geopolitical_risk_service.py` - Geopolitical risk analysis
- `global_market_service.py` - Global market data
- `corporate_action_service.py` - Corporate actions
- `insider_trading_service.py` - Insider trading data
- `currency_service.py` - Currency conversion
- `incremental_data_service.py` - Incremental data updates
- `multi_exchange_data_service.py` - Multi-exchange data fusion

### **Current Integration Points:**

- Angel One API for Indian stocks
- MySQL database for data storage
- Yahoo Finance for US stocks
- FRED API for economic data
- Multiple data sources and services

## 🎯 **Updated Target Architecture**

### **New Structure:**

```
main/
├── __init__.py
├── pipeline/
│   ├── __init__.py
│   ├── base_pipeline.py          # Base classes & interfaces
│   ├── core_pipeline.py          # Main orchestration (200-300 lines)
│   ├── data_processor.py         # Data processing logic (400-500 lines)
│   ├── model_trainer.py          # Model training logic (500-600 lines)
│   ├── strategy_analyzer.py      # Strategy analysis logic (400-500 lines)
│   ├── prediction_generator.py   # Prediction logic (400-500 lines)
│   └── report_generator.py       # Reporting logic (300-400 lines)
├── interfaces/
│   ├── __init__.py
│   ├── interactive_selector.py   # Interactive data selection (300-400 lines)
│   ├── user_interface.py         # Main UI logic (200-300 lines)
│   ├── angel_one_interface.py    # Angel One configuration UI (200-300 lines)
│   └── input_validator.py        # Input validation (100-200 lines)
├── utils/
│   ├── __init__.py
│   ├── pipeline_logger.py        # Enhanced logging (100-150 lines)
│   ├── error_handler.py          # Enhanced error handling (150-200 lines)
│   ├── formatters.py             # Price/currency formatting (100-150 lines)
│   ├── validators.py             # Data validation (100-200 lines)
│   └── service_manager.py        # Service coordination (150-200 lines)
├── config/
│   ├── __init__.py
│   ├── pipeline_config.py        # Pipeline configuration (100-150 lines)
│   ├── analysis_config.py        # Analysis parameters (100-150 lines)
│   ├── angel_one_config.py       # Angel One configuration (100-150 lines)
│   └── database_config.py        # Database configuration (100-150 lines)
├── services/
│   ├── __init__.py
│   ├── data_service_wrapper.py   # Data service integration (200-300 lines)
│   ├── model_service_wrapper.py  # Model service integration (200-300 lines)
│   ├── strategy_service_wrapper.py # Strategy service integration (200-300 lines)
│   ├── reporting_service_wrapper.py # Reporting service integration (200-300 lines)
│   ├── angel_one_manager.py      # Angel One API management (300-400 lines)
│   ├── database_manager.py       # Database management (200-300 lines)
│   └── api_coordinator.py        # API coordination and rate limiting (150-200 lines)
└── main.py                       # Entry point (100-150 lines)
```

## 🔄 **Updated Transformation Strategy**

### **Phase 1: Service Integration Layer (Week 1)**

#### **1.1 Create Service Wrappers**

```python
# main/services/data_service_wrapper.py
from src.core import DataService
from src.core.database_service import DatabaseService
from src.core.enhanced_angel_one_service import EnhancedAngelOneService
from typing import Dict, Any, Optional
import pandas as pd

class DataServiceWrapper:
    """Wrapper for data service with enhanced functionality"""

    def __init__(self, ticker: str, config: Dict[str, Any]):
        self.ticker = ticker
        self.config = config
        self.data_service = DataService(use_database=True)
        self.angel_service = None
        self.database_service = None

        # Initialize Angel One service if Indian stock
        if self._is_indian_stock(ticker):
            self.angel_service = EnhancedAngelOneService()

        # Initialize database service
        self.database_service = DatabaseService()

    def load_stock_data(self, period: str, interval: str = 'ONE_DAY') -> pd.DataFrame:
        """Load stock data from appropriate source"""
        try:
            if self._is_indian_stock(self.ticker) and self.angel_service:
                # Use Angel One for Indian stocks
                return self._load_angel_one_data(period, interval)
            else:
                # Use Yahoo Finance for US/International stocks
                return self._load_yahoo_finance_data(period)
        except Exception as e:
            # Fallback to Yahoo Finance
            return self._load_yahoo_finance_data(period)

    def _load_angel_one_data(self, period: str, interval: str) -> pd.DataFrame:
        """Load data from Angel One API"""
        # Implementation using enhanced_angel_one_service
        pass

    def _load_yahoo_finance_data(self, period: str) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        # Implementation using data_service
        pass

    def _is_indian_stock(self, ticker: str) -> bool:
        """Check if ticker is an Indian stock"""
        return ticker.endswith('.NS') or ticker.endswith('.BO')
```

#### **1.2 Create Angel One Manager**

```python
# main/services/angel_one_manager.py
from src.core.enhanced_angel_one_service import EnhancedAngelOneService
from src.core.angel_one_database_schema import AngelOneDatabaseSchema
from src.utils.rate_limiter import get_api_rate_limiter
from typing import Dict, Any, Optional
import pandas as pd

class AngelOneManager:
    """Manages Angel One API operations with rate limiting and caching"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.angel_service = EnhancedAngelOneService()
        self.db_schema = AngelOneDatabaseSchema()
        self.rate_limiter = get_api_rate_limiter()

    def get_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Get stock data with rate limiting and caching"""
        try:
            # Apply rate limiting
            return self.rate_limiter.call_with_retry(
                'angel_one',
                self._fetch_stock_data,
                ticker, period, interval
            )
        except Exception as e:
            raise Exception(f"Angel One data fetch failed: {e}")

    def _fetch_stock_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch stock data from Angel One API"""
        # Implementation using enhanced_angel_one_service
        pass

    def store_data_in_database(self, ticker: str, data: pd.DataFrame, interval: str):
        """Store data in database using Angel One schema"""
        # Implementation using angel_one_database_schema
        pass

    def get_cached_data(self, ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Get cached data from database"""
        # Implementation using database service
        pass
```

#### **1.3 Create Database Manager**

```python
# main/services/database_manager.py
from src.core.database_service import DatabaseService
from src.utils.database_pool import get_connection_pool
from typing import Dict, Any, Optional
import pandas as pd

class DatabaseManager:
    """Manages database operations with connection pooling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_service = DatabaseService()
        self.connection_pool = get_connection_pool()

    def store_stock_data(self, ticker: str, data: pd.DataFrame, source: str):
        """Store stock data in database"""
        try:
            with self.connection_pool.get_connection_context() as conn:
                # Store data using appropriate schema based on source
                if source == 'angel_one':
                    self._store_angel_one_data(conn, ticker, data)
                else:
                    self._store_yahoo_data(conn, ticker, data)
        except Exception as e:
            raise Exception(f"Database storage failed: {e}")

    def get_stock_data(self, ticker: str, period: str, source: str) -> Optional[pd.DataFrame]:
        """Get stock data from database"""
        try:
            with self.connection_pool.get_connection_context() as conn:
                if source == 'angel_one':
                    return self._get_angel_one_data(conn, ticker, period)
                else:
                    return self._get_yahoo_data(conn, ticker, period)
        except Exception as e:
            raise Exception(f"Database retrieval failed: {e}")

    def _store_angel_one_data(self, conn, ticker: str, data: pd.DataFrame):
        """Store Angel One data using appropriate schema"""
        # Implementation using angel_one_database_schema
        pass

    def _store_yahoo_data(self, conn, ticker: str, data: pd.DataFrame):
        """Store Yahoo Finance data"""
        # Implementation using standard database schema
        pass

    def _get_angel_one_data(self, conn, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Get Angel One data from database"""
        # Implementation
        pass

    def _get_yahoo_data(self, conn, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Get Yahoo Finance data from database"""
        # Implementation
        pass
```

### **Phase 2: Enhanced Data Processor (Week 1-2)**

#### **2.1 Updated Data Processor with Service Integration**

```python
# main/pipeline/data_processor.py
from .base_pipeline import BasePipelineComponent
from ..services.data_service_wrapper import DataServiceWrapper
from ..services.angel_one_manager import AngelOneManager
from ..services.database_manager import DatabaseManager
from typing import Dict, Any
import pandas as pd

class DataProcessor(BasePipelineComponent):
    """Enhanced data processor with service integration"""

    def __init__(self, ticker: str, config: Dict[str, Any]):
        super().__init__(ticker, config)
        self.data_wrapper = DataServiceWrapper(ticker, config)
        self.angel_manager = AngelOneManager(config) if self._is_indian_stock(ticker) else None
        self.db_manager = DatabaseManager(config)

    def validate_inputs(self, **kwargs) -> bool:
        """Validate data processing inputs"""
        required_params = ['period']
        return all(param in kwargs for param in required_params)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute enhanced data processing pipeline"""
        try:
            self.logger.info(f"Starting data processing for {self.ticker}")

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

            result = {
                'raw_data': raw_data,
                'cleaned_data': cleaned_data,
                'enhanced_data': enhanced_data,
                'enriched_data': enriched_data,
                'features': features,
                'data_source': self._get_data_source(),
                'success': True
            }

            self.logger.info(f"Data processing completed for {self.ticker}")
            return result

        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            return {'success': False, 'error': str(e)}

    def _load_stock_data(self, period: str, interval: str) -> pd.DataFrame:
        """Load stock data from appropriate source"""
        return self.data_wrapper.load_stock_data(period, interval)

    def _store_data_in_database(self, data: pd.DataFrame, interval: str):
        """Store data in database"""
        source = 'angel_one' if self._is_indian_stock(self.ticker) else 'yahoo_finance'
        self.db_manager.store_stock_data(self.ticker, data, source)

    def _clean_and_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        # Implementation from original clean_and_preprocess
        pass

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        # Implementation from original add_enhanced_technical_indicators
        pass

    def _enrich_with_external_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enrich data with economic and market indicators"""
        # Implementation using economic_data_service, global_market_service, etc.
        pass

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for model training"""
        # Implementation from original prepare_features
        pass

    def _get_data_source(self) -> str:
        """Get the data source used"""
        return 'angel_one' if self._is_indian_stock(self.ticker) else 'yahoo_finance'

    def _is_indian_stock(self, ticker: str) -> bool:
        """Check if ticker is an Indian stock"""
        return ticker.endswith('.NS') or ticker.endswith('.BO')
```

### **Phase 3: Enhanced Strategy Analyzer (Week 2-3)**

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
from typing import Dict, Any
import pandas as pd

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

    def validate_inputs(self, **kwargs) -> bool:
        """Validate strategy analysis inputs"""
        required_params = ['enhanced_data']
        return all(param in kwargs for param in required_params)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute comprehensive strategy analysis pipeline"""
        try:
            self.logger.info(f"Starting strategy analysis for {self.ticker}")

            # Step 1: Sentiment analysis
            sentiment_results = self._run_sentiment_analysis()

            # Step 2: Market factors analysis
            market_factors = self._run_market_factors()

            # Step 3: Economic indicators
            economic_indicators = self._run_economic_indicators()

            # Step 4: Geopolitical risk analysis
            geopolitical_risk = self._run_geopolitical_analysis()

            # Step 5: Global market analysis
            global_market = self._run_global_market_analysis()

            # Step 6: Corporate actions analysis
            corporate_actions = self._run_corporate_actions_analysis()

            # Step 7: Insider trading analysis
            insider_trading = self._run_insider_trading_analysis()

            # Step 8: Currency analysis
            currency_analysis = self._run_currency_analysis()

            # Step 9: Trading strategy
            trading_strategy = self._run_trading_strategy(kwargs['enhanced_data'])

            # Step 10: Backtesting
            backtest_results = self._run_backtesting(kwargs['enhanced_data'])

            # Step 11: Balance sheet analysis
            balance_sheet = self._run_balance_sheet_analysis()

            # Step 12: Event impact analysis
            event_impact = self._run_event_impact_analysis()

            result = {
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

            self.logger.info(f"Strategy analysis completed for {self.ticker}")
            return result

        except Exception as e:
            self.logger.error(f"Strategy analysis failed: {e}")
            return {'success': False, 'error': str(e)}

    def _run_sentiment_analysis(self) -> Dict[str, Any]:
        """Run sentiment analysis using strategy service"""
        # Implementation using strategy_service
        pass

    def _run_market_factors(self) -> Dict[str, Any]:
        """Run market factors analysis"""
        # Implementation using global_market_service
        pass

    def _run_economic_indicators(self) -> Dict[str, Any]:
        """Run economic indicators analysis"""
        # Implementation using economic_data_service and fred_api_service
        pass

    def _run_geopolitical_analysis(self) -> Dict[str, Any]:
        """Run geopolitical risk analysis"""
        # Implementation using geopolitical_risk_service
        pass

    def _run_global_market_analysis(self) -> Dict[str, Any]:
        """Run global market analysis"""
        # Implementation using global_market_service
        pass

    def _run_corporate_actions_analysis(self) -> Dict[str, Any]:
        """Run corporate actions analysis"""
        # Implementation using corporate_action_service
        pass

    def _run_insider_trading_analysis(self) -> Dict[str, Any]:
        """Run insider trading analysis"""
        # Implementation using insider_trading_service
        pass

    def _run_currency_analysis(self) -> Dict[str, Any]:
        """Run currency analysis"""
        # Implementation using currency_service
        pass

    def _run_trading_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run trading strategy analysis"""
        # Implementation from original run_trading_strategy
        pass

    def _run_backtesting(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtesting analysis"""
        # Implementation from original run_backtesting
        pass

    def _run_balance_sheet_analysis(self) -> Dict[str, Any]:
        """Run balance sheet analysis"""
        # Implementation from original run_balance_sheet_analysis
        pass

    def _run_event_impact_analysis(self) -> Dict[str, Any]:
        """Run event impact analysis"""
        # Implementation from original run_event_impact_analysis
        pass
```

### **Phase 4: Enhanced User Interface with Angel One Integration (Week 3-4)**

#### **4.1 Angel One Configuration Interface**

```python
# main/interfaces/angel_one_interface.py
from typing import Dict, Any, Optional
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

    def get_angel_one_limits(self, interval: str) -> int:
        """Get Angel One data limits for interval"""
        return self.angel_one_limits.get(interval, 2000)
```

#### **4.2 Enhanced User Interface**

```python
# main/interfaces/user_interface.py
from .interactive_selector import InteractiveDataSelector
from .angel_one_interface import AngelOneInterface
from typing import Dict, Any

class UserInterface:
    """Enhanced user interface with Angel One integration"""

    def __init__(self):
        self.interactive_selector = InteractiveDataSelector()
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

            # Get analysis type
            analysis_type = self._get_analysis_type()

            # Get analysis parameters
            if analysis_type == 'interactive':
                params = self.interactive_selector.run_interactive_selection()
            else:
                params = self._get_standard_parameters()

            # Get enhanced features preference
            use_enhanced = self._get_enhanced_features_preference()

            # Get database preferences
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

    def _get_ticker_input(self) -> str:
        """Get ticker input from user"""
        ticker = input("Enter stock ticker (e.g., AAPL, TCS, RELIANCE): ").strip().upper()
        if not ticker:
            ticker = "AAPL"
        return ticker

    def _is_indian_stock(self, ticker: str) -> bool:
        """Check if ticker is an Indian stock"""
        return ticker.endswith('.NS') or ticker.endswith('.BO')

    def _get_analysis_type(self) -> str:
        """Get analysis type from user"""
        print("\n🔧 Select Analysis Type:")
        print("1. Standard Analysis (Predefined parameters)")
        print("2. Interactive Analysis (Custom data selection)")

        analysis_choice = input("Enter your choice (1-2, default: 1): ").strip()
        return 'interactive' if analysis_choice == '2' else 'standard'

    def _get_standard_parameters(self) -> Dict[str, Any]:
        """Get standard analysis parameters"""
        return {
            'period': '1y',
            'interval': 'ONE_DAY',
            'days_ahead': 5,
            'use_enhanced': True
        }

    def _get_enhanced_features_preference(self) -> bool:
        """Get enhanced features preference"""
        use_enhanced = input("Use enhanced features? (y/n, default: y): ").strip().lower()
        return use_enhanced != 'n'

    def _get_database_preference(self) -> bool:
        """Get database usage preference"""
        use_database = input("Use database for data storage? (y/n, default: y): ").strip().lower()
        return use_database != 'n'
```

### **Phase 5: API Coordinator and Service Management (Week 4)**

#### **5.1 API Coordinator**

```python
# main/services/api_coordinator.py
from src.utils.rate_limiter import get_api_rate_limiter
from typing import Dict, Any, List
import asyncio
import time

class APICoordinator:
    """Coordinates multiple API calls with rate limiting and error handling"""

    def __init__(self):
        self.rate_limiter = get_api_rate_limiter()
        self.api_stats = {}

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

    def _load_angel_one_data(self, ticker: str, config: Dict[str, Any]):
        """Load Angel One data with rate limiting"""
        return self.rate_limiter.call_with_retry(
            'angel_one',
            self._fetch_angel_one_data,
            ticker, config
        )

    def _load_yahoo_data(self, ticker: str, config: Dict[str, Any]):
        """Load Yahoo Finance data"""
        return self.rate_limiter.call_with_retry(
            'yahoo_finance',
            self._fetch_yahoo_data,
            ticker, config
        )

    def _load_economic_data(self):
        """Load economic data from FRED API"""
        return self.rate_limiter.call_with_retry(
            'fred',
            self._fetch_economic_data
        )

    def _load_market_data(self):
        """Load global market data"""
        # Implementation
        pass

    def _load_currency_data(self):
        """Load currency data"""
        # Implementation
        pass

    def _fetch_angel_one_data(self, ticker: str, config: Dict[str, Any]):
        """Fetch Angel One data"""
        # Implementation
        pass

    def _fetch_yahoo_data(self, ticker: str, config: Dict[str, Any]):
        """Fetch Yahoo Finance data"""
        # Implementation
        pass

    def _fetch_economic_data(self):
        """Fetch economic data"""
        # Implementation
        pass
```

#### **5.2 Service Manager**

```python
# main/utils/service_manager.py
from typing import Dict, Any, Optional
import logging

class ServiceManager:
    """Manages and coordinates all services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}
        self.logger = logging.getLogger(__name__)

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

            self.logger.info("All services initialized successfully")
            return {'success': True, 'services': list(self.services.keys())}

        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_service(self, service_name: str):
        """Get a specific service"""
        return self.services.get(service_name)

    def _is_indian_stock(self, ticker: str) -> bool:
        """Check if ticker is an Indian stock"""
        return ticker.endswith('.NS') or ticker.endswith('.BO')

    def _init_data_service(self, ticker: str):
        """Initialize data service"""
        # Implementation
        pass

    def _init_model_service(self):
        """Initialize model service"""
        # Implementation
        pass

    def _init_strategy_service(self):
        """Initialize strategy service"""
        # Implementation
        pass

    def _init_reporting_service(self):
        """Initialize reporting service"""
        # Implementation
        pass

    def _init_angel_one_service(self):
        """Initialize Angel One service"""
        # Implementation
        pass

    def _init_database_service(self):
        """Initialize database service"""
        # Implementation
        pass

    def _init_economic_service(self):
        """Initialize economic data service"""
        # Implementation
        pass

    def _init_fred_service(self):
        """Initialize FRED API service"""
        # Implementation
        pass

    def _init_geopolitical_service(self):
        """Initialize geopolitical risk service"""
        # Implementation
        pass

    def _init_global_market_service(self):
        """Initialize global market service"""
        # Implementation
        pass

    def _init_corporate_action_service(self):
        """Initialize corporate action service"""
        # Implementation
        pass

    def _init_insider_trading_service(self):
        """Initialize insider trading service"""
        # Implementation
        pass

    def _init_currency_service(self):
        """Initialize currency service"""
        # Implementation
        pass
```

## 📋 **Updated Implementation Timeline**

### **Week 1: Service Integration Foundation**

- [ ] Create service wrapper classes
- [ ] Implement Angel One manager
- [ ] Create database manager
- [ ] Set up API coordinator

### **Week 2: Enhanced Components**

- [ ] Update data processor with service integration
- [ ] Implement model trainer with service integration
- [ ] Create enhanced strategy analyzer
- [ ] Add service management

### **Week 3: User Interface & Integration**

- [ ] Create Angel One configuration interface
- [ ] Update user interface with Angel One support
- [ ] Implement interactive selector enhancements
- [ ] Add database configuration options

### **Week 4: Testing & Validation**

- [ ] Test all service integrations
- [ ] Validate Angel One API functionality
- [ ] Test database operations
- [ ] Comprehensive integration testing

## 🎯 **Key Integration Points**

### **Angel One API Integration:**

- **Configuration management** for Indian stocks
- **Rate limiting** and error handling
- **Database storage** using Angel One schema
- **Fallback mechanisms** to Yahoo Finance

### **Database Integration:**

- **Connection pooling** for performance
- **Multiple schemas** (Angel One, Yahoo Finance)
- **Data caching** and retrieval
- **Incremental updates**

### **Service Coordination:**

- **Unified service management**
- **API rate limiting** across all services
- **Error handling** and fallback mechanisms
- **Performance optimization**

## 🎉 **Expected Benefits**

### **Enhanced Functionality:**

- **Complete Angel One integration** for Indian stocks
- **Robust database operations** with connection pooling
- **Comprehensive service coordination**
- **Advanced error handling** and fallback mechanisms

### **Performance Improvements:**

- **Optimized API calls** with rate limiting
- **Efficient database operations** with pooling
- **Cached data retrieval** for faster processing
- **Parallel service execution** where possible

### **Reliability Improvements:**

- **Fallback mechanisms** for API failures
- **Robust error handling** throughout
- **Service health monitoring**
- **Graceful degradation** when services are unavailable

This updated transformation plan ensures complete integration with all existing core services, Angel One API, and database systems while maintaining the modular polylithic architecture benefits.
