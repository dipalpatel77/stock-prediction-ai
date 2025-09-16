# AI Stock Predictor - API Workflow Reference

## 🎯 **Complete API Workflow Documentation**

This document provides a comprehensive reference for all API endpoints, methods, and workflow interactions in the AI Stock Predictor system.

## 📋 **Table of Contents**

1. [Core Pipeline API](#core-pipeline-api)
2. [Data Service API](#data-service-api)
3. [Model Service API](#model-service-api)
4. [Strategy Service API](#strategy-service-api)
5. [Database Service API](#database-service-api)
6. [Reporting Service API](#reporting-service-api)
7. [Integration APIs](#integration-apis)
8. [Error Handling API](#error-handling-api)
9. [Configuration API](#configuration-api)

---

## 🚀 **Core Pipeline API**

### **UnifiedAnalysisPipeline**

#### **Initialization**

```python
class UnifiedAnalysisPipeline:
    def __init__(self, ticker, max_workers=None, period_config="recommended"):
        """
        Initialize the Unified Analysis Pipeline.

        Args:
            ticker (str): Stock ticker symbol
            max_workers (int, optional): Number of worker threads
            period_config (str): Data period configuration

        Returns:
            UnifiedAnalysisPipeline: Initialized pipeline instance
        """
```

#### **Main Execution Methods**

##### **run_unified_analysis**

```python
def run_unified_analysis(self, period=None, days_ahead=5, use_enhanced=True):
    """
    Run complete unified analysis workflow.

    Args:
        period (str, optional): Data period ('1mo', '3mo', '6mo', '1y', '2y')
        days_ahead (int): Number of days for prediction horizon
        use_enhanced (bool): Whether to use enhanced analysis features

    Returns:
        bool: Success status of the analysis

    Workflow:
        1. PartA - Data Preprocessing
        2. PartB - Model Training
        3. PartC - Strategy Analysis
        4. Phase 1 - Enhanced Analysis
        5. Phase 2 - Economic Data
        6. Phase 3 - Risk Assessment
        7. Prediction Generation
        8. Report Generation
    """
```

##### **run_complete_analysis**

```python
def run_complete_analysis(self, period="1y", days_ahead=5):
    """
    Run complete analysis with all phases.

    Args:
        period (str): Data period for analysis
        days_ahead (int): Prediction horizon

    Returns:
        dict: Complete analysis results
    """
```

#### **Individual Phase Methods**

##### **run_partA_preprocessing**

```python
def run_partA_preprocessing(self, period=None):
    """
    Execute PartA data preprocessing workflow.

    Args:
        period (str, optional): Data period

    Returns:
        bool: Success status

    Workflow:
        1. Load stock data
        2. Preprocess data
        3. Validate data quality
        4. Store in database
    """
```

##### **run_partB_model_training**

```python
def run_partB_model_training(self):
    """
    Execute PartB model training workflow.

    Returns:
        bool: Success status

    Workflow:
        1. Prepare training data
        2. Train 15+ ML models
        3. Create ensemble model
        4. Evaluate model performance
        5. Save trained models
    """
```

##### **run_partC_strategy_analysis**

```python
def run_partC_strategy_analysis(self, use_enhanced=True):
    """
    Execute PartC strategy analysis workflow.

    Args:
        use_enhanced (bool): Use enhanced analysis features

    Returns:
        bool: Success status

    Workflow:
        1. Technical analysis
        2. Fundamental analysis
        3. Risk assessment
        4. Trading signal generation
    """
```

#### **Prediction Methods**

##### **generate_and_display_predictions**

```python
def generate_and_display_predictions(self, days_ahead=5):
    """
    Generate and display multi-timeframe predictions.

    Args:
        days_ahead (int): Number of days to predict ahead

    Returns:
        dict: Prediction results with confidence scores
    """
```

##### **generate_multi_day_predictions**

```python
def _generate_multi_day_predictions(self, X, days_ahead):
    """
    Generate multi-day predictions using ensemble models.

    Args:
        X (pd.DataFrame): Feature data
        days_ahead (int): Number of days to predict

    Returns:
        list: List of predicted prices for each day
    """
```

---

## 📊 **Data Service API**

### **DataService**

#### **Initialization**

```python
class DataService:
    def __init__(self, cache_dir="data/cache", period_config="recommended", use_database=True):
        """
        Initialize Data Service.

        Args:
            cache_dir (str): Cache directory path
            period_config (str): Period configuration
            use_database (bool): Enable database integration
        """
```

#### **Data Loading Methods**

##### **load_stock_data**

```python
def load_stock_data(self, ticker, period=None, interval='1d', force_refresh=False, start_date=None, end_date=None):
    """
    Load stock data with intelligent source selection.

    Args:
        ticker (str): Stock ticker symbol
        period (str, optional): Data period
        interval (str): Data interval
        force_refresh (bool): Force data refresh
        start_date (str, optional): Custom start date
        end_date (str, optional): Custom end date

    Returns:
        pd.DataFrame: Stock data

    Workflow:
        1. Check database for existing data
        2. Determine data source (Angel One vs Yahoo Finance)
        3. Download data if needed
        4. Store in database
        5. Return processed data
    """
```

##### **load_multi_exchange_data**

```python
def load_multi_exchange_data(self, ticker, period="1y", interval="1d"):
    """
    Load and fuse data from multiple exchanges.

    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period
        interval (str): Data interval

    Returns:
        pd.DataFrame: Fused multi-exchange data
    """
```

#### **Data Processing Methods**

##### **preprocess_data**

```python
def preprocess_data(self, df, timeframe='daily', target_col='Close'):
    """
    Preprocess stock data for analysis.

    Args:
        df (pd.DataFrame): Raw stock data
        timeframe (str): Analysis timeframe
        target_col (str): Target column name

    Returns:
        pd.DataFrame: Preprocessed data

    Workflow:
        1. Column normalization
        2. Date handling
        3. Missing value treatment
        4. Technical indicators
        5. Feature engineering
    """
```

##### **validate_data_quality**

```python
def validate_data_quality(self, df):
    """
    Validate data quality and completeness.

    Args:
        df (pd.DataFrame): Data to validate

    Returns:
        dict: Quality metrics
    """
```

---

## 🤖 **Model Service API**

### **ModelService**

#### **Initialization**

```python
class ModelService:
    def __init__(self):
        """Initialize Model Service with ML algorithms."""
```

#### **Model Training Methods**

##### **train_models**

```python
def train_models(self, data, target_col='Close'):
    """
    Train multiple ML models for prediction.

    Args:
        data (pd.DataFrame): Training data
        target_col (str): Target column

    Returns:
        tuple: (models_dict, scalers_dict, feature_importance)

    Algorithms:
        - RandomForestRegressor
        - GradientBoostingRegressor
        - XGBRegressor
        - LGBMRegressor
        - CatBoostRegressor
        - LinearRegression
        - Ridge, Lasso, ElasticNet
        - SVR, MLPRegressor
        - GaussianProcessRegressor
        - AdaBoostRegressor
        - ExtraTreesRegressor
        - HuberRegressor
        - KernelRidge
    """
```

##### **create_ensemble_model**

```python
def create_ensemble_model(self, models, scalers, X, y):
    """
    Create ensemble model from individual models.

    Args:
        models (dict): Trained models
        scalers (dict): Data scalers
        X (pd.DataFrame): Features
        y (pd.Series): Target

    Returns:
        object: Ensemble model
    """
```

#### **Prediction Methods**

##### **generate_predictions**

```python
def generate_predictions(self, models, scalers, data, days_ahead=5):
    """
    Generate predictions using trained models.

    Args:
        models (dict): Trained models
        scalers (dict): Data scalers
        data (pd.DataFrame): Input data
        days_ahead (int): Prediction horizon

    Returns:
        dict: Predictions with confidence scores
    """
```

##### **evaluate_models**

```python
def evaluate_models(self, models, X_test, y_test):
    """
    Evaluate model performance.

    Args:
        models (dict): Trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets

    Returns:
        dict: Model evaluation metrics
    """
```

---

## 📈 **Strategy Service API**

### **StrategyService**

#### **Initialization**

```python
class StrategyService:
    def __init__(self):
        """Initialize Strategy Service."""
```

#### **Analysis Methods**

##### **run_technical_analysis**

```python
def run_technical_analysis(self, data):
    """
    Perform technical analysis.

    Args:
        data (pd.DataFrame): Stock data

    Returns:
        dict: Technical analysis results

    Indicators:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Moving Averages
        - Volume indicators
    """
```

##### **run_fundamental_analysis**

```python
def run_fundamental_analysis(self, ticker):
    """
    Perform fundamental analysis.

    Args:
        ticker (str): Stock ticker

    Returns:
        dict: Fundamental analysis results

    Metrics:
        - P/E Ratio
        - P/B Ratio
        - Debt-to-Equity
        - ROE, ROA
        - Growth rates
    """
```

##### **assess_risk**

```python
def assess_risk(self, data, predictions):
    """
    Assess investment risk.

    Args:
        data (pd.DataFrame): Historical data
        predictions (dict): Model predictions

    Returns:
        dict: Risk assessment metrics

    Metrics:
        - Value at Risk (VaR)
        - Volatility
        - Sharpe Ratio
        - Maximum Drawdown
        - Beta
    """
```

#### **Signal Generation**

##### **generate_trading_signals**

```python
def generate_trading_signals(self, data, predictions):
    """
    Generate trading signals.

    Args:
        data (pd.DataFrame): Stock data
        predictions (dict): Model predictions

    Returns:
        dict: Trading signals and recommendations
    """
```

---

## 🗄️ **Database Service API**

### **DatabaseService**

#### **Initialization**

```python
class DatabaseService:
    def __init__(self, db_type="mysql", connection_string=None):
        """
        Initialize Database Service.

        Args:
            db_type (str): Database type ('mysql', 'postgresql', 'sqlite')
            connection_string (str): Database connection string
        """
```

#### **Data Storage Methods**

##### **store_stock_data**

```python
def store_stock_data(self, ticker, data, data_source="yfinance"):
    """
    Store stock data in database.

    Args:
        ticker (str): Stock ticker
        data (pd.DataFrame): Stock data
        data_source (str): Data source identifier

    Returns:
        bool: Success status
    """
```

##### **get_stock_data**

```python
def get_stock_data(self, ticker, start_date=None, end_date=None, limit=None):
    """
    Retrieve stock data from database.

    Args:
        ticker (str): Stock ticker
        start_date (str, optional): Start date filter
        end_date (str, optional): End date filter
        limit (int, optional): Maximum records

    Returns:
        pd.DataFrame: Stock data
    """
```

#### **Data Management Methods**

##### **update_stock_data**

```python
def update_stock_data(self, ticker, new_data, data_source="yfinance"):
    """
    Update existing stock data.

    Args:
        ticker (str): Stock ticker
        new_data (pd.DataFrame): New data
        data_source (str): Data source

    Returns:
        bool: Success status
    """
```

##### **get_data_quality_metrics**

```python
def get_data_quality_metrics(self, ticker):
    """
    Get data quality metrics.

    Args:
        ticker (str): Stock ticker

    Returns:
        dict: Quality metrics
    """
```

---

## 📋 **Reporting Service API**

### **ReportingService**

#### **Initialization**

```python
class ReportingService:
    def __init__(self):
        """Initialize Reporting Service."""
```

#### **Report Generation Methods**

##### **generate_comprehensive_report**

```python
def generate_comprehensive_report(self, analysis_results, ticker):
    """
    Generate comprehensive analysis report.

    Args:
        analysis_results (dict): Analysis results
        ticker (str): Stock ticker

    Returns:
        dict: Comprehensive report data
    """
```

##### **export_reports**

```python
def export_reports(self, report_data, ticker, formats=['html', 'csv', 'json']):
    """
    Export reports in multiple formats.

    Args:
        report_data (dict): Report data
        ticker (str): Stock ticker
        formats (list): Export formats

    Returns:
        dict: Export file paths
    """
```

#### **Visualization Methods**

##### **create_dashboard**

```python
def create_dashboard(self, data, predictions, ticker):
    """
    Create interactive dashboard.

    Args:
        data (pd.DataFrame): Stock data
        predictions (dict): Predictions
        ticker (str): Stock ticker

    Returns:
        str: Dashboard HTML content
    """
```

##### **generate_charts**

```python
def generate_charts(self, data, predictions):
    """
    Generate analysis charts.

    Args:
        data (pd.DataFrame): Stock data
        predictions (dict): Predictions

    Returns:
        dict: Chart data and configurations
    """
```

---

## 🔗 **Integration APIs**

### **Angel One Integration**

#### **AngelOneDataDownloader**

```python
class AngelOneDataDownloader:
    def authenticate(self):
        """Authenticate with Angel One API."""

    def get_historical_data(self, symbol_name, exchange, interval, days_back):
        """Get historical data from Angel One."""

    def get_symbol_token(self, symbol, exchange):
        """Get symbol token for Angel One API."""
```

### **Yahoo Finance Integration**

```python
def _download_from_yahoo(self, ticker, period, interval):
    """Download data from Yahoo Finance."""
```

### **FRED API Integration**

```python
class FredAPIService:
    def get_economic_data(self, series_ids):
        """Get economic data from FRED API."""
```

---

## ⚠️ **Error Handling API**

### **ErrorHandler**

```python
class ErrorHandler:
    @staticmethod
    def handle_analysis_error(context, error, fallback=""):
        """Handle analysis errors with fallback strategies."""

    @staticmethod
    def handle_data_error(ticker, error):
        """Handle data loading errors."""

    @staticmethod
    def handle_model_error(model_name, error):
        """Handle model training errors."""
```

### **PipelineLogger**

```python
class PipelineLogger:
    @staticmethod
    def success(message, details=""):
        """Log success messages."""

    @staticmethod
    def warning(message, details=""):
        """Log warning messages."""

    @staticmethod
    def error(message, details=""):
        """Log error messages."""
```

---

## ⚙️ **Configuration API**

### **AnalysisConfig**

```python
class AnalysisConfig:
    def __init__(self):
        """Initialize analysis configuration."""

    def get_model_config(self):
        """Get model configuration."""

    def get_data_config(self):
        """Get data configuration."""
```

### **DatabaseConfig**

```python
def get_database_config(preset="local"):
    """Get database configuration."""
```

### **AngelOneConfig**

```python
class AngelOneConfig:
    def get_interval_mapping(self):
        """Get interval mapping for Angel One API."""

    def get_max_days_limits(self):
        """Get maximum days limits for different intervals."""
```

---

## 🚀 **Usage Examples**

### **Basic Usage**

```python
# Initialize pipeline
pipeline = UnifiedAnalysisPipeline("RELIANCE")

# Run complete analysis
success = pipeline.run_unified_analysis(period="1y", days_ahead=5)

# Get predictions
predictions = pipeline.generate_and_display_predictions(days_ahead=5)
```

### **Advanced Usage**

```python
# Initialize with custom configuration
pipeline = UnifiedAnalysisPipeline(
    ticker="TCS",
    max_workers=8,
    period_config="comprehensive"
)

# Run individual phases
data_success = pipeline.run_partA_preprocessing(period="2y")
model_success = pipeline.run_partB_model_training()
strategy_success = pipeline.run_partC_strategy_analysis(use_enhanced=True)

# Generate custom predictions
predictions = pipeline.generate_and_display_predictions(days_ahead=10)
```

### **Database Integration**

```python
# Initialize with database
data_service = DataService(use_database=True)

# Load data (will use database if available)
data = data_service.load_stock_data("RELIANCE", period="1y")

# Store new data
success = data_service.db_service.store_stock_data("RELIANCE", data, "angel_one")
```

### **Multi-Exchange Data**

```python
# Load multi-exchange data
fusion_service = MultiExchangeDataService(use_database=True)
fused_data = fusion_service.load_multi_exchange_data("RELIANCE", period="1y")

# Get fusion quality metrics
metrics = fusion_service.get_fusion_quality_metrics(fused_data)
```

---

This comprehensive API reference provides detailed documentation for all workflow interactions in the AI Stock Predictor system, enabling developers to understand and utilize the complete system functionality.
