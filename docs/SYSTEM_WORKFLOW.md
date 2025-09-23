# AI Stock Predictor - Complete System Workflow Documentation

## 🎯 **System Overview**

The AI Stock Predictor is a comprehensive, multi-layered system that provides advanced stock market analysis and prediction capabilities. The system follows a structured workflow that integrates data collection, preprocessing, model training, analysis, and reporting.

## 📊 **High-Level Workflow Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  Command Line    │  Python API    │  Jupyter Notebook          │
│  run_analysis.py │  UnifiedPipeline│  notebook_interface.py     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│              UnifiedAnalysisPipeline                           │
│  • Coordinates all services                                    │
│  • Manages workflow execution                                  │
│  • Handles error recovery                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE SERVICES LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  DataService    │  ModelService   │  StrategyService           │
│  • Data loading │  • Model training│  • Trading strategies      │
│  • Preprocessing│  • Predictions   │  • Signal generation       │
│  • Validation   │  • Evaluation    │  • Risk management         │
├─────────────────────────────────────────────────────────────────┤
│  DatabaseService│  ReportingService│  IncrementalDataService   │
│  • Data storage │  • Report gen    │  • Smart updates           │
│  • Retrieval    │  • Visualization │  • Change detection        │
│  • Optimization │  • Export        │  • Merge strategies        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  Angel One API  │  Yahoo Finance  │  FRED API                  │
│  • Indian stocks│  • US stocks     │  • Economic data           │
│  • Real-time    │  • Historical    │  • Market indicators       │
│  • F&O data     │  • Multiple      │  • Currency rates          │
│                 │    intervals     │                            │
├─────────────────────────────────────────────────────────────────┤
│  Phase1Integration│ Phase2Integration│ Phase3Integration        │
│  • Basic analysis│  • Advanced ML   │  • Risk assessment        │
│  • Technical     │  • Ensemble      │  • Geopolitical           │
│    indicators    │    methods       │  • Corporate actions      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  MySQL Database │  File System    │  Cache System              │
│  • Stock data   │  • CSV files     │  • In-memory cache         │
│  • Metadata     │  • Models        │  • Session storage         │
│  • Quality      │  • Reports       │  • Temporary data          │
│    metrics      │  • Logs          │                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 **Detailed Workflow Steps**

### **Phase 1: System Initialization**

#### **1.1 Pipeline Initialization**

```python
# Entry Point: main/unified_analysis_pipeline.py
pipeline = UnifiedAnalysisPipeline(
    ticker="RELIANCE",
    max_workers=4,
    period_config="recommended"
)
```

**What Happens:**

- ✅ **Ticker Format Fixing**: Converts ticker to proper format
- ✅ **Worker Configuration**: Sets optimal number of workers
- ✅ **Service Initialization**: Initializes all core services
- ✅ **Configuration Loading**: Loads data periods and analysis config
- ✅ **Database Connection**: Establishes database connections

#### **1.2 Service Initialization**

```python
# Core Services Initialized:
self.data_service = DataService(use_database=True)
self.model_service = ModelService()
self.strategy_service = StrategyService()
self.reporting_service = ReportingService()
```

**Services Initialized:**

- **DataService**: Data loading, preprocessing, validation
- **ModelService**: ML model training and prediction
- **StrategyService**: Trading strategy analysis
- **ReportingService**: Report generation and export
- **DatabaseService**: Database operations
- **IncrementalDataService**: Smart data updates

### **Phase 2: Data Collection & Preprocessing (PartA)**

#### **2.1 Data Source Selection**

```python
def _download_stock_data(self, ticker, period, interval):
    if self._is_indian_stock(ticker):
        # Use Angel One API for Indian stocks
        return self._download_from_angel_one(ticker, period, interval)
    else:
        # Use Yahoo Finance for US stocks
        return self._download_from_yahoo(ticker, period, interval)
```

**Data Source Logic:**

- 🇮🇳 **Indian Stocks**: Angel One SmartAPI (NSE/BSE)
- 🇺🇸 **US Stocks**: Yahoo Finance
- 🔄 **Fallback**: Intelligent fallback mechanisms

#### **2.2 Data Download Process**

```python
# Angel One Data Download
angel_downloader = AngelOneDataDownloader()
angel_downloader.authenticate()  # TOTP-based authentication
data = angel_downloader.get_historical_data(
    symbol_name="RELIANCE",
    exchange="BSE",
    interval="ONE_DAY",
    days_back=365
)
```

**Download Steps:**

1. **Authentication**: TOTP-based Angel One authentication
2. **Symbol Resolution**: Map ticker to Angel One token
3. **Data Request**: Fetch historical data with optimal parameters
4. **Data Validation**: Validate data quality and completeness
5. **Database Storage**: Store data in MySQL database

#### **2.3 Data Preprocessing**

```python
def preprocess_data(self, df, timeframe='daily'):
    # Column normalization
    df = self._normalize_columns(df)

    # Date handling
    df = self._handle_dates(df)

    # Missing value handling
    df = self._handle_missing_values(df)

    # Technical indicators
    df = self._add_technical_indicators(df)

    # Feature engineering
    df = self._add_features(df)

    return df
```

**Preprocessing Steps:**

- ✅ **Column Normalization**: Standardize column names
- ✅ **Date Processing**: Handle timezone and date formats
- ✅ **Missing Value Handling**: Forward fill, interpolation
- ✅ **Technical Indicators**: RSI, MACD, Moving Averages
- ✅ **Feature Engineering**: Price changes, volatility, volume ratios

### **Phase 3: Model Training (PartB)**

#### **3.1 Model Initialization**

```python
# 15+ Machine Learning Algorithms
algorithms = [
    'RandomForestRegressor',
    'GradientBoostingRegressor',
    'XGBRegressor',
    'LGBMRegressor',
    'CatBoostRegressor',
    'LinearRegression',
    'Ridge', 'Lasso', 'ElasticNet',
    'SVR', 'MLPRegressor',
    'GaussianProcessRegressor',
    'AdaBoostRegressor',
    'ExtraTreesRegressor',
    'HuberRegressor',
    'KernelRidge'
]
```

#### **3.2 Model Training Process**

```python
def train_models(self, data):
    models = {}
    scalers = {}

    for algorithm in algorithms:
        try:
            # Data scaling
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # Model training
            model = algorithm()
            model.fit(X_scaled, y)

            # Model evaluation
            score = model.score(X_scaled, y)

            models[algorithm] = model
            scalers[algorithm] = scaler

        except Exception as e:
            print(f"Model {algorithm} training failed: {e}")

    return models, scalers
```

**Training Steps:**

1. **Data Preparation**: Feature selection and scaling
2. **Model Training**: Train 15+ algorithms in parallel
3. **Model Evaluation**: Cross-validation and scoring
4. **Model Persistence**: Save trained models
5. **Ensemble Creation**: Create ensemble model

### **Phase 4: Strategy Analysis (PartC)**

#### **4.1 Technical Analysis**

```python
def run_technical_analysis(self, data):
    # Technical indicators
    rsi = self._calculate_rsi(data['Close'])
    macd = self._calculate_macd(data['Close'])
    bollinger = self._calculate_bollinger_bands(data['Close'])

    # Trading signals
    signals = self._generate_trading_signals(data)

    return {
        'rsi': rsi,
        'macd': macd,
        'bollinger': bollinger,
        'signals': signals
    }
```

#### **4.2 Risk Assessment**

```python
def assess_risk(self, data, predictions):
    # Volatility analysis
    volatility = data['Close'].pct_change().std()

    # Value at Risk (VaR)
    var_95 = np.percentile(predictions, 5)
    var_99 = np.percentile(predictions, 1)

    # Risk metrics
    risk_metrics = {
        'volatility': volatility,
        'var_95': var_95,
        'var_99': var_99,
        'sharpe_ratio': self._calculate_sharpe_ratio(data)
    }

    return risk_metrics
```

### **Phase 5: Enhanced Analysis Integration**

#### **5.1 Phase 1: Enhanced Fundamental Analysis**

```python
def run_phase1_enhanced_analysis(self):
    # Enhanced fundamental analysis
    fundamental_data = self.phase1_integration.run_enhanced_analysis(self.ticker)

    # Economic indicators
    economic_data = self.economic_data_service.get_economic_indicators()

    # Market sentiment
    sentiment_data = self.sentiment_service.analyze_sentiment(self.ticker)

    return {
        'fundamental': fundamental_data,
        'economic': economic_data,
        'sentiment': sentiment_data
    }
```

#### **5.2 Phase 2: Economic Data & Regulatory Monitoring**

```python
def run_phase2_economic_analysis(self):
    # FRED API data
    fred_data = self.fred_service.get_economic_data()

    # World Bank data
    wb_data = self.world_bank_service.get_indicators()

    # Currency analysis
    currency_data = self.currency_service.get_currency_data()

    return {
        'fred': fred_data,
        'world_bank': wb_data,
        'currency': currency_data
    }
```

#### **5.3 Phase 3: Advanced Risk Assessment**

```python
def run_phase3_risk_analysis(self):
    # Geopolitical risk
    geo_risk = self.geopolitical_service.assess_risk()

    # Corporate actions
    corporate_actions = self.corporate_action_service.get_actions(self.ticker)

    # Insider trading
    insider_data = self.insider_trading_service.analyze_patterns(self.ticker)

    return {
        'geopolitical': geo_risk,
        'corporate_actions': corporate_actions,
        'insider_trading': insider_data
    }
```

### **Phase 6: Prediction Generation**

#### **6.1 Multi-Timeframe Predictions**

```python
def generate_predictions(self, days_ahead=5):
    predictions = {}

    # Short-term predictions (1-7 days)
    short_term = self._generate_short_term_predictions(days_ahead)

    # Medium-term predictions (1-4 weeks)
    medium_term = self._generate_medium_term_predictions()

    # Long-term predictions (1-12 months)
    long_term = self._generate_long_term_predictions()

    return {
        'short_term': short_term,
        'medium_term': medium_term,
        'long_term': long_term
    }
```

#### **6.2 Ensemble Prediction**

```python
def generate_ensemble_prediction(self, models, data):
    predictions = []

    for name, model in models.items():
        if hasattr(model, 'predict'):
            pred = model.predict(data)
            predictions.append(pred)

    # Weighted ensemble
    ensemble_pred = np.average(predictions, weights=self.model_weights)

    return ensemble_pred
```

### **Phase 7: Report Generation**

#### **7.1 Comprehensive Report Creation**

```python
def generate_comprehensive_report(self, analysis_results):
    report = {
        'executive_summary': self._create_executive_summary(),
        'technical_analysis': analysis_results['technical'],
        'fundamental_analysis': analysis_results['fundamental'],
        'predictions': analysis_results['predictions'],
        'risk_assessment': analysis_results['risk'],
        'recommendations': self._generate_recommendations()
    }

    return report
```

#### **7.2 Multi-Format Export**

```python
def export_reports(self, report_data):
    # JSON export
    self._export_json(report_data)

    # HTML report
    self._export_html(report_data)

    # CSV predictions
    self._export_csv_predictions(report_data['predictions'])

    # PDF report
    self._export_pdf(report_data)
```

## 🔄 **Data Flow Diagram**

```
User Input (Ticker)
        │
        ▼
┌─────────────────┐
│ System Init     │
│ • Services      │
│ • Config        │
│ • Database      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Data Collection │
│ • Angel One     │
│ • Yahoo Finance │
│ • Database      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Preprocessing   │
│ • Cleaning      │
│ • Features      │
│ • Validation    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Model Training  │
│ • 15+ Models    │
│ • Ensemble      │
│ • Validation    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Analysis        │
│ • Technical     │
│ • Fundamental   │
│ • Risk          │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Predictions     │
│ • Multi-time    │
│ • Confidence    │
│ • Validation    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Reports         │
│ • HTML/PDF      │
│ • CSV/JSON      │
│ • Dashboard     │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Output          │
│ • Files         │
│ • Database      │
│ • Display       │
└─────────────────┘
```

## ⚙️ **Configuration Workflow**

### **Configuration Loading**

```python
# Data Periods Configuration
period_configs = {
    'quick_check': '3mo',
    'recommended': '1y',
    'comprehensive': '2y',
    'angel_one': '6mo',
    'yfinance': '1y'
}

# Database Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '7874',
    'database': 'stock_data'
}

# Angel One Configuration
angel_config = {
    'api_key': 'your_api_key',
    'client_id': 'your_client_id',
    'pin': 'your_pin',
    'totp_secret': 'your_totp_secret'
}
```

## 🚀 **Execution Workflow**

### **Command Line Execution**

```bash
# Basic execution
python run_analysis.py --ticker RELIANCE

# Advanced execution
python run_analysis.py --ticker RELIANCE --period 2y --interval 1d --workers 4

# Interactive mode
python enhanced_unified_pipeline.py
```

### **Python API Execution**

```python
# Initialize pipeline
pipeline = UnifiedAnalysisPipeline("RELIANCE")

# Run complete analysis
success = pipeline.run_complete_analysis()

# Get predictions
predictions = pipeline.generate_and_display_predictions(days_ahead=5)

# Generate reports
reports = pipeline.generate_comprehensive_reports()
```

## 📊 **Output Workflow**

### **Generated Files**

```
data/
├── {ticker}_predictions.csv
├── {ticker}_short_term_predictions.csv
├── {ticker}_mid_term_predictions.csv
├── {ticker}_long_term_predictions.csv
└── {ticker}_comprehensive_report.html

models/
├── short_term/
├── mid_term/
├── long_term/
└── scalers/

reports/
├── {ticker}_analysis_report.json
├── {ticker}_prediction_report.html
└── {ticker}_validation_report.csv
```

### **Database Storage**

#### **Interval-Specific Storage System (NEW)**

The system now uses **interval-specific tables** optimized for different trading strategies:

```python
# Automatic routing to appropriate table based on interval
from src.core.interval_specific_storage import IntervalSpecificStorage

storage = IntervalSpecificStorage()
storage.store_data_by_interval(
    df=dataframe,
    ticker='RELIANCE',
    exchange='NSE',
    interval='FIVE_MINUTE',  # Automatically goes to intraday_5min table
    symbol_token='500325'
)
```

#### **Table Structure:**

| **Interval** | **Table**        | **Use Case**                                         |
| ------------ | ---------------- | ---------------------------------------------------- |
| 1-minute     | `intraday_1min`  | High-frequency trading, scalping                     |
| 5-minute     | `intraday_5min`  | Day trading, swing trading                           |
| 15-minute    | `intraday_15min` | Position trading, trend analysis                     |
| 30-minute    | `intraday_30min` | Trend following, technical analysis                  |
| 1-hour       | `hourly_data`    | Portfolio management, risk assessment                |
| 1-day        | `daily_data`     | Fundamental analysis, long-term investing            |
| Weekly       | `weekly_data`    | Trend analysis, performance metrics (auto-generated) |
| Monthly      | `monthly_data`   | Annual analysis, market cycles (auto-generated)      |

#### **Query Examples:**

```sql
-- High-frequency trading (1-minute data)
SELECT * FROM intraday_1min
WHERE ticker='RELIANCE'
AND datetime >= NOW() - INTERVAL 1 HOUR;

-- Day trading (5-minute data)
SELECT * FROM intraday_5min
WHERE ticker='RELIANCE'
AND DATE(datetime) = CURDATE();

-- Long-term analysis (Daily data)
SELECT * FROM daily_data
WHERE ticker='RELIANCE'
AND date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR);

-- Performance analysis (Weekly aggregates)
SELECT price_change_pct FROM weekly_data
WHERE ticker='RELIANCE'
ORDER BY week_start_date DESC LIMIT 52;
```

#### **Legacy Storage (Still Supported):**

```sql
-- Stock data storage
INSERT INTO stock_data (ticker, date, open, high, low, close, volume, data_source)
VALUES ('RELIANCE', '2025-09-16', 1500.00, 1520.00, 1495.00, 1510.00, 1000000, 'angel_one');

-- Prediction storage
INSERT INTO predictions (ticker, prediction_date, predicted_price, confidence, model_name)
VALUES ('RELIANCE', '2025-09-17', 1525.50, 0.85, 'ensemble');
```

## 🔧 **Error Handling Workflow**

### **Error Recovery Mechanisms**

```python
def handle_analysis_error(self, context, error):
    # Log error
    self.logger.error(f"Error in {context}: {error}")

    # Attempt recovery
    if context == "data_loading":
        return self._fallback_data_loading()
    elif context == "model_training":
        return self._fallback_model_training()
    elif context == "prediction":
        return self._fallback_prediction()

    # Return safe defaults
    return self._get_safe_defaults()
```

## 📈 **Performance Optimization Workflow**

### **Parallel Processing**

```python
# Multi-threaded data processing
with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    futures = []

    # Submit tasks
    for task in tasks:
        future = executor.submit(task)
        futures.append(future)

    # Collect results
    results = [future.result() for future in futures]
```

### **Caching Strategy**

```python
# Data caching
if cache_key in self.data_cache:
    return self.data_cache[cache_key]

# Model caching
if model_file.exists():
    model = joblib.load(model_file)
    return model
```

## 🎯 **Quality Assurance Workflow**

### **Data Validation**

```python
def validate_data_quality(self, data):
    # Check data completeness
    completeness = (1 - data.isnull().sum().sum() / data.size) * 100

    # Check data consistency
    consistency = self._check_data_consistency(data)

    # Check data freshness
    freshness = self._check_data_freshness(data)

    return {
        'completeness': completeness,
        'consistency': consistency,
        'freshness': freshness
    }
```

### **Model Validation**

```python
def validate_model_performance(self, model, test_data):
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)

    # Prediction accuracy
    predictions = model.predict(test_data)
    accuracy = mean_absolute_error(y_true, predictions)

    return {
        'cv_scores': cv_scores,
        'accuracy': accuracy,
        'confidence': np.mean(cv_scores)
    }
```

---

This comprehensive workflow documentation provides a complete understanding of how the AI Stock Predictor system operates from initialization to final output generation. Each phase is designed to be modular, scalable, and robust, ensuring reliable stock market analysis and prediction capabilities.
