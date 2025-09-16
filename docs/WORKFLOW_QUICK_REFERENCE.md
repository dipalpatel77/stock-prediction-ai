# AI Stock Predictor - Workflow Quick Reference

## 🚀 **Quick Start Workflow**

### **1. Basic Execution**

```bash
# Run analysis for a stock
python main/unified_analysis_pipeline.py

# Or use the enhanced pipeline
python enhanced_unified_pipeline.py
```

### **2. Programmatic Usage**

```python
from main.unified_analysis_pipeline import UnifiedAnalysisPipeline

# Initialize and run
pipeline = UnifiedAnalysisPipeline("RELIANCE")
success = pipeline.run_unified_analysis(period="1y", days_ahead=5)
```

## 📊 **Complete Workflow Steps**

### **Phase 1: System Initialization**

```
✅ Ticker Format Fixing
✅ Service Initialization
✅ Configuration Loading
✅ Database Connection
```

### **Phase 2: Data Collection (PartA)**

```
✅ Data Source Selection (Angel One vs Yahoo Finance)
✅ Data Download & Authentication
✅ Data Validation & Quality Check
✅ Database Storage
```

### **Phase 3: Model Training (PartB)**

```
✅ Data Preprocessing & Feature Engineering
✅ 15+ ML Model Training (Parallel)
✅ Ensemble Model Creation
✅ Model Evaluation & Persistence
```

### **Phase 4: Strategy Analysis (PartC)**

```
✅ Technical Analysis (RSI, MACD, Bollinger)
✅ Fundamental Analysis (P/E, P/B, Ratios)
✅ Risk Assessment (VaR, Volatility, Sharpe)
✅ Trading Signal Generation
```

### **Phase 5: Enhanced Analysis**

```
✅ Phase 1: Enhanced Fundamental Analysis
✅ Phase 2: Economic Data & Regulatory Monitoring
✅ Phase 3: Geopolitical Risk & Corporate Actions
```

### **Phase 6: Prediction Generation**

```
✅ Short-term Predictions (1-7 days)
✅ Medium-term Predictions (1-4 weeks)
✅ Long-term Predictions (1-12 months)
✅ Confidence Analysis & Validation
```

### **Phase 7: Report Generation**

```
✅ Comprehensive Report Creation
✅ Multi-format Export (HTML, PDF, CSV, JSON)
✅ Interactive Dashboard
✅ Database Storage
```

## 🔧 **Key Configuration Options**

### **Data Periods**

```python
period_configs = {
    'quick_check': '3mo',      # Quick analysis
    'recommended': '1y',       # Standard analysis
    'comprehensive': '2y',     # Detailed analysis
    'angel_one': '6mo',        # Angel One optimized
    'yfinance': '1y'           # Yahoo Finance optimized
}
```

### **Prediction Horizons**

```python
prediction_horizons = {
    'short_term': [1, 2, 3, 4, 5, 6, 7],           # Days
    'medium_term': [7, 14, 21, 28],                 # Weeks
    'long_term': [30, 90, 180, 365]                 # Months
}
```

### **ML Algorithms**

```python
algorithms = [
    'RandomForestRegressor',    # Ensemble
    'GradientBoostingRegressor', # Boosting
    'XGBRegressor',            # Extreme Gradient Boosting
    'LGBMRegressor',           # Light Gradient Boosting
    'CatBoostRegressor',       # Categorical Boosting
    'LinearRegression',        # Linear
    'Ridge', 'Lasso', 'ElasticNet', # Regularized Linear
    'SVR',                     # Support Vector Regression
    'MLPRegressor',            # Neural Network
    'GaussianProcessRegressor', # Gaussian Process
    'AdaBoostRegressor',       # Adaptive Boosting
    'ExtraTreesRegressor',     # Extra Trees
    'HuberRegressor',          # Robust Regression
    'KernelRidge'              # Kernel Ridge
]
```

## 📈 **Data Sources & Selection Logic**

### **Indian Stocks (Angel One)**

```python
if self._is_indian_stock(ticker):
    # Use Angel One API
    data = self._download_from_angel_one(ticker, period, interval)
    # Features: NSE/BSE data, F&O data, real-time prices
```

### **US Stocks (Yahoo Finance)**

```python
else:
    # Use Yahoo Finance
    data = self._download_from_yahoo(ticker, period, interval)
    # Features: Multiple exchanges, historical data, dividends
```

## 🗄️ **Database Integration**

### **Storage Strategy**

```python
# Automatic database storage
if self.use_database:
    success = self.db_service.store_stock_data(ticker, data, data_source)

# Data retrieval with freshness check
db_data = self.db_service.get_stock_data(ticker)
if days_old <= 1:  # Data is fresh
    return db_data
```

### **Database Schema**

```sql
-- Stock data table
CREATE TABLE stock_data (
    id INT PRIMARY KEY AUTO_INCREMENT,
    ticker VARCHAR(20),
    date DATE,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    data_source VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 📊 **Output Files Generated**

### **Prediction Files**

```
data/
├── {ticker}_predictions.csv              # Main predictions
├── {ticker}_short_term_predictions.csv   # 1-7 days
├── {ticker}_mid_term_predictions.csv     # 1-4 weeks
├── {ticker}_long_term_predictions.csv    # 1-12 months
└── {ticker}_comprehensive_report.html    # Full report
```

### **Model Files**

```
models/
├── short_term/                           # Short-term models
├── mid_term/                            # Medium-term models
├── long_term/                           # Long-term models
├── scalers/                             # Data scalers
└── cache/                               # Model cache
```

### **Report Files**

```
reports/
├── {ticker}_analysis_report.json        # JSON report
├── {ticker}_prediction_report.html      # HTML report
├── {ticker}_validation_report.csv       # Validation results
└── {ticker}_dashboard.html              # Interactive dashboard
```

## ⚡ **Performance Optimization**

### **Parallel Processing**

```python
# Multi-threaded execution
with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    futures = [executor.submit(task) for task in tasks]
    results = [future.result() for future in futures]
```

### **Caching Strategy**

```python
# Data caching
if cache_key in self.data_cache:
    return self.data_cache[cache_key]

# Model caching
if model_file.exists():
    return joblib.load(model_file)
```

### **Database Optimization**

```python
# Batch operations
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    self.db_service.batch_insert(batch)
```

## 🔍 **Error Handling & Recovery**

### **Data Errors**

```python
try:
    data = self.load_stock_data(ticker)
except DataError:
    # Fallback to alternative source
    data = self._fallback_data_loading(ticker)
```

### **Model Errors**

```python
try:
    model = self.train_model(algorithm, data)
except ModelError:
    # Skip failed model, continue with others
    self.logger.warning(f"Model {algorithm} failed, continuing...")
```

### **System Errors**

```python
try:
    result = self.run_analysis()
except SystemError:
    # Restart service and retry
    self._restart_service()
    result = self.run_analysis()
```

## 📋 **Quality Assurance**

### **Data Validation**

```python
def validate_data_quality(self, data):
    return {
        'completeness': (1 - data.isnull().sum().sum() / data.size) * 100,
        'consistency': self._check_data_consistency(data),
        'freshness': self._check_data_freshness(data)
    }
```

### **Model Validation**

```python
def validate_model_performance(self, model, test_data):
    cv_scores = cross_val_score(model, X, y, cv=5)
    return {
        'accuracy': np.mean(cv_scores),
        'confidence': np.std(cv_scores)
    }
```

## 🎯 **Best Practices**

### **1. Data Management**

- ✅ Use database for persistent storage
- ✅ Implement data freshness checks
- ✅ Validate data quality before processing
- ✅ Use appropriate data periods for analysis

### **2. Model Training**

- ✅ Train multiple algorithms in parallel
- ✅ Use cross-validation for model evaluation
- ✅ Implement ensemble methods for better accuracy
- ✅ Cache trained models for reuse

### **3. Error Handling**

- ✅ Implement graceful error recovery
- ✅ Use fallback strategies for failed operations
- ✅ Log all errors for debugging
- ✅ Continue processing despite individual failures

### **4. Performance**

- ✅ Use parallel processing where possible
- ✅ Implement caching for frequently accessed data
- ✅ Optimize database queries
- ✅ Monitor resource usage

## 🚀 **Quick Commands**

### **Run Analysis**

```bash
# Basic analysis
python main/unified_analysis_pipeline.py

# Enhanced analysis with custom parameters
python enhanced_unified_pipeline.py --ticker RELIANCE --period 2y --workers 8

# Interactive mode
python run_analysis.py --interactive
```

### **Database Operations**

```bash
# Migrate data to database
python scripts/database/migrate_to_database.py --preset local

# Test database connection
python scripts/database/test_database_connection.py
```

### **Validation**

```bash
# Quick validation
python scripts/validation/quick_validation.py --ticker RELIANCE

# Comprehensive validation
python scripts/validation/validation_dashboard.py --ticker RELIANCE
```

---

This quick reference provides essential information for efficiently using the AI Stock Predictor system workflow. For detailed information, refer to the complete documentation files.
