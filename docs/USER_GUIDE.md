# AI Stock Predictor - User Guide

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- MySQL database (for data storage)
- Angel One API credentials (for Indian stocks)
- Internet connection

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-stock-predictor

# Install dependencies
pip install -r requirements.txt

# Set up database
python scripts/database/setup_mysql_database.py

# Migrate existing data to database
python migrate_to_database.py --preset local
```

## 📊 System Overview

The AI Stock Predictor is a comprehensive, enterprise-grade system that provides:

- **Multi-source data integration** (Angel One for Indian stocks, Yahoo Finance for US stocks)
- **Advanced machine learning predictions** (15+ algorithms with ensemble methods)
- **Multi-timeframe analysis** (short-term, medium-term, long-term)
- **Database storage** for efficient data management and persistence
- **Real-time data updates** and incremental learning
- **Multi-exchange data fusion** (BSE/NSE integration for enhanced predictions)
- **Comprehensive reporting** (HTML, PDF, CSV, JSON formats)
- **Advanced validation** and accuracy tracking
- **Phase-based analysis** (Enhanced fundamental, economic, and risk analysis)

## 🎯 Core Features

### 1. Data Sources & Integration

- **Indian Stocks**: Angel One SmartAPI (NSE/BSE) with enhanced master data integration
- **US Stocks**: Yahoo Finance with comprehensive historical data
- **Economic Data**: FRED API, World Bank API for macroeconomic indicators
- **Multi-Exchange Fusion**: BSE/NSE data fusion for enhanced prediction accuracy
- **Real-time Data**: Live price feeds and market data updates
- **Database Integration**: MySQL storage with automatic data freshness checks

### 2. Advanced Prediction Algorithms

- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
- **Non-linear Models**: SVR, MLP Regressor, Gaussian Process
- **Advanced Models**: AdaBoost, Extra Trees, Huber Regressor, Kernel Ridge
- **Ensemble Creation**: Intelligent model combination with confidence scoring
- **Multi-timeframe Training**: Specialized models for different prediction horizons

### 3. Analysis Timeframes & Phases

- **Short-term**: 1-7 days (intraday patterns, technical indicators)
- **Medium-term**: 1-4 weeks (trend analysis, momentum indicators)
- **Long-term**: 1-12 months (fundamental analysis, strategic planning)
- **Phase 1**: Enhanced fundamental analysis with market sentiment
- **Phase 2**: Economic data integration and regulatory monitoring
- **Phase 3**: Geopolitical risk assessment and corporate actions analysis

### 4. Database & Storage Features

- **Interval-Specific Storage**: Purpose-built tables for different trading strategies
- **MySQL Integration**: High-performance database storage with optimized schemas
- **Automatic Data Routing**: Smart routing to appropriate tables based on interval
- **Data Quality Tracking**: Comprehensive quality metrics and monitoring
- **Incremental Updates**: Smart data merging and change detection
- **Multi-source Support**: Unified storage for different data sources
- **Performance Optimization**: Indexed queries and batch operations
- **Automatic Aggregation**: Daily data creates weekly/monthly aggregates

## 🛠️ How to Use the System

### Method 1: Command Line Interface

#### Basic Analysis

```bash
# Run analysis for a single stock
python run_analysis.py --ticker RELIANCE --period 1y

# Run with custom parameters
python run_analysis.py --ticker AAPL --period 2y --interval 1d --workers 4

# Run main unified pipeline
python main/unified_analysis_pipeline.py
```

#### Interactive Data Selection

```bash
# Use interactive data selector with custom parameters
python enhanced_unified_pipeline.py

# Interactive mode with data period selection
python enhanced_unified_pipeline.py --interactive
```

#### Custom Analysis

```bash
# Run custom analysis with specific configuration
python run_custom_analysis.py

# Notebook interface for interactive analysis
python notebook_interface.py
```

#### Interval-Specific Storage (NEW)

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

**Available Tables:**

- `intraday_1min` - High-frequency trading, scalping
- `intraday_5min` - Day trading, swing trading
- `intraday_15min` - Position trading, trend analysis
- `intraday_30min` - Trend following, technical analysis
- `hourly_data` - Portfolio management, risk assessment
- `daily_data` - Fundamental analysis, long-term investing
- `weekly_data` - Trend analysis, performance metrics (auto-generated)
- `monthly_data` - Annual analysis, market cycles (auto-generated)

#### Database Operations

```bash
# Test database connectivity
python scripts/database/test_database_connection.py

# Migrate data to database
python migrate_to_database.py --preset local

# Test Angel One integration
python test_angel_database_integration.py
```

### Method 2: Python Script Integration

#### Basic Usage

```python
from main.pipeline.core_pipeline import UnifiedAnalysisPipeline

# Initialize pipeline with database integration
pipeline = UnifiedAnalysisPipeline(
    ticker="RELIANCE",
    max_workers=4,
    period_config="recommended"
)

# Run complete analysis with all phases
success = pipeline.run_unified_analysis(period="1y", days_ahead=5, use_enhanced=True)

# Get multi-timeframe predictions
predictions = pipeline.generate_and_display_predictions(days_ahead=5)

# Access individual phase results
if success:
    print("✅ Analysis completed successfully")
    print(f"📊 Predictions generated: {len(predictions)} timeframes")
```

#### Advanced Usage with Multi-Exchange Data

```python
from src.core.data_service import DataService
from src.core.model_service import ModelService
from src.core.multi_exchange_data_service import MultiExchangeDataService

# Initialize services with database integration
data_service = DataService(use_database=True)
model_service = ModelService()
fusion_service = MultiExchangeDataService(use_database=True)

# Load multi-exchange data (BSE/NSE fusion)
fused_data = fusion_service.load_multi_exchange_data("RELIANCE", period="1y")

# Get fusion quality metrics
metrics = fusion_service.get_fusion_quality_metrics(fused_data)
print(f"📊 Fusion Coverage: {metrics.get('fusion_coverage', 0):.1f}%")
print(f"📊 Arbitrage Opportunities: {metrics.get('arbitrage_opportunities', 0)}")

# Preprocess enhanced data
processed_df = data_service.preprocess_data(fused_data)

# Train models with enhanced features
models, scalers, feature_importance = model_service.train_models(processed_df)

# Generate predictions with confidence scores
predictions = model_service.generate_predictions(models, scalers, processed_df, days_ahead=5)
```

#### Phase-Based Analysis

```python
# Run individual analysis phases
pipeline = UnifiedAnalysisPipeline("RELIANCE")

# Phase 1: Enhanced fundamental analysis
phase1_success = pipeline.run_phase1_enhanced_analysis()

# Phase 2: Economic data integration
phase2_success = pipeline.run_phase2_economic_analysis()

# Phase 3: Risk assessment
phase3_success = pipeline.run_phase3_risk_analysis()

# Generate comprehensive reports
reports = pipeline.generate_comprehensive_reports()
```

### Method 3: Jupyter Notebook Interface

```python
# Use notebook interface for interactive analysis
exec(open('notebook_interface.py').read())

# Or import the notebook class directly
from notebook_interface import StockAnalysisNotebook

# Create interactive analysis session
notebook = StockAnalysisNotebook()
notebook.run_interactive_analysis("RELIANCE")
```

## 📈 Data Management

### Database Configuration

#### Setup MySQL Database

```bash
# Create database and tables
python scripts/database/setup_mysql_database.py

# Migrate existing data
python migrate_to_database.py --preset local

# Test database connectivity
python scripts/database/test_database_connection.py
```

#### Enhanced Database Schema

- **stock_data**: Main stock price data with multi-source support
- **angel_one_stock_data**: Angel One specific data with enhanced metadata
- **angel_one_oi_data**: Open Interest data for derivatives
- **data_quality_metrics**: Comprehensive data quality tracking
- **prediction_accuracy**: Model performance tracking and validation
- **multi_exchange_data**: BSE/NSE fusion data storage
- **arbitrage_opportunities**: Cross-exchange arbitrage tracking
- **model_metadata**: Model training and performance metadata

### Data Sources Configuration

#### Angel One API Setup (Enhanced)

```python
# Configure Angel One credentials with enhanced features
from config.angel_one_config import AngelOneConfig

config = AngelOneConfig()
config.api_key = "your_api_key"
config.client_id = "your_client_id"
config.pin = "your_pin"
config.totp_secret = "your_totp_secret"

# Enhanced features
config.enable_master_data = True  # Use Angel Broking master data
config.max_days_limits = {
    'ONE_MINUTE': 30,
    'ONE_DAY': 2000,
    'ONE_HOUR': 400
}
config.interval_mapping = {
    '1d': 'ONE_DAY',
    '1h': 'ONE_HOUR',
    '1m': 'ONE_MINUTE'
}
```

#### Database Configuration (Enhanced)

```python
# Configure database connection with multiple presets
from config.database_config import get_database_config

# Local MySQL setup
db_config = get_database_config('local')
# Uses: host="localhost", user="root", password="7874", database="stock_data"

# Production setup
db_config = get_database_config('production')
# Uses: host="your-server", user="your-user", password="your-password", database="stock_data"

# Test setup
db_config = get_database_config('test')
# Uses: SQLite for testing
```

#### Multi-Exchange Configuration

```python
# Configure multi-exchange data fusion
from src.core.multi_exchange_data_service import MultiExchangeDataService

fusion_service = MultiExchangeDataService(use_database=True)

# Fusion parameters
fusion_service.fusion_methods = {
    'price': 'volume_weighted',  # volume_weighted, liquidity_weighted, vwap
    'volume': 'sum',            # sum, max, average
    'technical': 'average'      # average, weighted_average, best_signal
}

# Arbitrage detection
fusion_service.arbitrage_threshold = 0.5  # 0.5% price difference threshold
```

## 🔧 System Configuration

### Performance Optimization

#### Worker Configuration

```python
# Adjust based on your system
pipeline = UnifiedAnalysisPipeline(
    ticker="RELIANCE",
    max_workers=4,  # Adjust based on CPU cores
    period_config="recommended"
)
```

#### Memory Management

```python
# For large datasets
data_service = DataService(
    cache_dir="data/cache",
    use_database=True,  # Use database for large datasets
    period_config="comprehensive"
)
```

### Data Periods Configuration

#### Available Periods

- **quick_check**: 3 months (fast analysis)
- **recommended**: 1 year (balanced)
- **comprehensive**: 2 years (thorough analysis)
- **angel_one**: 6 months (optimized for Angel One)
- **yfinance**: 1 year (optimized for Yahoo Finance)

#### Custom Periods

```python
# Custom date range
df = data_service.load_stock_data_custom_dates(
    ticker="RELIANCE",
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

## 📊 Analysis Types

### 1. Short-term Analysis (1-7 days)

```python
# Focus on intraday and daily patterns
pipeline = UnifiedAnalysisPipeline(ticker="RELIANCE")
pipeline.run_short_term_analysis()
```

### 2. Medium-term Analysis (1-4 weeks)

```python
# Focus on weekly patterns and trends
pipeline.run_medium_term_analysis()
```

### 3. Long-term Analysis (1-12 months)

```python
# Focus on monthly and quarterly trends
pipeline.run_long_term_analysis()
```

### 4. Comprehensive Analysis

```python
# Run all analysis types
pipeline.run_complete_analysis()
```

## 🎯 Prediction Usage

### Basic Predictions

```python
# Get predictions for next 5 days
predictions = pipeline.generate_and_display_predictions(days_ahead=5)

# Access prediction data
for day, prediction in predictions.items():
    print(f"Day {day}: {prediction['price']:.2f} ({prediction['change']:+.2f}%)")
```

### Advanced Predictions

```python
# Get multi-timeframe predictions
predictions = pipeline.generate_multi_timeframe_predictions()

# Access different timeframes
short_term = predictions['short_term']  # 1-7 days
medium_term = predictions['medium_term']  # 1-4 weeks
long_term = predictions['long_term']  # 1-12 months
```

### Prediction Validation

```python
# Validate prediction accuracy
from scripts.validation.validation_dashboard import ValidationDashboard

validator = ValidationDashboard()
accuracy_report = validator.generate_accuracy_report("RELIANCE")
```

## 🔍 Monitoring and Validation

### Data Quality Monitoring

```python
from src.core.enhanced_angel_one_service import EnhancedAngelOneService

service = EnhancedAngelOneService()
quality_metrics = service.get_data_quality_metrics("RELIANCE", "NSE")

print(f"Data Completeness: {quality_metrics['data_completeness']:.2f}%")
print(f"Quality Score: {quality_metrics['data_quality_score']:.2f}")
```

### Model Performance Tracking

```python
# Check model accuracy
from scripts.validation.prediction_validator import PredictionValidator

validator = PredictionValidator()
accuracy = validator.validate_predictions("RELIANCE", days_back=30)
```

### System Health Check

```python
# Run system diagnostics
python scripts/testing/system_health_check.py
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Angel One Authentication Failed

```bash
# Check credentials
python -c "from src.utils.angel_one_data_downloader import AngelOneDataDownloader; AngelOneDataDownloader().authenticate()"
```

#### 2. Database Connection Issues

```bash
# Test database connection
python -c "from src.core.database_service import DatabaseService; DatabaseService('mysql', 'mysql://root:7874@localhost/stock_data')"
```

#### 3. Memory Issues

```python
# Reduce data period
data_service = DataService(period_config="quick_check")

# Use database instead of cache
data_service = DataService(use_database=True)
```

#### 4. Model Training Failures

```python
# Check data quality
df = data_service.load_stock_data("RELIANCE")
print(f"Data shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
```

### Performance Optimization

#### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_ticker_date ON stock_data(ticker, date);
CREATE INDEX idx_ticker ON stock_data(ticker);
```

#### 2. Cache Management

```python
# Clear cache if needed
import shutil
shutil.rmtree("data/cache")
```

#### 3. Worker Optimization

```python
# Adjust workers based on system
import os
max_workers = min(4, os.cpu_count())
```

## 📚 Advanced Features

### 1. Custom Model Training

```python
from src.core.model_service import ModelService

model_service = ModelService()
custom_models = model_service.train_custom_models(
    data=processed_df,
    algorithms=['RandomForest', 'XGBoost', 'LightGBM']
)
```

### 2. Batch Processing

```python
# Process multiple stocks
tickers = ["RELIANCE", "TCS", "INFY", "HDFC"]
results = {}

for ticker in tickers:
    pipeline = UnifiedAnalysisPipeline(ticker=ticker)
    results[ticker] = pipeline.run_complete_analysis()
```

### 3. Real-time Updates

```python
# Enable incremental updates
from src.core.incremental_data_service import IncrementalDataService

incremental_service = IncrementalDataService(use_database=True)
incremental_service.update_stock_data("RELIANCE")
```

### 4. Multi-currency Support

```python
# Generate reports in different currencies
from src.integrations.comprehensive_report_integration import ComprehensiveReportIntegration

report_service = ComprehensiveReportIntegration()
multi_currency_report = report_service.generate_multi_currency_report(
    ticker="RELIANCE",
    currencies=["INR", "USD", "EUR"]
)
```

## 📋 Best Practices

### 1. Data Management

- Use database storage for large datasets
- Regularly clean cache files
- Monitor data quality metrics
- Use appropriate data periods

### 2. Performance

- Adjust worker count based on system resources
- Use incremental updates for real-time data
- Monitor memory usage
- Optimize database queries

### 3. Predictions

- Validate predictions regularly
- Use multiple timeframes for comprehensive analysis
- Monitor model performance
- Update models periodically

### 4. System Maintenance

- Regular database backups
- Monitor system logs
- Update dependencies
- Test system health regularly

## 🆘 Support and Resources

### Documentation

- `docs/` - Complete documentation
- `ANGEL_ONE_OPTIMIZATION_SUMMARY.md` - Angel One integration details
- `ORGANIZATION_GUIDE.md` - Project structure guide

### Scripts

- `scripts/validation/` - Validation and testing scripts
- `scripts/database/` - Database management scripts
- `scripts/testing/` - System testing scripts

### Configuration

- `config/` - All configuration files
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (create as needed)

### Logs and Monitoring

- Check console output for real-time status
- Monitor database logs for data issues
- Use validation scripts for accuracy checks

## 🎯 Quick Reference

### Essential Commands

```bash
# Run analysis
python run_analysis.py --ticker RELIANCE

# Interactive mode
python enhanced_unified_pipeline.py

# Database setup
python scripts/database/setup_mysql_database.py

# System test
python test_angel_database_integration.py
```

### Key Files

- `main/unified_analysis_pipeline.py` - Main analysis pipeline
- `src/core/data_service.py` - Data management
- `src/core/model_service.py` - Model training
- `config/database_config.py` - Database configuration
- `config/angel_one_config.py` - Angel One configuration

### Important Classes

- `UnifiedAnalysisPipeline` - Main analysis class
- `DataService` - Data loading and preprocessing
- `ModelService` - Model training and prediction
- `DatabaseService` - Database operations
- `AngelOneDataDownloader` - Angel One API integration

---

**Note**: This system is designed for educational and research purposes. Always verify predictions with multiple sources and consider market risks before making investment decisions.
