# AI Stock Predictor - Quick Reference

## 🚀 Essential Commands

### Basic Analysis

```bash
# Single stock analysis
python run_analysis.py --ticker RELIANCE

# Interactive data selection
python enhanced_unified_pipeline.py

# Custom analysis
python run_custom_analysis.py
```

### Database Operations

```bash
# Setup database
python scripts/database/setup_mysql_database.py

# Migrate data
python migrate_to_database.py --preset local

# Test database integration
python test_angel_database_integration.py
```

### Validation & Testing

```bash
# Quick validation
python scripts/validation/quick_validation.py

# Comprehensive validation
python scripts/validation/validation_dashboard.py

# System health check
python scripts/testing/system_health_check.py
```

## 📊 Supported Tickers

### Indian Stocks (Angel One)

- **NSE**: RELIANCE, TCS, INFY, HDFC, HDFCBANK, ICICIBANK, KOTAKBANK, BHARTIARTL
- **BSE**: Add .BO suffix (e.g., RELIANCE.BO)

### US Stocks (Yahoo Finance)

- **NYSE/NASDAQ**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NFLX, NVDA

## ⚙️ Configuration

### Database Settings

```python
# Local MySQL (default)
host="localhost"
user="root"
password="7874"
database="stock_data"
```

### Angel One API

```python
# Required credentials
api_key="your_api_key"
client_id="your_client_id"
pin="your_pin"
totp_secret="your_totp_secret"
```

## 🎯 Analysis Types

| Type          | Period   | Use Case          |
| ------------- | -------- | ----------------- |
| Quick Check   | 3 months | Fast analysis     |
| Recommended   | 1 year   | Balanced analysis |
| Comprehensive | 2 years  | Thorough analysis |

## 📈 Prediction Timeframes

| Timeframe   | Duration    | Algorithms          |
| ----------- | ----------- | ------------------- |
| Short-term  | 1-7 days    | All 15+ algorithms  |
| Medium-term | 1-4 weeks   | Ensemble methods    |
| Long-term   | 1-12 months | Linear + Non-linear |

## 🔧 Performance Tuning

### Worker Configuration

```python
# CPU cores based
max_workers = min(4, os.cpu_count())

# Memory based
use_database = True  # For large datasets
```

### Data Periods

```python
# Fast analysis
period_config="quick_check"

# Balanced
period_config="recommended"

# Thorough
period_config="comprehensive"
```

## 🚨 Troubleshooting

### Common Issues

| Issue                     | Solution                 |
| ------------------------- | ------------------------ |
| Angel One auth failed     | Check credentials & TOTP |
| Database connection error | Verify MySQL is running  |
| Memory issues             | Use database storage     |
| Model training failed     | Check data quality       |

### Quick Fixes

```bash
# Clear cache
rm -rf data/cache/*

# Restart database
sudo systemctl restart mysql

# Check system health
python scripts/testing/system_health_check.py
```

## 📋 File Structure

```
ai-stock-predictor/
├── main/                    # Main pipeline
├── src/                     # Source code
│   ├── core/               # Core services
│   ├── analysis/           # Analysis modules
│   ├── integrations/       # Phase integrations
│   └── utils/              # Utilities
├── config/                 # Configuration
├── scripts/                # Helper scripts
├── docs/                   # Documentation
└── data/                   # Data storage
```

## 🎯 Key Classes

| Class                     | Purpose                      |
| ------------------------- | ---------------------------- |
| `UnifiedAnalysisPipeline` | Main analysis orchestrator   |
| `DataService`             | Data loading & preprocessing |
| `ModelService`            | Model training & prediction  |
| `DatabaseService`         | Database operations          |
| `AngelOneDataDownloader`  | Angel One API integration    |

## 📊 Output Files

### Predictions

- `{ticker}_predictions.csv` - Main predictions
- `{ticker}_short_term_predictions.csv` - 1-7 days
- `{ticker}_mid_term_predictions.csv` - 1-4 weeks
- `{ticker}_long_term_predictions.csv` - 1-12 months

### Reports

- `{ticker}_comprehensive_report.html` - Full analysis
- `{ticker}_validation_report.html` - Accuracy validation
- `{ticker}_data_quality_report.html` - Data quality metrics

## 🔍 Monitoring

### Data Quality Metrics

- **Completeness**: % of non-missing data
- **Quality Score**: Overall data quality (0-100)
- **Volatility**: Price and volume volatility
- **Coverage**: Date range coverage

### Model Performance

- **Accuracy**: Prediction accuracy percentage
- **MAPE**: Mean Absolute Percentage Error
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error

## 🆘 Emergency Commands

### Reset System

```bash
# Clear all cache
rm -rf data/cache/* models/cache/*

# Reset database
python scripts/database/reset_database.py

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Debug Mode

```bash
# Verbose logging
python run_analysis.py --ticker RELIANCE --verbose

# Debug database
python -c "from src.core.database_service import DatabaseService; print(DatabaseService('mysql', 'mysql://root:7874@localhost/stock_data'))"
```

---

**💡 Pro Tip**: Always start with `python enhanced_unified_pipeline.py` for the best user experience with interactive data selection and comprehensive analysis options.
