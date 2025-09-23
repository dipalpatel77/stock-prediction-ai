# AI Stock Predictor - Documentation

Welcome to the AI Stock Predictor documentation! This comprehensive system provides advanced stock market analysis and prediction capabilities using machine learning algorithms and real-time data integration.

## 📚 Documentation Overview

### 🚀 Getting Started

- **[User Guide](USER_GUIDE.md)** - Complete guide on how to use the system efficiently
- **[Quick Reference](QUICK_REFERENCE.md)** - Essential commands and quick access information
- **[System Architecture](SYSTEM_ARCHITECTURE.md)** - Technical architecture and component details

### 🔧 Technical Documentation

- **[Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)** - Common issues and solutions
- **[Database Implementation Guide](DATABASE_IMPLEMENTATION_GUIDE.md)** - Database setup and management
- **[Interval-Specific Storage Guide](INTERVAL_SPECIFIC_STORAGE_GUIDE.md)** - Advanced storage system for different trading strategies
- **[Angel One Optimization Summary](../ANGEL_ONE_OPTIMIZATION_SUMMARY.md)** - Angel One API integration details

### 📊 System Features

#### 🎯 Core Capabilities

- **Multi-source Data Integration**: Angel One (Indian stocks) + Yahoo Finance (US stocks)
- **Advanced ML Algorithms**: 15+ machine learning models for predictions
- **Multi-timeframe Analysis**: Short-term (1-7 days), Medium-term (1-4 weeks), Long-term (1-12 months)
- **Database Storage**: MySQL integration with interval-specific tables for optimized performance
- **Real-time Updates**: Incremental data updates and smart caching

#### 🔬 Analysis Types

- **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
- **Fundamental Analysis**: Economic indicators, market sentiment
- **Risk Assessment**: Volatility analysis, correlation studies
- **Prediction Models**: Ensemble methods, linear/non-linear models

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-stock-predictor

# Install dependencies
pip install -r requirements.txt

# Setup database
python scripts/database/setup_mysql_database.py
```

### 2. Basic Usage

```bash
# Run analysis for Indian stock
python run_analysis.py --ticker RELIANCE

# Run analysis for US stock
python run_analysis.py --ticker AAPL

# Interactive mode
python enhanced_unified_pipeline.py
```

### 3. Python API

```python
from main.unified_analysis_pipeline import UnifiedAnalysisPipeline

# Initialize and run analysis
pipeline = UnifiedAnalysisPipeline(ticker="RELIANCE")
pipeline.run_complete_analysis()

# Get predictions
predictions = pipeline.generate_and_display_predictions(days_ahead=5)
```

## 📈 Supported Markets

### 🇮🇳 Indian Markets (Angel One API)

- **NSE**: RELIANCE, TCS, INFY, HDFC, HDFCBANK, ICICIBANK, KOTAKBANK, BHARTIARTL
- **BSE**: Add .BO suffix (e.g., RELIANCE.BO)
- **Features**: Real-time data, F&O data, Open Interest, Corporate actions

### 🇺🇸 US Markets (Yahoo Finance)

- **NYSE/NASDAQ**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NFLX, NVDA
- **Features**: Historical data, multiple intervals, real-time prices

## 🛠️ System Components

### Core Services

- **DataService**: Data loading, preprocessing, and validation
- **ModelService**: Machine learning model training and prediction
- **DatabaseService**: Database operations and optimization
- **StrategyService**: Trading strategies and signal generation
- **ReportingService**: Report generation and visualization

### Integration Services

- **AngelOneDataDownloader**: Angel One SmartAPI integration
- **YahooFinanceDownloader**: Yahoo Finance data integration
- **FREDAPIService**: Economic data integration
- **CurrencyService**: Multi-currency support

### Analysis Modules

- **ShortTermAnalyzer**: 1-7 day analysis
- **MidTermAnalyzer**: 1-4 week analysis
- **LongTermAnalyzer**: 1-12 month analysis
- **EnhancedPriceForecaster**: Advanced prediction algorithms

## 📊 Output and Reports

### Prediction Files

- `{ticker}_predictions.csv` - Main predictions
- `{ticker}_short_term_predictions.csv` - Short-term forecasts
- `{ticker}_mid_term_predictions.csv` - Medium-term forecasts
- `{ticker}_long_term_predictions.csv` - Long-term forecasts

### Analysis Reports

- `{ticker}_comprehensive_report.html` - Complete analysis report
- `{ticker}_validation_report.html` - Prediction accuracy validation
- `{ticker}_data_quality_report.html` - Data quality metrics

### Model Files

- `models/short_term/` - Short-term prediction models
- `models/mid_term/` - Medium-term prediction models
- `models/long_term/` - Long-term prediction models
- `models/scalers/` - Data scaling objects

## 🔧 Configuration

### Database Configuration

```python
# Local MySQL setup
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '7874',
    'database': 'stock_data'
}
```

### Angel One API Configuration

```python
# Angel One credentials
ANGEL_ONE_CONFIG = {
    'api_key': 'your_api_key',
    'client_id': 'your_client_id',
    'pin': 'your_pin',
    'totp_secret': 'your_totp_secret'
}
```

## 📋 Performance Optimization

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for data and models
- **Network**: Stable internet for real-time data

### Optimization Tips

- Use database storage for large datasets
- Adjust worker count based on CPU cores
- Enable caching for frequently accessed data
- Use appropriate data periods for analysis

## 🚨 Troubleshooting

### Common Issues

1. **Angel One Authentication**: Check credentials and TOTP setup
2. **Database Connection**: Verify MySQL is running and accessible
3. **Memory Issues**: Use database storage and reduce data periods
4. **Import Errors**: Check Python path and dependencies

### Getting Help

- Check the [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)
- Run system health checks: `python scripts/testing/system_health_check.py`
- Validate data quality: `python scripts/validation/quick_validation.py`

## 📚 Additional Resources

### Scripts and Tools

- `scripts/validation/` - Validation and testing scripts
- `scripts/database/` - Database management scripts
- `scripts/testing/` - System testing and health checks

### Configuration Files

- `config/database_config.py` - Database configuration
- `config/angel_one_config.py` - Angel One API configuration
- `config/data_periods_config.py` - Data period configurations

### Example Scripts

- `run_analysis.py` - Basic analysis script
- `enhanced_unified_pipeline.py` - Interactive analysis
- `run_custom_analysis.py` - Custom analysis configuration

## 🎯 Best Practices

### Data Management

- Use database storage for large datasets
- Regularly clean cache files
- Monitor data quality metrics
- Use appropriate data periods

### Performance

- Adjust worker count based on system resources
- Use incremental updates for real-time data
- Monitor memory usage
- Optimize database queries

### Predictions

- Validate predictions regularly
- Use multiple timeframes for comprehensive analysis
- Monitor model performance
- Update models periodically

## 📞 Support

### Documentation

- **User Guide**: Complete usage instructions
- **Quick Reference**: Essential commands and features
- **System Architecture**: Technical implementation details
- **Troubleshooting**: Common issues and solutions

### System Health

- Run health checks regularly
- Monitor system logs
- Validate data quality
- Test prediction accuracy

---

**⚠️ Disclaimer**: This system is designed for educational and research purposes. Always verify predictions with multiple sources and consider market risks before making investment decisions.

**📧 Contact**: For technical support or feature requests, please refer to the troubleshooting guide or system documentation.
