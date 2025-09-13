# 🚀 AI Stock Predictor

A comprehensive stock prediction system with advanced analysis capabilities, multi-currency support, and enhanced date formatting.

## ✨ Features

- **Unified Analysis Pipeline**: Complete end-to-end stock analysis
- **Multi-Currency Support**: Real-time currency conversion and formatting
- **Enhanced Date Handling**: Multiple date formats with timezone support
- **Phase-Based Analysis**: 
  - Phase 1: Enhanced fundamental analysis
  - Phase 2: Economic data and regulatory monitoring
  - Phase 3: Geopolitical risk and corporate actions
- **Comprehensive Reports**: Multi-format output (JSON, HTML, CSV, TXT)
- **Real-time Data**: FRED API, World Bank API, Angel One integration

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python unified_analysis_pipeline.py
```

## 📊 Usage

### Interactive Mode
```bash
python unified_analysis_pipeline.py
```

### Programmatic Usage
```python
from unified_analysis_pipeline import UnifiedAnalysisPipeline

# Initialize pipeline
pipeline = UnifiedAnalysisPipeline("AAPL")

# Run comprehensive analysis
success = pipeline.run_unified_analysis(period="2y", days_ahead=365, use_enhanced=True)
```

## 🏗️ Architecture

```
unified_analysis_pipeline.py     # Main entry point
├── core/                        # Core services
│   ├── data_service.py         # Data loading and preprocessing
│   ├── model_service.py        # Model training and prediction
│   ├── strategy_service.py     # Strategy analysis
│   ├── economic_data_service.py # Economic data integration
│   ├── currency_service.py     # Currency conversion
│   └── enhanced_date_utils.py  # Date utilities
├── config/                      # Configuration
├── phase*_integration.py       # Phase integrations
└── comprehensive_report_integration.py # Report generation
```

## 📈 Analysis Types

1. **Intraday Forecast** (Hours)
2. **Short-term Forecast** (1-7 days)
3. **Medium-term Forecast** (1-4 weeks)
4. **Long-term Forecast** (1-12 months)
5. **Comprehensive Analysis** (All timeframes)
6. **Multi-Timeframe Analysis** (Short + Mid + Long term)

## 🔧 Configuration

### Data Periods
- **Recommended**: Balanced performance and accuracy
- **Performance**: Fast analysis with minimal data
- **Comprehensive**: Maximum data for accuracy

### Enhanced Features
- Economic indicators integration
- Multi-currency analysis
- Enhanced date formatting
- Risk assessment
- Trading recommendations

## 📊 Output Formats

- **JSON**: Structured data for programmatic use
- **HTML**: Rich formatted reports
- **CSV**: Tabular data for analysis
- **TXT**: Plain text summaries

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python tests/test_core_services.py
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions, please open an issue on GitHub.
