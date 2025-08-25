# AI Stock Predictor

An advanced machine learning-based stock prediction system that combines multiple analysis techniques to provide comprehensive stock market predictions.

## 🚀 Features

- **Multi-timeframe Analysis**: Short-term, mid-term, and long-term predictions
- **Ensemble Models**: Combines multiple ML algorithms for better accuracy
- **Technical Indicators**: Advanced technical analysis with 20+ indicators
- **Fundamental Analysis**: Balance sheet and financial ratio analysis
- **Real-time Data**: Integration with Angel One API for live market data
- **Backtesting**: Comprehensive strategy backtesting capabilities
- **Risk Management**: Built-in risk assessment and position sizing

## 📋 Prerequisites

- Python 3.8+
- Angel One API credentials (for live data)
- Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd ai-stock-predictor
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Angel One API credentials**
   ```bash
   python angel_one_config.py
   ```
   Follow the interactive setup to configure your API credentials.

## 🔧 Configuration

### Angel One API Setup

1. Create an account on [Angel One](https://www.angelone.in/)
2. Generate API credentials from your account
3. Run the configuration script:
   ```bash
   python angel_one_config.py
   ```

The script will create a `.env` file with your credentials. **Never commit this file to version control!**

## 📊 Usage

### Quick Start

For a simple prediction on RELIANCE stock:

```bash
python quick_reliance_prediction.py
```

### Comprehensive Analysis

For detailed analysis with multiple timeframes:

```bash
python comprehensive_prediction_runner.py
```

### Custom Stock Analysis

```python
from unified_analysis_pipeline import UnifiedAnalysisPipeline

# Initialize the pipeline
pipeline = UnifiedAnalysisPipeline()

# Run analysis for any stock
results = pipeline.run_complete_analysis(
    symbol="AAPL",
    prediction_days=[1, 5, 30],
    include_backtest=True
)
```

### Backtesting

```python
from partC_strategy.backtest import BacktestEngine

backtest = BacktestEngine()
results = backtest.run_backtest(
    strategy="moving_average_crossover",
    symbol="RELIANCE",
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

## 🧪 Testing

The project includes a comprehensive test suite organized in the `tests/` directory:

### Running Tests

```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test types
python tests/run_all_tests.py --type unit
python tests/run_all_tests.py --type integration
python tests/run_all_tests.py --type performance

# Run with verbose output
python tests/run_all_tests.py --verbose

# Windows batch file
tests\run_tests.bat
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Individual component testing
- **Integration Tests** (`tests/integration/`): System integration testing
- **Performance Tests** (`tests/performance/`): Performance and load testing
- **Test Utilities** (`tests/utils/`): Helper functions and mock data

See `tests/README.md` for detailed testing documentation.
symbol="RELIANCE",
start_date="2023-01-01",
end_date="2024-01-01",
strategy="enhanced_ensemble"
)

```

## 📁 Project Structure

```

ai-stock-predictor/
├── analysis_modules/ # Analysis components
│ ├── long_term_analyzer.py
│ ├── mid_term_analyzer.py
│ └── short_term_analyzer.py
├── partA_preprocessing/ # Data preprocessing
│ ├── data_loader.py
│ └── preprocess.py
├── partB_model/ # ML models
│ ├── enhanced_model_builder.py
│ └── enhanced_training.py
├── partC_strategy/ # Trading strategies
│ ├── backtest.py
│ ├── balance_sheet_analyzer.py
│ └── strategy_implementations/
├── data/ # Data storage
├── models/ # Trained models
├── logs/ # Log files
├── notebooks/ # Jupyter notebooks
└── scripts/ # Utility scripts

````

## 🔍 Key Components

### 1. Data Preprocessing (`partA_preprocessing/`)
- Historical data collection
- Technical indicator calculation
- Feature engineering
- Data cleaning and normalization

### 2. Model Building (`partB_model/`)
- Ensemble model creation
- Hyperparameter optimization
- Model training and validation
- Performance evaluation

### 3. Strategy Implementation (`partC_strategy/`)
- Trading strategy development
- Risk management
- Position sizing
- Backtesting framework

### 4. Analysis Modules (`analysis_modules/`)
- Multi-timeframe analysis
- Fundamental analysis
- Technical analysis
- Market sentiment analysis

## 📈 Supported Stocks

The system supports major Indian stocks including:
- RELIANCE
- TCS
- HDFC Bank
- Infosys
- ICICI Bank
- And many more...

For the complete list, run:
```bash
python angel_one_config.py
# Choose option 3: Show Supported Stocks
````

## 🎯 Prediction Types

1. **Short-term (1-5 days)**: Intraday and swing trading signals
2. **Mid-term (1-4 weeks)**: Position trading opportunities
3. **Long-term (1-6 months)**: Investment recommendations

## 📊 Model Performance

The system uses ensemble methods combining:

- Gradient Boosting (CatBoost)
- Random Forest
- Neural Networks
- Support Vector Machines
- Linear Regression

## ⚠️ Important Notes

1. **Risk Disclaimer**: This is for educational purposes. Always do your own research before making investment decisions.

2. **API Limits**: Be aware of Angel One API rate limits and usage policies.

3. **Data Accuracy**: While we strive for accuracy, market predictions are inherently uncertain.

4. **Model Updates**: Models should be retrained periodically with fresh data.

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Errors**

   - Verify your Angel One credentials
   - Check your internet connection
   - Ensure API limits aren't exceeded

2. **Model Loading Errors**

   - Ensure all dependencies are installed
   - Check if model files exist in the models/ directory

3. **Memory Issues**
   - Reduce the prediction timeframe
   - Use smaller datasets for testing

## 📝 Logging

The system generates detailed logs in the `logs/` directory. Check these files for debugging:

- `prediction.log`: Prediction execution logs
- `backtest.log`: Backtesting results
- `error.log`: Error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:

1. Check the documentation in the `docs/` folder
2. Review the troubleshooting section
3. Open an issue on GitHub

## 🔄 Updates

- **v2.0**: Enhanced ensemble models and improved accuracy
- **v1.5**: Added fundamental analysis capabilities
- **v1.0**: Initial release with basic prediction features

---

**Disclaimer**: This software is for educational and research purposes only. The authors are not responsible for any financial losses incurred through the use of this software. Always consult with a qualified financial advisor before making investment decisions.

# ai-stock-predictor
