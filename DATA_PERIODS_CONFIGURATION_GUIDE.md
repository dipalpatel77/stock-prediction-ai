# 📊 Data Periods Configuration Guide

## 🎯 Overview

This guide explains the new configurable data download periods system that allows you to customize how much historical data is downloaded for different types of analysis.

## 🔧 Issues Fixed

### ✅ **Angel One Symbol Lookup Fix**

- **Problem**: System was looking up `ICICIBANK.NS` (with suffix) in Angel One database
- **Solution**: Now correctly removes `.NS`, `.BO`, `.NSE`, `.BSE` suffixes before lookup
- **Result**: Indian stocks like `ICICIBANK.NS` will now be looked up as `ICICIBANK` in Angel One

### ✅ **Parameter Name Fix**

- **Problem**: `get_historical_data()` was receiving `symbol=` instead of `symbol_name=`
- **Solution**: Updated all method calls to use correct parameter names
- **Result**: No more "unexpected keyword argument" errors

## 📈 **Current vs New Data Periods**

### **Before (Fixed Periods):**

```python
# Hardcoded periods
DEFAULT_PERIOD = '2y'  # 2 years for everything
ANGEL_ONE_PERIOD = '30 days'  # 30 days for Angel One
```

### **After (Configurable Periods):**

```python
# Configurable periods based on use case
RECOMMENDED_PERIODS = {
    'default': '1y',              # 1 year - balanced
    'angel_one': '6mo',           # 6 months - faster for Indian stocks
    'yfinance': '1y',             # 1 year - good for international
    'quick_check': '3mo',         # 3 months - for quick analysis
    'comprehensive': '2y'         # 2 years - for detailed analysis
}
```

## 🎛️ **Configuration Options**

### **1. Recommended Configuration (Default)**

```python
RECOMMENDED_PERIODS = {
    'default': '1y',              # 1 year - balanced performance and data
    'angel_one': '6mo',           # 6 months - faster for Indian stocks
    'yfinance': '1y',             # 1 year - good for international stocks
    'quick_check': '3mo',         # 3 months - for quick analysis
    'comprehensive': '2y'         # 2 years - for detailed analysis
}
```

### **2. Performance Optimized**

```python
PERFORMANCE_PERIODS = {
    'default': '6mo',             # 6 months - faster processing
    'angel_one': '3mo',           # 3 months - very fast for Indian stocks
    'yfinance': '6mo',            # 6 months - faster for international
    'quick_check': '1mo',         # 1 month - very quick analysis
    'comprehensive': '1y'         # 1 year - still comprehensive but faster
}
```

### **3. Comprehensive Analysis**

```python
COMPREHENSIVE_PERIODS = {
    'default': '2y',              # 2 years - maximum data
    'angel_one': '1y',            # 1 year - max recommended for Angel One
    'yfinance': '3y',             # 3 years - extensive for international
    'quick_check': '6mo',         # 6 months - still quick but more data
    'comprehensive': '5y'         # 5 years - maximum comprehensive analysis
}
```

## 🔍 **Analysis-Specific Periods**

### **Technical Analysis**

- **Default**: `6mo` (6 months)
- **Reason**: Sufficient for most technical indicators
- **Best for**: RSI, MACD, Bollinger Bands, Moving Averages

### **Fundamental Analysis**

- **Default**: `2y` (2 years)
- **Reason**: Need more data for financial ratios and trends
- **Best for**: P/E ratios, revenue growth, profitability analysis

### **Sentiment Analysis**

- **Default**: `3mo` (3 months)
- **Reason**: Recent sentiment is more relevant
- **Best for**: News sentiment, social media analysis

### **Prediction Models**

- **Default**: `1y` (1 year)
- **Reason**: Balanced for ML model training
- **Best for**: Machine learning, AI predictions

## 📡 **Source-Specific Periods**

### **Angel One (Indian Stocks)**

- **Recommended**: `6mo` (6 months)
- **Performance**: `3mo` (3 months)
- **Comprehensive**: `1y` (1 year)
- **Reason**: Faster API, optimized for Indian market

### **yfinance (International Stocks)**

- **Recommended**: `1y` (1 year)
- **Performance**: `6mo` (6 months)
- **Comprehensive**: `3y` (3 years)
- **Reason**: More data available, slower API

## 🚀 **How to Use**

### **1. Basic Usage (Recommended)**

```python
from core.data_service import DataService

# Use recommended configuration (default)
ds = DataService()  # Uses 'recommended' periods
data = ds.load_stock_data("AAPL")  # Will use 1y for yfinance
data = ds.load_stock_data("RELIANCE.NS")  # Will use 6mo for Angel One
```

### **2. Performance Optimized**

```python
# Use performance configuration for faster processing
ds = DataService(period_config="performance")
data = ds.load_stock_data("AAPL")  # Will use 6mo for yfinance
data = ds.load_stock_data("RELIANCE.NS")  # Will use 3mo for Angel One
```

### **3. Comprehensive Analysis**

```python
# Use comprehensive configuration for maximum data
ds = DataService(period_config="comprehensive")
data = ds.load_stock_data("AAPL")  # Will use 3y for yfinance
data = ds.load_stock_data("RELIANCE.NS")  # Will use 1y for Angel One
```

### **4. Manual Period Override**

```python
# Override with specific period
ds = DataService(period_config="recommended")
data = ds.load_stock_data("AAPL", period="5y")  # Force 5 years
```

## 📊 **Performance Comparison**

| Configuration     | Indian Stocks | International Stocks | Processing Speed | Data Quality         |
| ----------------- | ------------- | -------------------- | ---------------- | -------------------- |
| **Performance**   | 3mo           | 6mo                  | ⚡⚡⚡ Fast      | ⭐⭐ Good            |
| **Recommended**   | 6mo           | 1y                   | ⚡⚡ Medium      | ⭐⭐⭐ Very Good     |
| **Comprehensive** | 1y            | 3y                   | ⚡ Slow          | ⭐⭐⭐⭐⭐ Excellent |

## 🎯 **Recommendations by Use Case**

### **Day Trading / Quick Checks**

```python
ds = DataService(period_config="performance")
# Use 3mo for Indian stocks, 6mo for international
```

### **Regular Analysis / Most Users**

```python
ds = DataService(period_config="recommended")  # Default
# Use 6mo for Indian stocks, 1y for international
```

### **Research / Backtesting**

```python
ds = DataService(period_config="comprehensive")
# Use 1y for Indian stocks, 3y for international
```

## 🔧 **Custom Configuration**

You can create your own configuration by modifying `config/data_periods_config.py`:

```python
# Add your custom configuration
CUSTOM_PERIODS = {
    'default': '9mo',             # Your preferred default
    'angel_one': '4mo',           # Your preferred for Indian stocks
    'yfinance': '9mo',            # Your preferred for international
    'quick_check': '2mo',         # Your preferred for quick checks
    'comprehensive': '1.5y'       # Your preferred for comprehensive
}

# Use it
ds = DataService(period_config="custom")
```

## 📈 **Benefits**

### **✅ Performance Benefits**

- **Faster Downloads**: Shorter periods = faster data retrieval
- **Reduced API Calls**: Fewer requests to data providers
- **Lower Memory Usage**: Less data to process and store
- **Faster Analysis**: Quicker model training and predictions

### **✅ Quality Benefits**

- **Optimized for Source**: Different periods for different APIs
- **Analysis-Specific**: Tailored periods for different analysis types
- **Flexible**: Easy to adjust based on your needs
- **Consistent**: Standardized across your application

### **✅ User Experience Benefits**

- **Configurable**: Choose what works best for you
- **Intelligent Defaults**: Smart defaults based on stock type
- **Easy to Use**: Simple configuration options
- **Backward Compatible**: Existing code continues to work

## 🎉 **Summary**

The new data periods configuration system provides:

1. **🔧 Fixed Issues**: Angel One symbol lookup and parameter errors resolved
2. **⚡ Performance**: Optimized periods for faster processing
3. **🎯 Flexibility**: Configurable periods for different use cases
4. **📊 Intelligence**: Smart defaults based on stock type and analysis
5. **🔄 Compatibility**: Backward compatible with existing code

**Choose your configuration based on your needs:**

- **Performance**: For speed and quick analysis
- **Recommended**: For balanced performance and quality (default)
- **Comprehensive**: For maximum data and detailed analysis

The system will automatically use the appropriate period for each stock type and data source! 🚀
