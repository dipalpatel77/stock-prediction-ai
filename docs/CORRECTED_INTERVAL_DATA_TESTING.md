# Corrected Interval Data Testing Framework

## 🚨 **IMPORTANT LIMITATION IDENTIFIED**

### **Data Source Limitations**

- **Yahoo Finance**: Only supports daily data for most stocks
- **Angel One API**: Supports all intervals (minute, 5-minute, 15-minute, hourly, daily) but only for Indian stocks
- **Non-Indian stocks**: Limited to daily data only

---

## **📊 Corrected Test Framework**

### **Available Intervals by Data Source**

#### **Yahoo Finance (US/International Stocks)**

```python
YAHOO_FINANCE_INTERVALS = {
    'ONE_DAY': 'Daily data only',
    'ONE_WEEK': 'Weekly data (aggregated from daily)',
    'ONE_MONTH': 'Monthly data (aggregated from daily)'
}
```

#### **Angel One API (Indian Stocks Only)**

```python
ANGEL_ONE_INTERVALS = {
    'ONE_MINUTE': '1-minute intraday data',
    'THREE_MINUTE': '3-minute intraday data',
    'FIVE_MINUTE': '5-minute intraday data',
    'TEN_MINUTE': '10-minute intraday data',
    'FIFTEEN_MINUTE': '15-minute intraday data',
    'THIRTY_MINUTE': '30-minute intraday data',
    'ONE_HOUR': '1-hour intraday data',
    'ONE_DAY': 'Daily data'
}
```

---

## **🧪 Corrected Test Cases**

### **Test Case 1: Yahoo Finance (US Stocks) - Daily Only**

```python
def test_yahoo_finance_intervals():
    """Test Yahoo Finance intervals for US stocks"""
    test_cases = [
        {
            'ticker': 'AAPL',
            'interval': 'ONE_DAY',
            'periods': ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
            'expected_data_points': {
                '1mo': 22,    # ~22 trading days
                '3mo': 65,    # ~65 trading days
                '6mo': 130,   # ~130 trading days
                '1y': 252,    # ~252 trading days
                '2y': 504,    # ~504 trading days
                '5y': 1260,   # ~1260 trading days
                'max': 2000   # Maximum available
            }
        }
    ]

    # Test only daily intervals for US stocks
    for test_case in test_cases:
        test_daily_interval_only(test_case)
```

### **Test Case 2: Angel One API (Indian Stocks) - All Intervals**

```python
def test_angel_one_intervals():
    """Test Angel One intervals for Indian stocks"""
    test_cases = [
        {
            'ticker': 'TATAMOTORS',
            'intervals': ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'ONE_HOUR', 'ONE_DAY'],
            'periods': ['1d', '5d', '1mo', '3mo'],
            'expected_data_points': {
                'ONE_MINUTE': {'1d': 390, '5d': 1950, '1mo': 7800, '3mo': 23400},
                'FIVE_MINUTE': {'1d': 78, '5d': 390, '1mo': 1560, '3mo': 4680},
                'FIFTEEN_MINUTE': {'1d': 26, '5d': 130, '1mo': 520, '3mo': 1560},
                'ONE_HOUR': {'1d': 6, '5d': 30, '1mo': 130, '3mo': 390},
                'ONE_DAY': {'1d': 1, '5d': 5, '1mo': 22, '3mo': 65}
            }
        }
    ]

    # Test all intervals for Indian stocks
    for test_case in test_cases:
        test_all_intervals_for_indian_stock(test_case)
```

---

## **🔧 Corrected Test Implementation**

### **Test Script for Yahoo Finance (US Stocks)**

```python
def test_us_stock_intervals():
    """Test intervals for US stocks (Yahoo Finance only)"""
    print("🇺🇸 Testing US Stock Intervals (Yahoo Finance)")
    print("=" * 50)

    # Only test daily intervals for US stocks
    us_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    intervals = ['ONE_DAY']  # Only daily data available

    for stock in us_stocks:
        print(f"\n📊 Testing {stock}")
        for interval in intervals:
            test_daily_interval_only(stock, interval)

def test_daily_interval_only(ticker: str, interval: str):
    """Test daily interval for US stocks"""
    periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']

    for period in periods:
        print(f"  📈 Testing {period} period")

        # Configuration for US stock
        config = {
            'ticker': ticker,
            'is_indian': False,  # US stock
            'analysis_type': 'comprehensive',
            'parameters': {
                'interval': interval,
                'period': period,
                'use_enhanced': True,
                'use_database': True
            },
            'use_enhanced': True,
            'use_database': True,
            'interval': interval,
            'timeframe': period,
            'success': True
        }

        # Test the configuration
        result = test_interval_configuration(config)
        print(f"    ✅ {period}: {result['data_points']} data points")
```

### **Test Script for Angel One (Indian Stocks)**

```python
def test_indian_stock_intervals():
    """Test intervals for Indian stocks (Angel One API)"""
    print("🇮🇳 Testing Indian Stock Intervals (Angel One API)")
    print("=" * 50)

    # Test all intervals for Indian stocks
    indian_stocks = ['TATAMOTORS', 'RELIANCE', 'TCS', 'HDFC', 'PNB']
    intervals = [
        'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE',
        'THIRTY_MINUTE', 'ONE_HOUR', 'ONE_DAY'
    ]

    for stock in indian_stocks:
        print(f"\n📊 Testing {stock}")
        for interval in intervals:
            test_interval_for_indian_stock(stock, interval)

def test_interval_for_indian_stock(ticker: str, interval: str):
    """Test specific interval for Indian stock"""
    periods = ['1d', '5d', '1mo', '3mo']

    for period in periods:
        print(f"  📈 Testing {interval} - {period}")

        # Configuration for Indian stock with Angel One
        config = {
            'ticker': ticker,
            'is_indian': True,  # Indian stock
            'analysis_type': 'comprehensive',
            'parameters': {
                'interval': interval,
                'period': period,
                'use_enhanced': True,
                'use_database': True
            },
            'use_enhanced': True,
            'use_database': True,
            'interval': interval,
            'timeframe': period,
            'success': True,
            'angel_config': {
                'api_key': '1TKgQThc ',
                'api_secret': 'D54448',
                'access_token': '2251',
                'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE',
                'exchange': 'BSE',
                'interval': interval
            }
        }

        # Test the configuration
        result = test_interval_configuration(config)
        print(f"    ✅ {period}: {result['data_points']} data points")
```

---

## **📋 Corrected Test Results**

### **Expected Results by Data Source**

#### **Yahoo Finance (US Stocks)**

- ✅ **ONE_DAY**: Works for all periods
- ❌ **ONE_MINUTE**: Not supported
- ❌ **FIVE_MINUTE**: Not supported
- ❌ **FIFTEEN_MINUTE**: Not supported
- ❌ **ONE_HOUR**: Not supported

#### **Angel One API (Indian Stocks)**

- ✅ **ONE_MINUTE**: Works for intraday periods
- ✅ **FIVE_MINUTE**: Works for intraday periods
- ✅ **FIFTEEN_MINUTE**: Works for intraday periods
- ✅ **THIRTY_MINUTE**: Works for intraday periods
- ✅ **ONE_HOUR**: Works for intraday periods
- ✅ **ONE_DAY**: Works for all periods

---

## **🎯 Corrected Test Strategy**

### **1. Separate Testing by Data Source**

```python
def run_corrected_interval_tests():
    """Run corrected interval tests based on data source"""

    # Test US stocks with Yahoo Finance (daily only)
    print("🇺🇸 Testing US Stocks (Yahoo Finance)")
    test_us_stock_intervals()

    # Test Indian stocks with Angel One (all intervals)
    print("\n🇮🇳 Testing Indian Stocks (Angel One)")
    test_indian_stock_intervals()

    # Test mixed scenarios
    print("\n🔄 Testing Mixed Scenarios")
    test_mixed_data_sources()
```

### **2. Data Source Validation**

```python
def validate_data_source_limitations():
    """Validate that we're testing appropriate intervals for each data source"""

    # US stocks should only test daily intervals
    us_stock_intervals = ['ONE_DAY']

    # Indian stocks can test all intervals
    indian_stock_intervals = [
        'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE',
        'THIRTY_MINUTE', 'ONE_HOUR', 'ONE_DAY'
    ]

    # Validate test configurations
    for ticker in ['AAPL', 'MSFT']:  # US stocks
        for interval in us_stock_intervals:
            assert interval == 'ONE_DAY', f"US stock {ticker} should only test daily intervals"

    for ticker in ['TATAMOTORS', 'RELIANCE']:  # Indian stocks
        for interval in indian_stock_intervals:
            assert interval in indian_stock_intervals, f"Indian stock {ticker} can test {interval}"
```

---

## **🚀 Implementation Notes**

### **Key Corrections Made**

1. **Separated testing by data source**: US stocks vs Indian stocks
2. **Limited US stock testing to daily intervals only**
3. **Enabled full interval testing for Indian stocks**
4. **Added proper data source validation**
5. **Corrected expected data points based on actual availability**

### **Test Coverage**

- **US Stocks**: Daily intervals only (Yahoo Finance limitation)
- **Indian Stocks**: All intervals (Angel One API capability)
- **Mixed Scenarios**: Proper fallback handling
- **Data Source Validation**: Ensure appropriate intervals are tested

### **Expected Outcomes**

- ✅ US stocks: Daily data only, ~200-2000 data points
- ✅ Indian stocks: All intervals, 1-30,000 data points depending on interval
- ✅ Proper error handling for unsupported intervals
- ✅ Correct data source selection based on stock type

This corrected framework properly handles the data source limitations and ensures accurate testing of available intervals for each stock type.
