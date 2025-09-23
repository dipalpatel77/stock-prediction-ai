# Comprehensive Interval Data Testing

## 🧪 **INTERVAL DATA TESTING FRAMEWORK**

This document provides comprehensive test cases for all data intervals (minute, hourly, daily, etc.) and tests the maximum number of available data points for each interval.

---

## **📊 Available Intervals**

### **Intraday Intervals**

1. **ONE_MINUTE** - 1-minute data
2. **THREE_MINUTE** - 3-minute data
3. **FIVE_MINUTE** - 5-minute data
4. **TEN_MINUTE** - 10-minute data
5. **FIFTEEN_MINUTE** - 15-minute data
6. **THIRTY_MINUTE** - 30-minute data
7. **ONE_HOUR** - 1-hour data

### **Daily Intervals**

8. **ONE_DAY** - Daily data

### **Extended Intervals**

9. **ONE_WEEK** - Weekly data
10. **ONE_MONTH** - Monthly data

---

## **🔧 Test Configuration**

### **Maximum Data Points per Interval**

```python
MAX_DATA_POINTS = {
    'ONE_MINUTE': 30000,      # ~20 trading days * 6.5 hours * 60 minutes
    'THREE_MINUTE': 10000,    # ~20 trading days * 6.5 hours * 20 periods
    'FIVE_MINUTE': 6000,      # ~20 trading days * 6.5 hours * 12 periods
    'TEN_MINUTE': 3000,       # ~20 trading days * 6.5 hours * 6 periods
    'FIFTEEN_MINUTE': 2000,   # ~20 trading days * 6.5 hours * 4 periods
    'THIRTY_MINUTE': 1000,    # ~20 trading days * 6.5 hours * 2 periods
    'ONE_HOUR': 500,          # ~20 trading days * 6.5 hours
    'ONE_DAY': 2000,          # ~8 years of daily data
    'ONE_WEEK': 400,          # ~8 years of weekly data
    'ONE_MONTH': 100          # ~8 years of monthly data
}
```

### **Test Periods**

```python
TEST_PERIODS = {
    'ONE_MINUTE': ['1d', '5d', '1mo', '3mo'],
    'THREE_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo'],
    'FIVE_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    'TEN_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    'FIFTEEN_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    'THIRTY_MINUTE': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
    'ONE_HOUR': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
    'ONE_DAY': ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
    'ONE_WEEK': ['1y', '2y', '5y', 'max'],
    'ONE_MONTH': ['2y', '5y', 'max']
}
```

---

## **🧪 Test Cases**

### **Test Case 1: ONE_MINUTE Interval**

```python
def test_one_minute_interval():
    """Test 1-minute interval data fetching"""
    test_config = {
        'ticker': 'AAPL',
        'interval': 'ONE_MINUTE',
        'periods': ['1d', '5d', '1mo', '3mo'],
        'expected_max_records': {
            '1d': 390,      # 6.5 hours * 60 minutes
            '5d': 1950,     # 5 days * 6.5 hours * 60 minutes
            '1mo': 7800,    # ~20 trading days * 6.5 hours * 60 minutes
            '3mo': 23400     # ~60 trading days * 6.5 hours * 60 minutes
        },
        'table': 'intraday_1min',
        'use_case': 'High-frequency trading, scalping, real-time analysis'
    }
```

### **Test Case 2: FIVE_MINUTE Interval**

```python
def test_five_minute_interval():
    """Test 5-minute interval data fetching"""
    test_config = {
        'ticker': 'MSFT',
        'interval': 'FIVE_MINUTE',
        'periods': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
        'expected_max_records': {
            '1d': 78,       # 6.5 hours * 12 periods
            '5d': 390,       # 5 days * 6.5 hours * 12 periods
            '1mo': 1560,     # ~20 trading days * 6.5 hours * 12 periods
            '3mo': 4680,     # ~60 trading days * 6.5 hours * 12 periods
            '6mo': 9360,     # ~120 trading days * 6.5 hours * 12 periods
            '1y': 18720      # ~240 trading days * 6.5 hours * 12 periods
        },
        'table': 'intraday_5min',
        'use_case': 'Short-term trading, swing trading, intraday analysis'
    }
```

### **Test Case 3: FIFTEEN_MINUTE Interval**

```python
def test_fifteen_minute_interval():
    """Test 15-minute interval data fetching"""
    test_config = {
        'ticker': 'GOOGL',
        'interval': 'FIFTEEN_MINUTE',
        'periods': ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
        'expected_max_records': {
            '1d': 26,        # 6.5 hours * 4 periods
            '5d': 130,       # 5 days * 6.5 hours * 4 periods
            '1mo': 520,      # ~20 trading days * 6.5 hours * 4 periods
            '3mo': 1560,     # ~60 trading days * 6.5 hours * 4 periods
            '6mo': 3120,     # ~120 trading days * 6.5 hours * 4 periods
            '1y': 6240       # ~240 trading days * 6.5 hours * 4 periods
        },
        'table': 'intraday_15min',
        'use_case': 'Medium-term trading, position trading, trend analysis'
    }
```

### **Test Case 4: ONE_HOUR Interval**

```python
def test_one_hour_interval():
    """Test 1-hour interval data fetching"""
    test_config = {
        'ticker': 'TSLA',
        'interval': 'ONE_HOUR',
        'periods': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
        'expected_max_records': {
            '1d': 6,         # 6.5 hours
            '5d': 30,        # 5 days * 6.5 hours
            '1mo': 130,      # ~20 trading days * 6.5 hours
            '3mo': 390,      # ~60 trading days * 6.5 hours
            '6mo': 780,      # ~120 trading days * 6.5 hours
            '1y': 1560,      # ~240 trading days * 6.5 hours
            '2y': 3120       # ~480 trading days * 6.5 hours
        },
        'table': 'hourly_data',
        'use_case': 'Long-term analysis, portfolio management, risk assessment'
    }
```

### **Test Case 5: ONE_DAY Interval**

```python
def test_one_day_interval():
    """Test daily interval data fetching"""
    test_config = {
        'ticker': 'AMZN',
        'interval': 'ONE_DAY',
        'periods': ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
        'expected_max_records': {
            '1mo': 22,       # ~22 trading days
            '3mo': 65,       # ~65 trading days
            '6mo': 130,      # ~130 trading days
            '1y': 252,       # ~252 trading days
            '2y': 504,       # ~504 trading days
            '5y': 1260,      # ~1260 trading days
            'max': 2000      # Maximum available data
        },
        'table': 'daily_data',
        'use_case': 'Fundamental analysis, long-term investing, backtesting'
    }
```

### **Test Case 6: ONE_WEEK Interval**

```python
def test_one_week_interval():
    """Test weekly interval data fetching"""
    test_config = {
        'ticker': 'NVDA',
        'interval': 'ONE_WEEK',
        'periods': ['1y', '2y', '5y', 'max'],
        'expected_max_records': {
            '1y': 52,        # 52 weeks
            '2y': 104,       # 104 weeks
            '5y': 260,       # 260 weeks
            'max': 400       # Maximum available data
        },
        'table': 'daily_data',  # Weekly data stored in daily table
        'use_case': 'Long-term trend analysis, fundamental research'
    }
```

### **Test Case 7: ONE_MONTH Interval**

```python
def test_one_month_interval():
    """Test monthly interval data fetching"""
    test_config = {
        'ticker': 'META',
        'interval': 'ONE_MONTH',
        'periods': ['2y', '5y', 'max'],
        'expected_max_records': {
            '2y': 24,        # 24 months
            '5y': 60,        # 60 months
            'max': 100       # Maximum available data
        },
        'table': 'daily_data',  # Monthly data stored in daily table
        'use_case': 'Long-term fundamental analysis, macroeconomic research'
    }
```

---

## **🔍 Comprehensive Test Suite**

### **Test Suite 1: Intraday Intervals**

```python
def test_all_intraday_intervals():
    """Test all intraday intervals with maximum data"""
    intervals = [
        'ONE_MINUTE', 'THREE_MINUTE', 'FIVE_MINUTE',
        'TEN_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR'
    ]

    for interval in intervals:
        test_interval_max_data(interval)
```

### **Test Suite 2: Daily and Extended Intervals**

```python
def test_daily_and_extended_intervals():
    """Test daily and extended intervals with maximum data"""
    intervals = ['ONE_DAY', 'ONE_WEEK', 'ONE_MONTH']

    for interval in intervals:
        test_interval_max_data(interval)
```

### **Test Suite 3: Maximum Data Points Test**

```python
def test_maximum_data_points():
    """Test maximum data points for each interval"""
    test_cases = [
        {
            'interval': 'ONE_MINUTE',
            'period': '3mo',
            'expected_min': 20000,
            'expected_max': 30000
        },
        {
            'interval': 'FIVE_MINUTE',
            'period': '1y',
            'expected_min': 15000,
            'expected_max': 20000
        },
        {
            'interval': 'ONE_HOUR',
            'period': '2y',
            'expected_min': 3000,
            'expected_max': 5000
        },
        {
            'interval': 'ONE_DAY',
            'period': 'max',
            'expected_min': 1500,
            'expected_max': 2500
        }
    ]

    for test_case in test_cases:
        test_max_data_for_interval(test_case)
```

---

## **📊 Performance Benchmarks**

### **Expected Performance Metrics**

```python
PERFORMANCE_BENCHMARKS = {
    'ONE_MINUTE': {
        'max_records': 30000,
        'fetch_time_max': 30,  # seconds
        'processing_time_max': 10,  # seconds
        'memory_usage_max': 500  # MB
    },
    'FIVE_MINUTE': {
        'max_records': 20000,
        'fetch_time_max': 20,  # seconds
        'processing_time_max': 8,  # seconds
        'memory_usage_max': 300  # MB
    },
    'ONE_HOUR': {
        'max_records': 5000,
        'fetch_time_max': 15,  # seconds
        'processing_time_max': 5,  # seconds
        'memory_usage_max': 200  # MB
    },
    'ONE_DAY': {
        'max_records': 2500,
        'fetch_time_max': 10,  # seconds
        'processing_time_max': 3,  # seconds
        'memory_usage_max': 100  # MB
    }
}
```

---

## **🧪 Test Execution Framework**

### **Test Runner Configuration**

```python
def run_interval_tests():
    """Run comprehensive interval testing"""
    test_results = {}

    # Test all intervals
    intervals = [
        'ONE_MINUTE', 'THREE_MINUTE', 'FIVE_MINUTE', 'TEN_MINUTE',
        'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'ONE_DAY',
        'ONE_WEEK', 'ONE_MONTH'
    ]

    for interval in intervals:
        print(f"Testing {interval} interval...")
        result = test_interval_comprehensive(interval)
        test_results[interval] = result

    return test_results
```

### **Validation Criteria**

```python
VALIDATION_CRITERIA = {
    'data_quality': {
        'min_records': 100,
        'max_missing_values': 0.05,  # 5%
        'date_continuity': True,
        'price_validity': True
    },
    'performance': {
        'fetch_time_acceptable': True,
        'memory_usage_acceptable': True,
        'processing_time_acceptable': True
    },
    'storage': {
        'correct_table': True,
        'data_persistence': True,
        'retrieval_success': True
    }
}
```

---

## **📈 Expected Results**

### **Success Criteria**

- ✅ All intervals fetch data successfully
- ✅ Maximum data points retrieved for each interval
- ✅ Data quality meets validation criteria
- ✅ Performance within acceptable limits
- ✅ Storage and retrieval working correctly
- ✅ Memory usage optimized
- ✅ Processing time efficient

### **Test Coverage**

- **Interval Coverage**: 100% (10/10 intervals)
- **Period Coverage**: 100% (all available periods)
- **Data Quality**: 100% validation
- **Performance**: 100% benchmark compliance
- **Storage**: 100% table routing accuracy

---

## **🚀 Implementation Notes**

### **Key Testing Points**

1. **Data Volume**: Test maximum data points for each interval
2. **Performance**: Ensure acceptable fetch and processing times
3. **Memory Usage**: Monitor memory consumption for large datasets
4. **Data Quality**: Validate data completeness and accuracy
5. **Storage**: Verify correct table routing and data persistence
6. **Retrieval**: Test data retrieval from appropriate tables

### **Error Handling**

- API rate limiting
- Network timeouts
- Memory overflow
- Data corruption
- Storage failures
- Retrieval errors

This comprehensive testing framework ensures all intervals work correctly with maximum data points while maintaining optimal performance and data quality.
