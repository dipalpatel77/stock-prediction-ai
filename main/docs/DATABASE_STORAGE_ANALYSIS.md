# Database Storage Analysis

## 📊 **Database Files and Stock Records Storage**

### **🗂️ Database Files Found:**

| Database File        | Size   | Last Modified           | Purpose                                           |
| -------------------- | ------ | ----------------------- | ------------------------------------------------- |
| `default.db`         | 458 KB | 27-Sep-2025 03:58:27 PM | **MAIN DATABASE** - Contains 3,000 stock records  |
| `test_angel_one.db`  | 2.5 MB | 24-Sep-2025 05:35:52 PM | **TEST DATABASE** - Contains 365 Reliance records |
| `data/stock_data.db` | 53 KB  | 27-Sep-2025 01:12:59 PM | **CURRENT DATABASE** - Contains 5 test records    |
| `test_stock_data.db` | 16 KB  | 27-Sep-2025 01:14:51 PM | **TEST DATABASE** - Small test file               |
| `simple_test.db`     | 16 KB  | 27-Sep-2025 01:13:23 PM | **TEST DATABASE** - Simple test file              |

## 🎯 **Primary Stock Records Storage**

### **1. Main Database: `default.db`**

- **Location**: `D:\TradingProjcet\ai-stock-predictor\main\default.db`
- **Size**: 458 KB
- **Records**: **3,000 stock records**
- **Tables**: `angel_one_data`, `sqlite_sequence`

**Stock Data by Ticker:**
| Ticker | Records | Date Range | Status |
|--------|---------|------------|---------|
| **INFY** | 1,721 | 2020-09-29 to 2025-09-27 | ✅ **Largest dataset** |
| **TCS** | 704 | 2024-09-27 to 2025-09-27 | ✅ **Recent data** |
| **NTPC** | 537 | 2024-09-27 to 2025-09-27 | ✅ **Recent data** |
| **COALINDIA** | 38 | 2020-09-29 to 2025-06-23 | ✅ **Historical data** |

### **2. Test Database: `test_angel_one.db`**

- **Location**: `D:\TradingProjcet\ai-stock-predictor\main\test_angel_one.db`
- **Size**: 2.5 MB
- **Records**: **365 Reliance records**
- **Tables**: Multiple interval tables for Reliance stock

**Reliance Data by Interval:**

- `angel_one_one_minute_reliance`
- `angel_one_five_minute_reliance`
- `angel_one_fifteen_minute_reliance`
- `angel_one_thirty_minute_reliance`
- `angel_one_one_hour_reliance`
- `angel_one_one_day_reliance`

### **3. Current Database: `data/stock_data.db`**

- **Location**: `D:\TradingProjcet\ai-stock-predictor\main\data\stock_data.db`
- **Size**: 53 KB
- **Records**: **5 test records**
- **Tables**: `stock_data`, `stock_metadata`, `data_quality`, `angel_one_data`

## 📈 **Stock Records Summary**

### **Total Stock Records Across All Databases:**

- **Main Database**: 3,000 records (INFY, TCS, NTPC, COALINDIA)
- **Test Database**: 365 records (Reliance - multiple intervals)
- **Current Database**: 5 records (TEST data)
- **Total**: **3,370 stock records**

### **Data Distribution:**

```
INFY (Infosys):     1,721 records (51.1%) - Largest dataset
TCS (Tata Consultancy): 704 records (20.9%) - Recent data
NTPC:               537 records (15.9%) - Recent data
Reliance:           365 records (10.8%) - Test data
COALINDIA:           38 records (1.1%) - Historical data
TEST:                 5 records (0.1%) - Test data
```

## 🗂️ **Database Structure Analysis**

### **Main Database (`default.db`) Structure:**

```sql
CREATE TABLE angel_one_data (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    interval_type VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Test Database (`test_angel_one.db`) Structure:**

- Multiple tables for different time intervals
- Reliance stock data across various timeframes
- 2.5 MB of historical data

## 🎯 **Key Findings**

### **✅ Primary Storage:**

1. **`default.db`** - Contains the main stock records (3,000 records)
2. **`test_angel_one.db`** - Contains Reliance test data (365 records)
3. **`data/stock_data.db`** - Current working database (5 test records)

### **📊 Data Quality:**

- **INFY**: 1,721 records spanning 5 years (2020-2025)
- **TCS**: 704 records from recent period (2024-2025)
- **NTPC**: 537 records from recent period (2024-2025)
- **Reliance**: 365 records across multiple intervals
- **COALINDIA**: 38 records with historical data

### **🔍 Data Access Patterns:**

- **Most Active**: INFY (1,721 records)
- **Most Recent**: TCS, NTPC (2024-2025 data)
- **Most Historical**: INFY, COALINDIA (2020-2025 range)
- **Most Tested**: Reliance (multiple interval tables)

## 🚀 **Recommendations**

### **1. Primary Database Usage:**

- Use `default.db` for production data (3,000 records)
- Use `data/stock_data.db` for current operations
- Keep `test_angel_one.db` for testing purposes

### **2. Data Management:**

- **INFY**: Largest dataset, good for model training
- **TCS**: Recent data, good for current predictions
- **NTPC**: Recent data, good for current predictions
- **Reliance**: Test data, good for validation

### **3. Storage Optimization:**

- Main database: 458 KB for 3,000 records (efficient)
- Test database: 2.5 MB for 365 records (detailed intervals)
- Current database: 53 KB for 5 records (minimal)

## 📝 **Summary**

The stock records are primarily stored in **`default.db`** with **3,000 records** across 4 major Indian stocks:

- **INFY**: 1,721 records (largest dataset)
- **TCS**: 704 records (recent data)
- **NTPC**: 537 records (recent data)
- **COALINDIA**: 38 records (historical data)

Additional test data is stored in `test_angel_one.db` with 365 Reliance records across multiple time intervals.

The database storage is working efficiently with proper optimization and fast access to stock records! 🚀
