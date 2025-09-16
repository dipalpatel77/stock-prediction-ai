# AI Stock Predictor - Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. Angel One API Issues

#### Problem: Authentication Failed

```
❌ Angel One authentication failed
```

**Solutions:**

```bash
# Check credentials
python -c "
from src.utils.angel_one_data_downloader import AngelOneDataDownloader
downloader = AngelOneDataDownloader()
print('API Key:', downloader.api_key[:10] + '...')
print('Client ID:', downloader.client_id)
print('Auth Status:', downloader.authenticate())
"

# Verify TOTP setup
python -c "
import pyotp
from config.angel_one_config import AngelOneConfig
config = AngelOneConfig()
totp = pyotp.TOTP(config.totp_secret)
print('Current TOTP:', totp.now())
"
```

**Prevention:**

- Ensure TOTP secret is correctly configured
- Check system time synchronization
- Verify API credentials are valid

#### Problem: Rate Limiting

```
❌ API rate limit exceeded
```

**Solutions:**

```python
# Add delays between requests
import time
time.sleep(1)  # 1 second delay

# Use batch requests when possible
# Reduce concurrent requests
```

#### Problem: Invalid Symbol

```
❌ Symbol not found on Angel One
```

**Solutions:**

```bash
# Check symbol format
python -c "
from src.utils.angel_one_data_downloader import AngelOneDataDownloader
downloader = AngelOneDataDownloader()
symbol_info = downloader.get_symbol_token('RELIANCE', 'NSE')
print('Symbol Info:', symbol_info)
"

# Use correct exchange
# NSE: RELIANCE, TCS, INFY
# BSE: Add .BO suffix
```

### 2. Database Connection Issues

#### Problem: MySQL Connection Failed

```
❌ Can't connect to MySQL server
```

**Solutions:**

```bash
# Check MySQL status
sudo systemctl status mysql
# or
brew services list | grep mysql

# Start MySQL
sudo systemctl start mysql
# or
brew services start mysql

# Test connection
mysql -u root -p7874 -e "SHOW DATABASES;"

# Check port
netstat -an | grep 3306
```

**Configuration Check:**

```python
# Test database connection
python -c "
from src.core.database_service import DatabaseService
try:
    db = DatabaseService('mysql', 'mysql://root:7874@localhost/stock_data')
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

#### Problem: Database Not Found

```
❌ Unknown database 'stock_data'
```

**Solutions:**

```bash
# Create database
python scripts/database/setup_mysql_database.py

# Manual creation
mysql -u root -p7874 -e "CREATE DATABASE stock_data;"
```

#### Problem: Permission Denied

```
❌ Access denied for user 'root'@'localhost'
```

**Solutions:**

```bash
# Reset MySQL password
sudo mysql -u root
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '7874';
FLUSH PRIVILEGES;
EXIT;

# Or create new user
mysql -u root -p
CREATE USER 'stock_user'@'localhost' IDENTIFIED BY 'stock_password';
GRANT ALL PRIVILEGES ON stock_data.* TO 'stock_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. Data Loading Issues

#### Problem: No Data Retrieved

```
❌ No data found for ticker
```

**Solutions:**

```python
# Check data source
python -c "
from src.core.data_service import DataService
data_service = DataService()
print('Is Indian stock:', data_service._is_indian_stock('RELIANCE'))
print('Is Indian stock:', data_service._is_indian_stock('AAPL'))
"

# Test individual sources
python -c "
from src.utils.angel_one_data_downloader import AngelOneDataDownloader
downloader = AngelOneDataDownloader()
data = downloader.get_historical_data('RELIANCE', 'NSE', 'ONE_DAY', days_back=30)
print('Angel One data shape:', data.shape if data is not None else 'None')
"
```

#### Problem: Insufficient Data

```
⚠️ Insufficient data points: 50 < 100
```

**Solutions:**

```python
# Adjust minimum data points
data_service = DataService()
data_service.min_data_points = 50  # Reduce requirement

# Use longer period
df = data_service.load_stock_data('RELIANCE', period='2y')

# Check data availability
python -c "
import yfinance as yf
ticker = yf.Ticker('RELIANCE.NS')
hist = ticker.history(period='2y')
print('Available data:', len(hist), 'records')
"
```

### 4. Model Training Issues

#### Problem: Model Training Failed

```
❌ Model training failed: ValueError
```

**Solutions:**

```python
# Check data quality
python -c "
from src.core.data_service import DataService
data_service = DataService()
df = data_service.load_stock_data('RELIANCE')
print('Data shape:', df.shape)
print('Missing values:', df.isnull().sum().sum())
print('Data types:', df.dtypes)
"

# Check for infinite values
import numpy as np
print('Infinite values:', np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
```

#### Problem: Memory Issues

```
❌ MemoryError during model training
```

**Solutions:**

```python
# Reduce data size
data_service = DataService(period_config='quick_check')

# Use fewer algorithms
from src.core.model_service import ModelService
model_service = ModelService()
models = model_service.train_models(df, algorithms=['RandomForest', 'XGBoost'])

# Enable garbage collection
import gc
gc.collect()
```

### 5. Performance Issues

#### Problem: Slow Data Loading

```
⏳ Data loading taking too long
```

**Solutions:**

```python
# Use database storage
data_service = DataService(use_database=True)

# Enable caching
data_service = DataService(cache_dir='data/cache')

# Reduce data period
df = data_service.load_stock_data('RELIANCE', period='3mo')
```

#### Problem: High Memory Usage

```
⚠️ High memory usage detected
```

**Solutions:**

```python
# Monitor memory usage
import psutil
print('Memory usage:', psutil.virtual_memory().percent)

# Clear cache
import shutil
shutil.rmtree('data/cache', ignore_errors=True)

# Use database instead of memory
data_service = DataService(use_database=True)
```

### 6. Import and Dependency Issues

#### Problem: Module Not Found

```
❌ ModuleNotFoundError: No module named 'src'
```

**Solutions:**

```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
cd /path/to/ai-stock-predictor
python run_analysis.py --ticker RELIANCE
```

#### Problem: Missing Dependencies

```
❌ ImportError: No module named 'mysql.connector'
```

**Solutions:**

```bash
# Install missing dependencies
pip install mysql-connector-python
pip install -r requirements.txt

# For specific packages
pip install yfinance pandas numpy scikit-learn xgboost lightgbm catboost
```

### 7. Configuration Issues

#### Problem: Configuration Not Found

```
❌ Configuration file not found
```

**Solutions:**

```bash
# Check configuration files
ls -la config/
cat config/database_config.py
cat config/angel_one_config.py

# Create missing configurations
cp config/database_config.py.example config/database_config.py
```

#### Problem: Invalid Configuration

```
❌ Invalid configuration value
```

**Solutions:**

```python
# Validate configuration
python -c "
from config.database_config import get_database_config
try:
    config = get_database_config('local')
    print('✅ Configuration valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

## 🔧 Diagnostic Tools

### 1. System Health Check

```bash
# Run comprehensive system check
python scripts/testing/system_health_check.py

# Check specific components
python -c "
from src.core.database_service import DatabaseService
from src.utils.angel_one_data_downloader import AngelOneDataDownloader

# Test database
try:
    db = DatabaseService('mysql', 'mysql://root:7874@localhost/stock_data')
    print('✅ Database: OK')
except Exception as e:
    print(f'❌ Database: {e}')

# Test Angel One
try:
    angel = AngelOneDataDownloader()
    if angel.authenticate():
        print('✅ Angel One: OK')
    else:
        print('❌ Angel One: Authentication failed')
except Exception as e:
    print(f'❌ Angel One: {e}')
"
```

### 2. Data Quality Check

```python
# Check data quality
python -c "
from src.core.data_service import DataService
data_service = DataService()
df = data_service.load_stock_data('RELIANCE')

print('Data Quality Report:')
print(f'Records: {len(df)}')
print(f'Date range: {df.index.min()} to {df.index.max()}')
print(f'Missing values: {df.isnull().sum().sum()}')
print(f'Duplicate dates: {df.index.duplicated().sum()}')
print(f'Price range: {df[\"Close\"].min():.2f} to {df[\"Close\"].max():.2f}')
"
```

### 3. Performance Profiling

```python
# Profile system performance
import time
import psutil

def profile_operation(func, *args, **kwargs):
    start_time = time.time()
    start_memory = psutil.virtual_memory().used

    result = func(*args, **kwargs)

    end_time = time.time()
    end_memory = psutil.virtual_memory().used

    print(f'Time: {end_time - start_time:.2f}s')
    print(f'Memory: {(end_memory - start_memory) / 1024 / 1024:.2f}MB')

    return result

# Usage
from src.core.data_service import DataService
data_service = DataService()
df = profile_operation(data_service.load_stock_data, 'RELIANCE')
```

## 🚀 Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_ticker_date ON stock_data(ticker, date);
CREATE INDEX idx_ticker ON stock_data(ticker);
CREATE INDEX idx_date ON stock_data(date);

-- Analyze table for query optimization
ANALYZE TABLE stock_data;

-- Check query performance
EXPLAIN SELECT * FROM stock_data WHERE ticker = 'RELIANCE' ORDER BY date DESC LIMIT 100;
```

### 2. Memory Optimization

```python
# Optimize memory usage
import gc

# Clear unused variables
del large_dataframe
gc.collect()

# Use chunked processing
def process_large_data(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        # Process chunk
        yield process_chunk(chunk)
```

### 3. Caching Strategy

```python
# Implement smart caching
import pickle
import os
from pathlib import Path

def cache_data(data, cache_key):
    cache_dir = Path('data/cache')
    cache_dir.mkdir(exist_ok=True)

    cache_file = cache_dir / f'{cache_key}.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def load_cached_data(cache_key):
    cache_file = Path('data/cache') / f'{cache_key}.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None
```

## 📋 Maintenance Tasks

### 1. Regular Maintenance

```bash
# Daily tasks
python scripts/database/cleanup_old_data.py
python scripts/validation/validate_data_quality.py

# Weekly tasks
python scripts/database/optimize_database.py
python scripts/testing/run_full_test_suite.py

# Monthly tasks
python scripts/database/backup_database.py
python scripts/analysis/performance_report.py
```

### 2. Log Monitoring

```bash
# Check system logs
tail -f logs/system.log
grep "ERROR" logs/system.log
grep "WARNING" logs/system.log

# Monitor database logs
tail -f /var/log/mysql/error.log
```

### 3. Backup and Recovery

```bash
# Database backup
mysqldump -u root -p7874 stock_data > backup_$(date +%Y%m%d).sql

# Restore from backup
mysql -u root -p7874 stock_data < backup_20240101.sql

# File system backup
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/
```

## 🆘 Emergency Procedures

### 1. System Reset

```bash
# Complete system reset
rm -rf data/cache/*
rm -rf models/cache/*
python scripts/database/reset_database.py
pip install -r requirements.txt --force-reinstall
```

### 2. Data Recovery

```bash
# Recover from backup
mysql -u root -p7874 stock_data < latest_backup.sql
tar -xzf data_backup_latest.tar.gz
```

### 3. Service Restart

```bash
# Restart all services
sudo systemctl restart mysql
sudo systemctl restart redis  # if using Redis
```

---

**💡 Pro Tip**: Always check the logs first when encountering issues. Most problems have clear error messages that point to the root cause.
