# AI Stock Predictor - Installation Guide

## 🚀 Complete Installation Guide

This guide provides step-by-step instructions for installing the AI Stock Predictor system with all its enhanced features.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher (recommended: Python 3.10+)
- **Memory**: Minimum 8GB RAM (recommended: 16GB+)
- **Storage**: Minimum 10GB free space
- **Internet**: Stable internet connection for data downloads

### Required Software

- **MySQL**: 8.0 or higher (for database storage)
- **Git**: For version control
- **pip**: Python package manager

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone <repository-url>
cd ai-stock-predictor

# Verify Python version
python --version
# Should be 3.8 or higher
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Verify activation
which python  # Should point to venv directory
```

### Step 3: Install Core Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install core requirements
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn; print('Core packages installed successfully')"
```

### Step 4: Install Database Dependencies

#### MySQL Installation

**Windows:**

```bash
# Download MySQL Installer from https://dev.mysql.com/downloads/installer/
# Install MySQL Server 8.0+
# Set root password (remember this for configuration)
```

**macOS:**

```bash
# Using Homebrew
brew install mysql
brew services start mysql

# Set root password
mysql_secure_installation
```

**Linux (Ubuntu/Debian):**

```bash
# Install MySQL
sudo apt update
sudo apt install mysql-server

# Secure installation
sudo mysql_secure_installation
```

#### PostgreSQL (Optional)

```bash
# Install PostgreSQL if needed
pip install psycopg2-binary
```

### Step 5: Install Technical Analysis Dependencies

#### TA-Lib Installation

**Windows:**

```bash
# Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Install the appropriate wheel file
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl
```

**macOS:**

```bash
# Install TA-Lib using Homebrew
brew install ta-lib
pip install TA-Lib
```

**Linux:**

```bash
# Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### Step 6: Install GPU Support (Optional)

#### CUDA Installation (for GPU acceleration)

**Check GPU compatibility:**

```bash
# Check if you have CUDA-compatible GPU
nvidia-smi
```

**Install CUDA Toolkit:**

```bash
# Download from https://developer.nvidia.com/cuda-downloads
# Install CUDA Toolkit 11.8 or 12.0+

# Install GPU-enabled packages
pip install tensorflow-gpu>=2.10.0
pip install torch-gpu>=2.0.0
```

### Step 7: Configure Database

```bash
# Create database
mysql -u root -p
```

```sql
-- In MySQL console
CREATE DATABASE stock_data;
CREATE USER 'stock_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON stock_data.* TO 'stock_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

### Step 8: Set Up Configuration

```bash
# Create configuration file
cp config/database_config.py.example config/database_config.py

# Edit database configuration
# Update with your MySQL credentials
```

### Step 9: Initialize Database Schema

```bash
# Set up database tables
python scripts/database/setup_mysql_database.py

# Test database connection
python scripts/database/test_database_connection.py
```

### Step 10: Configure Angel One API

```bash
# Create Angel One configuration
cp config/angel_one_config.py.example config/angel_one_config.py

# Edit with your Angel One credentials
# - API Key
# - Client ID
# - PIN
# - TOTP Secret
```

### Step 11: Test Installation

```bash
# Run system health check
python scripts/testing/system_health_check.py

# Test Angel One connectivity
python test_angel_database_integration.py

# Run sample analysis
python run_analysis.py --ticker RELIANCE --period 1mo
```

## 🔧 Optional Components

### Jupyter Notebook Support

```bash
# Install Jupyter for interactive analysis
pip install jupyter ipywidgets

# Start Jupyter
jupyter notebook
```

### Web Dashboard (Streamlit)

```bash
# Install Streamlit for web interface
pip install streamlit

# Run web dashboard
streamlit run dashboard.py
```

### API Server (FastAPI)

```bash
# Install FastAPI for REST API
pip install fastapi uvicorn

# Run API server
uvicorn api.main:app --reload
```

## 🚨 Troubleshooting

### Common Installation Issues

#### 1. TA-Lib Installation Failed

**Error**: `Microsoft Visual C++ 14.0 is required`

**Solution**:

```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use conda instead
conda install -c conda-forge ta-lib
```

#### 2. MySQL Connection Issues

**Error**: `Access denied for user 'root'@'localhost'`

**Solution**:

```bash
# Reset MySQL root password
sudo mysql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'new_password';
FLUSH PRIVILEGES;
```

#### 3. Memory Issues

**Error**: `MemoryError` during model training

**Solution**:

```python
# Reduce data period
data_service = DataService(period_config="quick_check")

# Use database instead of cache
data_service = DataService(use_database=True)

# Reduce worker count
pipeline = UnifiedAnalysisPipeline(max_workers=2)
```

#### 4. Angel One Authentication Issues

**Error**: `Authentication failed`

**Solution**:

```bash
# Check credentials
python -c "from src.utils.angel_one_data_downloader import AngelOneDataDownloader; AngelOneDataDownloader().authenticate()"

# Verify TOTP secret
python -c "import pyotp; print(pyotp.TOTP('your_totp_secret').now())"
```

### Performance Optimization

#### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_ticker_date ON stock_data(ticker, date);
CREATE INDEX idx_ticker ON stock_data(ticker);
CREATE INDEX idx_date ON stock_data(date);
```

#### 2. Memory Optimization

```python
# Use chunked processing for large datasets
def process_large_dataset(data, chunk_size=1000):
    for chunk in pd.read_csv(data, chunksize=chunk_size):
        process_chunk(chunk)
```

#### 3. GPU Optimization

```python
# Check GPU availability
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## 📊 Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All requirements installed successfully
- [ ] MySQL database running and accessible
- [ ] Database schema created
- [ ] Angel One API credentials configured
- [ ] TA-Lib installed and working
- [ ] System health check passed
- [ ] Sample analysis completed successfully

## 🎯 Quick Start After Installation

```bash
# 1. Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# 2. Run sample analysis
python run_analysis.py --ticker RELIANCE --period 1mo

# 3. Test interactive mode
python enhanced_unified_pipeline.py

# 4. Check database integration
python test_angel_database_integration.py
```

## 📚 Additional Resources

- **Documentation**: `docs/` directory
- **Configuration**: `config/` directory
- **Scripts**: `scripts/` directory
- **Examples**: `examples/` directory

## 🆘 Support

If you encounter issues during installation:

1. Check the troubleshooting section above
2. Review the logs in `logs/` directory
3. Run system health check: `python scripts/testing/system_health_check.py`
4. Check database connectivity: `python scripts/database/test_database_connection.py`

---

**Note**: This installation guide covers the complete setup for the enhanced AI Stock Predictor system. Follow the steps in order for the best results.
