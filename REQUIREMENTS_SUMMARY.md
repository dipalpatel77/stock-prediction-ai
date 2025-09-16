# AI Stock Predictor - Requirements Summary

## 📦 Updated Requirements Overview

The AI Stock Predictor system has been updated with comprehensive requirements to support all enhanced features including database integration, multi-exchange data fusion, advanced ML algorithms, and comprehensive reporting.

## 📋 Requirements Files

### 1. **requirements.txt** - Complete Installation

- **Purpose**: Full system installation with all features
- **Size**: ~170 packages
- **Use Case**: Production deployment, complete functionality
- **Features**: All ML algorithms, database support, reporting, validation

### 2. **requirements-minimal.txt** - Lightweight Installation

- **Purpose**: Core functionality only
- **Size**: ~15 packages
- **Use Case**: Quick setup, basic functionality
- **Features**: Core ML, basic data sources, essential utilities

### 3. **requirements-dev.txt** - Development Environment

- **Purpose**: Development and testing tools
- **Size**: ~200+ packages (includes base requirements)
- **Use Case**: Development, testing, code quality
- **Features**: Testing frameworks, code quality tools, documentation

## 🎯 Package Categories

### Core Machine Learning & Data Science

```
tensorflow>=2.10.0          # Deep learning framework
scikit-learn>=1.3.0         # Traditional ML algorithms
pandas>=2.0.0               # Data manipulation
numpy>=1.24.0               # Numerical computing
matplotlib>=3.7.0           # Basic plotting
seaborn>=0.12.0             # Statistical visualization
plotly>=5.15.0              # Interactive plotting
```

### Advanced ML Models

```
xgboost>=1.7.0              # Gradient boosting
lightgbm>=4.0.0             # Light gradient boosting
catboost>=1.2.0             # Categorical boosting
scikit-optimize>=0.9.0      # Hyperparameter optimization
```

### Financial Data & Market Data

```
yfinance>=0.2.18            # Yahoo Finance data
alpha-vantage>=2.3.1        # Alpha Vantage API
fredapi>=0.5.0              # Federal Reserve Economic Data
pandas-datareader>=0.10.0   # Multiple data sources
```

### Database & Storage

```
mysql-connector-python>=8.1.0  # MySQL database
sqlalchemy>=2.0.0              # Database ORM
psycopg2-binary>=2.9.0         # PostgreSQL support
pymongo>=4.4.0                 # MongoDB support
joblib>=1.3.0                  # Model persistence
h5py>=3.9.0                    # HDF5 file format
```

### API Integration & Web Services

```
requests>=2.31.0            # HTTP requests
beautifulsoup4>=4.12.0      # Web scraping
newsapi-python>=0.2.6       # News API
pyotp>=2.8.0                # TOTP for Angel One
websocket-client>=1.6.0     # WebSocket connections
```

### Natural Language Processing

```
nltk>=3.8.1                 # Natural language toolkit
textblob>=0.17.1            # Text processing
vaderSentiment>=3.3.2       # Sentiment analysis
transformers>=4.30.0        # Transformer models
torch>=2.0.0                # PyTorch framework
```

### Technical Analysis

```
ta>=0.10.2                  # Technical analysis library
talib-binary>=0.4.25        # TA-Lib binary
TA-Lib>=0.4.25              # Technical Analysis Library
```

### Reporting & Visualization

```
jinja2>=3.1.0               # Template engine
weasyprint>=59.0            # HTML to PDF
reportlab>=4.0.0            # PDF generation
openpyxl>=3.1.0             # Excel file handling
xlsxwriter>=3.1.0           # Excel writing
```

### Testing & Validation

```
pytest>=7.4.0               # Testing framework
pytest-cov>=4.1.0           # Coverage reporting
pytest-mock>=3.11.0         # Mocking utilities
unittest-xml-reporting>=3.2.0  # XML test reports
```

### Logging & Monitoring

```
loguru>=0.7.0               # Advanced logging
structlog>=23.1.0           # Structured logging
prometheus-client>=0.17.0   # Metrics collection
```

### Security & Authentication

```
cryptography>=41.0.0        # Cryptographic functions
pyjwt>=2.8.0                # JSON Web Tokens
bcrypt>=4.0.0               # Password hashing
```

## 🚀 Installation Options

### Option 1: Complete Installation

```bash
# Install all features
pip install -r requirements.txt
```

### Option 2: Minimal Installation

```bash
# Install core features only
pip install -r requirements-minimal.txt
```

### Option 3: Development Installation

```bash
# Install with development tools
pip install -r requirements-dev.txt
```

### Option 4: Selective Installation

```bash
# Install specific categories
pip install scikit-learn pandas numpy matplotlib  # Core ML
pip install mysql-connector-python sqlalchemy     # Database
pip install yfinance requests pyotp               # APIs
```

## 🔧 Platform-Specific Requirements

### Windows

- **TA-Lib**: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- **Visual C++**: Required for some packages
- **MySQL**: Use MySQL Installer

### macOS

- **TA-Lib**: `brew install ta-lib`
- **MySQL**: `brew install mysql`
- **Xcode**: Required for some compilations

### Linux (Ubuntu/Debian)

- **TA-Lib**: Compile from source
- **MySQL**: `sudo apt install mysql-server`
- **Build tools**: `sudo apt install build-essential`

## 📊 System Requirements

### Minimum Requirements

- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 10GB
- **OS**: Windows 10, macOS 10.15, Ubuntu 20.04

### Recommended Requirements

- **Python**: 3.10+
- **RAM**: 16GB+
- **Storage**: 50GB+
- **GPU**: CUDA-compatible (optional)
- **OS**: Latest versions

## 🎯 Feature Dependencies

### Core Features (Always Required)

- `pandas`, `numpy`, `scikit-learn`
- `yfinance`, `requests`
- `mysql-connector-python`

### Database Integration

- `mysql-connector-python`
- `sqlalchemy`
- `psycopg2-binary` (PostgreSQL)
- `pymongo` (MongoDB)

### Multi-Exchange Data Fusion

- `pandas`, `numpy`
- `requests`, `pyotp`
- `mysql-connector-python`

### Advanced ML Algorithms

- `xgboost`, `lightgbm`, `catboost`
- `tensorflow`, `torch`
- `scikit-optimize`

### Technical Analysis

- `ta`, `TA-Lib`
- `talib-binary`

### Reporting & Visualization

- `matplotlib`, `seaborn`, `plotly`
- `jinja2`, `weasyprint`
- `openpyxl`, `xlsxwriter`

### Natural Language Processing

- `nltk`, `textblob`
- `vaderSentiment`
- `transformers`, `torch`

## 🔍 Verification Commands

### Check Core Packages

```bash
python -c "import pandas, numpy, sklearn, yfinance, requests; print('✅ Core packages OK')"
```

### Check Database Support

```bash
python -c "import mysql.connector, sqlalchemy; print('✅ Database packages OK')"
```

### Check ML Packages

```bash
python -c "import xgboost, lightgbm, catboost; print('✅ Advanced ML packages OK')"
```

### Check Technical Analysis

```bash
python -c "import talib, ta; print('✅ Technical analysis packages OK')"
```

## 🚨 Common Issues & Solutions

### TA-Lib Installation Issues

```bash
# Windows: Download wheel file
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl

# macOS: Use Homebrew
brew install ta-lib
pip install TA-Lib

# Linux: Compile from source
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make && sudo make install
pip install TA-Lib
```

### Memory Issues

```bash
# Use minimal requirements for low-memory systems
pip install -r requirements-minimal.txt
```

### GPU Support

```bash
# Install CUDA toolkit first, then:
pip install tensorflow-gpu torch-gpu
```

## 📈 Performance Considerations

### Memory Usage

- **Minimal**: ~2GB RAM
- **Standard**: ~8GB RAM
- **Full**: ~16GB+ RAM

### Storage Requirements

- **Minimal**: ~2GB
- **Standard**: ~10GB
- **Full**: ~50GB+ (with models and data)

### Installation Time

- **Minimal**: ~5 minutes
- **Standard**: ~15 minutes
- **Full**: ~30+ minutes

---

This requirements summary provides a comprehensive overview of all dependencies needed for the enhanced AI Stock Predictor system. Choose the installation option that best fits your needs and system capabilities.
