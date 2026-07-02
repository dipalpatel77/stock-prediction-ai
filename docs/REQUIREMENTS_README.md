# Requirements Files Guide

This project provides multiple requirements files for different use cases:

## 📦 Available Requirements Files

### 1. `requirements.txt` - **Complete Installation**

- **Use for**: Full development environment with all features
- **Includes**: All ML libraries, visualization tools, testing frameworks
- **Size**: ~2GB installation
- **Best for**: Developers, researchers, full-featured deployments

### 2. `requirements-minimal.txt` - **Lightweight Installation**

- **Use for**: Basic functionality only
- **Includes**: Core ML libraries, essential financial data tools
- **Size**: ~500MB installation
- **Best for**: Quick setup, limited resources, basic predictions

### 3. `requirements-production.txt` - **Production Deployment**

- **Use for**: Production servers and cloud deployments
- **Includes**: Production-tested versions with security focus
- **Size**: ~1.5GB installation
- **Best for**: Live trading systems, production APIs

### 4. `requirements-dev.txt` - **Development Environment**

- **Use for**: Full development setup with testing and debugging tools
- **Includes**: All production requirements + development tools
- **Size**: ~3GB installation
- **Best for**: Contributors, advanced development

## 🚀 Quick Start

### For Basic Usage:

```bash
pip install -r requirements-minimal.txt
```

### For Full Development:

```bash
pip install -r requirements.txt
```

### For Production:

```bash
pip install -r requirements-production.txt
```

### For Development:

```bash
pip install -r requirements-dev.txt
```

## 📋 Core Dependencies Explained

### Essential Libraries:

- **scikit-learn**: Core ML algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **yfinance**: Stock data fetching
- **ta**: Technical analysis indicators

### Advanced ML:

- **xgboost**: Gradient boosting
- **lightgbm**: Fast gradient boosting
- **catboost**: Categorical boosting
- **tensorflow**: Deep learning

### Database:

- **mysql-connector-python**: MySQL connectivity
- **sqlalchemy**: Database ORM

### API Integration:

- **requests**: HTTP requests
- **pyotp**: TOTP authentication
- **aiohttp**: Async HTTP

## 🔧 Installation Tips

### 1. Virtual Environment (Recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. Conda Environment:

```bash
conda create -n ai-stock-predictor python=3.9
conda activate ai-stock-predictor
pip install -r requirements.txt
```

### 3. Docker (Production):

```dockerfile
FROM python:3.9-slim
COPY requirements-production.txt .
RUN pip install -r requirements-production.txt
```

## ⚠️ Common Issues

### 1. TensorFlow Installation:

```bash
# For CPU-only (recommended for most users)
pip install tensorflow-cpu

# For GPU support
pip install tensorflow-gpu
```

### 2. MySQL Connector Issues:

```bash
# Alternative MySQL connector
pip install PyMySQL
```

### 3. Memory Issues:

- Use `requirements-minimal.txt` for limited memory
- Consider using `--no-cache-dir` flag with pip

## 🔍 Version Compatibility

- **Python**: 3.8+ (recommended 3.9+)
- **Operating System**: Windows, Linux, macOS
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: 2GB+ free space

## 📊 Performance Optimization

### For Production:

1. Use `requirements-production.txt`
2. Pin specific versions
3. Use `--no-cache-dir` for smaller images
4. Consider using Alpine Linux base images

### For Development:

1. Use `requirements-dev.txt`
2. Include debugging tools
3. Add profiling libraries

## 🛡️ Security Considerations

- All production requirements include security scanning
- Use `safety` package to check for vulnerabilities
- Regular updates recommended
- Pin versions in production

## 📈 Monitoring Dependencies

The system includes monitoring tools:

- **loguru**: Advanced logging
- **structlog**: Structured logging
- **prometheus-client**: Metrics collection

## 🔄 Updating Requirements

1. **Test new versions** in development first
2. **Update requirements files** with new versions
3. **Test in staging** environment
4. **Deploy to production** after validation

## 📞 Support

For dependency issues:

1. Check Python version compatibility
2. Verify operating system support
3. Review error messages carefully
4. Consider using virtual environments
