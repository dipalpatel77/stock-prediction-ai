# 🚀 **HOW TO RUN THE AI STOCK PREDICTOR PROJECT**

## **📋 PREREQUISITES**

### **1. Python Requirements**

- **Python 3.8+** (Recommended: Python 3.10+)
- **pip** package manager

### **2. Required Dependencies**

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install yfinance requests beautifulsoup4 nltk
pip install sqlalchemy pymysql psycopg2-binary pymongo
pip install aiohttp websockets asyncio
pip install joblib xgboost lightgbm
pip install redis aioredis aiofiles
pip install psutil
```

### **3. Optional Dependencies (for advanced features)**

```bash
pip install tensorflow torch
pip install jupyter notebook
pip install streamlit dash
```

---

## **🚀 RUNNING THE PROJECT**

### **Method 1: Interactive Mode (Recommended for beginners)**

```bash
# Navigate to project directory
cd D:\TradingProjcet\ai-stock-predictor

# Run interactive mode
python main/main.py
```

**What happens:**

- Interactive prompts will guide you through the setup
- You'll be asked to enter stock ticker, analysis type, etc.
- Perfect for first-time users

### **Method 2: Quick Analysis Mode**

```bash
# Quick analysis for a single stock
python main/main.py --quick AAPL

# Quick analysis with custom period
python main/main.py --quick AAPL 1y

# Quick analysis for Indian stocks
python main/main.py --quick RELIANCE
```

**Examples:**

```bash
# US Stocks
python main/main.py --quick AAPL 6m
python main/main.py --quick MSFT 1y
python main/main.py --quick GOOGL 2y

# Indian Stocks (with Angel One integration)
python main/main.py --quick RELIANCE
python main/main.py --quick TCS
python main/main.py --quick INFY
```

### **Method 3: Batch Analysis Mode**

```bash
# Analyze multiple stocks at once
python main/main.py --batch AAPL,MSFT,GOOGL

# Batch analysis with custom period
python main/main.py --batch AAPL,MSFT,GOOGL 1y
```

**Examples:**

```bash
# US Stocks batch
python main/main.py --batch AAPL,MSFT,GOOGL,AMZN,TSLA

# Indian Stocks batch
python main/main.py --batch RELIANCE,TCS,INFY,HDFC,ICICIBANK
```

### **Method 4: Help and Options**

```bash
# Show help
python main/main.py --help
```

---

## **⚙️ CONFIGURATION OPTIONS**

### **1. Environment Variables (Optional)**

Create a `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=sqlite:///stock_data.db
# DATABASE_URL=mysql://user:password@localhost/stock_data
# DATABASE_URL=postgresql://user:password@localhost/stock_data

# Angel One API (for Indian stocks)
ANGEL_ONE_API_KEY=your_api_key
ANGEL_ONE_CLIENT_CODE=your_client_code
ANGEL_ONE_TOTP_SECRET=your_totp_secret

# Redis Configuration (for advanced caching)
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
```

### **2. Configuration File**

Create `config.json` in the project root:

```json
{
  "database": {
    "url": "sqlite:///stock_data.db",
    "pool_size": 10
  },
  "angel_one": {
    "api_key": "your_api_key",
    "client_code": "your_client_code",
    "totp_secret": "your_totp_secret"
  },
  "analysis": {
    "default_period": "1y",
    "use_enhanced_features": true,
    "enable_caching": true
  }
}
```

---

## **📊 ANALYSIS TYPES**

### **1. Comprehensive Analysis (Default)**

- Full data processing
- Multiple ML models
- Technical indicators
- Strategy analysis
- Predictions

### **2. Quick Analysis**

- Basic data processing
- Single ML model
- Essential indicators
- Fast execution

### **3. Interactive Analysis**

- Step-by-step guidance
- Custom configuration
- Real-time feedback
- Detailed results

---

## **🇮🇳 INDIAN STOCK SUPPORT**

### **Angel One Integration**

For Indian stocks, the system automatically uses Angel One API:

**Supported Indian Stocks:**

- RELIANCE, TCS, INFY, HDFC, ICICIBANK
- WIPRO, BHARTIARTL, ITC, SBIN, KOTAKBANK
- And many more...

**Configuration:**

```python
# The system will prompt for Angel One credentials
# Or you can set them in environment variables
```

---

## **🗄️ DATABASE SETUP**

### **1. SQLite (Default)**

No setup required - automatically creates database file.

### **2. MySQL**

```bash
# Install MySQL
# Create database
mysql -u root -p
CREATE DATABASE stock_data;
```

### **3. PostgreSQL**

```bash
# Install PostgreSQL
# Create database
createdb stock_data
```

### **4. MongoDB**

```bash
# Install MongoDB
# Start MongoDB service
mongod
```

---

## **📈 ADVANCED FEATURES**

### **1. Real-time Monitoring Dashboard**

```bash
# Start monitoring dashboard
python -c "
from main.services.monitoring_dashboard import MonitoringDashboard
import asyncio

async def start_dashboard():
    dashboard = MonitoringDashboard()
    await dashboard.start_dashboard()

asyncio.run(start_dashboard())
"
```

### **2. Auto-scaling Service**

```bash
# Start auto-scaling
python -c "
from main.services.auto_scaler import AutoScaler
import asyncio

async def start_scaling():
    scaler = AutoScaler()
    await scaler.start_auto_scaling()

asyncio.run(start_scaling())
"
```

### **3. Advanced Caching**

```bash
# Start Redis for advanced caching
redis-server

# Or use in-memory caching (default)
```

---

## **🔧 TROUBLESHOOTING**

### **Common Issues:**

#### **1. Import Errors**

```bash
# Make sure you're in the project directory
cd D:\TradingProjcet\ai-stock-predictor

# Install missing dependencies
pip install -r requirements.txt
```

#### **2. Database Connection Issues**

```bash
# For SQLite (default) - no setup needed
# For MySQL/PostgreSQL - check connection strings
# For MongoDB - ensure MongoDB is running
```

#### **3. Angel One API Issues**

```bash
# Check API credentials
# Ensure TOTP is working
# Check network connectivity
```

#### **4. Memory Issues**

```bash
# Reduce data size
python main/main.py --quick AAPL 3m

# Or modify memory settings in config
```

---

## **📊 EXAMPLE USAGE**

### **1. First-time User**

```bash
# Start with interactive mode
python main/main.py

# Follow the prompts:
# - Enter stock ticker: AAPL
# - Select analysis type: comprehensive
# - Configure Angel One (if needed)
# - Wait for results
```

### **2. Experienced User**

```bash
# Quick analysis
python main/main.py --quick AAPL 1y

# Batch analysis
python main/main.py --batch AAPL,MSFT,GOOGL 6m

# Indian stocks
python main/main.py --quick RELIANCE
```

### **3. Advanced User**

```bash
# Custom configuration
python -c "
from main.main import run_quick_analysis
result = run_quick_analysis('AAPL', '1y', use_enhanced=True)
print(result)
"
```

---

## **📈 EXPECTED OUTPUT**

### **Successful Run:**

```
🚀 Unified AI Stock Predictor - Polylithic Architecture
======================================================================
Started at: 2025-09-19 13:30:00

📋 Getting user inputs...
✅ User inputs received for ticker: AAPL

📊 Configuration Summary:
----------------------------------------
📈 Ticker: AAPL
🇮🇳 Indian Stock: False
📊 Data Source: Yahoo Finance
🔍 Analysis Type: comprehensive
⚡ Enhanced Features: True
🗄️ Database: True

🔧 Initializing pipeline for AAPL...
✅ Pipeline initialized successfully

🚀 Starting analysis...
🔄 Running comprehensive analysis...

======================================================================
✅ Analysis completed successfully!
⏱️ Execution time: 45.23 seconds

📊 Analysis Results:
----------------------------------------
📈 Data Processing: ✅ 252 records
   Data Source: Yahoo Finance
   Quality Score: 95.2%

🤖 Model Training: ✅ 7 models
   Best Model: random_forest
   Accuracy: 0.847

📊 Strategy Analysis: ✅ 5 components
   Sentiment: 0.72
   Risk Level: Medium

🔮 Predictions: ✅ Generated
   Short-term: $185.50
   Mid-term: $192.30
   Long-term: $205.80
```

---

## **🎯 QUICK START GUIDE**

### **For Beginners:**

1. Open terminal/command prompt
2. Navigate to project directory: `cd D:\TradingProjcet\ai-stock-predictor`
3. Run: `python main/main.py`
4. Follow the interactive prompts
5. Wait for results

### **For Advanced Users:**

1. Install dependencies: `pip install -r requirements.txt`
2. Configure database (optional)
3. Run: `python main/main.py --quick AAPL`
4. Check results

### **For Indian Stock Users:**

1. Get Angel One API credentials
2. Run: `python main/main.py --quick RELIANCE`
3. Enter Angel One credentials when prompted
4. Wait for results

---

## **🚀 READY TO RUN!**

The AI Stock Predictor is now ready to use! Choose your preferred method and start analyzing stocks with advanced AI capabilities.

**Happy Trading! 📈🚀**
