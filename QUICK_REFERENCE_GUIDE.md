# 🚀 QUICK REFERENCE GUIDE - CHOOSING THE RIGHT ANALYSIS TOOL

## 📊 **WHICH TOOL SHOULD I USE?**

### **🎯 QUICK DECISION TREE:**

```
What do you want to analyze?
├── Quick stock prediction?
│   ├── Simple analysis → run_stock_prediction.py AAPL simple
│   └── Advanced ML → run_stock_prediction.py AAPL advanced
├── Multi-timeframe analysis?
│   └── Short/Mid/Long term → enhanced_analysis_runner.py AAPL enhanced
└── Complete analysis with all modules?
    └── Full pipeline → unified_analysis_pipeline.py AAPL
```

---

## 🔧 **DETAILED USAGE GUIDE**

### **1. QUICK PREDICTIONS** (`run_stock_prediction.py`)

#### **Use When:**

- ✅ You want fast predictions
- ✅ You need real-time analysis
- ✅ You prefer advanced ML models
- ✅ You want model caching for speed

#### **Command Examples:**

```bash
# Simple prediction (fast)
python run_stock_prediction.py AAPL simple 5

# Advanced prediction (comprehensive)
python run_stock_prediction.py AAPL advanced 10

# Non-interactive mode
python run_stock_prediction.py MSFT advanced 7 --non-interactive
```

#### **Output:**

- Current price and predictions
- Model performance metrics
- Trading recommendations
- Confidence intervals

#### **Best For:**

- Day traders
- Quick decision making
- Real-time analysis
- Advanced ML enthusiasts

---

### **2. MULTI-TIMEFRAME ANALYSIS** (`enhanced_analysis_runner.py`)

#### **Use When:**

- ✅ You need short-term, mid-term, and long-term analysis
- ✅ You want specialized analysis for each timeframe
- ✅ You need comprehensive price forecasting
- ✅ You want confidence intervals and risk assessment

#### **Command Examples:**

```bash
# Enhanced multi-timeframe analysis
python enhanced_analysis_runner.py AAPL enhanced

# Basic multi-timeframe analysis
python enhanced_analysis_runner.py MSFT basic
```

#### **Output:**

- Short-term predictions (1-7 days)
- Mid-term predictions (1-4 weeks)
- Long-term predictions (1-12 months)
- Risk metrics and confidence levels
- Trading signals for each timeframe

#### **Best For:**

- Swing traders
- Position traders
- Long-term investors
- Portfolio managers

---

### **3. COMPLETE ANALYSIS PIPELINE** (`unified_analysis_pipeline.py`)

#### **Use When:**

- ✅ You want the most comprehensive analysis
- ✅ You need balance sheet analysis
- ✅ You want company event impact analysis
- ✅ You need economic indicators and market factors
- ✅ You want all project modules integrated

#### **Command Examples:**

```bash
# Complete analysis with all modules
python unified_analysis_pipeline.py AAPL

# Analysis with specific period
python unified_analysis_pipeline.py MSFT --period 5y --days-ahead 10
```

#### **Output:**

- All technical indicators
- Sentiment analysis
- Market factors
- Economic indicators
- Balance sheet analysis
- Company event impact
- Trading strategies
- Backtesting results

#### **Best For:**

- Fundamental analysts
- Research analysts
- Institutional investors
- Complete market analysis

---

## 📈 **COMPARISON MATRIX**

| Feature               | run_stock_prediction.py | enhanced_analysis_runner.py | unified_analysis_pipeline.py |
| --------------------- | ----------------------- | --------------------------- | ---------------------------- |
| **Speed**             | ⚡⚡⚡⚡⚡              | ⚡⚡⚡⚡                    | ⚡⚡⚡                       |
| **Comprehensiveness** | ⚡⚡⚡                  | ⚡⚡⚡⚡                    | ⚡⚡⚡⚡⚡                   |
| **ML Models**         | ⚡⚡⚡⚡⚡              | ⚡⚡⚡                      | ⚡⚡⚡⚡                     |
| **Timeframes**        | Single                  | Multiple                    | Single                       |
| **Real-time**         | ✅                      | ❌                          | ❌                           |
| **Balance Sheet**     | ❌                      | ❌                          | ✅                           |
| **Economic Factors**  | ❌                      | ❌                          | ✅                           |
| **Company Events**    | ❌                      | ❌                          | ✅                           |
| **Model Caching**     | ✅                      | ❌                          | ❌                           |

---

## 🎯 **RECOMMENDED USE CASES**

### **For Day Trading:**

```bash
python run_stock_prediction.py AAPL advanced 1
```

**Why:** Fast, real-time, advanced ML models

### **For Swing Trading:**

```bash
python enhanced_analysis_runner.py AAPL enhanced
```

**Why:** Multi-timeframe analysis, risk assessment

### **For Long-term Investing:**

```bash
python unified_analysis_pipeline.py AAPL
```

**Why:** Complete fundamental and technical analysis

### **For Research:**

```bash
python unified_analysis_pipeline.py AAPL
```

**Why:** All available data and analysis modules

### **For Quick Decisions:**

```bash
python run_stock_prediction.py AAPL simple 5
```

**Why:** Fast, simple, reliable predictions

---

## ⚡ **PERFORMANCE EXPECTATIONS**

### **Execution Times (Approximate):**

| Tool                           | Simple Mode   | Enhanced Mode | Complete Mode |
| ------------------------------ | ------------- | ------------- | ------------- |
| `run_stock_prediction.py`      | 30-60 seconds | 2-5 minutes   | N/A           |
| `enhanced_analysis_runner.py`  | 5-10 minutes  | 10-20 minutes | N/A           |
| `unified_analysis_pipeline.py` | N/A           | N/A           | 15-30 minutes |

### **Resource Usage:**

- **CPU:** All tools use multi-threading
- **Memory:** 2-4 GB RAM recommended
- **Storage:** 100-500 MB per analysis
- **Network:** Internet required for data download

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues:**

#### **1. "Module not found" errors:**

```bash
# Install missing packages
pip install xgboost lightgbm catboost
```

#### **2. "Timeout" errors:**

```bash
# Use simple mode for faster execution
python run_stock_prediction.py AAPL simple 5
```

#### **3. "No data available" errors:**

```bash
# Check internet connection
# Try different ticker symbol
python run_stock_prediction.py MSFT simple 5
```

#### **4. "Memory" errors:**

```bash
# Close other applications
# Use simple mode
# Reduce prediction days
```

---

## 📞 **GETTING HELP**

### **For Quick Questions:**

- Check the output messages
- Review generated CSV files
- Read the console output

### **For Technical Issues:**

- Check `logs/` directory for error logs
- Verify Python packages are installed
- Ensure internet connection is stable

### **For Best Results:**

- Use the right tool for your needs
- Start with simple mode
- Gradually increase complexity
- Review all generated files

---

## 🎉 **SUCCESS TIPS**

1. **Start Simple:** Begin with `run_stock_prediction.py` simple mode
2. **Graduate Up:** Move to advanced modes as you get comfortable
3. **Review Outputs:** Check all generated CSV files for insights
4. **Compare Results:** Use multiple tools for validation
5. **Keep Learning:** Each tool provides different insights

**Happy Trading! 📈**
