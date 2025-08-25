# 📁 FILE REFERENCE GUIDE - Project Structure & Dependencies

## 📅 Created: August 25, 2025

## 🎯 Purpose: Prevent accidental file deletions by documenting importance and dependencies

---

## 🚨 **CRITICAL FILES - NEVER DELETE**

### **Core System Files**

#### **1. `run_stock_prediction.py`** ⭐⭐⭐⭐⭐

- **Purpose**: Main prediction system - consolidated prediction engine
- **Importance**: CRITICAL - Primary system for stock predictions
- **Functionality**:
  - Data loading with multiple fallbacks
  - Feature engineering and model training
  - Prediction generation (simple & advanced modes)
  - Non-interactive CLI support
  - Model caching and optimization
- **Dependencies**: All ML libraries, data files, configuration
- **Status**: ✅ **PRESERVE** - Main system file

#### **2. `unified_analysis_pipeline.py`** ⭐⭐⭐⭐⭐

- **Purpose**: Data generation and enhanced analysis pipeline
- **Importance**: CRITICAL - Generates enhanced data files for predictions
- **Functionality**:
  - Generates `{TICKER}_partA_partC_enhanced.csv` files
  - Complete analysis with backtesting
  - Enhanced feature engineering
  - Market factor analysis
- **Dependencies**: partA, partB, partC modules
- **Status**: ✅ **PRESERVE** - Different from prediction system
- **Note**: Referenced in README.md and RUNNING_INSTRUCTIONS.md

#### **3. `angel_one_config.py`** ⭐⭐⭐⭐⭐

- **Purpose**: Angel One API configuration and setup
- **Importance**: CRITICAL - API access for Indian stock data
- **Functionality**:
  - API key management
  - Stock symbol mapping
  - Configuration validation
- **Dependencies**: .env file, Angel One API
- **Status**: ✅ **PRESERVE** - Essential for data access

#### **4. `angel_one_data_downloader.py`** ⭐⭐⭐⭐⭐

- **Purpose**: Downloads stock data from Angel One API
- **Importance**: CRITICAL - Primary data source for Indian stocks
- **Functionality**:
  - Real-time data download
  - Historical data retrieval
  - Data formatting and storage
- **Dependencies**: angel_one_config.py, .env file
- **Status**: ✅ **PRESERVE** - Essential for data acquisition

---

## 📚 **DOCUMENTATION FILES - PRESERVE**

### **User Documentation**

#### **5. `README.md`** ⭐⭐⭐⭐⭐

- **Purpose**: Project overview and main documentation
- **Importance**: CRITICAL - First point of reference for users
- **Content**: Project description, setup, usage examples
- **Status**: ✅ **PRESERVE** - Primary documentation

#### **6. `RUNNING_INSTRUCTIONS.md`** ⭐⭐⭐⭐⭐

- **Purpose**: Detailed instructions for running the system
- **Importance**: CRITICAL - Step-by-step user guide
- **Content**: Installation, usage, troubleshooting
- **Status**: ✅ **PRESERVE** - Essential user guide

#### **7. `ANGEL_ONE_SETUP.md`** ⭐⭐⭐⭐⭐

- **Purpose**: Angel One API setup instructions
- **Importance**: CRITICAL - Required for data access
- **Content**: API registration, configuration, troubleshooting
- **Status**: ✅ **PRESERVE** - Required for setup

#### **8. `IMPROVEMENTS_SUMMARY.md`** ⭐⭐⭐⭐

- **Purpose**: Documents recent system improvements
- **Importance**: HIGH - Tracks system evolution
- **Content**: Recent enhancements, optimizations, fixes
- **Status**: ✅ **PRESERVE** - Important for maintenance

#### **9. `DELETION_ANALYSIS.md`** ⭐⭐⭐⭐

- **Purpose**: Analysis of deleted files and cleanup history
- **Importance**: HIGH - Prevents repeat mistakes
- **Content**: What was deleted, why, lessons learned
- **Status**: ✅ **PRESERVE** - Important reference

#### **10. `CLEANUP_PLAN.md`** ⭐⭐⭐⭐

- **Purpose**: Planned cleanup strategy and execution
- **Importance**: HIGH - Guides future cleanup efforts
- **Content**: Safe deletion targets, preservation strategy
- **Status**: ✅ **PRESERVE** - Important for maintenance

---

## 🧪 **TESTING FILES - PRESERVE**

### **Test Suite**

#### **11. `tests/`** ⭐⭐⭐⭐⭐

- **Purpose**: Comprehensive test suite for system improvements
- **Importance**: CRITICAL - Validates system functionality
- **Functionality**:
  - Tests non-interactive mode
  - Tests data loading fallbacks
  - Tests model caching
  - Tests terminal interaction
  - Tests performance optimization
- **Dependencies**: run_stock_prediction.py
- **Status**: ✅ **PRESERVE** - Essential for validation

#### **12. `tests/run_tests.bat`** ⭐⭐⭐⭐

- **Purpose**: Windows batch script to run tests
- **Importance**: HIGH - Easy testing on Windows
- **Functionality**: Activates venv, runs test suite
- **Dependencies**: tests/run_all_tests.py, venv
- **Status**: ✅ **PRESERVE** - Convenient testing tool

---

## 📦 **CONFIGURATION FILES - PRESERVE**

### **Project Configuration**

#### **13. `requirements.txt`** ⭐⭐⭐⭐⭐

- **Purpose**: Python package dependencies
- **Importance**: CRITICAL - Required for installation
- **Content**: All required Python packages and versions
- **Status**: ✅ **PRESERVE** - Essential for setup

#### **14. `.env`** ⭐⭐⭐⭐⭐

- **Purpose**: Environment variables and API keys
- **Importance**: CRITICAL - Contains sensitive configuration
- **Content**: API keys, tokens, configuration settings
- **Status**: ✅ **PRESERVE** - Contains sensitive data
- **Note**: Should be in .gitignore

#### **15. `.gitignore`** ⭐⭐⭐⭐

- **Purpose**: Git ignore patterns
- **Importance**: HIGH - Prevents committing sensitive files
- **Content**: Patterns for files to ignore in git
- **Status**: ✅ **PRESERVE** - Important for version control

#### **16. `LICENSE`** ⭐⭐⭐⭐

- **Purpose**: Project license information
- **Importance**: HIGH - Legal requirements
- **Content**: MIT License terms
- **Status**: ✅ **PRESERVE** - Legal requirement

---

## 📁 **CORE MODULES - PRESERVE**

### **Data Preprocessing**

#### **17. `partA_preprocessing/`** ⭐⭐⭐⭐⭐

- **Purpose**: Data preprocessing and loading modules
- **Importance**: CRITICAL - Core data processing functionality
- **Contents**:
  - `data_loader.py` - Data loading utilities
  - `preprocess.py` - Data preprocessing functions
- **Dependencies**: May be imported by other modules
- **Status**: ✅ **PRESERVE** - Essential for modular architecture
- **Note**: Previously deleted by mistake, restored

### **Model Building**

#### **18. `partB_model/`** ⭐⭐⭐⭐⭐

- **Purpose**: Machine learning model building modules
- **Importance**: CRITICAL - Core ML functionality
- **Contents**:
  - Model training scripts
  - Model evaluation tools
  - Hyperparameter optimization
- **Dependencies**: ML libraries, data preprocessing
- **Status**: ✅ **PRESERVE** - Essential for ML functionality

### **Strategy Analysis**

#### **19. `partC_strategy/`** ⭐⭐⭐⭐⭐

- **Purpose**: Trading strategy and analysis modules
- **Importance**: CRITICAL - Core strategy functionality
- **Contents**:
  - Strategy implementations
  - Backtesting tools
  - Risk management
- **Dependencies**: Models, data, analysis modules
- **Status**: ✅ **PRESERVE** - Essential for trading strategies

### **Analysis Modules**

#### **20. `analysis_modules/`** ⭐⭐⭐⭐

- **Purpose**: Specialized analysis components
- **Importance**: HIGH - Advanced analysis functionality
- **Contents**:
  - Long-term analysis
  - Mid-term analysis
  - Short-term analysis
- **Dependencies**: Core modules, data
- **Status**: ✅ **PRESERVE** - Important for analysis

---

## 📊 **DATA & MODELS - PRESERVE**

### **Data Storage**

#### **21. `data/`** ⭐⭐⭐⭐⭐

- **Purpose**: Stock data storage directory
- **Importance**: CRITICAL - Contains all stock data files
- **Contents**:
  - CSV files with stock data
  - Enhanced data files
  - Historical data
- **Status**: ✅ **PRESERVE** - Essential data storage

### **Model Storage**

#### **22. `models/`** ⭐⭐⭐⭐⭐

- **Purpose**: Trained model storage
- **Importance**: CRITICAL - Contains trained ML models
- **Contents**:
  - Trained model files
  - Model cache
  - Model metadata
- **Status**: ✅ **PRESERVE** - Essential for predictions

---

## 🛠️ **UTILITY DIRECTORIES - CONDITIONAL**

### **Development Tools**

#### **23. `scripts/`** ⭐⭐⭐

- **Purpose**: Utility scripts for development
- **Importance**: MEDIUM - Development convenience
- **Contents**: Helper scripts, automation tools
- **Status**: ⚠️ **VERIFY BEFORE DELETE** - Check contents first

#### **24. `notebooks/`** ⭐⭐⭐

- **Purpose**: Jupyter notebooks for analysis
- **Importance**: MEDIUM - Research and development
- **Contents**: Analysis notebooks, experiments
- **Status**: ⚠️ **VERIFY BEFORE DELETE** - Check contents first

#### **25. `logs/`** ⭐⭐

- **Purpose**: Application log files
- **Importance**: LOW - Debugging and monitoring
- **Contents**: System logs, error logs
- **Status**: 🔄 **SAFE TO DELETE** - Can be regenerated

---

## 🔧 **DEVELOPMENT FILES - CONDITIONAL**

### **IDE and Build Files**

#### **26. `.vscode/`** ⭐⭐

- **Purpose**: VS Code configuration
- **Importance**: LOW - IDE-specific settings
- **Contents**: VS Code settings, launch configurations
- **Status**: 🔄 **SAFE TO DELETE** - Can be regenerated

#### **27. `venv/`** ⭐⭐

- **Purpose**: Python virtual environment
- **Importance**: LOW - Can be recreated
- **Contents**: Python packages, environment
- **Status**: 🔄 **SAFE TO DELETE** - Can be recreated with requirements.txt

#### **28. `__pycache__/`** ⭐

- **Purpose**: Python bytecode cache
- **Importance**: NONE - Temporary files
- **Contents**: Compiled Python files
- **Status**: 🔄 **SAFE TO DELETE** - Auto-generated

#### **29. `catboost_info/`** ⭐

- **Purpose**: CatBoost model training logs
- **Importance**: NONE - Temporary training files
- **Contents**: Training logs, temporary files
- **Status**: 🔄 **SAFE TO DELETE** - Auto-generated

---

## 🚨 **DELETION GUIDELINES**

### **NEVER DELETE (⭐⭐⭐⭐⭐)**

- Core system files (`run_stock_prediction.py`, `unified_analysis_pipeline.py`)
- Configuration files (`requirements.txt`, `.env`, `angel_one_config.py`)
- Documentation files (`README.md`, `RUNNING_INSTRUCTIONS.md`)
- Core modules (`partA_preprocessing/`, `partB_model/`, `partC_strategy/`)
- Data and models (`data/`, `models/`)
- Test files (`tests/` directory)

### **VERIFY BEFORE DELETE (⭐⭐⭐)**

- Utility directories (`scripts/`, `notebooks/`)
- Check contents and dependencies first
- Ensure no important functionality is lost

### **SAFE TO DELETE (⭐-⭐⭐)**

- Cache directories (`__pycache__/`, `catboost_info/`)
- Log files (`logs/`)
- IDE settings (`.vscode/`)
- Virtual environment (`venv/`)

---

## 📋 **DELETION CHECKLIST**

Before deleting any file, check:

1. **Is it marked as ⭐⭐⭐⭐⭐ (CRITICAL)?**

   - ❌ **NEVER DELETE**

2. **Is it marked as ⭐⭐⭐⭐ (HIGH IMPORTANCE)?**

   - ❌ **NEVER DELETE**

3. **Is it marked as ⭐⭐⭐ (MEDIUM IMPORTANCE)?**

   - ⚠️ **VERIFY CONTENTS FIRST**

4. **Is it marked as ⭐-⭐⭐ (LOW IMPORTANCE)?**

   - ✅ **SAFE TO DELETE**

5. **Additional Checks:**
   - [ ] Check for imports/references in other files
   - [ ] Check documentation references
   - [ ] Verify it's not part of a workflow
   - [ ] Ensure no data loss

---

## 🎯 **QUICK REFERENCE**

### **CRITICAL FILES (Never Delete):**

```
run_stock_prediction.py
unified_analysis_pipeline.py
angel_one_config.py
angel_one_data_downloader.py
README.md
RUNNING_INSTRUCTIONS.md
ANGEL_ONE_SETUP.md
requirements.txt
.env
tests/run_all_tests.py
partA_preprocessing/
partB_model/
partC_strategy/
data/
models/
```

### **SAFE TO DELETE:**

```
__pycache__/
catboost_info/
logs/
.vscode/
venv/
```

### **VERIFY FIRST:**

```
scripts/
notebooks/
```

---

## 📝 **MAINTENANCE NOTES**

- **Last Updated**: August 25, 2025
- **Purpose**: Prevent accidental file deletions
- **Usage**: Reference before any cleanup operations
- **Updates**: Update this guide when adding new files

**Remember**: When in doubt, preserve the file. It's better to keep a potentially unnecessary file than to lose an important one.
