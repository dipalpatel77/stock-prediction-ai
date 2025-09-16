# AI Stock Predictor - System Architecture

## 🏗️ System Overview

The AI Stock Predictor is a modular, scalable system designed for comprehensive stock market analysis and prediction. The system follows a layered architecture with clear separation of concerns.

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  Command Line    │  Python API    │  Jupyter Notebook          │
│  run_analysis.py │  UnifiedPipeline│  notebook_interface.py     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│              UnifiedAnalysisPipeline                           │
│  • Coordinates all services                                    │
│  • Manages workflow execution                                  │
│  • Handles error recovery                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CORE SERVICES LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│  DataService    │  ModelService   │  StrategyService           │
│  • Data loading │  • Model training│  • Trading strategies      │
│  • Preprocessing│  • Predictions   │  • Signal generation       │
│  • Validation   │  • Evaluation    │  • Risk management         │
├─────────────────────────────────────────────────────────────────┤
│  DatabaseService│  ReportingService│  IncrementalDataService   │
│  • Data storage │  • Report gen    │  • Smart updates           │
│  • Retrieval    │  • Visualization │  • Change detection        │
│  • Optimization │  • Export        │  • Merge strategies        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  Angel One API  │  Yahoo Finance  │  FRED API                  │
│  • Indian stocks│  • US stocks     │  • Economic data           │
│  • Real-time    │  • Historical    │  • Market indicators       │
│  • F&O data     │  • Multiple      │  • Currency rates          │
│                 │    intervals     │                            │
├─────────────────────────────────────────────────────────────────┤
│  Phase1Integration│ Phase2Integration│ Phase3Integration        │
│  • Basic analysis│  • Advanced ML   │  • Risk assessment        │
│  • Technical     │  • Ensemble      │  • Geopolitical           │
│    indicators    │    methods       │  • Corporate actions      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  MySQL Database │  File System    │  Cache System              │
│  • Stock data   │  • CSV files     │  • In-memory cache         │
│  • Metadata     │  • Models        │  • Session storage         │
│  • Quality      │  • Reports       │  • Temporary data          │
│    metrics      │  • Logs          │                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Component Details

### 1. User Interface Layer

#### Command Line Interface

- **`run_analysis.py`**: Basic analysis with command-line arguments
- **`enhanced_unified_pipeline.py`**: Interactive data selection and analysis
- **`run_custom_analysis.py`**: Custom analysis with specific configurations

#### Python API

- **`UnifiedAnalysisPipeline`**: Main orchestrator class
- **Direct service access**: For advanced users and custom implementations

#### Jupyter Notebook Interface

- **`notebook_interface.py`**: Interactive analysis with plotting capabilities
- **Real-time visualization**: Charts and graphs for analysis results

### 2. Orchestration Layer

#### UnifiedAnalysisPipeline

```python
class UnifiedAnalysisPipeline:
    """
    Main orchestrator that coordinates all services and manages workflow
    """
    def __init__(self, ticker, max_workers=None, period_config="recommended"):
        # Initialize all services
        self.data_service = DataService(use_database=True)
        self.model_service = ModelService()
        self.strategy_service = StrategyService()
        self.reporting_service = ReportingService()

    def run_complete_analysis(self):
        # Coordinate complete analysis workflow
        pass
```

**Responsibilities:**

- Service coordination and dependency management
- Workflow orchestration and error handling
- Resource management and optimization
- User interface and result presentation

### 3. Core Services Layer

#### DataService

```python
class DataService:
    """
    Centralized data loading, preprocessing, and validation
    """
    def load_stock_data(self, ticker, period, interval):
        # Load data from multiple sources
        pass

    def preprocess_data(self, df, timeframe):
        # Clean and prepare data for analysis
        pass
```

**Features:**

- Multi-source data integration (Angel One, Yahoo Finance)
- Intelligent data preprocessing and cleaning
- Database integration for efficient storage
- Data quality validation and monitoring

#### ModelService

```python
class ModelService:
    """
    Machine learning model training, evaluation, and prediction
    """
    def train_models(self, data):
        # Train multiple ML algorithms
        pass

    def generate_predictions(self, models, data):
        # Generate predictions using trained models
        pass
```

**Algorithms:**

- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
- **Non-linear Models**: SVR, MLP Regressor, Gaussian Process
- **Advanced Models**: AdaBoost, Extra Trees, Huber Regressor, Kernel Ridge

#### DatabaseService

```python
class DatabaseService:
    """
    High-performance database operations for stock data
    """
    def store_stock_data(self, ticker, data, data_source):
        # Store data in database
        pass

    def get_stock_data(self, ticker, start_date, end_date):
        # Retrieve data from database
        pass
```

**Features:**

- Multi-database support (MySQL, PostgreSQL, SQLite, MongoDB)
- Optimized queries and indexing
- Data integrity and consistency
- Performance monitoring and optimization

### 4. Integration Layer

#### Angel One API Integration

```python
class AngelOneDataDownloader:
    """
    Angel One SmartAPI integration for Indian stocks
    """
    def get_historical_data(self, symbol, exchange, interval):
        # Download historical data from Angel One
        pass

    def get_real_time_data(self, symbol, exchange):
        # Get real-time market data
        pass
```

**Features:**

- NSE/BSE stock data
- Real-time and historical data
- F&O (Futures & Options) data
- Open Interest data
- Corporate actions and dividends

#### Yahoo Finance Integration

```python
def _download_from_yahoo(self, ticker, period, interval):
    """
    Yahoo Finance integration for US stocks
    """
    # Download data from Yahoo Finance
    pass
```

**Features:**

- NYSE/NASDAQ stock data
- Multiple time intervals
- Historical data up to 2 years
- Real-time price updates

### 5. Data Layer

#### Database Schema

```sql
-- Main stock data table
CREATE TABLE stock_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(15,4),
    high DECIMAL(15,4),
    low DECIMAL(15,4),
    close DECIMAL(15,4),
    volume BIGINT,
    adj_close DECIMAL(15,4),
    data_source VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_data (ticker, date)
);

-- Angel One specific data
CREATE TABLE angel_one_stock_data (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    symbol_token VARCHAR(20) NOT NULL,
    date DATETIME NOT NULL,
    open DECIMAL(15,4) NOT NULL,
    high DECIMAL(15,4) NOT NULL,
    low DECIMAL(15,4) NOT NULL,
    close DECIMAL(15,4) NOT NULL,
    volume BIGINT NOT NULL,
    interval_type VARCHAR(20) NOT NULL,
    data_source VARCHAR(20) NOT NULL,
    UNIQUE KEY unique_data (ticker, exchange, date, interval_type)
);
```

#### File System Organization

```
data/
├── by_ticker/           # Organized by stock ticker
│   ├── RELIANCE/
│   ├── TCS/
│   └── INFY/
├── cache/               # Temporary cache files
└── archive/             # Archived old data

models/
├── short_term/          # Short-term prediction models
├── mid_term/            # Medium-term prediction models
├── long_term/           # Long-term prediction models
├── general/             # General purpose models
└── scalers/             # Data scaling objects
```

## 🔄 Data Flow

### 1. Data Ingestion Flow

```
User Request → DataService → Source Selection → Data Download → Validation → Storage
```

### 2. Analysis Flow

```
Data Retrieval → Preprocessing → Feature Engineering → Model Training → Prediction → Reporting
```

### 3. Update Flow

```
Incremental Check → Change Detection → Smart Merge → Database Update → Cache Invalidation
```

## 🚀 Performance Optimization

### 1. Database Optimization

- **Indexing**: Optimized indexes for common queries
- **Partitioning**: Data partitioning by date and ticker
- **Connection Pooling**: Efficient database connections
- **Query Optimization**: Optimized SQL queries

### 2. Caching Strategy

- **In-memory Cache**: Fast access to frequently used data
- **File Cache**: Persistent cache for processed data
- **Database Cache**: Query result caching
- **Model Cache**: Trained model persistence

### 3. Parallel Processing

- **Multi-threading**: Parallel data processing
- **Worker Pools**: Configurable worker counts
- **Batch Processing**: Efficient batch operations
- **Async Operations**: Non-blocking I/O operations

## 🔒 Security & Reliability

### 1. Data Security

- **API Key Management**: Secure credential storage
- **Database Security**: Encrypted connections
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: Graceful error recovery

### 2. System Reliability

- **Fault Tolerance**: Service failure recovery
- **Data Backup**: Regular data backups
- **Health Monitoring**: System health checks
- **Logging**: Comprehensive logging system

## 📊 Monitoring & Analytics

### 1. System Monitoring

- **Performance Metrics**: Response times, throughput
- **Resource Usage**: CPU, memory, disk usage
- **Error Tracking**: Error rates and types
- **Data Quality**: Data completeness and accuracy

### 2. Business Analytics

- **Prediction Accuracy**: Model performance tracking
- **Data Coverage**: Market coverage analysis
- **User Analytics**: Usage patterns and trends
- **Cost Analysis**: Resource cost optimization

## 🔧 Configuration Management

### 1. Environment Configuration

```python
# Database configuration
DATABASE_CONFIG = {
    'local': {
        'db_type': 'mysql',
        'host': 'localhost',
        'user': 'root',
        'password': '7874',
        'database': 'stock_data'
    }
}

# Angel One configuration
ANGEL_ONE_CONFIG = {
    'api_key': 'your_api_key',
    'client_id': 'your_client_id',
    'pin': 'your_pin',
    'totp_secret': 'your_totp_secret'
}
```

### 2. Feature Flags

```python
# Feature toggles
FEATURES = {
    'use_database': True,
    'enable_caching': True,
    'parallel_processing': True,
    'real_time_updates': True
}
```

## 🎯 Scalability Considerations

### 1. Horizontal Scaling

- **Microservices**: Service decomposition
- **Load Balancing**: Request distribution
- **Database Sharding**: Data partitioning
- **Container Orchestration**: Docker/Kubernetes support

### 2. Vertical Scaling

- **Resource Optimization**: CPU/Memory optimization
- **Database Tuning**: Query and index optimization
- **Caching Strategy**: Multi-level caching
- **Code Optimization**: Performance profiling

---

This architecture provides a robust, scalable foundation for comprehensive stock market analysis and prediction, with clear separation of concerns and modular design principles.
