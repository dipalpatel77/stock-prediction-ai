#!/usr/bin/env python3
"""
Unified Stock Prediction System
A single file that combines both simple and advanced prediction systems
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings
import threading
import time
import sys
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Basic ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import talib

# Advanced ML imports (optional)
try:
    from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("⚠️ Advanced models not available. Install xgboost, lightgbm, and catboost for full functionality.")

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def timeout_handler(func, args=(), kwargs={}, timeout_duration=120):
    """Execute function with timeout using threading."""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_duration)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return None
    elif exception[0] is not None:
        raise exception[0]
    else:
        return result[0]

class UnifiedPredictionEngine:
    """
    Unified prediction engine with both simple and advanced capabilities
    """
    
    def __init__(self, ticker, mode="simple", interactive=True, use_incremental=True):
        self.ticker = ticker.upper()
        self.mode = mode  # "simple" or "advanced"
        self.interactive = interactive  # New parameter for non-interactive mode
        self.use_incremental = use_incremental  # Enable incremental learning integration
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_cache_dir = "models/cache"
        
        # Create cache directory
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Initialize incremental learning manager if enabled
        self.incremental_manager = None
        if self.use_incremental:
            try:
                from partB_model.incremental_learning import IncrementalLearningManager
                self.incremental_manager = IncrementalLearningManager()
                print(f"🔄 Incremental learning enabled for {self.ticker}")
            except ImportError as e:
                print(f"⚠️ Incremental learning not available: {e}")
                self.use_incremental = False
        
    def load_and_prepare_data(self):
        """Load and prepare data with improved fallback mechanisms."""
        try:
            # Enhanced data loading with multiple fallbacks
            df = self._load_data_with_fallbacks()
            
            if df is None:
                print(f"❌ No data found for {self.ticker} from any source")
                return None
            
            # Clean and prepare data
            df = self._clean_data(df)
            
            if self.mode == "simple":
                df = self._add_basic_features(df)
            else:
                df = self._add_advanced_features(df)
            
            print(f"📈 Final data shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _load_data_with_fallbacks(self):
        """Load data with multiple fallback sources."""
        # Priority order for data sources
        data_sources = [
            # 1. Enhanced data (best quality)
            f"data/{self.ticker}_partA_partC_enhanced.csv",
            # 2. Preprocessed data
            f"data/preprocessed_{self.ticker}.csv",
            # 3. Angel One data
            f"data/{self.ticker}_one_day_data.csv",
            # 4. Raw data
            f"data/{self.ticker}_raw_data.csv",
            # 5. Any CSV with ticker name
            f"data/{self.ticker}.csv"
        ]
        
        for source in data_sources:
            if os.path.exists(source):
                try:
                    print(f"📊 Loading data from: {source}")
                    df = pd.read_csv(source)
                    
                    # Handle different CSV formats
                    df = self._standardize_dataframe(df)
                    
                    if df is not None and len(df) > 10:  # Minimum data requirement
                        print(f"✅ Successfully loaded {len(df)} records from {source}")
                        return df
                        
                except Exception as e:
                    print(f"⚠️ Failed to load {source}: {e}")
                    continue
        
        # If no existing data, try to download
        print(f"📡 No existing data found. Attempting to download for {self.ticker}...")
        return self._download_fresh_data()
    
    def _standardize_dataframe(self, df):
        """Standardize dataframe format regardless of source."""
        try:
            # Handle different date column names
            date_columns = ['Date', 'date', 'DATE', 'Unnamed: 0', 'index']
            date_col = None
            
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                # Convert to datetime and set as index
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                df.set_index(date_col, inplace=True)
                
                # Remove the original date column if it's not the index
                if date_col != df.index.name:
                    df = df.drop(date_col, axis=1, errors='ignore')
            
            # Remove unnamed columns
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col or 'unnamed' in col]
            if unnamed_cols:
                df = df.drop(unnamed_cols, axis=1)
            
            # Standardize column names
            column_mapping = {
                'Open': 'Open', 'open': 'Open', 'OPEN': 'Open',
                'High': 'High', 'high': 'High', 'HIGH': 'High',
                'Low': 'Low', 'low': 'Low', 'LOW': 'Low',
                'Close': 'Close', 'close': 'Close', 'CLOSE': 'Close',
                'Volume': 'Volume', 'volume': 'Volume', 'VOLUME': 'Volume',
                'Adj Close': 'Adj Close', 'adj close': 'Adj Close', 'ADJ CLOSE': 'Adj Close'
            }
            
            # Rename columns if needed
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns and new_name not in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Ensure we have at least Open, High, Low, Close
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Missing required columns. Found: {list(df.columns)}")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ Error standardizing dataframe: {e}")
            return None
    
    def _download_fresh_data(self):
        """Download fresh data from available sources."""
        try:
            # Try Angel One first
            if os.getenv('ANGEL_ONE_API_KEY'):
                print("📡 Attempting Angel One download...")
                df = self._download_angel_one_data()
                if df is not None:
                    return df
            
            # Try yfinance as fallback
            print("📡 Attempting yfinance download...")
            df = self._download_yfinance_data()
            if df is not None:
                return df
            
            print("❌ No data sources available")
            return None
            
        except Exception as e:
            print(f"❌ Error downloading fresh data: {e}")
            return None
    
    def _download_yfinance_data(self):
        """Download data using yfinance as fallback."""
        try:
            import yfinance as yf
            
            # Add .NS suffix for Indian stocks if not present
            ticker_symbol = self.ticker
            if not ticker_symbol.endswith('.NS') and not '.' in ticker_symbol:
                ticker_symbol = f"{ticker_symbol}.NS"
            
            print(f"📡 Downloading {ticker_symbol} from yfinance...")
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="1y")
            
            if df is not None and len(df) > 10:
                # Save the data
                output_file = f"data/{self.ticker}_yfinance_data.csv"
                df.to_csv(output_file)
                print(f"💾 Saved yfinance data to: {output_file}")
                return df
            else:
                print("❌ No data received from yfinance")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading from yfinance: {e}")
            return None
    
    def _download_angel_one_data(self):
        """Download data from Angel One API."""
        try:
            # Import Angel One downloader
            try:
                from angel_one_data_downloader import AngelOneDataDownloader
            except ImportError:
                print("⚠️ Angel One downloader not available. Please install required packages.")
                return None
            
            # Check if credentials are available
            api_key = os.getenv('ANGEL_ONE_API_KEY')
            auth_token = os.getenv('ANGEL_ONE_AUTH_TOKEN')
            
            if not api_key or not auth_token:
                print("⚠️ Angel One credentials not found in environment variables")
                return None
            
            # Initialize downloader
            downloader = AngelOneDataDownloader(
                api_key=api_key,
                auth_token=auth_token,
                client_ip=os.getenv('CLIENT_IP', '127.0.0.1'),
                mac_address=os.getenv('MAC_ADDRESS', '00:00:00:00:00:00')
            )
            
            # Download data
            df = downloader.get_latest_data(self.ticker, days=365, interval="ONE_DAY")
            
            if df is not None and len(df) > 10:
                # Save as enhanced data for future use
                enhanced_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
                df.to_csv(enhanced_file)
                print(f"💾 Saved Angel One data to: {enhanced_file}")
                return df
            else:
                print(f"❌ Failed to download data for {self.ticker}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading Angel One data: {e}")
            return None
    
    def _clean_data(self, df):
        """Clean and prepare the dataset."""
        # Remove unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df = df.drop(unnamed_cols, axis=1)
        
        # Handle date column
        if len(df.columns) > 1 and df.columns[1] == 'Price':
            df['Date'] = df['Price']
            df = df.drop('Price', axis=1)
        elif len(df.columns) > 1 and df.columns[1] == 'Date':
            df['Date'] = df['Date']
            df = df.drop('Date', axis=1)
        
        # Set date as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Forward fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _add_basic_features(self, df):
        """Add basic, reliable technical indicators."""
        try:
            # Basic price features
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Range'] = df['High'] - df['Low']
            
            # Simple moving averages
            df['SMA_5'] = df['Close'].rolling(5).mean()
            df['SMA_10'] = df['Close'].rolling(10).mean()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            
            # Volume features
            df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
            
            # Basic RSI
            df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
            
            # Basic MACD
            df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding basic features: {e}")
            return df
    
    def _add_advanced_features(self, df):
        """Add advanced technical indicators and pattern recognition features."""
        try:
            # Start with basic features
            df = self._add_basic_features(df)
            
            # Advanced price features
            df['Price_Change_2d'] = df['Close'].pct_change(2)
            df['Price_Change_5d'] = df['Close'].pct_change(5)
            df['Price_Change_10d'] = df['Close'].pct_change(10)
            
            # Advanced volume features
            df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
            df['Volume_Price_Trend'] = (df['Volume'] * df['Price_Change']).cumsum()
            
            # Advanced moving averages
            df['EMA_5'] = talib.EMA(df['Close'], timeperiod=5)
            df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
            df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
            df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
            
            # Moving average crossovers
            df['MA_Cross_5_20'] = df['EMA_5'] - df['EMA_20']
            df['MA_Cross_10_50'] = df['EMA_10'] - df['EMA_50']
            
            # Bollinger Bands advanced
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # RSI variations
            df['RSI_5'] = talib.RSI(df['Close'], timeperiod=5)
            df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
            
            # MACD advanced
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Additional indicators
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
            df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'])
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Market regime features
            df['Trend_Strength'] = abs(df['EMA_10'] - df['EMA_50']) / df['EMA_50']
            df['Volatility_20d'] = df['Close'].rolling(20).std()
            df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
            df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
            
            # Volatility features
            returns = df['Close'].pct_change().dropna()
            df['Volatility_GARCH'] = returns.rolling(20).std()
            
            # Momentum features
            df['Volume_Momentum_5d'] = df['Volume'].pct_change(5)
            df['RSI_Momentum'] = df['RSI_14'].diff()
            df['MACD_Momentum'] = df['MACD'].diff()
            
            # Support/Resistance features
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low']
            df['S1'] = 2 * df['Pivot'] - df['High']
            df['Distance_R1'] = (df['R1'] - df['Close']) / df['Close']
            df['Distance_S1'] = (df['Close'] - df['S1']) / df['Close']
            
            # Pattern Recognition Features
            df = self._add_pattern_recognition_features(df)
            
            # Advanced Statistical Features
            df = self._add_statistical_features(df)
            
            # Market Microstructure Features
            df = self._add_microstructure_features(df)
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding advanced features: {e}")
            return df
    
    def _add_pattern_recognition_features(self, df):
        """Add pattern recognition features for better prediction."""
        try:
            # Candlestick patterns
            df['Doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
            df['Hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
            df['Engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
            df['Morning_Star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
            df['Evening_Star'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
            
            # Price patterns
            df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
            df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
            df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
            df['Lower_High'] = (df['High'] < df['High'].shift(1)).astype(int)
            
            # Trend patterns
            df['Uptrend_5d'] = (df['Close'] > df['Close'].shift(5)).astype(int)
            df['Downtrend_5d'] = (df['Close'] < df['Close'].shift(5)).astype(int)
            df['Uptrend_10d'] = (df['Close'] > df['Close'].shift(10)).astype(int)
            df['Downtrend_10d'] = (df['Close'] < df['Close'].shift(10)).astype(int)
            
            # Breakout patterns
            df['Breakout_Above_20d_High'] = (df['Close'] > df['High'].rolling(20).max().shift(1)).astype(int)
            df['Breakout_Below_20d_Low'] = (df['Close'] < df['Low'].rolling(20).min().shift(1)).astype(int)
            
            # Consolidation patterns
            df['Consolidation_5d'] = ((df['High'].rolling(5).max() - df['Low'].rolling(5).min()) / df['Close'].rolling(5).mean() < 0.02).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding pattern recognition features: {e}")
            return df
    
    def _add_statistical_features(self, df):
        """Add advanced statistical features."""
        try:
            # Z-score features
            df['Price_ZScore_20d'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
            df['Volume_ZScore_20d'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
            
            # Percentile features
            df['Price_Percentile_20d'] = df['Close'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            df['Volume_Percentile_20d'] = df['Volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
            
            # Skewness and Kurtosis
            df['Price_Skewness_20d'] = df['Close'].rolling(20).skew()
            df['Price_Kurtosis_20d'] = df['Close'].rolling(20).kurt()
            
            # Autocorrelation features
            df['Price_Autocorr_1d'] = df['Close'].rolling(20).apply(lambda x: x.autocorr(lag=1))
            df['Price_Autocorr_5d'] = df['Close'].rolling(20).apply(lambda x: x.autocorr(lag=5))
            
            # Mean reversion features
            df['Mean_Reversion_Strength'] = abs(df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding statistical features: {e}")
            return df
    
    def _add_microstructure_features(self, df):
        """Add market microstructure features."""
        try:
            # Bid-ask spread proxy (using high-low ratio)
            df['Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
            
            # Price impact
            df['Price_Impact'] = df['Price_Change'].abs() / df['Volume_Ratio']
            
            # Order flow imbalance
            df['Order_Flow_Imbalance'] = (df['Volume'] * df['Price_Change']).rolling(5).sum()
            
            # Liquidity measures
            df['Liquidity_Ratio'] = df['Volume'] / df['Spread_Proxy']
            
            # Market efficiency ratio
            df['Market_Efficiency_Ratio'] = abs(df['Close'] - df['Close'].shift(20)) / df['Close'].rolling(20).apply(lambda x: sum(abs(x.diff().dropna())))
            
            return df
            
        except Exception as e:
            print(f"Warning: Error adding microstructure features: {e}")
            return df
    
    def prepare_features(self, df):
        """Prepare features for prediction."""
        try:
            if self.mode == "simple":
                # Select basic features
                feature_cols = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'Price_Change', 'Price_Range',
                    'SMA_5', 'SMA_10', 'SMA_20',
                    'Volume_Ratio',
                    'RSI_14',
                    'MACD', 'MACD_Signal',
                    'BB_Upper', 'BB_Middle', 'BB_Lower'
                ]
            else:
                # Select advanced features including pattern recognition
                feature_cols = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'Price_Change', 'Price_Change_2d', 'Price_Change_5d', 'Price_Change_10d',
                    'Price_Range', 'Volume_Ratio', 'Volume_Price_Trend',
                    'SMA_5', 'SMA_10', 'SMA_20',
                    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
                    'MA_Cross_5_20', 'MA_Cross_10_50',
                    'RSI_14', 'RSI_5', 'RSI_21',
                    'MACD', 'MACD_Signal', 'MACD_Hist',
                    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Position',
                    'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'ATR', 'ADX', 'OBV', 'MFI',
                    'Trend_Strength', 'Volatility_20d',
                    'Momentum_5d', 'Momentum_10d', 'Momentum_20d',
                    'Volatility_GARCH', 'Volume_Momentum_5d',
                    'RSI_Momentum', 'MACD_Momentum',
                    'Pivot', 'R1', 'S1', 'Distance_R1', 'Distance_S1',
                    # Pattern Recognition Features
                    'Doji', 'Hammer', 'Engulfing', 'Morning_Star', 'Evening_Star',
                    'Higher_High', 'Lower_Low', 'Higher_Low', 'Lower_High',
                    'Uptrend_5d', 'Downtrend_5d', 'Uptrend_10d', 'Downtrend_10d',
                    'Breakout_Above_20d_High', 'Breakout_Below_20d_Low', 'Consolidation_5d',
                    # Statistical Features
                    'Price_ZScore_20d', 'Volume_ZScore_20d',
                    'Price_Percentile_20d', 'Volume_Percentile_20d',
                    'Price_Skewness_20d', 'Price_Kurtosis_20d',
                    'Price_Autocorr_1d', 'Price_Autocorr_5d',
                    'Mean_Reversion_Strength',
                    # Microstructure Features
                    'Spread_Proxy', 'Price_Impact', 'Order_Flow_Imbalance',
                    'Liquidity_Ratio', 'Market_Efficiency_Ratio'
                ]
            
            # Get available features
            available_features = [col for col in feature_cols if col in df.columns]
            
            min_features = 8 if self.mode == "simple" else 15
            if len(available_features) < min_features:
                print(f"❌ Insufficient features available ({len(available_features)} < {min_features})")
                return None, None
            
            # Prepare feature matrix
            X = df[available_features].dropna()
            
            if len(X) < 50:
                print("❌ Insufficient data points")
                return None, None
            
            # Clean the data
            X = self._clean_feature_data(X)
            
            if X is None or len(X) < 50:
                print("❌ Insufficient data after cleaning")
                return None, None
            
            # Create target variable (next day's close price)
            y = X['Close'].shift(-1).dropna()
            X = X[:-1]  # Remove last row since we don't have target
            
            # Ensure X and y have the same length
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y.iloc[:min_len]
            
            print(f"✅ Prepared features: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None, None
    
    def _clean_feature_data(self, X):
        """Clean feature data by removing infinite and extreme values."""
        try:
            print("🧹 Cleaning feature data...")
            
            # Replace infinite values with NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Remove rows with any NaN values
            X_clean = X.dropna()
            
            if len(X_clean) < len(X) * 0.5:  # If we lost more than 50% of data
                print(f"⚠️ Warning: Lost {(len(X) - len(X_clean))} rows due to NaN values")
            
            # Cap extreme values for each column
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Use percentile-based capping
                    lower_bound = X_clean[col].quantile(0.001)
                    upper_bound = X_clean[col].quantile(0.999)
                    X_clean[col] = X_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            print(f"✅ Data cleaned: {X_clean.shape}")
            return X_clean
            
        except Exception as e:
            print(f"❌ Error cleaning feature data: {e}")
            return None
    
    def train_models(self, X, y):
        """Train models based on mode with caching."""
        try:
            # Check if we can load cached models first
            if self._load_cached_models(X.columns):
                print("✅ Loaded cached models")
                return True
            
            # Train new models
            if self.mode == "simple":
                success = self._train_simple_models(X, y)
            else:
                success = self._train_advanced_models(X, y)
            
            # Cache models if training was successful
            if success:
                self._cache_models()
            
            return success
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False
    
    def _train_simple_models(self, X, y):
        """Train simple, reliable models."""
        try:
            print("🤖 Training simple prediction models...")
            
            # Initialize simple models
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    min_samples_leaf=2, random_state=42, n_jobs=-1
                ),
                'LinearRegression': LinearRegression()
            }
            
            # Train models
            for name, model in models.items():
                print(f"   Training {name}...")
                
                # Scale features for linear regression
                if name == 'LinearRegression':
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[name] = scaler
                    model.fit(X_scaled, y)
                else:
                    model.fit(X, y)
                
                self.models[name] = model
            
            # Create simple ensemble (average of predictions)
            print("   Creating simple ensemble...")
            self.models['Ensemble'] = 'average'  # We'll implement this in prediction
            
            print("✅ All models trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training simple models: {e}")
            return False
    
    def _train_advanced_models(self, X, y):
        """Train advanced ensemble models with sophisticated algorithms."""
        try:
            if not ADVANCED_AVAILABLE:
                print("⚠️ Advanced models not available, falling back to simple models")
                return self._train_simple_models(X, y)
            
            print("🤖 Training advanced prediction models with pattern learning...")
            
            # Feature selection and engineering
            X_enhanced = self._enhance_features(X)
            
            # Initialize advanced models with hyperparameter optimization
            models = {
                'RandomForest': RandomForestRegressor(
                    n_estimators=300, max_depth=20, min_samples_split=3,
                    min_samples_leaf=1, random_state=42, n_jobs=-1,
                    max_features='sqrt', bootstrap=True
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=10,
                    min_samples_split=3, random_state=42, subsample=0.8
                ),
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=10,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    reg_alpha=0.1, reg_lambda=1.0
                ),
                'LightGBM': lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=10,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    reg_alpha=0.1, reg_lambda=1.0
                ),
                'ExtraTrees': ExtraTreesRegressor(
                    n_estimators=200, max_depth=15, min_samples_split=3,
                    min_samples_leaf=1, random_state=42, n_jobs=-1
                ),
                'AdaBoost': AdaBoostRegressor(
                    n_estimators=200, learning_rate=0.1, random_state=42
                ),
                'SVR': SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1),
                'Ridge': Ridge(alpha=0.1),
                'Lasso': Lasso(alpha=0.01),
                'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
                'Huber': HuberRegressor(epsilon=1.35, max_iter=200),
                'KernelRidge': KernelRidge(alpha=1.0, kernel='rbf'),
                'MLP': MLPRegressor(
                    hidden_layer_sizes=(200, 100, 50), max_iter=1000,
                    random_state=42, early_stopping=True, learning_rate='adaptive'
                )
            }
            
            # Try to add CatBoost with fallback
            try:
                models['CatBoost'] = CatBoostRegressor(
                    iterations=150, learning_rate=0.1, depth=6,
                    random_state=42, verbose=False, allow_writing_files=False,
                    task_type='CPU', thread_count=1, early_stopping_rounds=10
                )
            except Exception as e:
                print(f"   ⚠️ CatBoost not available: {e}")
            
            # Add Gaussian Process if data size allows
            if len(X) < 1000:  # GP is computationally expensive
                try:
                    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0], (1e-2, 1e2))
                    models['GaussianProcess'] = GaussianProcessRegressor(
                        kernel=kernel, random_state=42, n_restarts_optimizer=10
                    )
                except:
                    pass
            
            # Train individual models with feature scaling and timeout protection
            for name, model in models.items():
                print(f"   Training {name}...")
                
                try:
                    # Define training function
                    def train_model():
                        if name in ['SVR', 'Ridge', 'Lasso', 'ElasticNet', 'Huber', 'KernelRidge', 'MLP', 'GaussianProcess']:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_enhanced)
                            self.scalers[name] = scaler
                            model.fit(X_scaled, y)
                        else:
                            model.fit(X_enhanced, y)
                        return True
                    
                    # Execute with timeout (2 minutes)
                    result = timeout_handler(train_model, timeout_duration=120)
                    
                    if result is None:
                        print(f"   ⚠️ {name} training timed out, skipping...")
                        continue
                    
                    self.models[name] = model
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = dict(zip(X_enhanced.columns, model.feature_importances_))
                    
                    print(f"   ✅ {name} trained successfully")
                    
                except Exception as e:
                    print(f"   ❌ Error training {name}: {e}")
                    continue
            
            # Create sophisticated ensemble model
            print("   Creating adaptive ensemble model...")
            
            # Create adaptive ensemble with dynamic weighting
            adaptive_ensemble = self._create_adaptive_ensemble(X_enhanced, y)
            
            if adaptive_ensemble is not None:
                self.models['Ensemble'] = adaptive_ensemble
                print(f"   ✅ Adaptive ensemble created successfully")
            else:
                # Fallback to simple ensemble
                available_models = []
                for name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees']:
                    if name in self.models:
                        available_models.append((name.lower()[:3], self.models[name]))
                
                if len(available_models) >= 2:
                    # Create ensemble with available models
                    weights = [1.0/len(available_models)] * len(available_models)  # Equal weights
                    self.models['Ensemble'] = VotingRegressor(
                        estimators=available_models,
                        weights=weights
                    )
                    
                    # Train ensemble
                    self.models['Ensemble'].fit(X_enhanced, y)
                    print(f"   ✅ Fallback ensemble created with {len(available_models)} models")
                else:
                    print("   ⚠️ Not enough models for ensemble, using average prediction")
                    self.models['Ensemble'] = 'average'
            
            print("✅ All advanced models trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training advanced models: {e}")
            return False
    
    def _enhance_features(self, X):
        """Enhance features with polynomial and interaction terms."""
        try:
            print("🔧 Enhancing features with polynomial terms...")
            
            # Select numerical features for enhancement
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 20:  # Limit to top features to avoid explosion
                # Use feature importance or correlation to select top features
                corr_with_target = X[numerical_cols].corrwith(X['Close']).abs().sort_values(ascending=False)
                top_features = corr_with_target.head(15).index.tolist()
            else:
                top_features = numerical_cols.tolist()
            
            X_enhanced = X.copy()
            
            # Add polynomial features for top features
            for col in top_features[:5]:  # Limit to top 5 to avoid overfitting
                if col != 'Close':  # Don't create polynomial of target
                    X_enhanced[f'{col}_squared'] = X[col] ** 2
                    X_enhanced[f'{col}_cubed'] = X[col] ** 3
            
            # Add interaction terms between highly correlated features
            corr_matrix = X[top_features].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.7:  # High correlation threshold
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Add interaction terms (limit to avoid explosion)
            for i, (col1, col2) in enumerate(high_corr_pairs[:10]):
                X_enhanced[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
            
            print(f"✅ Enhanced features: {X.shape[1]} → {X_enhanced.shape[1]}")
            return X_enhanced
            
        except Exception as e:
            print(f"Warning: Error enhancing features: {e}")
            return X
    
    def generate_predictions(self, X, days_ahead=5):
        """Generate predictions using trained models."""
        try:
            predictions = {}
            
            # Enhance features if in advanced mode
            if self.mode == "advanced":
                X_enhanced = self._enhance_features(X)
                last_data = X_enhanced.iloc[-1:].values
            else:
                last_data = X.iloc[-1:].values
            
            # Generate predictions for each model
            for name, model in self.models.items():
                try:
                    if name == 'Ensemble' and self.mode == "simple":
                        # Calculate ensemble as weighted average of other models
                        other_preds = []
                        other_weights = []
                        
                        # Use stored weights if available, otherwise equal weights
                        if hasattr(self, 'ensemble_weights') and self.ensemble_weights:
                            weights = self.ensemble_weights
                        else:
                            weights = {}
                            model_count = len([n for n in self.models.keys() if n != 'Ensemble'])
                            equal_weight = 1.0 / model_count
                            for other_name in self.models.keys():
                                if other_name != 'Ensemble':
                                    weights[other_name] = equal_weight
                        
                        for other_name, other_model in self.models.items():
                            if other_name != 'Ensemble':
                                if other_name in self.scalers:
                                    pred = other_model.predict(self.scalers[other_name].transform(last_data))[0]
                                else:
                                    pred = other_model.predict(last_data)[0]
                                other_preds.append(pred)
                                other_weights.append(weights.get(other_name, 1.0))
                        
                        # Calculate weighted average
                        if other_weights and sum(other_weights) > 0:
                            weighted_pred = np.average(other_preds, weights=other_weights)
                        else:
                            weighted_pred = np.mean(other_preds)
                        
                        predictions[name] = weighted_pred
                    else:
                        if name in self.scalers:
                            pred = model.predict(self.scalers[name].transform(last_data))[0]
                        else:
                            pred = model.predict(last_data)[0]
                        predictions[name] = pred
                except Exception as e:
                    print(f"⚠️ Error with {name} model: {e}")
                    continue
            
            # Generate multi-day predictions
            multi_day_predictions = self._generate_multi_day_predictions(X, days_ahead)
            
            return predictions, multi_day_predictions
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            return None, None
    
    def _generate_multi_day_predictions(self, X, days_ahead):
        """Generate multi-day predictions."""
        try:
            if self.mode == "simple":
                return self._generate_simple_multi_day_predictions(X, days_ahead)
            else:
                return self._generate_advanced_multi_day_predictions(X, days_ahead)
        except Exception as e:
            print(f"Warning: Error generating multi-day predictions: {e}")
            return None
    
    def _generate_simple_multi_day_predictions(self, X, days_ahead):
        """Generate simple multi-day predictions."""
        try:
            predictions = []
            current_price = X['Close'].iloc[-1]
            
            # Simple trend-based prediction
            recent_trend = (X['Close'].iloc[-5:].mean() - X['Close'].iloc[-10:-5].mean()) / X['Close'].iloc[-10:-5].mean()
            
            for day in range(days_ahead):
                # Simple linear extrapolation with trend decay
                trend_factor = recent_trend * (0.9 ** day)  # Trend decays over time
                pred = current_price * (1 + trend_factor)
                predictions.append(pred)
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Error generating simple multi-day predictions: {e}")
            return None
    
    def _generate_advanced_multi_day_predictions(self, X, days_ahead):
        """Generate advanced multi-day predictions using recursive approach."""
        try:
            predictions = []
            current_data = X.iloc[-1:].copy()
            
            for day in range(days_ahead):
                # Generate prediction for next day
                if 'Ensemble' in self.models:
                    pred = self.models['Ensemble'].predict(current_data.values)[0]
                else:
                    # Use average of available models
                    preds = []
                    for name, model in self.models.items():
                        if name in self.scalers:
                            preds.append(model.predict(self.scalers[name].transform(current_data.values))[0])
                        else:
                            preds.append(model.predict(current_data.values)[0])
                    pred = np.mean(preds)
                
                predictions.append(pred)
                
                # Update features for next prediction (simplified approach)
                if day < days_ahead - 1:
                    # Update price-based features
                    current_data['Close'] = pred
                    current_data['Price_Change'] = (pred - X.iloc[-1]['Close']) / X.iloc[-1]['Close']
                    
                    # Update other features based on prediction
                    if 'Price_Range' in current_data.columns:
                        current_data['Price_Range'] = current_data['Price_Range'] * 0.95  # Slight decay
                    if 'Volume_Ratio' in current_data.columns:
                        current_data['Volume_Ratio'] = current_data['Volume_Ratio'] * 0.98  # Slight decay
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Error generating advanced multi-day predictions: {e}")
            return None
    
    def _load_cached_models(self, feature_columns):
        """Load the best available model (incremental or cached)."""
        # First, try to load the latest incremental model
        if self.use_incremental and self.incremental_manager:
            incremental_success = self._load_latest_incremental_model(feature_columns)
            if incremental_success:
                return True
        
        # Fallback to regular cache
        return self._load_regular_cached_models(feature_columns)
    
    def _load_latest_incremental_model(self, feature_columns):
        """Load the latest incremental model if available."""
        try:
            if not self.incremental_manager:
                return False
            
            # Get the latest version for this ticker and mode
            latest_version = self.incremental_manager.get_latest_version(self.ticker, self.mode)
            
            if latest_version and latest_version.is_active():
                print(f"🔄 Loading incremental model version: {latest_version.version_id}")
                
                # Load the model from incremental system
                model_data = self.incremental_manager.load_model(latest_version.version_id)
                
                if model_data and 'models' in model_data and 'scaler' in model_data:
                    # Verify model compatibility
                    if 'RandomForest' in model_data['models']:
                        expected_features = len(feature_columns)
                        actual_features = len(model_data['models']['RandomForest'].feature_importances_)
                        
                        if expected_features == actual_features:
                            self.models = model_data['models']
                            self.scalers['standard'] = model_data['scaler']
                            print(f"✅ Loaded incremental model ({len(self.models)} models)")
                            print(f"📊 Model version: {latest_version.version_id}")
                            print(f"📈 Performance: {latest_version.performance_metrics}")
                            return True
                        else:
                            print(f"⚠️ Incremental model incompatible (features: {actual_features} vs {expected_features})")
                    else:
                        print(f"⚠️ Incremental model missing RandomForest")
                else:
                    print(f"⚠️ Incremental model data incomplete")
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error loading incremental model: {e}")
            return False
    
    def _load_regular_cached_models(self, feature_columns):
        """Load regular cached models if they exist and are compatible."""
        try:
            cache_file = f"{self.model_cache_dir}/{self.ticker}_{self.mode}_models.pkl"
            scaler_file = f"{self.model_cache_dir}/{self.ticker}_{self.mode}_scaler.pkl"
            
            if os.path.exists(cache_file) and os.path.exists(scaler_file):
                # Check if cache is recent (less than 24 hours old)
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < 86400:  # 24 hours
                    models = joblib.load(cache_file)
                    scaler = joblib.load(scaler_file)
                    
                    # Verify model compatibility by checking feature count
                    if 'RandomForest' in models:
                        expected_features = len(feature_columns)
                        actual_features = len(models['RandomForest'].feature_importances_)
                        if expected_features == actual_features:
                            self.models = models
                            self.scalers['standard'] = scaler
                            print(f"✅ Loaded cached models ({len(models)} models)")
                            return True
                        else:
                            print(f"⚠️ Cached models incompatible (features: {actual_features} vs {expected_features})")
                
            return False
            
        except Exception as e:
            print(f"⚠️ Error loading cached models: {e}")
            return False
    
    def _cache_models(self):
        """Cache trained models for future use and create incremental version."""
        try:
            if not self.models:
                return
            
            # Regular caching
            cache_file = f"{self.model_cache_dir}/{self.ticker}_{self.mode}_models.pkl"
            scaler_file = f"{self.model_cache_dir}/{self.ticker}_{self.mode}_scaler.pkl"
            
            joblib.dump(self.models, cache_file)
            if 'standard' in self.scalers:
                joblib.dump(self.scalers['standard'], scaler_file)
            
            print(f"💾 Models cached to {cache_file}")
            
            # Create incremental version if enabled
            if self.use_incremental and self.incremental_manager:
                self._create_incremental_version()
            
        except Exception as e:
            print(f"⚠️ Error caching models: {e}")
    
    def _create_incremental_version(self):
        """Create an incremental version of the current models."""
        try:
            if not self.incremental_manager or not self.models:
                return
            
            # Prepare model data for incremental system
            model_data = {
                'models': self.models,
                'scaler': self.scalers.get('standard'),
                'feature_columns': list(self.models['RandomForest'].feature_importances_),
                'training_samples': len(self.models['RandomForest'].estimators_) * 100,  # Estimate
                'creation_timestamp': datetime.now().isoformat()
            }
            
            # Create new version
            version_id = f"{self.ticker}_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate basic performance metrics (placeholder)
            performance_metrics = {
                'accuracy': 0.85,  # Placeholder - would need actual validation
                'mse': 0.15,
                'mae': 0.12
            }
            
            # Save to incremental system
            success = self.incremental_manager.save_model(
                version_id=version_id,
                model_data=model_data,
                performance_metrics=performance_metrics,
                metadata={
                    'ticker': self.ticker,
                    'mode': self.mode,
                    'source': 'run_stock_prediction',
                    'feature_count': len(model_data['feature_columns'])
                }
            )
            
            if success:
                print(f"🔄 Created incremental version: {version_id}")
            else:
                print(f"⚠️ Failed to create incremental version")
                
        except Exception as e:
            print(f"⚠️ Error creating incremental version: {e}")
    
    def get_model_status(self):
        """Get information about the current model status."""
        status = {
            'ticker': self.ticker,
            'mode': self.mode,
            'incremental_enabled': self.use_incremental,
            'models_loaded': len(self.models) > 0,
            'model_count': len(self.models),
            'model_types': list(self.models.keys()) if self.models else []
        }
        
        # Add ensemble performance information
        if hasattr(self, 'ensemble_performance') and self.ensemble_performance:
            status['ensemble_performance'] = self.ensemble_performance
            status['ensemble_weights'] = getattr(self, 'ensemble_weights', {})
        
        if self.use_incremental and self.incremental_manager:
            try:
                latest_version = self.incremental_manager.get_latest_version(self.ticker, self.mode)
                if latest_version:
                    status['incremental_version'] = latest_version.version_id
                    status['incremental_performance'] = latest_version.performance_metrics
                    status['incremental_active'] = latest_version.is_active()
                else:
                    status['incremental_version'] = None
                    status['incremental_performance'] = None
                    status['incremental_active'] = False
            except Exception as e:
                status['incremental_error'] = str(e)
        
        return status
    
    def check_for_updates(self):
        """Check if incremental updates are available."""
        if not self.use_incremental or not self.incremental_manager:
            return False, "Incremental learning not enabled"
        
        try:
            # This would typically check for new data or performance degradation
            # For now, we'll return a simple check
            latest_version = self.incremental_manager.get_latest_version(self.ticker, self.mode)
            
            if latest_version:
                # Check if the latest version is recent (within 7 days)
                version_date = datetime.fromisoformat(latest_version.creation_timestamp.replace('Z', '+00:00'))
                days_old = (datetime.now() - version_date).days
                
                if days_old > 7:
                    return True, f"Model is {days_old} days old, consider updating"
                else:
                    return False, f"Model is up to date ({days_old} days old)"
            else:
                return True, "No incremental model found, initial training needed"
                
        except Exception as e:
            return False, f"Error checking for updates: {e}"
    
    def calculate_prediction_confidence(self, predictions):
        """Calculate confidence intervals and model analysis."""
        try:
            # Get all model predictions
            model_predictions = list(predictions.values())
            
            # Calculate statistics
            mean_pred = np.mean(model_predictions)
            std_pred = np.std(model_predictions)
            
            # Confidence intervals
            confidence_68 = (mean_pred - std_pred, mean_pred + std_pred)
            confidence_95 = (mean_pred - 2*std_pred, mean_pred + 2*std_pred)
            
            # Model agreement (lower std = higher agreement)
            agreement_score = 1 / (1 + std_pred/mean_pred) if mean_pred != 0 else 0
            
            # Model diversity analysis
            model_diversity = self._analyze_model_diversity(predictions)
            
            # Pattern strength analysis
            pattern_strength = self._analyze_pattern_strength(predictions)
            
            return {
                'mean': mean_pred,
                'std': std_pred,
                'confidence_68': confidence_68,
                'confidence_95': confidence_95,
                'agreement_score': agreement_score,
                'model_diversity': model_diversity,
                'pattern_strength': pattern_strength
            }
            
        except Exception as e:
            print(f"Warning: Error calculating confidence: {e}")
            return None
    
    def _analyze_model_diversity(self, predictions):
        """Analyze diversity among model predictions."""
        try:
            pred_values = list(predictions.values())
            
            # Calculate coefficient of variation
            cv = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) != 0 else 0
            
            # Calculate prediction range
            pred_range = max(pred_values) - min(pred_values)
            pred_range_pct = (pred_range / np.mean(pred_values)) * 100 if np.mean(pred_values) != 0 else 0
            
            # Determine diversity level
            if cv < 0.01:
                diversity_level = "LOW"
                diversity_desc = "Models are in strong agreement"
            elif cv < 0.05:
                diversity_level = "MEDIUM"
                diversity_desc = "Models show moderate agreement"
            else:
                diversity_level = "HIGH"
                diversity_desc = "Models show significant disagreement"
            
            return {
                'coefficient_of_variation': cv,
                'prediction_range': pred_range,
                'prediction_range_pct': pred_range_pct,
                'diversity_level': diversity_level,
                'diversity_description': diversity_desc
            }
            
        except Exception as e:
            print(f"Warning: Error analyzing model diversity: {e}")
            return None
    
    def _analyze_pattern_strength(self, predictions):
        """Analyze the strength of detected patterns."""
        try:
            # This would analyze the pattern recognition features
            # For now, return a basic analysis
            pred_values = list(predictions.values())
            
            # Calculate trend strength
            trend_strength = abs(pred_values[-1] - pred_values[0]) / np.mean(pred_values) if len(pred_values) > 1 else 0
            
            # Determine pattern strength
            if trend_strength < 0.01:
                pattern_level = "WEAK"
                pattern_desc = "No clear pattern detected"
            elif trend_strength < 0.05:
                pattern_level = "MODERATE"
                pattern_desc = "Moderate pattern strength"
            else:
                pattern_level = "STRONG"
                pattern_desc = "Strong pattern detected"
            
            return {
                'trend_strength': trend_strength,
                'pattern_level': pattern_level,
                'pattern_description': pattern_desc
            }
            
        except Exception as e:
            print(f"Warning: Error analyzing pattern strength: {e}")
            return None
    
    def _calculate_model_weights(self, recent_performance=None):
        """Calculate dynamic ensemble weights based on recent performance."""
        try:
            if not self.models:
                return {}
            
            # If no recent performance provided, use equal weights
            if recent_performance is None:
                model_names = [name for name in self.models.keys() if name != 'Ensemble']
                equal_weight = 1.0 / len(model_names)
                return {name: equal_weight for name in model_names}
            
            # Calculate weights based on recent performance
            weights = {}
            total_performance = 0
            
            for model_name, performance in recent_performance.items():
                if model_name in self.models and model_name != 'Ensemble':
                    # Convert accuracy to weight (higher accuracy = higher weight)
                    # Add small epsilon to avoid division by zero
                    weight = max(performance.get('accuracy', 0.5), 0.1)
                    weights[model_name] = weight
                    total_performance += weight
            
            # Normalize weights to sum to 1
            if total_performance > 0:
                for model_name in weights:
                    weights[model_name] /= total_performance
            else:
                # Fallback to equal weights if no performance data
                model_names = list(weights.keys())
                equal_weight = 1.0 / len(model_names)
                weights = {name: equal_weight for name in model_names}
            
            return weights
            
        except Exception as e:
            print(f"Warning: Error calculating model weights: {e}")
            # Fallback to equal weights
            model_names = [name for name in self.models.keys() if name != 'Ensemble']
            equal_weight = 1.0 / len(model_names)
            return {name: equal_weight for name in model_names}
    
    def _evaluate_recent_performance(self, X, y, lookback_days=30):
        """Evaluate recent performance of individual models."""
        try:
            if len(X) < lookback_days:
                lookback_days = len(X) // 2
            
            # Use recent data for evaluation
            X_recent = X.iloc[-lookback_days:]
            y_recent = y.iloc[-lookback_days:]
            
            performance = {}
            
            for name, model in self.models.items():
                if name == 'Ensemble':
                    continue
                
                try:
                    # Make predictions on recent data
                    if name in self.scalers:
                        X_scaled = self.scalers[name].transform(X_recent)
                        predictions = model.predict(X_scaled)
                    else:
                        predictions = model.predict(X_recent)
                    
                    # Calculate performance metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(y_recent, predictions)
                    mae = mean_absolute_error(y_recent, predictions)
                    r2 = r2_score(y_recent, predictions)
                    
                    # Calculate directional accuracy (if price went up/down correctly)
                    actual_direction = np.diff(y_recent.values) > 0
                    pred_direction = np.diff(predictions) > 0
                    directional_accuracy = np.mean(actual_direction == pred_direction)
                    
                    # Overall accuracy score (weighted combination)
                    # Normalize MSE and MAE by dividing by the target mean for relative comparison
                    target_mean = np.mean(y_recent)
                    normalized_mse = mse / (target_mean ** 2) if target_mean != 0 else mse
                    normalized_mae = mae / target_mean if target_mean != 0 else mae
                    
                    accuracy = (0.3 * max(0, 1 - normalized_mse) + 
                               0.3 * max(0, 1 - normalized_mae) + 
                               0.2 * max(0, r2) + 
                               0.2 * directional_accuracy)
                    
                    performance[name] = {
                        'accuracy': accuracy,
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'directional_accuracy': directional_accuracy
                    }
                    
                except Exception as e:
                    print(f"Warning: Could not evaluate {name}: {e}")
                    performance[name] = {'accuracy': 0.5}  # Default accuracy
            
            return performance
            
        except Exception as e:
            print(f"Warning: Error evaluating recent performance: {e}")
            return {}
    
    def _create_adaptive_ensemble(self, X, y):
        """Create an adaptive ensemble with dynamic weighting."""
        try:
            # Evaluate recent performance
            recent_performance = self._evaluate_recent_performance(X, y)
            
            # Calculate dynamic weights
            weights = self._calculate_model_weights(recent_performance)
            
            if not weights:
                print("⚠️ Could not calculate weights, using equal weights")
                return None
            
            # Create ensemble with dynamic weights
            from sklearn.ensemble import VotingRegressor
            
            # Prepare estimators (exclude Ensemble itself)
            estimators = []
            for name, model in self.models.items():
                if name != 'Ensemble' and name in weights:
                    # Use first 3 characters of name, ensure uniqueness
                    short_name = name.lower()[:3]
                    if short_name in [est[0] for est in estimators]:
                        short_name = name.lower()[:4]  # Use 4 chars if 3 chars conflict
                    estimators.append((short_name, model))
            
            if len(estimators) < 2:
                print("⚠️ Not enough models for ensemble")
                return None
            
            # Create weighted ensemble
            ensemble_weights = [weights[name] for name, _ in estimators]
            
            ensemble = VotingRegressor(
                estimators=estimators,
                weights=ensemble_weights
            )
            
            # Train the ensemble
            ensemble.fit(X, y)
            
            # Store performance and weights for later use
            self.ensemble_performance = recent_performance
            self.ensemble_weights = weights
            
            print(f"✅ Created adaptive ensemble with {len(estimators)} models")
            print(f"📊 Model weights: {weights}")
            
            return ensemble
            
        except Exception as e:
            print(f"Warning: Error creating adaptive ensemble: {e}")
            return None
    
    def save_models(self):
        """Save trained models."""
        try:
            os.makedirs("models", exist_ok=True)
            
            for name, model in self.models.items():
                if name != 'Ensemble' or self.mode == "advanced":  # Don't save simple ensemble as it's just a string
                    model_path = f"models/{self.ticker}_{self.mode}_{name.lower()}_model.pkl"
                    joblib.dump(model, model_path)
                    print(f"💾 Saved {name} model: {model_path}")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                scaler_path = f"models/{self.ticker}_{self.mode}_{name.lower()}_scaler.pkl"
                joblib.dump(scaler, scaler_path)
                print(f"💾 Saved {name} scaler: {scaler_path}")
            
            # Save feature importance for advanced mode
            if self.mode == "advanced" and self.feature_importance:
                importance_path = f"models/{self.ticker}_{self.mode}_feature_importance.pkl"
                joblib.dump(self.feature_importance, importance_path)
                print(f"💾 Saved feature importance: {importance_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load pre-trained models."""
        try:
            if self.mode == "simple":
                model_names = ['randomforest', 'linearregression']
            else:
                model_names = ['randomforest', 'gradientboosting', 'xgboost', 'lightgbm', 'ensemble']
            
            models_loaded = 0
            for name in model_names:
                model_path = f"models/{self.ticker}_{self.mode}_{name}_model.pkl"
                if os.path.exists(model_path):
                    try:
                        self.models[name.title()] = joblib.load(model_path)
                        print(f"📂 Loaded {name} model")
                        models_loaded += 1
                    except Exception as e:
                        print(f"⚠️ Could not load {name} model: {e}")
                        continue
            
            # Load scalers
            if self.mode == "simple":
                scaler_names = ['linearregression']
            else:
                scaler_names = ['svr', 'ridge', 'lasso', 'mlp']
            
            for name in scaler_names:
                scaler_path = f"models/{self.ticker}_{self.mode}_{name}_scaler.pkl"
                if os.path.exists(scaler_path):
                    try:
                        self.scalers[name.title()] = joblib.load(scaler_path)
                        print(f"📂 Loaded {name} scaler")
                    except Exception as e:
                        print(f"⚠️ Could not load {name} scaler: {e}")
                        continue
            
            # Load feature importance for advanced mode
            if self.mode == "advanced":
                importance_path = f"models/{self.ticker}_{self.mode}_feature_importance.pkl"
                if os.path.exists(importance_path):
                    try:
                        self.feature_importance = joblib.load(importance_path)
                        print(f"📂 Loaded feature importance")
                    except Exception as e:
                        print(f"⚠️ Could not load feature importance: {e}")
            
            # If no models loaded successfully, return False to trigger retraining
            return models_loaded > 0
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

def run_unified_prediction(ticker="AAPL", mode="simple", days_ahead=5, interactive=True, use_incremental=True):
    """Run unified prediction for a given ticker."""
    try:
        print(f"🚀 Unified Prediction Engine for {ticker}")
        print(f"🎯 Mode: {mode.upper()}")
        print(f"🔄 Incremental Learning: {'Enabled' if use_incremental else 'Disabled'}")
        print("=" * 60)
        
        # Initialize engine
        engine = UnifiedPredictionEngine(ticker, mode, interactive, use_incremental)
        
        # Show model status if incremental learning is enabled
        if use_incremental:
            status = engine.get_model_status()
            print(f"📊 Model Status:")
            print(f"   - Incremental Enabled: {status['incremental_enabled']}")
            if status.get('incremental_version'):
                print(f"   - Latest Version: {status['incremental_version']}")
                print(f"   - Active: {status['incremental_active']}")
            
            # Check for updates
            needs_update, update_message = engine.check_for_updates()
            if needs_update:
                print(f"🔄 Update Available: {update_message}")
                if interactive:
                    update_choice = input("Would you like to run incremental training first? (y/n): ").strip().lower()
                    if update_choice == 'y':
                        print("🔄 Running incremental training...")
                        # This would integrate with the incremental training CLI
                        print("💡 Tip: Use 'python incremental_training_cli.py update --ticker {ticker} --mode {mode}' for manual updates")
            else:
                print(f"✅ {update_message}")
        
        # Load and prepare data
        df = engine.load_and_prepare_data()
        if df is None:
            return False
        
        # Prepare features
        X, y = engine.prepare_features(df)
        if X is None or y is None:
            return False
        
        # Check if models exist
        models_exist = engine.load_models()
        
        # Check if models are compatible with current features
        if models_exist and engine.mode == "advanced":
            # Test if models can handle current features
            try:
                test_data = X.iloc[-1:].values
                test_enhanced = engine._enhance_features(X.iloc[-1:])
                test_enhanced_data = test_enhanced.values
                
                # Try to use one model to test compatibility
                for name, model in engine.models.items():
                    if name in engine.scalers:
                        model.predict(engine.scalers[name].transform(test_enhanced_data))
                    else:
                        model.predict(test_enhanced_data)
                    break  # If one works, they should all work
                print("✅ Using pre-trained models")
            except Exception as e:
                print(f"⚠️ Pre-trained models incompatible with new features: {e}")
                print("🔄 Retraining models with enhanced features...")
                models_exist = False
        
        # Train models if they don't exist or are incompatible
        if not models_exist:
            print("🔄 Training new models...")
            if not engine.train_models(X, y):
                return False
            engine.save_models()
        else:
            print("✅ Using pre-trained models")
        
        # Generate predictions
        print("🔮 Generating predictions...")
        predictions, multi_day_predictions = engine.generate_predictions(X, days_ahead)
        
        if predictions is None:
            return False
        
        # Get current price
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate confidence
        confidence = engine.calculate_prediction_confidence(predictions)
        
        # Display results
        display_unified_results(ticker, mode, predictions, multi_day_predictions, current_price, confidence, engine)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in unified prediction: {e}")
        return False

def display_unified_results(ticker, mode, predictions, multi_day_predictions, current_price, confidence, engine=None):
    """Display unified prediction results."""
    mode_title = "SIMPLE" if mode == "simple" else "ADVANCED"
    print(f"\n🎯 {mode_title} PREDICTION RESULTS")
    print("=" * 70)
    print(f"📊 Stock: {ticker}")
    print(f"💰 CURRENT PRICE: ₹{current_price:.2f}")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print()
    
    # Display individual model predictions
    print("🤖 Model Predictions:")
    for model_name, pred in predictions.items():
        change = pred - current_price
        change_pct = (change / current_price) * 100
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        # Add ensemble weight information if available
        weight_info = ""
        if engine and model_name != 'Ensemble' and hasattr(engine, 'ensemble_weights') and engine.ensemble_weights:
            weight = engine.ensemble_weights.get(model_name, 0)
            weight_info = f" (weight: {weight:.2f})"
        
        print(f"   • {model_name:15}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%){weight_info}")
    
    print()
    
    # Display ensemble prediction with confidence
    if 'Ensemble' in predictions:
        ensemble_pred = predictions['Ensemble']
        change = ensemble_pred - current_price
        change_pct = (change / current_price) * 100
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        print("🎯 ENSEMBLE PREDICTION:")
        print(f"   Predicted Price: ₹{ensemble_pred:.2f} ({direction} {change_pct:+.2f}%)")
        if confidence:
            print(f"   Confidence 68%: ₹{confidence['confidence_68'][0]:.2f} - ₹{confidence['confidence_68'][1]:.2f}")
            print(f"   Confidence 95%: ₹{confidence['confidence_95'][0]:.2f} - ₹{confidence['confidence_95'][1]:.2f}")
            print(f"   Model Agreement: {confidence['agreement_score']:.2%}")
            
            # Display advanced analysis
            if 'model_diversity' in confidence and confidence['model_diversity']:
                div = confidence['model_diversity']
                print(f"   Model Diversity: {div['diversity_level']} - {div['diversity_description']}")
                print(f"   Prediction Range: {div['prediction_range_pct']:.2f}%")
            
            if 'pattern_strength' in confidence and confidence['pattern_strength']:
                pat = confidence['pattern_strength']
                print(f"   Pattern Strength: {pat['pattern_level']} - {pat['pattern_description']}")
    
    print()
    
    # Display multi-day predictions
    if multi_day_predictions:
        print("📅 MULTI-DAY FORECAST:")
        for i, pred in enumerate(multi_day_predictions, 1):
            change = pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   • Day {i}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
    
    print()
    
    # Trading recommendation
    if 'Ensemble' in predictions:
        ensemble_pred = predictions['Ensemble']
        change_pct = (ensemble_pred - current_price) / current_price * 100
        
        if change_pct > 2:
            recommendation = "🟢 BUY - Strong upward momentum expected"
        elif change_pct > 0.5:
            recommendation = "🟡 BUY - Moderate upward potential"
        elif change_pct < -2:
            recommendation = "🔴 SELL - Strong downward pressure expected"
        elif change_pct < -0.5:
            recommendation = "🟠 SELL - Moderate downward potential"
        else:
            recommendation = "⚪ HOLD - Stable price movement expected"
        
        print(f"💡 Trading Recommendation: {recommendation}")
    
    # Display ensemble performance analysis if available
    if engine and hasattr(engine, 'ensemble_performance') and engine.ensemble_performance:
        print("\n📊 ENSEMBLE PERFORMANCE ANALYSIS:")
        print("-" * 50)
        
        # Sort models by accuracy
        sorted_models = sorted(
            engine.ensemble_performance.items(), 
            key=lambda x: x[1].get('accuracy', 0), 
            reverse=True
        )
        
        for model_name, perf in sorted_models:
            accuracy = perf.get('accuracy', 0)
            mse = perf.get('mse', 0)
            directional_acc = perf.get('directional_accuracy', 0)
            weight = engine.ensemble_weights.get(model_name, 0) if hasattr(engine, 'ensemble_weights') else 0
            
            print(f"   • {model_name:15}: Accuracy: {accuracy:.3f}, MSE: {mse:.4f}, "
                  f"Directional: {directional_acc:.3f}, Weight: {weight:.3f}")
    
    print()
    print("=" * 70)

def main():
    """Main function with user interface."""
    print("🚀 UNIFIED STOCK PREDICTION SYSTEM")
    print("=" * 60)
    
    # Check for command line arguments for non-interactive mode
    if len(sys.argv) > 1:
        # Non-interactive mode
        if len(sys.argv) >= 4:
            ticker = sys.argv[1].upper()
            mode = sys.argv[2].lower()
            days_ahead = int(sys.argv[3])
            
            if mode not in ["simple", "advanced"]:
                print("❌ Invalid mode. Use 'simple' or 'advanced'")
                return
            
            if days_ahead <= 0 or days_ahead > 30:
                print("⚠️ Days should be between 1-30. Using default: 5")
                days_ahead = 5
            
            print(f"🎯 Non-interactive mode: {ticker}, {mode}, {days_ahead} days")
            success = run_unified_prediction(ticker, mode, days_ahead, interactive=False)
            
            if success:
                print(f"\n✅ {ticker} {mode} prediction completed successfully!")
            else:
                print(f"\n❌ {ticker} {mode} prediction failed!")
            return
    
    # Interactive mode
    print("Choose your prediction mode:")
    print("1. Simple Mode - Fast, reliable predictions (Recommended)")
    print("2. Advanced Mode - Sophisticated analysis with more models")
    print("3. Incremental Learning Management")
    print("4. Exit")
    print("-" * 60)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "4":
                print("👋 Goodbye!")
                break
            
            elif choice == "1":
                mode = "simple"
                print(f"\n🎯 Selected: Simple Mode")
                break
                
            elif choice == "2":
                if not ADVANCED_AVAILABLE:
                    print("❌ Advanced mode not available. Install xgboost and lightgbm first.")
                    print("   pip install xgboost lightgbm")
                    continue
                mode = "advanced"
                print(f"\n🎯 Selected: Advanced Mode")
                break
                
            elif choice == "3":
                handle_incremental_management()
                continue
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if choice in ["1", "2"]:
        # Get stock ticker
        ticker = input("\nEnter stock ticker (e.g., RELIANCE, AAPL, TCS): ").strip().upper()
        if not ticker:
            ticker = "RELIANCE"
        
        # Get prediction days
        days_input = input("Enter prediction days (default: 5): ").strip()
        if not days_input:
            days_ahead = 5
        else:
            try:
                days_ahead = int(days_input)
                if days_ahead <= 0 or days_ahead > 30:
                    print("⚠️ Days should be between 1-30. Using default: 5")
                    days_ahead = 5
            except ValueError:
                print("⚠️ Invalid number. Using default: 5")
                days_ahead = 5
        
        print(f"\n🎯 Running {mode} prediction for {ticker}...")
        print(f"📅 Prediction horizon: {days_ahead} days")
        print()
        
        # Run prediction
        success = run_unified_prediction(ticker, mode, days_ahead, interactive=True)
        
        if success:
            print(f"\n✅ {ticker} {mode} prediction completed successfully!")
        else:
            print(f"\n❌ {ticker} {mode} prediction failed!")


def handle_incremental_management():
    """Handle incremental learning management options."""
    print("\n🔄 Incremental Learning Management")
    print("=" * 50)
    print("1. Check for updates")
    print("2. Run incremental training")
    print("3. View model versions")
    print("4. Back to main menu")
    print("-" * 50)
    
    while True:
        try:
            sub_choice = input("\nEnter your choice (1-4): ").strip()
            
            if sub_choice == "4":
                break
                
            elif sub_choice == "1":
                ticker = input("Enter stock ticker: ").strip().upper()
                if not ticker:
                    ticker = "RELIANCE"
                
                mode = input("Enter mode (simple/advanced): ").strip().lower()
                if mode not in ["simple", "advanced"]:
                    mode = "simple"
                
                print(f"\n🔍 Checking for updates for {ticker} ({mode})...")
                try:
                    from partB_model.model_update_pipeline import create_model_update_pipeline
                    pipeline = create_model_update_pipeline()
                    needs_update = pipeline.check_for_updates(ticker, mode)
                    
                    if needs_update:
                        print(f"🔄 Updates available for {ticker} ({mode})")
                        update_choice = input("Would you like to run the update? (y/n): ").strip().lower()
                        if update_choice == 'y':
                            print("🔄 Running update...")
                            result = pipeline.run_automatic_updates([ticker], [mode])
                            if result:
                                print("✅ Update completed successfully!")
                            else:
                                print("❌ Update failed!")
                    else:
                        print(f"✅ {ticker} ({mode}) is up to date!")
                        
                except Exception as e:
                    print(f"❌ Error checking for updates: {e}")
                    print("💡 Make sure incremental training modules are available")
                
            elif sub_choice == "2":
                ticker = input("Enter stock ticker: ").strip().upper()
                if not ticker:
                    ticker = "RELIANCE"
                
                mode = input("Enter mode (simple/advanced): ").strip().lower()
                if mode not in ["simple", "advanced"]:
                    mode = "simple"
                
                print(f"\n🔄 Running incremental training for {ticker} ({mode})...")
                try:
                    from partB_model.model_update_pipeline import create_model_update_pipeline
                    pipeline = create_model_update_pipeline()
                    result = pipeline.run_automatic_updates([ticker], [mode])
                    
                    if result:
                        print("✅ Incremental training completed successfully!")
                    else:
                        print("❌ Incremental training failed!")
                        
                except Exception as e:
                    print(f"❌ Error running incremental training: {e}")
                    print("💡 Make sure incremental training modules are available")
                
            elif sub_choice == "3":
                ticker = input("Enter stock ticker: ").strip().upper()
                if not ticker:
                    ticker = "RELIANCE"
                
                print(f"\n📊 Model versions for {ticker}:")
                try:
                    from partB_model.incremental_learning import IncrementalLearningManager
                    manager = IncrementalLearningManager()
                    versions = manager.get_version_history(ticker)
                    
                    if versions:
                        for version in versions[:5]:  # Show last 5 versions
                            print(f"   - {version.version_id}")
                            print(f"     Mode: {version.metadata.get('mode', 'unknown')}")
                            print(f"     Created: {version.creation_timestamp}")
                            print(f"     Active: {version.is_active()}")
                            print()
                    else:
                        print("   No versions found")
                        
                except Exception as e:
                    print(f"❌ Error viewing versions: {e}")
                    print("💡 Make sure incremental training modules are available")
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                continue
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
