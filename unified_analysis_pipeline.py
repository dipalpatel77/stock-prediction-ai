#!/usr/bin/env python3
"""
Unified Analysis Pipeline
Integrates partA (preprocessing), partB (model), and partC (strategy) modules
for comprehensive stock prediction analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import threading
import time
from queue import Queue
import warnings
import multiprocessing
from functools import partial
warnings.filterwarnings('ignore')

# Import partA modules (preprocessing)
from partA_preprocessing.data_loader import load_data
from partA_preprocessing.preprocess import clean_data, add_technical_indicators

# Import partB modules (model)
from partB_model.enhanced_model_builder import EnhancedModelBuilder
from partB_model.enhanced_training import EnhancedStockPredictor

# Import partC modules (strategy)
from partC_strategy.optimized_technical_indicators import OptimizedTechnicalIndicators
from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
from partC_strategy.optimized_trading_strategy import OptimizedTradingStrategy
from partC_strategy.enhanced_market_factors import EnhancedMarketFactors
from partC_strategy.economic_indicators import EconomicIndicators
from partC_strategy.backtest import BacktestStrategy
from partC_strategy.sector_trend import SectorTrendAnalyzer

class UnifiedAnalysisPipeline:
    """
    Unified analysis pipeline that integrates all three parts:
    - partA: Data preprocessing and loading
    - partB: Model building and training
    - partC: Strategy analysis and trading signals
    """
    
    def __init__(self, ticker, max_workers=None):
        self.ticker = ticker.upper()
        # Auto-detect optimal number of workers
        if max_workers is None:
            self.max_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 threads
        else:
            self.max_workers = max_workers
        self.results_queue = Queue()
        self.error_queue = Queue()
        self.start_time = None
        self.thread_lock = threading.Lock()
        self.progress = 0
        self.total_tasks = 0
        
        # Initialize modules
        self.data_loader = None
        self.model_builder = EnhancedModelBuilder()
        self.enhanced_predictor = EnhancedStockPredictor(ticker)
        self.technical_analyzer = OptimizedTechnicalIndicators()
        self.sentiment_analyzer = OptimizedSentimentAnalyzer()
        self.trading_strategy = OptimizedTradingStrategy(self.sentiment_analyzer)
        self.market_factors = EnhancedMarketFactors()
        self.economic_indicators = EconomicIndicators()
        self.backtest_strategy = BacktestStrategy()
        self.sector_analyzer = SectorTrendAnalyzer()
        
    def run_unified_analysis(self, period='2y', days_ahead=5, use_enhanced=True):
        """Run complete unified analysis using all three parts."""
        self.start_time = time.time()
        self.days_ahead = days_ahead  # Store for later use
        
        print("🚀 Unified AI Stock Predictor - Complete Analysis Pipeline")
        print("=" * 80)
        print(f"📊 Ticker: {self.ticker}")
        print(f"📅 Period: {period}")
        print(f"⏰ Prediction Days: {days_ahead}")
        print(f"🔧 Threads: {self.max_workers}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print()
        
        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        try:
            # Step 1: PartA - Data Preprocessing
            print("📥 Step 1: PartA - Data Preprocessing")
            print("-" * 50)
            data_processing_success = self.run_partA_preprocessing(period)
            
            if not data_processing_success:
                print("❌ PartA data processing failed")
                return False
            
            # Step 2: PartB - Model Training
            print("\n🤖 Step 2: PartB - Model Training")
            print("-" * 50)
            model_training_success = self.run_partB_model_training()
            
            if not model_training_success:
                print("❌ PartB model training failed")
                return False
            
            # Step 3: PartC - Strategy Analysis
            print("\n📈 Step 3: PartC - Strategy Analysis")
            print("-" * 50)
            strategy_analysis_success = self.run_partC_strategy_analysis(use_enhanced)
            
            if not strategy_analysis_success:
                print("⚠️ PartC strategy analysis failed or timed out - continuing with predictions")
                # Don't return False, continue with predictions
            
            # Step 4: Unified Report Generation (Optional)
            print("\n📋 Step 4: Unified Report Generation")
            print("-" * 50)
            try:
                report_generation_success = self.run_unified_report_generation(days_ahead, use_enhanced)
                if not report_generation_success:
                    print("⚠️ Report generation failed - continuing with predictions")
            except Exception as e:
                print(f"⚠️ Report generation error: {e} - continuing with predictions")
            
            # Step 5: Generate and Display Predictions
            print("\n🔮 Step 5: Generating Predictions")
            print("-" * 50)
            prediction_success = self.generate_and_display_predictions(days_ahead)
            
            # Calculate execution time
            execution_time = time.time() - self.start_time
            
            # Display final results
            self.display_unified_results(execution_time, use_enhanced)
            
            return True
            
        except Exception as e:
            print(f"❌ Unified analysis failed: {e}")
            return False
    
    def run_partA_preprocessing(self, period):
        """Run partA data preprocessing with parallel processing."""
        print("🔄 Starting partA data preprocessing...")
        
        def load_stock_data():
            """Load stock data using partA data_loader."""
            try:
                # Check if raw data already exists
                raw_data_file = f"data/{self.ticker}_raw_data.csv"
                if os.path.exists(raw_data_file):
                    df = pd.read_csv(raw_data_file)
                    if not df.empty:
                        return True, f"Using existing raw data with {len(df)} records"
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*2)  # 2 years
                
                # Use partA data loader
                df = load_data(self.ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if not df.empty:
                    df.to_csv(raw_data_file)
                    return True, f"Loaded {len(df)} records using partA data_loader"
                else:
                    return False, "No data loaded"
            except Exception as e:
                return False, f"Data loading error: {e}"
        
        def clean_and_preprocess():
            """Clean and preprocess data using partA preprocess module."""
            try:
                # Load raw data
                raw_data_file = f"data/{self.ticker}_raw_data.csv"
                if not os.path.exists(raw_data_file):
                    return False, "No raw data file found"
                
                # Read the raw data with proper handling
                df = pd.read_csv(raw_data_file)
                
                # Handle the malformed data structure - skip the first 3 rows for AAPL format
                if len(df) > 3 and df.iloc[0, 0] == 'Price' and df.iloc[1, 0] == 'Ticker':
                    # Skip the first 3 rows and use row 3 as header
                    df = pd.read_csv(raw_data_file, skiprows=3)
                    df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                elif len(df) > 2 and df.iloc[0, 0] == 'Price':
                    # Skip the first 2 rows and use row 2 as header (TSLA format)
                    df = pd.read_csv(raw_data_file, skiprows=2)
                    df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                
                # Convert numeric columns to proper types
                numeric_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Handle the specific format of the raw data
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif df.index.name == 'Date':
                    df.index = pd.to_datetime(df.index)
                
                # Use partA preprocess functions
                df = clean_data(df)
                df = add_technical_indicators(df)
                
                # Save preprocessed data
                df.to_csv(f"data/{self.ticker}_partA_preprocessed.csv")
                return True, f"Preprocessed {len(df)} records using partA modules"
            except Exception as e:
                return False, f"Preprocessing error: {e}"
        
        def add_enhanced_technical_indicators():
            """Add enhanced technical indicators using partC."""
            try:
                # Load partA preprocessed data
                data_file = f"data/{self.ticker}_partA_preprocessed.csv"
                if not os.path.exists(data_file):
                    return False, "No partA preprocessed data found"
                
                df = pd.read_csv(data_file)
                
                # Remove any unnamed columns that contain row numbers
                unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                if unnamed_cols:
                    df = df.drop(unnamed_cols, axis=1)
                
                # Handle the specific format where the second column contains dates
                if len(df.columns) > 1 and df.columns[1] == 'Price':
                    # The second column is 'Price' which contains dates
                    df['Date'] = df['Price']
                    df = df.drop('Price', axis=1)
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                elif len(df.columns) > 1 and df.columns[1] == 'Date':
                    # The second column is 'Date' which contains dates
                    df['Date'] = df['Date']
                    df = df.drop('Date', axis=1)
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                elif len(df.columns) > 1 and df.columns[1] == 'Price' and df.columns[2] == 'Adj Close':
                    # Special case for AAPL format where second column is 'Price' (dates) and third is 'Adj Close'
                    df['Date'] = df['Price']
                    df = df.drop('Price', axis=1)
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    # Standard format - set the first column as index
                    df.set_index(df.columns[0], inplace=True)
                    df.index = pd.to_datetime(df.index)
                
                # Add enhanced indicators using partC
                df_enhanced = self.technical_analyzer.add_all_indicators(df)
                
                # Save enhanced data
                df_enhanced.to_csv(f"data/{self.ticker}_partA_partC_enhanced.csv")
                return True, f"Added enhanced indicators to {len(df_enhanced)} records"
            except Exception as e:
                return False, f"Enhanced indicators error: {e}"
        
        # Run partA tasks sequentially to ensure proper data flow
        results = {}
        
        # Step 1: Load data
        print("🔄 Step 1: Loading stock data...")
        success, message = load_stock_data()
        results["load_data"] = (success, message)
        status = "✅" if success else "❌"
        print(f"{status} load_data: {message}")
        
        if not success:
            return False
        
        # Step 2: Preprocess data
        print("🔄 Step 2: Preprocessing data...")
        success, message = clean_and_preprocess()
        results["preprocess"] = (success, message)
        status = "✅" if success else "❌"
        print(f"{status} preprocess: {message}")
        
        if not success:
            return False
        
        # Step 3: Add enhanced indicators
        print("🔄 Step 3: Adding enhanced indicators...")
        success, message = add_enhanced_technical_indicators()
        results["enhance_indicators"] = (success, message)
        status = "✅" if success else "❌"
        print(f"{status} enhance_indicators: {message}")
        
        # Check if main data processing succeeded
        return results.get("load_data", (False, "Task not completed"))[0]
    
    def run_partB_model_training(self):
        """Run partB model training with parallel processing."""
        print("🔄 Starting partB model training...")
        
        def train_enhanced_model():
            """Train enhanced model using partB enhanced_training."""
            try:
                # Check if enhanced model already exists
                enhanced_model_path = f"models/{self.ticker}_enhanced.h5"
                if os.path.exists(enhanced_model_path):
                    return True, "Enhanced model already exists"
                
                # Load enhanced data
                data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
                if not os.path.exists(data_file):
                    return False, "No enhanced data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Use partB enhanced predictor with timeout
                df_prepared = self.enhanced_predictor.prepare_enhanced_data(df)
                
                # Set a timeout for training (5 minutes)
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Enhanced model training timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minutes timeout
                
                try:
                    self.enhanced_predictor.train_enhanced_model(df_prepared)
                    signal.alarm(0)  # Cancel timeout
                    
                    # Save model
                    self.enhanced_predictor.save_model()
                    
                    return True, "Enhanced model trained using partB"
                except TimeoutError:
                    signal.alarm(0)  # Cancel timeout
                    return False, "Enhanced model training timed out"
                    
            except Exception as e:
                return False, f"Enhanced model training error: {e}"
        
        def train_ensemble_models():
            """Train ensemble models using partB model_builder."""
            try:
                # Load enhanced data
                data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
                if not os.path.exists(data_file):
                    return False, "No enhanced data available"
                
                df = pd.read_csv(data_file)
                
                # Remove any unnamed columns that contain row numbers
                unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                if unnamed_cols:
                    df = df.drop(unnamed_cols, axis=1)
                
                # Handle the specific format where the second column contains dates
                if len(df.columns) > 1 and df.columns[1] == 'Price':
                    # The second column is 'Price' which contains dates
                    df['Date'] = df['Price']
                    df = df.drop('Price', axis=1)
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                elif len(df.columns) > 1 and df.columns[1] == 'Date':
                    # The second column is 'Date' which contains dates
                    df['Date'] = df['Date']
                    df = df.drop('Date', axis=1)
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                elif len(df.columns) > 1 and df.columns[1] == 'Price' and df.columns[2] == 'Adj Close':
                    # Special case for AAPL format where second column is 'Price' (dates) and third is 'Adj Close'
                    df['Date'] = df['Price']
                    df = df.drop('Price', axis=1)
                    df.set_index('Date', inplace=True)
                    df.index = pd.to_datetime(df.index)
                else:
                    # Standard format - set the first column as index
                    df.set_index(df.columns[0], inplace=True)
                    df.index = pd.to_datetime(df.index)
                
                # Prepare features - use essential columns and basic indicators
                essential_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
                basic_indicators = ['SMA_10', 'SMA_20', 'RSI_14', 'MACD']
                
                # Get available columns
                available_essential = [col for col in essential_cols if col in df.columns]
                available_indicators = [col for col in basic_indicators if col in df.columns]
                
                if not available_essential:
                    return False, "No essential columns available"
                
                # Use essential columns + basic indicators
                feature_cols = available_essential + available_indicators
                X = df[feature_cols].dropna()
                
                if len(X) == 0:
                    return False, "No valid data for training after filtering"
                
                # Create target variable (next day's close price)
                if 'Close' in X.columns:
                    y = X['Close'].shift(-1).dropna()
                    X = X[:-1]  # Remove last row since we don't have target
                    
                    # Ensure X and y have the same length
                    if len(X) != len(y):
                        min_len = min(len(X), len(y))
                        X = X.iloc[:min_len]
                        y = y.iloc[:min_len]
                    
                    print(f"Training data shape: X={X.shape}, y={y.shape}")
                else:
                    return False, "Close price column not found"
                
                # Use partB model builder
                ensemble_models = self.model_builder.create_ensemble_ml(X, y)
                
                # Save ensemble models
                for name, model in ensemble_models.items():
                    import joblib
                    joblib.dump(model, f"models/{self.ticker}_{name}_model.pkl")
                
                return True, "Ensemble models trained using partB"
            except Exception as e:
                return False, f"Ensemble training error: {e}"
        
        def validate_models():
            """Validate trained models."""
            try:
                # Check if models exist
                enhanced_exists = os.path.exists(f"models/{self.ticker}_enhanced.h5")
                rf_exists = os.path.exists(f"models/{self.ticker}_random_forest_model.pkl")
                gb_exists = os.path.exists(f"models/{self.ticker}_gradient_boost_model.pkl")
                
                if enhanced_exists and (rf_exists or gb_exists):
                    return True, "Models validated successfully"
                else:
                    return False, "Required models not found"
            except Exception as e:
                return False, f"Model validation error: {e}"
        
        # Run partB tasks in parallel
        tasks = [
            ("enhanced_model", train_enhanced_model),
            ("ensemble_models", train_ensemble_models),
            ("model_validation", validate_models)
        ]
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(task[1]): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    success, message = future.result()
                    results[task_name] = (success, message)
                    status = "✅" if success else "❌"
                    print(f"{status} {task_name}: {message}")
                except Exception as e:
                    results[task_name] = (False, f"Exception: {e}")
                    print(f"❌ {task_name}: Exception - {e}")
        
        # Check if main model training succeeded
        model_training_success = results.get("enhanced_model", (False, "Task not completed"))[0]
        
        # Generate predictions if model training was successful OR if ensemble models exist
        ensemble_models_exist = (os.path.exists(f"models/{self.ticker}_random_forest_model.pkl") or 
                               os.path.exists(f"models/{self.ticker}_gradient_boost_model.pkl"))
        
        if model_training_success or ensemble_models_exist:
            try:
                # Get days_ahead from the class or use default
                days_ahead = getattr(self, 'days_ahead', 5)
                print(f"\n🔮 Generating predictions using available models...")
                self.generate_and_display_predictions(days_ahead)
            except Exception as e:
                print(f"Warning: Prediction generation failed: {e}")
        
        return model_training_success or ensemble_models_exist
    
    def run_partC_strategy_analysis(self, use_enhanced):
        """Run partC strategy analysis with parallel processing."""
        print("🔄 Starting partC strategy analysis...")
        
        def run_sentiment_analysis():
            """Run sentiment analysis using partC."""
            try:
                # Add timeout to prevent infinite loops
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Sentiment analysis timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)  # 1 minute timeout
                
                try:
                    sentiment_df = self.sentiment_analyzer.analyze_stock_sentiment(self.ticker, days_back=30)
                    signal.alarm(0)  # Cancel timeout
                    
                    if not sentiment_df.empty:
                        sentiment_df.to_csv(f"data/{self.ticker}_sentiment_analysis.csv", index=False)
                        return True, "Sentiment analysis completed using partC"
                    else:
                        return False, "No sentiment data available"
                except TimeoutError:
                    signal.alarm(0)  # Cancel timeout
                    return False, "Sentiment analysis timed out"
                    
            except Exception as e:
                return False, f"Sentiment analysis error: {e}"
        
        def run_market_factors():
            """Run market factors analysis using partC."""
            try:
                market_data = self.market_factors.get_market_factors()
                
                if market_data:
                    market_df = pd.DataFrame([market_data])
                    market_df.to_csv(f"data/{self.ticker}_market_factors.csv", index=False)
                    return True, "Market factors completed using partC"
                else:
                    return False, "No market data available"
            except Exception as e:
                return False, f"Market factors error: {e}"
        
        def run_economic_indicators():
            """Run economic indicators analysis using partC."""
            try:
                economic_data = self.economic_indicators.get_all_indicators()
                
                if economic_data:
                    economic_df = pd.DataFrame([economic_data])
                    economic_df.to_csv(f"data/{self.ticker}_economic_indicators.csv", index=False)
                    return True, "Economic indicators completed using partC"
                else:
                    return False, "No economic data available"
            except Exception as e:
                return False, f"Economic indicators error: {e}"
        
        def run_trading_strategy():
            """Run trading strategy analysis using partC."""
            try:
                # Load enhanced data
                data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
                if not os.path.exists(data_file):
                    return False, "No enhanced data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Add timeout to prevent infinite loops
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Trading strategy analysis timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 minutes timeout
                
                try:
                    # Generate signals using partC
                    signals_df = self.trading_strategy.generate_enhanced_signals(self.ticker, df)
                    signal.alarm(0)  # Cancel timeout
                    
                    if not signals_df.empty:
                        signals_df.to_csv(f"data/{self.ticker}_trading_signals.csv")
                        return True, "Trading strategy completed using partC"
                    else:
                        return False, "No signals generated"
                except TimeoutError:
                    signal.alarm(0)  # Cancel timeout
                    return False, "Trading strategy analysis timed out"
                    
            except Exception as e:
                return False, f"Trading strategy error: {e}"
        
        def run_backtesting():
            """Run backtesting analysis using partC."""
            try:
                # Load signals data
                signals_file = f"data/{self.ticker}_trading_signals.csv"
                if not os.path.exists(signals_file):
                    return False, "No signals data available"
                
                signals_df = pd.read_csv(signals_file, index_col=0, parse_dates=True)
                
                # Run backtest using partC
                results = self.backtest_strategy.run_backtest(signals_df)
                
                if results:
                    results_df = pd.DataFrame([results])
                    results_df.to_csv(f"data/{self.ticker}_backtest_results.csv", index=False)
                    return True, "Backtesting completed using partC"
                else:
                    return False, "No backtest results"
            except Exception as e:
                return False, f"Backtesting error: {e}"
        
        # Run partC tasks in parallel
        tasks = [
            ("sentiment", run_sentiment_analysis),
            ("market_factors", run_market_factors),
            ("economic_indicators", run_economic_indicators),
            ("trading_strategy", run_trading_strategy),
            ("backtesting", run_backtesting)
        ]
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(task[1]): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    success, message = future.result()
                    results[task_name] = (success, message)
                    status = "✅" if success else "❌"
                    print(f"{status} {task_name}: {message}")
                except Exception as e:
                    results[task_name] = (False, f"Exception: {e}")
                    print(f"❌ {task_name}: Exception - {e}")
        
        return any(results.values())
    
    def run_unified_report_generation(self, days_ahead, use_enhanced):
        """Run unified report generation."""
        print("🔄 Starting unified report generation...")
        
        def generate_unified_summary():
            """Generate unified summary report."""
            try:
                data_files = [f for f in os.listdir('data') if f.startswith(self.ticker)]
                model_files = [f for f in os.listdir('models') if f.startswith(self.ticker)]
                
                summary = {
                    'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Ticker': self.ticker,
                    'Data_Files_Generated': len(data_files),
                    'Model_Files_Generated': len(model_files),
                    'Enhanced_Features': use_enhanced,
                    'Parts_Used': 'partA, partB, partC',
                    'Status': 'Completed'
                }
                
                summary_df = pd.DataFrame([summary])
                summary_df.to_csv(f'data/{self.ticker}_unified_summary.csv', index=False)
                return True, f"Unified summary generated ({len(data_files)} data files, {len(model_files)} model files)"
            except Exception as e:
                return False, f"Unified summary error: {e}"
        
        def generate_performance_report():
            """Generate performance report."""
            try:
                # Load backtest results
                backtest_file = f"data/{self.ticker}_backtest_results.csv"
                if os.path.exists(backtest_file):
                    backtest_df = pd.read_csv(backtest_file)
                    
                    performance = {
                        'Ticker': self.ticker,
                        'Total_Return': backtest_df['total_return'].iloc[0] if 'total_return' in backtest_df.columns else 'N/A',
                        'Sharpe_Ratio': backtest_df['sharpe_ratio'].iloc[0] if 'sharpe_ratio' in backtest_df.columns else 'N/A',
                        'Max_Drawdown': backtest_df['max_drawdown'].iloc[0] if 'max_drawdown' in backtest_df.columns else 'N/A',
                        'Win_Rate': backtest_df['win_rate'].iloc[0] if 'win_rate' in backtest_df.columns else 'N/A',
                        'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    performance_df = pd.DataFrame([performance])
                    performance_df.to_csv(f'data/{self.ticker}_performance_report.csv', index=False)
                    return True, "Performance report generated"
                else:
                    return False, "No backtest results available"
            except Exception as e:
                return False, f"Performance report error: {e}"
        
        # Run report generation tasks in parallel
        tasks = [
            ("unified_summary", generate_unified_summary),
            ("performance_report", generate_performance_report)
        ]
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(task[1]): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    success, message = future.result()
                    results[task_name] = (success, message)
                    status = "✅" if success else "❌"
                    print(f"{status} {task_name}: {message}")
                except Exception as e:
                    results[task_name] = (False, f"Exception: {e}")
                    print(f"❌ {task_name}: Exception - {e}")
        
        return any(results.values())
    
    def generate_and_display_predictions(self, days_ahead=5):
        """Generate and display predictions for the stock."""
        print("🔄 Starting prediction generation...")
        
        try:
            # Load enhanced data
            data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
            if not os.path.exists(data_file):
                print("❌ No enhanced data available for predictions")
                return False
            
            df = pd.read_csv(data_file)
            
            # Remove any unnamed columns that contain row numbers
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(unnamed_cols, axis=1)
            
            # Handle the specific format where the second column contains dates
            if len(df.columns) > 1 and df.columns[1] == 'Price':
                df['Date'] = df['Price']
                df = df.drop('Price', axis=1)
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
            elif len(df.columns) > 1 and df.columns[1] == 'Date':
                df['Date'] = df['Date']
                df = df.drop('Date', axis=1)
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
            elif len(df.columns) > 1 and df.columns[1] == 'Price' and df.columns[2] == 'Adj Close':
                df['Date'] = df['Price']
                df = df.drop('Price', axis=1)
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
            else:
                df.set_index(df.columns[0], inplace=True)
                df.index = pd.to_datetime(df.index)
            
            # Generate predictions using ensemble models
            print("📊 Generating ensemble predictions...")
            ensemble_predictions = self.generate_ensemble_predictions(df, days_ahead)
            
            # Generate predictions using enhanced model
            print("🚀 Generating enhanced model predictions...")
            enhanced_predictions = self.generate_enhanced_predictions(df, days_ahead)
            
            # Display predictions
            self.display_predictions(ensemble_predictions, enhanced_predictions, days_ahead)
            
            # Save predictions
            self.save_predictions(ensemble_predictions, enhanced_predictions)
            
            print("✅ Prediction generation completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Prediction generation error: {e}")
            return False
    
    def generate_ensemble_predictions(self, df, days_ahead):
        """Generate predictions using ensemble models."""
        try:
            # Load ensemble models
            rf_model_path = f"models/{self.ticker}_random_forest_model.pkl"
            gb_model_path = f"models/{self.ticker}_gradient_boost_model.pkl"
            
            if not (os.path.exists(rf_model_path) or os.path.exists(gb_model_path)):
                return None
            
            import joblib
            
            # Prepare features
            essential_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            basic_indicators = ['SMA_10', 'SMA_20', 'RSI_14', 'MACD']
            
            available_essential = [col for col in essential_cols if col in df.columns]
            available_indicators = [col for col in basic_indicators if col in df.columns]
            
            if not available_essential:
                return None
            
            feature_cols = available_essential + available_indicators
            X = df[feature_cols].dropna()
            
            if len(X) == 0:
                return None
            
            # Get last data point for prediction
            last_data = X.iloc[-1:].values
            
            predictions = []
            
            # Random Forest predictions
            if os.path.exists(rf_model_path):
                rf_model = joblib.load(rf_model_path)
                rf_pred = rf_model.predict(last_data)[0]
                predictions.append(('Random Forest', rf_pred))
            
            # Gradient Boost predictions
            if os.path.exists(gb_model_path):
                gb_model = joblib.load(gb_model_path)
                gb_pred = gb_model.predict(last_data)[0]
                predictions.append(('Gradient Boost', gb_pred))
            
            # SVR predictions
            svr_model_path = f"models/{self.ticker}_svr_model.pkl"
            if os.path.exists(svr_model_path):
                svr_model = joblib.load(svr_model_path)
                svr_pred = svr_model.predict(last_data)[0]
                predictions.append(('SVR', svr_pred))
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Ensemble prediction error: {e}")
            return None
    
    def generate_enhanced_predictions(self, df, days_ahead):
        """Generate predictions using enhanced model."""
        try:
            enhanced_model_path = f"models/{self.ticker}_enhanced.h5"
            
            if not os.path.exists(enhanced_model_path):
                return None
            
            # Use the enhanced predictor to make predictions
            predictions = self.enhanced_predictor.predict_enhanced(df, days_ahead)
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Enhanced prediction error: {e}")
            return None
    
    def display_predictions(self, ensemble_predictions, enhanced_predictions, days_ahead):
        """Display the generated predictions."""
        # Get current price from the data
        data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
        df = pd.read_csv(data_file)
        
        # Handle data format
        if len(df.columns) > 1 and df.columns[1] == 'Price':
            current_price = float(df.iloc[-1]['Close'])
        else:
            current_price = float(df.iloc[-1]['Close'])
        
        print("\n🎯 PREDICTION RESULTS")
        print("=" * 60)
        print(f"📊 Stock: {self.ticker}")
        print(f"📅 Prediction Period: {days_ahead} days")
        print(f"💰 CURRENT PRICE: ₹{current_price:.2f}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        print()
        
        # Display ensemble predictions
        if ensemble_predictions:
            print("🤖 Ensemble Model Predictions:")
            for model_name, pred in ensemble_predictions:
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   • {model_name}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        # Display enhanced predictions
        if enhanced_predictions:
            print(f"\n🚀 Enhanced Model Predictions ({days_ahead} days):")
            for i, pred in enumerate(enhanced_predictions, 1):
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   • Day {i}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        # Calculate average prediction
        all_predictions = []
        if ensemble_predictions:
            all_predictions.extend([pred for _, pred in ensemble_predictions])
        if enhanced_predictions:
            all_predictions.extend(enhanced_predictions)
        
        if all_predictions:
            avg_prediction = sum(all_predictions) / len(all_predictions)
            avg_change = avg_prediction - current_price
            avg_change_pct = (avg_change / current_price) * 100
            avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
            
            print(f"\n📊 Average Prediction: ₹{avg_prediction:.2f} ({avg_direction} {avg_change_pct:+.2f}%)")
            
            # Price summary
            print(f"\n📋 PRICE SUMMARY:")
            print(f"   Current Price: ₹{current_price:.2f}")
            print(f"   Predicted Price: ₹{avg_prediction:.2f}")
            print(f"   Expected Change: {avg_change_pct:+.2f}%")
            
            # Trading recommendation
            if avg_change_pct > 2:
                recommendation = "🟢 BUY - Strong upward momentum expected"
            elif avg_change_pct > 0.5:
                recommendation = "🟡 BUY - Moderate upward potential"
            elif avg_change_pct < -2:
                recommendation = "🔴 SELL - Strong downward pressure expected"
            elif avg_change_pct < -0.5:
                recommendation = "🟠 SELL - Moderate downward potential"
            else:
                recommendation = "⚪ HOLD - Stable price movement expected"
            
            print(f"\n💡 Trading Recommendation: {recommendation}")
    
    def save_predictions(self, ensemble_predictions, enhanced_predictions):
        """Save predictions to file."""
        try:
            # Get current price
            data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
            df = pd.read_csv(data_file)
            current_price = float(df.iloc[-1]['Close'])
            
            predictions_data = {
                'Date': [datetime.now().strftime('%Y-%m-%d')],
                'Ticker': [self.ticker],
                'Current_Price': [current_price],
            }
            
            if ensemble_predictions:
                for model_name, pred in ensemble_predictions:
                    predictions_data[f'{model_name}_Prediction'] = [pred]
            
            if enhanced_predictions:
                for i, pred in enumerate(enhanced_predictions, 1):
                    predictions_data[f'Enhanced_Day_{i}'] = [pred]
            
            predictions_df = pd.DataFrame(predictions_data)
            predictions_df.to_csv(f"data/{self.ticker}_latest_predictions.csv", index=False)
            print(f"\n💾 Predictions saved to: data/{self.ticker}_latest_predictions.csv")
            
        except Exception as e:
            print(f"Warning: Could not save predictions: {e}")
    
    def display_unified_results(self, execution_time, use_enhanced):
        """Display unified analysis results."""
        print("\n" + "=" * 80)
        print("🎯 UNIFIED ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"📊 Ticker: {self.ticker}")
        print(f"⏱️ Execution Time: {execution_time:.2f} seconds")
        print(f"🔧 Threads Used: {self.max_workers}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print()
        
        print("📁 Generated Files:")
        data_files = [f for f in os.listdir('data') if f.startswith(self.ticker)]
        model_files = [f for f in os.listdir('models') if f.startswith(self.ticker)]
        
        print("📊 Data Files:")
        for file in sorted(data_files):
            print(f"   • {file}")
        
        print("🤖 Model Files:")
        for file in sorted(model_files):
            print(f"   • {file}")
        
        print("\n🔗 Parts Integration:")
        print("   ✅ partA: Data preprocessing and loading")
        print("   ✅ partB: Model building and training")
        print("   ✅ partC: Strategy analysis and trading signals")
        
        print("\n🚀 Next Steps:")
        print("   1. Review trading signals in data/{ticker}_trading_signals.csv")
        print("   2. Check performance in data/{ticker}_performance_report.csv")
        print("   3. Analyze unified summary in data/{ticker}_unified_summary.csv")

def main():
    """Main function for unified analysis pipeline."""
    print("🚀 Unified AI Stock Predictor")
    print("=" * 50)
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
    
    period = input("Enter data period (default: 2y): ").strip()
    if not period:
        period = "2y"
    
    days_ahead = input("Enter prediction days (default: 5): ").strip()
    if not days_ahead:
        days_ahead = 5
    else:
        try:
            days_ahead = int(days_ahead)
        except ValueError:
            days_ahead = 5
    
    use_enhanced = input("Use enhanced features? (y/n, default: y): ").strip().lower()
    use_enhanced = use_enhanced != 'n'
    
    max_workers = input("Enter number of threads (default: auto): ").strip()
    if not max_workers:
        max_workers = None
    else:
        try:
            max_workers = int(max_workers)
        except ValueError:
            max_workers = None
    
    print(f"\n🎯 Starting unified analysis for {ticker}...")
    
    # Create and run unified pipeline
    pipeline = UnifiedAnalysisPipeline(ticker, max_workers)
    success = pipeline.run_unified_analysis(period, days_ahead, use_enhanced)
    
    if success:
        print("\n✅ Unified analysis completed successfully!")
    else:
        print("\n❌ Unified analysis failed!")

if __name__ == "__main__":
    main()
