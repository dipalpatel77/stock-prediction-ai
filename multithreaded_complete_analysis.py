#!/usr/bin/env python3
"""
Enhanced Multi-threaded Complete Analysis Runner
Runs the entire enhanced stock prediction pipeline with optimized parallel processing
Now integrates with unified pipeline for better partA, partB, partC integration
"""

import os
import sys
import subprocess
from datetime import datetime
import pandas as pd
import concurrent.futures
import threading
import time
from queue import Queue
import warnings
import multiprocessing
from functools import partial
import numpy as np
warnings.filterwarnings('ignore')

# Import unified pipeline
from unified_analysis_pipeline import UnifiedAnalysisPipeline

class EnhancedMultiThreadedAnalysisRunner:
    """Enhanced multi-threaded analysis runner with optimized parallel processing."""
    
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
        
        # Initialize unified pipeline
        self.unified_pipeline = UnifiedAnalysisPipeline(ticker, max_workers)
        
    def run_parallel_analysis(self, period='2y', days_ahead=5, use_enhanced=True):
        """Run complete analysis with parallel processing using unified pipeline."""
        self.start_time = time.time()
        
        print("🚀 Multi-threaded AI Stock Predictor - Complete Analysis Pipeline")
        print("=" * 80)
        print(f"📊 Ticker: {self.ticker}")
        print(f"📅 Period: {period}")
        print(f"⏰ Prediction Days: {days_ahead}")
        print(f"🔧 Threads: {self.max_workers}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print("🔗 Using Unified Pipeline (partA + partB + partC)")
        print()
        
        try:
            # Use unified pipeline for complete analysis
            success = self.unified_pipeline.run_unified_analysis(period, days_ahead, use_enhanced)
            
            # Calculate execution time
            execution_time = time.time() - self.start_time
            
            if success:
                print(f"\n✅ Unified analysis completed in {execution_time:.2f} seconds")
            else:
                print(f"\n❌ Unified analysis failed after {execution_time:.2f} seconds")
            
            return success
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return False
    
    def update_progress(self, task_name, success=True):
        """Update progress with thread safety."""
        with self.thread_lock:
            self.progress += 1
            percentage = (self.progress / self.total_tasks) * 100
            status = "✅" if success else "❌"
            print(f"{status} [{percentage:.1f}%] {task_name}")
    
    def run_parallel_data_processing(self, period):
        """Run data processing in parallel with enhanced threading."""
        print("🔄 Starting enhanced parallel data processing...")
        
        def download_and_preprocess():
            """Download and preprocess data with progress tracking."""
            try:
                # Use unified pipeline for data processing
                from unified_analysis_pipeline import UnifiedAnalysisPipeline
                
                pipeline = UnifiedAnalysisPipeline(self.ticker, max_workers=self.max_workers)
                success = pipeline.run_partA_preprocessing(period)
                
                if success:
                    self.update_progress("Main data processing")
                    return True, "Data downloaded and preprocessed using unified pipeline"
                else:
                    self.update_progress("Main data processing", False)
                    return False, "Data processing failed"
            except Exception as e:
                self.update_progress("Main data processing", False)
                return False, f"Data processing error: {e}"
        
        def download_sector_data():
            """Download sector comparison data with parallel downloads."""
            try:
                import yfinance as yf
                
                # Download sector ETFs for comparison
                sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI']  # Tech, Finance, Energy, Healthcare, Industrial
                sector_data = {}
                
                # Parallel download of sector data
                def download_single_etf(etf):
                    try:
                        etf_data = yf.download(etf, period=period, progress=False)
                        return etf, etf_data['Close']
                    except:
                        return etf, None
                
                # Use ThreadPoolExecutor for parallel downloads
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sector_etfs), 5)) as executor:
                    futures = [executor.submit(download_single_etf, etf) for etf in sector_etfs]
                    for future in concurrent.futures.as_completed(futures):
                        etf, data = future.result()
                        if data is not None:
                            sector_data[etf] = data
                
                if sector_data:
                    sector_df = pd.DataFrame(sector_data)
                    sector_df.to_csv(f"data/sector_comparison_{self.ticker}.csv")
                    self.update_progress("Sector data processing")
                    return True, "Sector data downloaded"
                self.update_progress("Sector data processing", False)
                return False, "No sector data available"
            except Exception as e:
                self.update_progress("Sector data processing", False)
                return False, f"Sector data error: {e}"
        
        def download_market_data():
            """Download market-wide data with parallel downloads."""
            try:
                import yfinance as yf
                
                # Download market indices
                market_data = {}
                indices = ['^GSPC', '^VIX', '^TNX']  # S&P 500, VIX, 10Y Treasury
                
                # Parallel download of market data
                def download_single_index(index):
                    try:
                        index_data = yf.download(index, period=period, progress=False)
                        return index, index_data['Close']
                    except:
                        return index, None
                
                # Use ThreadPoolExecutor for parallel downloads
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as executor:
                    futures = [executor.submit(download_single_index, index) for index in indices]
                    for future in concurrent.futures.as_completed(futures):
                        index, data = future.result()
                        if data is not None:
                            market_data[index] = data
                
                if market_data:
                    market_df = pd.DataFrame(market_data)
                    market_df.to_csv(f"data/market_data_{self.ticker}.csv")
                    self.update_progress("Market data processing")
                    return True, "Market data downloaded"
                self.update_progress("Market data processing", False)
                return False, "No market data available"
            except Exception as e:
                self.update_progress("Market data processing", False)
                return False, f"Market data error: {e}"
        
        def download_technical_indicators():
            """Download additional technical indicators data."""
            try:
                import yfinance as yf
                
                # Download additional technical data
                tech_data = {}
                tech_symbols = ['^TNX', '^VIX', '^DXY']  # Treasury, VIX, Dollar Index
                
                def download_tech_symbol(symbol):
                    try:
                        data = yf.download(symbol, period=period, progress=False)
                        return symbol, data['Close']
                    except:
                        return symbol, None
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(tech_symbols)) as executor:
                    futures = [executor.submit(download_tech_symbol, symbol) for symbol in tech_symbols]
                    for future in concurrent.futures.as_completed(futures):
                        symbol, data = future.result()
                        if data is not None:
                            tech_data[symbol] = data
                
                if tech_data:
                    tech_df = pd.DataFrame(tech_data)
                    tech_df.to_csv(f"data/technical_indicators_{self.ticker}.csv")
                    self.update_progress("Technical indicators processing")
                    return True, "Technical indicators downloaded"
                self.update_progress("Technical indicators processing", False)
                return False, "No technical indicators available"
            except Exception as e:
                self.update_progress("Technical indicators processing", False)
                return False, f"Technical indicators error: {e}"
        
        # Run all data processing tasks in parallel with enhanced threading
        tasks = [
            ("main_data", download_and_preprocess),
            ("sector_data", download_sector_data),
            ("market_data", download_market_data),
            ("technical_indicators", download_technical_indicators)
        ]
        
        self.total_tasks = len(tasks)
        self.progress = 0
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(task[1]): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    success, message = future.result()
                    results[task_name] = (success, message)
                except Exception as e:
                    results[task_name] = (False, f"Exception: {e}")
        
        # Check if main data processing succeeded
        return results.get("main_data", (False, "Task not completed"))[0]
    
    def run_parallel_model_training(self):
        """Run model training in parallel."""
        print("🔄 Starting parallel model training...")
        
        def train_lstm_model():
            """Train LSTM model."""
            try:
                # Use unified pipeline for model training
                from unified_analysis_pipeline import UnifiedAnalysisPipeline
                
                pipeline = UnifiedAnalysisPipeline(self.ticker, max_workers=self.max_workers)
                success = pipeline.run_partB_model_training()
                
                if success:
                    return True, "LSTM model trained and saved using unified pipeline"
                else:
                    return False, "Model training failed"
            except Exception as e:
                return False, f"LSTM training error: {e}"
        
        def train_ensemble_model():
            """Train ensemble model."""
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.preprocessing import StandardScaler
                import joblib
                
                # Load preprocessed data
                df = pd.read_csv(f"data/preprocessed_{self.ticker}.csv", index_col=0, parse_dates=True)
                
                # Prepare features
                feature_cols = [col for col in df.columns if col not in ['Date', 'Datetime']]
                X = df[feature_cols].dropna()
                y = X['Close'].shift(-1).dropna()
                X = X[:-1]  # Remove last row since we don't have target
                
                # Train Random Forest
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                
                # Save model
                joblib.dump(rf_model, f"models/{self.ticker}_rf_model.pkl")
                
                return True, "Ensemble model trained and saved"
            except Exception as e:
                return False, f"Ensemble training error: {e}"
        
        def validate_models():
            """Validate trained models."""
            try:
                # Check if models exist
                lstm_exists = os.path.exists(f"models/{self.ticker}_lstm.h5")
                scaler_exists = os.path.exists(f"models/{self.ticker}_scaler.pkl")
                rf_exists = os.path.exists(f"models/{self.ticker}_rf_model.pkl")
                
                if lstm_exists and scaler_exists:
                    return True, "Models validated successfully"
                else:
                    return False, "Required models not found"
            except Exception as e:
                return False, f"Model validation error: {e}"
        
        # Run model training tasks in parallel
        tasks = [
            ("lstm_model", train_lstm_model),
            ("ensemble_model", train_ensemble_model),
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
        return results.get("lstm_model", (False, "Task not completed"))[0]
    
    def run_parallel_enhanced_analysis(self):
        """Run enhanced analysis tasks in parallel."""
        print("🔄 Starting parallel enhanced analysis...")
        
        def run_sentiment_analysis():
            """Run sentiment analysis using optimized module."""
            try:
                from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
                
                analyzer = OptimizedSentimentAnalyzer()
                sentiment_df = analyzer.analyze_stock_sentiment(self.ticker, days_back=30)
                
                if not sentiment_df.empty:
                    sentiment_df.to_csv(f"data/{self.ticker}_sentiment_analysis.csv", index=False)
                    return True, "Sentiment analysis completed"
                else:
                    return False, "No sentiment data available"
            except Exception as e:
                return False, f"Sentiment analysis error: {e}"
        
        def run_sector_trends():
            """Run sector trends analysis."""
            try:
                from partC_strategy.sector_trend import SectorTrendAnalyzer
                
                analyzer = SectorTrendAnalyzer()
                sector_data = analyzer.analyze_sector_trends(self.ticker)
                
                if sector_data:
                    sector_df = pd.DataFrame([sector_data])
                    sector_df.to_csv(f"data/{self.ticker}_sector_trends.csv", index=False)
                    return True, "Sector trends completed"
                else:
                    return False, "No sector data available"
            except Exception as e:
                return False, f"Sector trends error: {e}"
        
        def run_economic_indicators():
            """Run economic indicators analysis."""
            try:
                from partC_strategy.economic_indicators import EconomicIndicators
                
                analyzer = EconomicIndicators()
                economic_data = analyzer.get_all_indicators()
                
                if economic_data:
                    economic_df = pd.DataFrame([economic_data])
                    economic_df.to_csv(f"data/{self.ticker}_economic_indicators.csv", index=False)
                    return True, "Economic indicators completed"
                else:
                    return False, "No economic data available"
            except Exception as e:
                return False, f"Economic indicators error: {e}"
        
        def run_market_factors():
            """Run market factors analysis."""
            try:
                from partC_strategy.enhanced_market_factors import EnhancedMarketFactors
                
                analyzer = EnhancedMarketFactors()
                market_data = analyzer.get_market_factors()
                
                if market_data:
                    market_df = pd.DataFrame([market_data])
                    market_df.to_csv(f"data/{self.ticker}_market_factors.csv", index=False)
                    return True, "Market factors completed"
                else:
                    return False, "No market data available"
            except Exception as e:
                return False, f"Market factors error: {e}"
        
        # Run enhanced analysis tasks in parallel
        tasks = [
            ("sentiment", run_sentiment_analysis),
            ("sector_trends", run_sector_trends),
            ("economic_indicators", run_economic_indicators),
            ("market_factors", run_market_factors)
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
                    print(f"{status} [{self.progress:.1f}%] {task_name}: {message}")
                except Exception as e:
                    results[task_name] = (False, f"Exception: {e}")
                    print(f"❌ [{self.progress:.1f}%] {task_name}: Exception - {e}")
                
                with self.thread_lock:
                    self.progress += 25.0
        
        return any(results.values())
    
    def run_parallel_strategy_analysis(self):
        """Run strategy analysis tasks in parallel."""
        print("🔄 Starting parallel strategy analysis...")
        
        def run_enhanced_strategy():
            """Run enhanced strategy analysis using optimized module."""
            try:
                from partC_strategy.optimized_trading_strategy import OptimizedTradingStrategy
                from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
                
                # Load data
                data_file = f"data/{self.ticker}_preprocessed.csv"
                if not os.path.exists(data_file):
                    return False, "No preprocessed data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Initialize strategy with sentiment analyzer
                sentiment_analyzer = OptimizedSentimentAnalyzer()
                strategy = OptimizedTradingStrategy(sentiment_analyzer)
                
                # Generate enhanced signals
                signals_df = strategy.generate_enhanced_signals(self.ticker, df)
                
                if not signals_df.empty:
                    signals_df.to_csv(f"data/{self.ticker}_enhanced_signals.csv")
                    return True, "Enhanced strategy completed"
                else:
                    return False, "No signals generated"
            except Exception as e:
                return False, f"Enhanced strategy error: {e}"
        
        def run_backtesting():
            """Run backtesting analysis."""
            try:
                from partC_strategy.backtest import BacktestStrategy
                
                # Load signals data
                signals_file = f"data/{self.ticker}_enhanced_signals.csv"
                if not os.path.exists(signals_file):
                    return False, "No signals data available"
                
                signals_df = pd.read_csv(signals_file, index_col=0, parse_dates=True)
                
                # Run backtest
                backtest = BacktestStrategy()
                results = backtest.run_backtest(signals_df)
                
                if results:
                    results_df = pd.DataFrame([results])
                    results_df.to_csv(f"data/{self.ticker}_backtest_results.csv", index=False)
                    return True, "Backtesting completed"
                else:
                    return False, "No backtest results"
            except Exception as e:
                return False, f"Backtesting error: {e}"
        
        def run_basic_strategy():
            """Run basic strategy analysis."""
            try:
                from partC_strategy.optimized_technical_indicators import OptimizedTechnicalIndicators
                
                # Load data
                data_file = f"data/{self.ticker}_preprocessed.csv"
                if not os.path.exists(data_file):
                    return False, "No preprocessed data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Add technical indicators
                analyzer = OptimizedTechnicalIndicators()
                df_with_indicators = analyzer.add_all_indicators(df)
                
                # Generate signals
                signals_df = analyzer.generate_signals(df_with_indicators)
                
                if not signals_df.empty:
                    signals_df.to_csv(f"data/{self.ticker}_signals.csv")
                    return True, "Basic strategy completed"
                else:
                    return False, "No signals generated"
            except Exception as e:
                return False, f"Basic strategy error: {e}"
        
        # Run strategy analysis tasks in parallel
        tasks = [
            ("enhanced_strategy", run_enhanced_strategy),
            ("backtesting", run_backtesting),
            ("basic_strategy", run_basic_strategy)
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
    
    def run_parallel_report_generation(self, days_ahead, use_enhanced):
        """Run report generation in parallel."""
        print("🔄 Starting parallel report generation...")
        
        def generate_summary_report():
            """Generate summary report."""
            try:
                data_files = [f for f in os.listdir('data') if f.startswith(self.ticker)]
                
                summary = {
                    'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Ticker': self.ticker,
                    'Files_Generated': len(data_files),
                    'Enhanced_Features': use_enhanced,
                    'Status': 'Completed'
                }
                
                summary_df = pd.DataFrame([summary])
                summary_df.to_csv(f'data/{self.ticker}_analysis_summary.csv', index=False)
                return True, f"Summary report generated ({len(data_files)} files)"
            except Exception as e:
                return False, f"Summary report error: {e}"
        
        def generate_performance_report():
            """Generate performance report."""
            try:
                # Load and analyze performance metrics
                metrics_files = [
                    f"data/{self.ticker}_backtest_metrics.csv",
                    f"data/{self.ticker}_enhanced_predictions.csv"
                ]
                
                performance_data = {}
                for file in metrics_files:
                    if os.path.exists(file):
                        df = pd.read_csv(file)
                        performance_data[file] = df.to_dict('records')
                
                # Save performance report
                if performance_data:
                    performance_df = pd.DataFrame(performance_data)
                    performance_df.to_csv(f"data/{self.ticker}_performance_report.csv", index=False)
                    return True, "Performance report generated"
                else:
                    return False, "No performance data available"
            except Exception as e:
                return False, f"Performance report error: {e}"
        
        def generate_decision_report():
            """Generate decision analysis report."""
            try:
                # Use unified pipeline for decision analysis
                from unified_analysis_pipeline import UnifiedAnalysisPipeline
                
                pipeline = UnifiedAnalysisPipeline(self.ticker, max_workers=self.max_workers)
                success = pipeline.run_unified_report_generation(5, True)  # Default values
                
                if success:
                    return True, "Decision report generated using unified pipeline"
                else:
                    return False, "Decision report generation failed"
            except Exception as e:
                return False, f"Decision report error: {e}"
        
        # Run report generation tasks in parallel
        tasks = [
            ("summary_report", generate_summary_report),
            ("performance_report", generate_performance_report),
            ("decision_report", generate_decision_report)
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
        
        return True
    
    def display_final_results(self, execution_time, use_enhanced):
        """Display final results."""
        print(f"\n{'='*80}")
        print("✅ Multi-threaded Analysis Pipeline Completed!")
        print(f"{'='*80}")
        print(f"⏱️ Total Execution Time: {execution_time:.2f} seconds")
        print(f"🔧 Threads Used: {self.max_workers}")
        print(f"📊 Ticker: {self.ticker}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print()
        
        print("📁 Generated Files:")
        data_files = [f for f in os.listdir('data') if f.startswith(self.ticker)]
        for file in sorted(data_files):
            print(f"    {file}")
        
        print(f"\n🎯 Key Results:")
        print(f"   📊 Enhanced predictions: data/{self.ticker}_enhanced_predictions.csv")
        print(f"   📈 Trading signals: data/{self.ticker}_signals.csv")
        if use_enhanced:
            print(f"   🏢 Market factors: data/{self.ticker}_market_factors.csv")
            print(f"   📰 Economic indicators: data/{self.ticker}_economic_indicators.csv")
            print(f"    Sentiment analysis: data/{self.ticker}_sentiment_analysis.csv")
        print(f"   📋 Analysis summary: data/{self.ticker}_analysis_summary.csv")
        
        print(f"\n💡 Performance Benefits:")
        print(f"   ⚡ Parallel processing reduced execution time")
        print(f"   🔧 Multi-threaded data loading and processing")
        print(f"   🤖 Concurrent model training")
        print(f"   📊 Parallel strategy analysis")
        print(f"   📋 Simultaneous report generation")
        
        print(f"\n⚠️ Important Notes:")
        print(f"   • Always verify predictions with additional research")
        print(f"   • Consider market conditions before trading")
        print(f"   • Use proper risk management")
        print(f"   • Past performance doesn't guarantee future results")

    def run_intraday_analysis(self, hours_ahead=6, use_enhanced=True):
        """Run intraday analysis with parallel processing."""
        self.start_time = time.time()
        
        print("🚀 Multi-threaded AI Stock Predictor - Intraday Analysis")
        print("=" * 80)
        print(f"📊 Ticker: {self.ticker}")
        print(f"⏰ Hours Ahead: {hours_ahead}")
        print(f" Threads: {self.max_workers}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print()
        
        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        try:
            # Step 1: Quick Data Processing (intraday needs recent data)
            print("📥 Step 1: Quick Intraday Data Processing")
            print("-" * 50)
            data_processing_success = self.run_quick_intraday_data_processing()
            
            if not data_processing_success:
                print("❌ Data processing failed")
                return False
            
            # Step 2: Intraday Model Training/Loading
            print("\n🤖 Step 2: Intraday Model Preparation")
            print("-" * 50)
            model_success = self.prepare_intraday_model()
            
            if not model_success:
                print("❌ Model preparation failed")
                return False
            
            # Step 3: Intraday Analysis
            print("\n🔍 Step 3: Intraday Analysis")
            print("-" * 50)
            analysis_success = self.run_intraday_enhanced_analysis()
            
            # Step 4: Intraday Strategy Analysis
            print("\n📈 Step 4: Intraday Strategy Analysis")
            print("-" * 50)
            strategy_success = self.run_intraday_strategy_analysis()
            
            # Step 5: Intraday Report Generation
            print("\n📋 Step 5: Intraday Report Generation")
            print("-" * 50)
            report_success = self.run_intraday_report_generation(hours_ahead, use_enhanced)
            
            # Calculate execution time
            execution_time = time.time() - self.start_time
            
            # Display final results
            self.display_intraday_results(execution_time, use_enhanced, hours_ahead)
            
            return True
            
        except Exception as e:
            print(f"❌ Intraday analysis failed: {e}")
            return False
    
    def run_quick_intraday_data_processing(self):
        """Run quick data processing for intraday analysis."""
        print("🔄 Starting quick intraday data processing...")
        
        def download_intraday_data():
            """Download recent intraday data."""
            try:
                import yfinance as yf
                
                # Download recent data (last 5 days with 1-hour intervals)
                stock = yf.Ticker(self.ticker)
                df = stock.history(period="5d", interval="1h")
                
                if not df.empty:
                    df.to_csv(f"data/{self.ticker}_intraday_data.csv")
                    return True, f"Downloaded {len(df)} intraday records"
                else:
                    return False, "No intraday data available"
            except Exception as e:
                return False, f"Data download error: {e}"
        
        def add_intraday_indicators():
            """Add intraday technical indicators."""
            try:
                from partC_strategy.optimized_technical_indicators import OptimizedTechnicalIndicators
                
                data_file = f"data/{self.ticker}_intraday_data.csv"
                if not os.path.exists(data_file):
                    return False, "No intraday data file found"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Add technical indicators optimized for intraday
                analyzer = OptimizedTechnicalIndicators()
                df_with_indicators = analyzer.add_all_indicators(df)
                
                df_with_indicators.to_csv(f"data/{self.ticker}_intraday_enhanced.csv")
                return True, f"Added indicators to {len(df_with_indicators)} records"
            except Exception as e:
                return False, f"Indicator error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("download_data", download_intraday_data),
            ("add_indicators", add_intraday_indicators)
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
        
        return all(success for success, _ in results.values())
    
    def prepare_intraday_model(self):
        """Prepare intraday prediction model."""
        print("🔄 Preparing intraday model...")
        
        def load_or_train_model():
            """Load existing model or train new one."""
            try:
                model_path = f"models/{self.ticker}_intraday_lstm.h5"
                
                if os.path.exists(model_path):
                    from tensorflow.keras.models import load_model
                    model = load_model(model_path)
                    return True, "Loaded existing intraday model"
                else:
                    # Use unified pipeline for model training
                    from unified_analysis_pipeline import UnifiedAnalysisPipeline
                    
                    pipeline = UnifiedAnalysisPipeline(self.ticker, max_workers=self.max_workers)
                    success = pipeline.run_partB_model_training()
                    
                    if success:
                        return True, "Trained new model using unified pipeline"
                    else:
                        return False, "Model training failed"
            except Exception as e:
                return False, f"Model error: {e}"
        
        def prepare_scaler():
            """Prepare data scaler for intraday."""
            try:
                from sklearn.preprocessing import MinMaxScaler
                import joblib
                
                scaler_path = f"models/{self.ticker}_intraday_scaler.pkl"
                
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    return True, "Loaded existing scaler"
                else:
                    # Create new scaler
                    data_file = f"data/{self.ticker}_intraday_enhanced.csv"
                    if os.path.exists(data_file):
                        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                        
                        # Prepare features for scaling
                        feature_cols = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        scaler = MinMaxScaler()
                        scaler.fit(df[feature_cols])
                        
                        # Save scaler
                        joblib.dump(scaler, scaler_path)
                        return True, "Created new scaler"
                    else:
                        return False, "No enhanced data for scaler"
            except Exception as e:
                return False, f"Scaler error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("model", load_or_train_model),
            ("scaler", prepare_scaler)
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
        
        return all(success for success, _ in results.values())
    
    def run_intraday_enhanced_analysis(self):
        """Run enhanced analysis for intraday."""
        print("🔄 Starting intraday enhanced analysis...")
        
        def run_intraday_sentiment():
            """Run sentiment analysis for intraday."""
            try:
                from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
                
                analyzer = OptimizedSentimentAnalyzer()
                sentiment_df = analyzer.analyze_stock_sentiment(self.ticker, days_back=1)  # Recent sentiment
                
                if not sentiment_df.empty:
                    sentiment_df.to_csv(f"data/{self.ticker}_intraday_sentiment.csv", index=False)
                    return True, "Intraday sentiment completed"
                else:
                    return False, "No sentiment data available"
            except Exception as e:
                return False, f"Sentiment error: {e}"
        
        def run_intraday_market_factors():
            """Run market factors for intraday."""
            try:
                from partC_strategy.enhanced_market_factors import EnhancedMarketFactors
                
                analyzer = EnhancedMarketFactors()
                market_data = analyzer.get_market_factors()
                
                if market_data:
                    market_df = pd.DataFrame([market_data])
                    market_df.to_csv(f"data/{self.ticker}_intraday_market_factors.csv", index=False)
                    return True, "Intraday market factors completed"
                else:
                    return False, "No market data available"
            except Exception as e:
                return False, f"Market factors error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("sentiment", run_intraday_sentiment),
            ("market_factors", run_intraday_market_factors)
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
    
    def run_intraday_strategy_analysis(self):
        """Run strategy analysis for intraday."""
        print("🔄 Starting intraday strategy analysis...")
        
        def run_intraday_signals():
            """Generate intraday trading signals."""
            try:
                from partC_strategy.optimized_trading_strategy import OptimizedTradingStrategy
                from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
                
                # Load intraday data
                data_file = f"data/{self.ticker}_intraday_enhanced.csv"
                if not os.path.exists(data_file):
                    return False, "No intraday data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Generate signals
                sentiment_analyzer = OptimizedSentimentAnalyzer()
                strategy = OptimizedTradingStrategy(sentiment_analyzer)
                signals_df = strategy.generate_enhanced_signals(self.ticker, df)
                
                if not signals_df.empty:
                    signals_df.to_csv(f"data/{self.ticker}_intraday_signals.csv")
                    return True, "Intraday signals generated"
                else:
                    return False, "No signals generated"
            except Exception as e:
                return False, f"Signals error: {e}"
        
        def run_intraday_predictions():
            """Generate intraday price predictions."""
            try:
                from tensorflow.keras.models import load_model
                import joblib
                
                # Load model and scaler
                model_path = f"models/{self.ticker}_intraday_lstm.h5"
                scaler_path = f"models/{self.ticker}_intraday_scaler.pkl"
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    return False, "Model or scaler not found"
                
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                
                # Load data
                data_file = f"data/{self.ticker}_intraday_enhanced.csv"
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Prepare features
                feature_cols = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                features = df[feature_cols].values
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make predictions (next 6 hours)
                predictions = []
                for i in range(6):
                    if len(features_scaled) > 0:
                        # Use last sequence for prediction
                        last_sequence = features_scaled[-1:].reshape(1, 1, -1)
                        pred = model.predict(last_sequence, verbose=0)
                        predictions.append(pred[0][0])
                
                # Create predictions DataFrame
                pred_df = pd.DataFrame({
                    'hour': range(1, len(predictions) + 1),
                    'predicted_price': predictions,
                    'timestamp': pd.Timestamp.now()
                })
                
                pred_df.to_csv(f"data/{self.ticker}_intraday_predictions.csv", index=False)
                return True, f"Generated {len(predictions)} intraday predictions"
            except Exception as e:
                return False, f"Predictions error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("signals", run_intraday_signals),
            ("predictions", run_intraday_predictions)
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
    
    def run_intraday_report_generation(self, hours_ahead, use_enhanced):
        """Generate intraday reports."""
        print("🔄 Starting intraday report generation...")
        
        def generate_intraday_summary():
            """Generate intraday summary report."""
            try:
                # Load all intraday data
                data_files = {}
                
                for file_type in ['predictions', 'signals', 'sentiment', 'market_factors']:
                    file_path = f"data/{self.ticker}_intraday_{file_type}.csv"
                    if os.path.exists(file_path):
                        data_files[file_type] = pd.read_csv(file_path)
                
                # Create summary
                summary = {
                    'ticker': self.ticker,
                    'analysis_type': 'intraday',
                    'hours_ahead': hours_ahead,
                    'timestamp': datetime.now().isoformat(),
                    'files_generated': list(data_files.keys()),
                    'enhanced_features': use_enhanced
                }
                
                # Save summary
                summary_df = pd.DataFrame([summary])
                summary_df.to_csv(f"data/{self.ticker}_intraday_summary.csv", index=False)
                
                return True, "Intraday summary generated"
            except Exception as e:
                return False, f"Summary error: {e}"
        
        def generate_intraday_decision_report():
            """Generate intraday decision report."""
            try:
                # Use unified pipeline for intraday decision analysis
                from unified_analysis_pipeline import UnifiedAnalysisPipeline
                
                pipeline = UnifiedAnalysisPipeline(self.ticker, max_workers=self.max_workers)
                success = pipeline.run_unified_report_generation(hours_ahead, True)
                
                if success:
                    return True, "Intraday decision report generated using unified pipeline"
                else:
                    return False, "Intraday decision report generation failed"
            except Exception as e:
                return False, f"Decision report error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("summary", generate_intraday_summary),
            ("decision_report", generate_intraday_decision_report)
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
    
    def display_intraday_results(self, execution_time, use_enhanced, hours_ahead):
        """Display intraday analysis results."""
        print(f"\n{'='*80}")
        print("✅ Intraday Analysis Pipeline Completed!")
        print(f"{'='*80}")
        print(f"⏱️ Total Execution Time: {execution_time:.2f} seconds")
        print(f"🔧 Threads Used: {self.max_workers}")
        print(f"📊 Ticker: {self.ticker}")
        print(f"⏰ Hours Ahead: {hours_ahead}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print()
        
        print("📁 Generated Intraday Files:")
        data_files = [f for f in os.listdir('data') if f.startswith(self.ticker) and 'intraday' in f]
        for file in sorted(data_files):
            print(f"    {file}")
        
        print(f"\n🎯 Key Intraday Results:")
        print(f"   📊 Intraday predictions: data/{self.ticker}_intraday_predictions.csv")
        print(f"   📈 Intraday signals: data/{self.ticker}_intraday_signals.csv")
        if use_enhanced:
            print(f"   🏢 Intraday market factors: data/{self.ticker}_intraday_market_factors.csv")
            print(f"    Intraday sentiment: data/{self.ticker}_intraday_sentiment.csv")
        print(f"   📋 Intraday summary: data/{self.ticker}_intraday_summary.csv")
        
        print(f"\n💡 Intraday Trading Benefits:")
        print(f"   ⚡ Faster execution with parallel processing")
        print(f"   🔧 Real-time data processing")
        print(f"   🤖 Quick model predictions")
        print(f"   📊 Hourly signal generation")
        print(f"   📋 Immediate decision reports")
        
        print(f"\n⚠️ Intraday Trading Notes:")
        print(f"   • Use tighter stop-losses (0.5-1%)")
        print(f"   • Monitor market conditions closely")
        print(f"   • Consider transaction costs")
        print(f"   • Past performance doesn't guarantee future results")

    def run_short_term_analysis(self, days_ahead=5, use_enhanced=True):
        """Run short-term analysis (1-7 days) with parallel processing."""
        try:
            from analysis_modules.short_term_analyzer import ShortTermAnalyzer
            
            # Create short-term analyzer
            analyzer = ShortTermAnalyzer(self.ticker, max_workers=self.max_workers)
            
            # Run short-term analysis
            success = analyzer.run_short_term_data_processing()
            if not success:
                return False
            
            success = analyzer.prepare_short_term_model()
            if not success:
                return False
            
            success = analyzer.run_short_term_enhanced_analysis()
            success = analyzer.run_short_term_strategy_analysis(days_ahead)
            success = analyzer.run_short_term_report_generation(days_ahead, use_enhanced)
            
            # Display results
            analyzer.display_short_term_results(time.time() - self.start_time, use_enhanced, days_ahead)
            
            return True
            
        except Exception as e:
            print(f"❌ Short-term analysis failed: {e}")
            return False
    
    def run_mid_term_analysis(self, weeks_ahead=2, use_enhanced=True):
        """Run mid-term analysis (1-4 weeks) with parallel processing."""
        try:
            from analysis_modules.mid_term_analyzer import MidTermAnalyzer
            
            # Create mid-term analyzer
            analyzer = MidTermAnalyzer(self.ticker, max_workers=self.max_workers)
            
            # Run mid-term analysis
            success = analyzer.run_mid_term_data_processing()
            if not success:
                return False
            
            success = analyzer.prepare_mid_term_model()
            if not success:
                return False
            
            success = analyzer.run_mid_term_enhanced_analysis()
            success = analyzer.run_mid_term_strategy_analysis(weeks_ahead)
            success = analyzer.run_mid_term_report_generation(weeks_ahead, use_enhanced)
            
            # Display results
            analyzer.display_mid_term_results(time.time() - self.start_time, use_enhanced, weeks_ahead)
            
            return True
            
        except Exception as e:
            print(f"❌ Mid-term analysis failed: {e}")
            return False
    
    def run_long_term_analysis(self, months_ahead=3, use_enhanced=True):
        """Run long-term analysis (1-12 months) with parallel processing."""
        try:
            from analysis_modules.long_term_analyzer import LongTermAnalyzer
            
            # Create long-term analyzer
            analyzer = LongTermAnalyzer(self.ticker, max_workers=self.max_workers)
            
            # Run long-term analysis
            success = analyzer.run_long_term_data_processing()
            if not success:
                return False
            
            success = analyzer.prepare_long_term_model()
            if not success:
                return False
            
            success = analyzer.run_long_term_enhanced_analysis()
            success = analyzer.run_long_term_strategy_analysis(months_ahead)
            success = analyzer.run_long_term_report_generation(months_ahead, use_enhanced)
            
            # Display results
            analyzer.display_long_term_results(time.time() - self.start_time, use_enhanced, months_ahead)
            
            return True
            
        except Exception as e:
            print(f"❌ Long-term analysis failed: {e}")
            return False

def main():
    """Main function with forecast type selection."""
    print("🚀 Multi-threaded AI Stock Predictor")
    print("=" * 50)
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").upper().strip()
    if not ticker:
        print("❌ No ticker provided")
        return
    
    # Display forecast type options
    print("\n📊 Select Forecast Type:")
    print("1. Intraday Forecast (1-24 hours)")
    print("2. Short-term Forecast (1-7 days)")
    print("3. Mid-term Forecast (1-4 weeks)")
    print("4. Long-term Forecast (1-12 months)")
    
    choice = input("\nEnter your choice (1-4) [default: 2]: ").strip() or "2"
    
    if choice == "1":
        # Intraday analysis
        print("\n⏰ Intraday Forecast Selected")
        hours_ahead = int(input("Enter hours ahead (1-24) [default: 6]: ").strip() or "6")
        use_enhanced = input("Use enhanced features? (y/n) [default: y]: ").strip().lower() or "y"
        threads = int(input("Enter number of threads (1-8) [default: 4]: ").strip() or "4")
        
        use_enhanced = use_enhanced == 'y'
        
        # Create and run intraday analysis
        runner = EnhancedMultiThreadedAnalysisRunner(ticker, max_workers=threads)
        success = runner.run_intraday_analysis(hours_ahead, use_enhanced)
        
    elif choice == "2":
        # Short-term analysis
        print("\n📅 Short-term Forecast Selected")
        days_ahead = int(input("Enter prediction days ahead (1-7) [default: 5]: ").strip() or "5")
        use_enhanced = input("Use enhanced features? (y/n) [default: y]: ").strip().lower() or "y"
        threads = int(input("Enter number of threads (1-8) [default: 4]: ").strip() or "4")
        
        use_enhanced = use_enhanced == 'y'
        
        # Create and run short-term analysis
        runner = EnhancedMultiThreadedAnalysisRunner(ticker, max_workers=threads)
        success = runner.run_short_term_analysis(days_ahead, use_enhanced)
        
    elif choice == "3":
        # Mid-term analysis
        print("\n📅 Mid-term Forecast Selected")
        weeks_ahead = int(input("Enter prediction weeks ahead (1-4) [default: 2]: ").strip() or "2")
        use_enhanced = input("Use enhanced features? (y/n) [default: y]: ").strip().lower() or "y"
        threads = int(input("Enter number of threads (1-8) [default: 4]: ").strip() or "4")
        
        use_enhanced = use_enhanced == 'y'
        
        # Create and run mid-term analysis
        runner = EnhancedMultiThreadedAnalysisRunner(ticker, max_workers=threads)
        success = runner.run_mid_term_analysis(weeks_ahead, use_enhanced)
        
    elif choice == "4":
        # Long-term analysis
        print("\n📅 Long-term Forecast Selected")
        months_ahead = int(input("Enter prediction months ahead (1-12) [default: 3]: ").strip() or "3")
        use_enhanced = input("Use enhanced features? (y/n) [default: y]: ").strip().lower() or "y"
        threads = int(input("Enter number of threads (1-8) [default: 4]: ").strip() or "4")
        
        use_enhanced = use_enhanced == 'y'
        
        # Create and run long-term analysis
        runner = EnhancedMultiThreadedAnalysisRunner(ticker, max_workers=threads)
        success = runner.run_long_term_analysis(months_ahead, use_enhanced)
        
    else:
        print("❌ Invalid choice. Please select 1-4.")
        return
    
    if success:
        print(f"\n✅ Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed. Check logs for details.")

if __name__ == "__main__":
    main()
