#!/usr/bin/env python3
"""
Mid-term Analysis Module (1-4 weeks)
Specialized for medium-term trading and trend following
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import time

class MidTermAnalyzer:
    """Mid-term analysis for 1-4 week predictions."""
    
    def __init__(self, ticker, max_workers=4):
        self.ticker = ticker
        self.max_workers = max_workers
        self.data_dir = "data"
        self.models_dir = "models"
    
    def run_mid_term_data_processing(self):
        """Run data processing optimized for mid-term analysis."""
        print("🔄 Starting mid-term data processing...")
        
        def download_mid_term_data():
            """Download historical data for mid-term analysis."""
            try:
                import yfinance as yf
                
                # Download historical data (last 1 year with daily intervals)
                stock = yf.Ticker(self.ticker)
                df = stock.history(period="1y", interval="1d")
                
                if not df.empty:
                    df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_data.csv")
                    return True, f"Downloaded {len(df)} mid-term records"
                else:
                    return False, "No mid-term data available"
            except Exception as e:
                return False, f"Data download error: {e}"
        
        def add_mid_term_indicators():
            """Add technical indicators optimized for mid-term."""
            try:
                from partC_strategy.optimized_technical_indicators import OptimizedTechnicalIndicators
                
                data_file = f"{self.data_dir}/{self.ticker}_mid_term_data.csv"
                if not os.path.exists(data_file):
                    return False, "No mid-term data file found"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Add technical indicators optimized for mid-term
                analyzer = OptimizedTechnicalIndicators()
                df_with_indicators = analyzer.add_all_indicators(df)
                
                df_with_indicators.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_enhanced.csv")
                return True, f"Added indicators to {len(df_with_indicators)} records"
            except Exception as e:
                return False, f"Indicator error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("download_data", download_mid_term_data),
            ("add_indicators", add_mid_term_indicators)
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
    
    def prepare_mid_term_model(self):
        """Prepare mid-term prediction model."""
        print("🔄 Preparing mid-term model...")
        
        def load_or_train_model():
            """Load existing model or train new one."""
            try:
                model_path = f"{self.models_dir}/{self.ticker}_mid_term_lstm.h5"
                
                if os.path.exists(model_path):
                    from tensorflow.keras.models import load_model
                    model = load_model(model_path)
                    return True, "Loaded existing mid-term model"
                else:
                    # Train new mid-term model
                    from partB_model.enhanced_training import EnhancedStockPredictor
                    
                    predictor = EnhancedStockPredictor(self.ticker)
                    model = predictor.train_mid_term_model()
                    
                    # Save model
                    model.save(model_path)
                    return True, "Trained new mid-term model"
            except Exception as e:
                return False, f"Model error: {e}"
        
        def prepare_scaler():
            """Prepare data scaler for mid-term."""
            try:
                from sklearn.preprocessing import MinMaxScaler
                import joblib
                
                scaler_path = f"{self.models_dir}/{self.ticker}_mid_term_scaler.pkl"
                
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    return True, "Loaded existing scaler"
                else:
                    # Create new scaler
                    data_file = f"{self.data_dir}/{self.ticker}_mid_term_enhanced.csv"
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
    
    def run_mid_term_enhanced_analysis(self):
        """Run enhanced analysis for mid-term."""
        print("🔄 Starting mid-term enhanced analysis...")
        
        def run_mid_term_sentiment():
            """Run sentiment analysis for mid-term."""
            try:
                from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
                
                analyzer = OptimizedSentimentAnalyzer()
                sentiment_df = analyzer.analyze_stock_sentiment(self.ticker, days_back=30)  # Monthly sentiment
                
                if not sentiment_df.empty:
                    sentiment_df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_sentiment.csv", index=False)
                    return True, "Mid-term sentiment completed"
                else:
                    return False, "No sentiment data available"
            except Exception as e:
                return False, f"Sentiment error: {e}"
        
        def run_mid_term_market_factors():
            """Run market factors for mid-term."""
            try:
                from partC_strategy.enhanced_market_factors import EnhancedMarketFactors
                
                analyzer = EnhancedMarketFactors()
                market_data = analyzer.get_market_factors()
                
                if market_data:
                    market_df = pd.DataFrame([market_data])
                    market_df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_market_factors.csv", index=False)
                    return True, "Mid-term market factors completed"
                else:
                    return False, "No market data available"
            except Exception as e:
                return False, f"Market factors error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("sentiment", run_mid_term_sentiment),
            ("market_factors", run_mid_term_market_factors)
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
    
    def run_mid_term_strategy_analysis(self, weeks_ahead):
        """Run strategy analysis for mid-term."""
        print("🔄 Starting mid-term strategy analysis...")
        
        def run_mid_term_signals():
            """Generate mid-term trading signals."""
            try:
                from partC_strategy.optimized_trading_strategy import OptimizedTradingStrategy
                from partC_strategy.optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
                
                # Load mid-term data
                data_file = f"{self.data_dir}/{self.ticker}_mid_term_enhanced.csv"
                if not os.path.exists(data_file):
                    return False, "No mid-term data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Generate signals
                sentiment_analyzer = OptimizedSentimentAnalyzer()
                strategy = OptimizedTradingStrategy(sentiment_analyzer)
                signals_df = strategy.generate_enhanced_signals(self.ticker, df)
                
                if not signals_df.empty:
                    signals_df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_signals.csv")
                    return True, "Mid-term signals generated"
                else:
                    return False, "No signals generated"
            except Exception as e:
                return False, f"Signals error: {e}"
        
        def run_mid_term_predictions():
            """Generate mid-term price predictions."""
            try:
                from tensorflow.keras.models import load_model
                import joblib
                
                # Load model and scaler
                model_path = f"{self.models_dir}/{self.ticker}_mid_term_lstm.h5"
                scaler_path = f"{self.models_dir}/{self.ticker}_mid_term_scaler.pkl"
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    return False, "Model or scaler not found"
                
                model = load_model(model_path)
                scaler = joblib.load(scaler_path)
                
                # Load data
                data_file = f"{self.data_dir}/{self.ticker}_mid_term_enhanced.csv"
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Prepare features
                feature_cols = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                features = df[feature_cols].values
                
                # Scale features
                features_scaled = scaler.transform(features)
                
                # Make predictions (next N weeks)
                predictions = []
                for i in range(weeks_ahead * 5):  # 5 trading days per week
                    if len(features_scaled) > 0:
                        # Use last sequence for prediction
                        last_sequence = features_scaled[-1:].reshape(1, 1, -1)
                        pred = model.predict(last_sequence, verbose=0)
                        predictions.append(pred[0][0])
                
                # Create predictions DataFrame
                pred_df = pd.DataFrame({
                    'day': range(1, len(predictions) + 1),
                    'predicted_price': predictions,
                    'timestamp': pd.Timestamp.now()
                })
                
                pred_df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_predictions.csv", index=False)
                return True, f"Generated {len(predictions)} mid-term predictions"
            except Exception as e:
                return False, f"Predictions error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("signals", run_mid_term_signals),
            ("predictions", run_mid_term_predictions)
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
    
    def run_mid_term_report_generation(self, weeks_ahead, use_enhanced):
        """Generate mid-term reports."""
        print("🔄 Starting mid-term report generation...")
        
        def generate_mid_term_summary():
            """Generate mid-term summary report."""
            try:
                # Load all mid-term data
                data_files = {}
                
                for file_type in ['predictions', 'signals', 'sentiment', 'market_factors']:
                    file_path = f"{self.data_dir}/{self.ticker}_mid_term_{file_type}.csv"
                    if os.path.exists(file_path):
                        data_files[file_type] = pd.read_csv(file_path)
                
                # Create summary
                summary = {
                    'ticker': self.ticker,
                    'analysis_type': 'mid_term',
                    'weeks_ahead': weeks_ahead,
                    'timestamp': datetime.now().isoformat(),
                    'files_generated': list(data_files.keys()),
                    'enhanced_features': use_enhanced
                }
                
                # Save summary
                summary_df = pd.DataFrame([summary])
                summary_df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_summary.csv", index=False)
                
                return True, "Mid-term summary generated"
            except Exception as e:
                return False, f"Summary error: {e}"
        
        def generate_mid_term_decision_report():
            """Generate mid-term decision report."""
            try:
                # Decision analysis functionality integrated into unified pipeline
                # Generate basic summary report
                summary = {
                    'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Ticker': self.ticker,
                    'Analysis_Type': 'Mid-term',
                    'Weeks_Ahead': weeks_ahead,
                    'Enhanced_Features': use_enhanced,
                    'Status': 'Completed'
                }
                
                summary_df = pd.DataFrame([summary])
                summary_df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_decision_summary.csv", index=False)
                
                return True, "Mid-term decision summary generated"
            except Exception as e:
                return False, f"Decision report error: {e}"
        
        # Run tasks in parallel
        tasks = [
            ("summary", generate_mid_term_summary),
            ("decision_report", generate_mid_term_decision_report)
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
    
    def display_mid_term_results(self, execution_time, use_enhanced, weeks_ahead):
        """Display mid-term analysis results."""
        print(f"\n{'='*80}")
        print("✅ Mid-term Analysis Pipeline Completed!")
        print(f"{'='*80}")
        print(f"⏱️ Total Execution Time: {execution_time:.2f} seconds")
        print(f"🔧 Threads Used: {self.max_workers}")
        print(f"📊 Ticker: {self.ticker}")
        print(f"📅 Weeks Ahead: {weeks_ahead}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print()
        
        print("📁 Generated Mid-term Files:")
        data_files = [f for f in os.listdir('data') if f.startswith(self.ticker) and 'mid_term' in f]
        for file in sorted(data_files):
            print(f"    {file}")
        
        print(f"\n🎯 Key Mid-term Results:")
        print(f"   📊 Mid-term predictions: data/{self.ticker}_mid_term_predictions.csv")
        print(f"   📈 Mid-term signals: data/{self.ticker}_mid_term_signals.csv")
        if use_enhanced:
            print(f"   🏢 Mid-term market factors: data/{self.ticker}_mid_term_market_factors.csv")
            print(f"   📰 Mid-term sentiment: data/{self.ticker}_mid_term_sentiment.csv")
        print(f"   📋 Mid-term summary: data/{self.ticker}_mid_term_summary.csv")
        
        print(f"\n💡 Mid-term Trading Benefits:")
        print(f"   ⚡ Balanced execution with parallel processing")
        print(f"   🔧 Historical data processing")
        print(f"   🤖 Trend-based model predictions")
        print(f"   📊 Weekly signal generation")
        print(f"   📋 Trend analysis reports")
        
        print(f"\n⚠️ Mid-term Trading Notes:")
        print(f"   • Use standard stop-losses (3-5%)")
        print(f"   • Monitor weekly market trends")
        print(f"   • Consider trend following strategies")
        print(f"   • Past performance doesn't guarantee future results")
