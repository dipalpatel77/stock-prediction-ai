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
            """Download historical data for mid-term analysis using incremental updates."""
            try:
                from src.core.data_service import DataService
                
                # Use incremental data service for efficiency
                data_service = DataService()
                df = data_service.load_stock_data_incremental(
                    ticker=self.ticker,
                    period="1y",
                    interval="1d",
                    force_refresh=False
                )
                
                if not df.empty:
                    # Save processed data
                    df.to_csv(f"{self.data_dir}/{self.ticker}_mid_term_data.csv")
                    return True, f"Downloaded {len(df)} mid-term records (incremental)"
                else:
                    return False, "No mid-term data available"
            except Exception as e:
                return False, f"Data download error: {e}"
        
        def add_mid_term_indicators():
            """Add technical indicators optimized for mid-term."""
            try:
                from src.core.strategy_service import StrategyService
                
                data_file = f"{self.data_dir}/{self.ticker}_mid_term_data.csv"
                if not os.path.exists(data_file):
                    return False, "No mid-term data file found"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Add technical indicators optimized for mid-term
                strategy_service = StrategyService()
                technical_data = strategy_service.get_technical_indicators(self.ticker)
                
                # Add technical indicators to dataframe
                for indicator, value in technical_data.items():
                    if isinstance(value, (int, float)):
                        df[f'technical_{indicator}'] = value
                
                df_with_indicators = df
                
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
                model_path = f"{self.models_dir}/{self.ticker}_mid_term_random_forest.pkl"
                
                if os.path.exists(model_path):
                    import joblib
                    model = joblib.load(model_path)
                    return True, "Loaded existing mid-term model"
                else:
                    # Train new mid-term model
                    from src.core.model_service import ModelService
                    
                    # Load data for training
                    data_file = f"{self.data_dir}/{self.ticker}_mid_term_data.csv"
                    if not os.path.exists(data_file):
                        return False, "No mid-term data file found for training"
                    
                    # Load data without setting index to avoid Date column issues
                    df = pd.read_csv(data_file)
                    
                    # Remove the Unnamed: 0 column if it exists
                    if 'Unnamed: 0' in df.columns:
                        df = df.drop('Unnamed: 0', axis=1)
                    
                    # Convert Date column to datetime and set as index
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.set_index('Date')
                    
                    # Remove any non-numeric columns that might cause issues
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    df = df[numeric_columns]
                    
                    # Handle NaN values by filling with forward fill and then backward fill
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    
                    # Drop any remaining rows with NaN values
                    df = df.dropna()
                    
                    model_service = ModelService()
                    
                    # Prepare features and target
                    if 'Close' in df.columns:
                        X = df.drop(['Close'], axis=1)
                        y = df['Close']
                        
                        # Train multiple models using all available algorithms
                        algorithms = [
                            'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost',
                            'linear_regression', 'ridge', 'lasso', 'elastic_net', 'svr', 'mlp', 'gaussian_process'
                        ]
                        
                        trained_models = {}
                        for algo in algorithms:
                            try:
                                result = model_service.train_model(algo, X, y, 'standard')
                                trained_models[algo] = result['model']
                                print(f"✅ Trained {algo} model for mid-term")
                            except Exception as e:
                                print(f"⚠️ Failed to train {algo}: {e}")
                                continue
                        
                        # Use the best performing model (random_forest as primary)
                        model = trained_models.get('random_forest')
                        if not model and trained_models:
                            model = list(trained_models.values())[0]
                    else:
                        return False, "No Close price column found for training"
                    
                    # Save model using joblib
                    import joblib
                    joblib.dump(model, model_path)
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
                    # Create new scaler using regular data file
                    data_file = f"{self.data_dir}/{self.ticker}_mid_term_data.csv"
                    if os.path.exists(data_file):
                        df = pd.read_csv(data_file)
                        
                        # Remove the Unnamed: 0 column if it exists
                        if 'Unnamed: 0' in df.columns:
                            df = df.drop('Unnamed: 0', axis=1)
                        
                        # Remove any non-numeric columns that might cause issues
                        numeric_columns = df.select_dtypes(include=[np.number]).columns
                        df = df[numeric_columns]
                        
                        # Handle NaN values
                        df = df.fillna(method='ffill').fillna(method='bfill').dropna()
                        
                        # Prepare features for scaling (exclude Close price)
                        feature_cols = [col for col in df.columns if col != 'Close']
                        if feature_cols:
                            scaler = MinMaxScaler()
                            scaler.fit(df[feature_cols])
                            
                            # Save scaler
                            joblib.dump(scaler, scaler_path)
                            return True, "Created new scaler"
                        else:
                            return False, "No feature columns for scaler"
                    else:
                        return False, "No data file for scaler"
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
                from src.core.strategy_service import StrategyService
                
                strategy_service = StrategyService()
                sentiment_df = strategy_service.analyze_sentiment(self.ticker, days_back=30)  # Monthly sentiment
                
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
                from src.core.strategy_service import StrategyService
                
                strategy_service = StrategyService()
                market_data = strategy_service.get_market_factors(self.ticker)
                
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
                from src.core.strategy_service import StrategyService
                from src.core.strategy_service import StrategyService
                
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
                import joblib
                
                # Load scikit-learn model and scaler
                model_path = f"{self.models_dir}/{self.ticker}_mid_term_random_forest.pkl"
                scaler_path = f"{self.models_dir}/{self.ticker}_mid_term_scaler.pkl"
                
                if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                    return False, "Model or scaler not found"
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Load data
                data_file = f"{self.data_dir}/{self.ticker}_mid_term_data.csv"
                if not os.path.exists(data_file):
                    return False, "Data file not found"
                
                df = pd.read_csv(data_file)
                
                # Remove the Unnamed: 0 column if it exists
                if 'Unnamed: 0' in df.columns:
                    df = df.drop('Unnamed: 0', axis=1)
                
                # Remove any non-numeric columns that might cause issues
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_columns]
                
                # Handle NaN values by filling with forward fill and then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Drop any remaining rows with NaN values
                df = df.dropna()
                
                # Prepare features (exclude target column)
                feature_cols = [col for col in df.columns if col != 'Close']
                if not feature_cols:
                    return False, "No feature columns found"
                
                # Use last row for prediction
                last_features = df[feature_cols].iloc[-1:].values
                
                # Handle NaN values
                if np.isnan(last_features).any():
                    # Fill NaN with median values
                    for i, col in enumerate(feature_cols):
                        if np.isnan(last_features[0, i]):
                            last_features[0, i] = df[col].median()
                
                # Scale features
                features_scaled = scaler.transform(last_features)
                
                # Make predictions (next N weeks)
                predictions = []
                for i in range(weeks_ahead * 5):  # 5 trading days per week
                    pred = model.predict(features_scaled)[0]
                    predictions.append(pred)
                
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
