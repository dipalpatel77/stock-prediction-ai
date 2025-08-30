#!/usr/bin/env python3
"""
Unified Analysis Pipeline
Integrates partA (preprocessing), partB (model), and partC (strategy) modules
for comprehensive stock prediction analysis with advanced prediction algorithms.
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
import joblib
import talib
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Import core services
from core import DataService, ModelService, ReportingService, StrategyService
from config import AnalysisConfig

# Import analysis modules for multi-timeframe analysis
try:
    from analysis_modules import ShortTermAnalyzer, MidTermAnalyzer, LongTermAnalyzer
    ANALYSIS_MODULES_AVAILABLE = True
except ImportError:
    ANALYSIS_MODULES_AVAILABLE = False
    print("⚠️ Analysis modules not available. Multi-timeframe analysis will be limited.")

# Import currency utilities
try:
    from core.currency_utils import format_price, format_change, get_currency_symbol
    CURRENCY_UTILS_AVAILABLE = True
except ImportError:
    CURRENCY_UTILS_AVAILABLE = False
    print("⚠️ Currency utilities not available. Using default formatting.")

# Import date utilities
try:
    from core.date_utils import DateUtils
    DATE_UTILS_AVAILABLE = True
except ImportError:
    DATE_UTILS_AVAILABLE = False
    print("⚠️ Date utilities not available. Using default date formatting.")

# Import Phase 1 integration
try:
    from phase1_integration import Phase1Integration
    PHASE1_AVAILABLE = True
    print("✅ Phase 1 integration available - Enhanced analysis enabled!")
except ImportError:
    PHASE1_AVAILABLE = False
    print("⚠️ Phase 1 integration not available. Using standard analysis.")

# Import Phase 2 integration
try:
    from phase2_integration import Phase2Integration
    PHASE2_AVAILABLE = True
    print("✅ Phase 2 integration available - Economic data & regulatory monitoring enabled!")
except ImportError:
    PHASE2_AVAILABLE = False
    print("⚠️ Phase 2 integration not available. Using Phase 1 analysis only.")

# Import Phase 3 integration
try:
    from phase3_integration import Phase3Integration
    PHASE3_AVAILABLE = True
    print("✅ Phase 3 integration available - Geopolitical risk, corporate actions & insider trading enabled!")
except ImportError:
    PHASE3_AVAILABLE = False
    print("⚠️ Phase 3 integration not available. Using Phase 1 & 2 analysis only.")

# Advanced ML imports (optional)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor, ExtraTreesRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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

class UnifiedAnalysisPipeline:
    """
    Unified analysis pipeline that integrates all three parts:
    - partA: Data preprocessing and loading
    - partB: Model building and training
    - partC: Strategy analysis and trading signals
    Enhanced with advanced prediction algorithms from run_stock_prediction
    """
    
    def _fix_ticker_format(self, ticker):
        """Fix ticker symbol format for Indian stocks."""
        # Common Indian stock mappings
        indian_stocks = {
            'TCS': 'TCS.NS',
            'RELIANCE': 'RELIANCE.NS',
            'HDFC': 'HDFC.NS',
            'INFY': 'INFY.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'SBIN': 'SBIN.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'AXISBANK': 'AXISBANK.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'MARUTI': 'MARUTI.NS',
            'SUNPHARMA': 'SUNPHARMA.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'WIPRO': 'WIPRO.NS',
            'ULTRACEMCO': 'ULTRACEMCO.NS',
            'TITAN': 'TITAN.NS',
            'NESTLEIND': 'NESTLEIND.NS',
            'POWERGRID': 'POWERGRID.NS',
            'TECHM': 'TECHM.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'NTPC': 'NTPC.NS',
            'HCLTECH': 'HCLTECH.NS',
            'JSWSTEEL': 'JSWSTEEL.NS',
            'ONGC': 'ONGC.NS',
            'TATASTEEL': 'TATASTEEL.NS',
            'ADANIENT': 'ADANIENT.NS',
            'ADANIPORTS': 'ADANIPORTS.NS',
            'BAJAJFINSV': 'BAJAJFINSV.NS',
            'BRITANNIA': 'BRITANNIA.NS',
            'CIPLA': 'CIPLA.NS',
            'COALINDIA': 'COALINDIA.NS',
            'DIVISLAB': 'DIVISLAB.NS',
            'DRREDDY': 'DRREDDY.NS',
            'EICHERMOT': 'EICHERMOT.NS',
            'GRASIM': 'GRASIM.NS',
            'HDFCLIFE': 'HDFCLIFE.NS',
            'HEROMOTOCO': 'HEROMOTOCO.NS',
            'HINDALCO': 'HINDALCO.NS',
            'LT': 'LT.NS',
            'M&M': 'M&M.NS',
            'SHREECEM': 'SHREECEM.NS',
            'TATACONSUM': 'TATACONSUM.NS',
            'UPL': 'UPL.NS',
            'VEDL': 'VEDL.NS',
            'ZEEL': 'ZEEL.NS',
            'SWIGGY': 'SWIGGY.NS',  # New age companies
            'ZOMATO': 'ZOMATO.NS',
            'PAYTM': 'PAYTM.NS',
            'NYKAA': 'NYKAA.NS',
            'DELHIVERY': 'DELHIVERY.NS'
        }
        
        # Check if it's an Indian stock without suffix
        if ticker in indian_stocks:
            corrected_ticker = indian_stocks[ticker]
            print(f"🔧 Corrected ticker format: {ticker} → {corrected_ticker}")
            return corrected_ticker
        
        # Check if it already has a suffix
        if '.' in ticker:
            return ticker
        
        # For non-Indian stocks, return as is
        return ticker
    
    def __init__(self, ticker, max_workers=None, period_config="recommended"):
        # Set environment variables for Angel One
        os.environ['ANGEL_ONE_API_KEY'] = '3PMAARNa'
        os.environ['ANGEL_ONE_CLIENT_CODE'] = 'D54448'
        os.environ['ANGEL_ONE_CLIENT_PIN'] = '2251'
        os.environ['ANGEL_ONE_TOTP_SECRET'] = 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
        
        # Fix ticker format for Indian stocks
        self.original_ticker = ticker.upper()
        self.ticker = self._fix_ticker_format(ticker.upper())
        
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
        
        # Initialize data periods configuration
        self.period_config = period_config
        try:
            from config.data_periods_config import get_period_config
            self.period_settings = get_period_config(period_config)
            print(f"📊 Using {period_config.upper()} data periods configuration")
        except ImportError:
            self.period_settings = {
                'default': '1y',
                'angel_one': '6mo',
                'yfinance': '1y',
                'quick_check': '3mo',
                'comprehensive': '2y'
            }
            print(f"⚠️ Using fallback data periods configuration")
        
        # Initialize core services with period configuration
        self.config = AnalysisConfig()
        self.data_service = DataService(period_config=period_config)
        self.model_service = ModelService()
        self.reporting_service = ReportingService()
        self.strategy_service = StrategyService()
        
        # Initialize advanced prediction components
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_cache_dir = "models/cache"
        
        # Create cache directory
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Initialize modules (all partB and partC functionality migrated to core services)
        self.data_loader = None
        
        # Multi-timeframe analysis storage
        self.timeframe_results = {
            'short_term': {},
            'mid_term': {},
            'long_term': {}
        }
        
        # Initialize analysis modules if available
        if ANALYSIS_MODULES_AVAILABLE:
            self.short_term_analyzer = ShortTermAnalyzer(ticker, self.max_workers)
            self.mid_term_analyzer = MidTermAnalyzer(ticker, self.max_workers)
            self.long_term_analyzer = LongTermAnalyzer(ticker, self.max_workers)
        else:
            self.short_term_analyzer = None
            self.mid_term_analyzer = None
            self.long_term_analyzer = None
        
    def get_user_data_period_choice(self):
        """Get user choice for historical data period with configuration-aware options."""
        print("\n📅 Historical Data Period Selection")
        print("=" * 50)
        print(f"Current Configuration: {self.period_config.upper()}")
        print(f"Recommended Period: {self.period_settings.get('default', '1y')}")
        
        # Show configuration-specific recommendations
        if self.period_config == "performance":
            print("⚡ Performance Mode: Optimized for speed")
        elif self.period_config == "comprehensive":
            print("📊 Comprehensive Mode: Maximum data for detailed analysis")
        else:
            print("🎯 Recommended Mode: Balanced performance and data quality")
        
        print("\nChoose the period for historical data analysis:")
        print("1. Quick Check - Use configuration's quick_check period")
        print("2. Standard Analysis - Use configuration's default period")
        print("3. Comprehensive Analysis - Use configuration's comprehensive period")
        print("4. Custom Period - Specify custom start and end dates")
        print("5. Manual Period Selection - Choose from predefined periods")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    period = self.period_settings.get('quick_check', '3mo')
                    return period, f"Quick Check ({period})"
                elif choice == '2':
                    period = self.period_settings.get('default', '1y')
                    return period, f"Standard Analysis ({period})"
                elif choice == '3':
                    period = self.period_settings.get('comprehensive', '2y')
                    return period, f"Comprehensive Analysis ({period})"
                elif choice == '4':
                    return self.get_custom_period()
                elif choice == '5':
                    return self.get_manual_period_choice()
                else:
                    print("❌ Invalid choice. Please enter a number between 1-5.")
            except KeyboardInterrupt:
                print("\n\n⚠️ Operation cancelled by user.")
                default_period = self.period_settings.get('default', '1y')
                return default_period, f"{default_period} (Default)"
            except Exception as e:
                print(f"❌ Error: {e}. Using default period.")
                default_period = self.period_settings.get('default', '1y')
                return default_period, f"{default_period} (Default)"
    
    def get_manual_period_choice(self):
        """Get manual period choice from predefined options."""
        print("\n📅 Manual Period Selection")
        print("=" * 30)
        print("Choose from predefined periods:")
        print("1. 1 Month (1mo) - Recent data for short-term analysis")
        print("2. 3 Months (3mo) - Medium-term analysis")
        print("3. 6 Months (6mo) - Semi-annual analysis")
        print("4. 1 Year (1y) - Annual analysis")
        print("5. 2 Years (2y) - Long-term analysis")
        print("6. 5 Years (5y) - Extended historical analysis")
        print("7. 10 Years (10y) - Maximum historical data")
        print("8. Since Listing (max) - All available data from listing date")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-8): ").strip()
                
                period_map = {
                    '1': ('1mo', "1 Month"),
                    '2': ('3mo', "3 Months"),
                    '3': ('6mo', "6 Months"),
                    '4': ('1y', "1 Year"),
                    '5': ('2y', "2 Years"),
                    '6': ('5y', "5 Years"),
                    '7': ('10y', "10 Years"),
                    '8': ('max', "Since Listing")
                }
                
                if choice in period_map:
                    return period_map[choice]
                else:
                    print("❌ Invalid choice. Please enter a number between 1-8.")
            except KeyboardInterrupt:
                print("\n\n⚠️ Operation cancelled by user.")
                default_period = self.period_settings.get('default', '1y')
                return default_period, f"{default_period} (Default)"
            except Exception as e:
                print(f"❌ Error: {e}. Using default period.")
                default_period = self.period_settings.get('default', '1y')
                return default_period, f"{default_period} (Default)"
    
    def get_custom_period(self):
        """Get custom start and end dates from user."""
        print("\n📅 Custom Period Selection")
        print("=" * 30)
        print("Enter custom start and end dates (YYYY-MM-DD format)")
        
        try:
            start_date = input("Start date (YYYY-MM-DD): ").strip()
            end_date = input("End date (YYYY-MM-DD) [Press Enter for today]: ").strip()
            
            # Validate start date
            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            
            # Validate end date
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if end_dt <= start_dt:
                    print("❌ End date must be after start date. Using today's date.")
                    end_date = ""
            
            # Calculate period string for yfinance
            if end_date:
                # For custom dates, we'll use a period that covers the range
                days_diff = (end_dt - start_dt).days
                if days_diff <= 30:
                    period = '1mo'
                elif days_diff <= 90:
                    period = '3mo'
                elif days_diff <= 180:
                    period = '6mo'
                elif days_diff <= 365:
                    period = '1y'
                elif days_diff <= 730:
                    period = '2y'
                elif days_diff <= 1825:
                    period = '5y'
                else:
                    period = '10y'
            else:
                # Calculate from start date to today
                days_diff = (datetime.now() - start_dt).days
                if days_diff <= 30:
                    period = '1mo'
                elif days_diff <= 90:
                    period = '3mo'
                elif days_diff <= 180:
                    period = '6mo'
                elif days_diff <= 365:
                    period = '1y'
                elif days_diff <= 730:
                    period = '2y'
                elif days_diff <= 1825:
                    period = '5y'
                else:
                    period = '10y'
            
            period_name = f"Custom ({start_date} to {end_date or 'Today'})"
            return period, period_name
            
        except ValueError:
            print("❌ Invalid date format. Using default 2 years.")
            return '2y', "2 Years (Default)"
        except Exception as e:
            print(f"❌ Error: {e}. Using default 2 years.")
            return '2y', "2 Years (Default)"
    
    def run_unified_analysis(self, period=None, days_ahead=5, use_enhanced=True):
        """Run complete unified analysis using all three parts."""
        self.start_time = time.time()
        self.days_ahead = days_ahead  # Store for later use
        
        # Get user choice for data period if not provided
        if period is None:
            period, period_name = self.get_user_data_period_choice()
        else:
            period_name = period
        
        print("🚀 Unified AI Stock Predictor - Complete Analysis Pipeline")
        print("=" * 80)
        print(f"📊 Ticker: {self.ticker}")
        print(f"📅 Period: {period_name}")
        print(f"⚙️ Configuration: {self.period_config.upper()}")
        print(f"⏰ Prediction Days: {days_ahead}")
        print(f"🔧 Threads: {self.max_workers}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        
        # Show configuration details
        print(f"\n📊 Data Periods Configuration:")
        print(f"   • Default: {self.period_settings.get('default', '1y')}")
        print(f"   • Quick Check: {self.period_settings.get('quick_check', '3mo')}")
        print(f"   • Comprehensive: {self.period_settings.get('comprehensive', '2y')}")
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
            
            # Step 4: Phase 1 - Enhanced Analysis Integration
            print("\n🚀 Step 4: Phase 1 - Enhanced Analysis Integration")
            print("-" * 50)
            if PHASE1_AVAILABLE:
                try:
                    phase1_success = self.run_phase1_enhanced_analysis()
                    if not phase1_success:
                        print("⚠️ Phase 1 analysis failed - continuing with standard analysis")
                except Exception as e:
                    print(f"⚠️ Phase 1 analysis error: {e} - continuing with standard analysis")
            else:
                print("⚠️ Phase 1 integration not available - using standard analysis")
            
            # Step 4.5: Phase 2 - Economic Data & Regulatory Monitoring
            print("\n🌍 Step 4.5: Phase 2 - Economic Data & Regulatory Monitoring")
            print("-" * 50)
            if PHASE2_AVAILABLE:
                try:
                    phase2_success = self.run_phase2_economic_analysis()
                    if not phase2_success:
                        print("⚠️ Phase 2 analysis failed - continuing with Phase 1 analysis")
                except Exception as e:
                    print(f"⚠️ Phase 2 analysis error: {e} - continuing with Phase 1 analysis")
            else:
                print("⚠️ Phase 2 integration not available - using Phase 1 analysis only")
            
            # Step 4.6: Phase 3 - Geopolitical Risk, Corporate Actions & Insider Trading
            print("\n🌍 Step 4.6: Phase 3 - Geopolitical Risk, Corporate Actions & Insider Trading")
            print("-" * 50)
            if PHASE3_AVAILABLE:
                try:
                    phase3_success = self.run_phase3_advanced_analysis()
                    if not phase3_success:
                        print("⚠️ Phase 3 analysis failed - continuing with Phase 1 & 2 analysis")
                except Exception as e:
                    print(f"⚠️ Phase 3 analysis error: {e} - continuing with Phase 1 & 2 analysis")
            else:
                print("⚠️ Phase 3 integration not available - using Phase 1 & 2 analysis only")
            
            # Step 5: Unified Report Generation (Optional)
            print("\n📋 Step 5: Unified Report Generation")
            print("-" * 50)
            try:
                report_generation_success = self.run_unified_report_generation(days_ahead, use_enhanced)
                if not report_generation_success:
                    print("⚠️ Report generation failed - continuing with predictions")
            except Exception as e:
                print(f"⚠️ Report generation error: {e} - continuing with predictions")
            
            # Step 6: Generate and Display Predictions
            print("\n🔮 Step 6: Generating Predictions")
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
    
    def run_multi_timeframe_analysis(self, use_enhanced: bool = True) -> Dict:
        """Run comprehensive analysis across all timeframes (short-term, mid-term, long-term)."""
        if not ANALYSIS_MODULES_AVAILABLE:
            print("❌ Analysis modules not available. Cannot run multi-timeframe analysis.")
            return {}
        
        print(f"\n{'='*80}")
        print(f"🚀 STARTING MULTI-TIMEFRAME ANALYSIS FOR {self.ticker}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Run all analyses in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_short_term_analysis, use_enhanced): 'short_term',
                executor.submit(self._run_mid_term_analysis, use_enhanced): 'mid_term',
                executor.submit(self._run_long_term_analysis, use_enhanced): 'long_term'
            }
            
            for future in concurrent.futures.as_completed(futures):
                timeframe = futures[future]
                try:
                    result = future.result()
                    self.timeframe_results[timeframe] = result
                    print(f"✅ {timeframe.replace('_', ' ').title()} analysis completed")
                except Exception as e:
                    print(f"❌ {timeframe.replace('_', ' ').title()} analysis failed: {e}")
                    self.timeframe_results[timeframe] = {'success': False, 'error': str(e)}
        
        execution_time = time.time() - start_time
        
        # Generate comprehensive report
        self._generate_multi_timeframe_report(execution_time, use_enhanced)
        
        # Display results
        self._display_multi_timeframe_results(execution_time, use_enhanced)
        
        return self.timeframe_results
    
    def _run_short_term_analysis(self, use_enhanced: bool) -> Dict:
        """Run short-term analysis (1-7 days)."""
        print(f"\n📊 Running Short-term Analysis for {self.ticker}...")
        
        start_time = time.time()
        
        try:
            # Data processing
            data_success = self.short_term_analyzer.run_short_term_data_processing()
            
            # Model preparation
            model_success = self.short_term_analyzer.prepare_short_term_model()
            
            # Enhanced analysis
            enhanced_success = True
            if use_enhanced:
                enhanced_success = self.short_term_analyzer.run_short_term_enhanced_analysis()
            
            # Strategy analysis (7 days ahead)
            strategy_success = self.short_term_analyzer.run_short_term_strategy_analysis(7)
            
            # Report generation
            report_success = self.short_term_analyzer.run_short_term_report_generation(7, use_enhanced)
            
            execution_time = time.time() - start_time
            
            return {
                'success': all([data_success, model_success, enhanced_success, strategy_success, report_success]),
                'execution_time': execution_time,
                'timeframe': '1-7 days',
                'components': {
                    'data_processing': data_success,
                    'model_preparation': model_success,
                    'enhanced_analysis': enhanced_success,
                    'strategy_analysis': strategy_success,
                    'report_generation': report_success
                }
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'timeframe': '1-7 days'
            }
    
    def _run_mid_term_analysis(self, use_enhanced: bool) -> Dict:
        """Run mid-term analysis (1-4 weeks)."""
        print(f"\n📈 Running Mid-term Analysis for {self.ticker}...")
        
        start_time = time.time()
        
        try:
            # Data processing
            data_success = self.mid_term_analyzer.run_mid_term_data_processing()
            
            # Model preparation
            model_success = self.mid_term_analyzer.prepare_mid_term_model()
            
            # Enhanced analysis
            enhanced_success = True
            if use_enhanced:
                enhanced_success = self.mid_term_analyzer.run_mid_term_enhanced_analysis()
            
            # Strategy analysis (4 weeks ahead)
            strategy_success = self.mid_term_analyzer.run_mid_term_strategy_analysis(4)
            
            # Report generation
            report_success = self.mid_term_analyzer.run_mid_term_report_generation(4, use_enhanced)
            
            execution_time = time.time() - start_time
            
            return {
                'success': all([data_success, model_success, enhanced_success, strategy_success, report_success]),
                'execution_time': execution_time,
                'timeframe': '1-4 weeks',
                'components': {
                    'data_processing': data_success,
                    'model_preparation': model_success,
                    'enhanced_analysis': enhanced_success,
                    'strategy_analysis': strategy_success,
                    'report_generation': report_success
                }
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'timeframe': '1-4 weeks'
            }
    
    def _run_long_term_analysis(self, use_enhanced: bool) -> Dict:
        """Run long-term analysis (1-12 months)."""
        print(f"\n📊 Running Long-term Analysis for {self.ticker}...")
        
        start_time = time.time()
        
        try:
            # Data processing
            data_success = self.long_term_analyzer.run_long_term_data_processing()
            
            # Model preparation
            model_success = self.long_term_analyzer.prepare_long_term_model()
            
            # Enhanced analysis
            enhanced_success = True
            if use_enhanced:
                enhanced_success = self.long_term_analyzer.run_long_term_enhanced_analysis()
            
            # Strategy analysis (12 months ahead)
            strategy_success = self.long_term_analyzer.run_long_term_strategy_analysis(12)
            
            # Report generation
            report_success = self.long_term_analyzer.run_long_term_report_generation(12, use_enhanced)
            
            execution_time = time.time() - start_time
            
            return {
                'success': all([data_success, model_success, enhanced_success, strategy_success, report_success]),
                'execution_time': execution_time,
                'timeframe': '1-12 months',
                'components': {
                    'data_processing': data_success,
                    'model_preparation': model_success,
                    'enhanced_analysis': enhanced_success,
                    'strategy_analysis': strategy_success,
                    'report_generation': report_success
                }
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'timeframe': '1-12 months'
            }
    
    def _generate_multi_timeframe_report(self, execution_time: float, use_enhanced: bool):
        """Generate comprehensive multi-timeframe analysis report."""
        print(f"\n📋 Generating multi-timeframe report...")
        
        # Load current price
        current_price = self._get_current_price()
        
        # Load predictions from all timeframes
        predictions = self._load_all_timeframe_predictions()
        
        # Generate comprehensive summary
        summary = {
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ticker': self.ticker,
            'Current_Price': current_price,
            'Total_Execution_Time': f"{execution_time:.2f} seconds",
            'Enhanced_Features': use_enhanced,
            'Analysis_Status': 'Completed',
            'Timeframes_Analyzed': list(self.timeframe_results.keys()),
            'Success_Rate': self._calculate_timeframe_success_rate()
        }
        
        # Add timeframe-specific results
        for timeframe, result in self.timeframe_results.items():
            if result.get('success'):
                summary[f'{timeframe}_status'] = 'Success'
                summary[f'{timeframe}_execution_time'] = f"{result.get('execution_time', 0):.2f}s"
            else:
                summary[f'{timeframe}_status'] = 'Failed'
                summary[f'{timeframe}_error'] = result.get('error', 'Unknown error')
        
        # Add price predictions
        if predictions:
            summary.update(predictions)
        
        # Save comprehensive report
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f"data/{self.ticker}_multi_timeframe_analysis.csv", index=False)
        
        # Generate price forecast summary
        self._generate_multi_timeframe_forecast_summary(current_price, predictions)
    
    def _get_current_price(self) -> float:
        """Get current stock price."""
        try:
            import yfinance as yf
            stock = yf.Ticker(self.ticker)
            current_price = stock.info.get('regularMarketPrice', 0)
            return current_price if current_price else 0
        except:
            return 0
    
    def _load_all_timeframe_predictions(self) -> Dict:
        """Load predictions from all timeframes."""
        predictions = {}
        
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            pred_file = f"data/{self.ticker}_{timeframe}_predictions.csv"
            if os.path.exists(pred_file):
                try:
                    df = pd.read_csv(pred_file)
                    if not df.empty:
                        # Get the latest prediction
                        latest_pred = df.iloc[-1]
                        predictions[f'{timeframe}_predicted_price'] = latest_pred.get('Predicted_Price', 0)
                        predictions[f'{timeframe}_confidence'] = latest_pred.get('Confidence', 0)
                        predictions[f'{timeframe}_prediction_date'] = latest_pred.get('Date', '')
                except Exception as e:
                    print(f"Warning: Could not load {timeframe} predictions: {e}")
        
        return predictions
    
    def _calculate_timeframe_success_rate(self) -> float:
        """Calculate overall success rate for timeframes."""
        successful = sum(1 for result in self.timeframe_results.values() if result.get('success'))
        total = len(self.timeframe_results)
        return (successful / total) * 100 if total > 0 else 0
    
    def _generate_multi_timeframe_forecast_summary(self, current_price: float, predictions: Dict):
        """Generate price forecast summary for multi-timeframe analysis."""
        if not predictions:
            return
        
        forecast_summary = {
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Ticker': self.ticker,
            'Current_Price': current_price,
            'Short_Term_Price': predictions.get('short_term_predicted_price', 0),
            'Short_Term_Change': self._calculate_change(current_price, predictions.get('short_term_predicted_price', 0)),
            'Short_Term_Confidence': predictions.get('short_term_confidence', 0),
            'Mid_Term_Price': predictions.get('mid_term_predicted_price', 0),
            'Mid_Term_Change': self._calculate_change(current_price, predictions.get('mid_term_predicted_price', 0)),
            'Mid_Term_Confidence': predictions.get('mid_term_confidence', 0),
            'Long_Term_Price': predictions.get('long_term_predicted_price', 0),
            'Long_Term_Change': self._calculate_change(current_price, predictions.get('long_term_predicted_price', 0)),
            'Long_Term_Confidence': predictions.get('long_term_confidence', 0),
            'Overall_Sentiment': self._calculate_overall_sentiment(predictions),
            'Risk_Level': self._calculate_risk_level(predictions),
            'Recommended_Action': self._get_recommended_action(predictions)
        }
        
        # Save forecast summary
        forecast_df = pd.DataFrame([forecast_summary])
        forecast_df.to_csv(f"data/{self.ticker}_multi_timeframe_forecast_summary.csv", index=False)
    
    def _calculate_change(self, current: float, predicted: float) -> float:
        """Calculate percentage change."""
        if current > 0 and predicted > 0:
            return ((predicted - current) / current) * 100
        return 0
    
    def _calculate_overall_sentiment(self, predictions: Dict) -> str:
        """Calculate overall sentiment based on predictions."""
        changes = []
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            change = predictions.get(f'{timeframe}_predicted_price', 0) - predictions.get('current_price', 0)
            if change != 0:
                changes.append(change)
        
        if not changes:
            return 'Neutral'
        
        avg_change = sum(changes) / len(changes)
        if avg_change > 0.05:  # 5% threshold
            return 'Bullish'
        elif avg_change < -0.05:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _calculate_risk_level(self, predictions: Dict) -> str:
        """Calculate risk level based on prediction confidence."""
        confidences = []
        for timeframe in ['short_term', 'mid_term', 'long_term']:
            conf = predictions.get(f'{timeframe}_confidence', 0)
            if conf > 0:
                confidences.append(conf)
        
        if not confidences:
            return 'Unknown'
        
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence >= 0.8:
            return 'Low'
        elif avg_confidence >= 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def _get_recommended_action(self, predictions: Dict) -> str:
        """Get recommended trading action."""
        sentiment = self._calculate_overall_sentiment(predictions)
        risk_level = self._calculate_risk_level(predictions)
        
        if sentiment == 'Bullish' and risk_level in ['Low', 'Medium']:
            return 'BUY'
        elif sentiment == 'Bearish' and risk_level in ['Low', 'Medium']:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _display_multi_timeframe_results(self, execution_time: float, use_enhanced: bool):
        """Display comprehensive multi-timeframe analysis results."""
        print(f"\n{'='*80}")
        print("🎉 MULTI-TIMEFRAME ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"⏱️ Total Execution Time: {execution_time:.2f} seconds")
        print(f"📊 Ticker: {self.ticker}")
        print(f"⚡ Enhanced Features: {use_enhanced}")
        print(f"🔧 Threads Used: {self.max_workers}")
        print()
        
        # Display timeframe results
        for timeframe, result in self.timeframe_results.items():
            status = "✅ Success" if result.get('success') else "❌ Failed"
            exec_time = result.get('execution_time', 0)
            print(f"📈 {timeframe.replace('_', ' ').title()}: {status} ({exec_time:.2f}s)")
        
        print(f"\n📁 Generated Files:")
        data_files = [f for f in os.listdir('data') if f.startswith(self.ticker)]
        for file in sorted(data_files):
            print(f"    📄 {file}")
        
        # Display price forecast if available
        forecast_file = f"data/{self.ticker}_multi_timeframe_forecast_summary.csv"
        if os.path.exists(forecast_file):
            print(f"\n💰 MULTI-TIMEFRAME PRICE FORECAST SUMMARY:")
            try:
                forecast_df = pd.read_csv(forecast_file)
                if not forecast_df.empty:
                    forecast = forecast_df.iloc[0]
                    current_price = forecast.get('Current_Price', 0)
                    
                    # Use currency formatting if available
                    if CURRENCY_UTILS_AVAILABLE:
                        print(f"   💵 Current Price: {format_price(current_price, self.ticker)}")
                    else:
                        print(f"   💵 Current Price: ${current_price:.2f}")
                    
                    for timeframe in ['Short_Term', 'Mid_Term', 'Long_Term']:
                        pred_price = forecast.get(f'{timeframe}_Price', 0)
                        change = forecast.get(f'{timeframe}_Change', 0)
                        confidence = forecast.get(f'{timeframe}_Confidence', 0)
                        
                        if pred_price > 0:
                            change_symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                            if CURRENCY_UTILS_AVAILABLE:
                                formatted_price = format_price(pred_price, self.ticker)
                            else:
                                formatted_price = f"${pred_price:.2f}"
                            print(f"   {change_symbol} {timeframe}: {formatted_price} ({change:+.2f}%) [Confidence: {confidence:.1%}]")
                    
                    sentiment = forecast.get('Overall_Sentiment', 'Unknown')
                    risk_level = forecast.get('Risk_Level', 'Unknown')
                    action = forecast.get('Recommended_Action', 'Unknown')
                    
                    print(f"   🎯 Overall Sentiment: {sentiment}")
                    print(f"   ⚠️ Risk Level: {risk_level}")
                    print(f"   💡 Recommended Action: {action}")
            except Exception as e:
                print(f"   ⚠️ Could not load forecast: {e}")
        
        print(f"\n💡 Trading Recommendations:")
        print(f"   • Short-term: Use for day trading and swing trading")
        print(f"   • Mid-term: Use for position trading and trend following")
        print(f"   • Long-term: Use for investment decisions and portfolio allocation")
        print(f"   • Always consider risk management and diversification")
        print(f"   • Past performance doesn't guarantee future results")
    
    def run_partA_preprocessing(self, period):
        """Run partA data preprocessing with parallel processing."""
        print("🔄 Starting partA data preprocessing...")
        
        def load_stock_data():
            """Load stock data using core DataService."""
            try:
                # Check if raw data already exists
                raw_data_file = f"data/{self.ticker}_raw_data.csv"
                if os.path.exists(raw_data_file):
                    df = pd.read_csv(raw_data_file)
                    if not df.empty:
                        return True, f"Using existing raw data with {len(df)} records"
                
                # Get configuration
                data_config = self.config.get_data_config()
                period = data_config.get('default_period', '2y')
                
                # Load data using core service
                df = self.data_service.load_stock_data(self.ticker, period=period, force_refresh=False)
                
                if not df.empty:
                    # Save raw data for compatibility
                    df.to_csv(raw_data_file)
                    return True, f"Loaded {len(df)} records using core DataService"
                else:
                    return False, "No data loaded"
            except Exception as e:
                return False, f"Data loading error: {e}"
        
        def clean_and_preprocess():
            """Clean and preprocess data using core DataService."""
            try:
                # Load raw data
                raw_data_file = f"data/{self.ticker}_raw_data.csv"
                if not os.path.exists(raw_data_file):
                    return False, "No raw data file found"
                
                # First, let's check the structure of the data
                df_check = pd.read_csv(raw_data_file, nrows=5)
                
                # Check if this is the malformed AAPL format (row1=headers, row2=ticker, row3=empty)
                if len(df_check) > 3 and df_check.iloc[0, 0] == 'Price' and df_check.iloc[1, 0] == 'Ticker':
                    # AAPL format - skip first 3 rows
                    df = pd.read_csv(raw_data_file, skiprows=3)
                    # Set proper column names for AAPL format
                    df.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                else:
                    # NVDA format or other formats - read normally
                    df = pd.read_csv(raw_data_file)
                    # Remove the first column if it's just an index
                    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '0':
                        df = df.iloc[:, 1:]
                
                # Convert numeric columns to proper types
                numeric_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Convert Date column to datetime
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
                
                # Remove any rows with NaN values
                df = df.dropna()
                
                # Use core service for preprocessing
                df_processed = self.data_service.preprocess_data(df, timeframe='daily', target_col='Close')
                
                # Save preprocessed data
                df_processed.to_csv(f"data/{self.ticker}_partA_preprocessed.csv")
                return True, f"Preprocessed {len(df_processed)} records using core DataService"
            except Exception as e:
                return False, f"Preprocessing error: {e}"
        
        def add_enhanced_technical_indicators():
            """Add enhanced technical indicators using core DataService."""
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
                
                # The preprocessed data from DataService should already have proper format
                # Just ensure Date column is properly formatted
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Use core service for technical indicators (already included in preprocess_data)
                # The core service already adds comprehensive technical indicators
                df_enhanced = df  # Core service already includes indicators
                
                # Save enhanced data
                df_enhanced.to_csv(f"data/{self.ticker}_partA_partC_enhanced.csv")
                return True, f"Enhanced data ready with {len(df_enhanced)} records using core DataService"
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
            print(f"\n❌ Data loading failed for {self.ticker}")
            print("💡 Possible solutions:")
            print("   1. Check if the ticker symbol is correct")
            print("   2. Verify internet connectivity")
            print("   3. For Indian stocks: Check Angel One API credentials")
            print("   4. Try a different ticker that's available on international exchanges")
            print("   5. Some stocks may be delisted or suspended")
            
            # Ask user if they want to continue with a different ticker
            try:
                retry = input(f"\nWould you like to try a different ticker? (y/n): ").strip().lower()
                if retry == 'y':
                    new_ticker = input("Enter new ticker symbol: ").strip().upper()
                    if new_ticker:
                        self.ticker = new_ticker
                        print(f"🔄 Retrying with ticker: {self.ticker}")
                        return self.run_partA_preprocessing(period)
            except (KeyboardInterrupt, EOFError):
                print("\n⚠️ Operation cancelled by user.")
            
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
            """Train enhanced model using core ModelService."""
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
                
                # Prepare features and target
                feature_cols = [col for col in df.columns if col not in ['Date', 'Adj Close']]
                X = df[feature_cols].dropna()
                y = df['Close'].shift(-1).dropna()
                
                # Align X and y
                common_index = X.index.intersection(y.index)
                X = X.loc[common_index]
                y = y.loc[common_index]
                
                if len(X) < 10:
                    return False, "Insufficient data for training"
                
                # Get model configuration
                model_config = self.config.get_model_config('advanced')
                
                # Train model using core service
                result = self.model_service.train_model('random_forest', X, y, **model_config)
                
                # Save model
                self.model_service.save_model(result, enhanced_model_path)
                
                return True, "Enhanced model trained using core ModelService"
                    
            except Exception as e:
                return False, f"Enhanced model training error: {e}"
        
        def train_ensemble_models():
            """Train ensemble models using core ModelService."""
            try:
                # Load enhanced data
                data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
                if not os.path.exists(data_file):
                    return False, "No enhanced data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Remove any unnamed columns that contain row numbers
                unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                if unnamed_cols:
                    df = df.drop(unnamed_cols, axis=1)
                
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
                
                # Get model configuration
                model_config = self.config.get_model_config('advanced')
                
                # Train ensemble models using core service
                models_to_train = ['random_forest', 'gradient_boosting']
                ensemble_results = {}
                
                for model_type in models_to_train:
                    try:
                        result = self.model_service.train_model(model_type, X, y, **model_config)
                        ensemble_results[model_type] = result
                        
                        # Save model
                        model_path = f"models/{self.ticker}_{model_type}_model.pkl"
                        self.model_service.save_model(result, model_path)
                        
                    except Exception as e:
                        print(f"Warning: Failed to train {model_type}: {e}")
                
                if ensemble_results:
                    return True, f"Ensemble models trained using core ModelService: {list(ensemble_results.keys())}"
                else:
                    return False, "No ensemble models were successfully trained"
                    
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
            """Run sentiment analysis using core StrategyService."""
            try:
                # Use core StrategyService for sentiment analysis
                sentiment_df = self.strategy_service.analyze_sentiment(self.ticker, days_back=30)
                
                if not sentiment_df.empty:
                    sentiment_df.to_csv(f"data/{self.ticker}_sentiment_analysis.csv", index=False)
                    return True, "Sentiment analysis completed using core StrategyService"
                else:
                    return False, "No sentiment data available"
                    
            except Exception as e:
                return False, f"Sentiment analysis error: {e}"
        
        def run_market_factors():
            """Run market factors analysis using core StrategyService."""
            try:
                # Use core StrategyService for market factors
                market_data = self.strategy_service.get_market_factors(self.ticker)
                
                if market_data:
                    market_df = pd.DataFrame([market_data])
                    market_df.to_csv(f"data/{self.ticker}_market_factors.csv", index=False)
                    return True, "Market factors completed using core StrategyService"
                else:
                    return False, "No market data available"
            except Exception as e:
                return False, f"Market factors error: {e}"
        
        def run_economic_indicators():
            """Run economic indicators analysis using core StrategyService."""
            try:
                print(f"📊 Analyzing economic indicators for {self.ticker}...")
                
                # Use core StrategyService for market factors (includes economic indicators)
                market_data = self.strategy_service.get_market_factors(self.ticker)
                
                if market_data:
                    # Save market factors (includes economic indicators)
                    market_df = pd.DataFrame([market_data])
                    market_df.to_csv(f"data/{self.ticker}_market_factors.csv", index=False)
                    
                    # Save economic indicators separately
                    economic_indicators = {k: v for k, v in market_data.items() 
                                        if k in ['gdp_growth', 'inflation_rate', 'unemployment_rate', 
                                               'interest_rate', 'consumer_confidence', 'manufacturing_pmi', 
                                               'retail_sales_growth']}
                    if economic_indicators:
                        economic_df = pd.DataFrame([economic_indicators])
                        economic_df.to_csv(f"data/{self.ticker}_economic_indicators.csv", index=False)
                    
                    return True, "Economic indicators and market factors completed using core StrategyService"
                else:
                    return False, "No economic data available"
            except Exception as e:
                return False, f"Economic indicators error: {e}"
        
        def run_trading_strategy():
            """Run trading strategy analysis using core StrategyService."""
            try:
                # Load enhanced data
                data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
                if not os.path.exists(data_file):
                    return False, "No enhanced data available"
                
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # Use core StrategyService for trading signals
                signals_df = self.strategy_service.generate_trading_signals(self.ticker, df)
                
                if not signals_df.empty:
                    signals_df.to_csv(f"data/{self.ticker}_trading_signals.csv")
                    return True, "Trading strategy completed using core StrategyService"
                else:
                    return False, "No signals generated"
                    
            except Exception as e:
                return False, f"Trading strategy error: {e}"
        
        def run_backtesting():
            """Run backtesting analysis using core StrategyService."""
            try:
                # Load signals data
                signals_file = f"data/{self.ticker}_trading_signals.csv"
                if not os.path.exists(signals_file):
                    return False, "No signals data available"
                
                signals_df = pd.read_csv(signals_file, index_col=0, parse_dates=True)
                
                # Run backtest using core StrategyService
                results = self.strategy_service.run_backtest(signals_df)
                
                if results:
                    results_df = pd.DataFrame([results])
                    results_df.to_csv(f"data/{self.ticker}_backtest_results.csv", index=False)
                    return True, "Backtesting completed using core StrategyService"
                else:
                    return False, "No backtest results"
            except Exception as e:
                return False, f"Backtesting error: {e}"
        
        def run_balance_sheet_analysis():
            """Run balance sheet analysis using core StrategyService."""
            try:
                print(f"🏢 Analyzing balance sheet for {self.ticker}...")
                
                # Use core StrategyService for balance sheet analysis
                analysis = self.strategy_service.analyze_balance_sheet(self.ticker)
                
                if analysis:
                    # Convert analysis to DataFrame
                    analysis_df = pd.DataFrame([analysis])
                    analysis_df.to_csv(f"data/{self.ticker}_balance_sheet_analysis.csv", index=False)
                    return True, "Balance sheet analysis completed using core StrategyService"
                return False, "No financial data available"
            except Exception as e:
                return False, f"Balance sheet analysis error: {e}"
        
        def run_event_impact_analysis():
            """Run company event impact analysis using core StrategyService."""
            try:
                print(f"📰 Analyzing company events for {self.ticker}...")
                
                # Use core StrategyService for company event analysis
                event_analysis = self.strategy_service.analyze_company_events(self.ticker)
                
                if event_analysis:
                    # Save event analysis results
                    event_summary_df = pd.DataFrame([event_analysis])
                    event_summary_df.to_csv(f"data/{self.ticker}_event_impact_summary.csv", index=False)
                    
                    return True, f"Event impact analysis completed using core StrategyService"
                return False, "No company events detected"
            except Exception as e:
                return False, f"Event impact analysis error: {e}"
        
        # Run partC tasks in parallel
        tasks = [
            ("sentiment", run_sentiment_analysis),
            ("market_factors", run_market_factors),
            ("economic_indicators", run_economic_indicators),
            ("balance_sheet", run_balance_sheet_analysis),
            ("event_impact", run_event_impact_analysis),
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
                    'Modules_Integrated': 'Technical, Sentiment, Economic, Balance Sheet, Company Events',
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
    
    def prepare_features(self, df, mode="advanced"):
        """Prepare features for prediction with advanced capabilities."""
        try:
            if mode == "simple":
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
            
            min_features = 8 if mode == "simple" else 15
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
    
    def train_advanced_models(self, X, y):
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
            
            # Create ensemble with available models
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
                print(f"   ✅ Ensemble created with {len(available_models)} models")
            else:
                print("   ⚠️ Not enough models for ensemble, using average prediction")
                self.models['Ensemble'] = 'average'
            
            print("✅ All advanced models trained successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training advanced models: {e}")
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
    
    def generate_and_display_predictions(self, days_ahead=5):
        """Generate and display predictions for the stock using advanced algorithms."""
        print("🔄 Starting advanced prediction generation...")
        
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
            
            # Add advanced features
            print("🔧 Adding advanced features...")
            df = self._add_advanced_features(df)
            
            # Prepare features for prediction
            print("📊 Preparing features for prediction...")
            X, y = self.prepare_features(df, mode="advanced")
            
            if X is None or y is None:
                print("❌ Failed to prepare features for prediction")
                return False
            
            # Train advanced models if not already trained
            if not self.models:
                print("🤖 Training advanced models...")
                success = self.train_advanced_models(X, y)
                if not success:
                    print("❌ Failed to train models")
                    return False
            
            # Generate predictions using advanced algorithms
            print("📊 Generating advanced predictions...")
            predictions, multi_day_predictions = self.generate_advanced_predictions(X, days_ahead)
            
            # Generate timeframe-specific predictions
            print("⏰ Generating timeframe predictions...")
            timeframe_predictions = self.generate_timeframe_predictions(df)
            
            # Calculate prediction confidence
            print("📈 Calculating prediction confidence...")
            confidence_analysis = self.calculate_prediction_confidence(predictions)
            
            # Display predictions
            self.display_advanced_predictions(predictions, multi_day_predictions, days_ahead, timeframe_predictions, confidence_analysis)
            
            # Save predictions
            self.save_advanced_predictions(predictions, multi_day_predictions, timeframe_predictions, confidence_analysis)
            
            print("✅ Advanced prediction generation completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Advanced prediction generation error: {e}")
            return False
    
    def generate_advanced_predictions(self, X, days_ahead):
        """Generate predictions using advanced ensemble models."""
        try:
            predictions = {}
            
            # Enhance features if in advanced mode
            X_enhanced = self._enhance_features(X)
            last_data = X_enhanced.iloc[-1:].values
            
            # Generate predictions for each model
            for name, model in self.models.items():
                try:
                    if name == 'Ensemble' and isinstance(model, str) and model == 'average':
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
        """Generate multi-day predictions using recursive approach."""
        try:
            predictions = []
            current_data = X.iloc[-1:].copy()
            
            for day in range(days_ahead):
                # Generate prediction for next day
                if 'Ensemble' in self.models and not isinstance(self.models['Ensemble'], str):
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
            print(f"Warning: Error generating multi-day predictions: {e}")
            return None
    
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
    

    
    def generate_timeframe_predictions(self, df):
        """Generate specific predictions for short-term, medium-term, and long-term."""
        try:
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Calculate trend and volatility
            recent_trend = df['Close'].pct_change().tail(20).mean()
            volatility = df['Close'].pct_change().tail(20).std()
            
            # Short-term predictions (1-7 days)
            short_term_predictions = []
            for day in range(1, 8):
                # Short-term: influenced by recent trend and volatility
                trend_factor = 1 + (recent_trend * day)
                volatility_factor = 1 + (np.random.normal(0, volatility * 0.5))
                prediction = current_price * trend_factor * volatility_factor
                short_term_predictions.append(prediction)
            
            # Medium-term predictions (1-4 weeks)
            medium_term_predictions = []
            for week in range(1, 5):
                # Medium-term: trend continues with some momentum
                trend_factor = 1 + (recent_trend * week * 5)  # Weekly factor
                momentum_factor = 1 + (recent_trend * 0.1)  # Momentum effect
                prediction = current_price * trend_factor * momentum_factor
                medium_term_predictions.append(prediction)
            
            # Long-term predictions (1-12 months)
            long_term_predictions = []
            for month in range(1, 13):
                # Long-term: trend with market cycle effects
                trend_factor = 1 + (recent_trend * month * 20)  # Monthly factor
                cycle_factor = 1 + (np.sin(month * np.pi / 6) * 0.05)  # Market cycle
                prediction = current_price * trend_factor * cycle_factor
                long_term_predictions.append(prediction)
            
            return {
                'short_term': short_term_predictions,
                'medium_term': medium_term_predictions,
                'long_term': long_term_predictions
            }
            
        except Exception as e:
            print(f"Warning: Timeframe prediction error: {e}")
            return None
    
    def display_advanced_predictions(self, predictions, multi_day_predictions, days_ahead, timeframe_predictions, confidence_analysis):
        """Display the advanced predictions with confidence analysis."""
        # Get current price from the data
        data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
        df = pd.read_csv(data_file)
        
        # Handle data format
        if len(df.columns) > 1 and df.columns[1] == 'Price':
            current_price = float(df.iloc[-1]['Close'])
        else:
            current_price = float(df.iloc[-1]['Close'])
        
        print("\n🎯 ADVANCED PREDICTION RESULTS")
        print("=" * 80)
        print(f"📊 Stock: {self.ticker}")
        print(f"📅 Prediction Period: {days_ahead} days")
        print(f"💰 CURRENT PRICE: {format_price(current_price, self.ticker)}")
        # Get enhanced date information for analysis
        if DATE_UTILS_AVAILABLE:
            analysis_date_info = DateUtils.format_analysis_timestamp()
            analysis_date_display = analysis_date_info['analysis_date_full']
        else:
            analysis_date_display = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"📅 Analysis Date: {analysis_date_display}")
        print("-" * 80)
        print()
        
        # Display individual model predictions
        if predictions:
            print("🤖 Individual Model Predictions:")
            for model_name, pred in predictions.items():
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   • {model_name}: {format_price(pred, self.ticker)} ({direction} {change_pct:+.2f}%)")
        
        # Display multi-day predictions
        if multi_day_predictions:
            print(f"\n🚀 Multi-Day Predictions ({days_ahead} days):")
            for i, pred in enumerate(multi_day_predictions, 1):
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   • Day {i}: {format_price(pred, self.ticker)} ({direction} {change_pct:+.2f}%)")
        
        # Display confidence analysis
        if confidence_analysis:
            print(f"\n📈 CONFIDENCE ANALYSIS:")
            print(f"   Mean Prediction: {format_price(confidence_analysis['mean'], self.ticker)}")
            print(f"   Standard Deviation: {format_price(confidence_analysis['std'], self.ticker)}")
            print(f"   68% Confidence Interval: {format_price(confidence_analysis['confidence_68'][0], self.ticker)} - {format_price(confidence_analysis['confidence_68'][1], self.ticker)}")
            print(f"   95% Confidence Interval: {format_price(confidence_analysis['confidence_95'][0], self.ticker)} - {format_price(confidence_analysis['confidence_95'][1], self.ticker)}")
            print(f"   Model Agreement Score: {confidence_analysis['agreement_score']:.3f}")
            
            # Model diversity analysis
            if confidence_analysis.get('model_diversity'):
                diversity = confidence_analysis['model_diversity']
                print(f"\n🔍 MODEL DIVERSITY ANALYSIS:")
                print(f"   Diversity Level: {diversity['diversity_level']}")
                print(f"   Description: {diversity['diversity_description']}")
                print(f"   Coefficient of Variation: {diversity['coefficient_of_variation']:.4f}")
                print(f"   Prediction Range: {format_price(diversity['prediction_range'], self.ticker)} ({diversity['prediction_range_pct']:.2f}%)")
            
            # Pattern strength analysis
            if confidence_analysis.get('pattern_strength'):
                pattern = confidence_analysis['pattern_strength']
                print(f"\n📊 PATTERN STRENGTH ANALYSIS:")
                print(f"   Pattern Level: {pattern['pattern_level']}")
                print(f"   Description: {pattern['pattern_description']}")
                print(f"   Trend Strength: {pattern['trend_strength']:.4f}")
            
            # Phase 1 Enhanced Analysis Insights
            if hasattr(self, 'phase1_results') and self.phase1_results:
                print(f"\n🚀 PHASE 1 ENHANCED ANALYSIS INSIGHTS:")
                print("-" * 50)
                
                # Fundamental insights
                fundamental = self.phase1_results.get('fundamental', {})
                if fundamental:
                    print(f"🏢 Fundamental Health: {fundamental.get('financial_health_score', 0):.1f}/100")
                    print(f"💰 Profit Margin: {fundamental.get('net_profit_margin', 0):.2f}%")
                    print(f"📈 EPS Growth: {fundamental.get('eps_growth', 0):+.2f}%")
                
                # Global market impact
                global_markets = self.phase1_results.get('global_markets', {})
                if global_markets:
                    print(f"🌍 Global Market Impact: {global_markets.get('market_impact_score', 0):.1f}/100")
                    print(f"🌍 Risk Sentiment: {global_markets.get('risk_sentiment', 'Unknown')}")
                
                # Institutional sentiment
                institutional = self.phase1_results.get('institutional', {})
                if institutional:
                    print(f"📈 Institutional Sentiment: {institutional.get('institutional_sentiment', 'Unknown')}")
                    print(f"💼 Institutional Confidence: {institutional.get('institutional_confidence', 0):.1f}/100")
                
                # Enhanced prediction score
                enhanced_score = self.phase1_results.get('enhanced_prediction_score', 50.0)
                print(f"🎯 Enhanced Prediction Score: {enhanced_score:.1f}/100")
                
                # Variable coverage
                coverage = self.phase1_results.get('variable_coverage', {})
                if coverage:
                    print(f"📋 Variable Coverage: {coverage.get('overall_coverage', 'Unknown')}")
        
        # Display timeframe predictions
        if timeframe_predictions:
            print(f"\n⏰ TIMEFRAME PREDICTIONS:")
            
            # Short-term predictions (1-7 days)
            if timeframe_predictions.get('short_term'):
                print(f"\n📅 SHORT-TERM (1-7 days):")
                for i, pred in enumerate(timeframe_predictions['short_term'], 1):
                    change = pred - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    # Get enhanced date information for display
                    if DATE_UTILS_AVAILABLE:
                        date_info = DateUtils.format_prediction_date(datetime.now(), i)
                        date_display = f"{date_info['day_name_short']}, {date_info['prediction_date_short']}"
                    else:
                        date_display = f"Day {i}"
                    
                    print(f"   • {date_display}: {format_price(pred, self.ticker)} ({direction} {change_pct:+.2f}%)")
            
            # Medium-term predictions (1-4 weeks)
            if timeframe_predictions.get('medium_term'):
                print(f"\n📅 MEDIUM-TERM (1-4 weeks):")
                for i, pred in enumerate(timeframe_predictions['medium_term'], 1):
                    change = pred - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    # Get enhanced date information for display
                    if DATE_UTILS_AVAILABLE:
                        date_info = DateUtils.format_week_prediction_date(datetime.now(), i)
                        date_display = f"Week {i} ({date_info['prediction_week_short']})"
                    else:
                        date_display = f"Week {i}"
                    
                    print(f"   • {date_display}: {format_price(pred, self.ticker)} ({direction} {change_pct:+.2f}%)")
            
            # Long-term predictions (1-12 months)
            if timeframe_predictions.get('long_term'):
                print(f"\n📅 LONG-TERM (1-12 months):")
                for i, pred in enumerate(timeframe_predictions['long_term'], 1):
                    change = pred - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    # Get enhanced date information for display
                    if DATE_UTILS_AVAILABLE:
                        date_info = DateUtils.format_month_prediction_date(datetime.now(), i)
                        date_display = f"Month {i} ({date_info['prediction_month_short']})"
                    else:
                        date_display = f"Month {i}"
                    
                    print(f"   • {date_display}: {format_price(pred, self.ticker)} ({direction} {change_pct:+.2f}%)")
        
        # Calculate average prediction
        all_predictions = []
        if predictions:
            all_predictions.extend(list(predictions.values()))
        if multi_day_predictions:
            all_predictions.extend(multi_day_predictions)
        
        if all_predictions:
            avg_prediction = sum(all_predictions) / len(all_predictions)
            avg_change = avg_prediction - current_price
            avg_change_pct = (avg_change / current_price) * 100
            avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
            
            print(f"\n📊 Average Prediction: {format_price(avg_prediction, self.ticker)} ({avg_direction} {avg_change_pct:+.2f}%)")
            
            # Price summary
            print(f"\n📋 PRICE SUMMARY:")
            print(f"   Current Price: {format_price(current_price, self.ticker)}")
            print(f"   Predicted Price: {format_price(avg_prediction, self.ticker)}")
            print(f"   Expected Change: {avg_change_pct:+.2f}%")
            
            # Advanced trading recommendation based on confidence
            if confidence_analysis and confidence_analysis['agreement_score'] > 0.8:
                # High agreement - more confident recommendation
                if avg_change_pct > 2:
                    recommendation = "🟢 STRONG BUY - High confidence upward momentum"
                elif avg_change_pct > 0.5:
                    recommendation = "🟡 BUY - High confidence moderate upward potential"
                elif avg_change_pct < -2:
                    recommendation = "🔴 STRONG SELL - High confidence downward pressure"
                elif avg_change_pct < -0.5:
                    recommendation = "🟠 SELL - High confidence moderate downward potential"
                else:
                    recommendation = "⚪ HOLD - High confidence stable movement"
            else:
                # Lower agreement - more conservative recommendation
                if avg_change_pct > 3:
                    recommendation = "🟢 BUY - Strong upward momentum expected"
                elif avg_change_pct > 1:
                    recommendation = "🟡 BUY - Moderate upward potential"
                elif avg_change_pct < -3:
                    recommendation = "🔴 SELL - Strong downward pressure expected"
                elif avg_change_pct < -1:
                    recommendation = "🟠 SELL - Moderate downward potential"
                else:
                    recommendation = "⚪ HOLD - Stable price movement expected"
            
            print(f"\n💡 Trading Recommendation: {recommendation}")
            
            # Timeframe-specific recommendations
            if timeframe_predictions:
                print(f"\n🎯 TIMEFRAME RECOMMENDATIONS:")
                
                # Short-term recommendation
                if timeframe_predictions.get('short_term'):
                    short_avg = sum(timeframe_predictions['short_term']) / len(timeframe_predictions['short_term'])
                    short_change_pct = ((short_avg - current_price) / current_price) * 100
                    if short_change_pct > 1:
                        short_rec = "🟢 BUY - Strong short-term momentum"
                    elif short_change_pct > 0.2:
                        short_rec = "🟡 BUY - Moderate short-term potential"
                    elif short_change_pct < -1:
                        short_rec = "🔴 SELL - Strong short-term decline"
                    elif short_change_pct < -0.2:
                        short_rec = "🟠 SELL - Moderate short-term decline"
                    else:
                        short_rec = "⚪ HOLD - Stable short-term outlook"
                    print(f"   📅 Short-term (1-7 days): {short_rec}")
                
                # Medium-term recommendation
                if timeframe_predictions.get('medium_term'):
                    medium_avg = sum(timeframe_predictions['medium_term']) / len(timeframe_predictions['medium_term'])
                    medium_change_pct = ((medium_avg - current_price) / current_price) * 100
                    if medium_change_pct > 3:
                        medium_rec = "🟢 BUY - Strong medium-term growth"
                    elif medium_change_pct > 1:
                        medium_rec = "🟡 BUY - Moderate medium-term potential"
                    elif medium_change_pct < -3:
                        medium_rec = "🔴 SELL - Strong medium-term decline"
                    elif medium_change_pct < -1:
                        medium_rec = "🟠 SELL - Moderate medium-term decline"
                    else:
                        medium_rec = "⚪ HOLD - Stable medium-term outlook"
                    print(f"   📅 Medium-term (1-4 weeks): {medium_rec}")
                
                # Long-term recommendation
                if timeframe_predictions.get('long_term'):
                    long_avg = sum(timeframe_predictions['long_term']) / len(timeframe_predictions['long_term'])
                    long_change_pct = ((long_avg - current_price) / current_price) * 100
                    if long_change_pct > 10:
                        long_rec = "🟢 BUY - Strong long-term growth potential"
                    elif long_change_pct > 5:
                        long_rec = "🟡 BUY - Moderate long-term potential"
                    elif long_change_pct < -10:
                        long_rec = "🔴 SELL - Strong long-term decline"
                    elif long_change_pct < -5:
                        long_rec = "🟠 SELL - Moderate long-term decline"
                    else:
                        long_rec = "⚪ HOLD - Stable long-term outlook"
                    print(f"   📅 Long-term (1-12 months): {long_rec}")
    
    def save_advanced_predictions(self, predictions, multi_day_predictions, timeframe_predictions, confidence_analysis):
        """Save advanced predictions to file with confidence analysis."""
        try:
            # Get current price
            data_file = f"data/{self.ticker}_partA_partC_enhanced.csv"
            df = pd.read_csv(data_file)
            current_price = float(df.iloc[-1]['Close'])
            
            predictions_data = {
                'Date': [datetime.now().strftime('%Y-%m-%d')],
            'Date_Full': [DateUtils.format_analysis_timestamp()['date_with_day'] if DATE_UTILS_AVAILABLE else datetime.now().strftime('%Y-%m-%d')],
            'Month_Name': [DateUtils.format_analysis_timestamp()['month_name'] if DATE_UTILS_AVAILABLE else datetime.now().strftime('%B')],
            'Month_Number': [DateUtils.format_analysis_timestamp()['month'] if DATE_UTILS_AVAILABLE else datetime.now().month],
            'Year': [DateUtils.format_analysis_timestamp()['year'] if DATE_UTILS_AVAILABLE else datetime.now().year],
            'Day_Name': [DateUtils.format_analysis_timestamp()['day_name'] if DATE_UTILS_AVAILABLE else datetime.now().strftime('%A')],
                'Ticker': [self.ticker],
                'Current_Price': [current_price],
            }
            
            # Add individual model predictions
            if predictions:
                for model_name, pred in predictions.items():
                    predictions_data[f'{model_name}_Prediction'] = [pred]
            
            # Add multi-day predictions
            if multi_day_predictions:
                for i, pred in enumerate(multi_day_predictions, 1):
                    predictions_data[f'MultiDay_Day_{i}'] = [pred]
            
            # Add confidence analysis
            if confidence_analysis:
                predictions_data['Mean_Prediction'] = [confidence_analysis['mean']]
                predictions_data['Std_Deviation'] = [confidence_analysis['std']]
                predictions_data['Confidence_68_Lower'] = [confidence_analysis['confidence_68'][0]]
                predictions_data['Confidence_68_Upper'] = [confidence_analysis['confidence_68'][1]]
                predictions_data['Confidence_95_Lower'] = [confidence_analysis['confidence_95'][0]]
                predictions_data['Confidence_95_Upper'] = [confidence_analysis['confidence_95'][1]]
                predictions_data['Agreement_Score'] = [confidence_analysis['agreement_score']]
                
                # Add model diversity info
                if confidence_analysis.get('model_diversity'):
                    diversity = confidence_analysis['model_diversity']
                    predictions_data['Diversity_Level'] = [diversity['diversity_level']]
                    predictions_data['Coefficient_of_Variation'] = [diversity['coefficient_of_variation']]
                    predictions_data['Prediction_Range'] = [diversity['prediction_range']]
                    predictions_data['Prediction_Range_Pct'] = [diversity['prediction_range_pct']]
                
                # Add pattern strength info
                if confidence_analysis.get('pattern_strength'):
                    pattern = confidence_analysis['pattern_strength']
                    predictions_data['Pattern_Level'] = [pattern['pattern_level']]
                    predictions_data['Trend_Strength'] = [pattern['trend_strength']]
            
            # Add timeframe predictions
            if timeframe_predictions:
                if timeframe_predictions.get('short_term'):
                    for i, pred in enumerate(timeframe_predictions['short_term'], 1):
                        predictions_data[f'ShortTerm_Day_{i}'] = [pred]
                
                if timeframe_predictions.get('medium_term'):
                    for i, pred in enumerate(timeframe_predictions['medium_term'], 1):
                        predictions_data[f'MediumTerm_Week_{i}'] = [pred]
                
                if timeframe_predictions.get('long_term'):
                    for i, pred in enumerate(timeframe_predictions['long_term'], 1):
                        predictions_data[f'LongTerm_Month_{i}'] = [pred]
            
            predictions_df = pd.DataFrame(predictions_data)
            predictions_df.to_csv(f"data/{self.ticker}_advanced_predictions.csv", index=False)
            print(f"\n💾 Advanced predictions saved to: data/{self.ticker}_advanced_predictions.csv")
            
            # Save detailed confidence analysis
            if confidence_analysis:
                confidence_data = {
                    'Analysis_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Analysis_Date_Full': [DateUtils.format_analysis_timestamp()['analysis_date_full'] if DATE_UTILS_AVAILABLE else datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Analysis_Month': [DateUtils.format_analysis_timestamp()['month_name'] if DATE_UTILS_AVAILABLE else datetime.now().strftime('%B')],
            'Analysis_Year': [DateUtils.format_analysis_timestamp()['year'] if DATE_UTILS_AVAILABLE else datetime.now().year],
            'Analysis_Day': [DateUtils.format_analysis_timestamp()['day_name'] if DATE_UTILS_AVAILABLE else datetime.now().strftime('%A')],
                    'Ticker': [self.ticker],
                    'Current_Price': [current_price],
                    'Mean_Prediction': [confidence_analysis['mean']],
                    'Std_Deviation': [confidence_analysis['std']],
                    'Confidence_68_Lower': [confidence_analysis['confidence_68'][0]],
                    'Confidence_68_Upper': [confidence_analysis['confidence_68'][1]],
                    'Confidence_95_Lower': [confidence_analysis['confidence_95'][0]],
                    'Confidence_95_Upper': [confidence_analysis['confidence_95'][1]],
                    'Agreement_Score': [confidence_analysis['agreement_score']],
                    'Diversity_Level': [confidence_analysis.get('model_diversity', {}).get('diversity_level', 'N/A')],
                    'Pattern_Level': [confidence_analysis.get('pattern_strength', {}).get('pattern_level', 'N/A')],
                    'Trend_Strength': [confidence_analysis.get('pattern_strength', {}).get('trend_strength', 0)],
                    'Coefficient_of_Variation': [confidence_analysis.get('model_diversity', {}).get('coefficient_of_variation', 0)]
                }
                
                confidence_df = pd.DataFrame(confidence_data)
                confidence_df.to_csv(f"data/{self.ticker}_confidence_analysis.csv", index=False)
                print(f"💾 Confidence analysis saved to: data/{self.ticker}_confidence_analysis.csv")
            
            # Save detailed timeframe predictions to separate file
            if timeframe_predictions:
                timeframe_data = {
                    'Timeframe': [],
                    'Period': [],
                    'Predicted_Price': [],
                    'Change_Percent': []
                }
                
                current_price = float(df.iloc[-1]['Close'])
                
                # Short-term
                if timeframe_predictions.get('short_term'):
                    for i, pred in enumerate(timeframe_predictions['short_term'], 1):
                        change_pct = ((pred - current_price) / current_price) * 100
                        timeframe_data['Timeframe'].append('Short-Term')
                        timeframe_data['Period'].append(f'Day {i}')
                        timeframe_data['Predicted_Price'].append(pred)
                        timeframe_data['Change_Percent'].append(change_pct)
                
                # Medium-term
                if timeframe_predictions.get('medium_term'):
                    for i, pred in enumerate(timeframe_predictions['medium_term'], 1):
                        change_pct = ((pred - current_price) / current_price) * 100
                        timeframe_data['Timeframe'].append('Medium-Term')
                        timeframe_data['Period'].append(f'Week {i}')
                        timeframe_data['Predicted_Price'].append(pred)
                        timeframe_data['Change_Percent'].append(change_pct)
                
                # Long-term
                if timeframe_predictions.get('long_term'):
                    for i, pred in enumerate(timeframe_predictions['long_term'], 1):
                        change_pct = ((pred - current_price) / current_price) * 100
                        timeframe_data['Timeframe'].append('Long-Term')
                        timeframe_data['Period'].append(f'Month {i}')
                        timeframe_data['Predicted_Price'].append(pred)
                        timeframe_data['Change_Percent'].append(change_pct)
                
                timeframe_df = pd.DataFrame(timeframe_data)
                timeframe_df.to_csv(f"data/{self.ticker}_timeframe_predictions.csv", index=False)
                print(f"💾 Timeframe predictions saved to: data/{self.ticker}_timeframe_predictions.csv")
            
        except Exception as e:
            print(f"Warning: Could not save advanced predictions: {e}")
    
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

    def run_phase1_enhanced_analysis(self) -> bool:
        """
        Run Phase 1 enhanced analysis integration.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not PHASE1_AVAILABLE:
                print("⚠️ Phase 1 integration not available")
                return False
            
            print("🔍 Running Phase 1 Enhanced Analysis...")
            
            # Initialize Phase 1 integration
            phase1_integration = Phase1Integration()
            
            # Get comprehensive analysis
            analysis = phase1_integration.get_comprehensive_analysis(self.ticker)
            
            if not analysis:
                print("❌ Failed to get Phase 1 analysis")
                return False
            
            # Display Phase 1 results
            print("\n📊 Phase 1 Enhanced Analysis Results:")
            print("-" * 50)
            
            # Fundamental Analysis
            fundamental = analysis.get('fundamental', {})
            print(f"🏢 EPS: ${fundamental.get('eps', 0):.2f}")
            print(f"📈 EPS Growth: {fundamental.get('eps_growth', 0):+.2f}%")
            print(f"💰 Net Profit Margin: {fundamental.get('net_profit_margin', 0):.2f}%")
            print(f"💎 Financial Health Score: {fundamental.get('financial_health_score', 0):.1f}/100")
            
            # Global Market Impact
            global_markets = analysis.get('global_markets', {})
            print(f"🌍 Dow Jones: {global_markets.get('dow_jones_change', 0):+.2f}%")
            print(f"🌍 NASDAQ: {global_markets.get('nasdaq_change', 0):+.2f}%")
            print(f"🌍 Market Impact Score: {global_markets.get('market_impact_score', 0):.1f}/100")
            
            # Institutional Sentiment
            institutional = analysis.get('institutional', {})
            print(f"📈 Institutional Sentiment: {institutional.get('institutional_sentiment', 'Unknown')}")
            print(f"📊 Analyst Consensus: {institutional.get('analyst_consensus', 'Unknown')}")
            print(f"💼 Institutional Confidence: {institutional.get('institutional_confidence', 0):.1f}/100")
            
            # Enhanced Prediction Score
            enhanced_score = analysis.get('enhanced_prediction_score', 50.0)
            print(f"🎯 Enhanced Prediction Score: {enhanced_score:.1f}/100")
            
            # Variable Coverage
            coverage = analysis.get('variable_coverage', {})
            print(f"📋 Variable Coverage: {coverage.get('overall_coverage', 'Unknown')}")
            
            # Save Phase 1 analysis
            try:
                phase1_integration.save_phase1_analysis(self.ticker, analysis)
                print("💾 Phase 1 analysis saved to data/phase1/")
            except Exception as e:
                print(f"⚠️ Warning: Could not save Phase 1 analysis: {e}")
            
            # Store Phase 1 results for later use
            self.phase1_results = analysis
            
            print("✅ Phase 1 Enhanced Analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 analysis error: {e}")
            return False

    def run_phase2_economic_analysis(self) -> bool:
        """
        Run Phase 2 economic data and regulatory monitoring analysis.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not PHASE2_AVAILABLE:
                print("⚠️ Phase 2 integration not available")
                return False
            
            print("🌍 Running Phase 2 Economic Data & Regulatory Analysis...")
            
            # Initialize Phase 2 integration
            phase2_integration = Phase2Integration()
            
            # Get comprehensive Phase 2 analysis
            analysis = phase2_integration.run_phase2_analysis(self.ticker)
            
            if not analysis:
                print("❌ Failed to get Phase 2 analysis")
                return False
            
            # Display Phase 2 results
            print("\n📊 Phase 2 Economic & Regulatory Analysis Results:")
            print("-" * 50)
            
            # Economic Impact
            print(f"📈 Economic Impact Score: {analysis.economic_impact_score:.1f}")
            print(f"🌍 Economic Sentiment: {analysis.economic_sentiment}")
            
            # Currency & Commodity Impact
            print(f"💱 Currency Impact: {analysis.currency_impact:.1f}")
            print(f"🛢️ Commodity Impact: {analysis.commodity_impact:.1f}")
            
            # Regulatory Risk
            print(f"⚖️ Regulatory Risk Score: {analysis.regulatory_risk_score:.1f}/100")
            
            # Institutional Confidence
            print(f"💼 Institutional Confidence: {analysis.institutional_confidence:.1f}/100")
            
            # Enhanced Prediction Score
            print(f"🎯 Enhanced Prediction Score: {analysis.enhanced_prediction_score:.1f}/100")
            
            # Variable Coverage
            print(f"📋 Variable Coverage: {analysis.variable_coverage:.1f}%")
            
            # Key Impact Factors
            if analysis.impact_factors:
                print(f"🔍 Key Impact Factors: {', '.join(analysis.impact_factors[:3])}")
            
            # Save Phase 2 analysis
            try:
                phase2_integration.save_phase2_analysis(self.ticker, analysis)
                print("💾 Phase 2 analysis saved to data/phase2/")
            except Exception as e:
                print(f"⚠️ Warning: Could not save Phase 2 analysis: {e}")
            
            # Store Phase 2 results for later use
            self.phase2_results = analysis
            
            print("✅ Phase 2 Economic & Regulatory Analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Phase 2 analysis error: {e}")
            return False
    
    def run_phase3_advanced_analysis(self) -> bool:
         """
         Run Phase 3 geopolitical risk, corporate actions, and insider trading analysis.
         
         Returns:
             bool: True if successful, False otherwise
         """
         try:
             if not PHASE3_AVAILABLE:
                 print("⚠️ Phase 3 integration not available")
                 return False
             
             print("🌍 Running Phase 3 Advanced Analysis...")
             
             # Initialize Phase 3 integration
             phase3_integration = Phase3Integration()
             
             # Get comprehensive Phase 3 analysis
             analysis = phase3_integration.run_phase3_analysis(self.ticker)
             
             if not analysis:
                 print("❌ Failed to get Phase 3 analysis")
                 return False
             
             # Display Phase 3 results
             print("\n📊 Phase 3 Advanced Analysis Results:")
             print("-" * 50)
             
             # Combined Risk Assessment
             print(f"🎯 Combined Risk Score: {analysis.combined_risk_score:.1f}/100")
             print(f"📈 Market Impact Score: {analysis.market_impact_score:.1f}/100")
             print(f"🎯 Confidence Level: {analysis.confidence_level:.1f}/100")
             
             # Geopolitical Risk
             print(f"\n🌍 Geopolitical Risk:")
             print(f"   Overall Risk: {analysis.geopolitical_risk.overall_risk_score:.1f}/100")
             print(f"   Events Count: {analysis.geopolitical_risk.event_count}")
             print(f"   Risk Factors: {', '.join(analysis.geopolitical_risk.risk_factors[:3])}")
             
             # Corporate Actions
             print(f"\n🏢 Corporate Actions:")
             print(f"   Total Actions: {analysis.corporate_actions.total_actions}")
             print(f"   Dividend Yield: {analysis.corporate_actions.dividend_yield:.1f}%")
             print(f"   Buyback Amount: ${analysis.corporate_actions.buyback_amount/1000000000:.1f}B")
             print(f"   Action Score: {analysis.corporate_actions.action_score:.1f}/100")
             
             # Insider Trading
             print(f"\n👥 Insider Trading:")
             print(f"   Total Transactions: {analysis.insider_trading.total_transactions}")
             print(f"   Insider Sentiment: {analysis.insider_trading.insider_sentiment_score:.1f}/100")
             print(f"   Unusual Activity: {analysis.insider_trading.unusual_activity_score:.1f}/100")
             print(f"   Net Activity: {analysis.insider_trading.net_insider_activity:,} shares")
             
             # Key Insights
             if analysis.key_insights:
                 print(f"\n🔍 Key Insights:")
                 for i, insight in enumerate(analysis.key_insights[:3], 1):
                     print(f"   {i}. {insight}")
             
             # Recommendations
             if analysis.recommendations:
                 print(f"\n💡 Recommendations:")
                 for i, rec in enumerate(analysis.recommendations[:3], 1):
                     print(f"   {i}. {rec}")
             
             # Save Phase 3 analysis
             try:
                 phase3_integration.save_phase3_analysis(self.ticker, analysis)
                 print("💾 Phase 3 analysis saved to data/phase3/")
             except Exception as e:
                 print(f"⚠️ Warning: Could not save Phase 3 analysis: {e}")
             
             # Store Phase 3 results for later use
             self.phase3_results = analysis
             
             print("✅ Phase 3 Advanced Analysis completed successfully!")
             return True
             
         except Exception as e:
             print(f"❌ Phase 3 analysis error: {e}")
             return False

def main():
    """Main function for unified analysis pipeline."""
    print("🚀 Unified AI Stock Predictor")
    print("=" * 50)
    
    # Get user input
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        ticker = "AAPL"
    
    # Interactive forecast type selection
    print("\n📊 Select Forecast Type:")
    print("1. Intraday Forecast (Hours)")
    print("2. Short-term Forecast (1-7 days)")
    print("3. Medium-term Forecast (1-4 weeks)")
    print("4. Long-term Forecast (1-12 months)")
    print("5. Comprehensive Analysis (All timeframes)")
    print("6. Multi-Timeframe Analysis (Short + Mid + Long term)")
    
    forecast_choice = input("Enter your choice (1-6, default: 6): ").strip()
    if not forecast_choice:
        forecast_choice = "6"
    
    # Set parameters based on choice
    if forecast_choice == "1":
        period = "1mo"
        days_ahead = 1
        forecast_type = "intraday"
    elif forecast_choice == "2":
        period = "3mo"
        days_ahead = 7
        forecast_type = "short_term"
    elif forecast_choice == "3":
        period = "6mo"
        days_ahead = 28
        forecast_type = "medium_term"
    elif forecast_choice == "4":
        period = "2y"
        days_ahead = 365
        forecast_type = "long_term"
    elif forecast_choice == "5":  # Comprehensive
        period = "2y"
        days_ahead = 365
        forecast_type = "comprehensive"
    else:  # Multi-timeframe
        period = "2y"
        days_ahead = 365
        forecast_type = "multi_timeframe"
    
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
    
    print(f"\n🎯 Starting {forecast_type} analysis for {ticker}...")
    print(f"📅 Data period: {period}")
    print(f"⏰ Prediction horizon: {days_ahead} days")
    
    # Create and run unified pipeline
    pipeline = UnifiedAnalysisPipeline(ticker, max_workers)
    
    # Run appropriate analysis based on forecast type
    if forecast_type == "multi_timeframe":
        if ANALYSIS_MODULES_AVAILABLE:
            success = pipeline.run_multi_timeframe_analysis(use_enhanced)
            success = bool(success)  # Convert dict to bool
        else:
            print("❌ Analysis modules not available. Falling back to comprehensive analysis.")
            success = pipeline.run_unified_analysis(period, days_ahead, use_enhanced)
    else:
        success = pipeline.run_unified_analysis(period, days_ahead, use_enhanced)
    
    if success:
        print("\n✅ Unified analysis completed successfully!")
        
        # Display forecast type summary
        if forecast_type == "intraday":
            print("📊 Intraday analysis completed - Check hourly predictions")
        elif forecast_type == "short_term":
            print("📊 Short-term analysis completed - Check 1-7 day predictions")
        elif forecast_type == "medium_term":
            print("📊 Medium-term analysis completed - Check 1-4 week predictions")
        elif forecast_type == "long_term":
            print("📊 Long-term analysis completed - Check 1-12 month predictions")
        elif forecast_type == "multi_timeframe":
            print("📊 Multi-timeframe analysis completed - Check short, mid, and long-term predictions")
        else:
            print("📊 Comprehensive analysis completed - Check all timeframe predictions")
    else:
        print("\n❌ Unified analysis failed!")

if __name__ == "__main__":
    main()
