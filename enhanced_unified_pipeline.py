#!/usr/bin/env python3
"""
Enhanced Unified Analysis Pipeline with Interactive Data Selection
Integrates Angel One API with user-configurable data parameters
"""

import sys
import os
sys.path.insert(0, '.')

from interactive_data_selector import InteractiveDataSelector
from main.unified_analysis_pipeline import UnifiedAnalysisPipeline
import pandas as pd
from datetime import datetime

class EnhancedUnifiedPipeline:
    """
    Enhanced unified pipeline with interactive data selection
    """
    
    def __init__(self):
        self.selector = InteractiveDataSelector()
        self.pipeline = None
        self.config = None
    
    def run_interactive_analysis(self, ticker):
        """
        Run analysis with interactive data selection
        """
        try:
            print(f"🚀 ENHANCED UNIFIED ANALYSIS PIPELINE")
            print(f"📊 Ticker: {ticker}")
            print("=" * 50)
            
            # Step 1: Interactive data selection
            print("\n🔧 STEP 1: DATA CONFIGURATION")
            print("-" * 30)
            self.config = self.selector.run_interactive_selection()
            
            if not self.config:
                print("❌ Configuration cancelled")
                return None
            
            # Step 2: Initialize pipeline with custom parameters
            print(f"\n🔧 STEP 2: PIPELINE INITIALIZATION")
            print("-" * 35)
            self.pipeline = UnifiedAnalysisPipeline(
                ticker=ticker,
                max_workers=4,
                period_config="recommended"
            )
            
            print(f"✅ Pipeline initialized for {ticker}")
            
            # Step 3: Load data with custom parameters
            print(f"\n🔧 STEP 3: DATA LOADING")
            print("-" * 25)
            
            # Convert interval to period format
            interval = self.config['interval']
            data_days = self.config['data_days']
            
            # Map interval to period
            if interval == 'ONE_DAY':
                if data_days <= 30:
                    period = '1mo'
                elif data_days <= 90:
                    period = '3mo'
                elif data_days <= 180:
                    period = '6mo'
                elif data_days <= 365:
                    period = '1y'
                else:
                    period = '2y'
            else:
                # For intraday data, use days
                period = f"{data_days}d"
            
            print(f"📊 Loading {data_days} days of {interval} data...")
            
            # Load data with correct interval - use enhanced Angel One service for all intervals
            from src.core.enhanced_angel_one_service import EnhancedAngelOneService
            angel_service = EnhancedAngelOneService()
            df = angel_service.get_optimal_historical_data(
                ticker.replace('.NS', ''),  # Remove .NS suffix for Angel One
                "BSE" if 'NS' in ticker else "NSE",
                interval,
                data_days
            )
            
            if df is not None and not df.empty:
                print(f"✅ Data loaded: {len(df)} records")
                print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
                print(f"💰 Latest price: ₹{df['Close'].iloc[-1]:.2f}")
            else:
                print("❌ Failed to load data")
                return None
            
            # Step 4: Prepare training data
            print(f"\n🔧 STEP 4: TRAINING DATA PREPARATION")
            print("-" * 40)
            
            training_days = self.config['training_days']
            total_records = len(df)
            
            # Calculate training records
            training_records = min(training_days, total_records)
            remaining_records = total_records - training_records
            
            print(f"📊 Total records: {total_records}")
            print(f"🎯 Training records: {training_records}")
            print(f"📊 Validation records: {remaining_records}")
            
            # Split data for training
            if remaining_records > 0:
                train_df = df.iloc[:training_records]
                val_df = df.iloc[training_records:]
                print(f"✅ Data split: {len(train_df)} training, {len(val_df)} validation")
            else:
                train_df = df
                val_df = pd.DataFrame()
                print(f"✅ Using all data for training: {len(train_df)} records")
            
            # Step 5: Run analysis
            print(f"\n🔧 STEP 5: ANALYSIS EXECUTION")
            print("-" * 30)
            
            # Set prediction horizon
            prediction_days = self.config['prediction_days']
            print(f"🔮 Prediction horizon: {prediction_days} days")
            
            # Run preprocessing on training data
            print("🔄 Running preprocessing...")
            result = self.pipeline.run_partA_preprocessing(period)
            
            if isinstance(result, dict) and result.get('load_stock_data', {}).get('success', False):
                print("✅ Preprocessing completed")
                
                # Run model training and predictions
                print("🔄 Running model training and predictions...")
                
                # Generate predictions
                predictions = self.pipeline.generate_and_display_predictions(days_ahead=prediction_days)
                
                if predictions:
                    print("✅ Predictions generated successfully")
                    
                    # Display results
                    self.display_analysis_results(predictions, prediction_days)
                    
                    return {
                        'success': True,
                        'config': self.config,
                        'data_loaded': len(df),
                        'training_records': training_records,
                        'validation_records': remaining_records,
                        'predictions': predictions
                    }
                else:
                    print("❌ Failed to generate predictions")
                    return None
            else:
                print("❌ Preprocessing failed")
                return None
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def display_analysis_results(self, predictions, prediction_days):
        """Display analysis results"""
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 30)
        print(f"🔮 Prediction Horizon: {prediction_days} days")
        print(f"📈 Data Configuration: {self.config['interval']}")
        print(f"📅 Training Data: {self.config['training_days']} days")
        print()
        
        if predictions:
            print("🎯 PREDICTIONS:")
            for i, (model, prediction) in enumerate(predictions.items(), 1):
                if isinstance(prediction, (int, float)):
                    print(f"   {i}. {model}: ₹{prediction:.2f}")
                else:
                    print(f"   {i}. {model}: {prediction}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Configuration used: {self.config}")
    
    def run_quick_analysis(self, ticker, interval='ONE_DAY', data_days=90, training_days=60, prediction_days=7):
        """
        Run quick analysis with predefined parameters
        """
        try:
            print(f"⚡ QUICK ANALYSIS - {ticker}")
            print("=" * 30)
            
            # Set quick config
            self.config = {
                'interval': interval,
                'data_days': data_days,
                'training_days': training_days,
                'prediction_days': prediction_days,
                'training_ratio': (training_days / data_days) * 100
            }
            
            print(f"📊 Quick config: {data_days} days data, {training_days} days training, {prediction_days} days prediction")
            
            # Initialize pipeline
            self.pipeline = UnifiedAnalysisPipeline(
                ticker=ticker,
                max_workers=4,
                period_config="recommended"
            )
            
            # Run analysis
            return self.run_interactive_analysis(ticker)
            
        except Exception as e:
            print(f"❌ Quick analysis failed: {e}")
            return None

def main():
    """Main function"""
    try:
        # Get ticker from user
        ticker = input("📊 Enter ticker symbol (e.g., RELIANCE, AAPL): ").strip().upper()
        
        if not ticker:
            print("❌ No ticker provided")
            return
        
        # Ask for analysis type
        print(f"\n🔧 SELECT ANALYSIS TYPE:")
        print("1. Interactive (Full customization)")
        print("2. Quick (Predefined parameters)")
        
        choice = input("🔢 Enter choice (1-2): ").strip()
        
        pipeline = EnhancedUnifiedPipeline()
        
        if choice == '1':
            # Interactive analysis
            result = pipeline.run_interactive_analysis(ticker)
        elif choice == '2':
            # Quick analysis
            result = pipeline.run_quick_analysis(ticker)
        else:
            print("❌ Invalid choice")
            return
        
        if result and result.get('success'):
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📊 Results: {result}")
        else:
            print(f"\n❌ Analysis failed")
            
    except KeyboardInterrupt:
        print("\n\n❌ Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
