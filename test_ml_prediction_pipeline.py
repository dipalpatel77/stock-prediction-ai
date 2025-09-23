#!/usr/bin/env python3
"""
ML Prediction Pipeline Test
Tests data retrieval from database and applies ML algorithms for predictions
"""

import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_ml_prediction_pipeline():
    """Test ML prediction pipeline with database data"""
    print("🧪 ML Prediction Pipeline Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import ML components
        from main.pipeline.data_processor import DataProcessor
        from main.pipeline.model_trainer import ModelTrainer
        from main.pipeline.prediction_generator import PredictionGenerator
        
        # Test database connection and data retrieval
        print("🔍 Testing database connection and data retrieval...")
        db_path = "stock_data.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check available data
        cursor.execute('''
            SELECT interval_type, COUNT(*) as record_count, 
                   MIN(datetime) as earliest_date, 
                   MAX(datetime) as latest_date
            FROM angel_one_stock_data 
            WHERE ticker = 'RELIANCE'
            GROUP BY interval_type
            ORDER BY record_count DESC
        ''')
        
        available_data = cursor.fetchall()
        print(f"📊 Available data in database:")
        for interval, count, earliest, latest in available_data:
            print(f"   {interval}: {count} records ({earliest} to {latest})")
        
        if not available_data:
            print("❌ No data found in database!")
            return False
        
        # Use the interval with most data for ML training
        best_interval = available_data[0][0]
        print(f"✅ Using {best_interval} data for ML training ({available_data[0][1]} records)")
        
        # Retrieve data from database
        print(f"\n📥 Retrieving {best_interval} data from database...")
        cursor.execute('''
            SELECT datetime, open_price, high_price, low_price, close_price, volume
            FROM angel_one_stock_data 
            WHERE ticker = 'RELIANCE' AND interval_type = ?
            ORDER BY datetime
        ''', (best_interval,))
        
        db_data = cursor.fetchall()
        print(f"✅ Retrieved {len(db_data)} records from database")
        
        # Convert to DataFrame
        df = pd.DataFrame(db_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"📊 Data shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        
        # Show sample data
        print(f"📈 Sample data (first 3 records):")
        for i, (idx, row) in enumerate(df.head(3).iterrows()):
            print(f"   {idx}: Open={row['open']:.2f}, High={row['high']:.2f}, Low={row['low']:.2f}, Close={row['close']:.2f}, Volume={row['volume']:.0f}")
        
        conn.close()
        
        # Initialize ML components
        print(f"\n🤖 Initializing ML components...")
        
        # Data Processor
        print("   📊 Initializing Data Processor...")
        data_processor = DataProcessor(ticker="RELIANCE")
        
        # Process data with technical indicators
        print("   🔄 Processing data with technical indicators...")
        processed_data = data_processor.execute(data=df, period="1mo", interval="ONE_MINUTE")
        
        if not processed_data.get('success', False):
            print(f"❌ Data processing failed: {processed_data.get('error', 'Unknown error')}")
            return False
        
        # Get the processed data from the response
        if 'enhanced_data' in processed_data:
            processed_df = processed_data['enhanced_data']
        elif 'cleaned_data' in processed_data:
            processed_df = processed_data['cleaned_data']
        elif 'raw_data' in processed_data:
            processed_df = processed_data['raw_data']
        else:
            print(f"❌ No processed data found in response")
            print(f"Available keys: {list(processed_data.keys())}")
            return False
        
        print(f"✅ Data processed successfully: {processed_df.shape}")
        
        # Show processed data features
        print(f"📊 Processed data columns: {list(processed_df.columns)}")
        print(f"📈 Sample processed data (last 3 records):")
        for i, (idx, row) in enumerate(processed_df.tail(3).iterrows()):
            close_val = row.get('Close', row.get('close', 'N/A'))
            volume_val = row.get('Volume', row.get('volume', 'N/A'))
            print(f"   {idx}: Close={close_val}, Volume={volume_val}")
            if 'SMA_20' in row:
                print(f"      SMA_20={row['SMA_20']:.2f}, RSI={row.get('RSI', 'N/A')}")
        
        # Model Trainer
        print(f"\n🧠 Initializing Model Trainer...")
        model_trainer = ModelTrainer(ticker="RELIANCE")
        
        # Train models
        print("   🔄 Training ML models...")
        training_results = model_trainer.execute(data=processed_df)
        
        if not training_results.get('success', False):
            print(f"❌ Model training failed: {training_results.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Model training successful!")
        print(f"📊 Training results: {training_results.get('summary', {})}")
        
        # Show model performance
        if 'model_performance' in training_results:
            perf = training_results['model_performance']
            print(f"🎯 Model Performance:")
            for model_name, metrics in perf.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    print(f"   {model_name}: Accuracy={metrics['accuracy']:.4f}, RMSE={metrics.get('rmse', 'N/A')}")
        
        # Prediction Generator
        print(f"\n🔮 Initializing Prediction Generator...")
        prediction_generator = PredictionGenerator(ticker="RELIANCE")
        
        # Generate predictions
        print("   🔄 Generating predictions...")
        prediction_results = prediction_generator.execute(
            data=processed_df, 
            models=training_results.get('models', {})
        )
        
        if not prediction_results.get('success', False):
            print(f"❌ Prediction generation failed: {prediction_results.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Predictions generated successfully!")
        
        # Show prediction results
        if 'predictions' in prediction_results:
            predictions = prediction_results['predictions']
            print(f"🔮 Prediction Results:")
            
            # Short-term predictions
            if 'short_term' in predictions:
                st_pred = predictions['short_term']
                print(f"   📈 Short-term (1-5 days):")
                for i, pred in enumerate(st_pred.get('predictions', [])[:3]):
                    print(f"      Day {i+1}: {pred:.2f}")
            
            # Mid-term predictions
            if 'mid_term' in predictions:
                mt_pred = predictions['mid_term']
                print(f"   📊 Mid-term (1-4 weeks):")
                for i, pred in enumerate(mt_pred.get('predictions', [])[:3]):
                    print(f"      Week {i+1}: {pred:.2f}")
            
            # Long-term predictions
            if 'long_term' in predictions:
                lt_pred = predictions['long_term']
                print(f"   📅 Long-term (1-12 months):")
                for i, pred in enumerate(lt_pred.get('predictions', [])[:3]):
                    print(f"      Month {i+1}: {pred:.2f}")
        
        # Show confidence analysis
        if 'confidence_analysis' in prediction_results:
            confidence = prediction_results['confidence_analysis']
            print(f"🎯 Confidence Analysis:")
            print(f"   Overall Confidence: {confidence.get('overall_confidence', 'N/A')}")
            print(f"   Model Agreement: {confidence.get('model_agreement', 'N/A')}")
            print(f"   Data Quality: {confidence.get('data_quality', 'N/A')}")
        
        return {
            'success': True,
            'data_retrieved': len(db_data),
            'data_processed': processed_df.shape[0],
            'models_trained': len(training_results.get('models', {})),
            'predictions_generated': len(prediction_results.get('predictions', {})),
            'training_results': training_results,
            'prediction_results': prediction_results
        }
        
    except Exception as e:
        print(f"❌ ML prediction pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_ml_report(results):
    """Generate ML prediction report"""
    print("\n" + "=" * 80)
    print("📊 ML PREDICTION PIPELINE TEST REPORT")
    print("=" * 80)
    
    if not results or not results.get('success', False):
        print("❌ No successful results to report")
        return
    
    print(f"\n📈 ML Pipeline Results:")
    print("-" * 30)
    print(f"   Data Retrieved: {results.get('data_retrieved', 0)} records")
    print(f"   Data Processed: {results.get('data_processed', 0)} records")
    print(f"   Models Trained: {results.get('models_trained', 0)} models")
    print(f"   Predictions Generated: {results.get('predictions_generated', 0)} prediction types")
    
    # Training results
    training_results = results.get('training_results', {})
    if training_results:
        print(f"\n🧠 Model Training Results:")
        print("-" * 25)
        
        if 'model_performance' in training_results:
            perf = training_results['model_performance']
            for model_name, metrics in perf.items():
                if isinstance(metrics, dict):
                    accuracy = metrics.get('accuracy', 'N/A')
                    rmse = metrics.get('rmse', 'N/A')
                    print(f"   {model_name}: Accuracy={accuracy}, RMSE={rmse}")
        
        if 'best_model' in training_results:
            print(f"   Best Model: {training_results['best_model']}")
    
    # Prediction results
    prediction_results = results.get('prediction_results', {})
    if prediction_results:
        print(f"\n🔮 Prediction Results:")
        print("-" * 20)
        
        predictions = prediction_results.get('predictions', {})
        for pred_type, pred_data in predictions.items():
            if isinstance(pred_data, dict) and 'predictions' in pred_data:
                preds = pred_data['predictions']
                if isinstance(preds, list) and len(preds) > 0:
                    print(f"   {pred_type}: {len(preds)} predictions")
                    print(f"      Latest: {preds[-1]:.2f}")
    
    # Save detailed report
    report_file = "docs/ML_PREDICTION_PIPELINE_REPORT.md"
    os.makedirs("docs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# ML Prediction Pipeline Test Report\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Data Retrieved:** {results.get('data_retrieved', 0)} records\n")
        f.write(f"**Data Processed:** {results.get('data_processed', 0)} records\n")
        f.write(f"**Models Trained:** {results.get('models_trained', 0)} models\n")
        f.write(f"**Predictions Generated:** {results.get('predictions_generated', 0)} prediction types\n\n")
        
        f.write("## Training Results\n\n")
        if training_results:
            f.write(f"**Best Model:** {training_results.get('best_model', 'N/A')}\n\n")
            
            if 'model_performance' in training_results:
                f.write("### Model Performance\n\n")
                perf = training_results['model_performance']
                for model_name, metrics in perf.items():
                    f.write(f"#### {model_name}\n\n")
                    f.write(f"- **Accuracy**: {metrics.get('accuracy', 'N/A')}\n")
                    f.write(f"- **RMSE**: {metrics.get('rmse', 'N/A')}\n")
                    f.write(f"- **R² Score**: {metrics.get('r2_score', 'N/A')}\n\n")
        
        f.write("## Prediction Results\n\n")
        if prediction_results:
            predictions = prediction_results.get('predictions', {})
            for pred_type, pred_data in predictions.items():
                f.write(f"### {pred_type.title()} Predictions\n\n")
                if isinstance(pred_data, dict) and 'predictions' in pred_data:
                    preds = pred_data['predictions']
                    if isinstance(preds, list):
                        f.write(f"**Number of Predictions:** {len(preds)}\n\n")
                        f.write("**Predictions:**\n")
                        for i, pred in enumerate(preds):
                            f.write(f"- Day {i+1}: {pred:.2f}\n")
                        f.write("\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main test function"""
    try:
        print("🧪 ML Prediction Pipeline Test")
        print("=" * 50)
        
        # Run the ML prediction pipeline test
        results = test_ml_prediction_pipeline()
        
        if results:
            # Generate comprehensive report
            generate_ml_report(results)
            
            # Calculate overall success
            overall_success = results.get('success', False)
            
            print(f"\n🎯 Overall Result: {'✅ ML PREDICTION PIPELINE SUCCESS' if overall_success else '❌ ML PREDICTION PIPELINE FAILED'}")
            
            if overall_success:
                print("\n🎉 ML prediction pipeline test successful!")
                print("✅ Data retrieved from database")
                print("✅ Data processed with technical indicators")
                print("✅ ML models trained successfully")
                print("✅ Predictions generated")
                print("✅ Comprehensive report generated")
                print("\n🚀 ML prediction pipeline is fully functional!")
            else:
                print("\n❌ ML prediction pipeline test failed!")
                print("❌ Check database data, ML components, and prediction generation")
                print("❌ Review detailed report for specific issues")
            
            return overall_success
        else:
            print("❌ Test failed")
            return False
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
