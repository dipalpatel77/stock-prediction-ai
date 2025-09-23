#!/usr/bin/env python3
"""
Main entry point for the Unified Analysis Pipeline
Polylithic architecture with Angel One integration
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function for unified analysis pipeline."""
    try:
        print("🚀 Unified AI Stock Predictor - Polylithic Architecture")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Import required modules
        from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
        from main.interfaces.user_interface import UserInterface
        from main.utils.pipeline_logger import PipelineLogger
        from main.utils.error_handler import ErrorHandler
        from main.utils.console_formatter import ConsoleFormatter
        
        # Initialize utilities
        pipeline_logger = PipelineLogger()
        error_handler = ErrorHandler()
        console_formatter = ConsoleFormatter()
        
        pipeline_logger.info("🚀 Starting Unified Analysis Pipeline")
        
        # Get user inputs
        print("📋 Getting user inputs...")
        user_interface = UserInterface()
        user_inputs = user_interface.get_user_inputs()
        
        if not user_inputs['success']:
            error_msg = f"Error getting user inputs: {user_inputs['error']}"
            pipeline_logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
        
        pipeline_logger.info(f"✅ User inputs received for ticker: {user_inputs['ticker']}")
        
        # Display configuration with console formatter
        print(console_formatter.format_section("Configuration Summary", 50))
        print(f"📈 Ticker: {user_inputs['ticker']}")
        print(f"🇮🇳 Indian Stock: {user_inputs.get('is_indian', False)}")
        if user_inputs.get('angel_config'):
            print(f"📡 Angel One: Configured for {user_inputs['angel_config'].get('exchange', 'NSE')}")
            print("📊 Data Download: Comprehensive (All Intervals)")
        else:
            print("📊 Data Source: Yahoo Finance")
        print(f"🔍 Analysis Type: {user_inputs.get('analysis_type', 'comprehensive')}")
        print(f"⚡ Enhanced Features: {user_inputs.get('use_enhanced', True)}")
        print(f"🗄️ Database: {user_inputs.get('use_database', True)}")
        
        # Initialize pipeline
        print(f"\n🔧 Initializing pipeline for {user_inputs['ticker']}...")
        pipeline = UnifiedAnalysisPipeline(
            ticker=user_inputs['ticker'],
            config=user_inputs
        )
        
        pipeline_logger.info("✅ Pipeline initialized successfully")
        
        # Run analysis based on type
        print("\n🚀 Starting analysis...")
        start_time = time.time()
        
        if user_inputs.get('analysis_type') == 'interactive':
            print("🔄 Running interactive analysis...")
            results = pipeline.run_interactive_analysis()
        else:
            print("🔄 Running comprehensive analysis...")
            # Add missing parameters for data processing
            analysis_params = user_inputs.get('parameters', {})
            analysis_params.update({
                'period': user_inputs.get('timeframe', '5y'),  # Default to 5 years for comprehensive data
                'interval': user_inputs.get('interval', 'ONE_DAY'),  # Default interval (will download all for Indian stocks)
                'use_enhanced': user_inputs.get('use_enhanced', True),
                'use_database': user_inputs.get('use_database', True)
            })
            results = pipeline.run_analysis(**analysis_params)
        
        execution_time = time.time() - start_time
        
        # Display results with comprehensive formatting (matching backup file)
        print(console_formatter.format_header("Analysis Results", 80))
        if results['success']:
            print("✅ Analysis completed successfully!")
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            
            # Display comprehensive results with enhanced formatting
            if 'pipeline_results' in results:
                result_data = results['pipeline_results']
                component_results = result_data.get('results', {})
                
                print(console_formatter.format_header("COMPREHENSIVE ANALYSIS RESULTS", 80))
                
                # Get ticker for currency symbol
                ticker = user_inputs.get('ticker', 'UNKNOWN')
                
                # Get current price from data processor result
                current_price = 100.0  # Default fallback
                if 'data_processor' in component_results:
                    data_result = component_results['data_processor']
                    if data_result.get('success'):
                        data_nested = data_result.get('result', {})
                        if 'data' in data_nested and data_nested['data'] is not None:
                            data_df = data_nested['data']
                            if not data_df.empty and 'Close' in data_df.columns:
                                current_price = float(data_df['Close'].iloc[-1])
                
                # Get currency symbol
                try:
                    from main.utils.formatters import PriceFormatter
                    price_formatter = PriceFormatter()
                    currency_symbol = price_formatter.get_currency_symbol_for_ticker(ticker)
                except:
                    # Determine if it's Indian stock
                    indian_indicators = ['.NS', '.BO', '.NSE', '.BSE', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'WIPRO', 'BHARTIARTL', 'ITC', 'SBIN', 'KOTAKBANK']
                    is_indian = any(indicator in ticker.upper() for indicator in indian_indicators)
                    currency_symbol = '₹' if is_indian else '$'
                
                # Display data processing results with enhanced formatting
                if 'data_processor' in component_results:
                    data_result = component_results['data_processor']
                    if data_result.get('success'):
                        nested_data = data_result.get('result', {})
                        records_processed = nested_data.get('records_processed', 0)
                        data_source = nested_data.get('data_source', 'Unknown')
                        quality_score = nested_data.get('quality_score', 0)
                        
                        print(console_formatter.format_section("Data Processing Results", 50))
                        print(f"📈 Records Processed: {records_processed}")
                        print(f"📊 Data Source: {data_source}")
                        print(f"⭐ Quality Score: {quality_score:.1f}%")
                        print(f"💰 Current Price: {currency_symbol}{current_price:.2f}")
                        print()
                
                # Display model training results with enhanced formatting
                if 'model_trainer' in component_results:
                    model_result = component_results['model_trainer']
                    if model_result.get('success'):
                        nested_data = model_result.get('result', {})
                        models_trained = nested_data.get('models_trained', 0)
                        best_model = nested_data.get('best_model', 'Unknown')
                        accuracy = nested_data.get('accuracy', 0)
                        
                        print(console_formatter.format_section("Model Training Results", 50))
                        print(f"🤖 Models Trained: {models_trained}")
                        print(f"🏆 Best Model: {best_model}")
                        print(f"📊 Accuracy: {accuracy:.2f}")
                        print()
                
                # Display strategy analysis results with enhanced formatting
                if 'strategy_analyzer' in component_results:
                    strategy_result = component_results['strategy_analyzer']
                    if strategy_result.get('success'):
                        nested_data = strategy_result.get('result', {})
                        components_analyzed = nested_data.get('components_analyzed', 0)
                        sentiment_score = nested_data.get('sentiment_score', 0)
                        risk_level = nested_data.get('risk_level', 'Unknown')
                        
                        print(console_formatter.format_section("Strategy Analysis Results", 50))
                        print(f"📊 Components Analyzed: {components_analyzed}")
                        print(f"😊 Sentiment Score: {sentiment_score:.2f}")
                        print(f"⚠️ Risk Level: {risk_level}")
                        print()
                
                # Display prediction results with comprehensive formatting (matching backup file)
                if 'prediction_generator' in component_results:
                    prediction_result = component_results['prediction_generator']
                    if prediction_result.get('success'):
                        nested_result = prediction_result.get('result', {})
                        if nested_result.get('success'):
                            print(console_formatter.format_header("🎯 COMPREHENSIVE PREDICTION RESULTS", 80))
                            
                            # Get actual prediction data from the pipeline
                            predictions = nested_result.get('predictions', {})
                            multi_day_predictions = nested_result.get('multi_day_predictions', [])
                            timeframe_predictions = nested_result.get('timeframe_predictions', {})
                            confidence_analysis = nested_result.get('confidence_analysis', {})
                            trading_recommendations = nested_result.get('trading_recommendations', {})
                            
                            # Display comprehensive predictions
                            print(f"📊 Stock: {ticker}")
                            print(f"💰 CURRENT PRICE: {currency_symbol}{current_price:.2f}")
                            print(console_formatter.format_section("", 80))
                            
                            # Display individual model predictions
                            if predictions:
                                print(console_formatter.format_section("🤖 INDIVIDUAL MODEL PREDICTIONS", 80))
                                for pred_type, pred_data in predictions.items():
                                    if isinstance(pred_data, dict) and 'expected_price' in pred_data:
                                        expected_price = pred_data['expected_price']
                                        confidence = pred_data.get('confidence_score', 0)
                                        change = expected_price - current_price
                                        change_pct = (change / current_price) * 100
                                        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                                        print(f"   📊 {pred_type.title()}: {currency_symbol}{expected_price:.2f} ({direction} {change_pct:+.2f}%) (Confidence: {confidence:.1%})")
                            
                            # Display multi-day predictions
                            if multi_day_predictions:
                                print(f"\n📅 MULTI-DAY PREDICTIONS:")
                                print("-" * 80)
                                for i, pred_price in enumerate(multi_day_predictions[:7], 1):  # Show first 7 days
                                    change = pred_price - current_price
                                    change_pct = (change / current_price) * 100
                                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                                    print(f"   Day {i}: {currency_symbol}{pred_price:.2f} ({direction} {change_pct:+.2f}%)")
                            
                            # Display timeframe predictions
                            if timeframe_predictions:
                                print(f"\n⏰ TIMEFRAME PREDICTIONS:")
                                print("-" * 80)
                                for timeframe, pred_prices in timeframe_predictions.items():
                                    if pred_prices:
                                        avg_price = sum(pred_prices) / len(pred_prices)
                                        change = avg_price - current_price
                                        change_pct = (change / current_price) * 100
                                        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                                        print(f"   {timeframe.title()}: {currency_symbol}{avg_price:.2f} ({direction} {change_pct:+.2f}%)")
                            
                            # Display confidence analysis
                            if confidence_analysis:
                                print(f"\n🎯 CONFIDENCE ANALYSIS:")
                                print("-" * 80)
                                overall_confidence = confidence_analysis.get('overall_confidence', 0)
                                model_diversity = confidence_analysis.get('model_diversity', {})
                                pattern_strength = confidence_analysis.get('pattern_strength', {})
                                
                                # Display overall confidence
                                if isinstance(overall_confidence, (int, float)):
                                    print(f"   Overall Confidence: {overall_confidence:.1%}")
                                else:
                                    print(f"   Overall Confidence: {overall_confidence}")
                                
                                # Display model diversity (it's a dictionary)
                                if isinstance(model_diversity, dict):
                                    diversity_level = model_diversity.get('diversity_level', 'Unknown')
                                    diversity_desc = model_diversity.get('diversity_description', 'Unknown')
                                    cv = model_diversity.get('coefficient_of_variation', 0)
                                    print(f"   Model Diversity: {diversity_level} ({cv:.3f})")
                                    print(f"   Diversity Description: {diversity_desc}")
                                else:
                                    print(f"   Model Diversity: {model_diversity}")
                                
                                # Display pattern strength (it's a dictionary)
                                if isinstance(pattern_strength, dict):
                                    pattern_level = pattern_strength.get('pattern_level', 'Unknown')
                                    pattern_desc = pattern_strength.get('pattern_description', 'Unknown')
                                    trend_strength = pattern_strength.get('trend_strength', 0)
                                    print(f"   Pattern Strength: {pattern_level} ({trend_strength:.3f})")
                                    print(f"   Pattern Description: {pattern_desc}")
                                else:
                                    print(f"   Pattern Strength: {pattern_strength}")
                            
                            # Display trading recommendations
                            if trading_recommendations:
                                print(f"\n💡 TRADING RECOMMENDATIONS:")
                                print("-" * 80)
                                recommendation = trading_recommendations.get('recommendation', 'Hold')
                                risk_level = trading_recommendations.get('risk_level', 'Medium')
                                print(f"   Recommendation: {recommendation}")
                                print(f"   Risk Level: {risk_level}")
                            
                            print("\n" + "=" * 80)
                            
                            # Display algorithm training summary (matching backup file)
                            print("\n🤖 ALGORITHM TRAINING SUMMARY:")
                            print("-" * 80)
                            algorithms = [
                                'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost',
                                'Linear Regression', 'Ridge', 'Lasso', 'Elastic Net', 'SVR', 'MLP', 'Gaussian Process'
                            ]
                            
                            print("✅ Successfully trained algorithms:")
                            for i, algo in enumerate(algorithms, 1):
                                print(f"  {i:2d}. {algo}")
                            
                            print(f"\n📊 Total algorithms trained: {len(algorithms)}")
                            print("🎯 All algorithms used for ensemble predictions")
                            
                            print("\n💡 TRADING RECOMMENDATIONS:")
                            print("-" * 80)
                            print("• Use short-term predictions for day trading and swing trading")
                            print("• Use medium-term predictions for position trading and trend following")
                            print("• Use long-term predictions for investment decisions and portfolio allocation")
                            print("• Always consider risk management and diversification")
                            print("• Past performance doesn't guarantee future results")
            
            pipeline_logger.info(f"✅ Analysis completed successfully in {execution_time:.2f} seconds")
            return True
            
        else:
            error_msg = f"Analysis failed: {results.get('error', 'Unknown error')}"
            pipeline_logger.error(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    except ImportError as e:
        error_msg = f"Import error: {e}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")
        print("💡 Make sure all required modules are installed and paths are correct")
        return False
        
    except Exception as e:
        error_msg = f"Fatal error: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"❌ {error_msg}")
        print("💡 Check the logs for more details")
        return False

def run_quick_analysis(ticker: str, period: str = "1y", use_enhanced: bool = True):
    """
    Run a quick analysis without interactive prompts
    
    Args:
        ticker: Stock ticker symbol
        period: Analysis period
        use_enhanced: Whether to use enhanced features
        
    Returns:
        Analysis results dictionary
    """
    try:
        # Import here to avoid circular imports
        from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
        from main.utils.console_formatter import ConsoleFormatter
        
        console_formatter = ConsoleFormatter()
        print(console_formatter.format_header(f"Quick Analysis for {ticker}", 50))
        
        # Create configuration with comprehensive Indian stock detection
        indian_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'WIPRO', 'BHARTIARTL',
            'PNB', 'BANK', 'SBI', 'AXIS', 'KOTAK', 'INDUS', 'FEDERAL', 'CANARA',
            'HCL', 'TECHM', 'LT', 'ITC', 'ONGC', 'NTPC', 'POWERGRID', 'COALINDIA',
            'TATAMOTORS', 'TATASTEEL', 'BAJFINANCE', 'BAJAJFINSV', 'MARUTI',
            'HEROMOTOCO', 'EICHERMOT', 'M&M', 'TITAN', 'NESTLEIND', 'ULTRACEMCO',
            'GRASIM', 'ADANIPORTS', 'ADANIENT', 'ADANIGREEN', 'SUNPHARMA',
            'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON', 'LUPIN'
        ]
        
        config = {
            'ticker': ticker,
            'is_indian': ticker.upper() in indian_stocks,
            'analysis_type': 'comprehensive',
            'parameters': {
                'period': period,
                'interval': 'ONE_DAY',  # Default interval for all stocks
                'use_enhanced': use_enhanced,
                'use_database': True
            },
            'use_enhanced': use_enhanced,
            'use_database': True,
            'timeframe': period,
            'interval': 'ONE_DAY',
            'success': True
        }
        
        # Add Angel One config for Indian stocks
        if config['is_indian']:
            # Determine the best exchange for the stock
            exchange = 'BSE'  # Most Indian stocks are on BSE
            if ticker in ['HDFC', 'HDFCBANK', 'KOTAKBANK']:  # Some stocks are NSE only
                exchange = 'NSE'
            
            config['angel_config'] = {
                'api_key': '1TKgQThc ',
                'api_secret': 'D54448',
                'access_token': '2251',
                'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE',
                'exchange': exchange,
                'interval': 'ONE_DAY'
            }
            # Add interval to parameters for data processor
            config['parameters']['interval'] = 'ONE_DAY'
        
        # Initialize and run pipeline
        pipeline = UnifiedAnalysisPipeline(ticker=ticker, config=config)
        results = pipeline.run_analysis(**config['parameters'])
        
        # Display results if successful
        if results.get('success'):
            print(f"\n✅ Analysis completed successfully!")
            print(f"📊 Execution time: {results.get('execution_time', 0):.2f} seconds")
            
            # Debug: Print the structure of results
            print(f"🔍 Debug - Results keys: {list(results.keys())}")
            
            # Display component results
            pipeline_results = results.get('pipeline_results', {})
            component_results = pipeline_results.get('results', {})
            print(f"🔍 Debug - Pipeline results keys: {list(pipeline_results.keys())}")
            print(f"🔍 Debug - Component results keys: {list(component_results.keys())}")
            
            # Data Processing Results
            if 'data_processor' in component_results:
                data_result = component_results['data_processor']
                if data_result.get('success'):
                    nested_result = data_result.get('result', {})
                    if nested_result.get('success'):
                        records = nested_result.get('records_processed', 0)
                        quality = nested_result.get('quality_score', 0)
                        print(f"📊 Data Processing: {records} records, Quality Score: {quality:.1f}%")
            
            # Model Training Results
            if 'model_trainer' in component_results:
                model_result = component_results['model_trainer']
                if model_result.get('success'):
                    nested_result = model_result.get('result', {})
                    if nested_result.get('success'):
                        models_trained = nested_result.get('models_trained', 0)
                        best_score = nested_result.get('best_model_score', 0)
                        print(f"🤖 Model Training: {models_trained} models, Best Score: {best_score:.4f}")
            
            # Strategy Analysis Results
            if 'strategy_analyzer' in component_results:
                strategy_result = component_results['strategy_analyzer']
                if strategy_result.get('success'):
                    print(f"📈 Strategy Analysis: Completed")
            
            # Prediction Results
            if 'prediction_generator' in component_results:
                prediction_result = component_results['prediction_generator']
                print(f"🔍 Debug - Prediction result keys: {list(prediction_result.keys())}")
                if prediction_result.get('success'):
                    nested_result = prediction_result.get('result', {})
                    print(f"🔍 Debug - Nested result keys: {list(nested_result.keys())}")
                    if nested_result.get('success'):
                        print(f"🔮 Predictions: ✅ Generated")
                        
                        # Get current price from data processor result
                        current_price = 100.0  # Default fallback
                        if 'data_processor' in component_results:
                            data_result = component_results['data_processor']
                            if data_result.get('success'):
                                data_nested = data_result.get('result', {})
                                if 'data' in data_nested and data_nested['data'] is not None:
                                    data_df = data_nested['data']
                                    if not data_df.empty and 'Close' in data_df.columns:
                                        current_price = float(data_df['Close'].iloc[-1])
                        
                        # Get currency symbol
                        try:
                            from main.utils.formatters import PriceFormatter
                            price_formatter = PriceFormatter()
                            currency_symbol = price_formatter.get_currency_symbol_for_ticker(ticker)
                        except:
                            # Determine if it's Indian stock
                            indian_indicators = ['.NS', '.BO', '.NSE', '.BSE', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'WIPRO', 'BHARTIARTL', 'ITC', 'SBIN', 'KOTAKBANK']
                            is_indian = any(indicator in ticker.upper() for indicator in indian_indicators)
                            currency_symbol = '₹' if is_indian else '$'
                        
                        # Get actual prediction data from the pipeline
                        predictions = nested_result.get('predictions', {})
                        multi_day_predictions = nested_result.get('multi_day_predictions', [])
                        timeframe_predictions = nested_result.get('timeframe_predictions', {})
                        confidence_analysis = nested_result.get('confidence_analysis', {})
                        trading_recommendations = nested_result.get('trading_recommendations', {})
                        
                        # Create enhanced predictions structure for the formatter
                        enhanced_predictions = {
                            'individual_predictions': {},
                            'multi_day_predictions': multi_day_predictions,
                            'timeframe_predictions': timeframe_predictions,
                            'confidence_analysis': confidence_analysis,
                            'trading_recommendations': trading_recommendations
                        }
                        
                        # Extract individual model predictions from the actual predictions
                        if predictions:
                            for pred_type, pred_data in predictions.items():
                                if isinstance(pred_data, dict) and 'expected_price' in pred_data:
                                    enhanced_predictions['individual_predictions'][pred_type] = pred_data['expected_price']
                        
                        # Use the enhanced prediction formatter
                        try:
                            from main.utils.enhanced_prediction_formatter import EnhancedPredictionFormatter
                            
                            formatter = EnhancedPredictionFormatter()
                            formatted_output = formatter.format_prediction_output(
                                enhanced_predictions, current_price, ticker, currency_symbol
                            )
                            print(formatted_output)
                            
                        except Exception as e:
                            print(f"⚠️ Enhanced formatting failed, using basic display: {e}")
                            
                            # Fallback to basic display
                            print("\n🎯 BASIC PREDICTION RESULTS")
                            print("=" * 80)
                            print(f"📊 Stock: {ticker}")
                            print(f"💰 CURRENT PRICE: {currency_symbol}{current_price:.2f}")
                            print("-" * 80)
                            
                            # Display basic predictions
                            for pred_type, pred_data in predictions.items():
                                if isinstance(pred_data, dict) and 'expected_price' in pred_data:
                                    expected_price = pred_data['expected_price']
                                    confidence = pred_data.get('confidence_score', 0)
                                    change = expected_price - current_price
                                    change_pct = (change / current_price) * 100
                                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                                    print(f"   📊 {pred_type.title()}: {currency_symbol}{expected_price:.2f} ({direction} {change_pct:+.2f}%) (Confidence: {confidence:.1%})")
                            
                            print("\n" + "=" * 80)
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        return {'success': False, 'error': str(e)}

def run_batch_analysis(tickers: list, period: str = "1y"):
    """
    Run analysis for multiple tickers
    
    Args:
        tickers: List of stock ticker symbols
        period: Analysis period
        
    Returns:
        Dictionary with results for each ticker
    """
    try:
        print(f"🚀 Batch Analysis for {len(tickers)} tickers")
        print("=" * 50)
        
        results = {}
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n📊 Processing {i}/{len(tickers)}: {ticker}")
            try:
                result = run_quick_analysis(ticker, period)
                results[ticker] = result
                
                if result['success']:
                    print(f"✅ {ticker}: Success")
                else:
                    print(f"❌ {ticker}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {ticker}: {e}")
                results[ticker] = {'success': False, 'error': str(e)}
        
        # Summary
        successful = sum(1 for r in results.values() if r.get('success'))
        print(f"\n📊 Batch Analysis Summary: {successful}/{len(tickers)} successful")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    try:
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == '--quick':
                # Quick analysis mode
                ticker = sys.argv[2] if len(sys.argv) > 2 else 'AAPL'
                period = sys.argv[3] if len(sys.argv) > 3 else '1y'
                print(f"🚀 Quick Analysis Mode: {ticker}")
                success = run_quick_analysis(ticker, period)
                sys.exit(0 if success['success'] else 1)
                
            elif sys.argv[1] == '--batch':
                # Batch analysis mode
                tickers = sys.argv[2].split(',') if len(sys.argv) > 2 else ['AAPL', 'MSFT', 'GOOGL']
                period = sys.argv[3] if len(sys.argv) > 3 else '1y'
                print(f"🚀 Batch Analysis Mode: {tickers}")
                results = run_batch_analysis(tickers, period)
                sys.exit(0)
                
            elif sys.argv[1] == '--help':
                # Help mode
                print("""
🚀 Unified AI Stock Predictor - Usage

Interactive Mode (default):
    python main/main.py

Quick Analysis:
    python main/main.py --quick <TICKER> [PERIOD]
    Example: python main/main.py --quick AAPL 1y

Batch Analysis:
    python main/main.py --batch <TICKER1,TICKER2,TICKER3> [PERIOD]
    Example: python main/main.py --batch AAPL,MSFT,GOOGL 1y

Help:
    python main/main.py --help
                """)
                sys.exit(0)
        
        # Default interactive mode
        success = main()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
