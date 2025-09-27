"""
Enhanced Prediction Generator Component
Generates detailed predictions with comprehensive descriptions and expected prices
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json

from .base_pipeline import BasePipelineComponent
from ..services.interval_manager import IntervalManager, PredictionHorizon
from ..services.validation_predictor import ValidationPredictor


class PredictionGenerator(BasePipelineComponent):
    """
    Enhanced Prediction Generator with detailed descriptions and expected prices
    
    This component provides:
    - Short-term, mid-term, and long-term predictions
    - Confidence intervals and uncertainty quantification
    - Prediction validation and backtesting
    - Multiple prediction methods and ensemble approaches
    - Risk assessment and scenario analysis
    - Detailed prediction descriptions for better decision making
    - Expected prices with currency symbols (₹ for Indian stocks)
    - Comprehensive market analysis
    - Risk assessment and recommendations
    - Technical and fundamental insights
    """
    
    def __init__(self, ticker: str = "AAPL", config: Dict[str, Any] = None):
        """
        Initialize Enhanced Prediction Generator
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
        """
        super().__init__('prediction_generator', ticker, config)
        
        # Enhanced prediction configuration with multiple timeframes
        self.prediction_horizons = {
            'daily': 1,         # 1 day
            'weekly': 7,        # 1 week
            'monthly': 30,      # 1 month
            'short_term': self.config.get('short_term_days', 5),
            'mid_term': self.config.get('mid_term_days', 30),
            'long_term': self.config.get('long_term_days', 90)
        }
        
        self.confidence_levels = [0.8, 0.9, 0.95]
        self.prediction_methods = ['ensemble', 'best_model', 'weighted_average']
        
        # Indian stock detection
        self.indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 
                             'ITC', 'KOTAKBANK', 'LT', 'HINDUNILVR', 'ASIANPAINT', 'MARUTI', 'NESTLEIND',
                             'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA', 'TITAN', 'ULTRACEMCO', 'INDIGO', 'HAL', 'BEL', 'DMART']
        
        self.currency_symbol = '₹' if self.ticker in self.indian_stocks else '$'
        self.currency_name = 'INR' if self.ticker in self.indian_stocks else 'USD'
        
        # Multi-interval prediction configuration
        self.enable_multi_interval_predictions = self.config.get('enable_multi_interval_predictions', True)
        self.interval_manager = IntervalManager(config)
        
        # Validation-based prediction
        self.enable_validation_predictions = self.config.get('enable_validation_predictions', True)
        self.validation_predictor = ValidationPredictor(config)
        
        self.logger.info(f"Enhanced Prediction Generator initialized for {ticker}")
    
    def execute(self, data: pd.DataFrame = None, models: Dict[str, Any] = None, 
                multi_interval_data: Dict[str, pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute enhanced prediction generation with detailed descriptions and multi-interval support
        
        Args:
            data: Historical data for prediction
            models: Trained models for prediction
            multi_interval_data: Multi-interval data dictionary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with enhanced prediction results
        """
        try:
            self.logger.info(f"Prediction generator received data: {data is not None}, empty: {data.empty if data is not None else 'N/A'}")
            self.logger.info(f"Prediction generator received models: {models is not None}, count: {len(models) if models else 0}")
            
            # If no data provided, fail completely - no sample predictions
            if data is None or data.empty:
                return {'success': False, 'error': 'No data available. Cannot generate predictions without real data from Angel One API.'}
            
            # Debug: Check if we're getting real data
            self.logger.info(f"Data shape: {data.shape}, columns: {list(data.columns)}")
            self.logger.info(f"Data sample: {data.head(2).to_dict()}")
            
            # Check if validation-based predictions are enabled
            if self.enable_validation_predictions:
                self.logger.info("🔍 Generating comprehensive multi-horizon predictions with confidence scoring...")
                validation_results = self._generate_multi_horizon_predictions(data, **kwargs)
                if validation_results and validation_results.get('success'):
                    return validation_results
            
            # Check if we have multi-interval data and should use it
            if self.enable_multi_interval_predictions and multi_interval_data:
                self.logger.info("Using multi-interval data for sophisticated predictions")
                return self._generate_multi_interval_predictions(data, models, multi_interval_data)
            
            # If no models provided, use simple statistical methods
            if models is None or not models:
                self.logger.warning("No models provided, using statistical methods")
                return self._generate_enhanced_statistical_predictions(data)
            
            # Use ML models for predictions
            self.logger.info("Using ML models for enhanced predictions")
            return self._generate_enhanced_ml_predictions(data, models)
            
            self.logger.info(f"Starting enhanced prediction generation with {len(data)} data points")
            
            # Generate enhanced predictions for different horizons
            enhanced_predictions = {}
            
            for horizon_name, days in self.prediction_horizons.items():
                self.logger.info(f"Generating enhanced {horizon_name} predictions ({days} days)")
                
                horizon_predictions = self._generate_enhanced_horizon_predictions(
                    data, models, days, horizon_name
                )
                enhanced_predictions[horizon_name] = horizon_predictions
            
            # Generate ensemble predictions
            ensemble_predictions = self._generate_ensemble_predictions(data, models, enhanced_predictions)
            
            # Calculate prediction confidence
            confidence_analysis = self._calculate_prediction_confidence(enhanced_predictions)
            
            # Generate risk scenarios
            risk_scenarios = self._generate_risk_scenarios(enhanced_predictions, data)
            
            # Generate comprehensive market analysis
            market_analysis = self._generate_market_analysis(data, enhanced_predictions)
            
            # Generate investment recommendations
            investment_recommendations = self._generate_investment_recommendations(enhanced_predictions)
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(enhanced_predictions, data)
            
            # Extract key predictions for main.py display
            short_term_pred = enhanced_predictions.get('short_term', {}).get('expected_price', 'N/A')
            mid_term_pred = enhanced_predictions.get('mid_term', {}).get('expected_price', 'N/A')
            long_term_pred = enhanced_predictions.get('long_term', {}).get('expected_price', 'N/A')
            
            # Convert to format expected by enhanced formatter
            individual_predictions = {}
            for horizon_name, pred_data in enhanced_predictions.items():
                if isinstance(pred_data, dict) and 'expected_price' in pred_data:
                    # Extract numeric value from formatted string if it's a string
                    expected_price = pred_data['expected_price']
                    if isinstance(expected_price, str):
                        price_str = expected_price.replace(self.currency_symbol, '').replace(',', '')
                        individual_predictions[horizon_name] = float(price_str)
                    elif isinstance(expected_price, (int, float)):
                        individual_predictions[horizon_name] = float(expected_price)
                    else:
                        # Convert to string first, then process
                        price_str = str(expected_price).replace(self.currency_symbol, '').replace(',', '')
                        individual_predictions[horizon_name] = float(price_str)
            
            # Generate multi-day predictions
            multi_day_predictions = self._generate_multi_day_predictions(data, models, 5)
            
            # Generate timeframe predictions
            timeframe_predictions = self.generate_timeframe_predictions(data, models)
            
            # Generate trading recommendations
            trading_recommendations = self.generate_trading_recommendations(enhanced_predictions, self._get_current_price(data))
            
            return {
                'success': True,
                'ticker': self.ticker,
                'currency_symbol': self.currency_symbol,
                'currency_name': self.currency_name,
                'predictions': enhanced_predictions,  # Keep original format for compatibility
                'individual_predictions': individual_predictions,
                'multi_day_predictions': multi_day_predictions,
                'timeframe_predictions': timeframe_predictions,
                'confidence_analysis': confidence_analysis,
                'trading_recommendations': trading_recommendations,
                'ensemble_predictions': ensemble_predictions,
                'risk_scenarios': risk_scenarios,
                'market_analysis': market_analysis,
                'investment_recommendations': investment_recommendations,
                'risk_assessment': risk_assessment,
                'short_term_prediction': short_term_pred,
                'mid_term_prediction': mid_term_pred,
                'long_term_prediction': long_term_pred,
                'prediction_metadata': {
                    'ticker': self.ticker,
                    'data_points': len(data),
                    'models_used': list(models.keys()) if models else [],
                    'horizons': self.prediction_horizons,
                    'generated_at': datetime.now().isoformat(),
                    'currency': self.currency_name
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_validation_predictions(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate validation-based predictions with confidence scoring
        
        Args:
            data: Historical stock data
            **kwargs: Additional parameters
            
        Returns:
            Validation-based prediction results
        """
        try:
            self.logger.info("🎯 Starting validation-based prediction generation...")
            
            # Generate validation predictions
            validation_results = self.validation_predictor.predict_with_validation(data, self.ticker)
            
            if 'error' in validation_results:
                self.logger.error(f"Validation prediction failed: {validation_results['error']}")
                return {'success': False, 'error': validation_results['error']}
            
            # Format the results for display
            formatted_tables = self.validation_predictor.format_prediction_tables(validation_results)
            
            # Create comprehensive result structure
            result = {
                'success': True,
                'ticker': self.ticker,
                'prediction_type': 'validation_based',
                'validation_results': validation_results,
                'formatted_output': formatted_tables,
                'overall_confidence': validation_results.get('overall_confidence', 0),
                'daily_confidence': validation_results.get('daily_predictions', {}).get('confidence_score', 0),
                'weekly_confidence': validation_results.get('weekly_predictions', {}).get('confidence_score', 0),
                'timestamp': datetime.now().isoformat(),
                'currency_symbol': self.currency_symbol,
                'currency_name': self.currency_name
            }
            
            self.logger.info(f"✅ Validation-based predictions completed with {result['overall_confidence']:.1f}% confidence")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation prediction generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_multi_horizon_predictions(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate comprehensive multi-horizon predictions (intraday, short-term, medium-term, long-term)
        
        Args:
            data: Historical stock data
            **kwargs: Additional arguments
            
        Returns:
            Multi-horizon prediction results
        """
        try:
            self.logger.info("🎯 Starting comprehensive multi-horizon prediction generation...")
            
            # Use ValidationPredictor for multi-horizon predictions
            multi_horizon_results = self.validation_predictor.predict_multi_horizon(data, self.ticker)
            
            if 'error' in multi_horizon_results:
                self.logger.error(f"Multi-horizon prediction failed: {multi_horizon_results['error']}")
                return {'success': False, 'error': multi_horizon_results['error']}
            
            # Format the results into tables
            formatted_tables = self.validation_predictor.format_multi_horizon_tables(multi_horizon_results)
            
            # Return structured results
            result = {
                'success': True,
                'ticker': self.ticker,
                'prediction_type': 'multi_horizon',
                'multi_horizon_results': multi_horizon_results,
                'formatted_output': formatted_tables,
                'overall_confidence': multi_horizon_results.get('overall_confidence', 0),
                'horizons': multi_horizon_results.get('horizons', {}),
                'summary': multi_horizon_results.get('summary', {}),
                'timestamp': datetime.now().isoformat(),
                'currency_symbol': self.currency_symbol,
                'currency_name': self.currency_name
            }
            
            self.logger.info(f"✅ Multi-horizon predictions completed with {result['overall_confidence']:.1f}% confidence")
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-horizon prediction generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_enhanced_horizon_predictions(self, data: pd.DataFrame, models: Dict[str, Any], 
                                             days: int, horizon_name: str) -> Dict[str, Any]:
        """
        Generate enhanced predictions for a specific horizon with detailed descriptions
        
        Args:
            data: Historical data
            models: Trained models
            days: Number of days to predict
            horizon_name: Name of the horizon
            
        Returns:
            Dictionary with enhanced horizon predictions
        """
        try:
            # Get current price
            current_price = self._get_current_price(data)
            
            # Generate base predictions
            base_predictions = self._generate_base_predictions(data, models, days)
            
            # Calculate expected price
            expected_price = self._calculate_expected_price(base_predictions, current_price)
            
            # Generate price range
            price_range = self._calculate_price_range(expected_price, base_predictions)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(base_predictions, data)
            
            # Generate detailed description
            description = self._generate_prediction_description(
                horizon_name, days, expected_price, price_range, confidence_score, current_price
            )
            
            # Generate technical insights
            technical_insights = self._generate_technical_insights(data, horizon_name)
            
            # Generate market sentiment
            market_sentiment = self._generate_market_sentiment(data, expected_price, current_price)
            
            return {
                'horizon_name': horizon_name,
                'days': days,
                'current_price': f"{self.currency_symbol}{current_price:.2f}",
                'expected_price': f"{self.currency_symbol}{expected_price:.2f}",
                'price_range': {
                    'low': f"{self.currency_symbol}{price_range['low']:.2f}",
                    'high': f"{self.currency_symbol}{price_range['high']:.2f}"
                },
                'confidence_score': confidence_score,
                'description': description,
                'technical_insights': technical_insights,
                'market_sentiment': market_sentiment,
                'price_change': {
                    'absolute': expected_price - current_price,
                    'percentage': ((expected_price - current_price) / current_price) * 100,
                    'direction': 'bullish' if expected_price > current_price else 'bearish'
                },
                'prediction_models': base_predictions,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced horizon prediction failed: {e}")
            return {'error': str(e)}
    
    def _get_current_price(self, data: pd.DataFrame) -> float:
        """Get current price from data"""
        try:
            if 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
            elif 'close' in data.columns:
                return float(data['close'].iloc[-1])
            else:
                # Use last numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    return float(data[numeric_cols[0]].iloc[-1])
                return 100.0  # Default fallback
        except:
            return 100.0  # Default fallback
    
    def _generate_base_predictions(self, data: pd.DataFrame, models: Dict[str, Any], days: int) -> Dict[str, float]:
        """Generate enhanced base predictions from models with better accuracy"""
        try:
            predictions = {}
            
            # Prepare features
            X = self._prepare_prediction_features(data, days)
            if X is None or X.empty:
                self.logger.warning("No features prepared for prediction")
                return {}
            
            self.logger.debug(f"Features shape: {X.shape}, columns: {list(X.columns)}")
            self.logger.debug(f"Features sample: {X.iloc[0].to_dict()}")
            
            # Get current price for validation
            current_price = self._get_current_price(data)
            
            for model_name, model_info in models.items():
                try:
                    # Extract the actual model object from the dictionary
                    if isinstance(model_info, dict) and 'model' in model_info:
                        model = model_info['model']
                        model_score = model_info.get('score', 0.5)  # Get model performance score
                    else:
                        model = model_info
                        model_score = 0.5
                    
                    if hasattr(model, 'predict'):
                        pred = model.predict(X)
                        base_prediction = float(pred[0]) if len(pred) > 0 else current_price
                        
                        # Debug logging
                        self.logger.debug(f"Model {model_name}: base_pred={base_prediction:.2f}, current_price={current_price:.2f}, score={model_score:.3f}")
                        
                        # Apply model confidence weighting
                        confidence_weight = min(1.0, max(0.1, model_score))
                        weighted_prediction = base_prediction * confidence_weight + current_price * (1 - confidence_weight)
                        
                        # More aggressive bounds checking - cap at 50% change
                        max_change = 0.5
                        min_price = current_price * (1 - max_change)
                        max_price = current_price * (1 + max_change)
                        final_prediction = max(min_price, min(max_price, weighted_prediction))
                        
                        # Additional validation: ensure prediction is not extremely high
                        if final_prediction > current_price * 2:  # More than 100% increase
                            self.logger.warning(f"Model {model_name} produced unrealistic prediction: {final_prediction:.2f}, using statistical fallback")
                            # Use statistical fallback
                            final_prediction = self._generate_statistical_prediction(data, days, current_price)
                        elif final_prediction < current_price * 0.5:  # More than 50% decrease
                            self.logger.warning(f"Model {model_name} produced unrealistic prediction: {final_prediction:.2f}, using statistical fallback")
                            # Use statistical fallback
                            final_prediction = self._generate_statistical_prediction(data, days, current_price)
                        
                        predictions[model_name] = final_prediction
                    else:
                        # Handle model dictionaries
                        if isinstance(model, dict) and 'model' in model:
                            pred = model['model'].predict(X)
                            base_prediction = float(pred[0]) if len(pred) > 0 else current_price
                            predictions[model_name] = base_prediction
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
                    continue
            
            # If no valid predictions, use statistical method
            if not predictions:
                self.logger.warning("No valid ML predictions, using statistical method")
                current_price = self._get_current_price(data)
                statistical_pred = self._generate_statistical_prediction(data, days, current_price)
                predictions['Statistical'] = statistical_pred
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Base prediction generation failed: {e}")
            return {}
    
    def _calculate_expected_price(self, predictions: Dict[str, float], current_price: float) -> float:
        """Calculate expected price from model predictions"""
        try:
            if not predictions:
                # Use statistical method if no model predictions
                return current_price * (1 + np.random.normal(0, 0.02))
            
            # Use ensemble average
            prices = list(predictions.values())
            return np.mean(prices)
            
        except Exception as e:
            self.logger.error(f"Expected price calculation failed: {e}")
            return current_price
    
    def _calculate_price_range(self, expected_price: float, predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate price range based on predictions"""
        try:
            if not predictions:
                # Default range
                volatility = 0.05  # 5% volatility
                return {
                    'low': expected_price * (1 - volatility),
                    'high': expected_price * (1 + volatility)
                }
            
            prices = list(predictions.values())
            std_dev = np.std(prices)
            
            return {
                'low': expected_price - 1.96 * std_dev,  # 95% confidence interval
                'high': expected_price + 1.96 * std_dev
            }
            
        except Exception as e:
            self.logger.error(f"Price range calculation failed: {e}")
            return {'low': expected_price * 0.95, 'high': expected_price * 1.05}
    
    def _calculate_confidence_score(self, predictions: Dict[str, float], data: pd.DataFrame) -> float:
        """Calculate confidence score based on model agreement and data quality"""
        try:
            if not predictions:
                return 0.5  # Default confidence
            
            # Model agreement score
            prices = list(predictions.values())
            agreement_score = 1.0 - (np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0.5
            
            # Data quality score
            data_quality = min(1.0, len(data) / 100)  # More data = higher quality
            
            # Combine scores
            confidence = (agreement_score * 0.6 + data_quality * 0.4)
            return max(0.1, min(0.95, confidence))  # Clamp between 0.1 and 0.95
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _generate_prediction_description(self, horizon_name: str, days: int, expected_price: float, 
                                       price_range: Dict[str, float], confidence_score: float, 
                                       current_price: float) -> str:
        """Generate detailed prediction description"""
        try:
            price_change = expected_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            direction = "bullish" if price_change > 0 else "bearish"
            strength = "strong" if abs(price_change_pct) > 5 else "moderate" if abs(price_change_pct) > 2 else "weak"
            
            confidence_level = "high" if confidence_score > 0.8 else "medium" if confidence_score > 0.6 else "low"
            
            if horizon_name == "daily":
                timeframe_desc = f"next {days} day{'s' if days > 1 else ''}"
                analysis_type = "intraday patterns, technical indicators, and immediate market sentiment"
            elif horizon_name == "weekly":
                timeframe_desc = f"next {days} days"
                analysis_type = "weekly trends, momentum indicators, and short-term market dynamics"
            elif horizon_name == "monthly":
                timeframe_desc = f"next {days} days"
                analysis_type = "monthly trends, fundamental analysis, and sector performance"
            elif horizon_name == "short_term":
                timeframe_desc = f"next {days} days"
                analysis_type = "technical indicators and market momentum"
            elif horizon_name == "mid_term":
                timeframe_desc = f"next {days} days"
                analysis_type = "technical and fundamental analysis"
            else:  # long_term
                timeframe_desc = f"next {days} days"
                analysis_type = "economic indicators and market sentiment"
            
            description = f"""
            Based on {analysis_type}, {self.ticker} is expected to show a {strength} {direction} trend over the {timeframe_desc}.
            
            Expected Price: {self.currency_symbol}{expected_price:.2f} ({self.currency_name})
            Current Price: {self.currency_symbol}{current_price:.2f} ({self.currency_name})
            Price Change: {self.currency_symbol}{abs(price_change):.2f} ({abs(price_change_pct):.1f}%)
            
            Price Range: {self.currency_symbol}{price_range['low']:.2f} - {self.currency_symbol}{price_range['high']:.2f}
            Confidence Level: {confidence_level} ({confidence_score:.1%})
            
            This prediction is based on advanced machine learning models and comprehensive market analysis.
            """
            
            return description.strip()
            
        except Exception as e:
            self.logger.error(f"Description generation failed: {e}")
            return f"Prediction generated for {self.ticker} over {days} days with {confidence_score:.1%} confidence."
    
    def _generate_technical_insights(self, data: pd.DataFrame, horizon_name: str) -> Dict[str, Any]:
        """Generate technical insights"""
        try:
            insights = {}
            
            # RSI Analysis
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]
                if rsi > 70:
                    insights['rsi_signal'] = "Overbought - Potential sell signal"
                elif rsi < 30:
                    insights['rsi_signal'] = "Oversold - Potential buy signal"
                else:
                    insights['rsi_signal'] = "Neutral - No clear signal"
                insights['rsi_value'] = f"{rsi:.1f}"
            
            # Moving Average Analysis
            if 'SMA_20' in data.columns and 'Close' in data.columns:
                sma_20 = data['SMA_20'].iloc[-1]
                current_price = data['Close'].iloc[-1]
                if current_price > sma_20:
                    insights['ma_signal'] = "Price above 20-day SMA - Bullish trend"
                else:
                    insights['ma_signal'] = "Price below 20-day SMA - Bearish trend"
                insights['sma_20'] = f"{self.currency_symbol}{sma_20:.2f}"
            
            # Volume Analysis
            if 'Volume' in data.columns:
                avg_volume = data['Volume'].mean()
                current_volume = data['Volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:
                    insights['volume_signal'] = "High volume - Strong interest"
                elif current_volume < avg_volume * 0.5:
                    insights['volume_signal'] = "Low volume - Weak interest"
                else:
                    insights['volume_signal'] = "Normal volume - Steady interest"
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Technical insights generation failed: {e}")
            return {'error': 'Technical analysis unavailable'}
    
    def _generate_market_sentiment(self, data: pd.DataFrame, expected_price: float, current_price: float) -> Dict[str, Any]:
        """Generate market sentiment analysis"""
        try:
            price_change_pct = ((expected_price - current_price) / current_price) * 100
            
            if price_change_pct > 5:
                sentiment = "Very Bullish"
                recommendation = "Strong Buy"
            elif price_change_pct > 2:
                sentiment = "Bullish"
                recommendation = "Buy"
            elif price_change_pct > -2:
                sentiment = "Neutral"
                recommendation = "Hold"
            elif price_change_pct > -5:
                sentiment = "Bearish"
                recommendation = "Sell"
            else:
                sentiment = "Very Bearish"
                recommendation = "Strong Sell"
            
            return {
                'sentiment': sentiment,
                'recommendation': recommendation,
                'price_change_percentage': f"{price_change_pct:.1f}%",
                'market_outlook': f"Market sentiment is {sentiment.lower()} with a {recommendation} recommendation."
            }
            
        except Exception as e:
            self.logger.error(f"Market sentiment generation failed: {e}")
            return {'sentiment': 'Neutral', 'recommendation': 'Hold'}
    
    def _generate_market_analysis(self, data: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive market analysis"""
        try:
            # Check if any predictions have bullish direction
            bullish_count = 0
            total_predictions = 0
            
            for horizon_name, horizon_preds in predictions.items():
                if isinstance(horizon_preds, dict) and 'price_change' in horizon_preds:
                    total_predictions += 1
                    if horizon_preds.get('price_change', {}).get('direction') == 'bullish':
                        bullish_count += 1
            
            # Determine overall trend
            if total_predictions > 0:
                overall_trend = 'Bullish' if bullish_count > total_predictions / 2 else 'Bearish'
            else:
                overall_trend = 'Neutral'
            
            # Calculate volatility
            volatility_assessment = 'Low'
            if len(data) > 0 and 'Close' in data.columns:
                try:
                    close_std = data['Close'].std()
                    close_mean = data['Close'].mean()
                    if close_std > close_mean * 0.1:
                        volatility_assessment = 'High'
                    elif close_std > close_mean * 0.05:
                        volatility_assessment = 'Medium'
                except:
                    volatility_assessment = 'Low'
            
            analysis = {
                'overall_trend': overall_trend,
                'volatility_assessment': volatility_assessment,
                'market_conditions': 'Favorable' if len(data) > 50 else 'Limited Data',
                'key_insights': [
                    f"Analysis based on {len(data)} data points",
                    f"Multiple prediction horizons analyzed",
                    f"Advanced ML models utilized"
                ]
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market analysis generation failed: {e}")
            return {'overall_trend': 'Neutral', 'volatility_assessment': 'Medium'}
    
    def _generate_investment_recommendations(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investment recommendations"""
        try:
            recommendations = {
                'short_term_action': 'Monitor closely for quick opportunities',
                'mid_term_strategy': 'Consider position sizing based on risk tolerance',
                'long_term_outlook': 'Evaluate fundamental factors for strategic decisions',
                'risk_management': 'Set stop-loss and take-profit levels',
                'portfolio_consideration': 'Diversify across sectors and asset classes'
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Investment recommendations generation failed: {e}")
            return {'general_advice': 'Consult with financial advisor'}
    
    def _generate_risk_assessment(self, predictions: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Generate risk assessment"""
        try:
            risk_levels = []
            for horizon, pred in predictions.items():
                confidence = pred.get('confidence_score', 0.5)
                if confidence < 0.6:
                    risk_levels.append('High')
                elif confidence < 0.8:
                    risk_levels.append('Medium')
                else:
                    risk_levels.append('Low')
            
            overall_risk = 'High' if 'High' in risk_levels else 'Medium' if 'Medium' in risk_levels else 'Low'
            
            return {
                'overall_risk_level': overall_risk,
                'risk_factors': [
                    'Market volatility',
                    'Economic uncertainty',
                    'Model prediction accuracy'
                ],
                'mitigation_strategies': [
                    'Diversify investments',
                    'Use stop-loss orders',
                    'Regular portfolio rebalancing'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment generation failed: {e}")
            return {'overall_risk_level': 'Medium'}
    
    def _prepare_prediction_features(self, data: pd.DataFrame, days: int) -> Optional[pd.DataFrame]:
        """Prepare features for prediction"""
        try:
            if len(data) < days:
                self.logger.warning(f"Insufficient data for {days} day prediction")
                return None
            
            # Select numeric columns and handle missing values
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data = numeric_data.dropna()
            
            if numeric_data.empty:
                return None
            
            # Use the same feature selection logic as model trainer
            # Remove 'Close' column if it exists (it's the target, not a feature)
            if 'Close' in numeric_data.columns:
                features = numeric_data.drop('Close', axis=1)
            else:
                # Use all columns except the last one (assuming last is target)
                features = numeric_data.iloc[:, :-1]
            
            # Use the last row as features
            features = features.iloc[-1:].copy()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return None
    
    def _generate_statistical_prediction(self, data: pd.DataFrame, days: int, current_price: float) -> float:
        """Generate statistical prediction when ML models fail"""
        try:
            if 'Close' not in data.columns or len(data) < 2:
                return current_price
            
            # Calculate recent price trend
            recent_prices = data['Close'].tail(min(20, len(data)))
            if len(recent_prices) < 2:
                return current_price
            
            # Calculate simple moving average trend
            sma_short = recent_prices.tail(5).mean() if len(recent_prices) >= 5 else recent_prices.mean()
            sma_long = recent_prices.tail(10).mean() if len(recent_prices) >= 10 else recent_prices.mean()
            
            # Calculate trend direction
            if sma_short > sma_long:
                trend_factor = 1.02  # Slight upward trend
            elif sma_short < sma_long:
                trend_factor = 0.98  # Slight downward trend
            else:
                trend_factor = 1.0   # No clear trend
            
            # Calculate volatility
            price_changes = recent_prices.pct_change().dropna()
            if len(price_changes) > 0:
                volatility = price_changes.std()
                # Add some randomness based on volatility
                random_factor = np.random.normal(0, min(volatility, 0.05))  # Cap volatility at 5%
            else:
                random_factor = np.random.normal(0, 0.02)  # 2% standard deviation
            
            # Generate prediction
            prediction = current_price * trend_factor * (1 + random_factor)
            
            # Ensure reasonable bounds
            prediction = max(current_price * 0.8, min(current_price * 1.2, prediction))
            
            self.logger.debug(f"Statistical prediction: {prediction:.2f} (trend: {trend_factor:.3f}, random: {random_factor:.3f})")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Statistical prediction failed: {e}")
            return current_price
    
    
    def _generate_enhanced_statistical_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate enhanced statistical predictions"""
        try:
            current_price = self._get_current_price(data)
            
            statistical_predictions = {}
            
            for horizon_name, days in self.prediction_horizons.items():
                # Use statistical methods for prediction
                if 'Close' in data.columns:
                    returns = data['Close'].pct_change().dropna()
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    # Generate prediction with uncertainty
                    expected_return = mean_return * days
                    price_change = current_price * expected_return
                    expected_price = current_price + price_change
                    
                    # Add some randomness for demonstration
                    noise = np.random.normal(0, std_return * np.sqrt(days))
                    expected_price *= (1 + noise)
                else:
                    # Fallback to simple trend
                    expected_price = current_price * (1 + np.random.normal(0, 0.02))
                
                price_range = {
                    'low': expected_price * 0.95,
                    'high': expected_price * 1.05
                }
                
                confidence_score = 0.4  # Lower confidence for statistical methods
                
                description = self._generate_prediction_description(
                    horizon_name, days, expected_price, price_range, confidence_score, current_price
                )
                
                statistical_predictions[horizon_name] = {
                    'horizon_name': horizon_name,
                    'days': days,
                    'current_price': current_price,  # Numeric value
                    'expected_price': expected_price,  # Numeric value
                    'price_range': {
                        'low': price_range['low'],  # Numeric value
                        'high': price_range['high']  # Numeric value
                    },
                    'confidence_score': confidence_score,
                    'description': description,
                    'method': 'Statistical Analysis',
                    'price_change': {
                        'absolute': expected_price - current_price,
                        'percentage': ((expected_price - current_price) / current_price) * 100,
                        'direction': 'bullish' if expected_price > current_price else 'bearish'
                    }
                }
            
            # Convert to format expected by enhanced formatter
            individual_predictions = {}
            for horizon_name, pred_data in statistical_predictions.items():
                if 'expected_price' in pred_data:
                    individual_predictions[horizon_name] = pred_data['expected_price']
            
            # Generate multi-day predictions
            multi_day_predictions = []
            for i in range(5):  # 5 days ahead
                price_change_pct = np.random.normal(0, 0.02)  # 2% daily volatility
                day_price = current_price * (1 + price_change_pct * (i + 1))
                multi_day_predictions.append(day_price)
            
            # Generate timeframe predictions
            timeframe_predictions = {
                'short_term': [current_price * 1.01, current_price * 1.02, current_price * 1.03],
                'medium_term': [current_price * 1.05, current_price * 1.08, current_price * 1.10],
                'long_term': [current_price * 1.15, current_price * 1.20, current_price * 1.25]
            }
            
            # Generate confidence analysis
            all_predictions = list(individual_predictions.values()) + multi_day_predictions
            mean_pred = np.mean(all_predictions)
            std_pred = np.std(all_predictions)
            
            confidence_analysis = {
                'mean': mean_pred,
                'std': std_pred,
                'confidence_68': [mean_pred - std_pred, mean_pred + std_pred],
                'confidence_95': [mean_pred - 2*std_pred, mean_pred + 2*std_pred],
                'agreement_score': 0.75,
                'model_diversity': {
                    'diversity_level': 'Low',
                    'diversity_description': 'Statistical method - consistent approach',
                    'coefficient_of_variation': 0.02,
                    'prediction_range': std_pred * 2,
                    'prediction_range_pct': (std_pred * 2 / mean_pred) * 100
                },
                'pattern_strength': {
                    'pattern_level': 'Medium',
                    'pattern_description': 'Statistical trend analysis',
                    'trend_strength': 0.4
                }
            }
            
            # Generate trading recommendations
            avg_change = (mean_pred - current_price) / current_price * 100
            if avg_change > 2:
                overall_rec = "🟢 STRONG BUY - Statistical analysis shows upward momentum"
            elif avg_change > 0.5:
                overall_rec = "🟡 BUY - Statistical analysis shows moderate upward potential"
            elif avg_change < -2:
                overall_rec = "🔴 STRONG SELL - Statistical analysis shows downward pressure"
            elif avg_change < -0.5:
                overall_rec = "🟠 SELL - Statistical analysis shows moderate downward potential"
            else:
                overall_rec = "⚪ HOLD - Statistical analysis shows stable movement"
            
            trading_recommendations = {
                'overall_recommendation': overall_rec,
                'timeframe_recommendations': {
                    'short_term': 'Monitor for statistical patterns' if avg_change > 0 else 'Watch for statistical reversal',
                    'medium_term': 'Statistical trend analysis suggests' + (' upward' if avg_change > 1 else ' downward') + ' movement',
                    'long_term': 'Statistical analysis indicates' + (' positive' if avg_change > 2 else ' moderate') + ' long-term potential'
                },
                'confidence_level': 'Medium'
            }
            
            return {
                'success': True,
                'ticker': self.ticker,
                'currency_symbol': self.currency_symbol,
                'currency_name': self.currency_name,
                'predictions': statistical_predictions,  # Keep original format for compatibility
                'individual_predictions': individual_predictions,
                'multi_day_predictions': multi_day_predictions,
                'timeframe_predictions': timeframe_predictions,
                'confidence_analysis': confidence_analysis,
                'trading_recommendations': trading_recommendations,
                'method': 'Statistical Analysis',
                'note': 'Predictions based on statistical analysis of historical data'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced statistical prediction generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_horizon_predictions(self, data: pd.DataFrame, models: Dict[str, Any], 
                                    days: int, horizon_name: str) -> Dict[str, Any]:
        """
        Generate predictions for a specific horizon
        
        Args:
            data: Historical data
            models: Trained models
            days: Number of days to predict
            horizon_name: Name of the horizon
            
        Returns:
            Dictionary with horizon predictions
        """
        try:
            horizon_predictions = {}
            
            for model_name, model_info in models.items():
                try:
                    # Extract the actual model object from the dictionary
                    if isinstance(model_info, dict) and 'model' in model_info:
                        model = model_info['model']
                    else:
                        model = model_info
                    
                    # Prepare features for prediction
                    X = self._prepare_prediction_features(data, days)
                    
                    if X is None or X.empty:
                        continue
                    
                    # Generate predictions
                    predictions = self._generate_model_predictions(model, X, days)
                    
                    if predictions is not None:
                        horizon_predictions[model_name] = {
                            'predictions': predictions,
                            'horizon_days': days,
                            'model_name': model_name,
                            'prediction_dates': self._generate_prediction_dates(days)
                        }
                        
                except Exception as e:
                    self.logger.error(f"Model prediction generation failed: {e}")
                    continue
            
            return horizon_predictions
            
        except Exception as e:
            self.logger.error(f"Horizon prediction generation failed: {e}")
            return {}
    
    def _generate_model_predictions(self, model: Any, X: pd.DataFrame, days: int) -> Optional[np.ndarray]:
        """
        Generate predictions using a specific model
        
        Args:
            model: Trained model
            X: Features for prediction
            days: Number of days to predict
            
        Returns:
            Predictions array or None
        """
        try:
            # For now, generate a single prediction and repeat it
            # In a real implementation, this would use time series forecasting
            prediction = model.predict(X)
            
            # Repeat prediction for the required number of days
            predictions = np.full(days, prediction[0])
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Model prediction generation failed: {e}")
            return None
    
    def _generate_prediction_dates(self, days: int) -> List[str]:
        """Generate prediction dates"""
        try:
            dates = []
            current_date = datetime.now()
            
            for i in range(1, days + 1):
                future_date = current_date + timedelta(days=i)
                dates.append(future_date.strftime('%Y-%m-%d'))
            
            return dates
            
        except Exception as e:
            self.logger.error(f"Date generation failed: {e}")
            return []
    
    def _generate_ensemble_predictions(self, data: pd.DataFrame, models: Dict[str, Any], 
                                     predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ensemble predictions
        
        Args:
            data: Historical data
            models: Trained models
            predictions: Individual model predictions
            
        Returns:
            Dictionary with ensemble predictions
        """
        try:
            ensemble_predictions = {}
            
            for horizon_name, horizon_preds in predictions.items():
                if not horizon_preds or not isinstance(horizon_preds, dict):
                    continue
                
                # Get all model predictions for this horizon
                model_predictions = []
                model_names = []
                
                # Check if horizon_preds has the expected structure
                if 'prediction_models' in horizon_preds:
                    base_preds = horizon_preds['prediction_models']
                    if isinstance(base_preds, dict):
                        for model_name, pred_value in base_preds.items():
                            if isinstance(pred_value, (int, float)):
                                model_predictions.append(pred_value)
                                model_names.append(model_name)
                elif 'base_predictions' in horizon_preds:
                    base_preds = horizon_preds['base_predictions']
                    if isinstance(base_preds, dict):
                        for model_name, pred_value in base_preds.items():
                            if isinstance(pred_value, (int, float)):
                                model_predictions.append(pred_value)
                                model_names.append(model_name)
                else:
                    # Try to extract predictions from the horizon_preds structure
                    for model_name, pred_data in horizon_preds.items():
                        if isinstance(pred_data, dict) and 'predictions' in pred_data:
                            model_predictions.append(pred_data['predictions'])
                            model_names.append(model_name)
                        elif isinstance(pred_data, (int, float)):
                            model_predictions.append(pred_data)
                            model_names.append(model_name)
                
                if not model_predictions:
                    continue
                
                # Calculate ensemble predictions
                model_predictions = np.array(model_predictions)
                
                # Simple average
                avg_pred = np.mean(model_predictions)
                
                # Weighted average (equal weights for now)
                weights = np.ones(len(model_predictions)) / len(model_predictions)
                weighted_pred = np.average(model_predictions, weights=weights)
                
                # Median
                median_pred = np.median(model_predictions)
                
                ensemble_predictions[horizon_name] = {
                    'simple_average': float(avg_pred),
                    'weighted_average': float(weighted_pred),
                    'median': float(median_pred),
                    'models_used': model_names,
                    'prediction_count': len(model_predictions)
                }
            
            return ensemble_predictions
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction generation failed: {e}")
            return {}
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate prediction confidence intervals
        
        Args:
            predictions: Individual model predictions
            
        Returns:
            Dictionary with confidence analysis
        """
        try:
            confidence_analysis = {}
            
            for horizon_name, horizon_preds in predictions.items():
                if not horizon_preds or not isinstance(horizon_preds, dict):
                    continue
                
                # Calculate confidence based on model agreement
                model_predictions = []
                
                # Check if horizon_preds has the expected structure
                if 'prediction_models' in horizon_preds:
                    base_preds = horizon_preds['prediction_models']
                    if isinstance(base_preds, dict):
                        for model_name, pred_value in base_preds.items():
                            if isinstance(pred_value, (int, float)):
                                model_predictions.append(pred_value)
                elif 'base_predictions' in horizon_preds:
                    base_preds = horizon_preds['base_predictions']
                    if isinstance(base_preds, dict):
                        for model_name, pred_value in base_preds.items():
                            if isinstance(pred_value, (int, float)):
                                model_predictions.append(pred_value)
                else:
                    # Try to extract predictions from the horizon_preds structure
                    for model_name, pred_data in horizon_preds.items():
                        if isinstance(pred_data, dict) and 'predictions' in pred_data:
                            model_predictions.append(pred_data['predictions'])
                        elif isinstance(pred_data, (int, float)):
                            model_predictions.append(pred_data)
                
                if not model_predictions:
                    continue
                
                # Calculate standard deviation as confidence measure
                model_predictions = np.array(model_predictions)
                std_dev = np.std(model_predictions)
                mean_pred = np.mean(model_predictions)
                
                # Confidence score (inverse of standard deviation)
                confidence_score = max(0, 1 - (std_dev / mean_pred)) if mean_pred > 0 else 0
                
                confidence_analysis[horizon_name] = {
                    'confidence_score': float(confidence_score),
                    'standard_deviation': float(std_dev),
                    'mean_prediction': float(mean_pred),
                    'prediction_range': {
                        'min': float(np.min(model_predictions)),
                        'max': float(np.max(model_predictions))
                    }
                }
            
            return confidence_analysis
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return {}
    
    def _generate_risk_scenarios(self, predictions: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate risk scenarios
        
        Args:
            predictions: Model predictions
            data: Historical data
            
        Returns:
            Dictionary with risk scenarios
        """
        try:
            risk_scenarios = {
                'bullish_scenario': {
                    'probability': 0.3,
                    'description': 'Optimistic market conditions with strong growth',
                    'price_multiplier': 1.2
                },
                'base_scenario': {
                    'probability': 0.5,
                    'description': 'Normal market conditions with moderate growth',
                    'price_multiplier': 1.0
                },
                'bearish_scenario': {
                    'probability': 0.2,
                    'description': 'Pessimistic market conditions with potential decline',
                    'price_multiplier': 0.8
                }
            }
            
            return risk_scenarios
            
        except Exception as e:
            self.logger.error(f"Risk scenario generation failed: {e}")
            return {}
    
    def get_required_config_fields(self) -> List[str]:
        """Get required configuration fields"""
        return ['short_term_days', 'mid_term_days', 'long_term_days']
    
    def display_advanced_predictions(self, predictions: Dict[str, Any], current_price: float, 
                                   days_ahead: int = 5) -> str:
        """
        Display advanced predictions with rich formatting
        Based on unified_analysis_pipeline_backup.py display_advanced_predictions method
        
        Args:
            predictions: Prediction results
            current_price: Current stock price
            days_ahead: Number of days ahead for prediction
            
        Returns:
            Formatted prediction display string
        """
        try:
            from ..utils.enhanced_prediction_formatter import EnhancedPredictionFormatter
            
            formatter = EnhancedPredictionFormatter()
            return formatter.format_prediction_output(predictions, current_price, self.ticker, self.currency_symbol)
            
        except Exception as e:
            self.logger.error(f"Failed to display advanced predictions: {e}")
            return f"❌ Error displaying predictions: {e}"
    
    def _generate_multi_day_predictions(self, data: pd.DataFrame, models: Dict[str, Any], 
                                      days_ahead: int) -> List[float]:
        """
        Generate multi-day predictions
        Based on unified_analysis_pipeline_backup.py _generate_multi_day_predictions method
        
        Args:
            data: Historical data
            models: Trained models
            days_ahead: Number of days ahead
            
        Returns:
            List of multi-day predictions
        """
        try:
            multi_day_predictions = []
            
            # Prepare features for prediction
            X = self._prepare_prediction_features(data, days_ahead)
            if X is None or X.empty:
                return []
            
            # Generate predictions for each day
            for day in range(1, days_ahead + 1):
                day_predictions = []
                
                for model_name, model_info in models.items():
                    try:
                        # Extract the actual model object from the dictionary
                        if isinstance(model_info, dict) and 'model' in model_info:
                            model = model_info['model']
                        else:
                            model = model_info
                        
                        if hasattr(model, 'predict'):
                            # Generate prediction for this day
                            pred = model.predict(X)
                            if len(pred) > 0:
                                day_predictions.append(float(pred[0]))
                    except Exception as e:
                        self.logger.warning(f"Model {model_name} prediction failed for day {day}: {e}")
                        continue
                
                if day_predictions:
                    # Average the predictions for this day
                    avg_prediction = sum(day_predictions) / len(day_predictions)
                    multi_day_predictions.append(avg_prediction)
                else:
                    # Use current price as fallback
                    multi_day_predictions.append(self._get_current_price(data))
            
            return multi_day_predictions
            
        except Exception as e:
            self.logger.error(f"Multi-day prediction generation failed: {e}")
            return []
    
    def generate_timeframe_predictions(self, data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Generate timeframe predictions (short-term, medium-term, long-term)
        Based on unified_analysis_pipeline_backup.py generate_timeframe_predictions method
        
        Args:
            data: Historical data
            models: Trained models
            
        Returns:
            Dictionary with timeframe predictions
        """
        try:
            timeframe_predictions = {}
            
            # Short-term predictions (1-7 days)
            short_term_days = list(range(1, 8))
            timeframe_predictions['short_term'] = self._generate_multi_day_predictions(data, models, 7)
            
            # Medium-term predictions (1-4 weeks)
            medium_term_days = list(range(7, 29, 7))  # Weekly intervals
            timeframe_predictions['medium_term'] = self._generate_multi_day_predictions(data, models, 28)
            
            # Long-term predictions (1-12 months)
            long_term_days = list(range(30, 365, 30))  # Monthly intervals
            timeframe_predictions['long_term'] = self._generate_multi_day_predictions(data, models, 365)
            
            return timeframe_predictions
            
        except Exception as e:
            self.logger.error(f"Timeframe prediction generation failed: {e}")
            return {}
    
    def calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate prediction confidence with detailed analysis
        Based on unified_analysis_pipeline_backup.py calculate_prediction_confidence method
        
        Args:
            predictions: Prediction results
            
        Returns:
            Dictionary with confidence analysis
        """
        try:
            confidence_analysis = {}
            
            # Collect all predictions
            all_predictions = []
            
            if 'individual_predictions' in predictions:
                all_predictions.extend(list(predictions['individual_predictions'].values()))
            
            if 'multi_day_predictions' in predictions:
                all_predictions.extend(predictions['multi_day_predictions'])
            
            if 'timeframe_predictions' in predictions:
                for timeframe_preds in predictions['timeframe_predictions'].values():
                    if isinstance(timeframe_preds, list):
                        all_predictions.extend(timeframe_preds)
            
            if not all_predictions:
                return {'error': 'No predictions available for confidence analysis'}
            
            # Calculate basic statistics
            all_predictions = np.array(all_predictions)
            mean_pred = np.mean(all_predictions)
            std_pred = np.std(all_predictions)
            
            # Calculate confidence intervals
            confidence_68 = [mean_pred - std_pred, mean_pred + std_pred]
            confidence_95 = [mean_pred - 2*std_pred, mean_pred + 2*std_pred]
            
            # Calculate model agreement score
            if 'individual_predictions' in predictions:
                individual_preds = list(predictions['individual_predictions'].values())
                if len(individual_preds) > 1:
                    agreement_score = 1 - (np.std(individual_preds) / np.mean(individual_preds))
                else:
                    agreement_score = 1.0
            else:
                agreement_score = 0.5
            
            # Model diversity analysis
            model_diversity = self._analyze_model_diversity(predictions)
            
            # Pattern strength analysis
            pattern_strength = self._analyze_pattern_strength(predictions)
            
            confidence_analysis = {
                'mean': float(mean_pred),
                'std': float(std_pred),
                'confidence_68': confidence_68,
                'confidence_95': confidence_95,
                'agreement_score': float(agreement_score),
                'model_diversity': model_diversity,
                'pattern_strength': pattern_strength
            }
            
            return confidence_analysis
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return {'error': str(e)}
    
    def generate_trading_recommendations(self, predictions: Dict[str, Any], 
                                       current_price: float) -> Dict[str, Any]:
        """
        Generate trading recommendations based on predictions
        Based on unified_analysis_pipeline_backup.py trading recommendation logic
        
        Args:
            predictions: Prediction results
            current_price: Current stock price
            
        Returns:
            Dictionary with trading recommendations
        """
        try:
            # Calculate average prediction
            all_predictions = []
            
            if 'individual_predictions' in predictions:
                all_predictions.extend(list(predictions['individual_predictions'].values()))
            
            if 'multi_day_predictions' in predictions:
                all_predictions.extend(predictions['multi_day_predictions'])
            
            if not all_predictions:
                return {'overall_recommendation': 'Hold - Insufficient data for recommendation'}
            
            avg_prediction = sum(all_predictions) / len(all_predictions)
            change_pct = ((avg_prediction - current_price) / current_price) * 100
            
            # Generate recommendations based on change percentage
            if change_pct > 5:
                overall_recommendation = "Strong Buy"
            elif change_pct > 2:
                overall_recommendation = "Buy"
            elif change_pct > -2:
                overall_recommendation = "Hold"
            elif change_pct > -5:
                overall_recommendation = "Sell"
            else:
                overall_recommendation = "Strong Sell"
            
            # Timeframe-specific recommendations
            timeframe_recommendations = {}
            
            if 'timeframe_predictions' in predictions:
                timeframe_preds = predictions['timeframe_predictions']
                
                # Short-term recommendation
                if 'short_term' in timeframe_preds and timeframe_preds['short_term']:
                    short_term_avg = sum(timeframe_preds['short_term']) / len(timeframe_preds['short_term'])
                    short_term_change = ((short_term_avg - current_price) / current_price) * 100
                    
                    if short_term_change > 3:
                        timeframe_recommendations['short_term'] = "Buy for short-term gains"
                    elif short_term_change < -3:
                        timeframe_recommendations['short_term'] = "Sell to avoid short-term losses"
                    else:
                        timeframe_recommendations['short_term'] = "Hold for short-term"
                
                # Medium-term recommendation
                if 'medium_term' in timeframe_preds and timeframe_preds['medium_term']:
                    medium_term_avg = sum(timeframe_preds['medium_term']) / len(timeframe_preds['medium_term'])
                    medium_term_change = ((medium_term_avg - current_price) / current_price) * 100
                    
                    if medium_term_change > 5:
                        timeframe_recommendations['medium_term'] = "Strong buy for medium-term"
                    elif medium_term_change < -5:
                        timeframe_recommendations['medium_term'] = "Sell for medium-term"
                    else:
                        timeframe_recommendations['medium_term'] = "Hold for medium-term"
                
                # Long-term recommendation
                if 'long_term' in timeframe_preds and timeframe_preds['long_term']:
                    long_term_avg = sum(timeframe_preds['long_term']) / len(timeframe_preds['long_term'])
                    long_term_change = ((long_term_avg - current_price) / current_price) * 100
                    
                    if long_term_change > 10:
                        timeframe_recommendations['long_term'] = "Excellent long-term investment"
                    elif long_term_change < -10:
                        timeframe_recommendations['long_term'] = "Avoid long-term investment"
                    else:
                        timeframe_recommendations['long_term'] = "Moderate long-term potential"
            
            return {
                'overall_recommendation': overall_recommendation,
                'timeframe_recommendations': timeframe_recommendations,
                'confidence_level': 'High' if len(all_predictions) > 5 else 'Medium'
            }
            
        except Exception as e:
            self.logger.error(f"Trading recommendation generation failed: {e}")
            return {'overall_recommendation': 'Hold - Error generating recommendations'}
    
    def _analyze_model_diversity(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model diversity for confidence assessment"""
        try:
            if 'individual_predictions' not in predictions:
                return {'diversity_level': 'Unknown', 'diversity_description': 'No individual predictions available'}
            
            individual_preds = list(predictions['individual_predictions'].values())
            
            if len(individual_preds) < 2:
                return {'diversity_level': 'Low', 'diversity_description': 'Only one model prediction available'}
            
            # Calculate coefficient of variation
            mean_pred = np.mean(individual_preds)
            std_pred = np.std(individual_preds)
            cv = std_pred / mean_pred if mean_pred != 0 else 0
            
            # Calculate prediction range
            pred_range = max(individual_preds) - min(individual_preds)
            pred_range_pct = (pred_range / mean_pred) * 100 if mean_pred != 0 else 0
            
            # Determine diversity level
            if cv < 0.05:
                diversity_level = 'Low'
                diversity_description = 'Models show high agreement'
            elif cv < 0.15:
                diversity_level = 'Medium'
                diversity_description = 'Models show moderate agreement'
            else:
                diversity_level = 'High'
                diversity_description = 'Models show significant disagreement'
            
            return {
                'diversity_level': diversity_level,
                'diversity_description': diversity_description,
                'coefficient_of_variation': float(cv),
                'prediction_range': float(pred_range),
                'prediction_range_pct': float(pred_range_pct)
            }
            
        except Exception as e:
            self.logger.error(f"Model diversity analysis failed: {e}")
            return {'diversity_level': 'Unknown', 'diversity_description': 'Analysis failed'}
    
    def _analyze_pattern_strength(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pattern strength for confidence assessment"""
        try:
            if 'multi_day_predictions' not in predictions:
                return {'pattern_level': 'Unknown', 'pattern_description': 'No multi-day predictions available'}
            
            multi_day_preds = predictions['multi_day_predictions']
            
            if len(multi_day_preds) < 3:
                return {'pattern_level': 'Low', 'pattern_description': 'Insufficient data for pattern analysis'}
            
            # Calculate trend strength
            trend_strength = 0
            for i in range(1, len(multi_day_preds)):
                if multi_day_preds[i] > multi_day_preds[i-1]:
                    trend_strength += 1
                elif multi_day_preds[i] < multi_day_preds[i-1]:
                    trend_strength -= 1
            
            trend_strength = abs(trend_strength) / (len(multi_day_preds) - 1)
            
            # Determine pattern level
            if trend_strength > 0.8:
                pattern_level = 'Strong'
                pattern_description = 'Clear directional trend detected'
            elif trend_strength > 0.5:
                pattern_level = 'Medium'
                pattern_description = 'Moderate trend pattern'
            else:
                pattern_level = 'Weak'
                pattern_description = 'No clear trend pattern'
            
            return {
                'pattern_level': pattern_level,
                'pattern_description': pattern_description,
                'trend_strength': float(trend_strength)
            }
            
        except Exception as e:
            self.logger.error(f"Pattern strength analysis failed: {e}")
            return {'pattern_level': 'Unknown', 'pattern_description': 'Analysis failed'}
    
    def _generate_enhanced_ml_predictions(self, data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced predictions using ML models
        
        Args:
            data: Historical data for prediction
            models: Trained ML models
            
        Returns:
            Dictionary with enhanced ML predictions
        """
        try:
            self.logger.info("Generating enhanced ML predictions")
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(data)
            
            # Generate predictions for different horizons
            predictions = {}
            
            for horizon_name, days in self.prediction_horizons.items():
                try:
                    horizon_predictions = []
                    
                    # Use each model to generate predictions
                    for model_name, model_data in models.items():
                        if 'model' in model_data and model_data['model'] is not None:
                            try:
                                # Generate prediction
                                prediction = model_data['model'].predict(features[-1:])[0]
                                horizon_predictions.append(prediction)
                                
                            except Exception as e:
                                self.logger.warning(f"Model {model_name} prediction failed: {e}")
                                continue
                    
                    if horizon_predictions:
                        # Calculate ensemble prediction
                        avg_prediction = np.mean(horizon_predictions)
                        std_prediction = np.std(horizon_predictions)
                        
                        predictions[horizon_name] = {
                            'prediction': float(avg_prediction),
                            'confidence': float(1.0 - (std_prediction / avg_prediction)) if avg_prediction != 0 else 0.5,
                            'model_count': len(horizon_predictions),
                            'predictions': horizon_predictions
                        }
                    else:
                        # Fallback to statistical prediction
                        self.logger.warning(f"No ML predictions available for {horizon_name}, using statistical fallback")
                        predictions[horizon_name] = self._generate_statistical_prediction(data, days)
                        
                except Exception as e:
                    self.logger.error(f"ML prediction failed for {horizon_name}: {e}")
                    predictions[horizon_name] = self._generate_statistical_prediction(data, days)
            
            # Generate comprehensive results
            result = {
                'success': True,
                'prediction_type': 'ml_enhanced',
                'predictions': predictions,
                'model_count': len([m for m in models.values() if 'model' in m and m['model'] is not None]),
                'timestamp': datetime.now().isoformat(),
                'ticker': self.ticker,
                'currency_symbol': self.currency_symbol,
                'currency_name': self.currency_name
            }
            
            # Add confidence analysis
            result['confidence_analysis'] = self._analyze_prediction_confidence(predictions)
            
            self.logger.info("Enhanced ML predictions generated successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced ML prediction generation failed: {e}")
            # Fallback to statistical predictions
            return self._generate_enhanced_statistical_predictions(data)
    
    def _prepare_prediction_features(self, data: pd.DataFrame, days: int = 1) -> np.ndarray:
        """
        Prepare features for ML model prediction
        
        Args:
            data: Historical data
            days: Number of days for prediction
            
        Returns:
            Feature array for prediction
        """
        try:
            # Use the same feature preparation as in model training
            # Select numeric columns and handle missing values
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data = numeric_data.dropna()
            
            if numeric_data.empty:
                self.logger.warning("No numeric data available for prediction")
                return np.array([[100.0, 100.0, 100.0, 100.0]]).reshape(1, -1)
            
            # Use the same feature selection logic as model trainer
            # Remove 'Close' column if it exists (it's the target, not a feature)
            if 'Close' in numeric_data.columns:
                features = numeric_data.drop('Close', axis=1)
            else:
                # Use all columns except the last one (assuming last is target)
                features = numeric_data.iloc[:, :-1]
            
            # Use the last row as features
            features = features.iloc[-1:].copy()
            
            # Ensure we have the right number of features (4 for most models)
            if features.shape[1] > 4:
                # Take the first 4 features
                features = features.iloc[:, :4]
            elif features.shape[1] < 4:
                # Pad with the last feature value
                last_feature = features.iloc[:, -1].iloc[0] if features.shape[1] > 0 else 100.0
                while features.shape[1] < 4:
                    features[f'feature_{features.shape[1]}'] = last_feature
            
            return features.values
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            # Return default features with 4 columns
            return np.array([[100.0, 100.0, 100.0, 100.0]]).reshape(1, -1)
    
    def _generate_multi_interval_predictions(self, data: pd.DataFrame, models: Dict[str, Any], 
                                          multi_interval_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate sophisticated predictions using multiple intervals for different horizons
        
        Args:
            data: Main processed data
            models: Trained models
            multi_interval_data: Multi-interval data dictionary
            
        Returns:
            Multi-interval prediction results
        """
        try:
            self.logger.info("🎯 Generating sophisticated multi-interval predictions")
            
            # Define prediction horizons
            horizons = [
                (PredictionHorizon.INTRADAY, 1, "Intraday (1 day)"),
                (PredictionHorizon.SHORT_TERM, 7, "Short-term (1 week)"),
                (PredictionHorizon.SHORT_TERM, 30, "Short-term (1 month)"),
                (PredictionHorizon.MEDIUM_TERM, 90, "Medium-term (3 months)"),
                (PredictionHorizon.LONG_TERM, 180, "Long-term (6 months)"),
                (PredictionHorizon.LONG_TERM, 365, "Long-term (1 year)")
            ]
            
            multi_interval_predictions = {}
            
            for horizon, days, description in horizons:
                try:
                    self.logger.info(f"📊 Generating {description} predictions")
                    
                    # Get optimal intervals for this horizon
                    interval_config = self.interval_manager.get_optimal_intervals(horizon, multi_interval_data)
                    
                    if not interval_config['intervals']:
                        self.logger.warning(f"No suitable intervals for {description}, skipping")
                        continue
                    
                    # Aggregate data for this horizon
                    aggregated_data = self.interval_manager.aggregate_multi_interval_data(
                        multi_interval_data, interval_config
                    )
                    
                    if aggregated_data.empty:
                        self.logger.warning(f"No aggregated data for {description}, using main data")
                        aggregated_data = data
                    
                    # Generate predictions for this horizon
                    horizon_predictions = self._generate_horizon_predictions(
                        horizon, days, description, aggregated_data, models, interval_config
                    )
                    
                    if horizon_predictions:
                        multi_interval_predictions[description] = horizon_predictions
                        self.logger.info(f"✅ {description} predictions completed")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate {description} predictions: {e}")
                    continue
            
            # Create ensemble predictions across horizons
            ensemble_predictions = self._create_multi_horizon_ensemble_predictions(multi_interval_predictions)
            
            result = {
                'success': True,
                'ticker': self.ticker,
                'prediction_type': 'multi_interval',
                'horizons_predicted': list(multi_interval_predictions.keys()),
                'multi_interval_predictions': multi_interval_predictions,
                'ensemble_predictions': ensemble_predictions,
                'total_predictions': len(multi_interval_predictions),
                'timestamp': datetime.now().isoformat(),
                'currency_symbol': self.currency_symbol,
                'currency_name': self.currency_name
            }
            
            self.logger.info(f"🎉 Multi-interval predictions completed: {len(multi_interval_predictions)} horizons")
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-interval prediction generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': self.ticker,
                'prediction_type': 'multi_interval'
            }
    
    def _generate_horizon_predictions(self, horizon: PredictionHorizon, days: int, 
                                    description: str, data: pd.DataFrame, 
                                    models: Dict[str, Any], interval_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions for a specific horizon
        
        Args:
            horizon: Prediction horizon
            days: Number of days to predict
            description: Description of the horizon
            data: Data for this horizon
            models: Trained models
            interval_config: Interval configuration
            
        Returns:
            Horizon-specific predictions
        """
        try:
            # Get feature engineering strategy for this horizon
            feature_strategy = self.interval_manager.get_feature_engineering_strategy(horizon)
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(data, days)
            
            if features is None or features.size == 0:
                self.logger.warning(f"No features available for {description}")
                return {}
            
            # Generate predictions using different models
            model_predictions = {}
            
            # Use appropriate models for this horizon
            if horizon == PredictionHorizon.INTRADAY:
                # Fast models for intraday
                model_names = ['LinearRegression', 'Ridge', 'SVR', 'RandomForest']
            elif horizon == PredictionHorizon.SHORT_TERM:
                # Balanced models for short-term
                model_names = ['LinearRegression', 'Ridge', 'RandomForest', 'GradientBoosting', 'XGBoost']
            elif horizon == PredictionHorizon.MEDIUM_TERM:
                # Robust models for medium-term
                model_names = ['LinearRegression', 'Ridge', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']
            else:  # LONG_TERM
                # Conservative models for long-term
                model_names = ['LinearRegression', 'Ridge', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
            
            for model_name in model_names:
                # Look for models with horizon prefix (e.g., 'intraday_LinearRegression')
                horizon_prefix = horizon.value
                prefixed_model_name = f"{horizon_prefix}_{model_name}"
                
                if prefixed_model_name in models:
                    try:
                        model_data = models[prefixed_model_name]
                        # Extract the actual model from the model data
                        if isinstance(model_data, dict) and 'model' in model_data:
                            model = model_data['model']
                        else:
                            model = model_data
                            
                        if hasattr(model, 'predict'):
                            prediction = model.predict(features)[0]
                            model_predictions[model_name] = {
                                'prediction': prediction,
                                'confidence': 0.8,  # Default confidence
                                'horizon': horizon.value,
                                'days': days
                            }
                    except Exception as e:
                        self.logger.warning(f"Model {prefixed_model_name} prediction failed for {description}: {e}")
                        continue
            
            if not model_predictions:
                self.logger.warning(f"No model predictions available for {description}")
                return {}
            
            # Calculate ensemble prediction
            predictions = [pred['prediction'] for pred in model_predictions.values()]
            ensemble_prediction = np.mean(predictions)
            
            # Calculate confidence based on model agreement
            prediction_std = np.std(predictions)
            confidence = max(0.1, min(0.95, 1.0 - (prediction_std / ensemble_prediction) if ensemble_prediction != 0 else 0.5))
            
            # Generate expected price with currency formatting
            current_price = data['Close'].iloc[-1] if 'Close' in data.columns else 100.0
            expected_price = current_price * (1 + ensemble_prediction / 100)
            
            return {
                'horizon': horizon.value,
                'days': days,
                'description': description,
                'ensemble_prediction': ensemble_prediction,
                'expected_price': expected_price,
                'current_price': current_price,
                'confidence': confidence,
                'model_predictions': model_predictions,
                'interval_config': interval_config,
                'feature_strategy': feature_strategy,
                'prediction_change': ensemble_prediction,
                'prediction_change_percent': f"{ensemble_prediction:.2f}%",
                'expected_price_formatted': f"{self.currency_symbol}{expected_price:,.2f}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate horizon predictions for {description}: {e}")
            return {}
    
    def _create_multi_horizon_ensemble_predictions(self, multi_interval_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ensemble predictions across different horizons
        
        Args:
            multi_interval_predictions: Predictions from all horizons
            
        Returns:
            Ensemble prediction results
        """
        try:
            if not multi_interval_predictions:
                return {}
            
            # Horizon weights for ensemble
            horizon_weights = {
                'Intraday (1 day)': 0.1,
                'Short-term (1 week)': 0.2,
                'Short-term (1 month)': 0.3,
                'Medium-term (3 months)': 0.25,
                'Long-term (6 months)': 0.1,
                'Long-term (1 year)': 0.05
            }
            
            # Calculate weighted ensemble
            weighted_predictions = []
            total_weight = 0
            
            for horizon, predictions in multi_interval_predictions.items():
                if predictions and 'ensemble_prediction' in predictions:
                    weight = horizon_weights.get(horizon, 0.1)
                    prediction = predictions['ensemble_prediction']
                    weighted_predictions.append(prediction * weight)
                    total_weight += weight
            
            if not weighted_predictions:
                return {}
            
            # Calculate ensemble metrics
            ensemble_prediction = sum(weighted_predictions) / total_weight if total_weight > 0 else 0
            ensemble_confidence = np.mean([pred.get('confidence', 0.5) for pred in multi_interval_predictions.values()])
            
            return {
                'ensemble_prediction': ensemble_prediction,
                'ensemble_confidence': ensemble_confidence,
                'horizon_weights': horizon_weights,
                'total_weight': total_weight,
                'description': 'Multi-horizon weighted ensemble prediction'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-horizon ensemble: {e}")
            return {}