"""
Enhanced Prediction Formatter
Rich console output with emojis, formatting, and comprehensive analysis
Based on unified_analysis_pipeline_backup.py display methods
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPredictionFormatter:
    """
    Enhanced prediction formatter with rich console output
    Provides comprehensive display methods from backup version
    """
    
    def __init__(self):
        """Initialize Enhanced Prediction Formatter"""
        self.currency_symbols = {
            'USD': '$',
            'INR': '₹',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CAD': 'C$',
            'AUD': 'A$'
        }
        
        # Indian stock indicators
        self.indian_stock_indicators = [
            '.NS', '.BO', '.NSE', '.BSE', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 
            'ICICIBANK', 'WIPRO', 'BHARTIARTL', 'ITC', 'SBIN', 'KOTAKBANK',
            'ONGC', 'NTPC', 'HAL', 'BEL', 'DMART', 'IOC', 'DLF', 'LICI', 
            'HINDUNILVR', 'ASIANPAINT', 'MARUTI', 'TITAN', 'NESTLEIND',
            'ULTRACEMCO', 'BAJFINANCE', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE',
            'POWERGRID', 'TECHM', 'SUNPHARMA', 'TATAMOTORS', 'AXISBANK',
            'INDUSINDBK', 'COALINDIA', 'GRASIM', 'JSWSTEEL', 'TATASTEEL',
            'ADANIPORTS', 'BAJAJ-AUTO', 'DRREDDY', 'EICHERMOT', 'HEROMOTOCO',
            'HINDALCO', 'HINDPETRO', 'LT', 'M&M', 'NTPC', 'RELIANCE',
            'TATACONSUM', 'TATASTEEL', 'UPL', 'WIPRO', 'PNB'
        ]
        
        logger.info("Enhanced Prediction Formatter initialized")
    
    def format_prediction_output(self, predictions: Dict[str, Any], current_price: float, 
                               ticker: str, currency_symbol: str = None) -> str:
        """
        Format comprehensive prediction output
        
        Args:
            predictions: Prediction results
            current_price: Current stock price
            ticker: Stock ticker
            currency_symbol: Currency symbol to use
            
        Returns:
            Formatted prediction output string
        """
        try:
            # Determine currency symbol
            if not currency_symbol:
                currency_symbol = self._get_currency_symbol(ticker)
            
            output = []
            
            # Header
            output.append(self._format_prediction_header(ticker, current_price, currency_symbol))
            
            # Individual model predictions
            if 'individual_predictions' in predictions:
                output.append(self._format_individual_model_predictions(
                    predictions['individual_predictions'], current_price, currency_symbol
                ))
            
            # Multi-day predictions
            if 'multi_day_predictions' in predictions:
                output.append(self._format_multi_day_predictions(
                    predictions['multi_day_predictions'], current_price, currency_symbol
                ))
            
            # Timeframe predictions
            if 'timeframe_predictions' in predictions:
                output.append(self._format_timeframe_predictions(
                    predictions['timeframe_predictions'], current_price, currency_symbol
                ))
            
            # Confidence analysis
            if 'confidence_analysis' in predictions:
                output.append(self._format_confidence_analysis(
                    predictions['confidence_analysis'], currency_symbol
                ))
            
            # Trading recommendations
            if 'trading_recommendations' in predictions:
                output.append(self._format_trading_recommendations(
                    predictions['trading_recommendations']
                ))
            
            # Price summary
            output.append(self._format_price_summary(current_price, predictions, currency_symbol))
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Failed to format prediction output: {e}")
            return f"❌ Error formatting predictions: {e}"
    
    def _format_prediction_header(self, ticker: str, current_price: float, currency_symbol: str) -> str:
        """Format prediction header"""
        try:
            analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            header = [
                "\n🎯 ADVANCED PREDICTION RESULTS",
                "=" * 80,
                f"📊 Stock: {ticker}",
                f"📅 Prediction Period: 5 days",
                f"💰 CURRENT PRICE: {self._format_price(current_price, currency_symbol)}",
                f"📅 Analysis Date: {analysis_date}",
                "-" * 80,
                ""
            ]
            
            return '\n'.join(header)
            
        except Exception as e:
            logger.error(f"Failed to format prediction header: {e}")
            return f"❌ Error formatting header: {e}"
    
    def _format_individual_model_predictions(self, predictions: Dict[str, float], 
                                           current_price: float, currency_symbol: str) -> str:
        """Format individual model predictions"""
        try:
            if not predictions:
                return ""
            
            output = ["🤖 Individual Model Predictions:"]
            
            for model_name, pred in predictions.items():
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                output.append(
                    f"   • {model_name}: {self._format_price(pred, currency_symbol)} "
                    f"({direction} {change_pct:+.2f}%)"
                )
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Failed to format individual model predictions: {e}")
            return f"❌ Error formatting individual predictions: {e}"
    
    def _format_multi_day_predictions(self, multi_day_predictions: List[float], 
                                     current_price: float, currency_symbol: str) -> str:
        """Format multi-day predictions"""
        try:
            if not multi_day_predictions:
                return ""
            
            output = [f"\n🚀 Multi-Day Predictions ({len(multi_day_predictions)} days):"]
            
            for i, pred in enumerate(multi_day_predictions, 1):
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                
                output.append(
                    f"   • Day {i}: {self._format_price(pred, currency_symbol)} "
                    f"({direction} {change_pct:+.2f}%)"
                )
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Failed to format multi-day predictions: {e}")
            return f"❌ Error formatting multi-day predictions: {e}"
    
    def _format_timeframe_predictions(self, timeframe_predictions: Dict[str, List[float]], 
                                     current_price: float, currency_symbol: str) -> str:
        """Format timeframe predictions"""
        try:
            if not timeframe_predictions:
                return ""
            
            output = ["\n⏰ TIMEFRAME PREDICTIONS:"]
            
            # Short-term predictions (1-7 days)
            if 'short_term' in timeframe_predictions:
                output.append("\n📅 SHORT-TERM (1-7 days):")
                for i, pred in enumerate(timeframe_predictions['short_term'], 1):
                    change = pred - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    date_display = self._format_prediction_date(datetime.now(), i)
                    output.append(
                        f"   • {date_display}: {self._format_price(pred, currency_symbol)} "
                        f"({direction} {change_pct:+.2f}%)"
                    )
            
            # Medium-term predictions (1-4 weeks)
            if 'medium_term' in timeframe_predictions:
                output.append("\n📅 MEDIUM-TERM (1-4 weeks):")
                for i, pred in enumerate(timeframe_predictions['medium_term'], 1):
                    change = pred - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    date_display = self._format_week_prediction_date(datetime.now(), i)
                    output.append(
                        f"   • {date_display}: {self._format_price(pred, currency_symbol)} "
                        f"({direction} {change_pct:+.2f}%)"
                    )
            
            # Long-term predictions (1-12 months)
            if 'long_term' in timeframe_predictions:
                output.append("\n📅 LONG-TERM (1-12 months):")
                for i, pred in enumerate(timeframe_predictions['long_term'], 1):
                    change = pred - current_price
                    change_pct = (change / current_price) * 100
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    
                    date_display = self._format_month_prediction_date(datetime.now(), i)
                    output.append(
                        f"   • {date_display}: {self._format_price(pred, currency_symbol)} "
                        f"({direction} {change_pct:+.2f}%)"
                    )
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Failed to format timeframe predictions: {e}")
            return f"❌ Error formatting timeframe predictions: {e}"
    
    def _format_confidence_analysis(self, confidence_analysis: Dict[str, Any], currency_symbol: str) -> str:
        """Format confidence analysis"""
        try:
            if not confidence_analysis:
                return ""
            
            output = ["\n📈 CONFIDENCE ANALYSIS:"]
            
            # Basic confidence metrics
            if 'mean' in confidence_analysis:
                output.append(f"   Mean Prediction: {self._format_price(confidence_analysis['mean'], currency_symbol)}")
            
            if 'std' in confidence_analysis:
                output.append(f"   Standard Deviation: {self._format_price(confidence_analysis['std'], currency_symbol)}")
            
            # Confidence intervals
            if 'confidence_68' in confidence_analysis:
                ci_68 = confidence_analysis['confidence_68']
                output.append(
                    f"   68% Confidence Interval: {self._format_price(ci_68[0], currency_symbol)} - "
                    f"{self._format_price(ci_68[1], currency_symbol)}"
                )
            
            if 'confidence_95' in confidence_analysis:
                ci_95 = confidence_analysis['confidence_95']
                output.append(
                    f"   95% Confidence Interval: {self._format_price(ci_95[0], currency_symbol)} - "
                    f"{self._format_price(ci_95[1], currency_symbol)}"
                )
            
            if 'agreement_score' in confidence_analysis:
                output.append(f"   Model Agreement Score: {confidence_analysis['agreement_score']:.3f}")
            
            # Model diversity analysis
            if 'model_diversity' in confidence_analysis:
                diversity = confidence_analysis['model_diversity']
                output.extend([
                    "\n🔍 MODEL DIVERSITY ANALYSIS:",
                    f"   Diversity Level: {diversity.get('diversity_level', 'Unknown')}",
                    f"   Description: {diversity.get('diversity_description', 'Unknown')}",
                    f"   Coefficient of Variation: {diversity.get('coefficient_of_variation', 0):.4f}",
                    f"   Prediction Range: {self._format_price(diversity.get('prediction_range', 0), currency_symbol)} "
                    f"({diversity.get('prediction_range_pct', 0):.2f}%)"
                ])
            
            # Pattern strength analysis
            if 'pattern_strength' in confidence_analysis:
                pattern = confidence_analysis['pattern_strength']
                output.extend([
                    "\n📊 PATTERN STRENGTH ANALYSIS:",
                    f"   Pattern Level: {pattern.get('pattern_level', 'Unknown')}",
                    f"   Description: {pattern.get('pattern_description', 'Unknown')}",
                    f"   Trend Strength: {pattern.get('trend_strength', 0):.4f}"
                ])
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Failed to format confidence analysis: {e}")
            return f"❌ Error formatting confidence analysis: {e}"
    
    def _format_trading_recommendations(self, recommendations: Dict[str, Any]) -> str:
        """Format trading recommendations"""
        try:
            if not recommendations:
                return ""
            
            output = []
            
            # Overall recommendation
            if 'overall_recommendation' in recommendations:
                output.append(f"\n💡 Trading Recommendation: {recommendations['overall_recommendation']}")
            
            # Timeframe-specific recommendations
            if 'timeframe_recommendations' in recommendations:
                output.append("\n🎯 TIMEFRAME RECOMMENDATIONS:")
                
                timeframe_recs = recommendations['timeframe_recommendations']
                
                if 'short_term' in timeframe_recs:
                    output.append(f"   📅 Short-term (1-7 days): {timeframe_recs['short_term']}")
                
                if 'medium_term' in timeframe_recs:
                    output.append(f"   📅 Medium-term (1-4 weeks): {timeframe_recs['medium_term']}")
                
                if 'long_term' in timeframe_recs:
                    output.append(f"   📅 Long-term (1-12 months): {timeframe_recs['long_term']}")
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Failed to format trading recommendations: {e}")
            return f"❌ Error formatting trading recommendations: {e}"
    
    def _format_price_summary(self, current_price: float, predictions: Dict[str, Any], 
                            currency_symbol: str) -> str:
        """Format price summary"""
        try:
            # Calculate average prediction
            all_predictions = []
            
            # Collect all predictions
            if 'individual_predictions' in predictions:
                all_predictions.extend(list(predictions['individual_predictions'].values()))
            
            if 'multi_day_predictions' in predictions:
                all_predictions.extend(predictions['multi_day_predictions'])
            
            if 'timeframe_predictions' in predictions:
                for timeframe_preds in predictions['timeframe_predictions'].values():
                    if isinstance(timeframe_preds, list):
                        all_predictions.extend(timeframe_preds)
            
            if not all_predictions:
                return ""
            
            avg_prediction = sum(all_predictions) / len(all_predictions)
            avg_change = avg_prediction - current_price
            avg_change_pct = (avg_change / current_price) * 100
            avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
            
            output = [
                f"\n📊 Average Prediction: {self._format_price(avg_prediction, currency_symbol)} "
                f"({avg_direction} {avg_change_pct:+.2f}%)",
                "",
                "📋 PRICE SUMMARY:",
                f"   Current Price: {self._format_price(current_price, currency_symbol)}",
                f"   Predicted Price: {self._format_price(avg_prediction, currency_symbol)}",
                f"   Expected Change: {avg_change_pct:+.2f}%"
            ]
            
            return '\n'.join(output)
            
        except Exception as e:
            logger.error(f"Failed to format price summary: {e}")
            return f"❌ Error formatting price summary: {e}"
    
    def _format_price(self, price: float, currency_symbol: str) -> str:
        """Format price with currency symbol"""
        try:
            if currency_symbol == '₹':
                return f"₹{price:.2f}"
            else:
                return f"{currency_symbol}{price:.2f}"
        except Exception as e:
            logger.error(f"Failed to format price: {e}")
            return f"{price:.2f}"
    
    def _get_currency_symbol(self, ticker: str) -> str:
        """Get currency symbol for ticker"""
        try:
            ticker_upper = ticker.upper()
            
            # Check if it's an Indian stock
            for indicator in self.indian_stock_indicators:
                if indicator in ticker_upper:
                    return '₹'
            
            # Default to USD for international stocks
            return '$'
            
        except Exception as e:
            logger.error(f"Failed to get currency symbol: {e}")
            return '$'
    
    def _format_prediction_date(self, base_date: datetime, days_ahead: int) -> str:
        """Format prediction date with day name"""
        try:
            prediction_date = base_date + timedelta(days=days_ahead)
            day_name = prediction_date.strftime('%a')
            date_short = prediction_date.strftime('%b %d')
            return f"{day_name}, {date_short}"
        except Exception as e:
            logger.error(f"Failed to format prediction date: {e}")
            return f"Day {days_ahead}"
    
    def _format_week_prediction_date(self, base_date: datetime, weeks_ahead: int) -> str:
        """Format week prediction date"""
        try:
            prediction_date = base_date + timedelta(weeks=weeks_ahead)
            date_short = prediction_date.strftime('%b %d')
            return f"Week {weeks_ahead} ({date_short})"
        except Exception as e:
            logger.error(f"Failed to format week prediction date: {e}")
            return f"Week {weeks_ahead}"
    
    def _format_month_prediction_date(self, base_date: datetime, months_ahead: int) -> str:
        """Format month prediction date"""
        try:
            # Simple month calculation
            prediction_date = base_date + timedelta(days=months_ahead * 30)
            date_short = prediction_date.strftime('%b %d')
            return f"Month {months_ahead} ({date_short})"
        except Exception as e:
            logger.error(f"Failed to format month prediction date: {e}")
            return f"Month {months_ahead}"
    
    def format_price_with_change(self, price: float, current_price: float, currency_symbol: str) -> str:
        """Format price with change percentage"""
        try:
            change = price - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            
            return f"{self._format_price(price, currency_symbol)} ({direction} {change_pct:+.2f}%)"
        except Exception as e:
            logger.error(f"Failed to format price with change: {e}")
            return self._format_price(price, currency_symbol)
    
    def get_formatter_status(self) -> Dict[str, Any]:
        """Get formatter status"""
        try:
            return {
                'currency_symbols': self.currency_symbols,
                'indian_stock_indicators': len(self.indian_stock_indicators),
                'initialized': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get formatter status: {e}")
            return {'error': str(e)}
