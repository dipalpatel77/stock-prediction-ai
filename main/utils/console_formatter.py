"""
Console Formatter Utility
Rich console output formatting with colors, emojis, and styling
Based on unified_analysis_pipeline_backup.py console formatting
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsoleFormatter:
    """
    Console formatter with rich output, colors, and styling
    Provides comprehensive console formatting utilities
    """
    
    def __init__(self):
        """Initialize Console Formatter"""
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'end': '\033[0m'
        }
        
        self.emoji_map = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'loading': '🔄',
            'rocket': '🚀',
            'chart': '📊',
            'money': '💰',
            'calendar': '📅',
            'robot': '🤖',
            'target': '🎯',
            'lightbulb': '💡',
            'clock': '⏰',
            'trend_up': '📈',
            'trend_down': '📉',
            'trend_side': '➡️',
            'star': '⭐',
            'fire': '🔥',
            'diamond': '💎',
            'shield': '🛡️',
            'flag': '🏁',
            'trophy': '🏆',
            'medal': '🥇',
            'thumbs_up': '👍',
            'thumbs_down': '👎',
            'check': '✔️',
            'cross': '✖️',
            'arrow_right': '➡️',
            'arrow_left': '⬅️',
            'arrow_up': '⬆️',
            'arrow_down': '⬇️'
        }
        
        logger.info("Console Formatter initialized")
    
    def format_header(self, title: str, width: int = 80, char: str = "=") -> str:
        """
        Format header with title and border
        
        Args:
            title: Header title
            width: Header width
            char: Border character
            
        Returns:
            Formatted header string
        """
        try:
            border = char * width
            title_line = f"{title:^{width}}"
            
            return f"\n{border}\n{title_line}\n{border}"
            
        except Exception as e:
            logger.error(f"Failed to format header: {e}")
            return f"\n{title}\n"
    
    def format_section(self, title: str, width: int = 80, char: str = "-") -> str:
        """
        Format section header
        
        Args:
            title: Section title
            width: Section width
            char: Border character
            
        Returns:
            Formatted section string
        """
        try:
            border = char * width
            return f"\n{title}\n{border}"
            
        except Exception as e:
            logger.error(f"Failed to format section: {e}")
            return f"\n{title}\n"
    
    def format_list_item(self, item: str, bullet: str = "•", indent: int = 3) -> str:
        """
        Format list item with bullet and indentation
        
        Args:
            item: List item text
            bullet: Bullet character
            indent: Indentation spaces
            
        Returns:
            Formatted list item string
        """
        try:
            return f"{' ' * indent}{bullet} {item}"
            
        except Exception as e:
            logger.error(f"Failed to format list item: {e}")
            return f"  {item}"
    
    def format_key_value(self, key: str, value: str, separator: str = ": ", indent: int = 3) -> str:
        """
        Format key-value pair
        
        Args:
            key: Key text
            value: Value text
            separator: Separator between key and value
            indent: Indentation spaces
            
        Returns:
            Formatted key-value string
        """
        try:
            return f"{' ' * indent}{key}{separator}{value}"
            
        except Exception as e:
            logger.error(f"Failed to format key-value: {e}")
            return f"  {key}: {value}"
    
    def format_prediction_header(self, ticker: str, current_price: float, 
                               currency_symbol: str, analysis_date: str) -> str:
        """
        Format prediction header with comprehensive information
        
        Args:
            ticker: Stock ticker
            current_price: Current stock price
            currency_symbol: Currency symbol
            analysis_date: Analysis date
            
        Returns:
            Formatted prediction header
        """
        try:
            header = [
                self.format_header("🎯 ADVANCED PREDICTION RESULTS", 80, "="),
                f"📊 Stock: {ticker}",
                f"📅 Prediction Period: 5 days",
                f"💰 CURRENT PRICE: {self._format_price(current_price, currency_symbol)}",
                f"📅 Analysis Date: {analysis_date}",
                self.format_section("", 80, "-"),
                ""
            ]
            
            return '\n'.join(header)
            
        except Exception as e:
            logger.error(f"Failed to format prediction header: {e}")
            return f"❌ Error formatting prediction header: {e}"
    
    def format_prediction_summary(self, predictions: Dict[str, Any], current_price: float, 
                                 currency_symbol: str) -> str:
        """
        Format prediction summary with key metrics
        
        Args:
            predictions: Prediction results
            current_price: Current stock price
            currency_symbol: Currency symbol
            
        Returns:
            Formatted prediction summary
        """
        try:
            summary = []
            
            # Calculate average prediction
            all_predictions = []
            if 'individual_predictions' in predictions:
                all_predictions.extend(list(predictions['individual_predictions'].values()))
            if 'multi_day_predictions' in predictions:
                all_predictions.extend(predictions['multi_day_predictions'])
            
            if all_predictions:
                avg_prediction = sum(all_predictions) / len(all_predictions)
                avg_change = avg_prediction - current_price
                avg_change_pct = (avg_change / current_price) * 100
                avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
                
                summary.extend([
                    f"\n📊 Average Prediction: {self._format_price(avg_prediction, currency_symbol)} "
                    f"({avg_direction} {avg_change_pct:+.2f}%)",
                    "",
                    "📋 PRICE SUMMARY:",
                    f"   Current Price: {self._format_price(current_price, currency_symbol)}",
                    f"   Predicted Price: {self._format_price(avg_prediction, currency_symbol)}",
                    f"   Expected Change: {avg_change_pct:+.2f}%"
                ])
            
            return '\n'.join(summary)
            
        except Exception as e:
            logger.error(f"Failed to format prediction summary: {e}")
            return f"❌ Error formatting prediction summary: {e}"
    
    def format_trading_recommendation(self, recommendation: str, confidence: float = None) -> str:
        """
        Format trading recommendation with color coding
        
        Args:
            recommendation: Trading recommendation
            confidence: Confidence level (0-1)
            
        Returns:
            Formatted trading recommendation
        """
        try:
            if confidence is not None:
                confidence_pct = confidence * 100
                return f"💡 Trading Recommendation: {recommendation} (Confidence: {confidence_pct:.1f}%)"
            else:
                return f"💡 Trading Recommendation: {recommendation}"
                
        except Exception as e:
            logger.error(f"Failed to format trading recommendation: {e}")
            return f"💡 Trading Recommendation: {recommendation}"
    
    def format_confidence_analysis(self, confidence_data: Dict[str, Any], currency_symbol: str) -> str:
        """
        Format confidence analysis with detailed metrics
        
        Args:
            confidence_data: Confidence analysis data
            currency_symbol: Currency symbol
            
        Returns:
            Formatted confidence analysis
        """
        try:
            analysis = ["\n📈 CONFIDENCE ANALYSIS:"]
            
            if 'mean' in confidence_data:
                analysis.append(f"   Mean Prediction: {self._format_price(confidence_data['mean'], currency_symbol)}")
            
            if 'std' in confidence_data:
                analysis.append(f"   Standard Deviation: {self._format_price(confidence_data['std'], currency_symbol)}")
            
            if 'confidence_68' in confidence_data:
                ci_68 = confidence_data['confidence_68']
                analysis.append(
                    f"   68% Confidence Interval: {self._format_price(ci_68[0], currency_symbol)} - "
                    f"{self._format_price(ci_68[1], currency_symbol)}"
                )
            
            if 'confidence_95' in confidence_data:
                ci_95 = confidence_data['confidence_95']
                analysis.append(
                    f"   95% Confidence Interval: {self._format_price(ci_95[0], currency_symbol)} - "
                    f"{self._format_price(ci_95[1], currency_symbol)}"
                )
            
            if 'agreement_score' in confidence_data:
                analysis.append(f"   Model Agreement Score: {confidence_data['agreement_score']:.3f}")
            
            return '\n'.join(analysis)
            
        except Exception as e:
            logger.error(f"Failed to format confidence analysis: {e}")
            return f"❌ Error formatting confidence analysis: {e}"
    
    def format_progress_bar(self, current: int, total: int, width: int = 50) -> str:
        """
        Format progress bar
        
        Args:
            current: Current progress
            total: Total progress
            width: Progress bar width
            
        Returns:
            Formatted progress bar
        """
        try:
            if total == 0:
                return "Progress: 0% [                    ]"
            
            percentage = (current / total) * 100
            filled_width = int((current / total) * width)
            bar = "█" * filled_width + "░" * (width - filled_width)
            
            return f"Progress: {percentage:.1f}% [{bar}]"
            
        except Exception as e:
            logger.error(f"Failed to format progress bar: {e}")
            return f"Progress: {current}/{total}"
    
    def format_status_message(self, message: str, status: str = "info") -> str:
        """
        Format status message with emoji and color
        
        Args:
            message: Status message
            status: Status type (success, error, warning, info)
            
        Returns:
            Formatted status message
        """
        try:
            emoji = self.emoji_map.get(status, "ℹ️")
            color = self.colors.get(status, self.colors['white'])
            
            return f"{emoji} {color}{message}{self.colors['end']}"
            
        except Exception as e:
            logger.error(f"Failed to format status message: {e}")
            return f"ℹ️ {message}"
    
    def format_table_row(self, row: List[str], widths: List[int]) -> str:
        """
        Format table row with specified column widths
        
        Args:
            row: Row data
            widths: Column widths
            
        Returns:
            Formatted table row
        """
        try:
            formatted_cells = []
            for i, cell in enumerate(row):
                width = widths[i] if i < len(widths) else 10
                formatted_cells.append(f"{str(cell):<{width}}")
            
            return " | ".join(formatted_cells)
            
        except Exception as e:
            logger.error(f"Failed to format table row: {e}")
            return " | ".join(str(cell) for cell in row)
    
    def format_table_header(self, headers: List[str], widths: List[int]) -> str:
        """
        Format table header with specified column widths
        
        Args:
            headers: Header data
            widths: Column widths
            
        Returns:
            Formatted table header
        """
        try:
            header_row = self.format_table_row(headers, widths)
            separator = "-" * len(header_row)
            
            return f"{header_row}\n{separator}"
            
        except Exception as e:
            logger.error(f"Failed to format table header: {e}")
            return " | ".join(str(header) for header in headers)
    
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
    
    def get_console_formatter_status(self) -> Dict[str, Any]:
        """Get console formatter status"""
        try:
            return {
                'colors_available': len(self.colors),
                'emojis_available': len(self.emoji_map),
                'initialized': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get console formatter status: {e}")
            return {'error': str(e)}


# Global instance for easy access
console_formatter = ConsoleFormatter()

