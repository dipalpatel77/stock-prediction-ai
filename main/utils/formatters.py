"""
Formatters
Price and currency formatting utilities
"""

import logging
from typing import Union, Optional, Dict, Any
from decimal import Decimal, ROUND_HALF_UP
import locale
from datetime import datetime


class PriceFormatter:
    """
    Price formatting utilities
    
    This formatter provides:
    - Price formatting with currency symbols
    - Percentage formatting
    - Number formatting with appropriate precision
    - Localized formatting
    """
    
    def __init__(self, currency: str = "USD", locale_code: str = "en_US"):
        """
        Initialize Price Formatter
        
        Args:
            currency: Default currency code
            locale_code: Locale code for formatting
        """
        self.currency = currency
        self.locale_code = locale_code
        
        # Currency symbols mapping
        self.currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'INR': '₹',
            'CNY': '¥',
            'CAD': 'C$',
            'AUD': 'A$',
            'CHF': 'CHF',
            'SEK': 'kr',
            'NOK': 'kr',
            'DKK': 'kr',
            'PLN': 'zł',
            'CZK': 'Kč',
            'HUF': 'Ft',
            'RUB': '₽',
            'BRL': 'R$',
            'MXN': '$',
            'ZAR': 'R',
            'KRW': '₩',
            'SGD': 'S$',
            'HKD': 'HK$',
            'TWD': 'NT$',
            'THB': '฿',
            'MYR': 'RM',
            'IDR': 'Rp',
            'PHP': '₱',
            'VND': '₫'
        }
        
        # Set locale
        try:
            locale.setlocale(locale.LC_ALL, locale_code)
        except locale.Error:
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, 'C')
                except locale.Error:
                    pass  # Use default formatting
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Price Formatter initialized for {currency}")
    
    def format_price(self, price: Union[float, int, Decimal], 
                    currency: Optional[str] = None, 
                    precision: int = 2,
                    show_currency: bool = True) -> str:
        """
        Format price with currency symbol
        
        Args:
            price: Price value
            currency: Currency code (uses default if None)
            precision: Decimal precision
            show_currency: Whether to show currency symbol
            
        Returns:
            Formatted price string
        """
        try:
            if currency is None:
                currency = self.currency
            
            # Convert to Decimal for precise formatting
            if isinstance(price, (int, float)):
                price = Decimal(str(price))
            
            # Round to specified precision
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            # Format number
            if precision == 0:
                formatted_price = f"{int(price)}"
            else:
                formatted_price = f"{price:.{precision}f}"
            
            # Add currency symbol if requested
            if show_currency:
                symbol = self.currency_symbols.get(currency, currency)
                if currency in ['USD', 'EUR', 'GBP', 'INR']:
                    return f"{symbol}{formatted_price}"
                else:
                    return f"{formatted_price} {symbol}"
            else:
                return formatted_price
                
        except Exception as e:
            self.logger.error(f"Failed to format price: {e}")
            return str(price)
    
    def format_percentage(self, value: Union[float, int, Decimal], 
                         precision: int = 2,
                         show_sign: bool = True) -> str:
        """
        Format percentage value
        
        Args:
            value: Percentage value
            precision: Decimal precision
            show_sign: Whether to show + sign for positive values
            
        Returns:
            Formatted percentage string
        """
        try:
            # Convert to Decimal for precise formatting
            if isinstance(value, (int, float)):
                value = Decimal(str(value))
            
            # Round to specified precision
            value = value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            # Format with sign
            if show_sign and value > 0:
                return f"+{value:.{precision}f}%"
            else:
                return f"{value:.{precision}f}%"
                
        except Exception as e:
            self.logger.error(f"Failed to format percentage: {e}")
            return f"{value}%"
    
    def format_large_number(self, number: Union[float, int, Decimal], 
                           precision: int = 1) -> str:
        """
        Format large numbers with K, M, B suffixes
        
        Args:
            number: Number to format
            precision: Decimal precision
            
        Returns:
            Formatted number string
        """
        try:
            if isinstance(number, (int, float)):
                number = Decimal(str(number))
            
            abs_number = abs(number)
            sign = "-" if number < 0 else ""
            
            if abs_number >= 1_000_000_000:
                formatted = f"{sign}{abs_number / 1_000_000_000:.{precision}f}B"
            elif abs_number >= 1_000_000:
                formatted = f"{sign}{abs_number / 1_000_000:.{precision}f}M"
            elif abs_number >= 1_000:
                formatted = f"{sign}{abs_number / 1_000:.{precision}f}K"
            else:
                formatted = f"{sign}{abs_number:.{precision}f}"
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Failed to format large number: {e}")
            return str(number)
    
    def format_volume(self, volume: Union[float, int, Decimal]) -> str:
        """
        Format trading volume
        
        Args:
            volume: Volume value
            
        Returns:
            Formatted volume string
        """
        try:
            return self.format_large_number(volume, precision=0)
        except Exception as e:
            self.logger.error(f"Failed to format volume: {e}")
            return str(volume)
    
    def format_market_cap(self, market_cap: Union[float, int, Decimal], 
                          currency: Optional[str] = None) -> str:
        """
        Format market capitalization
        
        Args:
            market_cap: Market cap value
            currency: Currency code
            
        Returns:
            Formatted market cap string
        """
        try:
            formatted_cap = self.format_large_number(market_cap, precision=1)
            if currency:
                symbol = self.currency_symbols.get(currency, currency)
                return f"{symbol}{formatted_cap}"
            else:
                return formatted_cap
                
        except Exception as e:
            self.logger.error(f"Failed to format market cap: {e}")
            return str(market_cap)


class CurrencyFormatter:
    """
    Currency formatting utilities
    
    This formatter provides:
    - Currency conversion formatting
    - Exchange rate formatting
    - Multi-currency support
    - Currency symbol handling
    """
    
    def __init__(self):
        """Initialize Currency Formatter"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Currency Formatter initialized")
    
    def format_exchange_rate(self, rate: Union[float, int, Decimal], 
                           from_currency: str, to_currency: str,
                           precision: int = 4) -> str:
        """
        Format exchange rate
        
        Args:
            rate: Exchange rate value
            from_currency: Source currency
            to_currency: Target currency
            precision: Decimal precision
            
        Returns:
            Formatted exchange rate string
        """
        try:
            if isinstance(rate, (int, float)):
                rate = Decimal(str(rate))
            
            formatted_rate = f"{rate:.{precision}f}"
            return f"1 {from_currency} = {formatted_rate} {to_currency}"
            
        except Exception as e:
            self.logger.error(f"Failed to format exchange rate: {e}")
            return f"1 {from_currency} = {rate} {to_currency}"
    
    def format_currency_conversion(self, amount: Union[float, int, Decimal],
                                 from_currency: str, to_currency: str,
                                 exchange_rate: Union[float, int, Decimal],
                                 precision: int = 2) -> str:
        """
        Format currency conversion
        
        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            exchange_rate: Exchange rate
            precision: Decimal precision
            
        Returns:
            Formatted conversion string
        """
        try:
            if isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            if isinstance(exchange_rate, (int, float)):
                exchange_rate = Decimal(str(exchange_rate))
            
            converted_amount = amount * exchange_rate
            formatted_amount = f"{converted_amount:.{precision}f}"
            
            return f"{amount} {from_currency} = {formatted_amount} {to_currency}"
            
        except Exception as e:
            self.logger.error(f"Failed to format currency conversion: {e}")
            return f"{amount} {from_currency} = {amount * exchange_rate} {to_currency}"
    
    def format_currency_pair(self, base_currency: str, quote_currency: str,
                           rate: Union[float, int, Decimal],
                           precision: int = 4) -> str:
        """
        Format currency pair
        
        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            rate: Exchange rate
            precision: Decimal precision
            
        Returns:
            Formatted currency pair string
        """
        try:
            if isinstance(rate, (int, float)):
                rate = Decimal(str(rate))
            
            formatted_rate = f"{rate:.{precision}f}"
            return f"{base_currency}/{quote_currency} = {formatted_rate}"
            
        except Exception as e:
            self.logger.error(f"Failed to format currency pair: {e}")
            return f"{base_currency}/{quote_currency} = {rate}"
    
    def format_currency_change(self, change: Union[float, int, Decimal],
                             currency: str, precision: int = 2) -> str:
        """
        Format currency change
        
        Args:
            change: Change value
            currency: Currency code
            precision: Decimal precision
            
        Returns:
            Formatted change string
        """
        try:
            if isinstance(change, (int, float)):
                change = Decimal(str(change))
            
            sign = "+" if change > 0 else ""
            formatted_change = f"{sign}{change:.{precision}f}"
            
            return f"{formatted_change} {currency}"
            
        except Exception as e:
            self.logger.error(f"Failed to format currency change: {e}")
            return f"{change} {currency}"


class NumberFormatter:
    """
    Number formatting utilities
    
    This formatter provides:
    - Number formatting with locale support
    - Scientific notation formatting
    - Compact number formatting
    - Precision control
    """
    
    def __init__(self, locale_code: str = "en_US"):
        """
        Initialize Number Formatter
        
        Args:
            locale_code: Locale code for formatting
        """
        self.locale_code = locale_code
        
        # Set locale
        try:
            locale.setlocale(locale.LC_ALL, locale_code)
        except locale.Error:
            try:
                locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, 'C')
                except locale.Error:
                    pass  # Use default formatting
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Number Formatter initialized for {locale_code}")
    
    def format_number(self, number: Union[float, int, Decimal],
                     precision: int = 2,
                     use_locale: bool = True) -> str:
        """
        Format number with locale support
        
        Args:
            number: Number to format
            precision: Decimal precision
            use_locale: Whether to use locale formatting
            
        Returns:
            Formatted number string
        """
        try:
            if use_locale:
                return locale.format_string(f"%.{precision}f", number, grouping=True)
            else:
                return f"{number:.{precision}f}"
                
        except Exception as e:
            self.logger.error(f"Failed to format number: {e}")
            return str(number)
    
    def format_scientific(self, number: Union[float, int, Decimal],
                         precision: int = 2) -> str:
        """
        Format number in scientific notation
        
        Args:
            number: Number to format
            precision: Decimal precision
            
        Returns:
            Formatted scientific notation string
        """
        try:
            return f"{number:.{precision}e}"
        except Exception as e:
            self.logger.error(f"Failed to format scientific notation: {e}")
            return str(number)
    
    def format_compact(self, number: Union[float, int, Decimal],
                      precision: int = 1) -> str:
        """
        Format number in compact notation
        
        Args:
            number: Number to format
            precision: Decimal precision
            
        Returns:
            Formatted compact notation string
        """
        try:
            if isinstance(number, (int, float)):
                number = Decimal(str(number))
            
            abs_number = abs(number)
            sign = "-" if number < 0 else ""
            
            if abs_number >= 1_000_000_000_000:
                formatted = f"{sign}{abs_number / 1_000_000_000_000:.{precision}f}T"
            elif abs_number >= 1_000_000_000:
                formatted = f"{sign}{abs_number / 1_000_000_000:.{precision}f}B"
            elif abs_number >= 1_000_000:
                formatted = f"{sign}{abs_number / 1_000_000:.{precision}f}M"
            elif abs_number >= 1_000:
                formatted = f"{sign}{abs_number / 1_000:.{precision}f}K"
            else:
                formatted = f"{sign}{abs_number:.{precision}f}"
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Failed to format compact notation: {e}")
            return str(number)
    
    def format_price_with_change(self, price: Union[float, int, str, Decimal], 
                               current_price: Union[float, int, str, Decimal],
                               currency: str = None, precision: int = 2) -> str:
        """
        Format price with change percentage and direction emoji
        
        Args:
            price: Price to format
            current_price: Current price for comparison
            currency: Currency code
            precision: Decimal precision
            
        Returns:
            Formatted price with change
        """
        try:
            # Convert to Decimal for precise calculation
            if isinstance(price, str):
                price = Decimal(price)
            elif isinstance(price, (int, float)):
                price = Decimal(str(price))
            elif not isinstance(price, Decimal):
                price = Decimal(str(price))
            
            if isinstance(current_price, str):
                current_price = Decimal(current_price)
            elif isinstance(current_price, (int, float)):
                current_price = Decimal(str(current_price))
            elif not isinstance(current_price, Decimal):
                current_price = Decimal(str(current_price))
            
            # Calculate change
            change = price - current_price
            change_pct = (change / current_price) * 100
            
            # Determine direction emoji
            if change > 0:
                direction = "📈"
            elif change < 0:
                direction = "📉"
            else:
                direction = "➡️"
            
            # Format price
            formatted_price = self.format_price(price, currency, precision)
            
            # Format change percentage
            change_pct_str = f"{change_pct:+.2f}%"
            
            return f"{formatted_price} ({direction} {change_pct_str})"
            
        except Exception as e:
            self.logger.error(f"Price with change formatting failed: {e}")
            return self.format_price(price, currency, precision)
    
    def format_prediction_summary(self, current_price: Union[float, int, str, Decimal],
                                 predicted_price: Union[float, int, str, Decimal],
                                 currency: str = None, precision: int = 2) -> str:
        """
        Format prediction summary with current and predicted prices
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted stock price
            currency: Currency code
            precision: Decimal precision
            
        Returns:
            Formatted prediction summary
        """
        try:
            # Convert to Decimal for precise calculation
            if isinstance(current_price, str):
                current_price = Decimal(current_price)
            elif isinstance(current_price, (int, float)):
                current_price = Decimal(str(current_price))
            elif not isinstance(current_price, Decimal):
                current_price = Decimal(str(current_price))
            
            if isinstance(predicted_price, str):
                predicted_price = Decimal(predicted_price)
            elif isinstance(predicted_price, (int, float)):
                predicted_price = Decimal(str(predicted_price))
            elif not isinstance(predicted_price, Decimal):
                predicted_price = Decimal(str(predicted_price))
            
            # Calculate change
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            
            # Determine direction emoji
            if change > 0:
                direction = "📈"
            elif change < 0:
                direction = "📉"
            else:
                direction = "➡️"
            
            # Format prices
            current_formatted = self.format_price(current_price, currency, precision)
            predicted_formatted = self.format_price(predicted_price, currency, precision)
            
            # Format change percentage
            change_pct_str = f"{change_pct:+.2f}%"
            
            return f"Current: {current_formatted} → Predicted: {predicted_formatted} ({direction} {change_pct_str})"
            
        except Exception as e:
            self.logger.error(f"Prediction summary formatting failed: {e}")
            return f"Current: {current_price} → Predicted: {predicted_price}"
    
    def get_currency_symbol_for_ticker(self, ticker: str) -> str:
        """
        Get currency symbol for stock ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Currency symbol
        """
        try:
            ticker_upper = ticker.upper()
            
            # Indian stock indicators
            indian_indicators = [
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
            
            # Check if it's an Indian stock
            for indicator in indian_indicators:
                if indicator in ticker_upper:
                    return '₹'
            
            # Default to USD for international stocks
            return '$'
            
        except Exception as e:
            self.logger.error(f"Failed to get currency symbol for ticker: {e}")
            return '$'
