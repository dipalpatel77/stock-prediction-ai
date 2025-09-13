#!/usr/bin/env python3
"""
Enhanced Date Utilities
========================

Provides comprehensive date handling with multiple formats and timezone support.
Features:
- Multiple date format support
- Timezone conversion
- Business day calculations
- Fiscal year handling
- Date range generation
- Holiday calendar support
- Market hours calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date, time
import pytz
import calendar
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DateRange:
    """Date range information"""
    start_date: datetime
    end_date: datetime
    duration_days: int
    duration_weeks: float
    duration_months: float
    business_days: int
    weekends: int
    holidays: int
    trading_days: int

@dataclass
class MarketHours:
    """Market trading hours information"""
    market_name: str
    timezone: str
    open_time: time
    close_time: time
    is_24h: bool
    trading_days: List[str]
    holidays: List[date]

@dataclass
class FiscalPeriod:
    """Fiscal period information"""
    fiscal_year: int
    quarter: int
    period_start: date
    period_end: date
    period_name: str
    days_in_period: int
    business_days_in_period: int

class EnhancedDateUtils:
    """
    Enhanced Date Utilities with comprehensive date handling
    
    Supports multiple formats, timezones, and business calculations
    """
    
    def __init__(self, default_timezone: str = 'UTC'):
        """Initialize enhanced date utilities"""
        self.default_timezone = pytz.timezone(default_timezone)
        
        # Date format templates
        self.date_formats = {
            'iso': '%Y-%m-%d',
            'iso_datetime': '%Y-%m-%d %H:%M:%S',
            'iso_datetime_tz': '%Y-%m-%d %H:%M:%S %Z',
            'us': '%m/%d/%Y',
            'us_datetime': '%m/%d/%Y %I:%M %p',
            'eu': '%d/%m/%Y',
            'eu_datetime': '%d/%m/%Y %H:%M',
            'compact': '%d-%m-%Y',
            'compact_datetime': '%d-%m-%Y %H:%M',
            'readable': '%B %d, %Y',
            'readable_datetime': '%B %d, %Y at %I:%M %p',
            'short': '%b %d, %Y',
            'short_datetime': '%b %d, %Y %I:%M %p',
            'day_month': '%d %B',
            'month_year': '%B %Y',
            'year_month': '%Y-%m',
            'weekday': '%A',
            'weekday_short': '%a',
            'month_name': '%B',
            'month_short': '%b',
            'day_name': '%A',
            'day_short': '%a',
            'time_12h': '%I:%M %p',
            'time_24h': '%H:%M',
            'time_seconds': '%H:%M:%S',
            'timestamp': '%Y%m%d_%H%M%S',
            'filename': '%Y%m%d_%H%M%S',
            'log': '%Y-%m-%d %H:%M:%S',
            'api': '%Y-%m-%dT%H:%M:%SZ',
            'excel': '%m/%d/%Y',
            'csv': '%Y-%m-%d',
            'json': '%Y-%m-%dT%H:%M:%S.%fZ'
        }
        
        # Market hours configuration
        self.market_hours = {
            'NYSE': MarketHours(
                market_name='New York Stock Exchange',
                timezone='America/New_York',
                open_time=time(9, 30),
                close_time=time(16, 0),
                is_24h=False,
                trading_days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                holidays=[]
            ),
            'NASDAQ': MarketHours(
                market_name='NASDAQ',
                timezone='America/New_York',
                open_time=time(9, 30),
                close_time=time(16, 0),
                is_24h=False,
                trading_days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                holidays=[]
            ),
            'LSE': MarketHours(
                market_name='London Stock Exchange',
                timezone='Europe/London',
                open_time=time(8, 0),
                close_time=time(16, 30),
                is_24h=False,
                trading_days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                holidays=[]
            ),
            'TSE': MarketHours(
                market_name='Tokyo Stock Exchange',
                timezone='Asia/Tokyo',
                open_time=time(9, 0),
                close_time=time(15, 0),
                is_24h=False,
                trading_days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                holidays=[]
            ),
            'NSE': MarketHours(
                market_name='National Stock Exchange of India',
                timezone='Asia/Kolkata',
                open_time=time(9, 15),
                close_time=time(15, 30),
                is_24h=False,
                trading_days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                holidays=[]
            ),
            'BSE': MarketHours(
                market_name='Bombay Stock Exchange',
                timezone='Asia/Kolkata',
                open_time=time(9, 15),
                close_time=time(15, 30),
                is_24h=False,
                trading_days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                holidays=[]
            ),
            'CRYPTO': MarketHours(
                market_name='Cryptocurrency Markets',
                timezone='UTC',
                open_time=time(0, 0),
                close_time=time(23, 59),
                is_24h=True,
                trading_days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                holidays=[]
            )
        }
        
        # Common holidays (US)
        self.us_holidays = [
            date(2024, 1, 1),   # New Year's Day
            date(2024, 1, 15),  # Martin Luther King Jr. Day
            date(2024, 2, 19),  # Presidents' Day
            date(2024, 3, 29),  # Good Friday
            date(2024, 5, 27),  # Memorial Day
            date(2024, 6, 19),  # Juneteenth
            date(2024, 7, 4),   # Independence Day
            date(2024, 9, 2),   # Labor Day
            date(2024, 11, 28), # Thanksgiving
            date(2024, 12, 25), # Christmas Day
            date(2025, 1, 1),   # New Year's Day
            date(2025, 1, 20),  # Martin Luther King Jr. Day
            date(2025, 2, 17),  # Presidents' Day
            date(2025, 4, 18),  # Good Friday
            date(2025, 5, 26),  # Memorial Day
            date(2025, 6, 19),  # Juneteenth
            date(2025, 7, 4),   # Independence Day
            date(2025, 9, 1),   # Labor Day
            date(2025, 11, 27), # Thanksgiving
            date(2025, 12, 25), # Christmas Day
        ]
        
        logger.info("Enhanced Date Utils initialized")
    
    def format_date(self, date_obj: Union[datetime, date], format_type: str = 'iso', 
                   timezone: str = None) -> str:
        """
        Format date with specified format
        
        Args:
            date_obj: Date or datetime object
            format_type: Format type from self.date_formats
            timezone: Target timezone (optional)
            
        Returns:
            Formatted date string
        """
        if timezone:
            if isinstance(date_obj, datetime):
                tz = pytz.timezone(timezone)
                if date_obj.tzinfo is None:
                    date_obj = self.default_timezone.localize(date_obj)
                date_obj = date_obj.astimezone(tz)
            else:
                # Convert date to datetime in specified timezone
                tz = pytz.timezone(timezone)
                date_obj = tz.localize(datetime.combine(date_obj, time.min))
        
        if format_type in self.date_formats:
            return date_obj.strftime(self.date_formats[format_type])
        else:
            return date_obj.strftime(format_type)
    
    def parse_date(self, date_string: str, format_type: str = 'auto', 
                  timezone: str = None) -> datetime:
        """
        Parse date string to datetime object
        
        Args:
            date_string: Date string to parse
            format_type: Format type or 'auto' for automatic detection
            timezone: Target timezone (optional)
            
        Returns:
            Parsed datetime object
        """
        if format_type == 'auto':
            # Try common formats
            formats_to_try = [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%B %d, %Y',
                '%b %d, %Y',
                '%d-%m-%Y',
                '%m-%d-%Y'
            ]
            
            for fmt in formats_to_try:
                try:
                    parsed_date = datetime.strptime(date_string, fmt)
                    if timezone:
                        tz = pytz.timezone(timezone)
                        parsed_date = tz.localize(parsed_date)
                    return parsed_date
                except ValueError:
                    continue
            
            raise ValueError(f"Unable to parse date: {date_string}")
        else:
            if format_type in self.date_formats:
                format_string = self.date_formats[format_type]
            else:
                format_string = format_type
            
            parsed_date = datetime.strptime(date_string, format_string)
            if timezone:
                tz = pytz.timezone(timezone)
                parsed_date = tz.localize(parsed_date)
            return parsed_date
    
    def get_date_range(self, start_date: Union[datetime, date], 
                      end_date: Union[datetime, date],
                      include_weekends: bool = True,
                      include_holidays: bool = True) -> DateRange:
        """
        Get comprehensive date range information
        
        Args:
            start_date: Start date
            end_date: End date
            include_weekends: Include weekends in calculation
            include_holidays: Include holidays in calculation
            
        Returns:
            DateRange object with comprehensive information
        """
        if isinstance(start_date, date):
            start_date = datetime.combine(start_date, time.min)
        if isinstance(end_date, date):
            end_date = datetime.combine(end_date, time.min)
        
        # Calculate duration
        duration = end_date - start_date
        duration_days = duration.days
        duration_weeks = duration_days / 7
        duration_months = duration_days / 30.44  # Average month length
        
        # Calculate business days
        business_days = 0
        weekends = 0
        holidays = 0
        
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            # Check if weekend
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                weekends += 1
            else:
                business_days += 1
                
                # Check if holiday
                if not include_holidays and current_date in self.us_holidays:
                    holidays += 1
                    business_days -= 1
            
            current_date += timedelta(days=1)
        
        trading_days = business_days - holidays if include_holidays else business_days
        
        return DateRange(
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            duration_weeks=duration_weeks,
            duration_months=duration_months,
            business_days=business_days,
            weekends=weekends,
            holidays=holidays,
            trading_days=trading_days
        )
    
    def get_fiscal_period(self, date_obj: Union[datetime, date], 
                         fiscal_year_start: int = 4) -> FiscalPeriod:
        """
        Get fiscal period information
        
        Args:
            date_obj: Date to analyze
            fiscal_year_start: Month when fiscal year starts (1-12)
            
        Returns:
            FiscalPeriod object
        """
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        year = date_obj.year
        month = date_obj.month
        
        # Determine fiscal year
        if month >= fiscal_year_start:
            fiscal_year = year
        else:
            fiscal_year = year - 1
        
        # Determine quarter
        fiscal_month = ((month - fiscal_year_start) % 12) + 1
        quarter = ((fiscal_month - 1) // 3) + 1
        
        # Calculate period start and end
        period_start_month = ((quarter - 1) * 3) + fiscal_year_start
        if period_start_month > 12:
            period_start_month -= 12
            period_start_year = fiscal_year + 1
        else:
            period_start_year = fiscal_year
        
        period_start = date(period_start_year, period_start_month, 1)
        
        # Calculate period end
        if quarter == 4:
            period_end = date(fiscal_year + 1, fiscal_year_start - 1, 
                            calendar.monthrange(fiscal_year + 1, fiscal_year_start - 1)[1])
        else:
            period_end_month = period_start_month + 2
            if period_end_month > 12:
                period_end_month -= 12
                period_end_year = period_start_year + 1
            else:
                period_end_year = period_start_year
            
            period_end = date(period_end_year, period_end_month, 
                            calendar.monthrange(period_end_year, period_end_month)[1])
        
        # Calculate days in period
        days_in_period = (period_end - period_start).days + 1
        
        # Calculate business days in period
        business_days_in_period = 0
        current_date = period_start
        while current_date <= period_end:
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                business_days_in_period += 1
            current_date += timedelta(days=1)
        
        period_name = f"Q{quarter} FY{fiscal_year}"
        
        return FiscalPeriod(
            fiscal_year=fiscal_year,
            quarter=quarter,
            period_start=period_start,
            period_end=period_end,
            period_name=period_name,
            days_in_period=days_in_period,
            business_days_in_period=business_days_in_period
        )
    
    def is_market_open(self, market: str, date_time: datetime = None) -> bool:
        """
        Check if market is open at given time
        
        Args:
            market: Market name (NYSE, NASDAQ, LSE, etc.)
            date_time: Date and time to check (default: now)
            
        Returns:
            True if market is open
        """
        if date_time is None:
            date_time = datetime.now()
        
        if market not in self.market_hours:
            logger.warning(f"Unknown market: {market}")
            return False
        
        market_info = self.market_hours[market]
        
        # Convert to market timezone
        market_tz = pytz.timezone(market_info.timezone)
        if date_time.tzinfo is None:
            date_time = self.default_timezone.localize(date_time)
        market_time = date_time.astimezone(market_tz)
        
        # Check if it's a trading day
        weekday = market_time.strftime('%A')
        if weekday not in market_info.trading_days:
            return False
        
        # Check if it's a holiday
        if market_time.date() in market_info.holidays:
            return False
        
        # Check if it's within trading hours
        if market_info.is_24h:
            return True
        
        current_time = market_time.time()
        return market_info.open_time <= current_time <= market_info.close_time
    
    def get_next_trading_day(self, market: str, date_obj: Union[datetime, date] = None) -> date:
        """
        Get next trading day for given market
        
        Args:
            market: Market name
            date_obj: Starting date (default: today)
            
        Returns:
            Next trading day
        """
        if date_obj is None:
            date_obj = date.today()
        elif isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        if market not in self.market_hours:
            logger.warning(f"Unknown market: {market}")
            return date_obj
        
        market_info = self.market_hours[market]
        
        # Start from next day
        current_date = date_obj + timedelta(days=1)
        
        # Find next trading day
        while True:
            weekday = current_date.strftime('%A')
            
            # Check if it's a trading day and not a holiday
            if (weekday in market_info.trading_days and 
                current_date not in market_info.holidays):
                return current_date
            
            current_date += timedelta(days=1)
    
    def get_previous_trading_day(self, market: str, date_obj: Union[datetime, date] = None) -> date:
        """
        Get previous trading day for given market
        
        Args:
            market: Market name
            date_obj: Starting date (default: today)
            
        Returns:
            Previous trading day
        """
        if date_obj is None:
            date_obj = date.today()
        elif isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        if market not in self.market_hours:
            logger.warning(f"Unknown market: {market}")
            return date_obj
        
        market_info = self.market_hours[market]
        
        # Start from previous day
        current_date = date_obj - timedelta(days=1)
        
        # Find previous trading day
        while True:
            weekday = current_date.strftime('%A')
            
            # Check if it's a trading day and not a holiday
            if (weekday in market_info.trading_days and 
                current_date not in market_info.holidays):
                return current_date
            
            current_date -= timedelta(days=1)
    
    def generate_date_series(self, start_date: Union[datetime, date], 
                           end_date: Union[datetime, date],
                           frequency: str = 'D',
                           include_weekends: bool = True,
                           include_holidays: bool = True) -> pd.DatetimeIndex:
        """
        Generate date series with specified frequency
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Frequency ('D', 'W', 'M', 'Q', 'Y', 'B' for business days)
            include_weekends: Include weekends
            include_holidays: Include holidays
            
        Returns:
            DatetimeIndex with date series
        """
        if isinstance(start_date, date):
            start_date = datetime.combine(start_date, time.min)
        if isinstance(end_date, date):
            end_date = datetime.combine(end_date, time.min)
        
        # Generate base series
        if frequency == 'B' or (not include_weekends):
            # Business days only
            date_range = pd.bdate_range(start=start_date, end=end_date, freq='B')
        else:
            date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Filter out holidays if requested
        if not include_holidays:
            date_range = date_range[~date_range.date.isin(self.us_holidays)]
        
        return date_range
    
    def get_timezone_info(self, timezone: str) -> Dict[str, Any]:
        """
        Get timezone information
        
        Args:
            timezone: Timezone name
            
        Returns:
            Dictionary with timezone information
        """
        try:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
            
            return {
                'timezone': timezone,
                'utc_offset': now.strftime('%z'),
                'dst_active': now.dst() != timedelta(0),
                'current_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'utc_time': now.utctimetuple()
            }
        except Exception as e:
            logger.error(f"Invalid timezone: {timezone}")
            return {
                'timezone': timezone,
                'error': str(e)
            }
    
    def convert_timezone(self, date_time: datetime, from_tz: str, to_tz: str) -> datetime:
        """
        Convert datetime from one timezone to another
        
        Args:
            date_time: Datetime object
            from_tz: Source timezone
            to_tz: Target timezone
            
        Returns:
            Converted datetime
        """
        from_timezone = pytz.timezone(from_tz)
        to_timezone = pytz.timezone(to_tz)
        
        if date_time.tzinfo is None:
            date_time = from_timezone.localize(date_time)
        
        return date_time.astimezone(to_timezone)
    
    def get_week_info(self, date_obj: Union[datetime, date]) -> Dict[str, Any]:
        """
        Get comprehensive week information
        
        Args:
            date_obj: Date to analyze
            
        Returns:
            Dictionary with week information
        """
        if isinstance(date_obj, datetime):
            date_obj = date_obj.date()
        
        # Get ISO week info
        iso_year, iso_week, iso_weekday = date_obj.isocalendar()
        
        # Get week start and end
        week_start = date_obj - timedelta(days=iso_weekday - 1)
        week_end = week_start + timedelta(days=6)
        
        # Get month info
        month_start = date_obj.replace(day=1)
        if date_obj.month == 12:
            month_end = date_obj.replace(year=date_obj.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = date_obj.replace(month=date_obj.month + 1, day=1) - timedelta(days=1)
        
        return {
            'date': date_obj,
            'iso_year': iso_year,
            'iso_week': iso_week,
            'iso_weekday': iso_weekday,
            'weekday_name': date_obj.strftime('%A'),
            'weekday_short': date_obj.strftime('%a'),
            'week_start': week_start,
            'week_end': week_end,
            'month_start': month_start,
            'month_end': month_end,
            'month_name': date_obj.strftime('%B'),
            'month_short': date_obj.strftime('%b'),
            'year': date_obj.year,
            'month': date_obj.month,
            'day': date_obj.day,
            'quarter': (date_obj.month - 1) // 3 + 1,
            'day_of_year': date_obj.timetuple().tm_yday,
            'days_in_month': calendar.monthrange(date_obj.year, date_obj.month)[1],
            'is_weekend': date_obj.weekday() >= 5,
            'is_month_start': date_obj.day == 1,
            'is_month_end': date_obj == month_end,
            'is_quarter_start': date_obj.month in [1, 4, 7, 10] and date_obj.day == 1,
            'is_year_start': date_obj.month == 1 and date_obj.day == 1,
            'is_year_end': date_obj.month == 12 and date_obj.day == 31
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced date utils
    date_utils = EnhancedDateUtils()
    
    print("📅 Enhanced Date Utils Test")
    print("=" * 50)
    
    # Test date formatting
    now = datetime.now()
    print(f"\n📊 Date Formatting:")
    print(f"ISO: {date_utils.format_date(now, 'iso')}")
    print(f"US: {date_utils.format_date(now, 'us')}")
    print(f"EU: {date_utils.format_date(now, 'eu')}")
    print(f"Readable: {date_utils.format_date(now, 'readable')}")
    print(f"Timestamp: {date_utils.format_date(now, 'timestamp')}")
    
    # Test date range
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    date_range = date_utils.get_date_range(start_date, end_date)
    print(f"\n📈 Date Range Analysis:")
    print(f"Duration: {date_range.duration_days} days")
    print(f"Business Days: {date_range.business_days}")
    print(f"Weekends: {date_range.weekends}")
    print(f"Trading Days: {date_range.trading_days}")
    
    # Test fiscal period
    fiscal_period = date_utils.get_fiscal_period(now)
    print(f"\n🏢 Fiscal Period:")
    print(f"Fiscal Year: {fiscal_period.fiscal_year}")
    print(f"Quarter: {fiscal_period.quarter}")
    print(f"Period: {fiscal_period.period_name}")
    print(f"Start: {fiscal_period.period_start}")
    print(f"End: {fiscal_period.period_end}")
    
    # Test market hours
    print(f"\n🏪 Market Hours:")
    markets = ['NYSE', 'NASDAQ', 'LSE', 'TSE', 'NSE']
    for market in markets:
        is_open = date_utils.is_market_open(market)
        print(f"{market}: {'🟢 Open' if is_open else '🔴 Closed'}")
    
    # Test timezone conversion
    print(f"\n🌍 Timezone Conversion:")
    ny_time = date_utils.convert_timezone(now, 'UTC', 'America/New_York')
    tokyo_time = date_utils.convert_timezone(now, 'UTC', 'Asia/Tokyo')
    print(f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"NY: {ny_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Tokyo: {tokyo_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Test week info
    week_info = date_utils.get_week_info(now)
    print(f"\n📅 Week Information:")
    print(f"Week: {week_info['iso_week']} of {week_info['iso_year']}")
    print(f"Weekday: {week_info['weekday_name']}")
    print(f"Week Start: {week_info['week_start']}")
    print(f"Week End: {week_info['week_end']}")
    print(f"Quarter: {week_info['quarter']}")
    print(f"Day of Year: {week_info['day_of_year']}")
    
    # Test date series generation
    print(f"\n📊 Date Series Generation:")
    business_days = date_utils.generate_date_series(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        frequency='B',
        include_weekends=False
    )
    print(f"Business days in January 2024: {len(business_days)}")
    print(f"First 5 business days: {business_days[:5].strftime('%Y-%m-%d').tolist()}")
    
    print("\n✅ Enhanced Date Utils test completed!")
