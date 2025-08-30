#!/usr/bin/env python3
"""
Date Utilities for Enhanced Date Formatting
===========================================

This module provides enhanced date formatting utilities that include:
- Exact date information (day, month, year)
- Month names in full and abbreviated forms
- Day of week information
- Enhanced timestamp formats
- Date range calculations
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import calendar

class DateUtils:
    """Enhanced date formatting utilities"""
    
    # Month names mapping
    MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    MONTH_NAMES_SHORT = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    
    DAY_NAMES = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday"
    }
    
    DAY_NAMES_SHORT = {
        0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu",
        4: "Fri", 5: "Sat", 6: "Sun"
    }
    
    @staticmethod
    def get_enhanced_date_info(date_obj: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get comprehensive date information including exact date and month details.
        
        Args:
            date_obj: datetime object (defaults to current time)
            
        Returns:
            Dictionary with comprehensive date information
        """
        if date_obj is None:
            date_obj = datetime.now()
        
        return {
            # Basic date components
            'year': date_obj.year,
            'month': date_obj.month,
            'day': date_obj.day,
            'hour': date_obj.hour,
            'minute': date_obj.minute,
            'second': date_obj.second,
            
            # Month information
            'month_name': DateUtils.MONTH_NAMES[date_obj.month],
            'month_name_short': DateUtils.MONTH_NAMES_SHORT[date_obj.month],
            'month_number': date_obj.month,
            
            # Day information
            'day_name': DateUtils.DAY_NAMES[date_obj.weekday()],
            'day_name_short': DateUtils.DAY_NAMES_SHORT[date_obj.weekday()],
            'day_of_week': date_obj.weekday(),
            'day_of_year': date_obj.timetuple().tm_yday,
            
            # Quarter information
            'quarter': (date_obj.month - 1) // 3 + 1,
            
            # Week information
            'week_of_year': date_obj.isocalendar()[1],
            
            # Formatted strings
            'date_iso': date_obj.strftime('%Y-%m-%d'),
            'date_full': date_obj.strftime('%Y-%m-%d'),
            'date_with_month_name': date_obj.strftime('%d %B %Y'),
            'date_with_month_name_short': date_obj.strftime('%d %b %Y'),
            'date_with_day': date_obj.strftime('%A, %d %B %Y'),
            'date_with_day_short': date_obj.strftime('%a, %d %b %Y'),
            
            # Time information
            'time_24h': date_obj.strftime('%H:%M:%S'),
            'time_12h': date_obj.strftime('%I:%M:%S %p'),
            
            # Full timestamp formats
            'timestamp_iso': date_obj.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp_full': date_obj.strftime('%A, %d %B %Y at %I:%M:%S %p'),
            'timestamp_short': date_obj.strftime('%d %b %Y %H:%M:%S'),
            
            # Analysis specific formats
            'analysis_date': date_obj.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_date_full': date_obj.strftime('%A, %d %B %Y at %I:%M:%S %p'),
            'analysis_date_short': date_obj.strftime('%d %b %Y %H:%M'),
            
            # File naming formats
            'filename_date': date_obj.strftime('%Y%m%d_%H%M%S'),
            'filename_date_short': date_obj.strftime('%Y%m%d'),
            
            # Raw datetime object
            'datetime': date_obj
        }
    
    @staticmethod
    def format_prediction_date(base_date: datetime, days_ahead: int) -> Dict[str, Any]:
        """
        Format date for prediction output with exact date and month information.
        
        Args:
            base_date: Base date for calculation
            days_ahead: Number of days ahead to predict
            
        Returns:
            Dictionary with prediction date information
        """
        prediction_date = base_date + timedelta(days=days_ahead)
        date_info = DateUtils.get_enhanced_date_info(prediction_date)
        
        # Add prediction-specific information
        date_info.update({
            'days_ahead': days_ahead,
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'prediction_date_full': prediction_date.strftime('%A, %d %B %Y'),
            'prediction_date_short': prediction_date.strftime('%d %b %Y'),
            'prediction_month': prediction_date.strftime('%B %Y'),
            'prediction_month_short': prediction_date.strftime('%b %Y'),
            'prediction_week': prediction_date.isocalendar()[1],
            'prediction_quarter': (prediction_date.month - 1) // 3 + 1,
        })
        
        return date_info
    
    @staticmethod
    def format_week_prediction_date(base_date: datetime, weeks_ahead: int) -> Dict[str, Any]:
        """
        Format date for weekly prediction output.
        
        Args:
            base_date: Base date for calculation
            weeks_ahead: Number of weeks ahead to predict
            
        Returns:
            Dictionary with weekly prediction date information
        """
        prediction_date = base_date + timedelta(weeks=weeks_ahead)
        date_info = DateUtils.get_enhanced_date_info(prediction_date)
        
        # Add weekly prediction-specific information
        date_info.update({
            'weeks_ahead': weeks_ahead,
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'prediction_date_full': prediction_date.strftime('%A, %d %B %Y'),
            'prediction_week_range': f"Week {prediction_date.isocalendar()[1]}, {prediction_date.strftime('%B %Y')}",
            'prediction_week_short': f"W{prediction_date.isocalendar()[1]} {prediction_date.strftime('%b %Y')}",
        })
        
        return date_info
    
    @staticmethod
    def format_month_prediction_date(base_date: datetime, months_ahead: int) -> Dict[str, Any]:
        """
        Format date for monthly prediction output.
        
        Args:
            base_date: Base date for calculation
            months_ahead: Number of months ahead to predict
            
        Returns:
            Dictionary with monthly prediction date information
        """
        # Calculate month ahead (approximate)
        prediction_date = base_date + timedelta(days=months_ahead * 30)
        date_info = DateUtils.get_enhanced_date_info(prediction_date)
        
        # Add monthly prediction-specific information
        date_info.update({
            'months_ahead': months_ahead,
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'prediction_date_full': prediction_date.strftime('%A, %d %B %Y'),
            'prediction_month_full': prediction_date.strftime('%B %Y'),
            'prediction_month_short': prediction_date.strftime('%b %Y'),
            'prediction_quarter': f"Q{(prediction_date.month - 1) // 3 + 1} {prediction_date.year}",
        })
        
        return date_info
    
    @staticmethod
    def get_date_range_info(start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Get information about a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with date range information
        """
        date_diff = end_date - start_date
        days_diff = date_diff.days
        
        return {
            'start_date': DateUtils.get_enhanced_date_info(start_date),
            'end_date': DateUtils.get_enhanced_date_info(end_date),
            'days_difference': days_diff,
            'weeks_difference': days_diff // 7,
            'months_difference': days_diff // 30,
            'years_difference': days_diff // 365,
            'range_description': DateUtils._get_range_description(days_diff),
            'range_short': f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}",
            'range_full': f"{start_date.strftime('%A, %d %B %Y')} to {end_date.strftime('%A, %d %B %Y')}"
        }
    
    @staticmethod
    def _get_range_description(days: int) -> str:
        """Get human-readable description of date range."""
        if days == 0:
            return "Same day"
        elif days == 1:
            return "1 day"
        elif days < 7:
            return f"{days} days"
        elif days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks > 1 else ''}"
        elif days < 365:
            months = days // 30
            return f"{months} month{'s' if months > 1 else ''}"
        else:
            years = days // 365
            return f"{years} year{'s' if years > 1 else ''}"
    
    @staticmethod
    def format_analysis_timestamp() -> Dict[str, Any]:
        """
        Get formatted timestamp for analysis output.
        
        Returns:
            Dictionary with analysis timestamp information
        """
        return DateUtils.get_enhanced_date_info()
    
    @staticmethod
    def format_file_timestamp() -> str:
        """
        Get formatted timestamp for file naming.
        
        Returns:
            Formatted timestamp string for file names
        """
        return datetime.now().strftime('%Y%m%d_%H%M%S')

# Convenience functions for backward compatibility
def get_enhanced_date_info(date_obj: Optional[datetime] = None) -> Dict[str, Any]:
    """Convenience function to get enhanced date information."""
    return DateUtils.get_enhanced_date_info(date_obj)

def format_prediction_date(base_date: datetime, days_ahead: int) -> Dict[str, Any]:
    """Convenience function to format prediction date."""
    return DateUtils.format_prediction_date(base_date, days_ahead)

def format_analysis_timestamp() -> Dict[str, Any]:
    """Convenience function to format analysis timestamp."""
    return DateUtils.format_analysis_timestamp()

def format_file_timestamp() -> str:
    """Convenience function to format file timestamp."""
    return DateUtils.format_file_timestamp()
