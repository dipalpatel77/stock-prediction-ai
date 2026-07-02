"""
Date Formatter Utility
Enhanced date formatting for predictions and analysis
Based on unified_analysis_pipeline_backup.py date utilities
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_period_to_days(period: str) -> int:
    """Convert a yfinance-style period string to number of days."""
    mapping = {
        '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
        '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
        'ytd': 365, 'max': 2000,
    }
    return mapping.get(period.lower(), 365)


class DateFormatter:
    """
    Enhanced date formatter for predictions and analysis
    Provides comprehensive date formatting utilities
    """
    
    def __init__(self):
        """Initialize Date Formatter"""
        self.month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        self.day_names = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]
        
        self.day_names_short = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        logger.info("Date Formatter initialized")
    
    def format_analysis_timestamp(self) -> Dict[str, Any]:
        """
        Format analysis timestamp with enhanced date information
        
        Returns:
            Dictionary with formatted date information
        """
        try:
            now = datetime.now()
            
            return {
                'analysis_date': now.strftime('%Y-%m-%d'),
                'analysis_date_full': now.strftime('%Y-%m-%d %H:%M:%S'),
                'date_with_day': now.strftime('%A, %B %d, %Y'),
                'month_name': now.strftime('%B'),
                'month': now.month,
                'year': now.year,
                'day_name': now.strftime('%A'),
                'day_name_short': now.strftime('%a'),
                'timestamp': now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to format analysis timestamp: {e}")
            return {'error': str(e)}
    
    def format_prediction_date(self, base_date: datetime, days_ahead: int) -> Dict[str, Any]:
        """
        Format prediction date with enhanced information
        
        Args:
            base_date: Base date for calculation
            days_ahead: Number of days ahead
            
        Returns:
            Dictionary with formatted prediction date information
        """
        try:
            prediction_date = base_date + timedelta(days=days_ahead)
            
            return {
                'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                'prediction_date_short': prediction_date.strftime('%b %d'),
                'day_name': prediction_date.strftime('%A'),
                'day_name_short': prediction_date.strftime('%a'),
                'month_name': prediction_date.strftime('%B'),
                'month': prediction_date.month,
                'year': prediction_date.year,
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            logger.error(f"Failed to format prediction date: {e}")
            return {'error': str(e)}
    
    def format_week_prediction_date(self, base_date: datetime, weeks_ahead: int) -> Dict[str, Any]:
        """
        Format week prediction date
        
        Args:
            base_date: Base date for calculation
            weeks_ahead: Number of weeks ahead
            
        Returns:
            Dictionary with formatted week prediction date information
        """
        try:
            prediction_date = base_date + timedelta(weeks=weeks_ahead)
            
            return {
                'prediction_week': prediction_date.strftime('%Y-%m-%d'),
                'prediction_week_short': prediction_date.strftime('%b %d'),
                'day_name': prediction_date.strftime('%A'),
                'day_name_short': prediction_date.strftime('%a'),
                'month_name': prediction_date.strftime('%B'),
                'month': prediction_date.month,
                'year': prediction_date.year,
                'weeks_ahead': weeks_ahead
            }
            
        except Exception as e:
            logger.error(f"Failed to format week prediction date: {e}")
            return {'error': str(e)}
    
    def format_month_prediction_date(self, base_date: datetime, months_ahead: int) -> Dict[str, Any]:
        """
        Format month prediction date
        
        Args:
            base_date: Base date for calculation
            months_ahead: Number of months ahead
            
        Returns:
            Dictionary with formatted month prediction date information
        """
        try:
            # Simple month calculation (30 days per month)
            prediction_date = base_date + timedelta(days=months_ahead * 30)
            
            return {
                'prediction_month': prediction_date.strftime('%Y-%m-%d'),
                'prediction_month_short': prediction_date.strftime('%b %d'),
                'day_name': prediction_date.strftime('%A'),
                'day_name_short': prediction_date.strftime('%a'),
                'month_name': prediction_date.strftime('%B'),
                'month': prediction_date.month,
                'year': prediction_date.year,
                'months_ahead': months_ahead
            }
            
        except Exception as e:
            logger.error(f"Failed to format month prediction date: {e}")
            return {'error': str(e)}
    
    def format_date_range(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Format date range with enhanced information
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with formatted date range information
        """
        try:
            duration = end_date - start_date
            
            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'start_date_formatted': start_date.strftime('%B %d, %Y'),
                'end_date_formatted': end_date.strftime('%B %d, %Y'),
                'duration_days': duration.days,
                'duration_weeks': duration.days // 7,
                'duration_months': duration.days // 30,
                'duration_years': duration.days // 365
            }
            
        except Exception as e:
            logger.error(f"Failed to format date range: {e}")
            return {'error': str(e)}
    
    def get_relative_date(self, base_date: datetime, period: str) -> Dict[str, Any]:
        """
        Get relative date based on period
        
        Args:
            base_date: Base date
            period: Period string (1d, 5d, 1w, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            
        Returns:
            Dictionary with relative date information
        """
        try:
            period_mapping = {
                '1d': timedelta(days=1),
                '5d': timedelta(days=5),
                '1w': timedelta(weeks=1),
                '1mo': timedelta(days=30),
                '3mo': timedelta(days=90),
                '6mo': timedelta(days=180),
                '1y': timedelta(days=365),
                '2y': timedelta(days=730),
                '5y': timedelta(days=1825)
            }
            
            if period not in period_mapping:
                return {'error': f'Unknown period: {period}'}
            
            target_date = base_date + period_mapping[period]
            
            return {
                'base_date': base_date.strftime('%Y-%m-%d'),
                'target_date': target_date.strftime('%Y-%m-%d'),
                'period': period,
                'days_difference': period_mapping[period].days,
                'formatted_period': self._format_period_name(period)
            }
            
        except Exception as e:
            logger.error(f"Failed to get relative date: {e}")
            return {'error': str(e)}
    
    def _format_period_name(self, period: str) -> str:
        """Format period name for display"""
        try:
            period_names = {
                '1d': '1 day',
                '5d': '5 days',
                '1w': '1 week',
                '1mo': '1 month',
                '3mo': '3 months',
                '6mo': '6 months',
                '1y': '1 year',
                '2y': '2 years',
                '5y': '5 years'
            }
            
            return period_names.get(period, period)
            
        except Exception as e:
            logger.error(f"Failed to format period name: {e}")
            return period
    
    def get_date_formatter_status(self) -> Dict[str, Any]:
        """Get date formatter status"""
        try:
            return {
                'month_names': len(self.month_names),
                'day_names': len(self.day_names),
                'day_names_short': len(self.day_names_short),
                'initialized': True,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get date formatter status: {e}")
            return {'error': str(e)}


# Global instance for easy access
date_formatter = DateFormatter()

