#!/usr/bin/env python3
"""
Data Periods Configuration
Configure historical data download periods for different analysis types
"""

# =============================================================================
# DATA DOWNLOAD PERIODS CONFIGURATION
# =============================================================================

# Default periods for different analysis types
DEFAULT_PERIODS = {
    # Standard analysis periods
    'quick_analysis': '3mo',      # 3 months - for quick checks
    'short_term': '6mo',          # 6 months - for short-term analysis
    'medium_term': '1y',          # 1 year - for medium-term analysis (RECOMMENDED)
    'long_term': '2y',            # 2 years - for comprehensive analysis
    'extensive': '5y',            # 5 years - for extensive historical analysis
    'maximum': 'max'              # Maximum available data
}

# Periods for different data sources
SOURCE_PERIODS = {
    'angel_one': {
        'default': '6mo',         # 6 months for Indian stocks (faster API)
        'short_term': '3mo',      # 3 months
        'medium_term': '1y',      # 1 year
        'long_term': '2y'         # 2 years (max recommended for Angel One)
    },
    'yfinance': {
        'default': '1y',          # 1 year for international stocks
        'short_term': '6mo',      # 6 months
        'medium_term': '2y',      # 2 years
        'long_term': '5y',        # 5 years
        'extensive': '10y'        # 10 years
    }
}

# Periods for different stock types
STOCK_TYPE_PERIODS = {
    'indian_stocks': {
        'default': '6mo',         # 6 months (Angel One optimized)
        'short_term': '3mo',      # 3 months
        'medium_term': '1y',      # 1 year
        'long_term': '2y'         # 2 years
    },
    'international_stocks': {
        'default': '1y',          # 1 year (yfinance optimized)
        'short_term': '6mo',      # 6 months
        'medium_term': '2y',      # 2 years
        'long_term': '5y'         # 5 years
    },
    'crypto': {
        'default': '1y',          # 1 year
        'short_term': '3mo',      # 3 months
        'medium_term': '2y',      # 2 years
        'long_term': '5y'         # 5 years
    }
}

# Periods for different analysis types
ANALYSIS_PERIODS = {
    'technical_analysis': {
        'default': '6mo',         # 6 months (sufficient for most indicators)
        'short_term': '3mo',      # 3 months
        'medium_term': '1y',      # 1 year
        'long_term': '2y'         # 2 years
    },
    'fundamental_analysis': {
        'default': '2y',          # 2 years (for financial ratios)
        'short_term': '1y',       # 1 year
        'medium_term': '3y',      # 3 years
        'long_term': '5y'         # 5 years
    },
    'sentiment_analysis': {
        'default': '3mo',         # 3 months (recent sentiment)
        'short_term': '1mo',      # 1 month
        'medium_term': '6mo',     # 6 months
        'long_term': '1y'         # 1 year
    },
    'prediction_models': {
        'default': '1y',          # 1 year (balanced for ML models)
        'short_term': '6mo',      # 6 months
        'medium_term': '2y',      # 2 years
        'long_term': '3y'         # 3 years
    }
}

# =============================================================================
# RECOMMENDED SETTINGS
# =============================================================================

# RECOMMENDED: Balanced approach for most users
RECOMMENDED_PERIODS = {
    'default': '1y',              # 1 year - balanced performance and data
    'angel_one': '6mo',           # 6 months - faster for Indian stocks
    'yfinance': '1y',             # 1 year - good for international stocks
    'quick_check': '3mo',         # 3 months - for quick analysis
    'comprehensive': '2y'         # 2 years - for detailed analysis
}

# PERFORMANCE OPTIMIZED: For faster processing
PERFORMANCE_PERIODS = {
    'default': '6mo',             # 6 months - faster processing
    'angel_one': '3mo',           # 3 months - very fast for Indian stocks
    'yfinance': '6mo',            # 6 months - faster for international
    'quick_check': '1mo',         # 1 month - very quick analysis
    'comprehensive': '1y'         # 1 year - still comprehensive but faster
}

# COMPREHENSIVE: For maximum data analysis
COMPREHENSIVE_PERIODS = {
    'default': '2y',              # 2 years - maximum data
    'angel_one': '1y',            # 1 year - max recommended for Angel One
    'yfinance': '3y',             # 3 years - extensive for international
    'quick_check': '6mo',         # 6 months - still quick but more data
    'comprehensive': '5y'         # 5 years - maximum comprehensive analysis
}

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_period_config(config_type: str = 'recommended') -> dict:
    """
    Get period configuration based on type.
    
    Args:
        config_type: 'recommended', 'performance', 'comprehensive', or 'custom'
        
    Returns:
        Dictionary with period configuration
    """
    configs = {
        'recommended': RECOMMENDED_PERIODS,
        'performance': PERFORMANCE_PERIODS,
        'comprehensive': COMPREHENSIVE_PERIODS
    }
    
    return configs.get(config_type, RECOMMENDED_PERIODS)

def get_period_for_analysis(analysis_type: str, stock_type: str = 'general') -> str:
    """
    Get appropriate period for specific analysis type.
    
    Args:
        analysis_type: Type of analysis ('technical', 'fundamental', 'sentiment', 'prediction')
        stock_type: Type of stock ('indian', 'international', 'crypto')
        
    Returns:
        Recommended period string
    """
    # Get analysis-specific periods
    analysis_periods = ANALYSIS_PERIODS.get(analysis_type, ANALYSIS_PERIODS['technical_analysis'])
    
    # Get stock-type specific periods
    stock_periods = STOCK_TYPE_PERIODS.get(f'{stock_type}_stocks', STOCK_TYPE_PERIODS['international_stocks'])
    
    # Return the more conservative (longer) period
    default_analysis = analysis_periods['default']
    default_stock = stock_periods['default']
    
    # Convert to days for comparison
    period_days = {
        '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '3y': 1095, '5y': 1825, '10y': 3650
    }
    
    analysis_days = period_days.get(default_analysis, 365)
    stock_days = period_days.get(default_stock, 365)
    
    return default_analysis if analysis_days >= stock_days else default_stock

def get_period_for_source(data_source: str, config_type: str = 'recommended') -> str:
    """
    Get appropriate period for specific data source.
    
    Args:
        data_source: 'angel_one' or 'yfinance'
        config_type: Configuration type
        
    Returns:
        Recommended period string
    """
    config = get_period_config(config_type)
    
    if data_source == 'angel_one':
        return config.get('angel_one', '6mo')
    elif data_source == 'yfinance':
        return config.get('yfinance', '1y')
    else:
        return config.get('default', '1y')

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_period(period: str) -> bool:
    """
    Validate if a period string is valid.
    
    Args:
        period: Period string to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    return period in valid_periods

def get_period_days(period: str) -> int:
    """
    Convert period string to approximate number of days.
    
    Args:
        period: Period string
        
    Returns:
        Approximate number of days
    """
    period_mapping = {
        '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
        '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
        'ytd': 365, 'max': 3650  # Approximations
    }
    
    return period_mapping.get(period, 365)

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("📊 Data Periods Configuration")
    print("=" * 50)
    
    # Example 1: Get recommended periods
    print("\n🎯 Recommended Periods:")
    recommended = get_period_config('recommended')
    for key, value in recommended.items():
        print(f"   {key}: {value}")
    
    # Example 2: Get period for specific analysis
    print("\n🔍 Analysis-Specific Periods:")
    analyses = ['technical_analysis', 'fundamental_analysis', 'sentiment_analysis', 'prediction_models']
    for analysis in analyses:
        period = get_period_for_analysis(analysis)
        print(f"   {analysis}: {period}")
    
    # Example 3: Get period for data source
    print("\n📡 Source-Specific Periods:")
    sources = ['angel_one', 'yfinance']
    for source in sources:
        period = get_period_for_source(source)
        print(f"   {source}: {period}")
    
    print("\n✅ Configuration loaded successfully!")
    print("💡 Use get_period_config() to get different configurations")
    print("💡 Use get_period_for_analysis() for analysis-specific periods")
    print("💡 Use get_period_for_source() for source-specific periods")
