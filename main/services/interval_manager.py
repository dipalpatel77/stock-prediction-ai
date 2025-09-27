#!/usr/bin/env python3
"""
Interval Manager Service
Handles sophisticated interval selection for different prediction horizons
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)

class PredictionHorizon(Enum):
    """Prediction horizon enumeration"""
    INTRADAY = "intraday"      # 1-24 hours
    SHORT_TERM = "short_term"  # 1-6 weeks  
    MEDIUM_TERM = "medium_term" # 1-6 months
    LONG_TERM = "long_term"    # 6+ months

class IntervalStrategy(Enum):
    """Interval strategy enumeration"""
    MINUTE_BASED = "minute_based"    # Use minute intervals
    DAILY_BASED = "daily_based"      # Use daily intervals
    MIXED = "mixed"                  # Use combination of intervals

class IntervalManager:
    """
    Sophisticated interval manager for multi-timeframe predictions
    
    This service provides:
    - Intelligent interval selection based on prediction horizon
    - Multi-timeframe data aggregation
    - Interval-specific feature engineering
    - Prediction horizon optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Interval Manager
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = {}
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.IntervalManager")
        
        # Interval strategies for different horizons
        self.horizon_strategies = {
            PredictionHorizon.INTRADAY: {
                'primary_intervals': ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE'],
                'lookback_days': 30,
                'strategy': IntervalStrategy.MINUTE_BASED,
                'description': 'Intraday predictions using minute-level data'
            },
            PredictionHorizon.SHORT_TERM: {
                'primary_intervals': ['ONE_DAY'],
                'lookback_days': 200,
                'strategy': IntervalStrategy.DAILY_BASED,
                'description': 'Short-term predictions using daily data from last 200 days'
            },
            PredictionHorizon.MEDIUM_TERM: {
                'primary_intervals': ['ONE_DAY'],
                'lookback_days': 500,
                'strategy': IntervalStrategy.DAILY_BASED,
                'description': 'Medium-term predictions using daily data from last 500 days'
            },
            PredictionHorizon.LONG_TERM: {
                'primary_intervals': ['ONE_DAY'],
                'lookback_days': 2000,
                'strategy': IntervalStrategy.DAILY_BASED,
                'description': 'Long-term predictions using daily data from last 2000 days'
            }
        }
        
        # Mixed strategy configuration
        self.mixed_strategy_config = {
            'intraday_weight': 0.3,    # 30% weight for intraday patterns
            'daily_weight': 0.7,       # 70% weight for daily patterns
            'minute_intervals': ['ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE'],
            'daily_intervals': ['ONE_DAY']
        }
        
        self.logger.info("Interval Manager initialized with sophisticated multi-horizon strategies")
    
    def get_optimal_intervals(self, horizon: PredictionHorizon, 
                            available_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get optimal intervals for a specific prediction horizon
        
        Args:
            horizon: Prediction horizon
            available_data: Available data intervals
            
        Returns:
            Dictionary with optimal interval configuration
        """
        try:
            if horizon not in self.horizon_strategies:
                raise ValueError(f"Unknown prediction horizon: {horizon}")
            
            strategy_config = self.horizon_strategies[horizon]
            
            # Check if we have the required data
            required_intervals = strategy_config['primary_intervals']
            available_intervals = []
            
            if available_data:
                available_intervals = list(available_data.keys())
            
            # Filter to only available intervals
            optimal_intervals = [interval for interval in required_intervals 
                               if interval in available_intervals] if available_data else required_intervals
            
            if not optimal_intervals:
                self.logger.warning(f"No optimal intervals available for {horizon}, using fallback")
                optimal_intervals = ['ONE_DAY']  # Fallback to daily data
            
            result = {
                'horizon': horizon,
                'intervals': optimal_intervals,
                'lookback_days': strategy_config['lookback_days'],
                'strategy': strategy_config['strategy'],
                'description': strategy_config['description'],
                'available_intervals': available_intervals,
                'confidence': self._calculate_interval_confidence(optimal_intervals, required_intervals)
            }
            
            self.logger.info(f"Optimal intervals for {horizon}: {optimal_intervals}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get optimal intervals for {horizon}: {e}")
            return {
                'horizon': horizon,
                'intervals': ['ONE_DAY'],
                'lookback_days': 200,
                'strategy': IntervalStrategy.DAILY_BASED,
                'description': 'Fallback to daily data',
                'confidence': 0.5
            }
    
    def get_mixed_strategy_intervals(self, available_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get intervals for mixed strategy (combining multiple timeframes)
        
        Args:
            available_data: Available data intervals
            
        Returns:
            Mixed strategy configuration
        """
        try:
            minute_intervals = [interval for interval in self.mixed_strategy_config['minute_intervals']
                              if interval in available_data]
            daily_intervals = [interval for interval in self.mixed_strategy_config['daily_intervals']
                             if interval in available_data]
            
            if not minute_intervals and not daily_intervals:
                self.logger.warning("No suitable intervals for mixed strategy")
                return {
                    'strategy': IntervalStrategy.DAILY_BASED,
                    'intervals': ['ONE_DAY'],
                    'weights': {'ONE_DAY': 1.0},
                    'confidence': 0.3
                }
            
            # Calculate weights based on available data
            weights = {}
            total_weight = 0
            
            if minute_intervals:
                minute_weight = self.mixed_strategy_config['intraday_weight'] / len(minute_intervals)
                for interval in minute_intervals:
                    weights[interval] = minute_weight
                    total_weight += minute_weight
            
            if daily_intervals:
                daily_weight = self.mixed_strategy_config['daily_weight'] / len(daily_intervals)
                for interval in daily_intervals:
                    weights[interval] = daily_weight
                    total_weight += daily_weight
            
            # Normalize weights
            if total_weight > 0:
                for interval in weights:
                    weights[interval] /= total_weight
            
            result = {
                'strategy': IntervalStrategy.MIXED,
                'intervals': minute_intervals + daily_intervals,
                'weights': weights,
                'minute_intervals': minute_intervals,
                'daily_intervals': daily_intervals,
                'confidence': self._calculate_mixed_confidence(minute_intervals, daily_intervals)
            }
            
            self.logger.info(f"Mixed strategy intervals: {result['intervals']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get mixed strategy intervals: {e}")
            return {
                'strategy': IntervalStrategy.DAILY_BASED,
                'intervals': ['ONE_DAY'],
                'weights': {'ONE_DAY': 1.0},
                'confidence': 0.3
            }
    
    def aggregate_multi_interval_data(self, data_dict: Dict[str, pd.DataFrame], 
                                    strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Aggregate data from multiple intervals based on strategy
        
        Args:
            data_dict: Dictionary of interval data
            strategy_config: Strategy configuration
            
        Returns:
            Aggregated DataFrame
        """
        try:
            if strategy_config['strategy'] == IntervalStrategy.MIXED:
                return self._aggregate_mixed_strategy_data(data_dict, strategy_config)
            else:
                return self._aggregate_single_strategy_data(data_dict, strategy_config)
                
        except Exception as e:
            self.logger.error(f"Failed to aggregate multi-interval data: {e}")
            # Return the most reliable data available
            if 'ONE_DAY' in data_dict:
                return data_dict['ONE_DAY']
            elif data_dict:
                return next(iter(data_dict.values()))
            else:
                return pd.DataFrame()
    
    def _aggregate_mixed_strategy_data(self, data_dict: Dict[str, pd.DataFrame], 
                                     strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate data using mixed strategy"""
        try:
            weights = strategy_config.get('weights', {})
            aggregated_data = None
            
            for interval, data in data_dict.items():
                if interval in weights and not data.empty:
                    weight = weights[interval]
                    
                    # Resample to common frequency (daily)
                    if interval != 'ONE_DAY':
                        data_resampled = self._resample_to_daily(data)
                    else:
                        data_resampled = data.copy()
                    
                    if aggregated_data is None:
                        aggregated_data = data_resampled * weight
                    else:
                        # Align indices and add weighted data
                        common_index = aggregated_data.index.intersection(data_resampled.index)
                        if len(common_index) > 0:
                            aggregated_data.loc[common_index] += data_resampled.loc[common_index] * weight
            
            return aggregated_data if aggregated_data is not None else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate mixed strategy data: {e}")
            return pd.DataFrame()
    
    def _aggregate_single_strategy_data(self, data_dict: Dict[str, pd.DataFrame], 
                                       strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate data using single strategy"""
        try:
            intervals = strategy_config.get('intervals', ['ONE_DAY'])
            
            # Use the first available interval
            for interval in intervals:
                if interval in data_dict and not data_dict[interval].empty:
                    return data_dict[interval]
            
            # Fallback to any available data
            if data_dict:
                return next(iter(data_dict.values()))
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to aggregate single strategy data: {e}")
            return pd.DataFrame()
    
    def _resample_to_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample minute/hour data to daily data"""
        try:
            if data.empty:
                return data
            
            # Ensure we have a datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'Date' in data.columns:
                    data = data.set_index('Date')
                elif 'date' in data.columns:
                    data = data.set_index('date')
                else:
                    return data
            
            # Resample to daily (OHLCV)
            daily_data = data.resample('D').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            return daily_data
            
        except Exception as e:
            self.logger.error(f"Failed to resample to daily: {e}")
            return data
    
    def _calculate_interval_confidence(self, optimal_intervals: List[str], 
                                     required_intervals: List[str]) -> float:
        """Calculate confidence based on interval availability"""
        try:
            if not required_intervals:
                return 0.5
            
            available_ratio = len(optimal_intervals) / len(required_intervals)
            
            # Boost confidence for having the primary interval
            if 'ONE_DAY' in optimal_intervals:
                available_ratio += 0.2
            
            return min(1.0, available_ratio)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate interval confidence: {e}")
            return 0.5
    
    def _calculate_mixed_confidence(self, minute_intervals: List[str], 
                                   daily_intervals: List[str]) -> float:
        """Calculate confidence for mixed strategy"""
        try:
            minute_score = len(minute_intervals) / len(self.mixed_strategy_config['minute_intervals'])
            daily_score = len(daily_intervals) / len(self.mixed_strategy_config['daily_intervals'])
            
            # Weighted average
            confidence = (minute_score * 0.3 + daily_score * 0.7)
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate mixed confidence: {e}")
            return 0.5
    
    def get_prediction_horizon_from_days(self, prediction_days: int) -> PredictionHorizon:
        """
        Determine prediction horizon from number of days
        
        Args:
            prediction_days: Number of days to predict
            
        Returns:
            Appropriate prediction horizon
        """
        try:
            if prediction_days <= 1:
                return PredictionHorizon.INTRADAY
            elif prediction_days <= 42:  # 6 weeks
                return PredictionHorizon.SHORT_TERM
            elif prediction_days <= 180:  # 6 months
                return PredictionHorizon.MEDIUM_TERM
            else:
                return PredictionHorizon.LONG_TERM
                
        except Exception as e:
            self.logger.error(f"Failed to determine prediction horizon: {e}")
            return PredictionHorizon.SHORT_TERM
    
    def get_feature_engineering_strategy(self, horizon: PredictionHorizon) -> Dict[str, Any]:
        """
        Get feature engineering strategy for specific horizon
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Feature engineering configuration
        """
        try:
            strategies = {
                PredictionHorizon.INTRADAY: {
                    'technical_indicators': ['RSI', 'MACD', 'Bollinger_Bands', 'Stochastic'],
                    'time_features': ['hour', 'minute', 'day_of_week'],
                    'volatility_features': True,
                    'momentum_features': True,
                    'volume_features': True
                },
                PredictionHorizon.SHORT_TERM: {
                    'technical_indicators': ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger_Bands'],
                    'time_features': ['day_of_week', 'month', 'quarter'],
                    'volatility_features': True,
                    'momentum_features': True,
                    'volume_features': True
                },
                PredictionHorizon.MEDIUM_TERM: {
                    'technical_indicators': ['SMA', 'EMA', 'RSI', 'MACD'],
                    'time_features': ['month', 'quarter', 'year'],
                    'volatility_features': True,
                    'momentum_features': True,
                    'volume_features': False
                },
                PredictionHorizon.LONG_TERM: {
                    'technical_indicators': ['SMA', 'EMA'],
                    'time_features': ['quarter', 'year'],
                    'volatility_features': False,
                    'momentum_features': True,
                    'volume_features': False
                }
            }
            
            return strategies.get(horizon, strategies[PredictionHorizon.SHORT_TERM])
            
        except Exception as e:
            self.logger.error(f"Failed to get feature engineering strategy: {e}")
            return {
                'technical_indicators': ['SMA', 'RSI'],
                'time_features': ['day_of_week'],
                'volatility_features': True,
                'momentum_features': True,
                'volume_features': True
            }

