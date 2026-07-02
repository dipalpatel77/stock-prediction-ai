#!/usr/bin/env python3
"""
Services Module
Essential services for the stock prediction pipeline
"""

from .data_service_wrapper import DataServiceWrapper
from .angel_one_manager import AngelOneManager
from .angel_one_service import AngelOneService
from .database_manager import DatabaseManager
from .api_coordinator import APICoordinator
from .technical_indicators_service import TechnicalIndicatorsService
from .feature_engineering_service import FeatureEngineeringService
from .currency_service import CurrencyService
from .fred_api_service import FREDAPIService
from .global_market_service import GlobalMarketService
# --- Incremental update system (three cooperating services) ---
# IncrementalDataService:      syncs raw OHLCV price/volume data from APIs
# IncrementalService:          manages ML model retraining cycles
# IncrementalUpdateService:    strategy coordinator — decides full vs incremental
from .incremental_data_service import IncrementalDataService
from .incremental_service import IncrementalService
from .incremental_update_service import IncrementalUpdateService
from .model_service import ModelService
from .report_generator import ReportGenerator
from .interval_manager import IntervalManager

__all__ = [
    'DataServiceWrapper',
    'AngelOneManager',
    'AngelOneService',
    'DatabaseManager',
    'APICoordinator',
    'TechnicalIndicatorsService',
    'FeatureEngineeringService',
    'CurrencyService',
    'FREDAPIService',
    'GlobalMarketService',
    'IncrementalDataService',
    'IncrementalService',
    'IncrementalUpdateService',
    'ModelService',
    'ReportGenerator',
    'IntervalManager'
]
