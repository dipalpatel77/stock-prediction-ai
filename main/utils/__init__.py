"""
Utils Module
Utility classes and functions for the polylithic pipeline

Exported below: pipeline-core utilities (always safe to import via `from main.utils import X`).

NOT exported here (import directly from their module when needed):
  console_formatter.py            — ANSI/emoji UI chrome, used by main.py
  enhanced_prediction_formatter.py — prediction result tables, used by prediction_generator.py
  date_formatter.py               — standalone date utilities
  simple_logger.py                — lightweight logger for standalone scripts (not pipeline)
  simple_error_handler.py         — lightweight error handler for standalone scripts (not pipeline)
  service_manager.py              — imported directly where needed
"""

from .pipeline_logger import PipelineLogger
from .error_handler import ErrorHandler
from .formatters import PriceFormatter, CurrencyFormatter
from .validators import DataValidator, ConfigValidator
# Service coordinator removed (over-engineered)
from .database_pool import get_connection_pool
from .rate_limiter import get_api_rate_limiter
from .model_cache import get_model_cache

__all__ = [
    'PipelineLogger',
    'ErrorHandler',
    'PriceFormatter',
    'CurrencyFormatter',
    'DataValidator',
    'ConfigValidator',
    # 'ServiceCoordinator',  # removed
    'get_connection_pool',
    'get_api_rate_limiter',
    'get_model_cache'
]
