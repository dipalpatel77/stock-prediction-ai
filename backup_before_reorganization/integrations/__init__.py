"""
Integrations Module
==================

This module contains all integration components that combine multiple services
and analysis modules to provide comprehensive stock analysis.

Phase 1: Enhanced fundamental analysis, global market tracking, institutional flows
Phase 2: Economic data APIs, currency/commodity tracking, regulatory monitoring  
Phase 3: Geopolitical risk, corporate actions, insider trading analysis
"""

from .phase1_integration import Phase1Integration
from .phase2_integration import Phase2Integration
from .phase3_integration import Phase3Integration

__all__ = [
    'Phase1Integration',
    'Phase2Integration', 
    'Phase3Integration'
]
