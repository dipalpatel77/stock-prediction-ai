#!/usr/bin/env python3
"""
Analysis Configuration
Unified configuration management for all analysis tools
"""

import os
from typing import Dict, List, Any
from pathlib import Path
import json

class AnalysisConfig:
    """Configuration management for all analysis types."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = self._load_default_config()
        
        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            # Data configuration
            'data': {
                'default_period': '2y',
                'default_interval': '1d',
                'cache_enabled': True,
                'cache_dir': 'data/cache',
                'timeout_seconds': 120,
                'min_data_points': 100,
                'max_missing_ratio': 0.1,
                'force_refresh': False
            },
            
            # Model configuration
            'models': {
                'simple': {
                    'models': ['random_forest', 'linear_regression'],
                    'scalers': ['standard', 'minmax'],
                    'ensemble': False,
                    'test_size': 0.2,
                    'random_state': 42
                },
                'advanced': {
                    'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost'],
                    'scalers': ['standard', 'robust'],
                    'ensemble': True,
                    'test_size': 0.2,
                    'random_state': 42
                },
                'full': {
                    'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'catboost', 
                              'svr', 'ridge', 'lasso', 'elastic_net', 'mlp', 'gaussian_process'],
                    'scalers': ['standard', 'minmax', 'robust'],
                    'ensemble': True,
                    'test_size': 0.2,
                    'random_state': 42
                },
                'cache_enabled': True,
                'cache_dir': 'models/cache',
                'save_models': True,
                'model_dir': 'models'
            },
            
            # Analysis configuration
            'analysis': {
                'timeframes': ['short_term', 'mid_term', 'long_term'],
                'enhanced_features': True,
                'parallel_processing': True,
                'max_workers': 4,
                'confidence_level': 0.95,
                'risk_assessment': True,
                'technical_indicators': True,
                'sentiment_analysis': True,
                'fundamental_analysis': True
            },
            
            # Technical indicators configuration
            'technical_indicators': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bollinger_period': 20,
                'bollinger_std': 2,
                'stochastic_k': 14,
                'stochastic_d': 3,
                'adx_period': 14,
                'williams_r_period': 14,
                'cci_period': 20,
                'atr_period': 14
            },
            
            # Sentiment analysis configuration
            'sentiment': {
                'enabled': True,
                'sources': ['news', 'social_media', 'analyst_ratings'],
                'vader_threshold': 0.1,
                'textblob_threshold': 0.1,
                'cache_sentiment': True
            },
            
            # Fundamental analysis configuration
            'fundamental': {
                'balance_sheet': True,
                'income_statement': True,
                'cash_flow': True,
                'financial_ratios': True,
                'company_events': True,
                'economic_indicators': True
            },
            
            # Reporting configuration
            'reporting': {
                'output_dir': 'reports',
                'save_charts': True,
                'chart_format': 'png',
                'chart_dpi': 300,
                'export_formats': ['csv', 'json'],
                'include_charts': True,
                'generate_dashboard': True
            },
            
            # Performance configuration
            'performance': {
                'enable_logging': True,
                'log_level': 'INFO',
                'log_file': 'logs/analysis.log',
                'progress_bar': True,
                'verbose_output': True,
                'save_intermediate_results': False
            },
            
            # API configuration
            'api': {
                'yfinance_timeout': 30,
                'max_retries': 3,
                'retry_delay': 1,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
    
    def get_config(self, section: str = None) -> Dict:
        """Get configuration for a specific section or entire config."""
        if section:
            return self.config.get(section, {})
        return self.config
    
    def set_config(self, section: str, key: str, value: Any):
        """Set a configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def update_config(self, updates: Dict):
        """Update configuration with new values."""
        for section, values in updates.items():
            if section not in self.config:
                self.config[section] = {}
            self.config[section].update(values)
    
    def load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            self.update_config(file_config)
            print(f"✅ Configuration loaded from {config_file}")
        except Exception as e:
            print(f"⚠️ Error loading configuration from {config_file}: {e}")
    
    def save_config(self, config_file: str = None):
        """Save configuration to file."""
        if not config_file:
            config_file = self.config_file or 'config/analysis_config.json'
        
        try:
            # Create directory if it doesn't exist
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Error saving configuration to {config_file}: {e}")
    
    def get_model_config(self, complexity: str = 'advanced') -> Dict:
        """Get model configuration for specific complexity level."""
        return self.config['models'].get(complexity, self.config['models']['advanced'])
    
    def get_data_config(self) -> Dict:
        """Get data configuration."""
        return self.config['data']
    
    def get_analysis_config(self) -> Dict:
        """Get analysis configuration."""
        return self.config['analysis']
    
    def get_technical_indicators_config(self) -> Dict:
        """Get technical indicators configuration."""
        return self.config['technical_indicators']
    
    def get_sentiment_config(self) -> Dict:
        """Get sentiment analysis configuration."""
        return self.config['sentiment']
    
    def get_fundamental_config(self) -> Dict:
        """Get fundamental analysis configuration."""
        return self.config['fundamental']
    
    def get_reporting_config(self) -> Dict:
        """Get reporting configuration."""
        return self.config['reporting']
    
    def get_performance_config(self) -> Dict:
        """Get performance configuration."""
        return self.config['performance']
    
    def get_api_config(self) -> Dict:
        """Get API configuration."""
        return self.config['api']
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate data configuration
        data_config = self.config['data']
        if data_config['timeout_seconds'] <= 0:
            errors.append("Data timeout must be positive")
        
        if data_config['min_data_points'] <= 0:
            errors.append("Minimum data points must be positive")
        
        # Validate model configuration
        for complexity in ['simple', 'advanced', 'full']:
            if complexity in self.config['models']:
                model_config = self.config['models'][complexity]
                if not model_config['models']:
                    errors.append(f"No models specified for {complexity} complexity")
                
                if model_config['test_size'] <= 0 or model_config['test_size'] >= 1:
                    errors.append(f"Invalid test size for {complexity} complexity")
        
        # Validate analysis configuration
        analysis_config = self.config['analysis']
        if analysis_config['max_workers'] <= 0:
            errors.append("Max workers must be positive")
        
        if analysis_config['confidence_level'] <= 0 or analysis_config['confidence_level'] >= 1:
            errors.append("Confidence level must be between 0 and 1")
        
        return errors
    
    def create_directories(self):
        """Create necessary directories based on configuration."""
        directories = [
            self.config['data']['cache_dir'],
            self.config['models']['cache_dir'],
            self.config['models']['model_dir'],
            self.config['reporting']['output_dir'],
            Path(self.config['performance']['log_file']).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_optimized_config(self, analysis_type: str) -> Dict:
        """Get optimized configuration for specific analysis type."""
        optimized_config = self.config.copy()
        
        if analysis_type == 'quick':
            # Optimize for speed
            optimized_config['data']['default_period'] = '6mo'
            optimized_config['models']['advanced']['models'] = ['random_forest', 'linear_regression']
            optimized_config['analysis']['parallel_processing'] = False
            optimized_config['analysis']['max_workers'] = 2
            optimized_config['reporting']['save_charts'] = False
            optimized_config['performance']['verbose_output'] = False
        
        elif analysis_type == 'comprehensive':
            # Optimize for completeness
            optimized_config['data']['default_period'] = '5y'
            optimized_config['models']['full']['models'] = self.config['models']['full']['models']
            optimized_config['analysis']['parallel_processing'] = True
            optimized_config['analysis']['max_workers'] = 8
            optimized_config['reporting']['save_charts'] = True
            optimized_config['performance']['verbose_output'] = True
        
        elif analysis_type == 'real_time':
            # Optimize for real-time analysis
            optimized_config['data']['default_period'] = '1mo'
            optimized_config['data']['cache_enabled'] = False
            optimized_config['models']['simple']['models'] = ['linear_regression']
            optimized_config['analysis']['parallel_processing'] = False
            optimized_config['analysis']['max_workers'] = 1
            optimized_config['reporting']['save_charts'] = False
            optimized_config['performance']['verbose_output'] = False
        
        return optimized_config
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("\n📋 Configuration Summary:")
        print("=" * 50)
        
        # Data configuration
        data_config = self.config['data']
        print(f"📊 Data Period: {data_config['default_period']}")
        print(f"📊 Cache Enabled: {data_config['cache_enabled']}")
        print(f"📊 Timeout: {data_config['timeout_seconds']}s")
        
        # Model configuration
        model_config = self.config['models']['advanced']
        print(f"🤖 Models: {', '.join(model_config['models'])}")
        print(f"🤖 Ensemble: {model_config['ensemble']}")
        
        # Analysis configuration
        analysis_config = self.config['analysis']
        print(f"🔍 Timeframes: {', '.join(analysis_config['timeframes'])}")
        print(f"🔍 Parallel Processing: {analysis_config['parallel_processing']}")
        print(f"🔍 Max Workers: {analysis_config['max_workers']}")
        
        # Reporting configuration
        reporting_config = self.config['reporting']
        print(f"📈 Output Directory: {reporting_config['output_dir']}")
        print(f"📈 Save Charts: {reporting_config['save_charts']}")
        print(f"📈 Export Formats: {', '.join(reporting_config['export_formats'])}")
        
        print("=" * 50)
    
    @classmethod
    def create_default_config_file(cls, config_file: str = 'config/analysis_config.json'):
        """Create a default configuration file."""
        config = cls()
        config.save_config(config_file)
        return config_file
    
    @classmethod
    def load_from_file(cls, config_file: str):
        """Create configuration instance from file."""
        return cls(config_file)

# Global configuration instance
default_config = AnalysisConfig()

def get_config(section: str = None) -> Dict:
    """Get global configuration."""
    return default_config.get_config(section)

def set_config(section: str, key: str, value: Any):
    """Set global configuration."""
    default_config.set_config(section, key, value)

def update_config(updates: Dict):
    """Update global configuration."""
    default_config.update_config(updates)
