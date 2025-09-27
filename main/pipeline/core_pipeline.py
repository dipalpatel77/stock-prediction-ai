"""
Core Pipeline
Main pipeline orchestrator with full service integration
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_pipeline import PipelineOrchestrator
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .strategy_analyzer import StrategyAnalyzer
from .prediction_generator import PredictionGenerator
from ..interfaces.user_interface import UserInterface
from ..utils.service_manager import ServiceManager


class UnifiedAnalysisPipeline:
    """
    Main pipeline orchestrator with full service integration
    
    This pipeline provides:
    - Complete analysis workflow orchestration
    - Service integration and management
    - Component coordination and execution
    - Error handling and recovery
    - Performance monitoring and reporting
    """
    
    def __init__(self, ticker: str = "AAPL", config: Dict[str, Any] = None):
        """
        Initialize Unified Analysis Pipeline
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
        """
        self.ticker = ticker
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize orchestrator and service manager
        self.orchestrator = PipelineOrchestrator(ticker, self.config)
        self.service_manager = ServiceManager(self.config)
        
        # Initialize components
        self._setup_components()
        
        # Initialize services
        self._initialize_services()
        
        self.logger.info(f"Unified Analysis Pipeline initialized for {ticker}")
    
    def _setup_components(self):
        """Setup all pipeline components"""
        try:
            # Data Processor
            data_processor = DataProcessor(self.ticker, self.config)
            self.orchestrator.add_component('data_processor', data_processor)
            
            # Model Trainer
            model_trainer = ModelTrainer(self.ticker, self.config)
            self.orchestrator.add_component('model_trainer', model_trainer)
            
            # Strategy Analyzer
            strategy_analyzer = StrategyAnalyzer(self.ticker, self.config)
            self.orchestrator.add_component('strategy_analyzer', strategy_analyzer)
            
            # Prediction Generator
            prediction_generator = PredictionGenerator(self.ticker, self.config)
            self.orchestrator.add_component('prediction_generator', prediction_generator)
            
            # Set execution order
            self.orchestrator.set_execution_order([
                'data_processor',
                'model_trainer', 
                'strategy_analyzer',
                'prediction_generator'
            ])
            
            # Set dependencies
            dependencies = {
                'model_trainer': ['data_processor'],
                'strategy_analyzer': ['data_processor'],
                'prediction_generator': ['data_processor', 'model_trainer']
            }
            self.orchestrator.set_dependencies(dependencies)
            
            self.logger.info("Pipeline components setup completed")
            
        except Exception as e:
            self.logger.error(f"Component setup failed: {e}")
            raise
    
    def _initialize_services(self):
        """Initialize all services"""
        try:
            service_result = self.service_manager.initialize_services(self.ticker)
            if not service_result['success']:
                raise Exception(f"Service initialization failed: {service_result['error']}")
            
            self.logger.info(f"Services initialized: {service_result['services']}")
            
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            raise
    
    def setup_pipeline_components(self) -> Dict[str, Any]:
        """
        Setup pipeline components
        
        Returns:
            Dictionary with setup results
        """
        try:
            self.logger.info("Setting up pipeline components")
            
            # Setup orchestrator
            if hasattr(self, 'orchestrator'):
                self.logger.info("Pipeline orchestrator already initialized")
            else:
                self.logger.warning("Pipeline orchestrator not initialized")
                return {'success': False, 'error': 'Orchestrator not initialized'}
            
            # Setup service manager
            if hasattr(self, 'service_manager'):
                self.logger.info("Service manager already initialized")
            else:
                self.logger.warning("Service manager not initialized")
                return {'success': False, 'error': 'Service manager not initialized'}
            
            self.logger.info("Pipeline components setup completed")
            return {'success': True, 'message': 'Pipeline components setup completed'}
            
        except Exception as e:
            self.logger.error(f"Pipeline components setup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def initialize_services(self) -> Dict[str, Any]:
        """
        Initialize all services
        
        Returns:
            Dictionary with initialization results
        """
        try:
            self.logger.info("Initializing services")
            
            # Initialize service manager
            if hasattr(self, 'service_manager'):
                service_result = self.service_manager.initialize_services(self.ticker)
                if service_result['success']:
                    self.logger.info("Services initialized successfully")
                    return service_result
                else:
                    self.logger.error(f"Service initialization failed: {service_result['error']}")
                    return service_result
            else:
                self.logger.error("Service manager not available")
                return {'success': False, 'error': 'Service manager not available'}
                
        except Exception as e:
            self.logger.error(f"Service initialization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline
        
        Args:
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting analysis for {self.ticker}")
            
            # Execute pipeline
            results = self.orchestrator.execute_pipeline(**kwargs)
            
            execution_time = time.time() - start_time
            
            # Add service status to results
            service_status = self.service_manager.get_service_status()
            
            return {
                'success': results['success'],
                'ticker': self.ticker,
                'execution_time': execution_time,
                'pipeline_results': results,
                'service_status': service_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': self.ticker,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_interactive_analysis(self, **kwargs) -> Dict[str, Any]:
        """
        Run interactive analysis with user interface
        
        Args:
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Initialize user interface
            user_interface = UserInterface()
            
            # Get user inputs
            user_inputs = user_interface.get_user_inputs()
            
            if not user_inputs['success']:
                return {
                    'success': False,
                    'error': user_inputs['error']
                }
            
            # Update configuration with user inputs
            self.config.update(user_inputs['data'])
            
            # Run analysis with updated configuration
            return self.run_analysis(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Interactive analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_component(self, component_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a specific component
        
        Args:
            component_name: Name of the component to run
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with component results
        """
        try:
            if component_name not in self.orchestrator.components:
                return {
                    'success': False,
                    'error': f"Component '{component_name}' not found"
                }
            
            component = self.orchestrator.components[component_name]
            result = component.run(**kwargs)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Component execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get pipeline status
        
        Returns:
            Dictionary with pipeline status
        """
        try:
            pipeline_status = self.orchestrator.get_pipeline_status()
            service_status = self.service_manager.get_service_status()
            
            return {
                'pipeline': pipeline_status,
                'services': service_status,
                'ticker': self.ticker,
                'config': self.config
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline status: {e}")
            return {
                'error': str(e)
            }
    
    def validate_pipeline(self) -> bool:
        """
        Validate pipeline configuration
        
        Returns:
            True if pipeline is valid, False otherwise
        """
        try:
            # Validate orchestrator
            if not self.orchestrator.validate_pipeline():
                return False
            
            # Validate services
            service_health = self.service_manager.get_all_service_health()
            unhealthy_services = [name for name, health in service_health.items() if not health]
            
            if unhealthy_services:
                self.logger.warning(f"Unhealthy services: {unhealthy_services}")
                # Don't fail validation for unhealthy services, just warn
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {e}")
            return False
    
    def get_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get analysis summary
        
        Args:
            results: Analysis results
            
        Returns:
            Dictionary with analysis summary
        """
        try:
            if not results.get('success', False):
                return {
                    'success': False,
                    'error': results.get('error', 'Unknown error')
                }
            
            pipeline_results = results.get('pipeline_results', {})
            service_status = results.get('service_status', {})
            
            summary = {
                'ticker': self.ticker,
                'success': True,
                'execution_time': results.get('execution_time', 0),
                'components_executed': len(pipeline_results.get('successful_components', [])),
                'components_failed': len(pipeline_results.get('failed_components', [])),
                'services_initialized': service_status.get('initialized_services', 0),
                'services_total': service_status.get('total_services', 0),
                'timestamp': results.get('timestamp', datetime.now().isoformat())
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get analysis summary: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Export analysis results to file
        
        Args:
            results: Analysis results
            filename: Optional filename for export
            
        Returns:
            Path to exported file
        """
        try:
            import json
            from pathlib import Path
            
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{self.ticker}_analysis_{timestamp}.json"
            
            export_path = Path('exports') / filename
            export_path.parent.mkdir(exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the pipeline"""
        try:
            # Shutdown services
            if hasattr(self.service_manager, 'shutdown'):
                self.service_manager.shutdown()
            
            self.logger.info("Pipeline shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Pipeline shutdown failed: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
