"""
Base Pipeline Component
Abstract base class for all pipeline components
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback


class BasePipelineComponent(ABC):
    """
    Abstract base class for all pipeline components
    
    This base class provides:
    - Common component interface
    - Timing and performance tracking
    - Error handling and logging
    - Component validation
    - Status reporting
    """
    
    def __init__(self, name: str, ticker: str, config: Dict[str, Any] = None):
        """
        Initialize base pipeline component
        
        Args:
            name: Component name
            ticker: Stock ticker symbol
            config: Configuration dictionary
        """
        self.name = name
        self.ticker = ticker
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.start_time = None
        self.end_time = None
        self.status = 'initialized'
        self.error_count = 0
        self.success_count = 0
        
        self.logger.info(f"Component '{name}' initialized for {ticker}")
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the component logic
        
        Args:
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary with execution results
        """
        pass
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the component with error handling and timing
        
        Args:
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary with execution results
        """
        try:
            self.start_time = time.time()
            self.status = 'running'
            
            self.logger.info(f"Starting execution of {self.name}")
            
            # Execute the component logic
            result = self.execute(**kwargs)
            
            self.end_time = time.time()
            self.status = 'completed'
            self.success_count += 1
            
            execution_time = self.end_time - self.start_time
            self.logger.info(f"Component '{self.name}' completed in {execution_time:.2f}s")
            
            return {
                'success': True,
                'component': self.name,
                'ticker': self.ticker,
                'execution_time': execution_time,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.end_time = time.time()
            self.status = 'failed'
            self.error_count += 1
            
            execution_time = self.end_time - self.start_time if self.start_time else 0
            
            self.logger.error(f"Component '{self.name}' failed after {execution_time:.2f}s: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'component': self.name,
                'ticker': self.ticker,
                'execution_time': execution_time,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status
        
        Returns:
            Dictionary with component status
        """
        return {
            'name': self.name,
            'ticker': self.ticker,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'execution_time': (self.end_time - self.start_time) if self.start_time and self.end_time else None,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0
        }
    
    def reset(self):
        """Reset component state"""
        self.start_time = None
        self.end_time = None
        self.status = 'initialized'
        self.error_count = 0
        self.success_count = 0
        self.logger.info(f"Component '{self.name}' reset")
    
    def validate_config(self) -> bool:
        """
        Validate component configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Basic validation - can be overridden by subclasses
            required_fields = self.get_required_config_fields()
            
            for field in required_fields:
                if field not in self.config:
                    self.logger.error(f"Required configuration field '{field}' not found")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_required_config_fields(self) -> List[str]:
        """
        Get list of required configuration fields
        
        Returns:
            List of required configuration field names
        """
        return ['ticker']  # Basic requirement - can be overridden by subclasses
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get component information
        
        Returns:
            Dictionary with component information
        """
        return {
            'name': self.name,
            'ticker': self.ticker,
            'type': self.__class__.__name__,
            'config': self.config,
            'status': self.get_status()
        }


class PipelineOrchestrator:
    """
    Pipeline orchestrator for managing multiple components
    
    This orchestrator provides:
    - Component registration and management
    - Pipeline execution coordination
    - Component dependency management
    - Execution monitoring and reporting
    - Error handling and recovery
    """
    
    def __init__(self, ticker: str, config: Dict[str, Any] = None):
        """
        Initialize pipeline orchestrator
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
        """
        self.ticker = ticker
        self.config = config or {}
        self.components = {}
        self.execution_order = []
        self.dependencies = {}
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Pipeline orchestrator initialized for {ticker}")
    
    def add_component(self, name: str, component: BasePipelineComponent):
        """
        Add a component to the pipeline
        
        Args:
            name: Component name
            component: Component instance
        """
        try:
            if not isinstance(component, BasePipelineComponent):
                raise ValueError(f"Component must be an instance of BasePipelineComponent")
            
            self.components[name] = component
            self.logger.info(f"Component '{name}' added to pipeline")
            
        except Exception as e:
            self.logger.error(f"Failed to add component '{name}': {e}")
    
    def remove_component(self, name: str):
        """
        Remove a component from the pipeline
        
        Args:
            name: Component name
        """
        try:
            if name in self.components:
                del self.components[name]
                if name in self.execution_order:
                    self.execution_order.remove(name)
                if name in self.dependencies:
                    del self.dependencies[name]
                self.logger.info(f"Component '{name}' removed from pipeline")
            else:
                self.logger.warning(f"Component '{name}' not found in pipeline")
                
        except Exception as e:
            self.logger.error(f"Failed to remove component '{name}': {e}")
    
    def set_execution_order(self, order: List[str]):
        """
        Set the execution order for components
        
        Args:
            order: List of component names in execution order
        """
        try:
            # Validate that all components exist
            for name in order:
                if name not in self.components:
                    raise ValueError(f"Component '{name}' not found in pipeline")
            
            self.execution_order = order
            self.logger.info(f"Execution order set: {order}")
            
        except Exception as e:
            self.logger.error(f"Failed to set execution order: {e}")
    
    def set_dependencies(self, dependencies: Dict[str, List[str]]):
        """
        Set component dependencies
        
        Args:
            dependencies: Dictionary mapping component names to their dependencies
        """
        try:
            # Validate dependencies
            for component, deps in dependencies.items():
                if component not in self.components:
                    raise ValueError(f"Component '{component}' not found in pipeline")
                
                for dep in deps:
                    if dep not in self.components:
                        raise ValueError(f"Dependency '{dep}' not found in pipeline")
            
            self.dependencies = dependencies
            self.logger.info(f"Dependencies set: {dependencies}")
            
        except Exception as e:
            self.logger.error(f"Failed to set dependencies: {e}")
    
    def execute_pipeline(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the entire pipeline
        
        Args:
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary with pipeline execution results
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting pipeline execution for {self.ticker}")
            
            results = {}
            execution_order = self._determine_execution_order()
            
            for component_name in execution_order:
                if component_name not in self.components:
                    self.logger.warning(f"Component '{component_name}' not found, skipping")
                    continue
                
                component = self.components[component_name]
                
                # Check dependencies
                if not self._check_dependencies(component_name, results):
                    self.logger.error(f"Dependencies not met for component '{component_name}'")
                    results[component_name] = {
                        'success': False,
                        'error': 'Dependencies not met'
                    }
                    continue
                
                # Execute component with previous results
                self.logger.info(f"Executing component '{component_name}'")
                
                # Pass data from previous components
                execution_kwargs = kwargs.copy()
                
                # Find data processor result (check all possible names)
                data_processor_result = None
                self.logger.info(f"Available results: {list(results.keys())}")
                
                # First, check for ticker-named component (actual data processor)
                if self.ticker in results and results[self.ticker].get('success'):
                    ticker_result = results[self.ticker]
                    # Check if data is in the nested 'result' key
                    nested_result = ticker_result.get('result', {})
                    self.logger.info(f"Checking ticker result {self.ticker}: success={ticker_result.get('success')}, has_data={'data' in nested_result}, data_is_none={nested_result.get('data') is None if 'data' in nested_result else 'N/A'}")
                    if 'data' in nested_result and nested_result['data'] is not None:
                        data_processor_result = nested_result
                        self.logger.info(f"Found data processor result in ticker component: {self.ticker}")
                
                # If not found, check all other components
                if not data_processor_result:
                    for result_name, result_data in results.items():
                        # Check if data is in the nested 'result' key
                        nested_result = result_data.get('result', {})
                        self.logger.info(f"Checking result {result_name}: success={result_data.get('success')}, has_data={'data' in nested_result}, data_is_none={nested_result.get('data') is None if 'data' in nested_result else 'N/A'}")
                        if result_data.get('success') and ('data' in nested_result and nested_result['data'] is not None):
                            # This looks like a data processor result
                            data_processor_result = nested_result
                            self.logger.info(f"Found data processor result in component: {result_name}")
                            break
                
                if data_processor_result:
                    # data_processor_result is already the nested result from the component
                    data = data_processor_result.get('data')
                    processed_data = data_processor_result.get('processed_data')
                    multi_interval_data = data_processor_result.get('multi_interval_data')
                    self.logger.info(f"Passing data to {component_name}: data={data is not None}, processed_data={processed_data is not None}, multi_interval_data={multi_interval_data is not None}")
                    execution_kwargs['data'] = data
                    execution_kwargs['processed_data'] = processed_data
                    execution_kwargs['multi_interval_data'] = multi_interval_data
                
                # Find model trainer result (check all possible names)
                model_trainer_result = None
                for result_name, result_data in results.items():
                    # Check if models are in the nested 'result' key
                    nested_result = result_data.get('result', {})
                    self.logger.info(f"Checking result {result_name}: success={result_data.get('success')}, has_models={'models' in nested_result}, models_is_none={nested_result.get('models') is None if 'models' in nested_result else 'N/A'}")
                    
                    # Check both nested and direct models
                    if result_data.get('success'):
                        if 'models' in nested_result and nested_result['models'] is not None:
                            # Models in nested result
                            model_trainer_result = nested_result
                            self.logger.info(f"Found model trainer result in component: {result_name} (nested)")
                            break
                        elif 'models' in result_data and result_data['models'] is not None:
                            # Models in direct result
                            model_trainer_result = result_data
                            self.logger.info(f"Found model trainer result in component: {result_name} (direct)")
                            break
                
                if model_trainer_result:
                    models = model_trainer_result.get('models')
                    training_results = model_trainer_result.get('training_results')
                    self.logger.info(f"Passing models to {component_name}: models={models is not None}, training_results={training_results is not None}")
                    execution_kwargs['models'] = models
                    execution_kwargs['training_results'] = training_results
                
                component_result = component.run(**execution_kwargs)
                results[component_name] = component_result
                
                # Debug: Show what we're storing
                if 'data' in component_result:
                    self.logger.info(f"Storing result for {component_name}: has_data={component_result['data'] is not None}")
                else:
                    self.logger.info(f"Storing result for {component_name}: no 'data' key found")
                
                # Stop execution if component failed and no recovery is possible
                if not component_result['success'] and not self._can_recover(component_name, component_result):
                    self.logger.error(f"Pipeline execution stopped due to failure in '{component_name}'")
                    break
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Determine overall success
            successful_components = [name for name, result in results.items() if result.get('success', False)]
            failed_components = [name for name, result in results.items() if not result.get('success', False)]
            
            self.logger.info(f"Pipeline execution completed in {execution_time:.2f}s")
            self.logger.info(f"Successful components: {successful_components}")
            if failed_components:
                self.logger.warning(f"Failed components: {failed_components}")
            
            return {
                'success': len(failed_components) == 0,
                'ticker': self.ticker,
                'execution_time': execution_time,
                'results': results,
                'successful_components': successful_components,
                'failed_components': failed_components,
                'total_components': len(self.components),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': self.ticker,
                'timestamp': datetime.now().isoformat()
            }
    
    def _determine_execution_order(self) -> List[str]:
        """
        Determine the execution order for components
        
        Returns:
            List of component names in execution order
        """
        if self.execution_order:
            return self.execution_order
        
        # Default order based on component types
        default_order = ['data_processor', 'model_trainer', 'strategy_analyzer', 'prediction_generator']
        
        # Filter to only include components that exist
        return [name for name in default_order if name in self.components]
    
    def _check_dependencies(self, component_name: str, results: Dict[str, Any]) -> bool:
        """
        Check if component dependencies are met
        
        Args:
            component_name: Name of the component
            results: Results from previous components
            
        Returns:
            True if dependencies are met, False otherwise
        """
        if component_name not in self.dependencies:
            return True
        
        dependencies = self.dependencies[component_name]
        
        for dep in dependencies:
            if dep not in results:
                self.logger.warning(f"Dependency '{dep}' not found in results")
                return False
            
            if not results[dep].get('success', False):
                self.logger.warning(f"Dependency '{dep}' failed")
                return False
        
        return True
    
    def _can_recover(self, component_name: str, result: Dict[str, Any]) -> bool:
        """
        Check if pipeline can recover from component failure
        
        Args:
            component_name: Name of the failed component
            result: Component execution result
            
        Returns:
            True if recovery is possible, False otherwise
        """
        # Critical components that cannot fail
        critical_components = ['data_processor']
        
        if component_name in critical_components:
            return False
        
        # Check if error is recoverable
        error = result.get('error', '')
        recoverable_errors = ['timeout', 'rate_limit', 'temporary']
        
        return any(recoverable_error in error.lower() for recoverable_error in recoverable_errors)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get pipeline status
        
        Returns:
            Dictionary with pipeline status
        """
        component_statuses = {}
        
        for name, component in self.components.items():
            component_statuses[name] = component.get_status()
        
        return {
            'ticker': self.ticker,
            'total_components': len(self.components),
            'execution_order': self.execution_order,
            'dependencies': self.dependencies,
            'component_statuses': component_statuses,
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_pipeline(self) -> bool:
        """
        Validate the pipeline configuration
        
        Returns:
            True if pipeline is valid, False otherwise
        """
        try:
            # Check if all components are valid
            for name, component in self.components.items():
                if not component.validate_config():
                    self.logger.error(f"Component '{name}' configuration is invalid")
                    return False
            
            # Check if execution order is valid
            execution_order = self._determine_execution_order()
            for name in execution_order:
                if name not in self.components:
                    self.logger.error(f"Component '{name}' in execution order not found")
                    return False
            
            # Check if dependencies are valid
            for component, deps in self.dependencies.items():
                if component not in self.components:
                    self.logger.error(f"Component '{component}' with dependencies not found")
                    return False
                
                for dep in deps:
                    if dep not in self.components:
                        self.logger.error(f"Dependency '{dep}' not found")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {e}")
            return False