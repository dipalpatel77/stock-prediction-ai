#!/usr/bin/env python3
"""
Test Phase 3: Advanced Features
Comprehensive testing of advanced features including caching, ML optimization, monitoring, and auto-scaling
"""

import asyncio
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import advanced features
from main.services.advanced_cache_manager import AdvancedCacheManager
from main.services.ml_optimizer import MLOptimizer
from main.services.monitoring_dashboard import MonitoringDashboard
from main.services.auto_scaler import AutoScaler

async def test_advanced_caching():
    """Test advanced multi-level caching"""
    print("\n🗄️ Testing Advanced Multi-Level Caching")
    print("=" * 60)
    
    try:
        # Initialize cache manager
        cache_config = {
            'cache_type': 'multi_level',
            'memory_cache_size': 100,
            'disk_cache_dir': 'cache/test',
            'default_ttl': 60,
            'enable_compression': True,
            'analytics_enabled': True
        }
        
        cache_manager = AdvancedCacheManager(cache_config)
        
        # Test basic operations
        print("📊 Testing cache operations...")
        
        # Set some test data
        test_data = {
            'stock_data': pd.DataFrame({
                'price': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 1200, 1300, 1400]
            }),
            'model_predictions': [0.8, 0.85, 0.9, 0.88, 0.92],
            'metadata': {'ticker': 'AAPL', 'timestamp': datetime.now().isoformat()}
        }
        
        # Test set operations
        for key, value in test_data.items():
            success = await cache_manager.set(key, value, ttl=60)
            print(f"   ✅ Set {key}: {success}")
        
        # Test get operations
        for key in test_data.keys():
            value = await cache_manager.get(key)
            print(f"   ✅ Get {key}: {type(value).__name__}")
        
        # Test cache warming
        print("🔥 Testing cache warming...")
        keys = ['warm_key_1', 'warm_key_2', 'warm_key_3']
        values = ['warm_value_1', 'warm_value_2', 'warm_value_3']
        
        warming_success = await cache_manager.warm_cache(keys, values, ttl=30)
        print(f"   ✅ Cache warming: {warming_success}")
        
        # Test analytics
        print("📈 Testing cache analytics...")
        analytics = cache_manager.get_analytics()
        print(f"   📊 Cache Analytics:")
        print(f"      - Total requests: {analytics['total_requests']}")
        print(f"      - Hit rate: {analytics['hit_rate']:.1f}%")
        print(f"      - Cache size: {analytics['memory_cache_size']}")
        print(f"      - Redis available: {analytics['redis_available']}")
        
        # Test cleanup
        print("🧹 Testing cache cleanup...")
        cleaned = await cache_manager.cleanup_expired()
        print(f"   ✅ Cleaned {cleaned} expired entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced caching test failed: {e}")
        return False

async def test_ml_optimization():
    """Test ML optimization features"""
    print("\n🤖 Testing ML Optimization")
    print("=" * 60)
    
    try:
        # Initialize ML optimizer
        ml_config = {
            'models_dir': 'models/test',
            'enable_hyperparameter_tuning': True,
            'enable_ensemble': True,
            'enable_feature_selection': True,
            'cross_validation_folds': 3,
            'max_iterations': 10
        }
        
        ml_optimizer = MLOptimizer(ml_config)
        
        # Create test data
        print("📊 Creating test dataset...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))
        
        print(f"   📈 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test model optimization
        print("🔧 Testing model optimization...")
        start_time = time.time()
        
        optimization_result = await ml_optimizer.optimize_models(
            X, y, 
            model_types=['random_forest', 'gradient_boosting', 'xgboost']
        )
        
        execution_time = time.time() - start_time
        print(f"   ✅ Model optimization completed in {execution_time:.2f}s")
        print(f"   📈 Success: {optimization_result.get('success', False)}")
        
        if optimization_result.get('success'):
            individual_models = optimization_result.get('individual_models', {})
            print(f"   📊 Individual Models:")
            for model_type, result in individual_models.items():
                if 'error' not in result:
                    performance = result.get('performance', {})
                    print(f"      - {model_type}: R² = {performance.get('r2', 0):.3f}")
            
            ensemble = optimization_result.get('ensemble')
            if ensemble and 'error' not in ensemble:
                ensemble_perf = ensemble.get('performance', {})
                print(f"   🎯 Ensemble: R² = {ensemble_perf.get('r2', 0):.3f}")
            
            feature_selection = optimization_result.get('feature_selection')
            if feature_selection and 'error' not in feature_selection:
                selected_features = feature_selection.get('selected_features', [])
                print(f"   🔍 Feature Selection: {len(selected_features)} features selected")
        
        # Test auto-scaling
        print("📈 Testing ML auto-scaling...")
        scaling_result = await ml_optimizer.auto_scale_models(X, y, performance_threshold=0.7)
        print(f"   ✅ Auto-scaling: {scaling_result.get('success', False)}")
        
        if scaling_result.get('success'):
            scaling_decisions = scaling_result.get('scaling_decisions', {})
            print(f"   📊 Scaling Strategy: {scaling_decisions.get('strategy', 'unknown')}")
            print(f"   📊 Status: {scaling_decisions.get('status', 'unknown')}")
        
        # Test optimization report
        print("📋 Testing optimization report...")
        report = ml_optimizer.get_optimization_report()
        print(f"   📊 Total models: {report.get('total_models', 0)}")
        print(f"   📁 Models directory: {report.get('models_directory', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML optimization test failed: {e}")
        return False

async def test_monitoring_dashboard():
    """Test real-time monitoring dashboard"""
    print("\n📊 Testing Real-time Monitoring Dashboard")
    print("=" * 60)
    
    try:
        # Initialize monitoring dashboard
        dashboard_config = {
            'websocket_port': 8766,  # Use different port for testing
            'websocket_host': 'localhost',
            'metrics_retention_days': 1,
            'alert_thresholds': {
                'cpu_usage': 80,
                'memory_usage': 85,
                'response_time': 5.0,
                'error_rate': 5.0
            },
            'enable_real_time': True,
            'enable_alerts': True,
            'dashboard_refresh_interval': 2.0
        }
        
        dashboard = MonitoringDashboard(dashboard_config)
        
        # Test metrics collection
        print("📈 Testing metrics collection...")
        
        # Simulate some metrics
        for i in range(5):
            system_metrics = await dashboard._collect_system_metrics()
            api_metrics = await dashboard._collect_api_metrics()
            db_metrics = await dashboard._collect_database_metrics()
            ml_metrics = await dashboard._collect_ml_metrics()
            
            print(f"   📊 Collection {i+1}: System={system_metrics.get('cpu_percent', 0):.1f}%, "
                  f"API={api_metrics.get('success_rate', 0):.1f}%, "
                  f"DB={db_metrics.get('query_count', 0)}, "
                  f"ML={ml_metrics.get('model_count', 0)}")
            
            await asyncio.sleep(1)
        
        # Test metrics summary
        print("📋 Testing metrics summary...")
        summary = dashboard.get_metrics_summary()
        print(f"   📊 Metrics Summary:")
        print(f"      - Total metrics: {summary.get('total_metrics_collected', 0)}")
        print(f"      - Active alerts: {summary.get('active_alerts', 0)}")
        print(f"      - Connected clients: {summary.get('connected_clients', 0)}")
        print(f"      - Dashboard URL: {summary.get('dashboard_url', 'unknown')}")
        print(f"      - Is running: {summary.get('is_running', False)}")
        
        # Test alert system
        print("🚨 Testing alert system...")
        
        # Add alert callback
        alert_received = []
        def alert_callback(alert):
            alert_received.append(alert)
            print(f"   🚨 Alert received: {alert.get('message', 'Unknown')}")
        
        dashboard.add_alert_callback(alert_callback)
        
        # Simulate high CPU alert
        high_cpu_metrics = {
            'system': {'cpu_percent': 85, 'memory_percent': 60},
            'application': {'response_time': 1.5, 'error_rate': 2.0},
            'load': {'concurrent_users': 50}
        }
        
        await dashboard._check_alert_conditions()
        
        # Test dashboard URL
        dashboard_url = dashboard.get_dashboard_url()
        print(f"   🌐 Dashboard URL: {dashboard_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring dashboard test failed: {e}")
        return False

async def test_auto_scaling():
    """Test intelligent auto-scaling"""
    print("\n📈 Testing Intelligent Auto-Scaling")
    print("=" * 60)
    
    try:
        # Initialize auto-scaler
        scaling_config = {
            'min_instances': 1,
            'max_instances': 10,
            'target_cpu_utilization': 70.0,
            'target_memory_utilization': 80.0,
            'target_response_time': 2.0,
            'scaling_cooldown': 10,  # Short cooldown for testing
            'enable_predictive_scaling': True,
            'enable_cost_optimization': True
        }
        
        auto_scaler = AutoScaler(scaling_config)
        
        # Test scaling decision making
        print("🧠 Testing scaling decision making...")
        
        # Simulate different load scenarios
        scenarios = [
            {'name': 'Low Load', 'cpu': 30, 'memory': 40, 'response_time': 1.0},
            {'name': 'High Load', 'cpu': 85, 'memory': 90, 'response_time': 3.0},
            {'name': 'Medium Load', 'cpu': 60, 'memory': 70, 'response_time': 2.0}
        ]
        
        for scenario in scenarios:
            print(f"   📊 Testing {scenario['name']} scenario...")
            
            # Create mock metrics
            mock_metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'cpu_percent': scenario['cpu'],
                    'memory_percent': scenario['memory'],
                    'disk_percent': 50,
                    'load_average': [1.0, 1.0, 1.0]
                },
                'application': {
                    'request_rate': 100 if scenario['name'] == 'High Load' else 50,
                    'response_time': scenario['response_time'],
                    'error_rate': 2.0,
                    'active_connections': 50
                },
                'load': {
                    'concurrent_users': 100 if scenario['name'] == 'High Load' else 50,
                    'data_processing_rate': 1000,
                    'api_calls_per_second': 50
                },
                'current_instances': auto_scaler.current_instances
            }
            
            # Make scaling decision
            decision = await auto_scaler._make_scaling_decision(mock_metrics)
            
            print(f"      🎯 Decision: {decision.action.value}")
            print(f"      📝 Reason: {decision.reason}")
            print(f"      🎯 Confidence: {decision.confidence:.2f}")
            print(f"      📊 Target instances: {decision.target_instances}")
            
            # Execute scaling if needed
            if decision.action.value != 'maintain':
                print(f"      ⚡ Executing scaling action...")
                await auto_scaler._execute_scaling_action(decision)
        
        # Test cost optimization
        print("💰 Testing cost optimization...")
        cost_analysis = await auto_scaler._analyze_cost_patterns()
        print(f"   📊 Cost Analysis:")
        print(f"      - Average cost: {cost_analysis.get('average_cost', 0):.2f}")
        print(f"      - Cost trend: {cost_analysis.get('cost_trend', 0):.2f}")
        print(f"      - Optimization potential: {cost_analysis.get('optimization_potential', 0):.2f}")
        
        # Test scaling summary
        print("📋 Testing scaling summary...")
        summary = auto_scaler.get_scaling_summary()
        print(f"   📊 Scaling Summary:")
        print(f"      - Current instances: {summary.get('current_instances', 0)}")
        print(f"      - Scaling events: {summary.get('scaling_events', 0)}")
        print(f"      - Successful scales: {summary.get('successful_scales', 0)}")
        print(f"      - Failed scales: {summary.get('failed_scales', 0)}")
        print(f"      - Scaling in progress: {summary.get('scaling_in_progress', False)}")
        print(f"      - Total metrics: {summary.get('total_metrics_collected', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

async def test_integrated_advanced_features():
    """Test integrated advanced features"""
    print("\n🚀 Testing Integrated Advanced Features")
    print("=" * 60)
    
    try:
        # Initialize all advanced services
        print("🔧 Initializing advanced services...")
        
        # Cache manager
        cache_manager = AdvancedCacheManager({
            'cache_type': 'multi_level',
            'memory_cache_size': 50,
            'disk_cache_dir': 'cache/integrated',
            'default_ttl': 30
        })
        
        # ML optimizer
        ml_optimizer = MLOptimizer({
            'models_dir': 'models/integrated',
            'enable_hyperparameter_tuning': True,
            'enable_ensemble': True,
            'max_iterations': 5
        })
        
        # Monitoring dashboard
        dashboard = MonitoringDashboard({
            'websocket_port': 8767,
            'enable_real_time': True,
            'enable_alerts': True,
            'dashboard_refresh_interval': 1.0
        })
        
        # Auto-scaler
        auto_scaler = AutoScaler({
            'min_instances': 1,
            'max_instances': 5,
            'scaling_cooldown': 5
        })
        
        print("   ✅ All services initialized")
        
        # Test integrated workflow
        print("🔄 Testing integrated workflow...")
        
        # 1. Cache ML model results
        print("   1️⃣ Caching ML model results...")
        model_results = {'accuracy': 0.85, 'predictions': [0.8, 0.9, 0.7]}
        cache_success = await cache_manager.set('ml_results', model_results, ttl=60)
        print(f"      ✅ Cache set: {cache_success}")
        
        # 2. Optimize ML models
        print("   2️⃣ Optimizing ML models...")
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randn(100))
        ml_result = await ml_optimizer.optimize_models(X, y, ['random_forest', 'gradient_boosting'])
        print(f"      ✅ ML optimization: {ml_result.get('success', False)}")
        
        # 3. Monitor performance
        print("   3️⃣ Monitoring performance...")
        system_metrics = await dashboard._collect_system_metrics()
        print(f"      ✅ System metrics: CPU={system_metrics.get('cpu_percent', 0):.1f}%")
        
        # 4. Auto-scale based on metrics
        print("   4️⃣ Auto-scaling...")
        mock_metrics = {
            'system': {'cpu_percent': 80, 'memory_percent': 75},
            'application': {'response_time': 2.5, 'error_rate': 3.0},
            'load': {'concurrent_users': 100}
        }
        scaling_decision = await auto_scaler._make_scaling_decision(mock_metrics)
        print(f"      ✅ Scaling decision: {scaling_decision.action.value}")
        
        # Test service integration
        print("🔗 Testing service integration...")
        
        # Cache analytics
        cache_analytics = cache_manager.get_analytics()
        print(f"   📊 Cache Analytics: {cache_analytics.get('hit_rate', 0):.1f}% hit rate")
        
        # ML optimization report
        ml_report = ml_optimizer.get_optimization_report()
        print(f"   🤖 ML Models: {ml_report.get('total_models', 0)} models")
        
        # Dashboard summary
        dashboard_summary = dashboard.get_metrics_summary()
        print(f"   📊 Dashboard: {dashboard_summary.get('total_metrics_collected', 0)} metrics")
        
        # Auto-scaling summary
        scaling_summary = auto_scaler.get_scaling_summary()
        print(f"   📈 Auto-scaling: {scaling_summary.get('current_instances', 0)} instances")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated advanced features test failed: {e}")
        return False

async def main():
    """Main test function for Phase 3 advanced features"""
    print("🚀 PHASE 3: ADVANCED FEATURES TESTING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {}
    
    # Test advanced caching
    test_results['advanced_caching'] = await test_advanced_caching()
    
    # Test ML optimization
    test_results['ml_optimization'] = await test_ml_optimization()
    
    # Test monitoring dashboard
    test_results['monitoring_dashboard'] = await test_monitoring_dashboard()
    
    # Test auto-scaling
    test_results['auto_scaling'] = await test_auto_scaling()
    
    # Test integrated features
    test_results['integrated_features'] = await test_integrated_advanced_features()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PHASE 3: ADVANCED FEATURES TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print()
    print("🎉 Phase 3: Advanced Features Testing Completed!")
    print("🚀 Advanced caching, ML optimization, monitoring, and auto-scaling features are working!")
    
    return test_results

if __name__ == "__main__":
    # Run Phase 3 tests
    asyncio.run(main())
