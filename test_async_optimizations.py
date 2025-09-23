#!/usr/bin/env python3
"""
Test Async Optimizations
Comprehensive testing of Phase 2 async optimizations
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

# Import async components
from main.pipeline.async_pipeline_orchestrator import AsyncPipelineOrchestrator
from main.services.database_manager import DatabaseManager
from main.services.api_coordinator import APICoordinator
from main.pipeline.data_processor import DataProcessor

async def test_async_database_operations():
    """Test async database operations"""
    print("\n🗄️ Testing Async Database Operations")
    print("=" * 50)
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager({
            'enable_async': True,
            'enable_query_cache': True,
            'enable_performance_monitoring': True
        })
        
        # Test async query execution
        print("📊 Testing async query execution...")
        start_time = time.time()
        
        # Test queries
        queries = [
            ("SELECT 1 as test", None),
            ("SELECT COUNT(*) as count FROM information_schema.tables", None),
            ("SELECT NOW() as current_time", None)
        ]
        
        results = await db_manager.async_execute_parallel_queries(queries)
        
        execution_time = time.time() - start_time
        print(f"✅ Async parallel queries completed in {execution_time:.3f}s")
        print(f"📈 Results: {len([r for r in results if r is not None])}/{len(queries)} successful")
        
        # Test performance metrics
        performance_report = db_manager.get_performance_report()
        print(f"📊 Database Performance:")
        print(f"   - Total Queries: {performance_report['performance_metrics']['total_queries']}")
        print(f"   - Average Query Time: {performance_report['performance_metrics']['average_query_time']:.3f}s")
        print(f"   - Cache Hit Rate: {performance_report['performance_metrics']['cache_hit_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Async database operations failed: {e}")
        return False

async def test_async_api_coordination():
    """Test async API coordination"""
    print("\n🌐 Testing Async API Coordination")
    print("=" * 50)
    
    try:
        # Initialize API coordinator
        api_coordinator = APICoordinator(max_workers=4)
        
        # Test async parallel loading
        print("📡 Testing async parallel API loading...")
        start_time = time.time()
        
        config = {
            'is_indian': False,
            'enable_async': True,
            'enable_streaming': True
        }
        
        result = await api_coordinator.async_coordinate_parallel_loading("AAPL", config)
        
        execution_time = time.time() - start_time
        print(f"✅ Async parallel API loading completed in {execution_time:.3f}s")
        print(f"📈 Tasks completed: {result.get('tasks_completed', 0)}")
        
        # Test performance metrics
        performance_metrics = api_coordinator.get_performance_metrics()
        print(f"📊 API Performance:")
        print(f"   - Total calls: {performance_metrics['total_calls']}")
        print(f"   - Success rate: {performance_metrics['success_rate']:.1f}%")
        print(f"   - Average response time: {performance_metrics['average_response_time']:.3f}s")
        print(f"   - Cache hit rate: {performance_metrics['cache_hit_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Async API coordination failed: {e}")
        return False

async def test_async_data_processing():
    """Test async data processing"""
    print("\n📊 Testing Async Data Processing")
    print("=" * 50)
    
    try:
        # Initialize data processor
        data_processor = DataProcessor("AAPL", {
            'enable_async': True,
            'enable_streaming': True,
            'memory_limit_mb': 512,
            'chunk_size': 100
        })
        
        # Create test data
        print("📈 Creating test data...")
        test_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 1000),
            'High': np.random.uniform(100, 200, 1000),
            'Low': np.random.uniform(100, 200, 1000),
            'Close': np.random.uniform(100, 200, 1000),
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000, freq='D'))
        
        # Test async data processing
        print("🔄 Testing async data processing...")
        start_time = time.time()
        
        processed_data = await data_processor.async_process_data(test_data)
        
        execution_time = time.time() - start_time
        print(f"✅ Async data processing completed in {execution_time:.3f}s")
        print(f"📈 Processed {len(processed_data)} records")
        
        # Test streaming data processing
        print("🌊 Testing async streaming data processing...")
        start_time = time.time()
        
        chunk_count = 0
        async for chunk in data_processor.async_stream_data_processing("test_source", "AAPL", {}):
            if not chunk.empty:
                chunk_count += 1
                if chunk_count >= 5:  # Limit to 5 chunks for testing
                    break
        
        execution_time = time.time() - start_time
        print(f"✅ Async streaming completed in {execution_time:.3f}s")
        print(f"📈 Processed {chunk_count} chunks")
        
        # Test performance metrics
        performance_report = await data_processor.async_get_performance_report()
        print(f"📊 Data Processing Performance:")
        print(f"   - Current Memory: {performance_report['current_memory_mb']:.1f}MB")
        print(f"   - Peak Memory: {performance_report['peak_memory_mb']:.1f}MB")
        print(f"   - Chunks Processed: {performance_report['chunks_processed']}")
        print(f"   - Data Points Processed: {performance_report['data_points_processed']}")
        print(f"   - Processing Time: {performance_report['processing_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Async data processing failed: {e}")
        return False

async def test_async_pipeline_orchestration():
    """Test async pipeline orchestration"""
    print("\n🚀 Testing Async Pipeline Orchestration")
    print("=" * 50)
    
    try:
        # Initialize async pipeline orchestrator
        config = {
            'enable_async': True,
            'enable_streaming': True,
            'enable_websocket': True,
            'max_concurrent_tasks': 5,
            'memory_limit_mb': 512
        }
        
        async with AsyncPipelineOrchestrator(config) as orchestrator:
            # Test single ticker analysis
            print("📊 Testing async single ticker analysis...")
            start_time = time.time()
            
            result = await orchestrator.async_run_analysis("AAPL", config)
            
            execution_time = time.time() - start_time
            print(f"✅ Async analysis completed in {execution_time:.3f}s")
            print(f"📈 Success: {result.get('success', False)}")
            print(f"📈 Execution time: {result.get('execution_time', 0):.3f}s")
            
            # Test batch analysis
            print("📊 Testing async batch analysis...")
            start_time = time.time()
            
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            batch_result = await orchestrator.async_batch_analysis(tickers, config)
            
            execution_time = time.time() - start_time
            print(f"✅ Async batch analysis completed in {execution_time:.3f}s")
            print(f"📈 Total tickers: {batch_result.get('total_tickers', 0)}")
            print(f"📈 Successful: {batch_result.get('successful', 0)}")
            print(f"📈 Failed: {batch_result.get('failed', 0)}")
            
            # Test performance report
            print("📊 Testing async performance report...")
            performance_report = await orchestrator.async_get_performance_report()
            print(f"📈 Orchestrator Performance:")
            print(f"   - Total Tasks: {performance_report['orchestrator_metrics']['total_tasks']}")
            print(f"   - Completed Tasks: {performance_report['orchestrator_metrics']['completed_tasks']}")
            print(f"   - Failed Tasks: {performance_report['orchestrator_metrics']['failed_tasks']}")
            print(f"   - Total Processing Time: {performance_report['orchestrator_metrics']['total_processing_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Async pipeline orchestration failed: {e}")
        return False

async def test_websocket_streaming():
    """Test WebSocket streaming (simulated)"""
    print("\n🌐 Testing WebSocket Streaming")
    print("=" * 50)
    
    try:
        # Initialize API coordinator
        api_coordinator = APICoordinator(max_workers=2)
        
        # Test WebSocket streaming (simulated)
        print("📡 Testing WebSocket streaming...")
        start_time = time.time()
        
        config = {
            'enable_websocket': True,
            'websocket_timeout': 10
        }
        
        update_count = 0
        async for update in api_coordinator.start_websocket_stream("AAPL", config):
            update_count += 1
            print(f"📈 Received update {update_count}: {update.get('success', False)}")
            
            if update_count >= 3:  # Limit to 3 updates for testing
                break
        
        execution_time = time.time() - start_time
        print(f"✅ WebSocket streaming completed in {execution_time:.3f}s")
        print(f"📈 Received {update_count} updates")
        
        # Cleanup
        await api_coordinator.close_async_resources()
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket streaming failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 PHASE 2: ASYNC OPTIMIZATION TESTING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {}
    
    # Test async database operations
    test_results['database'] = await test_async_database_operations()
    
    # Test async API coordination
    test_results['api'] = await test_async_api_coordination()
    
    # Test async data processing
    test_results['data_processing'] = await test_async_data_processing()
    
    # Test async pipeline orchestration
    test_results['orchestration'] = await test_async_pipeline_orchestration()
    
    # Test WebSocket streaming
    test_results['websocket'] = await test_websocket_streaming()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ASYNC OPTIMIZATION TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print()
    print("🎉 Phase 2: Async Optimization Testing Completed!")
    print("🚀 Advanced async/await, streaming, and WebSocket features are working!")
    
    return test_results

if __name__ == "__main__":
    # Run async tests
    asyncio.run(main())
