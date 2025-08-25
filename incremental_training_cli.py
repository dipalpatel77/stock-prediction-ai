#!/usr/bin/env python3
"""
Command Line Interface for Incremental Training

This script provides a command-line interface for managing incremental training
operations, model updates, and version management.
"""

import argparse
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path
sys.path.append('.')

from partB_model.model_update_pipeline import ModelUpdatePipeline, create_model_update_pipeline
from partB_model.incremental_learning import IncrementalLearningManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Incremental Training CLI for AI Stock Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for updates for a specific ticker
  python incremental_training_cli.py check-updates --ticker RELIANCE --mode simple
  
  # Run incremental update for a ticker
  python incremental_training_cli.py update --ticker RELIANCE --mode simple
  
  # Run automatic updates for multiple tickers
  python incremental_training_cli.py auto-update --tickers RELIANCE AAPL MSFT
  
  # View version history
  python incremental_training_cli.py versions --ticker RELIANCE
  
  # Rollback to a specific version
  python incremental_training_cli.py rollback --ticker RELIANCE --version-id RELIANCE_simple_20241201_143022
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check updates command
    check_parser = subparsers.add_parser('check-updates', help='Check if models need updates')
    check_parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    check_parser.add_argument('--mode', default='simple', choices=['simple', 'advanced'], 
                             help='Model mode (default: simple)')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update a specific model')
    update_parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    update_parser.add_argument('--mode', default='simple', choices=['simple', 'advanced'], 
                              help='Model mode (default: simple)')
    update_parser.add_argument('--force', action='store_true', 
                              help='Force update even if not needed')
    update_parser.add_argument('--full-retrain', action='store_true', 
                              help='Perform full retraining instead of incremental')
    
    # Auto update command
    auto_parser = subparsers.add_parser('auto-update', help='Run automatic updates for multiple tickers')
    auto_parser.add_argument('--tickers', nargs='+', required=True, 
                            help='List of stock ticker symbols')
    auto_parser.add_argument('--modes', nargs='+', default=['simple', 'advanced'], 
                            choices=['simple', 'advanced'], help='Model modes to update')
    auto_parser.add_argument('--output', help='Output file for results (JSON)')
    
    # Versions command
    versions_parser = subparsers.add_parser('versions', help='View model version history')
    versions_parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    versions_parser.add_argument('--mode', choices=['simple', 'advanced'], 
                                help='Filter by model mode')
    versions_parser.add_argument('--output', help='Output file for version history (JSON)')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to a specific version')
    rollback_parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    rollback_parser.add_argument('--version-id', required=True, help='Version ID to rollback to')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old model versions')
    cleanup_parser.add_argument('--ticker', required=True, help='Stock ticker symbol')
    cleanup_parser.add_argument('--mode', choices=['simple', 'advanced'], 
                               help='Filter by model mode')
    cleanup_parser.add_argument('--max-versions', type=int, default=10, 
                               help='Maximum versions to keep (default: 10)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show incremental learning status')
    status_parser.add_argument('--ticker', help='Filter by ticker symbol')
    status_parser.add_argument('--output', help='Output file for status (JSON)')
    
    return parser


def check_updates(args):
    """Check if models need updates"""
    pipeline = create_model_update_pipeline()
    
    logger.info(f"Checking for updates: {args.ticker} ({args.mode})")
    
    result = pipeline.check_for_updates(args.ticker, args.mode)
    
    print(f"\n📊 Update Check Results for {args.ticker} ({args.mode})")
    print("=" * 50)
    print(f"Needs Update: {'✅ Yes' if result['needs_update'] else '❌ No'}")
    print(f"Reason: {result['reason']}")
    print(f"Update Type: {result['update_type']}")
    
    if result['data_freshness']:
        freshness = result['data_freshness']
        print(f"Data Freshness: {freshness['days_since_last_update']} days")
        print(f"Latest Data: {freshness['latest_data_date']}")
    
    if result['latest_version']:
        print(f"Latest Version: {result['latest_version']}")
    
    return result


def update_model(args):
    """Update a specific model"""
    pipeline = create_model_update_pipeline()
    
    logger.info(f"Updating model: {args.ticker} ({args.mode})")
    
    if args.force or args.full_retrain:
        # Force full retraining
        result = pipeline.perform_full_retraining(args.ticker, args.mode)
    else:
        # Check if update is needed first
        update_check = pipeline.check_for_updates(args.ticker, args.mode)
        
        if not update_check['needs_update']:
            print(f"❌ No update needed for {args.ticker} ({args.mode})")
            print(f"Reason: {update_check['reason']}")
            return update_check
        
        # Prepare update data
        update_data = pipeline.prepare_update_data(args.ticker, args.mode)
        
        if update_data is None:
            print(f"❌ No update data available for {args.ticker} ({args.mode})")
            return {'success': False, 'error': 'No update data available'}
        
        # Perform incremental update
        result = pipeline.perform_incremental_update(args.ticker, args.mode, update_data)
    
    print(f"\n🔄 Update Results for {args.ticker} ({args.mode})")
    print("=" * 50)
    
    if result['success']:
        print("✅ Update completed successfully!")
        if 'version_id' in result:
            print(f"New Version: {result['version_id']}")
        if 'performance_improvement' in result:
            improvement = result['performance_improvement'] * 100
            print(f"Performance Improvement: {improvement:.2f}%")
        if 'backup_path' in result:
            print(f"Backup Created: {result['backup_path']}")
    else:
        print("❌ Update failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return result


def auto_update(args):
    """Run automatic updates for multiple tickers"""
    pipeline = create_model_update_pipeline()
    
    logger.info(f"Running automatic updates for {len(args.tickers)} tickers")
    
    results = pipeline.run_automatic_updates(args.tickers, args.modes)
    
    print(f"\n🤖 Automatic Update Results")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    for ticker, ticker_results in results.items():
        print(f"\n📈 {ticker}:")
        for mode, result in ticker_results.items():
            total_count += 1
            status = result['status']
            
            if status == 'completed':
                success_count += 1
                print(f"  ✅ {mode}: {status}")
            elif status == 'no_update_needed':
                print(f"  ⏭️  {mode}: {status}")
            elif status == 'no_data_available':
                print(f"  ⚠️  {mode}: {status}")
            else:
                print(f"  ❌ {mode}: {status}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
    
    print(f"\n📊 Summary: {success_count}/{total_count} updates completed successfully")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")
    
    return results


def show_versions(args):
    """Show model version history"""
    pipeline = create_model_update_pipeline()
    
    logger.info(f"Getting version history for {args.ticker}")
    
    versions = pipeline.get_model_version_history(args.ticker, args.mode)
    
    print(f"\n📚 Version History for {args.ticker}")
    if args.mode:
        print(f"Mode: {args.mode}")
    print("=" * 50)
    
    if not versions:
        print("No versions found.")
        return versions
    
    for i, version in enumerate(versions, 1):
        print(f"\n{i}. Version: {version['version_id']}")
        print(f"   Created: {version['created_at']}")
        print(f"   Mode: {version['metadata'].get('mode', 'N/A')}")
        print(f"   Training Samples: {version['training_samples']}")
        
        if version['performance_metrics']:
            metrics = version['performance_metrics']
            print(f"   RMSE: {metrics.get('rmse', 'N/A'):.6f}")
            print(f"   MAE: {metrics.get('mae', 'N/A'):.6f}")
        
        if 'improvement' in version['metadata']:
            improvement = version['metadata']['improvement'] * 100
            print(f"   Improvement: {improvement:.2f}%")
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(versions, f, indent=2, default=str)
        print(f"\nVersion history saved to: {args.output}")
    
    return versions


def rollback_model(args):
    """Rollback to a specific version"""
    pipeline = create_model_update_pipeline()
    
    logger.info(f"Rolling back {args.ticker} to version {args.version_id}")
    
    success = pipeline.rollback_model(args.ticker, args.version_id)
    
    print(f"\n🔄 Rollback Results for {args.ticker}")
    print("=" * 50)
    
    if success:
        print(f"✅ Successfully rolled back to version: {args.version_id}")
    else:
        print(f"❌ Failed to rollback to version: {args.version_id}")
    
    return success


def cleanup_versions(args):
    """Clean up old model versions"""
    pipeline = create_model_update_pipeline()
    
    logger.info(f"Cleaning up old versions for {args.ticker}")
    
    pipeline.cleanup_old_versions(args.ticker, args.mode, args.max_versions)
    
    print(f"\n🧹 Cleanup Results for {args.ticker}")
    print("=" * 50)
    print(f"✅ Cleanup completed. Keeping max {args.max_versions} versions.")
    
    # Show remaining versions
    versions = pipeline.get_model_version_history(args.ticker, args.mode)
    print(f"Remaining versions: {len(versions)}")


def show_status(args):
    """Show incremental learning status"""
    pipeline = create_model_update_pipeline()
    
    logger.info("Getting incremental learning status")
    
    # Get all tickers with versions
    all_tickers = list(pipeline.learning_manager.version_registry.keys())
    
    if args.ticker:
        all_tickers = [args.ticker] if args.ticker in all_tickers else []
    
    status_data = {}
    
    print(f"\n📊 Incremental Learning Status")
    print("=" * 50)
    
    for ticker in all_tickers:
        status_data[ticker] = {}
        
        for mode in ['simple', 'advanced']:
            # Get latest version
            latest_version = pipeline.learning_manager.get_latest_version(ticker, mode)
            
            # Check for updates
            update_check = pipeline.check_for_updates(ticker, mode)
            
            status_data[ticker][mode] = {
                'latest_version': latest_version.version_id if latest_version else None,
                'needs_update': update_check['needs_update'],
                'update_type': update_check['update_type'],
                'days_since_update': update_check['data_freshness']['days_since_last_update']
            }
            
            print(f"\n📈 {ticker} ({mode}):")
            if latest_version:
                print(f"   Latest Version: {latest_version.version_id}")
                print(f"   Created: {latest_version.created_at}")
            else:
                print(f"   Latest Version: None")
            
            print(f"   Needs Update: {'✅ Yes' if update_check['needs_update'] else '❌ No'}")
            print(f"   Update Type: {update_check['update_type']}")
            print(f"   Days Since Update: {update_check['data_freshness']['days_since_last_update']}")
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)
        print(f"\nStatus saved to: {args.output}")
    
    return status_data


def main():
    """Main function"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'check-updates':
            check_updates(args)
        elif args.command == 'update':
            update_model(args)
        elif args.command == 'auto-update':
            auto_update(args)
        elif args.command == 'versions':
            show_versions(args)
        elif args.command == 'rollback':
            rollback_model(args)
        elif args.command == 'cleanup':
            cleanup_versions(args)
        elif args.command == 'status':
            show_status(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
