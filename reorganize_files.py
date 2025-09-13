#!/usr/bin/env python3
"""
File Reorganization Script
Automatically reorganizes the AI Stock Predictor project files
"""

import os
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class FileReorganizer:
    """Handles file reorganization for the AI Stock Predictor project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_before_reorganization"
        
        # Define the new structure
        self.new_structure = {
            "src": {
                "core": [
                    "core/data_service.py",
                    "core/model_service.py", 
                    "core/strategy_service.py",
                    "core/database_service.py",
                    "core/report_generator.py",
                    "core/reporting_service.py",
                    "core/economic_data_service.py",
                    "core/currency_service.py",
                    "core/global_market_service.py",
                    "core/geopolitical_risk_service.py",
                    "core/corporate_action_service.py",
                    "core/insider_trading_service.py",
                    "core/incremental_data_service.py",
                    "core/incremental_service.py",
                    "core/fred_api_service.py"
                ],
                "analysis": [
                    "analysis_modules/short_term_analyzer.py",
                    "analysis_modules/mid_term_analyzer.py", 
                    "analysis_modules/long_term_analyzer.py",
                    "analysis_modules/enhanced_price_forecaster.py"
                ],
                "integrations": [
                    "integrations/phase1_integration.py",
                    "integrations/phase2_integration.py",
                    "integrations/phase3_integration.py",
                    "integrations/comprehensive_report_integration.py"
                ],
                "utils": [
                    "core/angel_one_data_downloader.py",
                    "core/angel_one_config.py",
                    "core/indian_stock_mapper.py",
                    "core/enhanced_date_utils.py"
                ]
            },
            "scripts": {
                "validation": [
                    "quick_validation.py",
                    "validation_dashboard.py", 
                    "prediction_validator.py"
                ],
                "database": [
                    "migrate_to_database.py",
                    "setup_mysql_database.py",
                    "mysql_connection_status.py"
                ],
                "testing": [
                    "test_incremental_efficiency.py"
                ]
            },
            "docs": [
                "VALIDATION_GUIDE.md",
                "DATABASE_IMPLEMENTATION_GUIDE.md",
                "INCREMENTAL_EFFICIENCY_IMPLEMENTATION.md",
                "FILE_ORGANIZATION_PLAN.md"
            ],
            "main": [
                "unified_analysis_pipeline.py"
            ]
        }
        
        # Data organization patterns
        self.data_patterns = {
            "raw_data": "*_raw_data.csv",
            "predictions": "*_predictions.csv", 
            "analysis": "*_analysis.csv",
            "summaries": "*_summary.csv",
            "enhanced": "*_enhanced.csv",
            "preprocessed": "*_preprocessed.csv"
        }
    
    def create_backup(self) -> bool:
        """Create a backup of the current structure."""
        try:
            print("🔄 Creating backup...")
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            # Copy important directories
            important_dirs = ["core", "analysis_modules", "integrations", "config", "data", "models", "tests"]
            for dir_name in important_dirs:
                src_dir = self.project_root / dir_name
                if src_dir.exists():
                    dst_dir = self.backup_dir / dir_name
                    shutil.copytree(src_dir, dst_dir)
            
            # Copy important files
            important_files = [
                "unified_analysis_pipeline.py",
                "quick_validation.py", 
                "validation_dashboard.py",
                "prediction_validator.py",
                "migrate_to_database.py",
                "setup_mysql_database.py",
                "requirements.txt",
                "README.md"
            ]
            
            for file_name in important_files:
                src_file = self.project_root / file_name
                if src_file.exists():
                    dst_file = self.backup_dir / file_name
                    shutil.copy2(src_file, dst_file)
            
            print(f"✅ Backup created at: {self.backup_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def create_new_structure(self) -> bool:
        """Create the new directory structure."""
        try:
            print("🔄 Creating new directory structure...")
            
            # Create main directories
            directories = [
                "src/core",
                "src/analysis", 
                "src/integrations",
                "src/utils",
                "scripts/validation",
                "scripts/database",
                "scripts/testing",
                "docs",
                "main",
                "data/raw/us_stocks",
                "data/raw/indian_stocks", 
                "data/processed/short_term",
                "data/processed/mid_term",
                "data/processed/long_term",
                "data/predictions",
                "data/analysis",
                "data/cache",
                "models/short_term",
                "models/mid_term", 
                "models/long_term",
                "models/scalers",
                "reports/daily",
                "reports/weekly",
                "reports/monthly",
                "reports/validation",
                "logs",
                "temp"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            print("✅ New directory structure created")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create structure: {e}")
            return False
    
    def move_source_files(self) -> bool:
        """Move source files to new locations."""
        try:
            print("🔄 Moving source files...")
            
            for category, subcategories in self.new_structure.items():
                if isinstance(subcategories, dict):
                    # Handle nested structure (src, scripts)
                    for subcategory, files in subcategories.items():
                        target_dir = self.project_root / category / subcategory
                        for file_path in files:
                            src_file = self.project_root / file_path
                            if src_file.exists():
                                dst_file = target_dir / src_file.name
                                shutil.move(str(src_file), str(dst_file))
                                print(f"  📁 Moved: {file_path} -> {category}/{subcategory}/")
                else:
                    # Handle flat structure (docs, main)
                    target_dir = self.project_root / category
                    for file_path in subcategories:
                        src_file = self.project_root / file_path
                        if src_file.exists():
                            dst_file = target_dir / src_file.name
                            shutil.move(str(src_file), str(dst_file))
                            print(f"  📁 Moved: {file_path} -> {category}/")
            
            print("✅ Source files moved")
            return True
            
        except Exception as e:
            print(f"❌ Failed to move source files: {e}")
            return False
    
    def organize_data_files(self) -> bool:
        """Organize data files by type and ticker."""
        try:
            print("🔄 Organizing data files...")
            
            data_dir = self.project_root / "data"
            if not data_dir.exists():
                print("⚠️ No data directory found")
                return True
            
            # Get all CSV files in data directory
            csv_files = list(data_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                file_name = csv_file.name
                
                # Determine file type and ticker
                ticker = self._extract_ticker(file_name)
                file_type = self._determine_file_type(file_name)
                
                if ticker and file_type:
                    # Create ticker directory if it doesn't exist
                    ticker_dir = data_dir / "by_ticker" / ticker
                    ticker_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move file to appropriate location
                    dst_file = ticker_dir / file_name
                    shutil.move(str(csv_file), str(dst_file))
                    print(f"  📊 Moved: {file_name} -> data/by_ticker/{ticker}/")
                else:
                    # Move to general location
                    general_dir = data_dir / "by_type" / "general"
                    general_dir.mkdir(parents=True, exist_ok=True)
                    dst_file = general_dir / file_name
                    shutil.move(str(csv_file), str(dst_file))
                    print(f"  📊 Moved: {file_name} -> data/by_type/general/")
            
            print("✅ Data files organized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to organize data files: {e}")
            return False
    
    def _extract_ticker(self, filename: str) -> str:
        """Extract ticker symbol from filename."""
        # Remove file extension
        name = filename.replace('.csv', '')
        
        # Split by underscore and take first part
        parts = name.split('_')
        if parts:
            ticker = parts[0]
            # Clean up ticker (remove any non-alphanumeric except dots)
            ticker = ''.join(c for c in ticker if c.isalnum() or c == '.')
            return ticker
        return ""
    
    def _determine_file_type(self, filename: str) -> str:
        """Determine the type of data file."""
        name = filename.lower()
        
        if 'raw_data' in name:
            return 'raw_data'
        elif 'predictions' in name:
            return 'predictions'
        elif 'analysis' in name:
            return 'analysis'
        elif 'summary' in name:
            return 'summaries'
        elif 'enhanced' in name:
            return 'enhanced'
        elif 'preprocessed' in name:
            return 'preprocessed'
        else:
            return 'general'
    
    def move_models(self) -> bool:
        """Move model files to organized structure."""
        try:
            print("🔄 Organizing model files...")
            
            models_dir = self.project_root / "models"
            if not models_dir.exists():
                print("⚠️ No models directory found")
                return True
            
            # Find all model files
            model_files = []
            for pattern in ["*.pkl", "*.h5", "*.joblib"]:
                model_files.extend(models_dir.glob(f"**/{pattern}"))
            
            for model_file in model_files:
                file_name = model_file.name
                
                # Determine timeframe
                if 'short_term' in file_name:
                    target_dir = models_dir / "short_term"
                elif 'mid_term' in file_name:
                    target_dir = models_dir / "mid_term"
                elif 'long_term' in file_name:
                    target_dir = models_dir / "long_term"
                elif 'scaler' in file_name:
                    target_dir = models_dir / "scalers"
                else:
                    target_dir = models_dir / "general"
                
                target_dir.mkdir(parents=True, exist_ok=True)
                dst_file = target_dir / file_name
                shutil.move(str(model_file), str(dst_file))
                print(f"  🤖 Moved: {file_name} -> models/{target_dir.name}/")
            
            print("✅ Model files organized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to organize model files: {e}")
            return False
    
    def move_cache_files(self) -> bool:
        """Move cache files to organized structure."""
        try:
            print("🔄 Organizing cache files...")
            
            # Move various cache directories
            cache_dirs = ["cache", "angel_data", "catboost_info"]
            
            for cache_dir_name in cache_dirs:
                cache_dir = self.project_root / cache_dir_name
                if cache_dir.exists():
                    target_dir = self.project_root / "data" / "cache" / cache_dir_name
                    target_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(cache_dir), str(target_dir))
                    print(f"  💾 Moved: {cache_dir_name} -> data/cache/")
            
            print("✅ Cache files organized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to organize cache files: {e}")
            return False
    
    def create_init_files(self) -> bool:
        """Create __init__.py files for Python packages."""
        try:
            print("🔄 Creating __init__.py files...")
            
            python_dirs = [
                "src",
                "src/core",
                "src/analysis", 
                "src/integrations",
                "src/utils",
                "scripts",
                "scripts/validation",
                "scripts/database",
                "scripts/testing",
                "main"
            ]
            
            for dir_name in python_dirs:
                init_file = self.project_root / dir_name / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"  📄 Created: {dir_name}/__init__.py")
            
            print("✅ __init__.py files created")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create __init__.py files: {e}")
            return False
    
    def reorganize(self, create_backup: bool = True) -> bool:
        """Main reorganization function."""
        print("🚀 Starting file reorganization...")
        print("=" * 60)
        
        try:
            # Step 1: Create backup
            if create_backup:
                if not self.create_backup():
                    return False
            
            # Step 2: Create new structure
            if not self.create_new_structure():
                return False
            
            # Step 3: Move source files
            if not self.move_source_files():
                return False
            
            # Step 4: Organize data files
            if not self.organize_data_files():
                return False
            
            # Step 5: Move models
            if not self.move_models():
                return False
            
            # Step 6: Move cache files
            if not self.move_cache_files():
                return False
            
            # Step 7: Create __init__.py files
            if not self.create_init_files():
                return False
            
            print("=" * 60)
            print("✅ File reorganization completed successfully!")
            print(f"📁 Backup available at: {self.backup_dir}")
            print("\n💡 Next steps:")
            print("1. Update import statements in your code")
            print("2. Test the reorganized structure")
            print("3. Update any hardcoded paths")
            print("4. Remove backup if everything works correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Reorganization failed: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Reorganize AI Stock Predictor files")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Skip creating backup (not recommended)")
    parser.add_argument("--project-root", default=".", 
                       help="Project root directory (default: current directory)")
    
    args = parser.parse_args()
    
    reorganizer = FileReorganizer(args.project_root)
    success = reorganizer.reorganize(create_backup=not args.no_backup)
    
    if success:
        print("\n🎉 Reorganization completed successfully!")
    else:
        print("\n❌ Reorganization failed. Check the errors above.")

if __name__ == "__main__":
    main()
