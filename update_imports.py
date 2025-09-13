#!/usr/bin/env python3
"""
Import Statement Updater
Updates import statements after file reorganization
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

class ImportUpdater:
    """Updates import statements after file reorganization."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
        # Define import mappings
        self.import_mappings = {
            # Core services
            "from core.data_service import": "from src.core.data_service import",
            "from core.model_service import": "from src.core.model_service import",
            "from core.strategy_service import": "from src.core.strategy_service import",
            "from core.database_service import": "from src.core.database_service import",
            "from core.report_generator import": "from src.core.report_generator import",
            "from core.reporting_service import": "from src.core.reporting_service import",
            "from core.economic_data_service import": "from src.core.economic_data_service import",
            "from core.currency_service import": "from src.core.currency_service import",
            "from core.global_market_service import": "from src.core.global_market_service import",
            "from core.geopolitical_risk_service import": "from src.core.geopolitical_risk_service import",
            "from core.corporate_action_service import": "from src.core.corporate_action_service import",
            "from core.insider_trading_service import": "from src.core.insider_trading_service import",
            "from core.incremental_data_service import": "from src.core.incremental_data_service import",
            "from core.incremental_service import": "from src.core.incremental_service import",
            "from core.fred_api_service import": "from src.core.fred_api_service import",
            
            # Analysis modules
            "from analysis_modules.short_term_analyzer import": "from src.analysis.short_term_analyzer import",
            "from analysis_modules.mid_term_analyzer import": "from src.analysis.mid_term_analyzer import",
            "from analysis_modules.long_term_analyzer import": "from src.analysis.long_term_analyzer import",
            "from analysis_modules.enhanced_price_forecaster import": "from src.analysis.enhanced_price_forecaster import",
            
            # Integrations
            "from integrations.phase1_integration import": "from src.integrations.phase1_integration import",
            "from integrations.phase2_integration import": "from src.integrations.phase2_integration import",
            "from integrations.phase3_integration import": "from src.integrations.phase3_integration import",
            "from integrations.comprehensive_report_integration import": "from src.integrations.comprehensive_report_integration import",
            
            # Utils
            "from core.angel_one_data_downloader import": "from src.utils.angel_one_data_downloader import",
            "from core.angel_one_config import": "from src.utils.angel_one_config import",
            "from core.indian_stock_mapper import": "from src.utils.indian_stock_mapper import",
            "from core.enhanced_date_utils import": "from src.utils.enhanced_date_utils import",
            
            # Config
            "from config.analysis_config import": "from config.analysis_config import",
            "from config.database_config import": "from config.database_config import",
            "from config.incremental_config import": "from config.incremental_config import",
            "from config.fred_api_config import": "from config.fred_api_config import",
            "from config.data_periods_config import": "from config.data_periods_config import",
            
            # Relative imports
            "from .core.data_service import": "from ..core.data_service import",
            "from .core.model_service import": "from ..core.model_service import",
            "from .analysis_modules.short_term_analyzer import": "from ..analysis.short_term_analyzer import",
            "from .integrations.phase1_integration import": "from ..integrations.phase1_integration import",
        }
        
        # File patterns to update
        self.file_patterns = [
            "src/**/*.py",
            "scripts/**/*.py", 
            "main/*.py",
            "tests/*.py"
        ]
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files that need import updates."""
        python_files = []
        
        for pattern in self.file_patterns:
            files = list(self.project_root.glob(pattern))
            python_files.extend(files)
        
        return python_files
    
    def update_file_imports(self, file_path: Path) -> bool:
        """Update import statements in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import mappings
            for old_import, new_import in self.import_mappings.items():
                content = content.replace(old_import, new_import)
            
            # Update relative imports based on file location
            content = self._update_relative_imports(content, file_path)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✅ Updated: {file_path.relative_to(self.project_root)}")
                return True
            else:
                print(f"  ⏭️ No changes: {file_path.relative_to(self.project_root)}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error updating {file_path}: {e}")
            return False
    
    def _update_relative_imports(self, content: str, file_path: Path) -> str:
        """Update relative imports based on file location."""
        # This is a simplified version - you might need more sophisticated logic
        # based on your specific import patterns
        
        # Count directory depth from project root
        relative_path = file_path.relative_to(self.project_root)
        depth = len(relative_path.parts) - 1
        
        # Update relative imports
        if depth > 0:
            # Files in subdirectories need to go up
            prefix = "." * depth
            content = re.sub(r'from \.\.', f'from {prefix}', content)
        
        return content
    
    def update_all_imports(self) -> bool:
        """Update imports in all Python files."""
        print("🔄 Updating import statements...")
        
        python_files = self.find_python_files()
        updated_count = 0
        
        for file_path in python_files:
            if self.update_file_imports(file_path):
                updated_count += 1
        
        print(f"✅ Updated {updated_count} files")
        return True
    
    def create_import_test_script(self) -> bool:
        """Create a test script to verify imports work."""
        try:
            test_script = self.project_root / "test_imports.py"
            
            test_content = '''#!/usr/bin/env python3
"""
Import Test Script
Tests that all imports work after reorganization
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test all imports."""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        from src.core.data_service import DataService
        from src.core.model_service import ModelService
        from src.core.strategy_service import StrategyService
        from src.core.database_service import DatabaseService
        print("✅ Core services imported successfully")
        
        # Test analysis imports
        from src.analysis.short_term_analyzer import ShortTermAnalyzer
        from src.analysis.mid_term_analyzer import MidTermAnalyzer
        from src.analysis.long_term_analyzer import LongTermAnalyzer
        print("✅ Analysis modules imported successfully")
        
        # Test integration imports
        from src.integrations.phase1_integration import Phase1Integration
        from src.integrations.phase2_integration import Phase2Integration
        from src.integrations.phase3_integration import Phase3Integration
        print("✅ Integration modules imported successfully")
        
        # Test config imports
        from config.analysis_config import AnalysisConfig
        from config.database_config import DatabaseConfig
        print("✅ Config modules imported successfully")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
'''
            
            with open(test_script, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            print("✅ Created import test script: test_imports.py")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create test script: {e}")
            return False

def main():
    """Main function."""
    updater = ImportUpdater()
    
    print("🚀 Starting import statement updates...")
    print("=" * 60)
    
    try:
        # Update imports
        if not updater.update_all_imports():
            return False
        
        # Create test script
        if not updater.create_import_test_script():
            return False
        
        print("=" * 60)
        print("✅ Import updates completed!")
        print("\n💡 Next steps:")
        print("1. Run: python test_imports.py")
        print("2. Fix any remaining import issues")
        print("3. Test your main application")
        
        return True
        
    except Exception as e:
        print(f"❌ Import update failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
