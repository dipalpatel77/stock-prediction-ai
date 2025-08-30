#!/usr/bin/env python3
"""
AI Stock Predictor - Project Cleanup Script
Safely removes temporary, generated, and non-essential files
"""

import os
import shutil
import glob
from pathlib import Path
import sys

class ProjectCleanup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.deleted_files = []
        self.deleted_dirs = []
        self.total_size_saved = 0
        
    def get_file_size(self, file_path):
        """Get file size in bytes."""
        try:
            return os.path.getsize(file_path)
        except:
            return 0
    
    def get_dir_size(self, dir_path):
        """Get directory size in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(dir_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += self.get_file_size(filepath)
        except:
            pass
        return total_size
    
    def format_size(self, size_bytes):
        """Format size in human readable format."""
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def safe_delete_file(self, file_path):
        """Safely delete a file and track statistics."""
        if os.path.exists(file_path):
            size = self.get_file_size(file_path)
            try:
                os.remove(file_path)
                self.deleted_files.append(str(file_path))
                self.total_size_saved += size
                return True, size
            except Exception as e:
                print(f"❌ Failed to delete {file_path}: {e}")
                return False, 0
        return False, 0
    
    def safe_delete_dir(self, dir_path):
        """Safely delete a directory and track statistics."""
        if os.path.exists(dir_path):
            size = self.get_dir_size(dir_path)
            try:
                shutil.rmtree(dir_path)
                self.deleted_dirs.append(str(dir_path))
                self.total_size_saved += size
                return True, size
            except Exception as e:
                print(f"❌ Failed to delete {dir_path}: {e}")
                return False, 0
        return False, 0
    
    def cleanup_generated_data(self):
        """Clean up generated data files."""
        print("🗑️ Cleaning up generated data files...")
        
        # Data directory files
        data_patterns = [
            "data/*.csv",
            "data/*.pkl",
            "data/cache/*",
            "data/angel_one_cache/*"
        ]
        
        for pattern in data_patterns:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    success, size = self.safe_delete_file(file_path)
                    if success:
                        print(f"   ✅ Deleted: {file_path} ({self.format_size(size)})")
        
        # Angel data cache and exports
        angel_patterns = [
            "angel_data/cache/*",
            "angel_data/exports/*"
        ]
        
        for pattern in angel_patterns:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    success, size = self.safe_delete_file(file_path)
                    if success:
                        print(f"   ✅ Deleted: {file_path} ({self.format_size(size)})")
    
    def cleanup_generated_models(self):
        """Clean up generated model files."""
        print("🗑️ Cleaning up generated model files...")
        
        model_patterns = [
            "models/*.pkl",
            "models/*.h5",
            "models/cache/*"
        ]
        
        for pattern in model_patterns:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    success, size = self.safe_delete_file(file_path)
                    if success:
                        print(f"   ✅ Deleted: {file_path} ({self.format_size(size)})")
    
    def cleanup_cache_directories(self):
        """Clean up cache directories."""
        print("🗑️ Cleaning up cache directories...")
        
        cache_dirs = [
            "logs",
            "catboost_info",
            "reports"
        ]
        
        for dir_name in cache_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                success, size = self.safe_delete_dir(dir_path)
                if success:
                    print(f"   ✅ Deleted directory: {dir_name} ({self.format_size(size)})")
    
    def cleanup_python_cache(self):
        """Clean up Python cache files."""
        print("🗑️ Cleaning up Python cache...")
        
        # Find all __pycache__ directories
        for root, dirs, files in os.walk("."):
            for dir_name in dirs:
                if dir_name == "__pycache__":
                    cache_dir = os.path.join(root, dir_name)
                    success, size = self.safe_delete_dir(cache_dir)
                    if success:
                        print(f"   ✅ Deleted: {cache_dir} ({self.format_size(size)})")
        
        # Find all .pyc files
        for root, dirs, files in os.walk("."):
            for file_name in files:
                if file_name.endswith(".pyc"):
                    pyc_file = os.path.join(root, file_name)
                    success, size = self.safe_delete_file(pyc_file)
                    if success:
                        print(f"   ✅ Deleted: {pyc_file} ({self.format_size(size)})")
    
    def cleanup_development_files(self):
        """Clean up development files."""
        print("🗑️ Cleaning up development files...")
        
        dev_dirs = [
            ".vscode",
            "tests",
            "notebooks"
        ]
        
        for dir_name in dev_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                success, size = self.safe_delete_dir(dir_path)
                if success:
                    print(f"   ✅ Deleted directory: {dir_name} ({self.format_size(size)})")
    
    def cleanup_test_files(self):
        """Clean up test files."""
        print("🗑️ Cleaning up test files...")
        
        test_files = [
            "test_historical_data_selection.py",
            "simple_mapper_test.py",
            "test_unified_with_mapper.py",
            "test_caching.py"
        ]
        
        for file_name in test_files:
            file_path = Path(file_name)
            if file_path.exists():
                success, size = self.safe_delete_file(file_path)
                if success:
                    print(f"   ✅ Deleted: {file_name} ({self.format_size(size)})")
    
    def cleanup_documentation_files(self):
        """Clean up documentation files (optional)."""
        print("🗑️ Cleaning up documentation files...")
        
        doc_patterns = [
            "*SUMMARY.md",
            "*GUIDE.md",
            "*COMPARISON.md"
        ]
        
        for pattern in doc_patterns:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path) and file_path != "README.md":
                    success, size = self.safe_delete_file(file_path)
                    if success:
                        print(f"   ✅ Deleted: {file_path} ({self.format_size(size)})")
    
    def run_cleanup(self, include_tests=True, include_docs=False):
        """Run the complete cleanup process."""
        print("🚀 AI Stock Predictor - Project Cleanup")
        print("=" * 50)
        print("This script will safely remove temporary and generated files.")
        print("Core functionality will be preserved.")
        print()
        
        # Phase 1: Safe deletions
        self.cleanup_generated_data()
        self.cleanup_generated_models()
        self.cleanup_cache_directories()
        self.cleanup_python_cache()
        self.cleanup_development_files()
        
        # Phase 2: Optional deletions
        if include_tests:
            self.cleanup_test_files()
        
        if include_docs:
            self.cleanup_documentation_files()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print cleanup summary."""
        print("\n" + "=" * 50)
        print("📊 CLEANUP SUMMARY")
        print("=" * 50)
        print(f"🗑️ Files deleted: {len(self.deleted_files)}")
        print(f"🗑️ Directories deleted: {len(self.deleted_dirs)}")
        print(f"💾 Total space saved: {self.format_size(self.total_size_saved)}")
        print()
        
        if self.deleted_files:
            print("📄 Deleted files:")
            for file_path in self.deleted_files[:10]:  # Show first 10
                print(f"   - {file_path}")
            if len(self.deleted_files) > 10:
                print(f"   ... and {len(self.deleted_files) - 10} more files")
        
        if self.deleted_dirs:
            print("\n📁 Deleted directories:")
            for dir_path in self.deleted_dirs:
                print(f"   - {dir_path}")
        
        print("\n✅ Cleanup completed successfully!")
        print("🎯 Core functionality preserved.")
        print("🔄 Generated files can be recreated when needed.")

def main():
    """Main function."""
    cleanup = ProjectCleanup()
    
    # Parse command line arguments
    include_tests = "--keep-tests" not in sys.argv
    include_docs = "--clean-docs" in sys.argv
    
    print("Options:")
    print(f"  Include test files: {include_tests}")
    print(f"  Include documentation: {include_docs}")
    print()
    
    # Confirm before proceeding
    response = input("Proceed with cleanup? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Cleanup cancelled.")
        return
    
    # Run cleanup
    cleanup.run_cleanup(include_tests=include_tests, include_docs=include_docs)

if __name__ == "__main__":
    main()
