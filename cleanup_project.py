#!/usr/bin/env python3
"""
Project Cleanup Script
Cleans up duplicate files, old data, and optimizes the project structure
"""

import os
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime, timedelta

class ProjectCleanup:
    """Handles project cleanup and optimization."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.cleanup_log = []
        
        # Define cleanup patterns
        self.cleanup_patterns = {
            "duplicate_files": [
                "**/*.pyc",
                "**/__pycache__",
                "**/*.pyo", 
                "**/*.pyd",
                "**/.DS_Store",
                "**/Thumbs.db"
            ],
            "temp_files": [
                "**/*.tmp",
                "**/*.temp",
                "**/*.log",
                "**/*.bak",
                "**/*.swp",
                "**/*.swo"
            ],
            "cache_dirs": [
                "**/cache",
                "**/.cache",
                "**/tmp",
                "**/.tmp"
            ]
        }
        
        # File size thresholds (in MB)
        self.size_thresholds = {
            "large_files": 100,  # 100MB
            "medium_files": 50,  # 50MB
            "small_files": 1     # 1MB
        }
    
    def log_action(self, action: str, details: str):
        """Log cleanup actions."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}: {details}"
        self.cleanup_log.append(log_entry)
        print(f"  {action}: {details}")
    
    def find_duplicate_files(self) -> List[Tuple[Path, Path]]:
        """Find duplicate files in the project."""
        print("🔍 Searching for duplicate files...")
        
        duplicates = []
        file_hashes = {}
        
        # Get all files
        all_files = []
        for pattern in ["**/*.csv", "**/*.json", "**/*.pkl", "**/*.h5"]:
            all_files.extend(self.project_root.glob(pattern))
        
        for file_path in all_files:
            if file_path.is_file():
                try:
                    # Simple duplicate detection by name and size
                    file_key = (file_path.name, file_path.stat().st_size)
                    
                    if file_key in file_hashes:
                        duplicates.append((file_hashes[file_key], file_path))
                    else:
                        file_hashes[file_key] = file_path
                        
                except Exception as e:
                    self.log_action("ERROR", f"Could not process {file_path}: {e}")
        
        return duplicates
    
    def remove_duplicate_files(self, duplicates: List[Tuple[Path, Path]]) -> int:
        """Remove duplicate files."""
        print(f"🗑️ Removing {len(duplicates)} duplicate files...")
        
        removed_count = 0
        for original, duplicate in duplicates:
            try:
                # Keep the one in a more organized location
                if "data/by_ticker" in str(original) or "data/by_type" in str(original):
                    # Keep original, remove duplicate
                    duplicate.unlink()
                    self.log_action("REMOVED", f"Duplicate: {duplicate.relative_to(self.project_root)}")
                    removed_count += 1
                elif "data/by_ticker" in str(duplicate) or "data/by_type" in str(duplicate):
                    # Keep duplicate, remove original
                    original.unlink()
                    self.log_action("REMOVED", f"Original: {original.relative_to(self.project_root)}")
                    removed_count += 1
                else:
                    # Remove the one with longer path (usually less organized)
                    if len(str(original)) > len(str(duplicate)):
                        original.unlink()
                        self.log_action("REMOVED", f"Longer path: {original.relative_to(self.project_root)}")
                    else:
                        duplicate.unlink()
                        self.log_action("REMOVED", f"Longer path: {duplicate.relative_to(self.project_root)}")
                    removed_count += 1
                    
            except Exception as e:
                self.log_action("ERROR", f"Could not remove duplicate: {e}")
        
        return removed_count
    
    def clean_cache_files(self) -> int:
        """Clean cache and temporary files."""
        print("🧹 Cleaning cache and temporary files...")
        
        removed_count = 0
        
        for pattern in self.cleanup_patterns["duplicate_files"]:
            for file_path in self.project_root.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        self.log_action("REMOVED", f"Cache file: {file_path.relative_to(self.project_root)}")
                        removed_count += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        self.log_action("REMOVED", f"Cache dir: {file_path.relative_to(self.project_root)}")
                        removed_count += 1
                except Exception as e:
                    self.log_action("ERROR", f"Could not remove {file_path}: {e}")
        
        return removed_count
    
    def clean_temp_files(self) -> int:
        """Clean temporary files."""
        print("🧹 Cleaning temporary files...")
        
        removed_count = 0
        
        for pattern in self.cleanup_patterns["temp_files"]:
            for file_path in self.project_root.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        self.log_action("REMOVED", f"Temp file: {file_path.relative_to(self.project_root)}")
                        removed_count += 1
                except Exception as e:
                    self.log_action("ERROR", f"Could not remove {file_path}: {e}")
        
        return removed_count
    
    def archive_old_data(self, days_old: int = 30) -> int:
        """Archive old data files."""
        print(f"📦 Archiving data older than {days_old} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archived_count = 0
        
        # Create archive directory
        archive_dir = self.project_root / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        # Find old data files
        data_patterns = ["data/**/*.csv", "data/**/*.json"]
        
        for pattern in data_patterns:
            for file_path in self.project_root.glob(pattern):
                try:
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if file_time < cutoff_date:
                            # Create archive subdirectory
                            relative_path = file_path.relative_to(self.project_root)
                            archive_path = archive_dir / relative_path
                            archive_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Move file to archive
                            shutil.move(str(file_path), str(archive_path))
                            self.log_action("ARCHIVED", f"Old data: {relative_path}")
                            archived_count += 1
                            
                except Exception as e:
                    self.log_action("ERROR", f"Could not archive {file_path}: {e}")
        
        return archived_count
    
    def optimize_large_files(self) -> Dict[str, int]:
        """Identify and report large files."""
        print("📊 Analyzing file sizes...")
        
        size_stats = {
            "large_files": 0,
            "medium_files": 0,
            "small_files": 0,
            "total_size": 0
        }
        
        large_files = []
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    size_stats["total_size"] += size_mb
                    
                    if size_mb > self.size_thresholds["large_files"]:
                        size_stats["large_files"] += 1
                        large_files.append((file_path, size_mb))
                    elif size_mb > self.size_thresholds["medium_files"]:
                        size_stats["medium_files"] += 1
                    else:
                        size_stats["small_files"] += 1
                        
                except Exception as e:
                    self.log_action("ERROR", f"Could not analyze {file_path}: {e}")
        
        # Report large files
        if large_files:
            print("\n📋 Large files found:")
            for file_path, size_mb in sorted(large_files, key=lambda x: x[1], reverse=True):
                print(f"  📁 {file_path.relative_to(self.project_root)}: {size_mb:.1f} MB")
        
        return size_stats
    
    def clean_empty_directories(self) -> int:
        """Remove empty directories."""
        print("🗂️ Removing empty directories...")
        
        removed_count = 0
        
        # Find empty directories (bottom-up)
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):  # Directory is empty
                        dir_path.rmdir()
                        self.log_action("REMOVED", f"Empty dir: {dir_path.relative_to(self.project_root)}")
                        removed_count += 1
                except Exception as e:
                    self.log_action("ERROR", f"Could not remove {dir_path}: {e}")
        
        return removed_count
    
    def generate_cleanup_report(self) -> bool:
        """Generate a cleanup report."""
        try:
            report_file = self.project_root / "cleanup_report.md"
            
            report_content = f"""# 🧹 Project Cleanup Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Cleanup Summary

### Actions Performed:
{chr(10).join(f"- {log}" for log in self.cleanup_log)}

### File Size Analysis:
- Total project size: {self.optimize_large_files()['total_size']:.1f} MB
- Large files (>100MB): {self.optimize_large_files()['large_files']}
- Medium files (50-100MB): {self.optimize_large_files()['medium_files']}
- Small files (<50MB): {self.optimize_large_files()['small_files']}

## 💡 Recommendations

1. **Regular Cleanup**: Run this script weekly to maintain project health
2. **Archive Old Data**: Consider archiving data older than 30 days
3. **Monitor Large Files**: Keep an eye on files larger than 100MB
4. **Version Control**: Use .gitignore to exclude cache and temp files

## 🔄 Next Steps

1. Review the cleanup log above
2. Test your application to ensure nothing important was removed
3. Update .gitignore if needed
4. Consider setting up automated cleanup

---
*This report was generated by the Project Cleanup Script*
"""
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"✅ Cleanup report saved: {report_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate report: {e}")
            return False
    
    def run_cleanup(self, archive_days: int = 30, remove_duplicates: bool = True) -> bool:
        """Run the complete cleanup process."""
        print("🚀 Starting project cleanup...")
        print("=" * 60)
        
        try:
            total_removed = 0
            
            # Step 1: Remove duplicate files
            if remove_duplicates:
                duplicates = self.find_duplicate_files()
                removed = self.remove_duplicate_files(duplicates)
                total_removed += removed
                print(f"✅ Removed {removed} duplicate files")
            
            # Step 2: Clean cache files
            removed = self.clean_cache_files()
            total_removed += removed
            print(f"✅ Removed {removed} cache files")
            
            # Step 3: Clean temp files
            removed = self.clean_temp_files()
            total_removed += removed
            print(f"✅ Removed {removed} temporary files")
            
            # Step 4: Archive old data
            archived = self.archive_old_data(archive_days)
            print(f"✅ Archived {archived} old data files")
            
            # Step 5: Remove empty directories
            removed = self.clean_empty_directories()
            total_removed += removed
            print(f"✅ Removed {removed} empty directories")
            
            # Step 6: Analyze file sizes
            size_stats = self.optimize_large_files()
            print(f"✅ Analyzed {size_stats['total_size']:.1f} MB of data")
            
            # Step 7: Generate report
            self.generate_cleanup_report()
            
            print("=" * 60)
            print(f"🎉 Cleanup completed! Removed {total_removed} items")
            print("📋 Check cleanup_report.md for details")
            
            return True
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up AI Stock Predictor project")
    parser.add_argument("--no-duplicates", action="store_true",
                       help="Skip duplicate file removal")
    parser.add_argument("--archive-days", type=int, default=30,
                       help="Archive data older than N days (default: 30)")
    parser.add_argument("--project-root", default=".",
                       help="Project root directory (default: current directory)")
    
    args = parser.parse_args()
    
    cleanup = ProjectCleanup(args.project_root)
    success = cleanup.run_cleanup(
        archive_days=args.archive_days,
        remove_duplicates=not args.no_duplicates
    )
    
    if success:
        print("\n🎉 Project cleanup completed successfully!")
    else:
        print("\n❌ Project cleanup failed. Check the errors above.")

if __name__ == "__main__":
    main()
