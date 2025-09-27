#!/usr/bin/env python3
"""
AI Stock Predictor - Installation Script
Automated installation with different options
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements(requirements_file):
    """Install requirements from specified file"""
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file {requirements_file} not found")
        return False
    
    print(f"📦 Installing from {requirements_file}...")
    return run_command(f"pip install -r {requirements_file}", f"Install {requirements_file}")

def create_virtual_environment(env_name="venv"):
    """Create virtual environment"""
    if os.path.exists(env_name):
        print(f"✅ Virtual environment {env_name} already exists")
        return True
    
    return run_command(f"python -m venv {env_name}", f"Create virtual environment {env_name}")

def activate_virtual_environment(env_name="venv"):
    """Show activation instructions"""
    if os.name == 'nt':  # Windows
        activate_script = f"{env_name}\\Scripts\\activate"
    else:  # Linux/Mac
        activate_script = f"source {env_name}/bin/activate"
    
    print(f"🔧 To activate virtual environment, run:")
    print(f"   {activate_script}")
    print()

def main():
    parser = argparse.ArgumentParser(description="AI Stock Predictor Installation Script")
    parser.add_argument(
        "--type", 
        choices=["minimal", "full", "production", "dev"], 
        default="full",
        help="Installation type (default: full)"
    )
    parser.add_argument(
        "--venv", 
        action="store_true",
        help="Create virtual environment"
    )
    parser.add_argument(
        "--upgrade", 
        action="store_true",
        help="Upgrade pip before installation"
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true",
        help="Install without cache"
    )
    
    args = parser.parse_args()
    
    print("🚀 AI Stock Predictor Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip if requested
    if args.upgrade:
        run_command("python -m pip install --upgrade pip", "Upgrade pip")
    
    # Create virtual environment if requested
    if args.venv:
        if not create_virtual_environment():
            sys.exit(1)
        activate_virtual_environment()
        print("⚠️  Please activate the virtual environment before continuing")
        print("   Run the activation command above, then run this script again")
        return
    
    # Determine requirements file
    requirements_map = {
        "minimal": "requirements-minimal.txt",
        "full": "requirements.txt", 
        "production": "requirements-production.txt",
        "dev": "requirements-dev.txt"
    }
    
    requirements_file = requirements_map[args.type]
    
    # Prepare pip command
    pip_cmd = "pip install"
    if args.no_cache:
        pip_cmd += " --no-cache-dir"
    
    # Install requirements
    if not install_requirements(requirements_file):
        sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print(f"📋 Installed: {args.type} requirements")
    print("\n🔧 Next steps:")
    print("1. Configure your API keys in .env file")
    print("2. Run: python main.py --help")
    print("3. Test with: python main.py --quick RELIANCE")
    
    if args.type == "dev":
        print("\n🛠️  Development tools installed:")
        print("- pytest: Testing framework")
        print("- black: Code formatting")
        print("- flake8: Code linting")
        print("- mypy: Type checking")

if __name__ == "__main__":
    main()
