#!/usr/bin/env python3
"""
News Sentiment Analysis Installation Script
Installs required dependencies for news sentiment analysis
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package}: {e}")
        return False

def install_spacy_model():
    """Install spaCy English model"""
    try:
        logger.info("Installing spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        logger.info("✅ spaCy English model installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install spaCy model: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 News Sentiment Analysis Installation")
    print("=" * 50)
    print("This script will install all required dependencies for news sentiment analysis.")
    print("=" * 50)
    
    # Required packages
    packages = [
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "spacy>=3.4.0",
        "newspaper3k>=0.2.8",
        "beautifulsoup4>=4.11.0",
        "requests>=2.28.0",
        "yfinance>=0.1.87",
        "nltk>=3.7",
        "textblob>=0.17.1"
    ]
    
    print(f"📦 Installing {len(packages)} packages...")
    
    success_count = 0
    failed_packages = []
    
    for package in packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    # Install spaCy model
    print("\n📚 Installing spaCy English model...")
    if install_spacy_model():
        success_count += 1
    else:
        failed_packages.append("spacy-en_core_web_sm")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Installation Summary")
    print("=" * 50)
    print(f"✅ Successfully installed: {success_count}/{len(packages) + 1}")
    
    if failed_packages:
        print(f"❌ Failed packages: {failed_packages}")
        print("\n💡 You can try installing failed packages manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
    else:
        print("🎉 All packages installed successfully!")
        print("\n📰 News sentiment analysis is now ready to use!")
        print("   Run: python main.py")
        print("   Select 'Yes' for news sentiment analysis when prompted")

if __name__ == "__main__":
    main()
