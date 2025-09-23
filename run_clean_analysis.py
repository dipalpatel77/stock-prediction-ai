#!/usr/bin/env python3
"""
Clean Analysis Runner - Suppresses all repetitive output and warnings
"""

import os
import sys
import subprocess
import warnings
from contextlib import redirect_stderr, redirect_stdout

def suppress_output():
    """Suppress all warnings and verbose output."""
    warnings.filterwarnings('ignore')
    
    # Set environment variables to suppress ML library output
    os.environ.update({
        'LIGHTGBM_VERBOSE': '0',
        'XGBOOST_VERBOSE': '0', 
        'CATBOOST_VERBOSE': '0',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'PYTHONWARNINGS': 'ignore',
        'LIGHTGBM_LOG_LEVEL': 'FATAL',
        'XGBOOST_LOG_LEVEL': 'FATAL',
        'OMP_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1'
    })

def run_clean_analysis():
    """Run the analysis with clean output."""
    suppress_output()
    
    # Import and run the main pipeline
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Import the main pipeline
        from main.main import main
        
        # Run with suppressed output
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                with redirect_stdout(devnull):
                    # Only show essential output
                    print("🚀 Starting AI Stock Predictor...")
                    print("📊 Running analysis with clean output...")
                    
                    # Run the main function
                    main()
                    
                    print("✅ Analysis completed successfully!")
                    
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_clean_analysis()
    sys.exit(0 if success else 1)
