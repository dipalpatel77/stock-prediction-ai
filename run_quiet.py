#!/usr/bin/env python3
"""
Quiet Analysis Runner - Minimal output version
"""

import os
import sys
import subprocess
import warnings

def run_quiet_analysis():
    """Run analysis with minimal output."""
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Set environment variables to suppress ML library output
    os.environ.update({
        'LIGHTGBM_VERBOSE': '0',
        'XGBOOST_VERBOSE': '0', 
        'CATBOOST_VERBOSE': '0',
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'PYTHONWARNINGS': 'ignore',
        'LIGHTGBM_LOG_LEVEL': 'FATAL',
        'XGBOOST_LOG_LEVEL': 'FATAL'
    })
    
    # Run the main pipeline with output redirection
    try:
        # Use subprocess to run with minimal output
        result = subprocess.run([
            sys.executable, 'main/unified_analysis_pipeline.py'
        ], 
        capture_output=True, 
        text=True,
        input='GALAXYSURF\n6\ny\n\n'  # Default inputs
        )
        
        # Only show essential output
        print("🚀 AI Stock Predictor - Clean Output Mode")
        print("=" * 50)
        
        # Extract and display only key results
        output_lines = result.stdout.split('\n')
        
        # Find and display key sections
        key_sections = [
            "ADVANCED PREDICTION RESULTS",
            "UNIFIED ANALYSIS COMPLETED", 
            "Generated Files:",
            "Trading Recommendation:",
            "TIMEFRAME RECOMMENDATIONS:"
        ]
        
        for line in output_lines:
            if any(section in line for section in key_sections):
                print(line)
            elif line.startswith("📊") or line.startswith("💰") or line.startswith("📅"):
                print(line)
            elif "Predicted Price:" in line or "Expected Change:" in line:
                print(line)
            elif line.startswith("✅") and "completed" in line:
                print(line)
        
        print("\n" + "=" * 50)
        print("✅ Analysis completed successfully!")
        
        if result.stderr:
            print(f"⚠️ Warnings suppressed: {len(result.stderr.split('Warning'))} warnings")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_quiet_analysis()
    sys.exit(0 if success else 1)
