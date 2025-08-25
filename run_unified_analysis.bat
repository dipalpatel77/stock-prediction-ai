@echo off
title Unified AI Stock Predictor - Complete Analysis Pipeline
color 0A

echo.
echo ========================================
echo   Unified AI Stock Predictor
echo   Complete Analysis Pipeline
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo ✅ Python found
echo.

:: Check if required packages are installed
echo 🔍 Checking required packages...
python -c "import pandas, numpy, yfinance, sklearn, tensorflow, concurrent.futures" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Some required packages are missing
    echo Installing required packages...
    pip install pandas numpy yfinance scikit-learn tensorflow
    if errorlevel 1 (
        echo ❌ Failed to install required packages
        pause
        exit /b 1
    )
)

echo ✅ Required packages found
echo.

:: Run the unified analysis pipeline
echo 🚀 Starting Unified Analysis Pipeline...
echo.

python unified_analysis_pipeline.py

if errorlevel 1 (
    echo.
    echo ❌ Unified analysis failed!
    echo Check the error messages above for details.
) else (
    echo.
    echo ✅ Unified analysis completed successfully!
    echo.
    echo 📁 Check the generated files in the 'data' and 'models' folders
    echo 📊 Review trading signals and performance reports
)

echo.
echo Press any key to exit...
pause >nul
