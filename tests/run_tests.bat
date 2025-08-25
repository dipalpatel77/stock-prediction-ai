@echo off
echo ========================================
echo STOCK PREDICTION SYSTEM IMPROVEMENTS TEST
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Running improvements test...
echo.

REM Run the test script
python tests\integration\test_system_improvements.py

echo.
echo ========================================
echo Test completed!
echo ========================================

REM Keep window open
pause
