@echo off
REM This batch file installs the required Python packages and runs the Qlib Visual Tool.

echo [Step 1/2] Installing required Python packages from requirements.txt...
pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo *******************************************************************
    echo *  ERROR: Failed to install Python packages.                      *
    echo *  Please ensure Python and pip are correctly installed and try again. *
    echo *******************************************************************
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [Step 2/2] All packages installed. Starting the Qlib Visual Tool...
echo You can close this window after the application has opened in your browser.
echo.

streamlit run app.py

pause
