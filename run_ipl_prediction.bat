@echo off
REM IPL First Innings Score Prediction Web App Launcher
REM This batch file launches the Streamlit web interface

echo ================================================
echo    IPL First Innings Score Prediction
echo    Streamlit Web App
echo ================================================
echo.
echo Starting web server...
echo Your browser will open automatically.
echo.
echo Press Ctrl+C to stop the server
echo.

REM Change to the project directory
cd /d "%~dp0"

REM Run Streamlit app using the virtual environment
"%~dp0.venv\Scripts\streamlit.exe" run streamlit_app.py
