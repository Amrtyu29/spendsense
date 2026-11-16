@echo off
rem Double-click to start the SpendSense dashboard.
cd /d "%~dp0"
echo.
echo  Starting SpendSense. Your browser will open at http://localhost:8501
echo  Keep this window open while you use the app. Close it to stop.
echo.
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8501"
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
if errorlevel 1 (
  echo.
  echo  Could not start. Install the requirements first:  pip install -r requirements.txt
  pause
)
