@echo off
cd /d "%~dp0"
echo Generating navigation demo GIF ...
python nav_demo.py
if errorlevel 1 (
  echo.
  echo Failed. Check your python environment.
  pause
  exit /b 1
)
echo.
echo Done: nav_demo.gif generated.
start "" explorer.exe "%~dp0"
pause
