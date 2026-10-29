@echo off
cd /d "%~dp0"
echo Opening live simulation window (close window to exit) ...
python nav_demo.py --live
if errorlevel 1 (
  echo.
  echo Failed. Check python environment / matplotlib GUI backend.
  pause
)
