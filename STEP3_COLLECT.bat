@echo off
echo ========================================
echo STEP 3: Collecting data
echo ========================================
echo This automatically downloads all datasets!
echo No manual downloads needed - everything is automatic.
echo This may take 10-30 minutes...
echo.
python training/data_collector.py
echo.
echo Press any key to continue to STEP 4...
pause >nul

