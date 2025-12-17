@echo off
echo ========================================
echo STEP 5: Converting to training format
echo ========================================
echo This may take 1-2 minutes...
python training/dataset_converter.py
echo.
echo Press any key to continue to STEP 6...
pause >nul


