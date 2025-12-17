@echo off
echo ========================================
echo STEP 3: Collecting data
echo ========================================
echo.
echo IMPORTANT: This automatically downloads ALL datasets!
echo.
echo NO MANUAL DOWNLOADS NEEDED!
echo - FDA data downloads automatically
echo - PubMed data downloads automatically  
echo - MedQuAD downloads automatically
echo - MedMCQA downloads automatically
echo - All other datasets download automatically
echo.
echo This may take 10-30 minutes...
echo Just wait - everything happens automatically!
echo.
python training/data_collector.py
echo.
echo Press any key to continue to STEP 4...
pause >nul

