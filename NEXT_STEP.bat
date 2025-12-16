@echo off
echo ========================================
echo Next Steps After HuggingFace Login
echo ========================================
echo.
echo STEP 1: Request Access to Mistral Model
echo.
echo Opening Mistral model page...
start https://huggingface.co/mistralai/Mistral-7B-v0.1
echo.
echo In the browser that just opened:
echo 1. Click "Agree and access repository" button
echo 2. That's it!
echo.
echo Press any key after you've clicked "Agree and access repository"...
pause >nul
echo.
echo ========================================
echo STEP 2: Start Training Setup
echo ========================================
echo.
echo Now we'll set up everything for training.
echo.
echo Press any key to continue to STEP 1 (Setup)...
pause >nul
echo.
call STEP1_SETUP.bat

