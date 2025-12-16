@echo off
echo ========================================
echo Fix Token Error - Step by Step
echo ========================================
echo.
echo The error means your token is INVALID.
echo.
echo ========================================
echo STEP 1: Open Browser
echo ========================================
echo.
echo I'll open the token page for you...
echo.
start https://huggingface.co/settings/tokens
echo.
echo ========================================
echo STEP 2: Create New Token
echo ========================================
echo.
echo In the browser that just opened:
echo.
echo 1. Make sure you're LOGGED IN to HuggingFace
echo 2. Delete ALL old tokens (if any)
echo 3. Click "New token"
echo 4. Name it: training
echo 5. Select "Read" permission
echo 6. Click "Generate token"
echo 7. COPY THE ENTIRE TOKEN (starts with hf_)
echo 8. Save it in Notepad temporarily
echo.
echo ========================================
echo STEP 3: Login
echo ========================================
echo.
echo Press any key when you have your token ready...
pause >nul
echo.
echo Now paste your token when prompted:
echo (Right-click to paste in PowerShell)
echo.
hf auth login
echo.
echo ========================================
echo STEP 4: Verify
echo ========================================
echo.
hf auth whoami
echo.
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Login worked!
    echo ========================================
    echo.
    echo Now request access to Mistral:
    echo 1. Go to: https://huggingface.co/mistralai/Mistral-7B-v0.1
    echo 2. Click "Agree and access repository"
    echo.
    echo Then you can continue with STEP 1!
    echo.
) else (
    echo.
    echo ========================================
    echo ERROR: Login failed
    echo ========================================
    echo.
    echo Your token is still invalid. Check:
    echo - Token starts with hf_
    echo - No spaces before/after
    echo - Copied completely (all characters)
    echo - Token is fresh (just created)
    echo - You're logged into HuggingFace website
    echo.
    echo Try creating a NEW token and try again.
    echo.
)
pause

