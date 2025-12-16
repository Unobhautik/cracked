@echo off
echo ========================================
echo Fix HuggingFace Login
echo ========================================
echo.
echo The old command is deprecated. Using new command...
echo.
echo Step 1: Open token page in browser
start https://huggingface.co/settings/tokens
echo.
echo Step 2: Create a FRESH token
echo IMPORTANT: Make sure you're LOGGED IN to HuggingFace first!
echo.
echo In the browser:
echo 1. Delete ALL old tokens
echo 2. Click "New token"
echo 3. Name it: training
echo 4. Select "Read" permission
echo 5. Click "Generate token"
echo 6. COPY THE ENTIRE TOKEN (starts with hf_)
echo 7. Save it in Notepad temporarily
echo.
echo Press any key after you have your token ready...
pause >nul
echo.
echo Step 2: Login with new command
hf auth login
echo.
echo Step 3: Verify login
echo.
hf auth whoami
echo.
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Login successful!
    echo ========================================
    echo.
    echo Now request access to Mistral:
    echo 1. Go to: https://huggingface.co/mistralai/Mistral-7B-v0.1
    echo 2. Click "Agree and access repository"
    echo.
) else (
    echo.
    echo ========================================
    echo Login failed. Check your token!
    echo ========================================
    echo.
    echo Make sure:
    echo - Token starts with hf_
    echo - No spaces before/after token
    echo - Token is copied completely
    echo.
)
pause

