@echo off
echo ========================================
echo How to Paste Token in PowerShell
echo ========================================
echo.
echo IMPORTANT: You CANNOT use Ctrl+V to paste!
echo.
echo ========================================
echo STEP 1: Copy Your Token
echo ========================================
echo.
echo 1. Go to: https://huggingface.co/settings/tokens
echo 2. Copy your token (Ctrl+C)
echo 3. Make sure it starts with hf_ and has no spaces
echo.
echo ========================================
echo STEP 2: Login Command
echo ========================================
echo.
echo I'll run the login command now.
echo.
echo WHEN IT ASKS FOR TOKEN:
echo - DO NOT use Ctrl+V
echo - RIGHT-CLICK in the PowerShell window
echo - Your token will paste (you won't see it, that's normal!)
echo - Press Enter
echo.
echo Press any key to start login...
pause >nul
echo.
hf auth login
echo.
echo ========================================
echo STEP 3: Verify
echo ========================================
echo.
echo Checking if login worked...
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
) else (
    echo.
    echo ========================================
    echo ERROR: Login failed
    echo ========================================
    echo.
    echo Your token might be wrong. Try:
    echo 1. Create a NEW token
    echo 2. Make sure you copy it completely
    echo 3. Right-click to paste (not Ctrl+V)
    echo.
)
pause

