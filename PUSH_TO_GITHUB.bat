@echo off
echo ========================================
echo Push Code to GitHub
echo ========================================
echo.
echo Repository: https://github.com/Unobhautik/cracked
echo.

REM Check if git is installed
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Git is not installed!
    echo.
    echo Please install Git from: https://git-scm.com/download/win
    echo Then run this script again.
    pause
    exit /b 1
)

echo Step 1: Initializing git (if not already done)...
git init

echo.
echo Step 2: Adding all files...
git add .

echo.
echo Step 3: Checking if .gitignore exists...
if not exist .gitignore (
    echo Creating .gitignore file...
    (
        echo .env
        echo __pycache__/
        echo *.pyc
        echo *.pyo
        echo *.pyd
        echo .venv/
        echo venv/
        echo *.db
        echo *.db-journal
        echo training/data/raw/
        echo training/data/processed/
        echo training/models/
        echo training/logs/
        echo .DS_Store
        echo *.log
    ) > .gitignore
    git add .gitignore
    echo .gitignore created!
)

echo.
echo Step 4: Committing changes...
git commit -m "Initial commit: Medical AI Agentic System with Training Pipeline"

echo.
echo Step 5: Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/Unobhautik/cracked.git

echo.
echo Step 6: Pushing to GitHub...
echo.
echo You may be asked for GitHub username and password/token.
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS! Code pushed to GitHub!
    echo ========================================
    echo.
    echo Your code is now at:
    echo https://github.com/Unobhautik/cracked
    echo.
) else (
    echo.
    echo ========================================
    echo ERROR: Push failed
    echo ========================================
    echo.
    echo Common issues:
    echo 1. Branch might be 'master' instead of 'main'
    echo 2. Need GitHub authentication
    echo.
    echo Trying with 'master' branch...
    git branch -M master
    git push -u origin master
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo SUCCESS! Pushed to 'master' branch!
    ) else (
        echo.
        echo Still failed. You may need to:
        echo 1. Set up GitHub authentication
        echo 2. Use GitHub Personal Access Token
        echo.
        echo See: https://docs.github.com/en/authentication
    )
)

pause

