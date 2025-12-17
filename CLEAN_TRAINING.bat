@echo off
echo ========================================
echo Cleaning Previous Training Files
echo ========================================
echo.
echo This will remove:
echo - Previous model checkpoints
echo - Training logs
echo - Cached model files (will re-download)
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul
echo.

echo Removing training models...
if exist "training\models" (
    rmdir /s /q "training\models"
    echo [OK] Removed training models
) else (
    echo [SKIP] No models folder found
)

echo.
echo Removing training logs...
if exist "training\logs" (
    rmdir /s /q "training\logs"
    echo [OK] Removed training logs
) else (
    echo [SKIP] No logs folder found
)

echo.
echo Removing processed datasets (will reprocess)...
if exist "training\data\processed" (
    rmdir /s /q "training\data\processed"
    echo [OK] Removed processed data
) else (
    echo [SKIP] No processed data folder found
)

echo.
echo Removing converted datasets (will reconvert)...
if exist "training\datasets" (
    rmdir /s /q "training\datasets"
    echo [OK] Removed converted datasets
) else (
    echo [SKIP] No datasets folder found
)

echo.
echo Clearing HuggingFace cache (optional - will re-download models)...
echo Do you want to clear HuggingFace cache? This will delete downloaded models.
echo You'll need to re-download them, but it frees up space.
set /p clear_cache="Clear cache? (y/n): "
if /i "%clear_cache%"=="y" (
    if exist "%USERPROFILE%\.cache\huggingface" (
        echo Clearing HuggingFace cache...
        rmdir /s /q "%USERPROFILE%\.cache\huggingface"
        echo [OK] Cleared HuggingFace cache
    )
)

echo.
echo ========================================
echo Cleanup Complete!
echo ========================================
echo.
echo You can now start fresh training.
echo.
pause

