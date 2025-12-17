@echo off
echo ============================================================
echo Medical AI Model Training - Complete Pipeline
echo ============================================================
echo.
echo This will run the complete training pipeline:
echo 1. Check dependencies
echo 2. Collect medical data
echo 3. Process data
echo 4. Convert to training format
echo 5. Train the model
echo.
echo NOTE: If you have limited resources (CPU or low GPU memory),
echo       the script will automatically use CPU-optimized training.
echo.
echo This may take several hours. You can stop with Ctrl+C.
echo.
pause

python training/run_complete_training.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo Training failed. Trying fast training version...
    echo ============================================================
    echo.
    python training/train_model_fast.py
)

pause
