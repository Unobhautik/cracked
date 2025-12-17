@echo off
echo ========================================
echo Clean and Start Fresh Training
echo ========================================
echo.
echo Step 1: Cleaning previous training files...
call CLEAN_TRAINING.bat
echo.

echo ========================================
echo Step 2: Starting Fresh Training
echo ========================================
echo.
echo Now we'll start the training pipeline fresh.
echo.

echo Step 1: Setup directories...
python training/setup_training.py
echo.

echo Step 2: Install dependencies (if needed)...
pip install -r training/requirements_training.txt
echo.

echo Step 3: Collect data (automatic downloads)...
python training/data_collector.py
echo.

echo Step 4: Process data...
python training/data_processor.py
echo.

echo Step 5: Convert to training format...
python training/dataset_converter.py
echo.

echo Step 6: Train model (using smaller model for low memory)...
python training/train_model_ultra_low_memory.py --model Qwen/Qwen2-1.5B
echo.

echo ========================================
echo Training Complete!
echo ========================================
pause


