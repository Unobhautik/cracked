@echo off
echo ========================================
echo STEP 6: Training (Ultra Low Memory)
echo ========================================
echo.
echo Using smaller model (Qwen2-1.5B) for very low memory systems.
echo This model is much smaller and will work on systems with 4GB or less GPU RAM.
echo.
echo Training will be slower but will work!
echo.
python training/train_model_ultra_low_memory.py --model Qwen/Qwen2-1.5B
echo.
echo ========================================
echo Training Complete!
echo ========================================
pause


