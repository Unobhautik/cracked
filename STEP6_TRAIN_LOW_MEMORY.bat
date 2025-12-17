@echo off
echo ========================================
echo STEP 6: Training Model (Low Memory Mode)
echo ========================================
echo.
echo This version is optimized for systems with limited GPU RAM.
echo.
echo Using low-memory optimizations:
echo - Smaller batch size (1)
echo - Shorter sequences (512 tokens)
echo - CPU offloading enabled
echo - Gradient checkpointing
echo.
echo This may be slower but will work on your system!
echo.
python training/train_model_low_memory.py --model mistralai/Mistral-7B-v0.1 --epochs 2 --seq-length 512
echo.
echo ========================================
echo Training Complete!
echo ========================================
pause


