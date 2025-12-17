@echo off
echo ========================================
echo STEP 6: Training your model
echo ========================================
echo.
echo NOTE: This is optimized for low-memory systems.
echo If you get memory errors, use STEP6_TRAIN_LOW_MEMORY.bat instead.
echo.
echo This will take 2-6 hours depending on your GPU.
echo.
echo Starting training...
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 2
echo.
echo ========================================
echo Training Complete!
echo ========================================
pause

