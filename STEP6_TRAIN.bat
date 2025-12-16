@echo off
echo ========================================
echo STEP 6: Training your model
echo ========================================
echo This will take 2-6 hours depending on your GPU.
echo Make sure you have a GPU with 16GB+ VRAM!
echo.
echo Starting training...
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 3
echo.
echo ========================================
echo Training Complete!
echo ========================================
pause

