@echo off
echo ========================================
echo STEP 0: HuggingFace Setup (One-Time)
echo ========================================
echo.
echo This is a ONE-TIME setup before training.
echo.
echo Step 1: Installing HuggingFace CLI...
pip install huggingface_hub
echo.
echo Step 2: Login to HuggingFace
echo You need to:
echo 1. Get token from: https://huggingface.co/settings/tokens
echo 2. Click "New token" and copy it (make sure it starts with hf_)
echo 3. Paste it when prompted below
echo.
echo Using new command: hf auth login
hf auth login
echo.
echo Step 3: Request access to Mistral model
echo.
echo Please open this link in your browser:
echo https://huggingface.co/mistralai/Mistral-7B-v0.1
echo.
echo Then click "Agree and access repository"
echo.
echo Press any key after you've done this...
pause >nul
echo.
echo ========================================
echo HuggingFace setup complete!
echo ========================================
echo.
echo Now you can run STEP1_SETUP.bat
pause

