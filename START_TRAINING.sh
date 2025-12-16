#!/bin/bash

echo "========================================"
echo "Medical AI Model Training - Linux/Mac"
echo "========================================"
echo ""

echo "Step 1: Setting up directories..."
python training/setup_training.py
echo ""

echo "Step 2: Installing dependencies..."
pip install -r training/requirements_training.txt
echo ""

echo "Step 3: Starting data collection and training pipeline..."
echo "This will automatically download all datasets - no manual work needed!"
echo ""
python training/run_training_pipeline.py
echo ""

echo "Step 4: Training model..."
echo "This may take 2-6 hours depending on your GPU."
echo ""
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 3
echo ""

echo "========================================"
echo "Training Complete!"
echo "========================================"

