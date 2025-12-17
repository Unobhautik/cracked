# 🚀 Model Training Quick Start Guide

## Overview

This guide will help you train your medical AI model efficiently on your personal PC. Everything is optimized for limited hardware and includes comprehensive error handling.

## Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended, 8GB+ VRAM) or CPU (much slower)
- Internet connection (for downloading models and data)
- HuggingFace account (free) - request access to Mistral: https://huggingface.co/mistralai/Mistral-7B-v0.1

## Quick Start (One Command!)

### Windows:
```bash
START_TRAINING.bat
```

### Linux/Mac:
```bash
python training/run_complete_training.py
```

That's it! The script will:
1. ✅ Check and install dependencies
2. ✅ Collect medical data automatically
3. ✅ Process and clean data
4. ✅ Convert to training format
5. ✅ Train your model

## Step-by-Step (If You Prefer)

### Step 1: Install Dependencies
```bash
pip install -r training/requirements_training.txt
```

### Step 2: Collect Data
```bash
python training/data_collector.py
```
This automatically downloads:
- FDA drug labels (via API)
- PubMed abstracts (via API)
- Medical datasets from HuggingFace

### Step 3: Process Data
```bash
python training/data_processor.py
```

### Step 4: Convert to Training Format
```bash
python training/dataset_converter.py
```

### Step 5: Train Model
```bash
python training/train_model_efficient.py
```

## Training Options

### Efficient Training (Recommended)
```bash
python training/train_model_efficient.py --epochs 3 --batch-size 2
```
- Uses QLoRA (4-bit quantization)
- Works on 8GB+ GPU
- Training time: 2-4 hours

### Ultra Low Memory (For 4-6GB GPU)
```bash
python training/train_model_ultra_low_memory.py --epochs 2
```
- More aggressive memory optimizations
- Training time: 3-6 hours

### Custom Options
```bash
python training/train_model_efficient.py \
    --model mistralai/Mistral-7B-v0.1 \
    --epochs 3 \
    --batch-size 2 \
    --seq-length 1024
```

## Using Your Trained Model

### Option 1: Test Your Model
```bash
python training/model_integration.py
```

### Option 2: Use with Agentic AI System

1. **Set environment variables** (create `.env` file):
```env
USE_CUSTOM_MODEL=true
CUSTOM_MODEL_PATH=training/models/medical_ai_model
BASE_MODEL=mistralai/Mistral-7B-v0.1
```

2. **Update your code** to use `config_custom_model.py`:
```python
# In medical_ai.py, change:
from config import DEFAULT_MODEL
# To:
from config_custom_model import DEFAULT_MODEL
```

3. **Run your system** - it will automatically use your trained model!

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `--batch-size 1`
- Reduce sequence length: `--seq-length 512`
- Use ultra low memory script
- Close other GPU applications

### Slow Training
- Normal on CPU (expect 10-20 hours)
- GPU recommended for reasonable speed
- Reduce epochs if needed

### Model Not Found
- Make sure you requested access to Mistral on HuggingFace
- Check internet connection
- Verify HuggingFace login: `huggingface-cli login`

### Data Collection Fails
- Check internet connection
- Some APIs may have rate limits (script handles this)
- You can skip and use existing data if available

## What Gets Created

After training, you'll have:
```
training/models/medical_ai_model/
├── adapter_config.json
├── adapter_model.bin
└── tokenizer files
```

## Integration with Agentic AI

Your trained model seamlessly integrates with:
- ✅ All medical agents (booking, cancellation, symptom analysis, etc.)
- ✅ Safety layer
- ✅ Memory system
- ✅ Frontend (FastAPI)

Just set `USE_CUSTOM_MODEL=true` and your system will use your trained model instead of OpenAI!

## Performance Tips

1. **Use GPU**: Training on CPU is 10-20x slower
2. **Monitor Memory**: Use `nvidia-smi` to check GPU usage
3. **TensorBoard**: Monitor training progress:
   ```bash
   tensorboard --logdir training/models/medical_ai_model/runs
   ```
4. **Save Checkpoints**: Training saves checkpoints automatically

## Next Steps

After training:
1. Test your model: `python training/model_integration.py`
2. Evaluate performance: `python training/evaluate_model.py`
3. Integrate with system: Set `USE_CUSTOM_MODEL=true`
4. Deploy: Your model is ready for production use!

## Support

If you encounter issues:
1. Check error messages carefully
2. Verify all dependencies are installed
3. Check GPU memory availability
4. Review training logs in `training/models/medical_ai_model/runs/`

## Notes

- Training on personal PC is optimized but may be slower than cloud VMs
- Model quality improves with more data and epochs
- You can stop training (Ctrl+C) and resume later
- Checkpoints are saved automatically

Good luck with your training! 🎉


