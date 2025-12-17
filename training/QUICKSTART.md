# Quick Start Guide - Training Your Medical AI Model

## 🚀 Fast Track (5 Steps)

### Step 1: Install Dependencies
```bash
pip install -r training/requirements_training.txt
```

### Step 2: Collect Data
```bash
python training/data_collector.py
```
⏱️ Takes: 10-30 minutes (depending on internet speed)

### Step 3: Process Data
```bash
python training/data_processor.py
```
⏱️ Takes: 2-5 minutes

### Step 4: Convert to Training Format
```bash
python training/dataset_converter.py
```
⏱️ Takes: 1-2 minutes

### Step 5: Train Model
```bash
# For Mistral 7B (Recommended)
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora

# Or run all steps at once:
python training/run_training_pipeline.py
```
⏱️ Takes: 2-6 hours (depending on GPU)

## 📋 Prerequisites Checklist

- [ ] NVIDIA GPU with 16GB+ VRAM
- [ ] Python 3.8+
- [ ] HuggingFace account (for model access)
- [ ] 100GB+ free disk space

## 🎯 What You'll Get

After training, you'll have:
- ✅ Your own fine-tuned medical AI model
- ✅ Model that understands medical terminology
- ✅ Safety-aware responses
- ✅ Integration-ready model

## 🔧 Integration

After training, integrate your model:

```python
from training.integrate_model import CustomMedicalModel

# Load your model
model = CustomMedicalModel(
    model_path="training/models/medical_ai_model",
    base_model="mistralai/Mistral-7B-v0.1"
)

# Use it
response = model.generate("What are the side effects of aspirin?")
```

## 📊 Expected Results

- **Training Loss**: Should decrease from ~2.5 to ~1.5
- **Validation Loss**: Should track training loss
- **Model Size**: ~14GB (base) + ~100MB (LoRA weights)
- **Inference Speed**: ~10-50 tokens/second (depending on GPU)

## ⚠️ Common Issues

**Out of Memory?**
- Reduce batch size: `--batch-size 2`
- Use QLoRA: `--use-qlora`

**Slow Training?**
- Use smaller model
- Reduce max sequence length

**Poor Results?**
- Train for more epochs: `--epochs 5`
- Collect more data
- Check data quality

## 📚 Next Steps

1. Evaluate: `python training/evaluate_model.py --model-path training/models/medical_ai_model`
2. Integrate: See `training/integrate_model.py`
3. Deploy: Use vLLM or TGI for production

## 🆘 Need Help?

Check `training/README_TRAINING.md` for detailed documentation.



