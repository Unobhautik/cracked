# Medical AI Model Training Guide

Complete guide for training your own medical AI model.

## Overview

This training pipeline creates a fine-tuned medical AI model from:
- **Base Models**: Mistral 7B, Llama 3 8B, or Qwen 7B
- **Data Sources**: FDA, PubMed, NHS, CDC, MedQuAD, HuggingFace medical datasets
- **Training Method**: LoRA/QLoRA for efficient fine-tuning

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM (for QLoRA) or 24GB+ (for full fine-tuning)
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space for datasets and models

### Software Requirements
```bash
# Install training dependencies
pip install -r training/requirements_training.txt

# For QLoRA (4-bit training)
pip install bitsandbytes
```

### Access Requirements
- **HuggingFace Account**: For downloading models and datasets
  - Request access to Llama models: https://huggingface.co/meta-llama
  - Request access to Mistral models: https://huggingface.co/mistralai
- **PubMed API**: Free, no registration needed
- **FDA API**: Free, no registration needed

## Quick Start

### Option 1: Run Complete Pipeline
```bash
python training/run_training_pipeline.py
```

This will:
1. Collect data from all sources
2. Process and clean the data
3. Convert to instruction format
4. Prepare for training

### Option 2: Run Steps Individually

#### Step 1: Collect Data
```bash
python training/data_collector.py
```
Collects data from:
- FDA drug labels (500+ examples)
- PubMed abstracts (500+ examples)
- HuggingFace medical datasets
- MedQuAD Q&A dataset

**Output**: `training/data/raw/*.json`

#### Step 2: Process Data
```bash
python training/data_processor.py
```
Cleans and processes raw data:
- Removes HTML, normalizes text
- Extracts relevant fields
- Structures data consistently

**Output**: `training/data/processed/*.json`

#### Step 3: Convert to Training Format
```bash
python training/dataset_converter.py
```
Converts to instruction-following JSONL:
- Q&A pairs
- Drug information
- Symptom analysis
- Safety refusal examples
- Emergency escalation
- Triage recommendations

**Output**: 
- `training/datasets/medical_instruction_dataset_train.jsonl`
- `training/datasets/medical_instruction_dataset_val.jsonl`

#### Step 4: Train Model
```bash
# Mistral 7B (Recommended)
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 3

# Llama 3 8B
python training/train_model.py --model meta-llama/Llama-3-8b --use-qlora --epochs 3

# Qwen 7B
python training/train_model.py --model Qwen/Qwen-7B --use-qlora --epochs 3
```

**Training Options**:
- `--use-qlora`: Use 4-bit quantization (requires 16GB VRAM)
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size (default: 4, adjust based on VRAM)
- `--learning-rate`: Learning rate (default: 2e-4)

**Output**: `training/models/medical_ai_model/`

## Training Configuration

### LoRA Parameters
- **r**: 16 (rank)
- **alpha**: 32 (scaling)
- **dropout**: 0.05
- **target_modules**: ["q_proj", "v_proj", "k_proj", "o_proj"]

### Training Parameters
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 2048
- **Gradient Accumulation**: 4

Adjust these in `training/train_model.py` if needed.

## Monitoring Training

Training logs are saved to TensorBoard:
```bash
tensorboard --logdir training/models/medical_ai_model/runs
```

Watch for:
- **Loss**: Should decrease over time
- **Perplexity**: Should decrease
- **Validation Loss**: Should track training loss

## Using Your Trained Model

After training, integrate your model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "training/models/medical_ai_model")
model = model.merge_and_unload()  # Merge LoRA weights

# Use the model
prompt = "<s>[INST] What are the side effects of ibuprofen? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in training config
- Increase `gradient_accumulation_steps`
- Use QLoRA (4-bit) instead of full fine-tuning

### Slow Training
- Use smaller model (7B instead of 13B+)
- Reduce `max_seq_length`
- Use gradient checkpointing

### Poor Results
- Increase training epochs
- Add more training data
- Adjust learning rate
- Check data quality

## Next Steps

1. **Evaluate Model**: Use evaluation script to test on medical Q&A
2. **Deploy Model**: Use vLLM or TGI for serving
3. **Integrate**: Replace OpenAI API calls with your model
4. **Iterate**: Collect more data, retrain with improvements

## Resources

- [HuggingFace LoRA Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Medical LLM Fine-tuning](https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face)

## Support

For issues or questions:
1. Check training logs
2. Verify data quality
3. Review model configuration
4. Check GPU memory usage



