# Copy-Paste Commands - Training Your Medical AI Model

## 🎯 Everything You Need (Copy & Paste)

### Step 1: Setup (Run Once)
```bash
python training/setup_training.py
```

### Step 2: Install Dependencies (Run Once)
```bash
pip install -r training/requirements_training.txt
```

### Step 3: Collect Data (Automated - No Manual Downloads Needed!)
```bash
python training/data_collector.py
```
**What this does automatically:**
- Downloads FDA drug labels from API
- Downloads PubMed abstracts from API
- Downloads HuggingFace medical datasets (MedQuAD, MedMCQA, etc.)
- **NO manual downloads needed!** Everything is automated.

### Step 4: Process Data
```bash
python training/data_processor.py
```

### Step 5: Convert to Training Format
```bash
python training/dataset_converter.py
```

### Step 6: Train Model
```bash
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 3
```

---

## 🚀 OR: Run Everything at Once
```bash
python training/run_training_pipeline.py
```
This runs steps 3, 4, and 5 automatically!

---

## 📋 Complete Sequence (Copy All)

```bash
# Setup
python training/setup_training.py

# Install dependencies
pip install -r training/requirements_training.txt

# Run complete pipeline (collects, processes, converts)
python training/run_training_pipeline.py

# Train model (after data is ready)
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 3
```

---

## ❓ FAQ: Do I Need to Download Datasets Manually?

### ✅ NO! Everything is Automated

**The pipeline automatically downloads:**
- ✅ FDA data (via API - free, no registration)
- ✅ PubMed abstracts (via API - free, no registration)
- ✅ HuggingFace datasets (MedQuAD, MedMCQA, etc. - automatic download)
- ✅ All medical datasets mentioned in your document

**You DON'T need to:**
- ❌ Manually download any datasets
- ❌ Visit FDA/PubMed websites
- ❌ Download files from Kaggle/HuggingFace manually
- ❌ Extract or unzip anything

**The only thing you need:**
- ✅ HuggingFace account (free) - for accessing models like Mistral/Llama
- ✅ Internet connection - for downloading data and models
- ✅ GPU with 16GB+ VRAM - for training

---

## 🔑 HuggingFace Setup (One-Time)

If you don't have HuggingFace access yet:

1. **Create free account**: https://huggingface.co/join
2. **Get access tokens**:
   ```bash
   # Install huggingface-cli
   pip install huggingface_hub
   
   # Login (will open browser)
   huggingface-cli login
   ```
3. **Request model access** (if needed):
   - Mistral: https://huggingface.co/mistralai/Mistral-7B-v0.1
   - Llama: https://huggingface.co/meta-llama/Llama-3-8b
   - Click "Agree and access repository"

---

## ⚡ Quick Test (Verify Everything Works)

```bash
# Test data collection (small sample)
python -c "from training.data_collector import MedicalDataCollector; c = MedicalDataCollector(); c.collect_fda_drug_labels(limit=10); print('Data collection works!')"
```

---

## 🎯 What Gets Downloaded Automatically?

When you run `python training/data_collector.py`:

1. **FDA Drug Labels** → `training/data/raw/fda_drug_labels.json`
2. **PubMed Abstracts** → `training/data/raw/pubmed_abstracts.json`
3. **MedQuAD Dataset** → `training/data/raw/medquad.json`
4. **MedMCQA Dataset** → `training/data/raw/hf_openlifescienceai_medmcqa.json`
5. **Other HF Medical Datasets** → `training/data/raw/hf_*.json`

**All automatic! No manual work needed.**

---

## 💡 Troubleshooting

**If data collection fails:**
```bash
# Check internet connection
ping google.com

# Try again with smaller limit
python -c "from training.data_collector import MedicalDataCollector; c = MedicalDataCollector(); c.collect_fda_drug_labels(limit=100)"
```

**If HuggingFace download fails:**
```bash
# Login to HuggingFace
huggingface-cli login

# Try again
python training/data_collector.py
```

---

## 📊 Expected Output

After running data collection, you should see:
```
training/data/raw/
├── fda_drug_labels.json (500+ drug labels)
├── pubmed_abstracts.json (500+ abstracts)
├── medquad.json (1000+ Q&A pairs)
└── hf_*.json (various medical datasets)
```

**All created automatically!**


