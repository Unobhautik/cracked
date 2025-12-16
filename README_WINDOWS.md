# 🪟 Windows Training Guide - SUPER SIMPLE

## ⚠️ IMPORTANT: Do This FIRST (One-Time Setup)

### STEP 0: HuggingFace Setup
**Double-click:** `STEP0_HUGGINGFACE_SETUP.bat`

OR run these commands:

```bash
pip install huggingface_hub
huggingface-cli login
```

Then:
1. Get token from: https://huggingface.co/settings/tokens
2. Click "New token" → Copy it
3. Paste when prompted
4. Request access: https://huggingface.co/mistralai/Mistral-7B-v0.1
5. Click "Agree and access repository"

**This is ONE-TIME only!** After this, you can train anytime.

---

## Then Run These Steps ONE BY ONE

### Method 1: Double-Click Files (Easiest!)

1. **STEP 0** (if not done): `STEP0_HUGGINGFACE_SETUP.bat`
2. **STEP 1**: `STEP1_SETUP.bat`
3. **STEP 2**: `STEP2_INSTALL.bat`
4. **STEP 3**: `STEP3_COLLECT.bat`
5. **STEP 4**: `STEP4_PROCESS.bat`
6. **STEP 5**: `STEP5_CONVERT.bat`
7. **STEP 6**: `STEP6_TRAIN.bat`

**That's it!** Each file will pause when done so you know when to run the next one.

---

### Method 2: Copy-Paste Commands (One at a Time)

Open PowerShell or Command Prompt in the project folder (`E:\agentmed`)

#### STEP 0 (One-Time Setup):
```bash
pip install huggingface_hub
huggingface-cli login
```
Then get token from https://huggingface.co/settings/tokens and request access to https://huggingface.co/mistralai/Mistral-7B-v0.1

#### Command 1:
```bash
python training/setup_training.py
```
**Wait for:** "Setup complete!" then run Command 2

#### Command 2:
```bash
pip install -r training/requirements_training.txt
```
**Wait for:** Installation to finish (5-10 minutes), then run Command 3

#### Command 3:
```bash
python training/data_collector.py
```
**Wait for:** "Data collection complete!" (10-30 minutes), then run Command 4
**Note:** This automatically downloads everything - no manual work needed!

#### Command 4:
```bash
python training/data_processor.py
```
**Wait for:** "Processing complete!" (2-5 minutes), then run Command 5

#### Command 5:
```bash
python training/dataset_converter.py
```
**Wait for:** "Dataset conversion complete!" (1-2 minutes), then run Command 6

#### Command 6:
```bash
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 3
```
**Wait for:** "Training complete!" (2-6 hours)

---

## ✅ Done!

After Command 6 finishes, your model is trained and ready!

---

## 📝 Quick Checklist

- [ ] **STEP 0**: HuggingFace setup (one-time)
- [ ] Step 1: Setup ✓
- [ ] Step 2: Install ✓
- [ ] Step 3: Collect data ✓ (automatic downloads)
- [ ] Step 4: Process ✓
- [ ] Step 5: Convert ✓
- [ ] Step 6: Train ✓

---

## 🆘 Help

**If a command fails:**
- Make sure you're in the right folder: `cd E:\agentmed`
- Check the error message
- Wait for the previous step to fully complete before running the next

**If HuggingFace login fails:**
- Make sure you created account: https://huggingface.co/join
- Get token from: https://huggingface.co/settings/tokens
- Run `huggingface-cli login` again

**Remember:** One command at a time, wait for each to finish!
