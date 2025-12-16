# Windows Training Guide - One Command at a Time

## 📋 Follow These Steps ONE BY ONE

Copy and paste each command, wait for it to finish, then move to the next step.

---

### STEP 1: Setup Directories
```bash
python training/setup_training.py
```
**Wait for:** "Setup complete!"

---

### STEP 2: Install Dependencies
```bash
pip install -r training/requirements_training.txt
```
**Wait for:** Installation to complete (may take 5-10 minutes)

---

### STEP 3: Collect Data (Downloads Everything Automatically!)
```bash
python training/data_collector.py
```
**Wait for:** "Data collection complete!" (may take 10-30 minutes)
**This automatically downloads all datasets - you don't need to do anything!**

---

### STEP 4: Process Data
```bash
python training/data_processor.py
```
**Wait for:** "Processing complete!" (takes 2-5 minutes)

---

### STEP 5: Convert to Training Format
```bash
python training/dataset_converter.py
```
**Wait for:** "Dataset conversion complete!" (takes 1-2 minutes)

---

### STEP 6: Train Your Model
```bash
python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora --epochs 3
```
**Wait for:** "Training complete!" (takes 2-6 hours depending on your GPU)

---

## ✅ That's It!

After Step 6 finishes, your model is ready!

---

## 🆘 If Something Goes Wrong

**If Step 2 fails (pip install):**
- Make sure you're in the project folder: `cd E:\agentmed`
- Try: `python -m pip install -r training/requirements_training.txt`

**If Step 3 fails (data collection):**
- Check your internet connection
- Wait a few minutes and try again

**If Step 6 fails (training):**
- Make sure you have a GPU with 16GB+ VRAM
- If not, you can't train on CPU (too slow)

---

## 📝 Quick Reference

Just remember:
1. Setup
2. Install
3. Collect (automatic downloads)
4. Process
5. Convert
6. Train

One command at a time, wait for each to finish!

