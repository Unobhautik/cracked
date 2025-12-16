# 🔑 HuggingFace & Mistral Setup (One-Time)

## What You Need to Do BEFORE Training

### Step 1: Create HuggingFace Account (Free)
1. Go to: https://huggingface.co/join
2. Sign up (it's free!)
3. Verify your email

### Step 2: Install HuggingFace CLI
```bash
pip install huggingface_hub
```

### Step 3: Login to HuggingFace
```bash
huggingface-cli login
```
- This will ask for your token
- Get token from: https://huggingface.co/settings/tokens
- Click "New token" → Name it "training" → Copy the token
- Paste it when prompted

### Step 4: Request Access to Models (If Needed)

**For Mistral 7B:**
1. Go to: https://huggingface.co/mistralai/Mistral-7B-v0.1
2. Click "Agree and access repository"
3. That's it!

**For Llama 3 (if you want to use it instead):**
1. Go to: https://huggingface.co/meta-llama/Llama-3-8b
2. Request access (may take a few hours/days)
3. Wait for approval email

**For Qwen (if you want to use it instead):**
1. Go to: https://huggingface.co/Qwen/Qwen-7B
2. Click "Agree and access repository"
3. That's it!

---

## ✅ Quick Setup Commands (Copy-Paste)

Run these BEFORE starting training:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login (will ask for token)
huggingface-cli login
```

Then request access to Mistral model (link above).

---

## 🎯 That's It!

After this one-time setup, the training scripts will automatically:
- ✅ Download the model when training starts
- ✅ Download all datasets automatically
- ✅ Everything else is handled!

---

## 📝 What Gets Downloaded Automatically

**When you run training:**
- Mistral 7B model (~14GB) - downloaded automatically
- All medical datasets - downloaded automatically
- Everything else - downloaded automatically

**You DON'T need to:**
- ❌ Manually download model files
- ❌ Download datasets
- ❌ Extract anything

---

## 🆘 Troubleshooting

**"Access denied" error?**
- Make sure you requested access to the model
- Make sure you're logged in: `huggingface-cli whoami`

**"Token not found" error?**
- Run: `huggingface-cli login` again
- Get new token from: https://huggingface.co/settings/tokens

**"Model not found" error?**
- Check you requested access: https://huggingface.co/mistralai/Mistral-7B-v0.1
- Click "Agree and access repository" button

