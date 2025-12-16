# 📤 Push Code to GitHub

## Quick Method

### Option 1: Double-Click (Easiest!)
Double-click: `PUSH_TO_GITHUB.bat`

### Option 2: Manual Commands

**Step 1: Initialize Git (if not done)**
```bash
git init
```

**Step 2: Add All Files**
```bash
git add .
```

**Step 3: Commit**
```bash
git commit -m "Initial commit: Medical AI Agentic System"
```

**Step 4: Add Remote**
```bash
git remote add origin https://github.com/Unobhautik/cracked.git
```

**Step 5: Push**
```bash
git push -u origin main
```

If it says "master" instead of "main":
```bash
git branch -M main
git push -u origin main
```

---

## 🔐 GitHub Authentication

If you get authentication errors:

### Option 1: Use Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name it: `git-push`
4. Select scope: `repo` (full control)
5. Copy the token
6. When git asks for password, paste the token instead

### Option 2: Use GitHub CLI
```bash
# Install GitHub CLI
winget install GitHub.cli

# Login
gh auth login

# Then push
git push -u origin main
```

---

## 📋 What Gets Pushed

✅ **Pushed:**
- All Python files
- Configuration files
- Documentation
- Training scripts
- Batch files

❌ **NOT Pushed (in .gitignore):**
- `.env` (your API keys - keep private!)
- `*.db` (database files)
- `training/data/raw/` (large datasets)
- `training/models/` (large model files)
- `__pycache__/` (Python cache)
- `.venv/` (virtual environment)

---

## 🆘 Troubleshooting

**"Repository not found" error?**
- Make sure repository exists: https://github.com/Unobhautik/cracked
- Check you have write access

**"Authentication failed" error?**
- Use Personal Access Token (see above)
- Or use GitHub CLI

**"Branch 'main' does not exist" error?**
```bash
git branch -M main
git push -u origin main
```

**"Nothing to commit" error?**
- Files might already be committed
- Try: `git status` to see what's changed

---

## ✅ After Push

Your code will be at:
https://github.com/Unobhautik/cracked

You can:
- View files online
- Share with others
- Clone on other machines
- Continue development

---

## 🔒 Security Note

**IMPORTANT:** The `.gitignore` file ensures:
- Your `.env` file (with API keys) is NOT pushed
- Large model files are NOT pushed
- Sensitive data stays local

**Never commit:**
- API keys
- Passwords
- Personal tokens
- Large files (>100MB)

