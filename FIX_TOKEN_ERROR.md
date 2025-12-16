# 🔧 Fix "401 Unauthorized" Error

## The Problem
Your token is **invalid** or **copied incorrectly**. This means:
- Token doesn't exist
- Token was deleted
- Token copied with extra spaces
- Wrong token copied

## ✅ Step-by-Step Fix

### Step 1: Go to HuggingFace Website
Open your browser and go to:
```
https://huggingface.co/settings/tokens
```

**Make sure you're LOGGED IN** to HuggingFace website first!

### Step 2: Delete ALL Old Tokens
1. Look at your tokens list
2. **Delete ALL existing tokens** (click delete/trash icon)
3. This ensures you start fresh

### Step 3: Create NEW Token
1. Click **"New token"** button
2. **Name**: Type `training` (or any name you want)
3. **Type**: Select **"Read"** (that's enough for downloading models)
4. Click **"Generate token"**

### Step 4: COPY THE ENTIRE TOKEN
**IMPORTANT:**
- The token starts with `hf_`
- It's about 40-50 characters long
- **COPY THE ENTIRE TOKEN** - don't miss any characters
- **NO SPACES** before or after
- Copy it immediately - you can only see it once!

Example format: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 5: Save Token Somewhere Safe
- Paste it in Notepad temporarily
- Or save in a text file
- You'll need it for login

### Step 6: Login in Terminal
```bash
hf auth login
```

When it asks:
- **"Enter your token"**: Right-click to paste (don't type it!)
- **"Add token as git credential?"**: Type `n` and press Enter

### Step 7: Verify It Worked
```bash
hf auth whoami
```

You should see your HuggingFace username (not an error).

---

## 🆘 If Still Not Working

### Check 1: Are you logged into HuggingFace website?
- Go to: https://huggingface.co
- Make sure you see your profile/username
- If not, log in first!

### Check 2: Token Format
Your token MUST:
- ✅ Start with `hf_`
- ✅ Be 40-50 characters long
- ✅ Have NO spaces
- ✅ Be copied completely

### Check 3: Try Manual Login
```bash
hf auth login --token hf_YOUR_TOKEN_HERE
```
Replace `hf_YOUR_TOKEN_HERE` with your actual token.

### Check 4: Create New Account (If Needed)
If you don't have a HuggingFace account:
1. Go to: https://huggingface.co/join
2. Sign up (free)
3. Verify your email
4. Then create token

---

## ✅ Quick Test

After login, test with:
```bash
hf auth whoami
```

If you see your username → **SUCCESS!**
If you see an error → Token is still wrong, create a new one.

---

## 📝 Common Mistakes

❌ **Wrong**: Copying token with spaces
❌ **Wrong**: Missing first or last characters
❌ **Wrong**: Using old/deleted token
❌ **Wrong**: Not logged into HuggingFace website

✅ **Right**: Copy entire token, no spaces, fresh token

