# 📋 How to Paste Token in PowerShell

## The Problem
When you type `hf auth login`, it asks for your token but **hides what you type** (like a password). You can't use Ctrl+V to paste.

## ✅ Solution: Right-Click to Paste

### Step 1: Copy Your Token
1. Go to: https://huggingface.co/settings/tokens
2. Copy your token (Ctrl+C)

### Step 2: In PowerShell
1. Type: `hf auth login`
2. Press Enter
3. When it says "Enter your token (input will not be visible):"
4. **RIGHT-CLICK** in the PowerShell window (don't use Ctrl+V!)
5. Your token will be pasted (you won't see it, but it's there)
6. Press Enter

### Step 3: When Asked About Git
- It will ask: "Add token as git credential? (Y/n)"
- Type: `n`
- Press Enter

---

## 🎯 Visual Guide

```
PowerShell Window:
┌─────────────────────────────────────┐
│ PS E:\agentmed> hf auth login      │
│                                     │
│ Enter your token:                   │
│ [RIGHT-CLICK HERE] ← Paste here!    │
│                                     │
└─────────────────────────────────────┘
```

**Don't type! Just right-click!**

---

## 🔍 How to Know It Worked

After pasting and pressing Enter:

### ✅ SUCCESS looks like:
```
Token has been saved to C:\Users\YourName\.huggingface\token
```

### ❌ ERROR looks like:
```
401 Client Error: Unauthorized
Invalid username or password
```

If you get an error, your token is wrong - create a new one!

---

## 🆘 Alternative: Use Token Directly

If right-click doesn't work, you can pass the token directly:

```bash
hf auth login --token hf_YOUR_TOKEN_HERE
```

Replace `hf_YOUR_TOKEN_HERE` with your actual token.

Example:
```bash
hf auth login --token hf_abc123xyz789...
```

---

## 💡 Pro Tip

1. Copy token to Notepad first
2. Verify it starts with `hf_` and has no spaces
3. Then right-click paste in PowerShell

---

## ✅ Quick Test After Login

```bash
hf auth whoami
```

If you see your username → **SUCCESS!**
If you see an error → Token is wrong, try again.

