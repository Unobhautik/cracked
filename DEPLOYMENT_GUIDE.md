# 🚀 Deployment Guide - MedicalAI Hackathon Project

## Quick Deployment Options

### Option 1: Railway (Easiest - Recommended) ⭐
- **Backend:** Railway.app
- **Frontend:** Vercel
- **Database:** Railway PostgreSQL (included)

### Option 2: Render
- **Backend:** Render.com
- **Frontend:** Render.com or Vercel
- **Database:** Render PostgreSQL

### Option 3: Fly.io
- **Backend:** Fly.io
- **Frontend:** Vercel
- **Database:** Fly.io PostgreSQL

---

## 📦 Backend Deployment (Railway)

### Step 1: Prepare Backend

Create `Procfile` in backend root:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Create `runtime.txt` (if using specific Python version):
```
python-3.11.0
```

### Step 2: Deploy to Railway

1. **Sign up:** https://railway.app
2. **New Project** → **Deploy from GitHub**
3. **Select your repository**
4. **Add Service** → **Empty Service**
5. **Settings** → **Add Source** → Select your backend folder
6. **Variables** → Add these:
   ```
   OPENAI_API_KEY=sk-your-key-here
   PORT=8000
   DATABASE_URL=postgresql://... (auto-generated if you add PostgreSQL)
   ```
7. **Deploy!**

Railway will auto-detect Python and install dependencies from `requirements.txt`.

### Step 3: Get Your Backend URL

Railway gives you: `https://your-app.railway.app`

**Update frontend `.env`:**
```
REACT_APP_API_URL=https://your-app.railway.app
```

---

## 🎨 Frontend Deployment (Vercel)

### Step 1: Prepare Frontend

Create `vercel.json` in frontend root:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

Update `package.json` scripts:
```json
{
  "scripts": {
    "build": "react-scripts build",
    "start": "react-scripts start"
  }
}
```

### Step 2: Deploy to Vercel

1. **Sign up:** https://vercel.com
2. **New Project** → **Import Git Repository**
3. **Select your repository**
4. **Root Directory:** `frontend`
5. **Environment Variables:**
   ```
   REACT_APP_API_URL=https://your-backend.railway.app
   ```
6. **Deploy!**

Vercel gives you: `https://your-app.vercel.app`

---

## 🐳 Docker Deployment (Alternative)

### Backend Dockerfile

Create `backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

Create `frontend/Dockerfile`:
```dockerfile
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Build
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/medicalai
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    environment:
      - REACT_APP_API_URL=http://localhost:8000

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=medicalai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Run locally:
```bash
docker-compose up --build
```

---

## 🗄️ Database Setup

### Option 1: Railway PostgreSQL (Easiest)

1. In Railway project, click **+ New** → **Database** → **PostgreSQL**
2. Railway auto-generates `DATABASE_URL`
3. Add to backend environment variables

### Option 2: Render PostgreSQL

1. Go to Render dashboard
2. **New** → **PostgreSQL**
3. Copy connection string
4. Add to backend environment variables

### Option 3: Supabase (Free Tier)

1. Sign up: https://supabase.com
2. Create project
3. Get connection string from Settings → Database
4. Add to backend environment variables

---

## 🔧 Environment Variables Checklist

### Backend (.env or Railway Variables)
```bash
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://...
PORT=8000
ENVIRONMENT=production
```

### Frontend (.env or Vercel Variables)
```bash
REACT_APP_API_URL=https://your-backend.railway.app
```

---

## ✅ Deployment Checklist

### Backend
- [ ] Code pushed to GitHub
- [ ] `requirements.txt` includes all dependencies
- [ ] `Procfile` created (for Railway)
- [ ] Environment variables set
- [ ] Database connected
- [ ] CORS configured for frontend URL
- [ ] API health check working
- [ ] Deployed and accessible

### Frontend
- [ ] Code pushed to GitHub
- [ ] `package.json` has build script
- [ ] Environment variables set
- [ ] API URL points to deployed backend
- [ ] Build succeeds
- [ ] Deployed and accessible
- [ ] Can connect to backend API

### Testing
- [ ] Backend API responds to health check
- [ ] Frontend loads correctly
- [ ] Chat interface works
- [ ] API calls succeed (check browser console)
- [ ] No CORS errors
- [ ] PDF upload works (if implemented)

---

## 🐛 Common Issues & Fixes

### Issue: CORS Error
**Fix:** Update backend CORS settings:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Environment Variables Not Working
**Fix:** 
- Check variable names match exactly
- Restart deployment after adding variables
- For React, variables must start with `REACT_APP_`

### Issue: Build Fails
**Fix:**
- Check `requirements.txt` / `package.json` for all dependencies
- Check build logs for specific errors
- Test build locally first: `npm run build` or `pip install -r requirements.txt`

### Issue: Database Connection Fails
**Fix:**
- Check `DATABASE_URL` format
- Ensure database is accessible (not blocked by firewall)
- Check database credentials

### Issue: API Returns 500 Errors
**Fix:**
- Check backend logs
- Ensure OpenAI API key is valid
- Check all dependencies installed
- Verify environment variables set correctly

---

## 🚀 Quick Deploy Commands

### Railway CLI (Optional)
```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

### Vercel CLI (Optional)
```bash
npm i -g vercel
vercel login
vercel
```

---

## 📊 Monitoring & Logs

### Railway
- View logs in Railway dashboard
- Real-time log streaming
- Error tracking

### Vercel
- View logs in Vercel dashboard
- Analytics included
- Performance monitoring

---

## 🔒 Security Checklist

- [ ] API keys in environment variables (not in code)
- [ ] CORS restricted to frontend URL
- [ ] HTTPS enabled (automatic on Railway/Vercel)
- [ ] Database credentials secure
- [ ] No sensitive data in logs
- [ ] Rate limiting (if time permits)

---

## 🎯 Production Checklist

Before demo:
- [ ] Both frontend and backend deployed
- [ ] URLs working and accessible
- [ ] All features tested
- [ ] No console errors
- [ ] Mobile responsive
- [ ] Fast loading times
- [ ] Error handling works
- [ ] Emergency detection works

---

## 📝 Deployment URLs Template

Save these for your team:

```
Backend API: https://your-backend.railway.app
Frontend: https://your-app.vercel.app
API Docs: https://your-backend.railway.app/docs
Health Check: https://your-backend.railway.app/api/health
```

---

## 🆘 Need Help?

- **Railway Docs:** https://docs.railway.app
- **Vercel Docs:** https://vercel.com/docs
- **Docker Docs:** https://docs.docker.com
- **FastAPI Docs:** https://fastapi.tiangolo.com/deployment/

---

**Good luck with deployment! 🚀**


