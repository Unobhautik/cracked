# 🔌 API Integration Guide - For Backend Developer

## How to Integrate with MedicalAI System

This guide shows you exactly how to connect your FastAPI backend to the existing `medical_ai.py` system.

---

## 📋 Understanding the Current System

### Main Entry Point: `medical_ai.py`

The system has these key functions:
- `setup_agents()` - Creates all 8 agents
- `create_medical_team()` - Creates the routing team
- `setup_memory()` - Sets up conversation memory
- `check_api_key()` - Validates OpenAI API key

### Current Flow:
1. User sends message → `team.run(message, user_id=user_id)`
2. Team automatically routes to appropriate agent
3. Agent processes and returns response

---

## 🔧 Integration Options

### Option 1: Direct Team Integration (Recommended)

Use the team directly - it handles routing automatically:

```python
from fastapi import FastAPI
from medical_ai import setup_agents, create_medical_team, setup_memory, check_api_key

app = FastAPI()

# Initialize once at startup
agents = None
team = None
memory = None

@app.on_event("startup")
async def startup():
    global agents, team, memory
    
    if not check_api_key():
        raise RuntimeError("OpenAI API key missing!")
    
    memory = setup_memory()
    agents = setup_agents()
    team = create_medical_team(agents, memory)

@app.post("/api/chat")
async def chat(message: str, user_id: str = "default_user"):
    # Team automatically routes to correct agent
    response = team.run(message, user_id=user_id)
    
    # Extract response text
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    return {"response": response_text}
```

### Option 2: Individual Agent Access

Access specific agents directly:

```python
from medical_ai import setup_agents

agents = setup_agents()

# Use specific agent
@app.post("/api/symptoms")
async def analyze_symptoms(symptoms: str, user_id: str = "default_user"):
    response = agents['symptom_analyzer'].run(
        f"Analyze these symptoms: {symptoms}",
        user_id=user_id
    )
    return {"response": response.content}
```

---

## 📝 Complete Integration Example

Here's a complete working example:

```python
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Import medical AI system
from medical_ai import (
    setup_agents,
    create_medical_team,
    setup_memory,
    check_api_key
)

load_dotenv()

app = FastAPI(title="MedicalAI API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
agents = None
team = None
memory = None

@app.on_event("startup")
async def startup():
    """Initialize AI system"""
    global agents, team, memory
    
    if not check_api_key():
        raise RuntimeError("Set OPENAI_API_KEY in .env file")
    
    print("Initializing MedicalAI system...")
    memory = setup_memory()
    agents = setup_agents()
    team = create_medical_team(agents, memory)
    print("✅ MedicalAI ready!")

# Request models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str
    agent_used: Optional[str] = None
    is_emergency: bool = False

# Endpoints
@app.get("/")
async def root():
    return {"status": "ok", "message": "MedicalAI API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - auto-routes to appropriate agent"""
    if not team:
        raise HTTPException(500, "AI system not initialized")
    
    try:
        # Team automatically routes to correct agent
        response = team.run(request.message, user_id=request.user_id)
        
        # Extract response
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Check for emergency (basic check - you can enhance this)
        is_emergency = any(word in request.message.lower() for word in [
            "chest pain", "can't breathe", "emergency"
        ])
        
        return ChatResponse(
            response=response_text,
            agent_used="auto-routed",
            is_emergency=is_emergency
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@app.post("/api/symptoms")
async def analyze_symptoms(request: ChatRequest):
    """Direct symptom analysis"""
    if not agents or 'symptom_analyzer' not in agents:
        raise HTTPException(500, "Symptom analyzer not available")
    
    response = agents['symptom_analyzer'].run(
        request.message,
        user_id=request.user_id
    )
    
    return {"response": response.content}

@app.post("/api/drug-info")
async def drug_info(request: ChatRequest):
    """Drug information"""
    if not agents or 'drug_info' not in agents:
        raise HTTPException(500, "Drug info agent not available")
    
    response = agents['drug_info'].run(
        request.message,
        user_id=request.user_id
    )
    
    return {"response": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 🔍 Understanding Response Objects

### Team Response
```python
response = team.run("I have chest pain", user_id="user123")

# Response object has:
response.content  # The text response
response.messages  # Full message history
response.member_responses  # Individual agent responses (if enabled)
```

### Agent Response
```python
response = agents['symptom_analyzer'].run("I have a fever", user_id="user123")

# Same structure
response.content  # The text response
```

---

## 🎯 Available Agents

All agents are available in the `agents` dictionary:

```python
agents = {
    'booking': BookingAgent,
    'cancellation': CancellationAgent,
    'reschedule': RescheduleAgent,
    'pdf_analyzer': PDFAnalyzerAgent,
    'reminder': ReminderAgent,
    'tips': TipsAgent,
    'symptom_analyzer': SymptomAnalyzerAgent,
    'drug_info': DrugInfoAgent,
}
```

---

## 📤 Streaming Responses (Advanced)

If you want to stream responses (like ChatGPT):

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""
    
    def generate():
        # Use team's streaming capability
        for chunk in team.run_stream(request.message, user_id=request.user_id):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

Note: You may need to check if `run_stream` is available in your Agno version.

---

## 🔐 Safety Layer Integration

The system has a safety layer. You can use it:

```python
from safety_layer import get_safety_layer

safety = get_safety_layer()

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Check safety first
    risk = safety.assess_risk_level(request.message)
    
    if risk["is_emergency"]:
        return {
            "response": "🚨 EMERGENCY: Please call 911 immediately!",
            "is_emergency": True,
            "risk_level": "emergency"
        }
    
    # Continue with normal processing
    response = team.run(request.message, user_id=request.user_id)
    return {"response": response.content}
```

---

## 📄 PDF Upload Handling

For PDF analysis:

```python
from fastapi import UploadFile, File
import tempfile
import os

@app.post("/api/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), user_id: str = "default_user"):
    """Analyze uploaded PDF"""
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Analyze PDF
        response = agents['pdf_analyzer'].run(
            f"Analyze this PDF: {tmp_path}",
            user_id=user_id
        )
        
        return {"response": response.content}
    finally:
        # Clean up
        os.unlink(tmp_path)
```

---

## 🗄️ Database Integration

The system uses SQLite by default. For production, you might want PostgreSQL:

```python
# The system already uses SQLiteStorage
# For PostgreSQL, you'd need to modify config.py
# But for hackathon, SQLite is fine!
```

---

## ⚡ Performance Tips

1. **Initialize once:** Don't recreate agents on each request
2. **Use async:** FastAPI is async, but Agno might be sync - wrap if needed
3. **Caching:** Consider caching common queries
4. **Rate limiting:** Add rate limiting for production

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(request: Request, chat_req: ChatRequest):
    # Your code
```

---

## 🐛 Error Handling

Always wrap agent calls in try-except:

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = team.run(request.message, user_id=request.user_id)
        return {"response": response.content}
    except Exception as e:
        # Log error
        print(f"Error: {e}")
        # Return user-friendly message
        raise HTTPException(
            status_code=500,
            detail="Sorry, I encountered an error. Please try again."
        )
```

---

## ✅ Testing Your Integration

Test locally:

```bash
# Terminal 1: Start backend
uvicorn main:app --reload

# Terminal 2: Test API
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a headache", "user_id": "test123"}'
```

Or use the Swagger UI:
```
http://localhost:8000/docs
```

---

## 📚 Key Files to Understand

1. **`medical_ai.py`** - Main system entry point
2. **`config.py`** - Agent configurations
3. **`agents/*.py`** - Individual agent implementations
4. **`safety_layer.py`** - Safety checks
5. **`rag_system.py`** - RAG system (if you need to query it)

---

## 🎯 Quick Checklist

- [ ] Import `medical_ai` functions
- [ ] Initialize agents/team at startup
- [ ] Create API endpoints
- [ ] Handle errors
- [ ] Add CORS
- [ ] Test locally
- [ ] Deploy

---

## 🆘 Common Issues

**Issue: "Module not found"**
- Make sure you're in the project root
- Check Python path includes project directory

**Issue: "OpenAI API key not found"**
- Create `.env` file with `OPENAI_API_KEY=sk-...`
- Or set environment variable

**Issue: "Agents not initialized"**
- Make sure startup event runs
- Check for errors in startup

**Issue: "Response is None"**
- Check response object structure
- Use `response.content` or `str(response)`

---

**You've got this! The AI system is ready - just wrap it in an API! 🚀**


