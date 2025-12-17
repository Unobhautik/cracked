"""
FastAPI Backend Example - Integration with MedicalAI System
This is a starter template for the Backend Developer to build upon.

To use this:
1. Install: pip install fastapi uvicorn python-dotenv
2. Run: uvicorn backend_api_example:app --reload
3. Visit: http://localhost:8000/docs for API documentation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your medical AI system
from medical_ai import (
    setup_agents,
    create_medical_team,
    setup_memory,
    check_api_key
)

app = FastAPI(
    title="MedicalAI API",
    description="Agentic AI Healthcare System API",
    version="1.0.0"
)

# CORS middleware (allow frontend to call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI system (do this once at startup)
agents = None
team = None
memory = None

@app.on_event("startup")
async def startup_event():
    """Initialize AI system when server starts"""
    global agents, team, memory
    
    if not check_api_key():
        raise RuntimeError("OpenAI API key not found! Set OPENAI_API_KEY in .env")
    
    memory = setup_memory()
    agents = setup_agents()
    team = create_medical_team(agents, memory)
    print("✅ MedicalAI system initialized!")

# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    agent_used: Optional[str] = None
    risk_level: Optional[str] = None
    is_emergency: bool = False

class SymptomAnalysisRequest(BaseModel):
    symptoms: str
    user_id: str = "default_user"

class DrugInfoRequest(BaseModel):
    drug_name: str
    user_id: str = "default_user"

class AppointmentRequest(BaseModel):
    doctor_type: str
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    user_id: str = "default_user"

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "MedicalAI API is running",
        "agents_available": [
            "booking", "cancellation", "reschedule", 
            "pdf_analyzer", "reminder", "tips",
            "symptom_analyzer", "drug_info"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "ai_system_initialized": team is not None,
        "agents_count": len(agents) if agents else 0
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - routes to appropriate agent automatically
    
    Example request:
    {
        "message": "I have chest pain",
        "user_id": "user123"
    }
    """
    if not team:
        raise HTTPException(status_code=500, detail="AI system not initialized")
    
    try:
        # Get response from team (this routes to appropriate agent)
        response = team.run(
            request.message,
            user_id=request.user_id
        )
        
        # Extract response text
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Check for emergency (you might want to add safety layer check here)
        is_emergency = any(keyword in request.message.lower() for keyword in [
            "chest pain", "can't breathe", "difficulty breathing",
            "severe pain", "unconscious", "bleeding heavily"
        ])
        
        return ChatResponse(
            response=response_text,
            agent_used="auto-routed",
            risk_level="emergency" if is_emergency else "normal",
            is_emergency=is_emergency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/api/symptoms", response_model=ChatResponse)
async def analyze_symptoms(request: SymptomAnalysisRequest):
    """
    Analyze symptoms using symptom_analyzer agent
    
    Example request:
    {
        "symptoms": "I have a fever and headache",
        "user_id": "user123"
    }
    """
    if not team or 'symptom_analyzer' not in agents:
        raise HTTPException(status_code=500, detail="Symptom analyzer not available")
    
    try:
        # Route to symptom analyzer
        response = agents['symptom_analyzer'].run(
            f"Analyze these symptoms: {request.symptoms}",
            user_id=request.user_id
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return ChatResponse(
            response=response_text,
            agent_used="symptom_analyzer",
            risk_level="normal",
            is_emergency=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing symptoms: {str(e)}")

@app.post("/api/drug-info", response_model=ChatResponse)
async def get_drug_info(request: DrugInfoRequest):
    """
    Get drug information using drug_info agent
    
    Example request:
    {
        "drug_name": "aspirin",
        "user_id": "user123"
    }
    """
    if not team or 'drug_info' not in agents:
        raise HTTPException(status_code=500, detail="Drug info agent not available")
    
    try:
        response = agents['drug_info'].run(
            f"Tell me about {request.drug_name}",
            user_id=request.user_id
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return ChatResponse(
            response=response_text,
            agent_used="drug_info",
            risk_level="normal",
            is_emergency=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting drug info: {str(e)}")

@app.post("/api/book-appointment", response_model=ChatResponse)
async def book_appointment(request: AppointmentRequest):
    """
    Book an appointment using booking agent
    
    Example request:
    {
        "doctor_type": "cardiologist",
        "preferred_date": "2024-01-15",
        "preferred_time": "morning",
        "user_id": "user123"
    }
    """
    if not team or 'booking' not in agents:
        raise HTTPException(status_code=500, detail="Booking agent not available")
    
    try:
        message = f"I need to book an appointment with a {request.doctor_type}"
        if request.preferred_date:
            message += f" on {request.preferred_date}"
        if request.preferred_time:
            message += f" in the {request.preferred_time}"
        
        response = agents['booking'].run(
            message,
            user_id=request.user_id
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return ChatResponse(
            response=response_text,
            agent_used="booking",
            risk_level="normal",
            is_emergency=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error booking appointment: {str(e)}")

@app.post("/api/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), user_id: str = "default_user"):
    """
    Analyze a medical PDF using pdf_analyzer agent
    
    Example: POST /api/analyze-pdf with file upload
    """
    if not team or 'pdf_analyzer' not in agents:
        raise HTTPException(status_code=500, detail="PDF analyzer not available")
    
    try:
        # Save uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze PDF
        response = agents['pdf_analyzer'].run(
            f"Analyze this PDF: {file_path}",
            user_id=user_id
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up temp file
        os.remove(file_path)
        
        return ChatResponse(
            response=response_text,
            agent_used="pdf_analyzer",
            risk_level="normal",
            is_emergency=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing PDF: {str(e)}")

@app.post("/api/health-tips", response_model=ChatResponse)
async def get_health_tips(request: ChatRequest):
    """
    Get health tips using tips agent
    
    Example request:
    {
        "message": "I want tips for managing diabetes",
        "user_id": "user123"
    }
    """
    if not team or 'tips' not in agents:
        raise HTTPException(status_code=500, detail="Tips agent not available")
    
    try:
        response = agents['tips'].run(
            request.message,
            user_id=request.user_id
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return ChatResponse(
            response=response_text,
            agent_used="tips",
            risk_level="normal",
            is_emergency=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting health tips: {str(e)}")

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


