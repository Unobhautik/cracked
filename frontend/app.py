"""
AgentMed - Web Server
FastAPI backend for AgentMed frontend
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import json
import asyncio
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from medical_ai import setup_memory, setup_agents, create_medical_team
from safety_layer import get_safety_layer

# Global instances
memory = None
team = None
safety = None

# Create uploads directory - relative to this script's location
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global memory, team, safety
    print("Initializing AgentMed...")
    memory = setup_memory()
    agents = setup_agents()
    team = create_medical_team(agents, memory)
    safety = get_safety_layer()
    print("AgentMed ready!")
    print("\n" + "="*60)
    print("🌐 Server is running!")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("   or http://127.0.0.1:8000")
    print("="*60 + "\n")
    yield
    # Shutdown (if needed)
    print("\nShutting down AgentMed...")

app = FastAPI(title="AgentMed - Healthcare AI Assistant", lifespan=lifespan)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "web_user"

class ChatResponse(BaseModel):
    response: str
    agent_used: Optional[str] = None
    risk_level: Optional[str] = None
    emergency: Optional[bool] = False

@app.get("/")
async def read_root():
    """Serve the frontend"""
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(str(index_path))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Handle chat messages"""
    global team, safety
    
    if not team:
        raise HTTPException(status_code=503, detail="Medical AI system not initialized")
    
    try:
        # Safety check
        risk_assessment = safety.assess_risk_level(message.message)
        
        # Get response from team
        response = team.run(message.message, stream=False, user_id=message.user_id)
        response_text = str(response.content) if hasattr(response, 'content') else str(response)
        
        # Add safety disclaimer
        if risk_assessment["risk_level"] != "low":
            response_text = safety.add_safety_disclaimer(response_text)
        
        return ChatResponse(
            response=response_text,
            agent_used="medical_team",
            risk_level=risk_assessment["risk_level"],
            emergency=risk_assessment["is_emergency"]
        )
    except Exception as e:
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            risk_level="error"
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time chat"""
    await websocket.accept()
    global team, safety
    
    user_id = "ws_user"
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if not user_message:
                continue
            
            # Safety check
            risk_assessment = safety.assess_risk_level(user_message)
            
            # Send emergency warning if needed
            if risk_assessment["is_emergency"]:
                await websocket.send_json({
                    "type": "emergency",
                    "message": risk_assessment["messages"][0] if risk_assessment["messages"] else "Please seek emergency medical attention immediately."
                })
            
            # Get response
            try:
                response = team.run(user_message, stream=False, user_id=user_id)
                response_text = str(response.content) if hasattr(response, 'content') else str(response)
                
                # Add safety disclaimer
                if risk_assessment["risk_level"] != "low":
                    response_text = safety.add_safety_disclaimer(response_text)
                
                await websocket.send_json({
                    "type": "response",
                    "message": response_text,
                    "risk_level": risk_assessment["risk_level"],
                    "emergency": risk_assessment["is_emergency"]
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AgentMed"}

@app.get("/api/agents")
async def get_agents():
    """Get list of available agents"""
    return {
        "agents": [
            {"name": "Booking Agent", "id": "booking", "description": "Book medical appointments with the right specialists"},
            {"name": "Cancellation Agent", "id": "cancellation", "description": "Cancel appointments and handle refunds"},
            {"name": "Reschedule Agent", "id": "reschedule", "description": "Reschedule appointments to better times"},
            {"name": "PDF Analyzer", "id": "pdf_analyzer", "description": "Analyze medical PDFs and reports"},
            {"name": "Reminder Agent", "id": "reminder", "description": "Set up medication reminders and schedules"},
            {"name": "Health Tips", "id": "tips", "description": "Get personalized health and wellness tips"},
            {"name": "Symptom Analyzer", "id": "symptom_analyzer", "description": "Analyze symptoms with verified medical sources"},
            {"name": "Drug Information", "id": "drug_info", "description": "Get drug information and check interactions"}
        ]
    }

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF file uploads"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOADS_DIR / filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse({
            "success": True,
            "filename": filename,
            "file_path": str(file_path),
            "message": f"PDF uploaded successfully: {file.filename}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/api/chat-with-pdf")
async def chat_with_pdf(message: ChatMessage, pdf_path: Optional[str] = None):
    """Handle chat messages with PDF context"""
    global team, safety
    
    if not team:
        raise HTTPException(status_code=503, detail="Medical AI system not initialized")
    
    try:
        # Enhance message with PDF path if provided
        enhanced_message = message.message
        if pdf_path and os.path.exists(pdf_path):
            enhanced_message = f"Please analyze the PDF at {pdf_path}. {message.message}"
        
        # Safety check
        risk_assessment = safety.assess_risk_level(enhanced_message)
        
        # Get response from team
        response = team.run(enhanced_message, stream=False, user_id=message.user_id)
        response_text = str(response.content) if hasattr(response, 'content') else str(response)
        
        # Add safety disclaimer
        if risk_assessment["risk_level"] != "low":
            response_text = safety.add_safety_disclaimer(response_text)
        
        return ChatResponse(
            response=response_text,
            agent_used="medical_team",
            risk_level=risk_assessment["risk_level"],
            emergency=risk_assessment["is_emergency"]
        )
    except Exception as e:
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            risk_level="error"
        )

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 for localhost access, or 0.0.0.0 for all interfaces
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

