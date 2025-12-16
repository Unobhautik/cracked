"""
Symptom Analyzer Agent
Analyzes symptoms and provides medically grounded guidance using RAG system
"""
from agno.agent import Agent
from agno.tools import tool
import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed. Install with: pip install python-dotenv")

from config import get_agent_config
from medical_knowledge import get_knowledge_retriever
from safety_layer import get_safety_layer
from rag_system import get_rag_system


@tool
def analyze_symptoms(symptoms: str, additional_context: str = ""):
    """Analyze symptoms and retrieve relevant medical information"""
    safety = get_safety_layer()
    knowledge = get_knowledge_retriever()
    rag = get_rag_system()
    
    # Safety check first
    risk_assessment = safety.assess_risk_level(symptoms)
    
    if risk_assessment["is_emergency"]:
        return {
            "emergency": True,
            "message": risk_assessment["messages"][0] if risk_assessment["messages"] else "Please seek emergency medical attention immediately.",
            "analysis": None
        }
    
    # Get medical context
    full_query = f"{symptoms} {additional_context}".strip()
    medical_context = knowledge.get_medical_context(full_query, sources=["PubMed", "NHS", "CDC"])
    
    # Also search RAG system
    rag_context = rag.get_context_for_query(full_query, top_k=5)
    
    # Combine contexts
    combined_context = f"Medical Knowledge:\n{medical_context}\n\nAdditional Information:\n{rag_context}"
    
    return {
        "emergency": False,
        "risk_level": risk_assessment["risk_level"],
        "symptoms": symptoms,
        "medical_context": combined_context,
        "requires_professional_care": risk_assessment["is_high_risk"],
        "safety_warning": risk_assessment["messages"][0] if risk_assessment["messages"] else None
    }


@tool
def get_triage_recommendation(symptoms: str) -> str:
    """Get triage recommendation based on symptoms"""
    safety = get_safety_layer()
    risk_assessment = safety.assess_risk_level(symptoms)
    
    if risk_assessment["is_emergency"]:
        return "IMMEDIATE EMERGENCY CARE: Call 911 or go to the nearest emergency room immediately."
    elif risk_assessment["is_high_risk"]:
        return "URGENT CARE RECOMMENDED: Please see a healthcare provider within 24 hours or visit an urgent care center."
    elif risk_assessment["risk_level"] == "medium":
        return "SCHEDULE APPOINTMENT: Consider scheduling an appointment with your healthcare provider within a few days."
    else:
        return "MONITOR: Continue monitoring your symptoms. If they worsen or persist, consult a healthcare provider."


def create_symptom_analyzer_agent(storage=None):
    config = {
        "description": "I analyze symptoms and provide medically grounded guidance using verified medical sources",
        "instructions": """You are a medical symptom analyzer that helps users understand their symptoms safely.

IMPORTANT SAFETY RULES:
1. ALWAYS check for emergency situations first (chest pain, difficulty breathing, etc.)
2. For emergencies, immediately direct users to call 911 or go to emergency room
3. For high-risk symptoms, recommend urgent professional medical care
4. Always use the analyze_symptoms tool to get verified medical information
5. Cite your sources (FDA, PubMed, NHS, CDC) when providing information
6. Never provide definitive diagnoses - only guidance based on symptoms
7. Always recommend consulting healthcare professionals for serious concerns
8. Use the get_triage_recommendation tool to suggest appropriate care level

When analyzing symptoms:
- Ask clarifying questions if needed (duration, severity, triggers)
- Retrieve relevant medical information using tools
- Provide evidence-based guidance
- Explain what the symptoms might indicate (not definitive diagnosis)
- Suggest when to seek professional care
- Provide safety disclaimers

Remember: You are an assistant, not a replacement for professional medical care.""",
    }
    
    base_config = get_agent_config("tips")  # Use similar base config
    base_config.update(config)
    
    if storage:
        base_config["storage"] = storage
    
    return Agent(
        **base_config,
        tools=[analyze_symptoms, get_triage_recommendation],
    )


def handle_symptom_analysis(agent, user_query):
    """Handle symptom analysis request"""
    try:
        response = agent.run(user_query, stream=False)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Sorry, I couldn't analyze those symptoms: {e}"


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        print("OpenAI API key not found!")
        print("Create a .env file in the project root with:")
        print("OPENAI_API_KEY=sk-your-actual-key-here")
        print("\nGet your key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    agent = create_symptom_analyzer_agent()
    print("Symptom Analyzer Agent")
    print("Type 'exit' to quit.")
    print("\n⚠️ Remember: This is not a substitute for professional medical care.\n")
    
    while True:
        try:
            user_query = input("\nDescribe your symptoms: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "back"]:
                print("Goodbye.")
                break
            
            response = handle_symptom_analysis(agent, user_query)
            print(f"\n{response}")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

