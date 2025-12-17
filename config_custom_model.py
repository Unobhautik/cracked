"""
Configuration with Custom Model Support
Allows switching between OpenAI and custom trained models
"""
from pathlib import Path
import os

# Try to import custom model integration
try:
    from training.model_integration import get_medical_model, MedicalModelWrapper
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    CUSTOM_MODEL_AVAILABLE = False
    print("⚠️  Custom model integration not available. Using OpenAI only.")

from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "agentmed_history.db"
DEFAULT_STORAGE = SqliteStorage(
    table_name="medical_ai_sessions",
    db_file=str(DB_FILE),
)

# Check if we should use custom model
USE_CUSTOM_MODEL = os.getenv("USE_CUSTOM_MODEL", "false").lower() == "true"
CUSTOM_MODEL_PATH = os.getenv("CUSTOM_MODEL_PATH", "training/models/medical_ai_model")
BASE_MODEL_NAME = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-v0.1")

# Create model instance
if USE_CUSTOM_MODEL and CUSTOM_MODEL_AVAILABLE:
    try:
        # Check if model exists
        model_path = Path(CUSTOM_MODEL_PATH)
        if model_path.exists():
            print(f"✓ Using custom trained model from: {CUSTOM_MODEL_PATH}")
            _custom_model_wrapper = get_medical_model(
                use_custom=True,
                model_path=CUSTOM_MODEL_PATH,
                base_model=BASE_MODEL_NAME
            )
            # Create a wrapper that works with agno
            class CustomModelAdapter:
                """Adapter to make custom model work with agno framework"""
                def __init__(self, model_wrapper: MedicalModelWrapper):
                    self.model_wrapper = model_wrapper
                    self.id = "custom_medical_model"
                
                def complete(self, messages, **kwargs):
                    """Complete method compatible with agno"""
                    # Extract user message
                    user_message = ""
                    for msg in messages:
                        if msg.get("role") == "user":
                            user_message = msg.get("content", "")
                    
                    if not user_message:
                        # Try to get from last message
                        if messages:
                            user_message = str(messages[-1].get("content", ""))
                    
                    response = self.model_wrapper.generate(
                        user_message,
                        max_tokens=kwargs.get("max_tokens", 512),
                        temperature=kwargs.get("temperature", 0.7)
                    )
                    
                    # Return in agno format
                    class Response:
                        def __init__(self, content):
                            self.content = content
                    
                    return Response(response)
            
            DEFAULT_MODEL = CustomModelAdapter(_custom_model_wrapper)
            print("✓ Custom model loaded and ready!")
        else:
            print(f"⚠️  Custom model not found at {CUSTOM_MODEL_PATH}")
            print("   Falling back to OpenAI API")
            DEFAULT_MODEL = OpenAIChat(id="gpt-4o-mini")
    except Exception as e:
        print(f"⚠️  Error loading custom model: {e}")
        print("   Falling back to OpenAI API")
        DEFAULT_MODEL = OpenAIChat(id="gpt-4o-mini")
else:
    DEFAULT_MODEL = OpenAIChat(id="gpt-4o-mini")

DEFAULT_AGENT_CONFIG = {
    "model": DEFAULT_MODEL,
    "markdown": True,
    "add_history_to_messages": True,
    "storage": DEFAULT_STORAGE,
}

AGENT_CONFIGS = {
    "booking": {
        "description": "I help people book doctor appointments",
        "instructions": "Always remember previous conversation context. If the user mentioned symptoms like fever, pain, or other health concerns before, use that information to suggest appropriate doctor types. Ask for any missing appointment details (doctor type, date, time, location) and only use the book_appointment tool when they provide all required information.",
    },
    "cancellation": {
        "description": "I help people cancel their appointments",
        "instructions": "Remember any booking details mentioned previously. Always ask for their booking confirmation number first. Only use the cancel_appointment tool when they provide the confirmation ID.",
    },
    "tips": {
        "description": "I give helpful health and wellness advice",
        "instructions": "Remember any health concerns or symptoms mentioned previously. If the user mentioned specific symptoms like fever, pain, etc., provide relevant tips for those conditions. Ask what kind of health tips they're looking for. Use the get_health_tips tool only when they specify a topic or ask for general tips.",
    },
    "reschedule": {
        "description": "I help people reschedule their appointments",
        "instructions": "Remember any appointment details mentioned previously. Ask about their current appointment details and when they'd like to reschedule it to. Help them find a better time.",
    },
    "reminder": {
        "description": "I help people set up medication reminders",
        "instructions": "Remember any medication details mentioned previously. Ask what medications they're taking, how often, and when. Help them create a reminder system that works for them.",
    },
    "pdf_analyzer": {
        "description": "I analyze medical documents and reports",
        "instructions": "You are a medical document analyzer. When given a PDF file, analyze it thoroughly and answer questions about the medical content. Remember any previous questions about the same document. Provide detailed, accurate analysis of medical reports, test results, prescriptions, and other healthcare documents. If the user mentions symptoms or health concerns in the conversation, relate the PDF analysis to those concerns when relevant.",
    },
}

def get_agent_config(agent_name):
    config = DEFAULT_AGENT_CONFIG.copy()
    if agent_name in AGENT_CONFIGS:
        config.update(AGENT_CONFIGS[agent_name])
    return config


def create_storage(table_name=None):
    if table_name:
        return SqliteStorage(
            table_name=table_name,
            db_file=str(DB_FILE),
        )
    return DEFAULT_STORAGE


