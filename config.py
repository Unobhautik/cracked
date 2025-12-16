from pathlib import Path
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "agentmed_history.db"
DEFAULT_STORAGE = SqliteStorage(
    table_name="medical_ai_sessions",
    db_file=str(DB_FILE),
)

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
