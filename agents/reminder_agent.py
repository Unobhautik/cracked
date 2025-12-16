from agno.agent import Agent
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

def create_reminder_agent(storage=None):
    config = get_agent_config("reminder")
    if storage:
        config["storage"] = storage
    
    return Agent(**config)

def handle_reminder_request(agent, user_query):
    try:
        response = agent.run(user_query, stream=False)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Sorry, I couldn't set up those reminders: {e}"


# For testing individual agents
if __name__ == "__main__":
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        print("OpenAI API key not found!")
        print("Create a .env file in the project root with:")
        print("OPENAI_API_KEY=sk-your-actual-key-here")
        print("\nGet your key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    print("API key found, testing reminder agent...")
    agent = create_reminder_agent()
    print("Reminder Agent Test:")
    print(handle_reminder_request(agent, "I need medication reminders"))