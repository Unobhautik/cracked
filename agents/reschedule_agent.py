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
from agno.tools import tool

@tool
def reschedule_appointment(confirmation_id, new_date=None, new_time=None):
    """Reschedule an existing appointment using confirmation id and new date/time"""
    if not confirmation_id:
        return "I need the confirmation id to reschedule the appointment."
    if not new_date and not new_time:
        return "Please provide a new date and/or new time to reschedule."
    details = []
    if new_date:
        details.append(f"date={new_date}")
    if new_time:
        details.append(f"time={new_time}")
    return f"Appointment {confirmation_id} rescheduled successfully (" + ", ".join(details) + ")"

def create_reschedule_agent(storage=None):
    config = get_agent_config("reschedule")
    if storage:
        config["storage"] = storage
    
    return Agent(**config, tools=[reschedule_appointment])

def handle_reschedule_request(agent, user_query):
    try:
        response = agent.run(user_query, stream=False)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Sorry, I couldn't help reschedule that: {e}"


def save_chat_markdown(chat_log, title="# Reschedule Agent Chat"):
    try:
        default_name = "reschedule_chat.md"
        filename = input("Save as (default reschedule_chat.md): ").strip() or default_name
        if not filename.endswith(".md"):
            filename += ".md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n")
            for line in chat_log:
                f.write(line + "\n\n")
        print(f"Saved chat to {filename}")
    except Exception as e:
        print(f"Could not save chat: {e}")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        print("OpenAI API key not found!")
        print("Create a .env file in the project root with:")
        print("OPENAI_API_KEY=sk-your-actual-key-here")
        print("\nGet your key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    agent = create_reschedule_agent()
    print("Reschedule Agent")
    print("Type 'back' to exit.")

    chat_log = []

    while True:
        try:
            user_query = input("You: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["back", "exit", "quit"]:
                print("Goodbye.")
                break
            response = handle_reschedule_request(agent, user_query)
            print(response)
            chat_log.append(f"You: {user_query}")
            chat_log.append(f"Agent: {response}")

            if "reschedule" in response.lower() or "moved" in response.lower() or "new time" in response.lower():
                ans = input("Would you like to save this chat? (y/n): ").strip().lower()
                if ans.startswith("y"):
                    save_chat_markdown(chat_log)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

