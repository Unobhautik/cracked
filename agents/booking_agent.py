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


@tool
def book_appointment(doctor_type, date, time, location, patient_name=None):
    """Book a medical appointment with the given details"""
    if not all([doctor_type, date, time, location]):
        return "I need doctor type, date, time, and location to book your appointment."
    
    booking_id = f"BK{abs(hash(f'{doctor_type}{date}{time}')) % 100000:06d}"
    return f"Appointment booked! Your confirmation is {booking_id}. See {doctor_type} on {date} at {time} in {location}."


def create_booking_agent(storage=None):
    config = get_agent_config("booking")
    if storage:
        config["storage"] = storage
    
    return Agent(
        **config,
        tools=[book_appointment],
    )


def handle_booking_request(agent, user_query):
    try:
        response = agent.run(user_query, stream=False)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Sorry, I couldn't book that appointment: {e}"


def save_chat_markdown(chat_log):
    try:
        default_name = "booking_chat.md"
        filename = input("Save as (default booking_chat.md): ").strip() or default_name
        if not filename.endswith(".md"):
            filename += ".md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Booking Agent Chat\n\n")
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
    
    agent = create_booking_agent()
    print("Booking Agent")
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
            response = handle_booking_request(agent, user_query)
            print(response)
            chat_log.append(f"You: {user_query}")
            chat_log.append(f"Agent: {response}")
            
            if response.startswith("Appointment booked!"):
                ans = input("Would you like to save this chat? (y/n): ").strip().lower()
                if ans.startswith("y"):
                    save_chat_markdown(chat_log)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
