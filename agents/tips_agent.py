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
from rag_system import get_rag_system


@tool
def get_health_tips(topic=None):
    """Get health and wellness tips, optionally for a specific topic, using medical knowledge sources"""
    knowledge = get_knowledge_retriever()
    rag = get_rag_system()
    
    if not topic:
        # General tips with medical context
        general_context = rag.get_context_for_query("general health wellness tips", top_k=3)
        return f"""Here are some general health tips based on medical guidelines:
• Drink plenty of water and get 7-9 hours of sleep
• Eat more veggies, lean protein, and fiber
• Exercise for 150 minutes per week
• Take short breaks to manage stress
• See a doctor if you have ongoing health issues

Medical Guidance:
{general_context if general_context and 'No relevant' not in general_context else 'Based on general health recommendations from medical sources.'}"""

    # Get topic-specific tips with medical knowledge
    medical_context = knowledge.get_medical_context(f"health tips for {topic}", sources=["NHS", "CDC"])
    rag_context = rag.get_context_for_query(f"health tips {topic}", top_k=5)
    
    combined_context = f"{medical_context}\n\n{rag_context}" if rag_context and 'No relevant' not in rag_context else medical_context
    
    return f"""Here are some evidence-based tips for {topic}:
• Start with small, achievable goals
• Keep a consistent routine
• Cut back on processed foods and sugary drinks
• Fit in short activity bursts when busy
• Talk to a healthcare provider if symptoms persist

Medical Information:
{combined_context if combined_context and 'No medical information found' not in combined_context else 'Based on general health recommendations. Consult healthcare providers for personalized advice.'}"""


def create_tips_agent(storage=None):
    config = get_agent_config("tips")
    if storage:
        config["storage"] = storage
    
    return Agent(
        **config,
        tools=[get_health_tips],
    )


def handle_tips_request(agent, user_query):
    try:
        response = agent.run(user_query, stream=False)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Sorry, I couldn't get those tips: {e}"


def save_chat_markdown(chat_log, title="# Health Tips Chat"):
    try:
        default_name = "tips_chat.md"
        filename = input("Save as (default tips_chat.md): ").strip() or default_name
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

    agent = create_tips_agent()
    print("Health Tips Agent")
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
            response = handle_tips_request(agent, user_query)
            print(response)
            chat_log.append(f"You: {user_query}")
            chat_log.append(f"Agent: {response}")

            if response:
                ans = input("Would you like to save this chat? (y/n): ").strip().lower()
                if ans.startswith("y"):
                    save_chat_markdown(chat_log)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

