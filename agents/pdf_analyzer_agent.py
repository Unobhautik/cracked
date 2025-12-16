from agno.agent import Agent
from agno.media import File
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
def analyze_medical_pdf(pdf_path: str, user_query: str = "Give me a general overview of this medical document."):
    """Analyze a medical PDF document and answer questions about it"""
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            return f"Sorry, I couldn't find the PDF file at {pdf_path}. Please check the file path and try again."
        
        file_obj = File(filepath=Path(pdf_path))
        
        if not user_query.strip():
            user_query = "Give me a general overview of this medical document."
        
        # Add context about the file being analyzed
        enhanced_query = f"Please analyze this medical PDF document. {user_query}"
        
        # Create a temporary agent to process the file
        from agno.models.openai import OpenAIChat
        temp_agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions="You are a medical document analyzer. Analyze the provided PDF and answer questions about it."
        )
        
        response = temp_agent.run(enhanced_query, files=[file_obj], stream=False)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except FileNotFoundError:
        return f"Sorry, I couldn't find the PDF file at {pdf_path}. Please check the file path and try again."
    except PermissionError:
        return f"Sorry, I don't have permission to access the file at {pdf_path}. Please check file permissions."
    except Exception as e:
        return f"Sorry, I couldn't analyze that PDF: {str(e)}. Please make sure the file exists and is accessible."

def create_pdf_analyzer_agent(storage=None):
    config = get_agent_config("pdf_analyzer")
    if storage:
        config["storage"] = storage
    
    return Agent(**config, tools=[analyze_medical_pdf])

def save_chat_markdown(chat_log, title="# PDF Analyzer Chat"):
    try:
        default_name = "pdf_chat.md"
        filename = input("Save as (default pdf_chat.md): ").strip() or default_name
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

    agent = create_pdf_analyzer_agent()
    print("PDF Analyzer Agent")
    print("Type 'back' to exit.")

    chat_log = []

    while True:
        try:
            user_query = input("Question about your PDF: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["back", "exit", "quit"]:
                print("Goodbye.")
                break
            pdf_path = input("Full path to the PDF: ").strip().strip('"')
            response = analyze_medical_pdf(pdf_path, user_query)
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

