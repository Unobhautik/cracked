import os
import sys
from pathlib import Path
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("You'll need to install python-dotenv first: pip install python-dotenv")
    sys.exit(1)

from agents.booking_agent import create_booking_agent, handle_booking_request
from agents.cancellation_agent import create_cancellation_agent, handle_cancellation_request
from agents.reschedule_agent import create_reschedule_agent, handle_reschedule_request
from agents.pdf_analyzer_agent import create_pdf_analyzer_agent
from agents.reminder_agent import create_reminder_agent, handle_reminder_request
from agents.tips_agent import create_tips_agent, handle_tips_request
from agents.symptom_analyzer_agent import create_symptom_analyzer_agent, handle_symptom_analysis
from agents.drug_info_agent import create_drug_info_agent, handle_drug_info_request
from safety_layer import get_safety_layer
from human_review import get_human_review_workflow
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.manager import MemoryManager
from agno.memory.v2.memory import Memory
from config import DEFAULT_STORAGE


def check_api_key():
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        print("OpenAI API key not found!")
        print("Create a .env file in this folder and add:")
        print("OPENAI_API_KEY=sk-your-actual-key-here")
        print("\nYou can get one from: https://platform.openai.com/api-keys")
        return False
    return True


def setup_memory():
    memory_db = SqliteMemoryDb(table_name="medical_memory", db_file="agentmed_history.db")
    memory = Memory(
        db=memory_db,
        memory_manager=MemoryManager(
            memory_capture_instructions="""\
                CRITICALLY IMPORTANT: Remember EVERYTHING the patient mentions in the conversation:
                - All symptoms mentioned (pain, discomfort, conditions like disc bulging, back pain, etc.)
                - Medical history or conditions discussed
                - Previous questions and answers
                - Context about what they're asking for (medicines, doctors, treatments)
                - Any typos or variations (e.g., 'dic' means 'disc', 'doc' means 'doctor')
                
                When patient mentions symptoms, conditions, or concerns, store them permanently for this conversation.
                Always recall previous symptoms when answering new questions.
                If patient mentioned 'lower back pain' and 'disc bulging', remember that for medicine suggestions, doctor recommendations, etc.
            """,
            model=OpenAIChat(id="gpt-4o-mini"),
        ),
    )
    return memory


def setup_agents():
    # Create shared storage for conversation memory
    shared_storage = DEFAULT_STORAGE
    
    agents = {
        'booking': create_booking_agent(storage=shared_storage),
        'cancellation': create_cancellation_agent(storage=shared_storage),
        'reschedule': create_reschedule_agent(storage=shared_storage),
        'pdf_analyzer': create_pdf_analyzer_agent(storage=shared_storage),
        'reminder': create_reminder_agent(storage=shared_storage),
        'tips': create_tips_agent(storage=shared_storage),
        'symptom_analyzer': create_symptom_analyzer_agent(storage=shared_storage),
        'drug_info': create_drug_info_agent(storage=shared_storage),
    }
    return agents


def create_medical_team(agents, memory):
    return Team(
        name="Medical Assistant Team",
        mode="route",
        model=OpenAIChat(id="gpt-4o-mini"),
        members=[
            agents['booking'],
            agents['cancellation'],
            agents['reschedule'],
            agents['pdf_analyzer'],
            agents['reminder'],
            agents['tips'],
            agents['symptom_analyzer'],
            agents['drug_info'],
        ],
        memory=memory,
        instructions=[
            "You are a medical assistant that maintains full conversation context and uses verified medical sources.",
            "CRITICAL: ALWAYS remember and reference the ENTIRE conversation history. If the user mentioned symptoms earlier (like lower back pain, disc bulging), REMEMBER and use that context in all responses.",
            "When patient mentions symptoms, conditions, or concerns, remember them for the ENTIRE conversation and reference them when relevant.",
            "UNDERSTAND TYPOS: If user types 'dic', 'doc', 'medicne', 'sugest' - understand they mean 'disc', 'doctor', 'medicine', 'suggest'. Be intelligent about interpreting user intent.",
            "SAFETY FIRST: Check for emergencies (chest pain, difficulty breathing, etc.) and immediately direct to emergency services if needed.",
            "MEDICINE REQUESTS: When user asks for medicine suggestions (e.g., 'suggest me medicines', 'what medicine should I take'), use the 'Drug Information Agent' member. This agent MUST provide SPECIFIC medication recommendations with FDA information using its tools.",
            "For symptom analysis: Use the symptom_analyzer member to get medically grounded information from FDA, PubMed, NHS, and CDC sources.",
            "For drug/medicine information: Use the 'Drug Information Agent' member. This agent MUST PROVIDE SPECIFIC MEDICATION SUGGESTIONS when asked, using suggest_medications_for_condition tool first, then get_drug_information for each medication.",
            "For tips: Route to tips agent with CURRENT symptoms from conversation context.",
            "For booking: Route to booking agent and suggest appropriate specialist based on CURRENT symptoms from conversation.",
            "For rescheduling: Route to reschedule agent and use booking ID if available from current session.",
            "For PDF analysis: Route to pdf_analyzer agent when user provides a PDF path.",
            "CONTEXT AWARENESS: Always reference previous conversation. If user said 'lower back pain' and 'disc bulging' earlier, remember that when they ask for medicines or doctor recommendations.",
            "Always cite medical sources (FDA, PubMed, NHS, CDC) when providing medical information.",
            "Never provide definitive diagnoses - provide guidance and suggestions based on symptoms and verified medical information.",
            "Always recommend consulting healthcare professionals for serious concerns, but still provide helpful information and suggestions.",
        ],
        markdown=True,
        show_members_responses=True,
        enable_agentic_context=True,
        add_datetime_to_instructions=True,
        success_criteria="Provide accurate, safe, and medically grounded assistance while maintaining conversation context.",
    )


def _looks_like_pdf_path(text: str) -> str | None:
    text = text.strip().strip('"')
    # Windows path with backslashes
    win_path = re.match(r"^[A-Za-z]:\\[^\n]*\.pdf$", text)
    # Windows path with forward slashes
    win_path_forward = re.match(r"^[A-Za-z]:/[^\n]*\.pdf$", text)
    # Posix path
    posix_like = re.match(r"^/.+\.pdf$", text)
    # Any string ending with .pdf
    pdf_ending = text.lower().endswith('.pdf')
    
    if win_path or win_path_forward or posix_like or pdf_ending:
        return text
    return None


def run_unified_chat():
    if not check_api_key():
        return
    
    memory = setup_memory()
    agents = setup_agents()
    team = create_medical_team(agents, memory)
    
    print("\nMedical Assistant")
    print("Type 'exit' or 'quit' to end.")

    last_pdf_path: str | None = None
    last_booking_id: str | None = None
    current_symptoms = []
    user_id = "default_user"
    
    # Initialize safety layer and human review workflow
    safety = get_safety_layer()
    review_workflow = get_human_review_workflow()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break

            # Safety check first
            risk_assessment = safety.assess_risk_level(user_input)
            needs_human_review = review_workflow.should_route_to_human(risk_assessment)
            
            if risk_assessment["is_emergency"]:
                print("\n" + "="*60)
                print("🚨 EMERGENCY DETECTED")
                print("="*60)
                for msg in risk_assessment["messages"]:
                    print(msg)
                print("="*60 + "\n")
                # Still process but with emergency context
                user_input = f"EMERGENCY SITUATION: {user_input}"
            elif risk_assessment["is_high_risk"]:
                print("\n⚠️ HIGH RISK WARNING:")
                for msg in risk_assessment["messages"]:
                    print(msg)
                print()

            # Update context based on user input - capture ALL medical mentions
            medical_keywords = ["fever", "pain", "itch", "ache", "sore", "rash", "red", "swollen", "sciatica", 
                              "suffering", "experiencing", "symptom", "symptoms", "disc", "bulging", "back", 
                              "headache", "numbness", "tingling", "weakness", "dizziness", "nausea", "vomiting",
                              "diarrhea", "constipation", "cough", "sneeze", "congestion", "bleeding", "injury"]
            if any(keyword in user_input.lower() for keyword in medical_keywords):
                current_symptoms.append(user_input)
                # Also store in memory context for the team
                context_note = f"Patient mentioned: {user_input}"
            
            # Check if input looks like a PDF path
            inferred_path = _looks_like_pdf_path(user_input)
            if inferred_path:
                last_pdf_path = inferred_path
                team.print_response(
                    f"I have a PDF file at {inferred_path}. What would you like me to do with it?",
                    stream=True,
                    user_id=user_id
                )
                continue

            # Enhanced context for rescheduling
            if "reschedule" in user_input.lower() and last_booking_id:
                enhanced_query = f"Reschedule appointment {last_booking_id}. {user_input}"
                team.print_response(
                    enhanced_query,
                    stream=True,
                    user_id=user_id
                )
                continue

            # If we have a PDF path and user wants analysis, add context
            if last_pdf_path and any(k in user_input.lower() for k in ["analyze", "analysis", "extract", "summary", "summarize", "overview", "what", "who", "how", "when", "where", "why"]):
                enhanced_query = f"Please analyze the PDF at {last_pdf_path}. {user_input}"
                team.print_response(
                    enhanced_query,
                    stream=True,
                    user_id=user_id
                )
                continue

            # Add context for medicine/drug requests - route to drug_info agent
            if any(word in user_input.lower() for word in ["medicine", "medicines", "medication", "medications", "drug", "drugs", "suggest medicine", "what medicine", "which medicine", "take for"]):
                context_info = ""
                if current_symptoms:
                    context_info = f" Patient symptoms from conversation: {', '.join(current_symptoms[-3:])}. "
                # Also check conversation history for symptoms
                enhanced_query = f"{context_info}User is asking for medicine suggestions. Provide SPECIFIC medication recommendations with FDA information. {user_input}"
                team.print_response(
                    enhanced_query,
                    stream=True,
                    user_id=user_id
                )
                continue

            # Add context for tips requests
            if any(word in user_input.lower() for word in ["tips", "advice", "suggest", "help with"]):
                context_info = ""
                if current_symptoms:
                    context_info = f" Patient symptoms: {', '.join(current_symptoms[-3:])}. "
                enhanced_query = f"{context_info}{user_input}"
                team.print_response(
                    enhanced_query,
                    stream=True,
                    user_id=user_id
                )
                continue

            # Add context for booking requests
            if any(word in user_input.lower() for word in ["book", "appointment", "schedule", "see doctor", "need doc", "which doctor", "best doctor"]):
                context_info = ""
                if current_symptoms:
                    context_info = f" Patient symptoms from conversation: {', '.join(current_symptoms[-3:])}. "
                enhanced_query = f"{context_info}Based on the symptoms mentioned, suggest appropriate specialist. {user_input}"
                team.print_response(
                    enhanced_query,
                    stream=True,
                    user_id=user_id
                )
                continue

            # Use the team for all other interactions
            # Get response (we'll capture it for human review if needed)
            response_text = ""
            try:
                # For now, we'll use print_response which streams
                # In production, you'd capture the full response
                team.print_response(
                    user_input,
                    stream=True,
                    user_id=user_id
                )
                
                # If human review is needed, create review request
                if needs_human_review:
                    print("\n" + "="*60)
                    print("📋 This case has been flagged for clinician review")
                    print("="*60)
                    review_request = review_workflow.create_review_request(
                        user_query=user_input,
                        risk_assessment=risk_assessment,
                        ai_response=response_text or "Response generated (see above)",
                        context={"symptoms": current_symptoms[-2:] if current_symptoms else []}
                    )
                    print(f"Review ID: {review_request['review_id']}")
                    print("A clinician will review this case and provide feedback.")
                    print("="*60 + "\n")
            except Exception as e:
                print(f"Error processing request: {e}")
                
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_unified_chat()
