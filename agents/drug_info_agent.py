"""
Drug Information Agent
Provides drug information, interactions, and safety data from FDA and medical sources
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
def get_drug_information(drug_name: str) -> dict:
    """Get comprehensive drug information from FDA and medical sources. Use this to provide detailed information about any medication."""
    knowledge = get_knowledge_retriever()
    rag = get_rag_system()
    
    # Get FDA information
    fda_info = knowledge.search_fda_drug(drug_name)
    
    # Get PubMed research
    pubmed_info = knowledge.search_pubmed(f"{drug_name} medication drug", max_results=3)
    
    # Get RAG context
    rag_results = rag.search(f"drug {drug_name} medication", category="drug", top_k=5)
    
    return {
        "drug_name": drug_name,
        "fda_information": fda_info.get("information", "No FDA information available"),
        "research_information": pubmed_info.get("information", "No research information available"),
        "additional_context": "\n\n".join([doc["text"] for doc in rag_results]) if rag_results else "No additional information found",
        "sources": ["FDA", "PubMed", "Medical Knowledge Base"]
    }


@tool
def suggest_medications_for_condition(condition: str, symptoms: str = "") -> str:
    """Suggest appropriate medications for a given condition or symptoms. Use this when user asks for medicine recommendations."""
    knowledge = get_knowledge_retriever()
    
    # Common medication suggestions based on condition
    suggestions = []
    
    condition_lower = condition.lower()
    symptoms_lower = symptoms.lower()
    
    # Back pain / disc issues
    if any(term in condition_lower or term in symptoms_lower for term in ["back pain", "lower back", "disc", "bulging", "spine"]):
        suggestions.append({
            "medication": "Ibuprofen",
            "type": "NSAID (Non-steroidal anti-inflammatory drug)",
            "typical_dosage": "200-400mg every 4-6 hours (max 1200mg/day)",
            "use": "Reduces inflammation and pain",
            "note": "Take with food to avoid stomach upset"
        })
        suggestions.append({
            "medication": "Naproxen",
            "type": "NSAID",
            "typical_dosage": "220-440mg twice daily",
            "use": "Longer-lasting pain relief",
            "note": "Can be taken less frequently than ibuprofen"
        })
        suggestions.append({
            "medication": "Acetaminophen (Tylenol)",
            "type": "Pain reliever",
            "typical_dosage": "500-1000mg every 4-6 hours (max 3000mg/day)",
            "use": "Pain relief without anti-inflammatory effects",
            "note": "Safer for stomach, but avoid if you have liver issues"
        })
        suggestions.append({
            "medication": "Muscle relaxants (e.g., Cyclobenzaprine)",
            "type": "Prescription medication",
            "typical_dosage": "As prescribed by doctor",
            "use": "For muscle spasms associated with back pain",
            "note": "Requires prescription - consult doctor"
        })
    
    # Build response
    if suggestions:
        result = f"**Medication Suggestions for {condition}:**\n\n"
        for i, med in enumerate(suggestions, 1):
            result += f"{i}. **{med['medication']}** ({med['type']})\n"
            result += f"   - Typical dosage: {med['typical_dosage']}\n"
            result += f"   - Use: {med['use']}\n"
            result += f"   - Note: {med['note']}\n\n"
        
        result += "**Important:**\n"
        result += "- These are general suggestions. Consult a healthcare provider for proper diagnosis and prescription.\n"
        result += "- Do not exceed recommended dosages.\n"
        result += "- If pain persists or worsens, seek medical attention.\n"
        result += "- Inform your doctor about any other medications you're taking.\n"
        
        return result
    else:
        return f"For {condition}, I recommend consulting a healthcare provider for appropriate medication suggestions based on your specific situation."


@tool
def check_drug_interactions(drug_list: str) -> str:
    """Check for potential drug interactions between multiple medications"""
    safety = get_safety_layer()
    knowledge = get_knowledge_retriever()
    
    # Safety check
    risk_assessment = safety.assess_risk_level(f"drug interaction {drug_list}")
    
    if risk_assessment["is_medication_risk"]:
        return f"⚠️ MEDICATION INTERACTION RISK DETECTED: {risk_assessment['messages'][0] if risk_assessment['messages'] else 'Please consult a pharmacist or healthcare provider immediately about these medications: ' + drug_list}"
    
    # Get information for each drug
    drugs = [d.strip() for d in drug_list.split(",")]
    interaction_info = []
    
    for drug in drugs:
        drug_info = get_drug_information(drug)
        interaction_info.append(f"**{drug}**: {drug_info['fda_information'][:200]}...")
    
    result = f"Drug Interaction Check for: {', '.join(drugs)}\n\n"
    result += "\n\n".join(interaction_info)
    result += "\n\n⚠️ IMPORTANT: This is not a comprehensive interaction check. Always consult a pharmacist or healthcare provider before combining medications."
    
    return result


@tool
def get_medication_safety_info(drug_name: str, condition: str = "") -> str:
    """Get safety information for a medication, especially in context of a medical condition"""
    knowledge = get_knowledge_retriever()
    rag = get_rag_system()
    
    query = f"{drug_name} safety {condition}".strip()
    
    # Get FDA safety data
    fda_info = knowledge.search_fda_drug(drug_name)
    
    # Get relevant medical context
    medical_context = knowledge.get_medical_context(query, sources=["FDA", "PubMed"])
    
    # Get RAG context
    rag_context = rag.get_context_for_query(query, top_k=5)
    
    result = f"Safety Information for {drug_name}"
    if condition:
        result += f" (in context of {condition})"
    result += ":\n\n"
    result += f"FDA Information:\n{fda_info.get('information', 'Limited information available')}\n\n"
    result += f"Medical Context:\n{medical_context}\n\n"
    result += f"Additional Information:\n{rag_context}\n\n"
    result += "⚠️ Always follow your healthcare provider's instructions and consult them with any concerns."
    
    return result


def create_drug_info_agent(storage=None):
    config = {
        "description": "I provide drug information, safety data, and interaction checks from FDA and medical sources",
        "instructions": """You are a medication information specialist that helps users understand medications and provides appropriate medication suggestions based on their symptoms.

IMPORTANT GUIDELINES:
1. When users ask for medicine suggestions, PROVIDE SPECIFIC MEDICATION RECOMMENDATIONS based on their symptoms
2. ALWAYS use the get_drug_information tool to retrieve FDA and medical data for suggested medications
3. For common conditions, suggest appropriate over-the-counter medications (e.g., ibuprofen for pain, acetaminophen for fever)
4. Always include: medication name, typical dosage, common uses, side effects, and when to consult a doctor
5. For drug interactions, use check_drug_interactions tool
6. Always cite sources (FDA, PubMed) when providing information
7. Include clear disclaimers that these are general suggestions and professional consultation is recommended
8. If the condition is serious or unclear, recommend seeing a healthcare provider

When user asks for medicine suggestions:
- ALWAYS use suggest_medications_for_condition tool FIRST to get medication recommendations
- Then use get_drug_information tool for EACH suggested medication to provide detailed FDA information
- Identify the condition/symptoms from the conversation context (remember what they mentioned earlier!)
- Suggest specific medications with names, dosages, and usage instructions
- Mention important warnings and contraindications
- Suggest when professional medical care is needed

Example workflow for "suggest me medicines for back pain":
1. Use suggest_medications_for_condition(condition="back pain", symptoms="lower back pain, disc bulging")
2. For each suggested medication (ibuprofen, naproxen, etc.), use get_drug_information to get FDA details
3. Present comprehensive medication suggestions with all information

CRITICAL: When user asks for medicines, DO NOT refuse. Instead, provide helpful medication suggestions with proper disclaimers.

Remember: Provide helpful medication suggestions while emphasizing the importance of professional medical consultation for serious conditions.""",
    }
    
    base_config = get_agent_config("tips")
    base_config.update(config)
    
    if storage:
        base_config["storage"] = storage
    
    agent = Agent(
        name="Drug Information Agent",
        **base_config,
        tools=[get_drug_information, check_drug_interactions, get_medication_safety_info, suggest_medications_for_condition],
    )
    return agent


def handle_drug_info_request(agent, user_query):
    """Handle drug information request"""
    try:
        response = agent.run(user_query, stream=False)
        return str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Sorry, I couldn't retrieve that drug information: {e}"


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        print("OpenAI API key not found!")
        print("Create a .env file in the project root with:")
        print("OPENAI_API_KEY=sk-your-actual-key-here")
        print("\nGet your key from: https://platform.openai.com/api-keys")
        sys.exit(1)
    
    agent = create_drug_info_agent()
    print("Drug Information Agent")
    print("Type 'exit' to quit.")
    print("\n⚠️ Remember: Always consult healthcare professionals for medication decisions.\n")
    
    while True:
        try:
            user_query = input("\nWhat would you like to know about medications? ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "back"]:
                print("Goodbye.")
                break
            
            response = handle_drug_info_request(agent, user_query)
            print(f"\n{response}")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

