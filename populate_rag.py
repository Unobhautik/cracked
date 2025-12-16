"""
Script to populate RAG system with initial medical knowledge
This is a helper script - you can add your own medical documents here
"""
from rag_system import get_rag_system
import json


def add_sample_medical_knowledge():
    """Add sample medical knowledge to RAG system"""
    rag = get_rag_system()
    
    # Sample medical knowledge entries
    # In production, you would load these from FDA, PubMed, NHS, CDC APIs or datasets
    
    sample_documents = [
        {
            "text": "Chest pain can be a sign of a heart attack. If you experience severe chest pain, especially with shortness of breath, nausea, or pain radiating to the arm, call emergency services immediately. Other causes include angina, acid reflux, or muscle strain.",
            "source": "NHS",
            "category": "symptom",
            "metadata": {"condition": "chest pain", "urgency": "high"}
        },
        {
            "text": "Fever is usually a sign that your body is fighting an infection. A temperature above 38°C (100.4°F) is considered a fever. Most fevers are not serious, but seek medical attention if fever persists for more than 3 days, is very high (above 40°C/104°F), or is accompanied by severe symptoms.",
            "source": "NHS",
            "category": "symptom",
            "metadata": {"condition": "fever", "urgency": "medium"}
        },
        {
            "text": "Aspirin (acetylsalicylic acid) is used to reduce pain, fever, and inflammation. Common side effects include stomach irritation and increased bleeding risk. Do not give aspirin to children under 16 due to risk of Reye's syndrome. Consult a doctor before use if you have stomach ulcers, bleeding disorders, or are taking blood thinners.",
            "source": "FDA",
            "category": "drug",
            "metadata": {"drug": "aspirin", "class": "NSAID"}
        },
        {
            "text": "High blood pressure (hypertension) is a common condition that increases the risk of heart disease and stroke. Lifestyle changes include reducing salt intake, regular exercise, maintaining healthy weight, limiting alcohol, and managing stress. Medication may be needed if lifestyle changes are insufficient.",
            "source": "CDC",
            "category": "condition",
            "metadata": {"condition": "hypertension", "type": "chronic"}
        },
        {
            "text": "Diabetes management involves monitoring blood sugar levels, following a healthy diet, regular physical activity, and taking medications as prescribed. Type 1 diabetes requires insulin, while Type 2 may be managed with oral medications, insulin, or both. Regular check-ups are essential.",
            "source": "NHS",
            "category": "condition",
            "metadata": {"condition": "diabetes", "type": "chronic"}
        },
    ]
    
    print("Adding sample medical knowledge to RAG system...")
    added_count = 0
    
    for doc in sample_documents:
        success = rag.add_document(
            text=doc["text"],
            source=doc["source"],
            category=doc["category"],
            metadata=doc.get("metadata", {})
        )
        if success:
            added_count += 1
    
    print(f"Successfully added {added_count} documents to the RAG system.")
    print("\nNote: This is just sample data. For production use, you should:")
    print("1. Download medical datasets from FDA, PubMed, NHS, CDC")
    print("2. Process and chunk the documents appropriately")
    print("3. Add them to the RAG system using this script as a template")
    print("\nSee Agentic_AI_Healthcare.docx for detailed instructions on data sources.")


if __name__ == "__main__":
    add_sample_medical_knowledge()

