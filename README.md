# MedicalAI - Agentic AI Healthcare System

An intelligent, medically-grounded healthcare management system built with **Agno AI** agents and RAG (Retrieval-Augmented Generation). This system integrates verified medical knowledge from FDA, PubMed, NHS, and CDC sources to provide safe, evidence-based healthcare assistance.

## 🏥 What is MedicalAI?

MedicalAI is an advanced agentic AI system that combines multiple specialized health assistants with a comprehensive medical knowledge base. Each agent has a specific role and uses verified medical sources (FDA, PubMed, NHS, CDC) to provide safe, grounded healthcare guidance. The system includes safety layers, emergency detection, and human-in-the-loop workflows for high-risk cases.

## ✨ What Can It Do?

### 📅 **Appointment Booking Agent**
- Books doctor appointments with the right specialists
- Finds available time slots that work for you
- Helps you choose the right hospital or clinic
- Talks like a helpful friend who knows the healthcare system

### ❌ **Appointment Cancellation Agent** 
- Cancels appointments when life gets in the way
- Explains cancellation policies in plain English
- Helps with refunds if applicable
- Understanding and non-judgmental

### 🔄 **Appointment Rescheduling Agent**
- Changes appointment times when you need to
- Finds new slots that fit your schedule better
- Flexible and accommodating
- Like having a helpful receptionist on speed dial

### 📄 **Medical PDF Analyzer Agent**
- Reads your medical reports and explains what they mean
- Translates medical jargon into normal English
- Highlights important findings and what to ask your doctor about
- Think of it as that friend who went to med school

### 💊 **Medication Reminder Agent**
- Sets up medication schedules that work for you
- Gives safety tips and warns about interactions
- Helps you never miss a dose
- Caring and safety-focused

### 💡 **Health Tips Agent**
- Gives personalized wellness advice using medical knowledge sources
- Shares practical tips backed by NHS and CDC guidelines
- Considers your lifestyle and health concerns
- Like chatting with a knowledgeable wellness friend

### 🔍 **Symptom Analyzer Agent** (NEW)
- Analyzes symptoms using verified medical sources (FDA, PubMed, NHS, CDC)
- Provides triage recommendations (emergency, urgent, routine)
- Retrieves relevant medical research and guidelines
- Safety-first approach with emergency detection

### 💊 **Drug Information Agent** (NEW)
- Provides comprehensive drug information from FDA sources
- Checks for drug interactions between multiple medications
- Retrieves safety data and contraindications
- Helps understand medication usage and side effects

## 🏗️ How It's Built

The system uses **Agno AI** with a comprehensive architecture:

### Core Components:
1. **RAG System** - Vector database (LanceDB) with medical knowledge embeddings
2. **Medical Knowledge Retrieval** - Integration with FDA, PubMed, NHS, and CDC APIs
3. **Safety Layer** - Emergency detection and risk classification
4. **Human-in-the-Loop** - Workflow for high-risk case review
5. **Agentic Team** - 8 specialized agents working together

### Medical Knowledge Sources:
- **FDA** - Drug labels, adverse events, medication guidelines
- **PubMed** - Biomedical research abstracts and articles
- **NHS** - Condition information, symptoms, treatment guidance
- **CDC** - Disease guidelines, prevention, emergency symptoms

```
agentmed/
├── agents/
│   ├── __init__.py
│   ├── booking_agent.py          # Appointment booking specialist
│   ├── cancellation_agent.py     # Cancellation helper
│   ├── reschedule_agent.py       # Rescheduling expert
│   ├── pdf_analyzer_agent.py     # Medical report reader
│   ├── reminder_agent.py         # Medication buddy
│   ├── tips_agent.py              # Wellness advisor (enhanced with RAG)
│   ├── symptom_analyzer_agent.py # Symptom analysis with medical sources
│   └── drug_info_agent.py        # Drug information and interactions
├── rag_system.py                  # RAG system with vector database
├── medical_knowledge.py           # Medical knowledge retrieval (FDA, PubMed, etc.)
├── safety_layer.py                # Safety checks and emergency detection
├── human_review.py                # Human-in-the-loop workflow
├── medical_ai.py                  # Main unified interface
├── config.py                      # Configuration
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── agentmed_history.db            # Chat history (auto-created)
└── medical_rag.db                 # Vector database (auto-created)
```

## 🚀 Getting Started

### What You Need
- Python 3.8+
- OpenAI API key (get one from [OpenAI](https://platform.openai.com/api-keys))

### Quick Setup

1. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the project folder:
   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Run the system:**
   ```bash
   python medical_ai.py
   ```
   
   **Note**: The first run will download the embedding model (~80MB) for the RAG system.

## 📱 Using MedicalAI

### Starting Up
Just run the command and you'll see a friendly menu:
```bash
python medical_ai_cli.py
```

### The Interface
The system feels like chatting with helpful friends:
- **Natural language** - No need to memorize commands
- **Conversational** - Each agent has its own personality
- **Helpful** - They guide you through each process
- **Friendly** - No corporate robot speak

### Example Conversations

#### Booking an Appointment
```
You: "I need to see a cardiologist next week"
Agent: "Hey! Let's get you booked with a cardiologist. 
       What day works best for you? And do you have a 
       preference for morning or afternoon?"
```

#### Understanding a Medical Report
```
You: "Can you explain this blood test report?"
Agent: "Sure thing! Let me take a look at your results. 
       I see your cholesterol levels are a bit high - 
       let me break down what that means and what 
       you should ask your doctor about."
```

## 🔧 Customizing Agents

Each agent is in its own file, so you can easily:
- **Change personalities** - Make them more formal, casual, or specialized
- **Add new features** - Extend what each agent can do
- **Modify instructions** - Adjust how they respond to users
- **Add new agents** - Create specialized assistants for other health needs

## 🛡️ Privacy & Security

- **Local storage** - All chat history and vector database stay on your computer
- **No data sharing** - Your medical info never leaves your system
- **Secure API calls** - Only sends queries to OpenAI and medical APIs, not your full data
- **Safety layer** - Automatic emergency detection and risk assessment
- **Human review** - High-risk cases are flagged for clinician review
- **You control everything** - Delete databases anytime

## 🚨 Safety Features

The system includes multiple safety layers:

1. **Emergency Detection** - Automatically detects emergency keywords (chest pain, difficulty breathing, etc.)
2. **Risk Classification** - Categorizes queries as low, medium, high, or emergency risk
3. **Medical Grounding** - All medical information comes from verified sources (FDA, PubMed, NHS, CDC)
4. **Human Review Workflow** - High-risk cases are automatically flagged for clinician review
5. **Safety Disclaimers** - All responses include appropriate medical disclaimers
6. **No Diagnoses** - System provides guidance only, never definitive diagnoses

## 🚨 Important Notes

- **Not medical advice** - This system provides information and guidance based on verified medical sources, but is NOT a substitute for professional medical care. Always consult healthcare professionals for medical decisions.
- **Emergency situations** - The system will detect emergencies and direct you to call 911 or go to the emergency room, but in a real emergency, don't wait - call emergency services immediately.
- **API costs** - OpenAI charges for API usage, so keep an eye on your usage
- **Medical knowledge** - The RAG system starts empty. You can populate it with medical documents from FDA, PubMed, NHS, and CDC sources.
- **Local files** - PDF analysis works with files on your computer
- **Internet required** - Medical knowledge retrieval requires internet access for FDA and PubMed APIs

## 🆘 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Make sure you created a `.env` file
- Check that the key starts with `sk-`
- Verify the file is in the same folder as the script

**"Can't import agents"**
- Make sure you're in the right folder
- Check that all the agent files are present
- Try running `pip install -r requirements.txt` again

**PDF analysis fails**
- Check that the file path is correct
- Make sure the PDF isn't password-protected
- Verify the file isn't corrupted

## 🔮 What's Next?

The system is designed for expansion:

### Immediate Enhancements:
- Populate RAG system with medical documents from FDA, PubMed, NHS, CDC
- Fine-tune custom medical LLM (Mistral/Llama/Qwen) as described in the architecture document
- Add more specialized agents (lab results analyzer, treatment planner, etc.)
- Web interface for easier access
- Mobile app integration

### Advanced Features:
- Custom fine-tuned medical model (replacing OpenAI API)
- Integration with hospital appointment systems
- Real-time medical data updates
- Multi-language support
- Telemedicine integration
- Patient record management

### Architecture Document:
See `Agentic_AI_Healthcare.docx` for the complete system architecture, including:
- Custom LLM fine-tuning pipeline
- RAG system detailed design
- Training datasets and sources
- Deployment strategies
- Commercialization roadmap

## 💬 Support

If you run into issues:
1. Check the troubleshooting section above
2. Make sure all dependencies are installed
3. Verify your OpenAI API key is working
4. Check that all agent files are present

---

**Built with ❤️ using Agno AI - Making healthcare AI feel human**
