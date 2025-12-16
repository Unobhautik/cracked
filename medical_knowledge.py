"""
Medical Knowledge Retrieval Tools
Integrates with FDA, PubMed, NHS, and CDC data sources
"""
import requests
import json
from typing import List, Dict, Optional
from rag_system import get_rag_system


class MedicalKnowledgeRetriever:
    """Retrieves medical information from various sources"""
    
    def __init__(self):
        self.rag = get_rag_system()
    
    def search_fda_drug(self, drug_name: str) -> Dict:
        """Search FDA drug information"""
        try:
            # OpenFDA API endpoint
            url = f"https://api.fda.gov/drug/label.json"
            params = {
                "search": f"openfda.brand_name:{drug_name}",
                "limit": 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    return {
                        "source": "FDA",
                        "drug_name": drug_name,
                        "information": json.dumps(result, indent=2),
                        "success": True
                    }
            
            # Fallback: search RAG system
            rag_results = self.rag.search(f"FDA drug {drug_name}", source="FDA", category="drug", top_k=3)
            if rag_results:
                return {
                    "source": "FDA (RAG)",
                    "drug_name": drug_name,
                    "information": "\n\n".join([doc["text"] for doc in rag_results]),
                    "success": True
                }
            
            return {
                "source": "FDA",
                "drug_name": drug_name,
                "information": f"Limited information available for {drug_name}. Please consult a healthcare provider.",
                "success": False
            }
        except Exception as e:
            return {
                "source": "FDA",
                "drug_name": drug_name,
                "information": f"Error retrieving FDA data: {str(e)}",
                "success": False
            }
    
    def search_pubmed(self, query: str, max_results: int = 3) -> Dict:
        """Search PubMed for medical research"""
        try:
            # PubMed E-utilities API
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=10)
            if search_response.status_code == 200:
                search_data = search_response.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if pmids:
                    # Fetch abstracts
                    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(pmids),
                        "retmode": "xml"
                    }
                    
                    fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)
                    if fetch_response.status_code == 200:
                        # Parse XML (simplified - in production use proper XML parser)
                        abstracts = []
                        content = fetch_response.text
                        # Simple extraction (for production, use xml.etree.ElementTree)
                        if "AbstractText" in content:
                            return {
                                "source": "PubMed",
                                "query": query,
                                "information": f"Found {len(pmids)} relevant research articles. PMIDs: {', '.join(pmids)}",
                                "pmids": pmids,
                                "success": True
                            }
            
            # Fallback: search RAG system
            rag_results = self.rag.search(f"PubMed {query}", source="PubMed", top_k=max_results)
            if rag_results:
                return {
                    "source": "PubMed (RAG)",
                    "query": query,
                    "information": "\n\n".join([doc["text"] for doc in rag_results]),
                    "success": True
                }
            
            return {
                "source": "PubMed",
                "query": query,
                "information": f"Limited research found for '{query}'. Please consult healthcare professionals.",
                "success": False
            }
        except Exception as e:
            return {
                "source": "PubMed",
                "query": query,
                "information": f"Error retrieving PubMed data: {str(e)}",
                "success": False
            }
    
    def search_nhs_condition(self, condition: str) -> Dict:
        """Search NHS condition information"""
        try:
            # NHS doesn't have a public API, so we'll use RAG system
            rag_results = self.rag.search(f"NHS {condition}", source="NHS", category="condition", top_k=5)
            
            if rag_results:
                return {
                    "source": "NHS",
                    "condition": condition,
                    "information": "\n\n".join([doc["text"] for doc in rag_results]),
                    "success": True
                }
            
            return {
                "source": "NHS",
                "condition": condition,
                "information": f"Information about {condition} from NHS guidelines. Please consult NHS website or healthcare provider for detailed information.",
                "success": False
            }
        except Exception as e:
            return {
                "source": "NHS",
                "condition": condition,
                "information": f"Error retrieving NHS data: {str(e)}",
                "success": False
            }
    
    def search_cdc_guidelines(self, topic: str) -> Dict:
        """Search CDC guidelines"""
        try:
            # CDC doesn't have a simple API, so we'll use RAG system
            rag_results = self.rag.search(f"CDC {topic}", source="CDC", category="guideline", top_k=5)
            
            if rag_results:
                return {
                    "source": "CDC",
                    "topic": topic,
                    "information": "\n\n".join([doc["text"] for doc in rag_results]),
                    "success": True
                }
            
            return {
                "source": "CDC",
                "topic": topic,
                "information": f"CDC guidelines for {topic}. Please consult CDC website for official guidelines.",
                "success": False
            }
        except Exception as e:
            return {
                "source": "CDC",
                "topic": topic,
                "information": f"Error retrieving CDC data: {str(e)}",
                "success": False
            }
    
    def get_medical_context(self, query: str, sources: Optional[List[str]] = None) -> str:
        """Get comprehensive medical context from all sources"""
        if sources is None:
            sources = ["FDA", "PubMed", "NHS", "CDC"]
        
        context_parts = []
        
        if "FDA" in sources:
            # Try to extract drug name from query
            fda_info = self.search_fda_drug(query)
            if fda_info.get("success"):
                context_parts.append(f"FDA Information:\n{fda_info['information']}")
        
        if "PubMed" in sources:
            pubmed_info = self.search_pubmed(query)
            if pubmed_info.get("success"):
                context_parts.append(f"PubMed Research:\n{pubmed_info['information']}")
        
        if "NHS" in sources:
            nhs_info = self.search_nhs_condition(query)
            if nhs_info.get("success"):
                context_parts.append(f"NHS Guidelines:\n{nhs_info['information']}")
        
        if "CDC" in sources:
            cdc_info = self.search_cdc_guidelines(query)
            if cdc_info.get("success"):
                context_parts.append(f"CDC Guidelines:\n{cdc_info['information']}")
        
        if not context_parts:
            # Fallback to RAG search
            rag_context = self.rag.get_context_for_query(query, top_k=5)
            if rag_context and "No relevant" not in rag_context:
                context_parts.append(f"Medical Knowledge Base:\n{rag_context}")
        
        return "\n\n".join(context_parts) if context_parts else "No medical information found. Please consult a healthcare professional."


# Global instance
_knowledge_retriever = None

def get_knowledge_retriever() -> MedicalKnowledgeRetriever:
    """Get or create global knowledge retriever instance"""
    global _knowledge_retriever
    if _knowledge_retriever is None:
        _knowledge_retriever = MedicalKnowledgeRetriever()
    return _knowledge_retriever

