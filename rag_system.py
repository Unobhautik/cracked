"""
RAG (Retrieval-Augmented Generation) System for Medical Knowledge
Integrates FDA, PubMed, NHS, and CDC data sources
"""
import os
from pathlib import Path
from typing import List, Dict, Optional
import json

try:
    import lancedb
    from lancedb.pydantic import Vector, LanceModel
    LANCEDB_AVAILABLE = True
except (ImportError, ValueError, AttributeError, Exception) as e:
    LANCEDB_AVAILABLE = False
    # Silently handle - RAG will work in limited mode
    Vector = None
    LanceModel = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


# MedicalDocument schema - only defined if LanceDB is available
if LANCEDB_AVAILABLE and LanceModel:
    class MedicalDocument(LanceModel):
        """Schema for medical documents in vector DB"""
        id: str
        text: str
        source: str  # "FDA", "PubMed", "NHS", "CDC"
        category: str  # "drug", "symptom", "condition", "guideline"
        metadata: str  # JSON string with additional info
        vector: Vector(384)  # Default embedding dimension
else:
    MedicalDocument = None


class RAGSystem:
    """RAG system for medical knowledge retrieval"""
    
    def __init__(self, db_path: str = "medical_rag.db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.db = None
        self.embedding_model = None
        self.table = None
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
        
        if LANCEDB_AVAILABLE:
            try:
                self.db = lancedb.connect(db_path)
                # Try to open existing table or create new one
                try:
                    self.table = self.db.open_table("medical_knowledge")
                    print("Opened existing medical knowledge database")
                except:
                    self.table = None
                    print("Medical knowledge database will be created on first document insertion")
            except Exception as e:
                print(f"Warning: Could not initialize LanceDB: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.embedding_model:
            # Fallback: return zero vector
            return [0.0] * 384
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 384
    
    def add_document(self, text: str, source: str, category: str, metadata: Optional[Dict] = None):
        """Add a medical document to the vector database"""
        if not LANCEDB_AVAILABLE or not self.db:
            print("LanceDB not available. Document not added.")
            return False
        
        try:
            doc_id = f"{source}_{category}_{hash(text) % 1000000}"
            embedding = self._get_embedding(text)
            metadata_str = json.dumps(metadata or {})
            
            doc = {
                "id": doc_id,
                "text": text,
                "source": source,
                "category": category,
                "metadata": metadata_str,
                "vector": embedding
            }
            
            if self.table is None:
                # Create table with first document
                self.table = self.db.create_table("medical_knowledge", data=[doc], mode="overwrite")
                print(f"Created medical knowledge table with first document")
            else:
                self.table.add([doc])
                print(f"Added document from {source} ({category})")
            
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, source: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """Search for relevant medical documents"""
        if not LANCEDB_AVAILABLE or not self.table:
            return []
        
        try:
            query_embedding = self._get_embedding(query)
            
            # Build search query
            search_query = self.table.search(query_embedding).limit(top_k)
            
            # Apply filters if provided
            if source:
                search_query = search_query.where(f"source = '{source}'")
            if category:
                search_query = search_query.where(f"category = '{category}'")
            
            results = search_query.to_pandas()
            
            # Convert to list of dicts
            documents = []
            for _, row in results.iterrows():
                documents.append({
                    "text": row["text"],
                    "source": row["source"],
                    "category": row["category"],
                    "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
                    "score": getattr(row, "_distance", 0.0)  # Similarity score
                })
            
            return documents
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 5) -> str:
        """Get formatted context string for RAG"""
        documents = self.search(query, top_k=top_k)
        
        if not documents:
            return "No relevant medical information found in knowledge base."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc["source"]
            category = doc["category"]
            text = doc["text"][:500]  # Limit text length
            context_parts.append(f"[{i}] Source: {source} ({category})\n{text}")
        
        return "\n\n".join(context_parts)


# Global RAG instance
_rag_instance = None

def get_rag_system() -> RAGSystem:
    """Get or create global RAG system instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance

