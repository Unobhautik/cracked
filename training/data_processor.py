"""
Data Processing and Cleaning Pipeline
Cleans and processes raw medical data into training-ready format
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import html


class MedicalDataProcessor:
    """Processes and cleans medical data"""
    
    def __init__(self, raw_data_dir: str = "training/data/raw", 
                 processed_dir: str = "training/data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean text: remove HTML, normalize whitespace, etc."""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = text.strip()
        
        return text
    
    def process_fda_data(self) -> List[Dict]:
        """Process FDA drug label data"""
        print("Processing FDA drug labels...")
        input_file = self.raw_data_dir / "fda_drug_labels.json"
        
        if not input_file.exists():
            print(f"File not found: {input_file}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        
        for item in data:
            try:
                # Extract relevant information
                brand_name = item.get("openfda", {}).get("brand_name", [""])[0] if item.get("openfda") else ""
                generic_name = item.get("openfda", {}).get("generic_name", [""])[0] if item.get("openfda") else ""
                indications = self.clean_text(item.get("indications_and_usage", [""])[0] if isinstance(item.get("indications_and_usage"), list) else item.get("indications_and_usage", ""))
                warnings = self.clean_text(item.get("warnings", [""])[0] if isinstance(item.get("warnings"), list) else item.get("warnings", ""))
                dosage = self.clean_text(item.get("dosage_and_administration", [""])[0] if isinstance(item.get("dosage_and_administration"), list) else item.get("dosage_and_administration", ""))
                
                if brand_name or generic_name:
                    processed.append({
                        "source": "FDA",
                        "drug_name": brand_name or generic_name,
                        "generic_name": generic_name,
                        "indications": indications,
                        "warnings": warnings,
                        "dosage": dosage,
                        "raw_data": item
                    })
            except Exception as e:
                print(f"Error processing FDA item: {e}")
                continue
        
        print(f"Processed {len(processed)} FDA drug labels")
        return processed
    
    def process_pubmed_data(self) -> List[Dict]:
        """Process PubMed abstracts"""
        print("Processing PubMed abstracts...")
        input_file = self.raw_data_dir / "pubmed_abstracts.json"
        
        if not input_file.exists():
            print(f"File not found: {input_file}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        
        for item in data:
            try:
                title = self.clean_text(item.get("title", ""))
                abstract = self.clean_text(item.get("abstract", ""))
                
                if title and abstract:
                    processed.append({
                        "source": "PubMed",
                        "pmid": item.get("pmid", ""),
                        "title": title,
                        "abstract": abstract,
                        "full_text": f"{title}. {abstract}"
                    })
            except Exception as e:
                print(f"Error processing PubMed item: {e}")
                continue
        
        print(f"Processed {len(processed)} PubMed abstracts")
        return processed
    
    def process_medquad_data(self) -> List[Dict]:
        """Process MedQuAD Q&A dataset"""
        print("Processing MedQuAD data...")
        input_file = self.raw_data_dir / "medquad.json"
        
        if not input_file.exists():
            print(f"File not found: {input_file}")
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed = []
        
        for item in data:
            try:
                question = self.clean_text(item.get("question", ""))
                answer = self.clean_text(item.get("answer", ""))
                context = self.clean_text(item.get("context", ""))
                
                if question and answer:
                    processed.append({
                        "source": "MedQuAD",
                        "question": question,
                        "answer": answer,
                        "context": context
                    })
            except Exception as e:
                print(f"Error processing MedQuAD item: {e}")
                continue
        
        print(f"Processed {len(processed)} MedQuAD Q&A pairs")
        return processed
    
    def process_huggingface_data(self) -> List[Dict]:
        """Process HuggingFace medical datasets"""
        print("Processing HuggingFace datasets...")
        processed = []
        
        # Process all HF datasets
        for file in self.raw_data_dir.glob("hf_*.json"):
            print(f"Processing {file.name}...")
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    # Generic processing - adapt based on dataset structure
                    if isinstance(item, dict):
                        # Try to extract question/answer or instruction/response
                        question = item.get("question") or item.get("instruction") or item.get("input", "")
                        answer = item.get("answer") or item.get("response") or item.get("output", "")
                        
                        if question and answer:
                            processed.append({
                                "source": f"HF_{file.stem}",
                                "question": self.clean_text(str(question)),
                                "answer": self.clean_text(str(answer)),
                                "raw_data": item
                            })
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        print(f"Processed {len(processed)} HuggingFace examples")
        return processed
    
    def save_processed_data(self, data: List[Dict], filename: str):
        """Save processed data to JSON file"""
        output_file = self.processed_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data)} items to {output_file}")


def main():
    """Main processing function"""
    processor = MedicalDataProcessor()
    
    print("=" * 60)
    print("Medical Data Processing Pipeline")
    print("=" * 60)
    
    # Process all data sources
    all_processed = []
    
    fda_data = processor.process_fda_data()
    if fda_data:
        processor.save_processed_data(fda_data, "fda_processed.json")
        all_processed.extend(fda_data)
    
    pubmed_data = processor.process_pubmed_data()
    if pubmed_data:
        processor.save_processed_data(pubmed_data, "pubmed_processed.json")
        all_processed.extend(pubmed_data)
    
    medquad_data = processor.process_medquad_data()
    if medquad_data:
        processor.save_processed_data(medquad_data, "medquad_processed.json")
        all_processed.extend(medquad_data)
    
    hf_data = processor.process_huggingface_data()
    if hf_data:
        processor.save_processed_data(hf_data, "hf_processed.json")
        all_processed.extend(hf_data)
    
    # Save combined processed data
    processor.save_processed_data(all_processed, "all_processed.json")
    
    print("\n" + "=" * 60)
    print(f"Processing complete! Total processed items: {len(all_processed)}")
    print(f"Processed data saved to: {processor.processed_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()



