"""
Data Collection Script for Medical AI Training
Collects data from FDA, PubMed, NHS, CDC, and HuggingFace datasets
"""
import os
import json
import requests
import time
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET
from datasets import load_dataset
import pandas as pd


class MedicalDataCollector:
    """Collects medical data from various sources"""
    
    def __init__(self, output_dir: str = "training/data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_fda_drug_labels(self, limit: int = 1000):
        """Collect FDA drug label data"""
        print("Collecting FDA drug labels...")
        url = "https://api.fda.gov/drug/label.json"
        all_data = []
        
        # Collect in batches
        skip = 0
        batch_size = 100
        
        while len(all_data) < limit:
            params = {
                "limit": min(batch_size, limit - len(all_data)),
                "skip": skip
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    if not results:
                        break
                    all_data.extend(results)
                    skip += len(results)
                    print(f"Collected {len(all_data)} FDA drug labels...")
                    time.sleep(0.5)  # Rate limiting
                else:
                    print(f"Error: {response.status_code}")
                    break
            except Exception as e:
                print(f"Error collecting FDA data: {e}")
                break
        
        # Save to file
        output_file = self.output_dir / "fda_drug_labels.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(all_data)} FDA drug labels to {output_file}")
        return all_data
    
    def collect_pubmed_abstracts(self, query: str = "medical treatment", max_results: int = 1000):
        """Collect PubMed abstracts"""
        print(f"Collecting PubMed abstracts for query: {query}...")
        
        # Search for articles
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 10000),
            "retmode": "json"
        }
        
        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])
                
                # Fetch abstracts
                abstracts = []
                batch_size = 100
                
                for i in range(0, len(pmids), batch_size):
                    batch_pmids = pmids[i:i+batch_size]
                    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(batch_pmids),
                        "retmode": "xml"
                    }
                    
                    try:
                        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=30)
                        if fetch_response.status_code == 200:
                            # Parse XML (simplified - in production use proper parser)
                            root = ET.fromstring(fetch_response.text)
                            for article in root.findall(".//PubmedArticle"):
                                pmid = article.find(".//PMID")
                                title = article.find(".//ArticleTitle")
                                abstract = article.find(".//AbstractText")
                                
                                if title is not None and abstract is not None:
                                    abstracts.append({
                                        "pmid": pmid.text if pmid is not None else "",
                                        "title": title.text if title is not None else "",
                                        "abstract": abstract.text if abstract is not None else ""
                                    })
                        
                        time.sleep(0.3)  # Rate limiting
                    except Exception as e:
                        print(f"Error fetching batch: {e}")
                        continue
                    
                    print(f"Collected {len(abstracts)} PubMed abstracts...")
                
                # Save to file
                output_file = self.output_dir / "pubmed_abstracts.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(abstracts, f, indent=2, ensure_ascii=False)
                
                print(f"Saved {len(abstracts)} PubMed abstracts to {output_file}")
                return abstracts
        except Exception as e:
            print(f"Error collecting PubMed data: {e}")
            return []
    
    def collect_huggingface_datasets(self):
        """Collect medical datasets from HuggingFace"""
        print("Collecting HuggingFace medical datasets...")
        
        datasets_to_load = [
            "openlifescienceai/medmcqa",  # Medical MCQ dataset
            "medalpaca/medical_meadow_medical_flashcards",  # Medical flashcards
            "medalpaca/medical_meadow_wikidoc",  # WikiDoc medical content
        ]
        
        all_data = {}
        
        for dataset_name in datasets_to_load:
            try:
                print(f"Loading {dataset_name}...")
                dataset = load_dataset(dataset_name, split="train")
                
                # Convert to list
                data_list = list(dataset)
                
                # Save to file
                safe_name = dataset_name.replace("/", "_")
                output_file = self.output_dir / f"hf_{safe_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent=2, ensure_ascii=False)
                
                all_data[dataset_name] = data_list
                print(f"Saved {len(data_list)} examples from {dataset_name}")
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
        
        return all_data
    
    def collect_medquad(self):
        """Collect MedQuAD dataset from HuggingFace"""
        print("Collecting MedQuAD dataset...")
        try:
            dataset = load_dataset("abachaa/MedQuAD", split="train")
            data_list = list(dataset)
            
            output_file = self.output_dir / "medquad.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(data_list)} MedQuAD examples to {output_file}")
            return data_list
        except Exception as e:
            print(f"Error loading MedQuAD: {e}")
            return []


def main():
    """Main collection function"""
    collector = MedicalDataCollector()
    
    print("=" * 60)
    print("Medical Data Collection Pipeline")
    print("=" * 60)
    print("\nNOTE: This will AUTOMATICALLY download all datasets.")
    print("No manual downloads needed! Everything is automated.\n")
    
    # Collect from different sources
    print("1. Collecting FDA drug labels (from API)...")
    collector.collect_fda_drug_labels(limit=500)  # Start with 500
    
    print("\n2. Collecting PubMed abstracts (from API)...")
    collector.collect_pubmed_abstracts(query="medical treatment", max_results=500)
    
    print("\n3. Collecting HuggingFace datasets (automatic download)...")
    collector.collect_huggingface_datasets()
    
    print("\n4. Collecting MedQuAD (automatic download)...")
    collector.collect_medquad()
    
    print("\n" + "=" * 60)
    print("✅ Data collection complete!")
    print(f"📁 Raw data saved to: {collector.output_dir}")
    print("\nAll datasets downloaded automatically - no manual work needed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


