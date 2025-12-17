"""
Dataset Converter: Converts processed data to instruction-following JSONL format
for fine-tuning LLMs
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import random


class InstructionDatasetConverter:
    """Converts medical data to instruction-following format"""
    
    def __init__(self, processed_dir: str = "training/data/processed",
                 output_dir: str = "training/datasets"):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_instruction_example(self, data_item: Dict, template_type: str = "qa") -> Dict:
        """Create instruction-following example from data item"""
        
        if template_type == "qa":
            # Q&A format
            question = data_item.get("question", "")
            answer = data_item.get("answer", "")
            context = data_item.get("context", "")
            
            if not question or not answer:
                return None
            
            instruction = question
            if context:
                instruction = f"Context: {context}\n\nQuestion: {question}"
            
            return {
                "instruction": instruction,
                "input": "",
                "output": answer
            }
        
        elif template_type == "drug_info":
            # Drug information format
            drug_name = data_item.get("drug_name", "")
            indications = data_item.get("indications", "")
            warnings = data_item.get("warnings", "")
            dosage = data_item.get("dosage", "")
            
            if not drug_name:
                return None
            
            instruction = f"Tell me about the medication {drug_name}."
            output_parts = []
            
            if indications:
                output_parts.append(f"**Uses:** {indications}")
            if dosage:
                output_parts.append(f"**Dosage:** {dosage}")
            if warnings:
                output_parts.append(f"**Warnings:** {warnings}")
            
            if not output_parts:
                return None
            
            output = "\n\n".join(output_parts)
            output += "\n\n⚠️ **Important**: Always consult a healthcare provider before taking any medication."
            
            return {
                "instruction": instruction,
                "input": "",
                "output": output
            }
        
        elif template_type == "symptom_analysis":
            # Symptom analysis format
            abstract = data_item.get("abstract", "")
            title = data_item.get("title", "")
            
            if not abstract:
                return None
            
            # Extract symptoms/conditions from title/abstract
            instruction = f"Analyze these medical symptoms and provide guidance: {title}"
            output = f"Based on medical research: {abstract}\n\n⚠️ **Important**: This is informational only. Consult a healthcare provider for proper diagnosis."
            
            return {
                "instruction": instruction,
                "input": "",
                "output": output
            }
        
        elif template_type == "safety_refusal":
            # Safety refusal examples
            dangerous_requests = [
                "Prescribe me medication for [condition]",
                "Diagnose my symptoms: [symptoms]",
                "Tell me if I should stop taking [medication]",
                "What's the dosage for [drug] without a prescription?"
            ]
            
            safe_responses = [
                "I cannot prescribe medications. Please consult a licensed healthcare provider who can properly assess your condition and prescribe appropriate treatment.",
                "I cannot provide diagnoses. While I can provide general information about symptoms, only a qualified healthcare professional can diagnose medical conditions. Please consult a doctor.",
                "I cannot advise on stopping medications. Changes to medication should only be made under the supervision of your healthcare provider. Please consult your doctor before making any changes.",
                "I cannot provide specific dosages without proper medical evaluation. Dosage depends on many factors including age, weight, medical history, and other medications. Please consult a pharmacist or doctor."
            ]
            
            request = random.choice(dangerous_requests)
            response = random.choice(safe_responses)
            
            return {
                "instruction": request,
                "input": "",
                "output": response
            }
        
        elif template_type == "emergency_escalation":
            # Emergency escalation examples
            emergency_scenarios = [
                ("I'm having severe chest pain", "🚨 EMERGENCY: Severe chest pain requires immediate medical attention. Please call emergency services (911) immediately or go to the nearest emergency room. Do not delay."),
                ("I can't breathe properly", "🚨 EMERGENCY: Difficulty breathing is a medical emergency. Call 911 immediately or go to the nearest emergency room right away."),
                ("I think I'm having a heart attack", "🚨 EMERGENCY: If you're experiencing symptoms of a heart attack, call 911 immediately. Do not drive yourself. Emergency medical services can provide life-saving treatment en route to the hospital."),
            ]
            
            scenario, response = random.choice(emergency_scenarios)
            
            return {
                "instruction": scenario,
                "input": "",
                "output": response
            }
        
        elif template_type == "triage_recommendation":
            # Triage examples
            triage_examples = [
                ("I have a mild headache", "MONITOR: A mild headache can often be managed with rest, hydration, and over-the-counter pain relievers. If it persists for more than 24-48 hours or worsens, consult a healthcare provider."),
                ("I have a high fever (104°F) that won't go down", "URGENT CARE: A high fever that doesn't respond to treatment requires prompt medical attention. Please see a healthcare provider within 24 hours or visit an urgent care center."),
                ("I have persistent back pain for 2 weeks", "SCHEDULE APPOINTMENT: Persistent back pain lasting more than a week should be evaluated by a healthcare provider. Consider scheduling an appointment with your primary care doctor or a specialist."),
            ]
            
            scenario, response = random.choice(triage_examples)
            
            return {
                "instruction": scenario,
                "input": "",
                "output": response
            }
        
        return None
    
    def convert_to_jsonl(self, data: List[Dict], output_file: Path, 
                        template_types: List[str] = None):
        """Convert data to JSONL format"""
        if template_types is None:
            template_types = ["qa", "drug_info", "symptom_analysis"]
        
        examples = []
        
        for item in data:
            source = item.get("source", "")
            
            # Determine template type based on source
            if "FDA" in source or "drug" in source.lower():
                template = "drug_info"
            elif "PubMed" in source:
                template = "symptom_analysis"
            elif "MedQuAD" in source or "question" in item:
                template = "qa"
            else:
                template = "qa"
            
            if template in template_types:
                example = self.create_instruction_example(item, template)
                if example:
                    examples.append(example)
        
        # Add safety and emergency examples
        for _ in range(min(100, len(examples) // 10)):  # 10% safety examples
            safety_ex = self.create_instruction_example({}, "safety_refusal")
            if safety_ex:
                examples.append(safety_ex)
            
            emergency_ex = self.create_instruction_example({}, "emergency_escalation")
            if emergency_ex:
                examples.append(emergency_ex)
            
            triage_ex = self.create_instruction_example({}, "triage_recommendation")
            if triage_ex:
                examples.append(triage_ex)
        
        # Shuffle examples
        random.shuffle(examples)
        
        # Write to JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"Created {len(examples)} training examples in {output_file}")
        return examples
    
    def split_dataset(self, jsonl_file: Path, train_ratio: float = 0.9):
        """Split dataset into train and validation sets"""
        examples = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        random.shuffle(examples)
        split_idx = int(len(examples) * train_ratio)
        
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Save splits
        train_file = jsonl_file.parent / f"{jsonl_file.stem}_train.jsonl"
        val_file = jsonl_file.parent / f"{jsonl_file.stem}_val.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"Split dataset: {len(train_examples)} train, {len(val_examples)} validation")
        return train_file, val_file


def main():
    """Main conversion function"""
    converter = InstructionDatasetConverter()
    
    print("=" * 60)
    print("Dataset Conversion to Instruction Format")
    print("=" * 60)
    
    # Load processed data
    processed_file = converter.processed_dir / "all_processed.json"
    
    if not processed_file.exists():
        print(f"Error: Processed data not found at {processed_file}")
        print("Please run data_processor.py first!")
        return
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} processed items")
    
    # Convert to JSONL
    output_file = converter.output_dir / "medical_instruction_dataset.jsonl"
    examples = converter.convert_to_jsonl(data, output_file)
    
    # Split into train/val
    train_file, val_file = converter.split_dataset(output_file)
    
    print("\n" + "=" * 60)
    print("Dataset conversion complete!")
    print(f"Training file: {train_file}")
    print(f"Validation file: {val_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()



