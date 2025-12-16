"""
Model Evaluation Script
Evaluates trained medical AI model on test set
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import List, Dict


class ModelEvaluator:
    """Evaluates medical AI model"""
    
    def __init__(self, model_path: str, base_model: str = "mistralai/Mistral-7B-v0.1"):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base, str(self.model_path))
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt for model"""
        if "mistral" in self.base_model.lower():
            prompt = f"<s>[INST] {instruction}\n{input_text} [/INST]"
        elif "llama" in self.base_model.lower():
            prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction}\n{input_text} [/INST]"
        elif "qwen" in self.base_model.lower():
            prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        
        return prompt
    
    def generate_response(self, instruction: str, input_text: str = "", max_length: int = 512) -> str:
        """Generate response from model"""
        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        elif "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        elif "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def evaluate_on_dataset(self, test_file: str, num_samples: int = 50):
        """Evaluate model on test dataset"""
        print(f"Evaluating on {test_file}...")
        
        # Load test data
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                if line.strip():
                    test_data.append(json.loads(line))
        
        results = []
        
        for i, item in enumerate(test_data):
            print(f"Evaluating {i+1}/{len(test_data)}...")
            
            instruction = item.get("instruction", "")
            expected = item.get("output", "")
            
            generated = self.generate_response(instruction)
            
            results.append({
                "instruction": instruction,
                "expected": expected,
                "generated": generated,
                "match": expected.lower() in generated.lower() or generated.lower() in expected.lower()
            })
        
        # Calculate metrics
        accuracy = sum(r["match"] for r in results) / len(results) if results else 0
        
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Total samples: {len(results)}")
        
        # Save results
        results_file = Path(test_file).parent / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "accuracy": accuracy,
                "total_samples": len(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {results_file}")
        
        return results


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Medical AI Model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Base model name")
    parser.add_argument("--test-file", type=str,
                       default="training/datasets/medical_instruction_dataset_val.jsonl",
                       help="Test dataset file")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model_path, args.base_model)
    evaluator.load_model()
    evaluator.evaluate_on_dataset(args.test_file, args.num_samples)


if __name__ == "__main__":
    main()


