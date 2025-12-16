"""
Model Integration Script
Shows how to integrate your trained model into the medical AI system
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path


class CustomMedicalModel:
    """Wrapper for custom medical model to replace OpenAI API"""
    
    def __init__(self, model_path: str, base_model: str = "mistralai/Mistral-7B-v0.1"):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading custom medical model from {self.model_path}...")
        
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
        
        print("Custom model loaded successfully!")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt based on model type"""
        if "mistral" in self.base_model.lower():
            prompt = f"<s>[INST] {instruction}\n{input_text} [/INST]"
        elif "llama" in self.base_model.lower():
            prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction}\n{input_text} [/INST]"
        elif "qwen" in self.base_model.lower():
            prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        
        return prompt
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response (OpenAI-like interface)"""
        formatted_prompt = self.format_prompt(prompt)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
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


# Example usage
if __name__ == "__main__":
    # Initialize your custom model
    custom_model = CustomMedicalModel(
        model_path="training/models/medical_ai_model",
        base_model="mistralai/Mistral-7B-v0.1"
    )
    
    # Test the model
    test_prompts = [
        "What are the side effects of ibuprofen?",
        "I have a fever and headache. What should I do?",
        "Tell me about diabetes management."
    ]
    
    print("\n" + "=" * 60)
    print("Testing Custom Medical Model")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = custom_model.generate(prompt)
        print(f"Response: {response}")
        print("-" * 60)
    
    print("\nTo integrate into your system:")
    print("1. Replace OpenAI API calls with CustomMedicalModel")
    print("2. Update config.py to use custom model")
    print("3. Test with your agents")


