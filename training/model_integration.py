"""
Seamless Model Integration for Agentic AI System
Allows switching between OpenAI API and custom trained model
"""
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Try to import transformers (optional - only needed for custom model)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MedicalModelWrapper:
    """
    Unified wrapper that works with both OpenAI API and custom trained models
    Seamlessly integrates with existing agentic AI system
    """
    
    def __init__(self, use_custom_model: bool = False, 
                 model_path: Optional[str] = None,
                 base_model: Optional[str] = None):
        """
        Initialize model wrapper
        
        Args:
            use_custom_model: If True, use custom trained model; if False, use OpenAI API
            model_path: Path to trained model (required if use_custom_model=True)
            base_model: Base model name (e.g., "mistralai/Mistral-7B-v0.1")
        """
        self.use_custom_model = use_custom_model
        self.model = None
        self.tokenizer = None
        self.openai_client = None
        
        if use_custom_model:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Transformers not available. Install with: "
                    "pip install transformers peft torch"
                )
            if not model_path:
                raise ValueError("model_path required when use_custom_model=True")
            self._load_custom_model(model_path, base_model)
        else:
            self._setup_openai()
    
    def _load_custom_model(self, model_path: str, base_model: Optional[str]):
        """Load custom trained model"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading custom medical model from {model_path}...")
        
        # Try to detect base model from config
        if not base_model:
            config_file = model_path / "adapter_config.json"
            if config_file.exists():
                import json
                with open(config_file) as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")
        
        if not base_model:
            base_model = "mistralai/Mistral-7B-v0.1"  # Default
            print(f"⚠️  Base model not specified, using default: {base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load LoRA weights
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base, str(model_path))
        self.model.eval()
        
        print("✓ Custom model loaded successfully!")
    
    def _setup_openai(self):
        """Setup OpenAI API client"""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.openai_client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def format_prompt(self, instruction: str, input_text: str = "", base_model: str = "") -> str:
        """Format prompt based on model type"""
        if "mistral" in base_model.lower() or (self.model and "mistral" in str(type(self.model))):
            if input_text:
                prompt = f"<s>[INST] {instruction}\n{input_text} [/INST]"
            else:
                prompt = f"<s>[INST] {instruction} [/INST]"
        elif "llama" in base_model.lower():
            if input_text:
                prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction}\n{input_text} [/INST]"
            else:
                prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction} [/INST]"
        else:
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return prompt
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.7, **kwargs) -> str:
        """
        Generate response (unified interface for both OpenAI and custom model)
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments
        
        Returns:
            Generated text response
        """
        if self.use_custom_model:
            return self._generate_custom(prompt, max_tokens, temperature)
        else:
            return self._generate_openai(prompt, max_tokens, temperature)
    
    def _generate_custom(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using custom model"""
        # Format prompt
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        elif "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        elif "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def _generate_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using OpenAI API"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content


# Global model instance
_global_model: Optional[MedicalModelWrapper] = None


def get_medical_model(use_custom: bool = False, 
                     model_path: Optional[str] = None,
                     base_model: Optional[str] = None) -> MedicalModelWrapper:
    """
    Get or create global medical model instance
    
    Args:
        use_custom: Use custom trained model instead of OpenAI
        model_path: Path to trained model (if use_custom=True)
        base_model: Base model name (if use_custom=True)
    
    Returns:
        MedicalModelWrapper instance
    """
    global _global_model
    
    # Check environment variable for default behavior
    if not use_custom:
        use_custom = os.getenv("USE_CUSTOM_MODEL", "false").lower() == "true"
        if use_custom and not model_path:
            model_path = os.getenv("CUSTOM_MODEL_PATH", "training/models/medical_ai_model")
            base_model = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-v0.1")
    
    if _global_model is None or _global_model.use_custom_model != use_custom:
        _global_model = MedicalModelWrapper(
            use_custom_model=use_custom,
            model_path=model_path,
            base_model=base_model
        )
    
    return _global_model


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Medical Model Integration Test")
    print("="*60)
    
    # Test custom model if available
    model_path = Path("training/models/medical_ai_model")
    if model_path.exists():
        print("\nTesting custom model...")
        try:
            model = get_medical_model(
                use_custom=True,
                model_path=str(model_path)
            )
            
            test_prompts = [
                "What are the side effects of ibuprofen?",
                "I have a fever and headache. What should I do?",
            ]
            
            for prompt in test_prompts:
                print(f"\nPrompt: {prompt}")
                response = model.generate(prompt, max_tokens=256)
                print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ Custom model test failed: {e}")
            print("Make sure you've completed training first!")
    else:
        print("\n⚠️  Custom model not found. Train a model first:")
        print("  python training/run_complete_training.py")
    
    # Test OpenAI (if API key available)
    if os.getenv("OPENAI_API_KEY"):
        print("\nTesting OpenAI API...")
        try:
            model = get_medical_model(use_custom=False)
            response = model.generate("What is diabetes?", max_tokens=100)
            print(f"OpenAI Response: {response[:200]}...")
        except Exception as e:
            print(f"❌ OpenAI test failed: {e}")
    else:
        print("\n⚠️  OpenAI API key not found. Skipping OpenAI test.")
    
    print("\n" + "="*60)
    print("Integration test complete!")
    print("="*60)
    print("\nTo use in your agentic AI system:")
    print("1. Set USE_CUSTOM_MODEL=true in .env file")
    print("2. Set CUSTOM_MODEL_PATH=training/models/medical_ai_model")
    print("3. Update config.py to use get_medical_model()")


