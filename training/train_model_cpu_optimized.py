"""
CPU-Optimized Model Training Script
Ultra-efficient for systems with limited resources (CPU or low GPU memory)
Uses aggressive optimizations and smaller model if needed
"""
import os
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import sys

# Check dependencies
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install: pip install -r training/requirements_training.txt")
    sys.exit(1)


@dataclass
class CPUOptimizedConfig:
    """Ultra-optimized configuration for low-resource systems"""
    # Use smaller model for CPU/low memory
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"  # Much smaller, faster
    use_qlora: bool = True
    output_dir: str = "training/models/medical_ai_model"
    dataset_path: str = "training/datasets/medical_instruction_dataset_train.jsonl"
    val_dataset_path: str = "training/datasets/medical_instruction_dataset_val.jsonl"
    
    # Minimal LoRA config
    lora_r: int = 8  # Very small rank
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])  # Only 2 modules
    
    # Minimal training config
    num_epochs: int = 2  # Fewer epochs
    batch_size: int = 1  # Minimum batch
    gradient_accumulation_steps: int = 32  # Large accumulation
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    max_seq_length: int = 512  # Shorter sequences
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100


class CPUOptimizedTrainer:
    """Ultra-efficient trainer for low-resource systems"""
    
    def __init__(self, config: CPUOptimizedConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def check_requirements(self):
        """Check if all requirements are met"""
        print("Checking requirements...")
        
        if not torch.cuda.is_available():
            print("⚠️  Running on CPU - this will be VERY slow (10-20 hours)")
            print("   Consider using a cloud GPU service or smaller dataset")
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU detected: {gpu_memory:.1f}GB")
            
            if gpu_memory < 6:
                print("⚠️  Very low GPU memory. Using aggressive optimizations.")
        
        # Check dataset files
        train_path = Path(self.config.dataset_path)
        val_path = Path(self.config.val_dataset_path)
        
        if not train_path.exists():
            print(f"❌ Training dataset not found: {train_path}")
            sys.exit(1)
        
        if not val_path.exists():
            print(f"⚠️  Validation dataset not found: {val_path}")
            self._create_val_split(train_path, val_path)
        
        print("✓ All requirements met!")
        return True
    
    def _create_val_split(self, train_path: Path, val_path: Path):
        """Create validation split if missing"""
        examples = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        val_size = max(10, len(examples) // 10)
        val_examples = examples[:val_size]
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"Created validation set with {len(val_examples)} examples")
    
    def load_model_and_tokenizer(self):
        """Load model with maximum efficiency"""
        print(f"\nLoading model: {self.config.base_model}")
        print("Using smaller model optimized for low-resource systems...")
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print("✓ Tokenizer loaded")
            
            # Configure quantization
            if self.config.use_qlora and torch.cuda.is_available():
                print("Configuring 4-bit QLoRA...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
                print("⚠️  Using full precision (CPU mode)")
            
            # Load model
            print("Loading base model...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = "cpu"
                model_kwargs["low_cpu_mem_usage"] = True
            
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **model_kwargs
            )
            
            print("✓ Base model loaded")
            
            # Prepare for QLoRA training
            if self.config.use_qlora and bnb_config:
                print("Preparing model for QLoRA training...")
                self.model = prepare_model_for_kbit_training(self.model)
                print("✓ Model prepared for QLoRA")
            
            # Enable gradient checkpointing
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            
            # Apply minimal LoRA
            print("Applying minimal LoRA adapters...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            print("✓ LoRA adapters applied")
            print("\n" + "="*60)
            print("Model loaded successfully!")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            raise
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format instruction following prompt"""
        if "phi" in self.config.base_model.lower():
            # Phi-3 format
            if input_text:
                prompt = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n{output}<|end|>"
            else:
                prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
        elif "mistral" in self.config.base_model.lower():
            if input_text:
                prompt = f"<s>[INST] {instruction}\n{input_text} [/INST] {output}</s>"
            else:
                prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
        else:
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
    
    def preprocess_function(self, examples):
        """Preprocess dataset for training with proper padding"""
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        prompts = [
            self.format_prompt(inst, inp, out)
            for inst, inp, out in zip(instructions, inputs, outputs)
        ]
        
        # Tokenize with padding to max_length for consistent batching
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",  # Critical: pad to max_length
        )
        
        # Create labels (copy of input_ids)
        labels = model_inputs["input_ids"].copy()
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def load_dataset(self):
        """Load and preprocess dataset"""
        print("\nLoading dataset...")
        
        def load_jsonl(file_path):
            data = {"instruction": [], "input": [], "output": []}
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                item = json.loads(line)
                                data["instruction"].append(item.get("instruction", ""))
                                data["input"].append(item.get("input", ""))
                                data["output"].append(item.get("output", ""))
                            except json.JSONDecodeError as e:
                                print(f"⚠️  Skipping invalid JSON on line {line_num}: {e}")
                                continue
                return data
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                raise
        
        train_data = load_jsonl(self.config.dataset_path)
        val_data = load_jsonl(self.config.val_dataset_path)
        
        if len(train_data["instruction"]) == 0:
            raise ValueError("Training dataset is empty!")
        
        # Limit dataset size for CPU training (use subset)
        if not torch.cuda.is_available():
            print("⚠️  CPU mode: Using subset of data for faster training")
            max_samples = 5000  # Limit to 5000 samples for CPU
            for key in train_data:
                train_data[key] = train_data[key][:max_samples]
            for key in val_data:
                val_data[key] = val_data[key][:min(500, len(val_data[key]))]
        
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        print(f"✓ Loaded {len(train_dataset)} training examples")
        print(f"✓ Loaded {len(val_dataset)} validation examples")
        
        print("Preprocessing datasets (this may take a while)...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=100,  # Smaller batches for preprocessing
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=100,
            remove_columns=val_dataset.column_names
        )
        
        print("✓ Datasets preprocessed")
        return train_dataset, val_dataset
    
    def train(self):
        """Train the model"""
        print("\n" + "="*60)
        print("Starting CPU-Optimized Medical AI Model Training")
        print("="*60)
        
        self.check_requirements()
        self.load_model_and_tokenizer()
        train_dataset, val_dataset = self.load_dataset()
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Only use fp16 on GPU
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # No workers for stability
            optim="paged_adamw_8bit" if self.config.use_qlora and torch.cuda.is_available() else "adamw_torch",
            report_to="tensorboard",
            run_name="medical_ai_training_cpu_optimized",
            save_total_limit=2,  # Keep only 2 checkpoints
            ddp_find_unused_parameters=False,
        )
        
        # Data collator with proper padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60)
        if not torch.cuda.is_available():
            print("⚠️  WARNING: Training on CPU will be VERY slow!")
            print("   Estimated time: 10-20 hours")
            print("   Consider using a smaller dataset or cloud GPU")
        print(f"Training for {self.config.num_epochs} epochs")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print("="*60 + "\n")
        
        try:
            trainer.train()
            
            print("\n" + "="*60)
            print("Training complete!")
            print("="*60)
            
            print("\nSaving final model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            print(f"\n✓ Model saved to: {output_dir}")
            print("\nNext steps:")
            print("1. Test your model: python training/integrate_model.py")
            print("2. Integrate with agentic AI: Update config.py")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            raise


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Medical AI Model (CPU Optimized)")
    parser.add_argument("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                       help="Base model (default: Phi-3-mini for low resources)")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    config = CPUOptimizedConfig(
        base_model=args.model,
        num_epochs=args.epochs,
        max_seq_length=args.seq_length,
    )
    
    trainer = CPUOptimizedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()


