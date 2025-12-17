"""
Efficient Model Training Script for Personal PC
Optimized for limited GPU memory with comprehensive error handling
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
class EfficientTrainingConfig:
    """Optimized configuration for personal PC training"""
    base_model: str = "mistralai/Mistral-7B-v0.1"
    use_qlora: bool = True  # Use 4-bit QLoRA for maximum efficiency
    output_dir: str = "training/models/medical_ai_model"
    dataset_path: str = "training/datasets/medical_instruction_dataset_train.jsonl"
    val_dataset_path: str = "training/datasets/medical_instruction_dataset_val.jsonl"
    
    # LoRA config (balanced for quality and efficiency)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training config (optimized for personal PC)
    num_epochs: int = 3
    batch_size: int = 2  # Small batch for limited memory
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 1024  # Balanced length
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50


class EfficientMedicalModelTrainer:
    """Efficient trainer with comprehensive error handling"""
    
    def __init__(self, config: EfficientTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def check_requirements(self):
        """Check if all requirements are met"""
        print("Checking requirements...")
        
        # Check GPU
        if not torch.cuda.is_available():
            print("⚠️  WARNING: No GPU detected. Training will be very slow on CPU.")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Training cancelled.")
                sys.exit(0)
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            
            if gpu_memory < 8:
                print("⚠️  WARNING: GPU memory is very low. Training may fail.")
                print("Consider using train_model_ultra_low_memory.py instead.")
        
        # Check dataset files
        train_path = Path(self.config.dataset_path)
        val_path = Path(self.config.val_dataset_path)
        
        if not train_path.exists():
            print(f"❌ Training dataset not found: {train_path}")
            print("Please run data collection and processing first!")
            sys.exit(1)
        
        if not val_path.exists():
            print(f"⚠️  Validation dataset not found: {val_path}")
            print("Creating validation split from training data...")
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
        
        # Take 10% for validation
        val_size = max(10, len(examples) // 10)
        val_examples = examples[:val_size]
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"Created validation set with {len(val_examples)} examples")
    
    def load_model_and_tokenizer(self):
        """Load model with maximum efficiency"""
        print(f"\nLoading model: {self.config.base_model}")
        print("This may take a few minutes...")
        
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
            
            # Configure quantization for QLoRA
            if self.config.use_qlora:
                print("Configuring 4-bit QLoRA...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            # Load model
            print("Loading base model (this may take a while)...")
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": torch.float16,
            }
            
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **model_kwargs
            )
            
            print("✓ Base model loaded")
            
            # Prepare for QLoRA training
            if self.config.use_qlora:
                print("Preparing model for QLoRA training...")
                self.model = prepare_model_for_kbit_training(self.model)
                print("✓ Model prepared for QLoRA")
            
            # Enable gradient checkpointing for memory savings
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            
            # Apply LoRA
            print("Applying LoRA adapters...")
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
            print("\nTroubleshooting:")
            print("1. Check internet connection (needs to download model)")
            print("2. Verify HuggingFace access to the model")
            print("3. Check available disk space")
            print("4. Try a smaller model or use CPU offloading")
            raise
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format instruction following prompt"""
        if "mistral" in self.config.base_model.lower():
            if input_text:
                prompt = f"<s>[INST] {instruction}\n{input_text} [/INST] {output}</s>"
            else:
                prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
        elif "llama" in self.config.base_model.lower():
            if input_text:
                prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction}\n{input_text} [/INST] {output}</s>"
            else:
                prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction} [/INST] {output}</s>"
        elif "qwen" in self.config.base_model.lower():
            if input_text:
                prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:
                prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
    
    def preprocess_function(self, examples):
        """Preprocess dataset for training"""
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        prompts = [
            self.format_prompt(inst, inp, out)
            for inst, inp, out in zip(instructions, inputs, outputs)
        ]
        
        # Tokenize with padding - data collator will handle batching
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",  # Pad to max_length for consistent batching
        )
        
        # Create labels (copy of input_ids for causal LM)
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
        
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        print(f"✓ Loaded {len(train_dataset)} training examples")
        print(f"✓ Loaded {len(val_dataset)} validation examples")
        
        print("Preprocessing datasets...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        print("✓ Datasets preprocessed")
        return train_dataset, val_dataset
    
    def train(self):
        """Train the model"""
        print("\n" + "="*60)
        print("Starting Efficient Medical AI Model Training")
        print("="*60)
        
        # Check requirements
        self.check_requirements()
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Load dataset
        train_dataset, val_dataset = self.load_dataset()
        
        # Setup training arguments
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
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            optim="paged_adamw_8bit" if self.config.use_qlora else "adamw_torch",
            report_to="tensorboard",
            run_name="medical_ai_training_efficient",
            save_total_limit=3,  # Keep only last 3 checkpoints
        )
        
        # Use data collator with proper padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
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
        print(f"Training for {self.config.num_epochs} epochs")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Total steps: ~{len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps) * self.config.num_epochs}")
        print("\nMonitor progress with: tensorboard --logdir training/models/medical_ai_model/runs")
        print("="*60 + "\n")
        
        try:
            trainer.train()
            
            print("\n" + "="*60)
            print("Training complete!")
            print("="*60)
            
            # Save final model
            print("\nSaving final model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            print(f"\n✓ Model saved to: {output_dir}")
            print("\nNext steps:")
            print("1. Test your model: python training/integrate_model.py")
            print("2. Integrate with agentic AI: Update config.py to use custom model")
            print("="*60)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n❌ Out of memory error!")
                print("\nSolutions:")
                print("1. Reduce batch_size in config (try 1)")
                print("2. Reduce max_seq_length (try 512)")
                print("3. Use train_model_ultra_low_memory.py instead")
                print("4. Close other applications using GPU")
            else:
                print(f"\n❌ Training error: {e}")
            raise
        except Exception as e:
            print(f"\n❌ Unexpected error during training: {e}")
            raise


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Medical AI Model (Efficient)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--seq-length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--no-qlora", action="store_true",
                       help="Disable QLoRA (uses more memory)")
    
    args = parser.parse_args()
    
    config = EfficientTrainingConfig(
        base_model=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.seq_length,
        use_qlora=not args.no_qlora,
    )
    
    trainer = EfficientMedicalModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

