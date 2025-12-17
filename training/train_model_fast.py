"""
Fast Training Script - Completes in 6-7 hours on CPU
Ultra-aggressive optimizations for quick training
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
class FastTrainingConfig:
    """Ultra-fast configuration for 6-7 hour training"""
    # Use smallest available model
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B - smallest good model
    use_qlora: bool = True
    output_dir: str = "training/models/medical_ai_model"
    dataset_path: str = "training/datasets/medical_instruction_dataset_train.jsonl"
    val_dataset_path: str = "training/datasets/medical_instruction_dataset_val.jsonl"
    
    # Minimal LoRA
    lora_r: int = 4  # Very small rank
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Fast training config
    num_epochs: int = 1  # Single epoch for speed
    batch_size: int = 1
    gradient_accumulation_steps: int = 64  # Large accumulation
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 384  # Short sequences
    save_steps: int = 5000  # Save less frequently
    eval_steps: int = 5000
    logging_steps: int = 200
    max_samples: int = 2000  # Use only 2000 samples


class FastTrainer:
    """Ultra-fast trainer for 6-7 hour completion"""
    
    def __init__(self, config: FastTrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def check_requirements(self):
        """Check requirements"""
        print("Checking requirements...")
        
        if not torch.cuda.is_available():
            print("⚠️  Running on CPU")
            print("   Fast mode: Using minimal data and 1 epoch")
            print("   Estimated time: 6-7 hours")
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ GPU detected: {gpu_memory:.1f}GB")
            print("   Fast mode will complete in 2-3 hours on GPU")
        
        train_path = Path(self.config.dataset_path)
        if not train_path.exists():
            print(f"❌ Training dataset not found: {train_path}")
            sys.exit(1)
        
        val_path = Path(self.config.val_dataset_path)
        if not val_path.exists():
            self._create_val_split(train_path, val_path)
        
        print("✓ All requirements met!")
        return True
    
    def _create_val_split(self, train_path: Path, val_path: Path):
        """Create validation split"""
        examples = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        val_size = max(10, min(200, len(examples) // 10))
        val_examples = examples[:val_size]
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        print(f"Created validation set with {len(val_examples)} examples")
    
    def load_model_and_tokenizer(self):
        """Load model quickly"""
        print(f"\nLoading model: {self.config.base_model}")
        print("Fast mode: Using smallest model for speed...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print("✓ Tokenizer loaded")
            
            # Only use QLoRA on GPU
            if self.config.use_qlora and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
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
            
            print("Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **model_kwargs
            )
            
            print("✓ Base model loaded")
            
            if self.config.use_qlora and bnb_config:
                self.model = prepare_model_for_kbit_training(self.model)
            
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            # Minimal LoRA
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
            
            print("✓ Model ready for fast training")
            
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            raise
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format prompt"""
        if "phi" in self.config.base_model.lower():
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
        """Preprocess with proper padding"""
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        prompts = [
            self.format_prompt(inst, inp, out)
            for inst, inp, out in zip(instructions, inputs, outputs)
        ]
        
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",  # Critical for batching
        )
        
        labels = model_inputs["input_ids"].copy()
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def load_dataset(self):
        """Load minimal dataset"""
        print("\nLoading dataset (fast mode: using subset)...")
        
        def load_jsonl(file_path, max_samples=None):
            data = {"instruction": [], "input": [], "output": []}
            count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and (max_samples is None or count < max_samples):
                            try:
                                item = json.loads(line)
                                data["instruction"].append(item.get("instruction", ""))
                                data["input"].append(item.get("input", ""))
                                data["output"].append(item.get("output", ""))
                                count += 1
                            except json.JSONDecodeError:
                                continue
                return data
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                raise
        
        # Load only subset for fast training
        train_data = load_jsonl(self.config.dataset_path, max_samples=self.config.max_samples)
        val_data = load_jsonl(self.config.val_dataset_path, max_samples=200)
        
        if len(train_data["instruction"]) == 0:
            raise ValueError("Training dataset is empty!")
        
        print(f"✓ Loaded {len(train_data['instruction'])} training examples (subset for speed)")
        print(f"✓ Loaded {len(val_data['instruction'])} validation examples")
        
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        print("Preprocessing datasets...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=100,
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
        """Fast training"""
        print("\n" + "="*60)
        print("FAST TRAINING MODE - 6-7 Hours on CPU")
        print("="*60)
        print("Using:")
        print(f"  - Model: {self.config.base_model}")
        print(f"  - Training samples: {self.config.max_samples}")
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Sequence length: {self.config.max_seq_length}")
        print("="*60)
        
        self.check_requirements()
        self.load_model_and_tokenizer()
        train_dataset, val_dataset = self.load_dataset()
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate steps
        total_steps = len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps) * self.config.num_epochs
        
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
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            optim="paged_adamw_8bit" if self.config.use_qlora and torch.cuda.is_available() else "adamw_torch",
            report_to="tensorboard",
            run_name="medical_ai_fast_training",
            save_total_limit=1,  # Keep only 1 checkpoint
            ddp_find_unused_parameters=False,
            max_steps=total_steps,  # Explicit step limit
        )
        
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
        print("Starting FAST training...")
        print("="*60)
        print(f"Total steps: {total_steps}")
        if not torch.cuda.is_available():
            print("Estimated time: 6-7 hours on CPU")
        else:
            print("Estimated time: 2-3 hours on GPU")
        print("="*60 + "\n")
        
        try:
            trainer.train()
            
            print("\n" + "="*60)
            print("✅ FAST TRAINING COMPLETE!")
            print("="*60)
            
            print("\nSaving model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            print(f"\n✓ Model saved to: {output_dir}")
            print("\nNote: This is a fast-trained model with minimal data.")
            print("You can retrain with more data later for better quality.")
            print("\nNext steps:")
            print("1. Test: python training/model_integration.py")
            print("2. Use with agentic AI: Set USE_CUSTOM_MODEL=true")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            raise


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast Training - 6-7 Hours on CPU")
    parser.add_argument("--samples", type=int, default=2000,
                       help="Number of training samples (default: 2000)")
    parser.add_argument("--seq-length", type=int, default=384,
                       help="Sequence length (default: 384)")
    
    args = parser.parse_args()
    
    config = FastTrainingConfig(
        max_samples=args.samples,
        max_seq_length=args.seq_length,
    )
    
    trainer = FastTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()


