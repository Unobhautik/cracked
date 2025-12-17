"""
Ultra-Minimal Training for 2GB GPU
Only use if you want to try GPU training (may not work)
"""
import os
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import sys

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
    sys.exit(1)


@dataclass
class UltraMinimalConfig:
    """Config for 2GB GPU - may not work"""
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    use_qlora: bool = True
    output_dir: str = "training/models/medical_ai_model"
    dataset_path: str = "training/datasets/medical_instruction_dataset_train.jsonl"
    val_dataset_path: str = "training/datasets/medical_instruction_dataset_val.jsonl"
    
    # Ultra-minimal LoRA
    lora_r: int = 2  # Absolute minimum
    lora_alpha: int = 4
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj"])  # Only 1 module!
    
    # Ultra-minimal training
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 128  # Very large
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    max_seq_length: int = 256  # Very short
    save_steps: int = 10000
    eval_steps: int = 10000
    logging_steps: int = 500
    max_samples: int = 1000  # Only 1000 samples


class UltraMinimalTrainer:
    """Trainer for 2GB GPU - may fail"""
    
    def __init__(self, config: UltraMinimalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def check_gpu(self):
        """Check if GPU is available and has enough memory"""
        if not torch.cuda.is_available():
            print("❌ No GPU detected. Use train_model_fast.py for CPU training.")
            sys.exit(1)
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory < 2.5:
            print("⚠️  WARNING: GPU has less than 2.5GB VRAM")
            print("   This training may fail due to out of memory errors.")
            print("   Recommendation: Use CPU training instead (train_model_fast.py)")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(0)
    
    def load_model_and_tokenizer(self):
        """Load with maximum memory efficiency"""
        print(f"\nLoading model: {self.config.base_model}")
        print("⚠️  Ultra-minimal mode for 2GB GPU")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 4-bit QLoRA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load with CPU offloading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                max_memory={0: "1.5GiB", "cpu": "10GiB"},  # Limit GPU usage
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.gradient_checkpointing_enable()
            
            # Ultra-minimal LoRA
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
            
            print("✓ Model loaded (ultra-minimal mode)")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n❌ Out of memory! Your GPU is too small.")
                print("   Please use CPU training instead:")
                print("   python training/train_model_fast.py")
                sys.exit(1)
            raise
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format prompt"""
        if "phi" in self.config.base_model.lower():
            if input_text:
                prompt = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n{output}<|end|>"
            else:
                prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
        else:
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt
    
    def preprocess_function(self, examples):
        """Preprocess with padding"""
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
            padding="max_length",
        )
        
        labels = model_inputs["input_ids"].copy()
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def load_dataset(self):
        """Load minimal dataset"""
        print("\nLoading dataset (ultra-minimal: 1000 samples)...")
        
        def load_jsonl(file_path, max_samples=None):
            data = {"instruction": [], "input": [], "output": []}
            count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and (max_samples is None or count < max_samples):
                        try:
                            item = json.loads(line)
                            data["instruction"].append(item.get("instruction", ""))
                            data["input"].append(item.get("input", ""))
                            data["output"].append(item.get("output", ""))
                            count += 1
                        except:
                            continue
            return data
        
        train_data = load_jsonl(self.config.dataset_path, max_samples=self.config.max_samples)
        val_data = load_jsonl(self.config.val_dataset_path, max_samples=100)
        
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=50,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=50,
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset
    
    def train(self):
        """Train with ultra-minimal settings"""
        print("\n" + "="*60)
        print("ULTRA-MINIMAL TRAINING (2GB GPU)")
        print("="*60)
        print("⚠️  WARNING: This may fail due to low GPU memory!")
        print("="*60)
        
        self.check_gpu()
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
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            optim="paged_adamw_8bit",
            report_to="tensorboard",
            run_name="medical_ai_ultra_minimal",
            save_total_limit=1,
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
        
        print("\nStarting training (may fail due to low GPU memory)...")
        
        try:
            trainer.train()
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            print("✓ Training complete!")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n❌ Out of memory error!")
                print("   Your GPU (2GB) is too small for this training.")
                print("   Please use CPU training instead:")
                print("   python training/train_model_fast.py")
            raise


if __name__ == "__main__":
    config = UltraMinimalConfig()
    trainer = UltraMinimalTrainer(config)
    trainer.train()


