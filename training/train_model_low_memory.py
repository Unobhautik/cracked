"""
Low-Memory Training Script for Medical AI Model
Optimized for systems with limited GPU RAM (8GB or less)
"""
import os
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
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
from datasets import load_dataset


@dataclass
class LowMemoryModelConfig:
    """Model configuration optimized for low memory"""
    base_model: str = "mistralai/Mistral-7B-v0.1"
    use_qlora: bool = False  # Using 8-bit instead of 4-bit QLoRA
    output_dir: str = "training/models/medical_ai_model"
    dataset_path: str = "training/datasets/medical_instruction_dataset_train.jsonl"
    val_dataset_path: str = "training/datasets/medical_instruction_dataset_val.jsonl"
    
    # LoRA config (smaller for memory efficiency)
    lora_r: int = 8  # Reduced from 16
    lora_alpha: int = 16  # Reduced from 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])  # Fewer modules
    
    # Training config (ultra-low memory)
    num_epochs: int = 2  # Reduced epochs
    batch_size: int = 1  # Minimum batch size
    gradient_accumulation_steps: int = 16  # High accumulation
    learning_rate: float = 1e-4  # Lower learning rate
    warmup_steps: int = 50
    max_seq_length: int = 512  # Much shorter sequences
    save_steps: int = 1000  # Save less frequently
    eval_steps: int = 1000
    logging_steps: int = 50


class LowMemoryMedicalModelTrainer:
    """Trainer optimized for low-memory systems"""
    
    def __init__(self, config: LowMemoryModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self):
        """Load base model with maximum memory efficiency"""
        print(f"Loading model: {self.config.base_model}")
        print("[WARNING] Low-memory mode: Using aggressive optimizations")
        
        # Use 8-bit quantization instead of 4-bit (allows CPU offloading)
        # 8-bit is less memory efficient but allows CPU offloading
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Use 8-bit instead of 4-bit for CPU offloading support
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Custom device map for CPU offloading with 8-bit
        max_memory = None
        device_map = "auto"
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_memory < 12:
                # Very low GPU memory - use CPU offloading
                max_memory = {0: f"{int(gpu_memory * 0.6)}GiB", "cpu": "50GiB"}
                print(f"[WARNING] Low GPU memory detected ({gpu_memory:.1f}GB). Using CPU offloading.")
        
        # Load model with 8-bit quantization (allows CPU offloading)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for 8-bit
        )
        
        # 8-bit models don't need prepare_model_for_kbit_training
        # But we enable gradient checkpointing for memory savings
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Apply LoRA with smaller rank
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
        
        print("[SUCCESS] Model loaded with low-memory optimizations!")
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format instruction following prompt"""
        if "mistral" in self.config.base_model.lower():
            prompt = f"<s>[INST] {instruction}\n{input_text} [/INST] {output}</s>"
        elif "llama" in self.config.base_model.lower():
            prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction}\n{input_text} [/INST] {output}</s>"
        elif "qwen" in self.config.base_model.lower():
            prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        
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
        
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=False,
        )
        
        labels = model_inputs["input_ids"].copy()
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def load_dataset(self):
        """Load and preprocess dataset"""
        print("Loading dataset...")
        
        def load_jsonl(file_path):
            data = {"instruction": [], "input": [], "output": []}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data["instruction"].append(item.get("instruction", ""))
                        data["input"].append(item.get("input", ""))
                        data["output"].append(item.get("output", ""))
            return data
        
        train_data = load_jsonl(self.config.dataset_path)
        val_data = load_jsonl(self.config.val_dataset_path)
        
        from datasets import Dataset
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
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
        
        print(f"Loaded {len(train_dataset)} training examples")
        print(f"Loaded {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset
    
    def train(self):
        """Train the model with low-memory optimizations"""
        print("=" * 60)
        print("Starting Low-Memory Medical AI Model Training")
        print("=" * 60)
        print("[WARNING] Optimized for systems with limited GPU RAM")
        print("=" * 60)
        
        self.load_model_and_tokenizer()
        train_dataset, val_dataset = self.load_dataset()
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
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
            fp16=True,  # Use float16 for memory
            gradient_checkpointing=True,  # Critical!
            dataloader_pin_memory=False,
            dataloader_num_workers=0,  # Reduce workers
            optim="adamw_torch",
            report_to="tensorboard",
            run_name="medical_ai_training_low_mem",
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print("\nStarting training with low-memory optimizations...")
        print("This may be slower but will work on limited GPU memory.")
        trainer.train()
        
        print("\nSaving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Model saved to: {self.config.output_dir}")
        print("=" * 60)


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Medical AI Model (Low Memory)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                       help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    config = LowMemoryModelConfig(
        base_model=args.model,
        num_epochs=args.epochs,
        max_seq_length=args.seq_length,
    )
    
    trainer = LowMemoryMedicalModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

