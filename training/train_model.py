"""
Fine-tuning Script for Medical AI Model
Uses LoRA/QLoRA for efficient training
Supports: Mistral 7B, Llama 3 8B, Qwen 7B
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
class ModelConfig:
    """Model configuration"""
    base_model: str = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Llama-3-8b" or "Qwen/Qwen-7B"
    use_qlora: bool = True  # Use QLoRA for 4-bit quantization
    output_dir: str = "training/models/medical_ai_model"
    dataset_path: str = "training/datasets/medical_instruction_dataset_train.jsonl"
    val_dataset_path: str = "training/datasets/medical_instruction_dataset_val.jsonl"
    
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Training config
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50


class MedicalModelTrainer:
    """Trainer for medical AI model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        print(f"Loading model: {self.config.base_model}")
        
        # Configure quantization for QLoRA
        if self.config.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.use_qlora else torch.float16
        )
        
        # Prepare for QLoRA training
        if self.config.use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
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
        
        print("Model and tokenizer loaded successfully!")
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format instruction following prompt"""
        # Mistral format
        if "mistral" in self.config.base_model.lower():
            prompt = f"<s>[INST] {instruction}\n{input_text} [/INST] {output}</s>"
        # Llama format
        elif "llama" in self.config.base_model.lower():
            prompt = f"<s>[INST] <<SYS>>\nYou are a helpful medical assistant.\n<</SYS>>\n\n{instruction}\n{input_text} [/INST] {output}</s>"
        # Qwen format
        elif "qwen" in self.config.base_model.lower():
            prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        # Default format
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        
        return prompt
    
    def preprocess_function(self, examples):
        """Preprocess dataset for training"""
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        # Format prompts
        prompts = [
            self.format_prompt(inst, inp, out)
            for inst, inp, out in zip(instructions, inputs, outputs)
        ]
        
        # Tokenize
        model_inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=False,
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = model_inputs["input_ids"].copy()
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def load_dataset(self):
        """Load and preprocess dataset"""
        print("Loading dataset...")
        
        # Load JSONL files
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
        
        # Convert to datasets format
        from datasets import Dataset
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        # Preprocess
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
        """Train the model"""
        print("=" * 60)
        print("Starting Medical AI Model Training")
        print("=" * 60)
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Load dataset
        train_dataset, val_dataset = self.load_dataset()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=not self.config.use_qlora,  # QLoRA uses bfloat16
            bf16=self.config.use_qlora,
            report_to="tensorboard",
            run_name="medical_ai_training",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save final model
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
    
    parser = argparse.ArgumentParser(description="Train Medical AI Model")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                       choices=["mistralai/Mistral-7B-v0.1", "meta-llama/Llama-3-8b", "Qwen/Qwen-7B"],
                       help="Base model to fine-tune")
    parser.add_argument("--use-qlora", action="store_true", default=True,
                       help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Create config
    config = ModelConfig(
        base_model=args.model,
        use_qlora=args.use_qlora,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Train
    trainer = MedicalModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()


