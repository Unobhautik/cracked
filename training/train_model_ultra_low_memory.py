"""
Ultra Low-Memory Training Script
For systems with very limited GPU RAM (4GB or less)
Uses smaller model or CPU training
"""
import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset


class UltraLowMemoryTrainer:
    """Trainer for systems with very limited GPU memory"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B", output_dir: str = "training/models/medical_ai_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
    
    def load_model_and_tokenizer(self):
        """Load smaller model that fits in low memory"""
        print(f"Loading smaller model: {self.model_name}")
        print("[WARNING] Using smaller model for ultra-low memory systems")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model without quantization first (smaller models don't need it)
        # Use CPU if GPU is too small
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < 6:
                    print(f"[WARNING] Very low GPU memory ({gpu_memory:.1f}GB). Using CPU training.")
                    device_map = "cpu"
                else:
                    device_map = "auto"
            except:
                device_map = "cpu"
        else:
            device_map = "cpu"
            print("[WARNING] No GPU detected. Training on CPU (will be very slow).")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        )
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=4,  # Very small rank
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("[SUCCESS] Model loaded!")
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """Format prompt"""
        if "qwen" in self.model_name.lower():
            prompt = f"<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        elif "mistral" in self.model_name.lower():
            prompt = f"<s>[INST] {instruction}\n{input_text} [/INST] {output}</s>"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        return prompt
    
    def preprocess_function(self, examples):
        """Preprocess dataset"""
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        
        prompts = [
            self.format_prompt(inst, inp, out)
            for inst, inp, out in zip(instructions, inputs, outputs)
        ]
        
        model_inputs = self.tokenizer(
            prompts,
            max_length=256,  # Very short for ultra-low memory
            truncation=True,
            padding=False,
        )
        
        labels = model_inputs["input_ids"].copy()
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def load_dataset(self):
        """Load dataset"""
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
        
        train_data = load_jsonl("training/datasets/medical_instruction_dataset_train.jsonl")
        val_data = load_jsonl("training/datasets/medical_instruction_dataset_val.jsonl")
        
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
        return train_dataset, val_dataset
    
    def train(self):
        """Train model"""
        print("=" * 60)
        print("Ultra Low-Memory Training")
        print("=" * 60)
        
        self.load_model_and_tokenizer()
        train_dataset, val_dataset = self.load_dataset()
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            learning_rate=5e-5,
            warmup_steps=20,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            optim="adamw_torch",
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
        
        print("\nStarting training...")
        trainer.train()
        
        print("\nSaving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("\n[SUCCESS] Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-1.5B",
                       help="Smaller model (Qwen2-1.5B, Qwen2-0.5B, etc.)")
    args = parser.parse_args()
    
    trainer = UltraLowMemoryTrainer(model_name=args.model)
    trainer.train()

