"""
Complete Training Pipeline Runner
Runs the entire pipeline: collection -> processing -> conversion -> training
"""
import subprocess
import sys
from pathlib import Path


def run_step(script_name: str, description: str):
    """Run a training pipeline step"""
    print("\n" + "=" * 60)
    print(f"Step: {description}")
    print("=" * 60)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Error: {script_path} not found!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False


def main():
    """Run complete training pipeline"""
    print("=" * 60)
    print("Medical AI Model Training Pipeline")
    print("=" * 60)
    print("\nThis will:")
    print("1. Collect data from FDA, PubMed, HuggingFace")
    print("2. Process and clean the data")
    print("3. Convert to instruction-following format")
    print("4. Train the model with LoRA/QLoRA")
    print("\nThis may take several hours depending on your hardware.")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    steps = [
        ("data_collector.py", "Data Collection"),
        ("data_processor.py", "Data Processing"),
        ("dataset_converter.py", "Dataset Conversion"),
    ]
    
    # Run collection, processing, conversion
    for script, description in steps:
        if not run_step(script, description):
            print(f"\nPipeline stopped at: {description}")
            print("Please fix errors and rerun.")
            return
    
    # Training step (optional - user can run separately)
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print("\nTo train the model, run:")
    print("  python training/train_model.py --model mistralai/Mistral-7B-v0.1 --use-qlora")
    print("\nOr for Llama 3:")
    print("  python training/train_model.py --model meta-llama/Llama-3-8b --use-qlora")
    print("\nOr for Qwen:")
    print("  python training/train_model.py --model Qwen/Qwen-7B --use-qlora")
    print("\nNote: Training requires a GPU with at least 16GB VRAM for QLoRA.")


if __name__ == "__main__":
    main()


