"""
Complete Training Pipeline Runner
One-command solution to run entire training pipeline with error handling
"""
import subprocess
import sys
from pathlib import Path
import os


def print_step(step_num: int, description: str):
    """Print step header"""
    print("\n" + "="*70)
    print(f"STEP {step_num}: {description}")
    print("="*70)


def run_python_script(script_path: Path, description: str) -> bool:
    """Run a Python script with error handling"""
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        print(f"Running: {script_path.name}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=script_path.parent.parent
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print_step(0, "Checking Dependencies")
    
    required_packages = [
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("\nInstalling requirements...")
        try:
            requirements_file = Path(__file__).parent / "requirements_training.txt"
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )
            print("✓ Requirements installed")
        except Exception as e:
            print(f"❌ Failed to install requirements: {e}")
            print("Please run manually: pip install -r training/requirements_training.txt")
            return False
    else:
        print("✓ All dependencies installed")
    
    return True


def main():
    """Run complete training pipeline"""
    print("="*70)
    print("MEDICAL AI MODEL TRAINING PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("1. Check dependencies")
    print("2. Collect medical data (FDA, PubMed, HuggingFace)")
    print("3. Process and clean the data")
    print("4. Convert to training format")
    print("5. Train the model")
    print("\nThis may take several hours depending on your hardware.")
    print("You can stop at any time with Ctrl+C and resume later.")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Step 0: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please fix and rerun.")
        return
    
    # Step 1: Collect data
    print_step(1, "Data Collection")
    data_collector = Path(__file__).parent / "data_collector.py"
    if not run_python_script(data_collector, "Data Collection"):
        print("\n⚠️  Data collection failed. Check if you have internet connection.")
        print("You can skip this step if data already exists.")
        skip = input("Skip data collection and use existing data? (y/n): ").strip().lower()
        if skip != 'y':
            return
    
    # Step 2: Process data
    print_step(2, "Data Processing")
    data_processor = Path(__file__).parent / "data_processor.py"
    if not run_python_script(data_processor, "Data Processing"):
        print("\n❌ Data processing failed. Please check errors above.")
        return
    
    # Step 3: Convert to training format
    print_step(3, "Dataset Conversion")
    dataset_converter = Path(__file__).parent / "dataset_converter.py"
    if not run_python_script(dataset_converter, "Dataset Conversion"):
        print("\n❌ Dataset conversion failed. Please check errors above.")
        return
    
    # Step 4: Train model
    print_step(4, "Model Training")
    print("\n⚠️  This is the longest step (2-6 hours on GPU, 10-20 hours on CPU)")
    print("You can monitor progress with TensorBoard in another terminal:")
    print("  tensorboard --logdir training/models/medical_ai_model/runs")
    
    # Check if GPU is available
    import torch
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: No GPU detected!")
        print("Training options:")
        print("  1. Fast training (6-7 hours) - Minimal data, 1 epoch")
        print("  2. CPU-optimized (10-20 hours) - More data, 2 epochs")
        choice = input("\nChoose option (1/2) [default: 1]: ").strip() or "1"
        use_fast = choice == "1"
        use_cpu_optimized = choice == "2"
    else:
        use_fast = False
        use_cpu_optimized = False
    
    response = input("\nStart training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training skipped. You can run it later with:")
        if use_cpu_optimized:
            print("  python training/train_model_cpu_optimized.py")
        else:
            print("  python training/train_model_efficient.py")
        return
    
    # Choose training script based on system
    if use_fast:
        train_script = Path(__file__).parent / "train_model_fast.py"
        print("\nUsing FAST training (6-7 hours on CPU, minimal data)")
    elif use_cpu_optimized:
        train_script = Path(__file__).parent / "train_model_cpu_optimized.py"
        print("\nUsing CPU-optimized training (10-20 hours, more data)")
    else:
        train_script = Path(__file__).parent / "train_model_efficient.py"
        print("\nUsing efficient training (2-4 hours on GPU)")
    
    if not run_python_script(train_script, "Model Training"):
        print("\n❌ Training failed. Check errors above.")
        print("\nTroubleshooting:")
        print("- If out of memory: Use train_model_cpu_optimized.py")
        print("- If padding errors: Fixed in latest version")
        print("- Check GPU memory usage")
        print("- Reduce batch size or sequence length")
        return
    
    # Success!
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print("\nYour trained model is saved at:")
    print("  training/models/medical_ai_model/")
    print("\nNext steps:")
    print("1. Test your model:")
    print("   python training/integrate_model.py")
    print("\n2. Integrate with your agentic AI system:")
    print("   Update config.py to use your custom model")
    print("\n3. Use in production:")
    print("   The model is ready to use with your medical AI agents!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("You can resume from where you left off by running individual steps.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

