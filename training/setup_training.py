"""
Setup script to create training directory structure
"""
from pathlib import Path


def setup_directories():
    """Create necessary directories for training"""
    directories = [
        "training/data/raw",
        "training/data/processed",
        "training/datasets",
        "training/models",
        "training/logs",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created: {dir_path}")
    
    print("\n[OK] Training directory structure ready!")


if __name__ == "__main__":
    print("Setting up training directories...")
    setup_directories()
    print("\nSetup complete! You can now run the training pipeline.")

