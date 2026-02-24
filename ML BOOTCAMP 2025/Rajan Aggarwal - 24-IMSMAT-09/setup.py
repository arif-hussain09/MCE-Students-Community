import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_directories():
    """Create necessary directories"""
    directories = [
        "models", "data", "output", "test_images", 
        "trained_models", "utils", "data/train", "data/val"
    ]
    
    for category in ['violence', 'abuse', 'adult_content', 'scary', 'normal']:
        directories.extend([
            f"data/train/{category}",
            f"data/val/{category}"
        ])
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    print("=== Image Cryptography System Setup ===")
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    print("\n✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add training images to data/train/ and data/val/ directories")
    print("2. Run: python test_basic.py")
    print("3. Run: python test_full_processing.py")
    print("4. (Optional) Train custom model: python train_model.py")

if __name__ == "__main__":
    main()