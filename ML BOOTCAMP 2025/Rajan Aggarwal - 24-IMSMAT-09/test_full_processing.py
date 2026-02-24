from image_crypto_system import ImageCryptographySystem
from utils.image_utils import create_test_image, save_image
import os
import json

def test_full_processing():
    """Test complete processing pipeline"""
    print("=== Full Processing Pipeline Test ===")
    
    # Create test image
    test_image = create_test_image()
    test_image_path = "test_images/full_test.jpg"
    os.makedirs("test_images", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    save_image(test_image, test_image_path)
    
    # Initialize system
    crypto_system = ImageCryptographySystem(password="secure_test_password")
    
    try:
        # Process image
        result = crypto_system.process_image(
            image_path=test_image_path,
            output_path="output/processed_full_test.jpg",
            encryption_enabled=True,
            blur_enabled=True
        )
        
        print("✓ Image processing completed")
        print(f"Encryption key: {result['encryption_key'][:20]}...")
        
        # Test decryption
        if os.path.exists("output/processed_full_test_metadata.json"):
            decrypted_image = crypto_system.decrypt_image(
                processed_image_path="output/processed_full_test.jpg",
                metadata_path="output/processed_full_test_metadata.json",
                encryption_key=result['encryption_key']
            )
            
            save_image(decrypted_image, "output/decrypted_full_test.jpg")
            print("✓ Image decryption completed")
        
        print("Full processing test successful!")
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_full_processing()