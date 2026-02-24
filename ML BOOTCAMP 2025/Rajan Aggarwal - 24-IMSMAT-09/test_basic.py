from image_crypto_system import ImageCryptographySystem
from utils.image_utils import create_test_image, save_image
import cv2
import os

def test_basic_functionality():
    """Test basic system functionality"""
    print("=== Basic Functionality Test ===")
    
    # Create test image
    test_image = create_test_image()
    test_image_path = "test_images/basic_test.jpg"
    os.makedirs("test_images", exist_ok=True)
    save_image(test_image, test_image_path)
    print(f"Created test image: {test_image_path}")
    
    # Initialize system
    crypto_system = ImageCryptographySystem(password="test_password_123")
    print("Initialized crypto system")
    
    # Test content analysis
    content_analysis = crypto_system.analyze_image_content(test_image)
    print(f"Content analysis: {content_analysis}")
    
    # Test object detection
    inappropriate_regions = crypto_system.detect_inappropriate_regions(test_image)
    print(f"Detected regions: {len(inappropriate_regions)}")
    
    # Test blur functionality
    if len(inappropriate_regions) > 0:
        region = inappropriate_regions[0]
        x1, y1, x2, y2 = region[:4]
        blurred_image = crypto_system.apply_advanced_blur(test_image, (x1, y1, x2, y2), 0.8)
        save_image(blurred_image, "output/blurred_test.jpg")
        print("Applied blur test successful")
    
    print("Basic functionality test completed!")

if __name__ == "__main__":
    test_basic_functionality()