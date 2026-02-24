import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Tuple

def load_image(image_path: str) -> np.ndarray:
    """Load image from path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to path"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height))
    
    # Create canvas and center image
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def create_test_image() -> np.ndarray:
    """Create a test image for demonstration"""
    # Create a simple test image
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Blue
    cv2.rectangle(image, (200, 50), (300, 150), (0, 255, 0), -1)  # Green
    cv2.rectangle(image, (350, 50), (450, 150), (0, 0, 255), -1)  # Red
    
    # Add text
    cv2.putText(image, "Test Image", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image