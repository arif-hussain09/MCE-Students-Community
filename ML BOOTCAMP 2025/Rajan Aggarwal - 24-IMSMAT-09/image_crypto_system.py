import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image, ImageFilter
import hashlib
from cryptography.fernet import Fernet
import base64
import json
import os
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class ContentDetectionCNN(nn.Module):
    """
    Custom CNN for detecting inappropriate content in images
    """
    def __init__(self, num_classes=5):
        super(ContentDetectionCNN, self).__init__()
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify final layer for our classes
        # Classes: violence, abuse, adult_content, scary, normal
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Add additional layers for better feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Use both backbone and custom features
        backbone_features = self.backbone(x)
        
        # Custom feature extraction
        custom_features = self.feature_extractor(x)
        custom_features = custom_features.view(custom_features.size(0), -1)
        custom_output = self.classifier(custom_features)
        
        # Combine outputs
        combined = (backbone_features + custom_output) / 2
        return combined

class ObjectDetectionYOLO:
    """
    YOLO-style object detection for localizing inappropriate content
    """
    def __init__(self):
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        
    def detect_objects(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect objects and return bounding boxes
        Returns: List of (x1, y1, x2, y2, confidence)
        """
        height, width = image.shape[:2]
        
        # Simulate object detection (in real implementation, use trained YOLO)
        # For demo purposes, we'll use simple gradient-based detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection to find potential objects
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 50:  # Minimum size threshold
                    boxes.append((x, y, x + w, y + h, 0.8))
        
        return boxes

class AdvancedEncryption:
    """
    Advanced encryption system for image regions
    """
    def __init__(self, password: str = None):
        if password:
            key = hashlib.sha256(password.encode()).digest()
            self.key = base64.urlsafe_b64encode(key)
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
    def encrypt_region(self, image_region: np.ndarray) -> Tuple[bytes, np.ndarray]:
        """
        Encrypt image region and return encrypted data + noise pattern
        """
        # Convert region to bytes
        region_bytes = cv2.imencode('.png', image_region)[1].tobytes()
        
        # Encrypt the region
        encrypted_data = self.cipher.encrypt(region_bytes)
        
        # Create noise pattern to replace the region
        noise_pattern = self._generate_noise_pattern(image_region.shape)
        
        return encrypted_data, noise_pattern
    
    def decrypt_region(self, encrypted_data: bytes) -> np.ndarray:
        """
        Decrypt image region
        """
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_data)
            nparr = np.frombuffer(decrypted_bytes, np.uint8)
            image_region = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image_region
        except Exception as e:
            print(f"Decryption failed: {e}")
            return None
    
    def _generate_noise_pattern(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate sophisticated noise pattern
        """
        # Create multiple noise layers
        noise1 = np.random.randint(0, 256, shape, dtype=np.uint8)
        noise2 = np.random.normal(128, 64, shape).astype(np.uint8)
        
        # Blend noises
        alpha = np.random.random(shape)
        blended_noise = (alpha * noise1 + (1 - alpha) * noise2).astype(np.uint8)
        
        # Add some structure to make it less obviously random
        kernel = np.ones((5, 5), np.float32) / 25
        blended_noise = cv2.filter2D(blended_noise, -1, kernel)
        
        return blended_noise

class ImageCryptographySystem:
    """
    Main system combining ML detection with advanced encryption
    """
    def __init__(self, model_path: str = None, password: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.content_model = ContentDetectionCNN()
        if model_path and os.path.exists(model_path):
            self.content_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.content_model.to(self.device)
        self.content_model.eval()
        
        self.object_detector = ObjectDetectionYOLO()
        self.encryptor = AdvancedEncryption(password)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Content categories
        self.categories = ['violence', 'abuse', 'adult_content', 'scary', 'normal']
        self.inappropriate_threshold = 0.6
        
    def analyze_image_content(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image content using deep learning
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.content_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        # Create results dictionary
        results = {}
        for i, category in enumerate(self.categories):
            results[category] = float(probabilities[i])
            
        return results
    
    def detect_inappropriate_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Detect inappropriate regions in image
        Returns: List of (x1, y1, x2, y2, category, confidence)
        """
        # Get object bounding boxes
        boxes = self.object_detector.detect_objects(image)
        
        inappropriate_regions = []
        
        for box in boxes:
            x1, y1, x2, y2, obj_confidence = box
            
            # Extract region
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                continue
                
            # Analyze region content
            content_analysis = self.analyze_image_content(region)
            
            # Check if any inappropriate category exceeds threshold
            for category, confidence in content_analysis.items():
                if category != 'normal' and confidence > self.inappropriate_threshold:
                    inappropriate_regions.append((x1, y1, x2, y2, category, confidence))
                    break
                    
        return inappropriate_regions
    
    def apply_advanced_blur(self, image: np.ndarray, region: Tuple[int, int, int, int], intensity: float = 1.0) -> np.ndarray:
        """
        Apply sophisticated blurring techniques
        """
        x1, y1, x2, y2 = region
        region_img = image[y1:y2, x1:x2].copy()
        
        # Apply multiple blur techniques based on intensity
        if intensity > 0.8:
            # Heavy blur for high-risk content
            blurred = cv2.GaussianBlur(region_img, (51, 51), 20)
            blurred = cv2.medianBlur(blurred, 21)
        elif intensity > 0.6:
            # Medium blur
            blurred = cv2.GaussianBlur(region_img, (31, 31), 15)
            blurred = cv2.bilateralFilter(blurred, 15, 80, 80)
        else:
            # Light blur
            blurred = cv2.GaussianBlur(region_img, (21, 21), 10)
        
        # Add pixelation effect
        if intensity > 0.7:
            h, w = blurred.shape[:2]
            pixel_size = max(10, int(20 * intensity))
            temp = cv2.resize(blurred, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Apply result to original image
        result = image.copy()
        result[y1:y2, x1:x2] = blurred
        
        return result
    
    def encrypt_image_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Dict]:
        """
        Encrypt specified regions of the image
        """
        encrypted_data = {}
        result_image = image.copy()
        
        for i, region in enumerate(regions):
            x1, y1, x2, y2 = region
            region_img = image[y1:y2, x1:x2].copy()
            
            # Encrypt region
            encrypted_bytes, noise_pattern = self.encryptor.encrypt_region(region_img)
            
            # Store encrypted data
            encrypted_data[f'region_{i}'] = {
                'data': base64.b64encode(encrypted_bytes).decode('utf-8'),
                'bounds': region,
                'shape': region_img.shape
            }
            
            # Replace region with noise
            result_image[y1:y2, x1:x2] = noise_pattern
            
        return result_image, encrypted_data
    
    def process_image(self, image_path: str, output_path: str = None, 
                     encryption_enabled: bool = True, blur_enabled: bool = True,
                     save_metadata: bool = True) -> Dict:
        """
        Main processing function
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Analyze overall content
        content_analysis = self.analyze_image_content(image)
        print(f"Content analysis: {content_analysis}")
        
        # Detect inappropriate regions
        inappropriate_regions = self.detect_inappropriate_regions(image)
        print(f"Found {len(inappropriate_regions)} inappropriate regions")
        
        processed_image = image.copy()
        metadata = {
            'original_path': image_path,
            'content_analysis': content_analysis,
            'inappropriate_regions': inappropriate_regions,
            'processing_applied': []
        }
        
        if inappropriate_regions:
            # Apply blur if enabled
            if blur_enabled:
                for region_data in inappropriate_regions:
                    x1, y1, x2, y2, category, confidence = region_data
                    processed_image = self.apply_advanced_blur(
                        processed_image, (x1, y1, x2, y2), confidence
                    )
                metadata['processing_applied'].append('blur')
                print("Applied advanced blurring to inappropriate regions")
            
            # Apply encryption if enabled
            if encryption_enabled:
                regions_to_encrypt = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in inappropriate_regions]
                processed_image, encrypted_data = self.encrypt_image_regions(
                    processed_image, regions_to_encrypt
                )
                metadata['encrypted_regions'] = encrypted_data
                metadata['processing_applied'].append('encryption')
                print("Applied encryption to inappropriate regions")
        
        # Save processed image
        if output_path:
            cv2.imwrite(output_path, processed_image)
            print(f"Saved processed image to: {output_path}")
            
            # Save metadata
            if save_metadata:
                metadata_path = output_path.rsplit('.', 1)[0] + '_metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Saved metadata to: {metadata_path}")
        
        return {
            'processed_image': processed_image,
            'metadata': metadata,
            'encryption_key': base64.b64encode(self.encryptor.key).decode('utf-8')
        }
    
    def decrypt_image(self, processed_image_path: str, metadata_path: str, 
                     encryption_key: str) -> np.ndarray:
        """
        Decrypt processed image using metadata and encryption key
        """
        # Load processed image and metadata
        image = cv2.imread(processed_image_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create decryptor with provided key
        key = base64.b64decode(encryption_key.encode())
        decryptor = AdvancedEncryption()
        decryptor.key = key
        decryptor.cipher = Fernet(key)
        
        # Decrypt regions if they exist
        if 'encrypted_regions' in metadata:
            for region_id, region_data in metadata['encrypted_regions'].items():
                encrypted_bytes = base64.b64decode(region_data['data'].encode())
                x1, y1, x2, y2 = region_data['bounds']
                
                # Decrypt region
                decrypted_region = decryptor.decrypt_region(encrypted_bytes)
                if decrypted_region is not None:
                    # Resize if necessary
                    expected_shape = region_data['shape']
                    if decrypted_region.shape != expected_shape:
                        decrypted_region = cv2.resize(
                            decrypted_region, 
                            (expected_shape[1], expected_shape[0])
                        )
                    
                    # Restore region
                    image[y1:y2, x1:x2] = decrypted_region
        
        return image

# Training utilities (for custom model training)
class ModelTrainer:
    """
    Utility class for training the content detection model
    """
    def __init__(self, model: ContentDetectionCNN, device: torch.device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self, dataloader, epoch: int) -> float:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss

# Example usage and testing
def main():
    """
    Example usage of the Image Cryptography System
    """
    # Initialize system
    crypto_system = ImageCryptographySystem(password="secure_password_123")
    
    # Example image processing
    try:
        # Process an image
        result = crypto_system.process_image(
            image_path="test_image.jpg",
            output_path="processed_image.jpg",
            encryption_enabled=True,
            blur_enabled=True
        )
        
        print("Processing completed successfully!")
        print(f"Encryption key: {result['encryption_key']}")
        
        # Example decryption
        decrypted_image = crypto_system.decrypt_image(
            processed_image_path="processed_image.jpg",
            metadata_path="processed_image_metadata.json",
            encryption_key=result['encryption_key']
        )
        
        cv2.imwrite("decrypted_image.jpg", decrypted_image)
        print("Decryption completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a test image available or modify the path accordingly.")

if __name__ == "__main__":
    main()