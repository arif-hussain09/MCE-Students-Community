import torch

# Model Configuration
MODEL_CONFIG = {
    'num_classes': 5,
    'input_size': 224,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50
}

# Content Categories
CONTENT_CATEGORIES = {
    0: 'violence',
    1: 'abuse', 
    2: 'adult_content',
    3: 'scary',
    4: 'normal'
}

# Processing Configuration
PROCESSING_CONFIG = {
    'inappropriate_threshold': 0.6,
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'min_region_area': 1000,
    'min_region_size': 50
}

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")