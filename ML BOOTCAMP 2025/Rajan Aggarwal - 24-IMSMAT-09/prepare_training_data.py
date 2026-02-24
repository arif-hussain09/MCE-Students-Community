import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class ContentDataset(Dataset):
    """Dataset for content classification training"""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Expected directory structure:
        # data_dir/
        #   ├── violence/
        #   ├── abuse/
        #   ├── adult_content/
        #   ├── scary/
        #   └── normal/
        
        categories = ['violence', 'abuse', 'adult_content', 'scary', 'normal']
        
        for idx, category in enumerate(categories):
            category_path = os.path.join(data_dir, category)
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(category_path, filename),
                            idx
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_data_loaders(data_dir: str, batch_size: int = 32):
    """Create training and validation data loaders"""
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ContentDataset(
        os.path.join(data_dir, 'train'), 
        transform=transform_train
    )
    val_dataset = ContentDataset(
        os.path.join(data_dir, 'val'), 
        transform=transform_val
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Data preparation utilities created")
    print("To use: organize your training data in the following structure:")
    print("""
    data/
      train/
        ├── violence/
        ├── abuse/
        ├── adult_content/
        ├── scary/
        └── normal/
      val/
        ├── violence/
        ├── abuse/
        ├── adult_content/
        ├── scary/
        └── normal/
    """)