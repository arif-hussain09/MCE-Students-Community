import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json

from image_crypto_system import ContentDetectionCNN, ModelTrainer
from prepare_training_data import create_data_loaders
from config import MODEL_CONFIG, DEVICE

def train_content_model(data_dir: str, model_save_path: str = "trained_models/content_model.pth"):
    """Train the content detection model"""
    
    print(f"Training on device: {DEVICE}")
    
    # Create data loaders
    print("Loading training data...")
    train_loader, val_loader = create_data_loaders(
        data_dir, 
        batch_size=MODEL_CONFIG['batch_size']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = ContentDetectionCNN(num_classes=MODEL_CONFIG['num_classes'])
    model.to(DEVICE)
    
    # Initialize trainer
    trainer = ModelTrainer(model, DEVICE)
    
    # Training loop
    best_loss = float('inf')
    training_history = []
    
    print("Starting training...")
    for epoch in range(MODEL_CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{MODEL_CONFIG['epochs']}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss = validate_model(model, val_loader, DEVICE)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ New best model saved (val_loss: {val_loss:.4f})")
        
        # Update scheduler
        trainer.scheduler.step()
        
        # Save history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
    
    # Save training history
    history_path = model_save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining completed! Best model saved to: {model_save_path}")
    return model

def validate_model(model, val_loader, device):
    """Validate the model"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Validating"):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss

if __name__ == "__main__":
    # Example usage
    data_directory = "data"  # Update this path
    if os.path.exists(data_directory):
        train_content_model(data_directory)
    else:
        print(f"Data directory not found: {data_directory}")
        print("Please organize your training data first using prepare_training_data.py")