"""
Training script for Image Cryptography Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import ImageCryptographyModel
import numpy as np


class CryptographyLoss(nn.Module):
    """Custom loss function for image cryptography"""
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super(CryptographyLoss, self).__init__()
        self.alpha = alpha  # Reconstruction loss weight
        self.beta = beta    # Encryption quality weight
        self.gamma = gamma  # Perceptual loss weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, original, encrypted, decrypted):
        """
        Calculate total loss
        Args:
            original: Original image
            encrypted: Encrypted image
            decrypted: Decrypted image
        """
        # Reconstruction loss: decrypted should match original
        reconstruction_loss = self.mse(decrypted, original)
        
        # Encryption quality: encrypted should be different from original
        # We want this to be maximized, so we use negative correlation
        encryption_quality = -torch.mean(torch.abs(encrypted - original))
        
        # Ensure encrypted image has good statistical properties
        # Encrypted image should have normalized distribution
        encrypted_flat = encrypted.view(encrypted.size(0), -1)
        mean_loss = torch.mean(encrypted_flat) ** 2
        std_loss = (torch.std(encrypted_flat) - 0.5) ** 2
        
        total_loss = (self.alpha * reconstruction_loss + 
                     self.beta * encryption_quality + 
                     self.gamma * (mean_loss + std_loss))
        
        return total_loss, reconstruction_loss, encryption_quality


def generate_random_keys(batch_size, key_size, device):
    """Generate random encryption keys"""
    return torch.randn(batch_size, key_size).to(device)


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        batch_size = images.size(0)
        
        # Generate random keys for this batch
        keys = generate_random_keys(batch_size, model.key_size, device)
        
        # Forward pass
        encrypted, decrypted = model(images, keys)
        
        # Calculate loss
        loss, recon_loss, enc_quality = criterion(images, encrypted, decrypted)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    
    return avg_loss, avg_recon


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            batch_size = images.size(0)
            
            keys = generate_random_keys(batch_size, model.key_size, device)
            encrypted, decrypted = model(images, keys)
            
            loss, recon_loss, _ = criterion(images, encrypted, decrypted)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    
    return avg_loss, avg_recon


def save_sample_images(model, dataloader, device, save_path, epoch):
    """Save sample encrypted and decrypted images"""
    model.eval()
    
    # Get one batch
    images, _ = next(iter(dataloader))
    images = images[:4].to(device)  # Take 4 images
    
    with torch.no_grad():
        keys = generate_random_keys(images.size(0), model.key_size, device)
        encrypted, decrypted = model(images, keys)
        
        # Test with wrong key
        wrong_keys = generate_random_keys(images.size(0), model.key_size, device)
        wrong_decrypted = model.decrypt(encrypted, wrong_keys)
    
    # Convert to numpy for plotting
    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    encrypted = encrypted.cpu().numpy().transpose(0, 2, 3, 1)
    decrypted = decrypted.cpu().numpy().transpose(0, 2, 3, 1)
    wrong_decrypted = wrong_decrypted.cpu().numpy().transpose(0, 2, 3, 1)
    
    # Normalize to [0, 1] for display
    images = (images + 1) / 2
    encrypted = (encrypted + 1) / 2
    decrypted = (decrypted + 1) / 2
    wrong_decrypted = (wrong_decrypted + 1) / 2
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        axes[i, 0].imshow(np.clip(images[i], 0, 1))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.clip(encrypted[i], 0, 1))
        axes[i, 1].set_title('Encrypted')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(np.clip(decrypted[i], 0, 1))
        axes[i, 2].set_title('Decrypted (Correct Key)')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(np.clip(wrong_decrypted[i], 0, 1))
        axes[i, 3].set_title('Decrypted (Wrong Key)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'samples_epoch_{epoch}.png'))
    plt.close()


def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    key_size = 128
    image_size = 64
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset (using CIFAR-10 as example)
    print('Loading dataset...')
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    print('Initializing model...')
    model = ImageCryptographyModel(key_size=key_size).to(device)
    
    # Loss and optimizer
    criterion = CryptographyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print('Starting training...')
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        
        # Train
        train_loss, train_recon = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_recon = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Train Recon: {train_recon:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Recon: {val_recon:.4f}')
        
        # Save sample images
        if epoch % 5 == 0:
            save_sample_images(model, val_loader, device, 'samples', epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'checkpoints/best_model.pth')
            print('Saved best model!')
        
        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.savefig('training_curves.png')
    plt.close()
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()
