"""
Image Cryptography using Deep Learning
Neural Network-based Image Encryption and Decryption
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class EncryptionNetwork(nn.Module):
    """Neural network for encrypting images"""
    def __init__(self, key_size=128):
        super(EncryptionNetwork, self).__init__()
        self.key_size = key_size
        
        # Key embedding network
        self.key_embed = nn.Sequential(
            nn.Linear(key_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        
        # Image encoding path
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256)
        )
        
        # Key-conditioned transformation
        self.key_transform = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Decoding path for encrypted image
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, image, key):
        """
        Args:
            image: Input image tensor [B, 3, H, W]
            key: Encryption key tensor [B, key_size]
        Returns:
            encrypted_image: Encrypted image [B, 3, H, W]
        """
        # Embed the key
        key_embedded = self.key_embed(key)
        
        # Encode image
        encoded = self.encoder(image)
        
        # Apply key-based transformation
        # Reshape key embedding to match spatial dimensions
        B, C, H, W = encoded.shape
        key_spatial = key_embedded.view(B, -1, 1, 1).expand(B, -1, H, W)
        
        # Combine encoded image with key information
        encoded = encoded + key_spatial[:, :C, :, :]
        encoded = self.key_transform(encoded)
        
        # Decode to encrypted image
        encrypted = self.decoder(encoded)
        
        return encrypted


class DecryptionNetwork(nn.Module):
    """Neural network for decrypting images"""
    def __init__(self, key_size=128):
        super(DecryptionNetwork, self).__init__()
        self.key_size = key_size
        
        # Key embedding network (same architecture as encryption)
        self.key_embed = nn.Sequential(
            nn.Linear(key_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        
        # Encrypted image encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256)
        )
        
        # Key-conditioned inverse transformation
        self.key_transform = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Decoding path for decrypted image
        self.decoder = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, encrypted_image, key):
        """
        Args:
            encrypted_image: Encrypted image tensor [B, 3, H, W]
            key: Decryption key tensor [B, key_size]
        Returns:
            decrypted_image: Decrypted image [B, 3, H, W]
        """
        # Embed the key
        key_embedded = self.key_embed(key)
        
        # Encode encrypted image
        encoded = self.encoder(encrypted_image)
        
        # Apply key-based inverse transformation
        B, C, H, W = encoded.shape
        key_spatial = key_embedded.view(B, -1, 1, 1).expand(B, -1, H, W)
        
        # Combine with key information
        encoded = encoded + key_spatial[:, :C, :, :]
        encoded = self.key_transform(encoded)
        
        # Decode to original image
        decrypted = self.decoder(encoded)
        
        return decrypted


class ImageCryptographyModel(nn.Module):
    """Complete image cryptography model with encryption and decryption"""
    def __init__(self, key_size=128):
        super(ImageCryptographyModel, self).__init__()
        self.encryption_net = EncryptionNetwork(key_size)
        self.decryption_net = DecryptionNetwork(key_size)
        self.key_size = key_size
        
    def forward(self, image, key):
        """
        Forward pass: encrypt then decrypt
        Args:
            image: Input image [B, 3, H, W]
            key: Encryption/decryption key [B, key_size]
        Returns:
            encrypted: Encrypted image
            decrypted: Decrypted image (should match original)
        """
        encrypted = self.encryption_net(image, key)
        decrypted = self.decryption_net(encrypted, key)
        return encrypted, decrypted
    
    def encrypt(self, image, key):
        """Encrypt an image"""
        return self.encryption_net(image, key)
    
    def decrypt(self, encrypted_image, key):
        """Decrypt an image"""
        return self.decryption_net(encrypted_image, key)
