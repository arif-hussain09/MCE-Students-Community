"""
Inference script for Image Cryptography
Encrypt and decrypt custom images
"""

import torch
import numpy as np
from PIL import Image
import argparse
import os
from model import ImageCryptographyModel
import matplotlib.pyplot as plt


def load_model(checkpoint_path, key_size=128, device='cpu'):
    """Load trained model from checkpoint"""
    model = ImageCryptographyModel(key_size=key_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs with loss: {checkpoint['loss']:.4f}")
    return model


def preprocess_image(image_path, size=64):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize to [-1, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = (image_array - 0.5) / 0.5
    
    # Convert to tensor [1, 3, H, W]
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
    
    return image_tensor


def postprocess_image(tensor):
    """Convert tensor back to image"""
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu().detach()
    
    # Denormalize from [-1, 1] to [0, 1]
    image_array = tensor.numpy().transpose(1, 2, 0)
    image_array = (image_array * 0.5) + 0.5
    image_array = np.clip(image_array, 0, 1)
    
    # Convert to PIL Image
    image = Image.fromarray((image_array * 255).astype(np.uint8))
    return image


def save_key(key, filepath):
    """Save encryption key to file"""
    np.save(filepath, key.cpu().numpy())
    print(f"Key saved to {filepath}")


def load_key(filepath, device):
    """Load encryption key from file"""
    key = np.load(filepath)
    return torch.from_numpy(key).to(device)


def encrypt_image(model, image_path, output_dir, key_size=128, device='cpu'):
    """Encrypt an image and save the result"""
    # Preprocess image
    image = preprocess_image(image_path).to(device)
    
    # Generate random key
    key = torch.randn(1, key_size).to(device)
    
    # Encrypt
    with torch.no_grad():
        encrypted = model.encrypt(image, key)
    
    # Save encrypted image
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    encrypted_image = postprocess_image(encrypted)
    encrypted_path = os.path.join(output_dir, f'{base_name}_encrypted.png')
    encrypted_image.save(encrypted_path)
    print(f"Encrypted image saved to {encrypted_path}")
    
    # Save key
    key_path = os.path.join(output_dir, f'{base_name}_key.npy')
    save_key(key, key_path)
    
    return encrypted_path, key_path


def decrypt_image(model, encrypted_path, key_path, output_dir, device='cpu'):
    """Decrypt an encrypted image using the key"""
    # Load encrypted image
    encrypted = preprocess_image(encrypted_path).to(device)
    
    # Load key
    key = load_key(key_path, device)
    
    # Decrypt
    with torch.no_grad():
        decrypted = model.decrypt(encrypted, key)
    
    # Save decrypted image
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(encrypted_path))[0]
    base_name = base_name.replace('_encrypted', '')
    
    decrypted_image = postprocess_image(decrypted)
    decrypted_path = os.path.join(output_dir, f'{base_name}_decrypted.png')
    decrypted_image.save(decrypted_path)
    print(f"Decrypted image saved to {decrypted_path}")
    
    return decrypted_path


def visualize_process(original_path, encrypted_path, decrypted_path, output_path):
    """Visualize the encryption-decryption process"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Load images
    original = Image.open(original_path)
    encrypted = Image.open(encrypted_path)
    decrypted = Image.open(decrypted_path)
    
    # Plot
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(encrypted)
    axes[1].set_title('Encrypted Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(decrypted)
    axes[2].set_title('Decrypted Image', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def test_wrong_key(model, encrypted_path, output_dir, key_size=128, device='cpu'):
    """Test decryption with wrong key to verify security"""
    encrypted = preprocess_image(encrypted_path).to(device)
    
    # Generate a different random key
    wrong_key = torch.randn(1, key_size).to(device)
    
    # Try to decrypt with wrong key
    with torch.no_grad():
        wrong_decrypted = model.decrypt(encrypted, wrong_key)
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    wrong_decrypted_image = postprocess_image(wrong_decrypted)
    wrong_path = os.path.join(output_dir, 'decrypted_with_wrong_key.png')
    wrong_decrypted_image.save(wrong_path)
    print(f"Wrong key decryption saved to {wrong_path}")
    
    return wrong_path


def main():
    parser = argparse.ArgumentParser(description='Image Cryptography Inference')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['encrypt', 'decrypt', 'both'],
                       help='Operation mode')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--key', type=str, default=None,
                       help='Path to encryption key (for decryption)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--key-size', type=int, default=128,
                       help='Encryption key size')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of the process')
    parser.add_argument('--test-wrong-key', action='store_true',
                       help='Test decryption with wrong key')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.model, key_size=args.key_size, device=device)
    
    if args.mode == 'encrypt':
        print('\n--- Encrypting Image ---')
        encrypted_path, key_path = encrypt_image(
            model, args.image, args.output, args.key_size, device
        )
        
        if args.test_wrong_key:
            print('\n--- Testing Wrong Key ---')
            test_wrong_key(model, encrypted_path, args.output, args.key_size, device)
    
    elif args.mode == 'decrypt':
        if args.key is None:
            raise ValueError("--key must be provided for decryption")
        
        print('\n--- Decrypting Image ---')
        decrypted_path = decrypt_image(
            model, args.image, args.key, args.output, device
        )
    
    elif args.mode == 'both':
        print('\n--- Encrypting Image ---')
        encrypted_path, key_path = encrypt_image(
            model, args.image, args.output, args.key_size, device
        )
        
        print('\n--- Decrypting Image ---')
        decrypted_path = decrypt_image(
            model, encrypted_path, key_path, args.output, device
        )
        
        if args.visualize:
            print('\n--- Creating Visualization ---')
            visualize_process(
                args.image,
                encrypted_path,
                decrypted_path,
                os.path.join(args.output, 'process_visualization.png')
            )
        
        if args.test_wrong_key:
            print('\n--- Testing Wrong Key ---')
            test_wrong_key(model, encrypted_path, args.output, args.key_size, device)


if __name__ == '__main__':
    main()
