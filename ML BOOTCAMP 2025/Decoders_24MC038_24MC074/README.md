# Image Cryptography using Deep Learning

A neural network-based approach to image encryption and decryption using deep learning. This project implements an end-to-end trainable system that can encrypt images with a key and decrypt them back to the original form.

## Features

- 🔐 Neural Network-based Encryption
- 🔑 Key-Dependent Security  
- 📊 Comprehensive Metrics (MSE, PSNR, SSIM)
- 🎨 Visual Analysis Tools
- 🚀 Easy to Use CLI
- 📈 Training Visualization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Demo (1 minute)
```bash
python demo.py
```

### 2. Train Model (15-30 minutes on GPU)
```bash
python train.py
```

### 3. Encrypt Your Image
```bash
python inference.py --mode encrypt --image photo.jpg --output encrypted/
```

### 4. Decrypt Image
```bash
python inference.py --mode decrypt \
  --image encrypted/photo_encrypted.png \
  --key encrypted/photo_key.npy \
  --output decrypted/
```

### 5. Full Pipeline with Visualization
```bash
python inference.py --mode both \
  --image photo.jpg \
  --output results/ \
  --visualize \
  --test-wrong-key
```

## Project Structure

```
image_cryptography_dl/
├── model.py              # Neural network architecture
├── train.py              # Training script
├── inference.py          # Encrypt/decrypt interface
├── demo.py              # Quick demonstration
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## How It Works

The system uses two neural networks:

1. **Encryption Network**: Transforms image using a secret key
2. **Decryption Network**: Recovers original image with correct key

Both networks use encoder-decoder architecture with residual blocks and key-conditioned transformations.

## Expected Results

After training:
- **Reconstruction PSNR**: > 30 dB
- **Encryption Correlation**: < 0.1  
- **Wrong Key PSNR**: < 15 dB

## License

MIT License - See LICENSE file

## Disclaimer

This is an experimental/educational project. 