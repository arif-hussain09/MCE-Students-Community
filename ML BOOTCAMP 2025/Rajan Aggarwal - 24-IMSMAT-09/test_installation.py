import torch
import cv2
import numpy as np
from PIL import Image
from cryptography.fernet import Fernet

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV version: {cv2.__version__}")
print("All dependencies installed successfully!")