import argparse
import os
import sys
from PIL import Image
from image_crypto_system import ImageCryptographySystem

def main():
    parser = argparse.ArgumentParser(description="Image Cryptography System")
    parser.add_argument('command', choices=['process', 'decrypt'], help='Command to execute')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--password', help='Encryption password')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--metadata', help='Metadata file path (for decryption)')
    parser.add_argument('--key', help='Encryption key (for decryption)')
    parser.add_argument('--no-blur', action='store_true', help='Disable blur processing')
    parser.add_argument('--no-encrypt', action='store_true', help='Disable encryption')
    
    args = parser.parse_args()
    
    # Initialize system
    crypto_system = ImageCryptographySystem(
        model_path=args.model,
        password=args.password
    )
    
    if args.command == 'process':
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
        
        output_path = args.output or f"processed_{os.path.basename(args.input)}"
        
        try:
            result = crypto_system.process_image(
                image_path=args.input,
                output_path=output_path,
                encryption_enabled=not args.no_encrypt,
                blur_enabled=not args.no_blur
            )
            
            print(f"✓ Processing completed: {output_path}")
            print(f"Encryption key: {result['encryption_key']}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            sys.exit(1)
    
    elif args.command == 'decrypt':
        if not all([args.metadata, args.key]):
            print("Error: Decryption requires --metadata and --key arguments")
            sys.exit(1)
        
        try:
            decrypted_image = crypto_system.decrypt_image(
                processed_image_path=args.input,
                metadata_path=args.metadata,
                encryption_key=args.key
            )
            
            output_path = args.output or f"decrypted_{os.path.basename(args.input)}"
            # Convert numpy array to PIL Image and save
            Image.fromarray(decrypted_image).save(output_path)
            print(f"✓ Decryption completed: {output_path}")
            
        except Exception as e:
            print(f"Error decrypting image: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()