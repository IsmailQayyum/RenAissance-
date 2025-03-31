import os
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import urllib.request
import zipfile

from renaissance_gan import (
    DEVICE, BATCH_SIZE, EPOCHS, NOISE_DIM, IMAGE_SIZE,
    DATASET_PATH, OUTPUT_PATH, CHECKPOINT_PATH,
    RenaissanceTextDataset, Generator, Discriminator, 
    train_gan, generate_and_save_samples, load_spanish_text
)
from text_generation import (
    generate_specific_pages, evaluate_model, print_evaluation_report
)

def download_fonts():
    """Download the EB Garamond font for better Renaissance text generation"""
    font_dir = "EB_Garamond"
    if not os.path.exists(font_dir):
        print("Downloading EB Garamond font...")
        os.makedirs(font_dir, exist_ok=True)
        
        # Download from Google Fonts
        font_url = "https://fonts.google.com/download?family=EB%20Garamond"
        zip_path = os.path.join(font_dir, "eb_garamond.zip")
        
        try:
            urllib.request.urlretrieve(font_url, zip_path)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(font_dir)
                
            # Remove the zip file
            os.remove(zip_path)
            print("Font downloaded and extracted successfully.")
        except Exception as e:
            print(f"Error downloading font: {e}")
            print("Will use default fonts instead.")
    else:
        print("EB Garamond font directory already exists.")

def generate_synthetic_dataset():
    """Generate a synthetic Renaissance text dataset for training"""
    try:
        from renaissance_data_generator import generate_dataset
        generate_dataset()
        print("Synthetic dataset generated successfully.")
        return "synthetic_renaissance_data"  # Return the path to the generated dataset
    except Exception as e:
        print(f"Error generating synthetic dataset: {e}")
        print("Will use the original dataset instead.")
        return None

def main():
    parser = argparse.ArgumentParser(description='Renaissance Text GAN')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'generate', 'prepare'],
                        help='Train the model, generate images, or prepare dataset')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained generator model for generation mode')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--num_pages', type=int, default=5,
                        help='Number of pages to generate')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic dataset instead of Rodrigo corpus')
    
    args = parser.parse_args()
    
    # Create required directories
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
    # Prepare mode for downloading resources and generating synthetic data
    if args.mode == 'prepare':
        download_fonts()
        synthetic_path = generate_synthetic_dataset()
        print("Preparation completed.")
        return
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Initialize generator regardless of mode
    generator = Generator(NOISE_DIM).to(DEVICE)
    
    # Determine dataset path
    dataset_path = "synthetic_renaissance_data" if args.use_synthetic else DATASET_PATH
    
    if args.mode == 'train':
        print(f"Starting Renaissance Text GAN training...")
        print(f"Using device: {DEVICE}")
        print(f"Using {'synthetic' if args.use_synthetic else 'Rodrigo corpus'} dataset")
        
        # Download fonts if needed
        download_fonts()
        
        # Generate synthetic dataset if needed and requested
        if args.use_synthetic and not os.path.exists("synthetic_renaissance_data"):
            print("Synthetic dataset not found, generating...")
            generate_synthetic_dataset()
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        dataset = RenaissanceTextDataset(dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Initialize discriminator
        discriminator = Discriminator().to(DEVICE)
        
        print(f"Model architecture:")
        print(f"Generator parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}")
        print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad)}")
        
        # Train the model
        print(f"Starting training for {args.epochs} epochs...")
        generator, discriminator = train_gan(generator, discriminator, dataloader, epochs=args.epochs)
        
        # Generate samples from the trained model
        print("Generating and saving sample images...")
        generate_and_save_samples(generator, epoch=args.epochs)
        
        # Save final model
        torch.save(generator.state_dict(), os.path.join(CHECKPOINT_PATH, "generator_final.pt"))
        torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_PATH, "discriminator_final.pt"))
        
        # Load real samples for evaluation
        print("Evaluating model performance...")
        real_samples = []
        for i, (imgs) in enumerate(dataloader):
            if i >= 5:  # Get first 5 batches for evaluation
                break
            real_samples.append(imgs)
        
        evaluation_results = evaluate_model(generator, real_samples)
        print_evaluation_report(evaluation_results)
        
    elif args.mode == 'generate':
        # Download fonts if needed
        download_fonts()
        
        # Load trained model
        model_path = args.model if args.model else os.path.join(CHECKPOINT_PATH, "generator_final.pt")
        print(f"Loading trained generator from {model_path}...")
        
        try:
            generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
            generator.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Load Spanish text from Rodrigo corpus or use Don Quixote
        print("Loading historical Spanish text...")
        spanish_text = load_spanish_text()
        
        # Generate pages
        print(f"Generating {args.num_pages} pages of Renaissance-style text...")
        enhanced_pages = generate_specific_pages(generator, spanish_text, num_pages=args.num_pages)
        
        print(f"Pages generated and saved to {OUTPUT_PATH}")
        
        # Load some real samples for evaluation
        print("Evaluating generated pages...")
        dataset = RenaissanceTextDataset(dataset_path, transform=transform)
        real_samples = [dataset[i] for i in range(min(args.num_pages, len(dataset)))]
        
        evaluation_results = evaluate_model(generator, real_samples)
        print_evaluation_report(evaluation_results)
    
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main() 