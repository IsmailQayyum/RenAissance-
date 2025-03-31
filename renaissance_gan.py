import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
import cv2
from torchvision.utils import save_image

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# The Rodrigo corpus images appear to be line snippets, not full pages
# Use a smaller image size that's more appropriate for this content
IMAGE_SIZE = 128  # Reduced size for better training with line images 
CHANNELS = 1
BATCH_SIZE = 16  # Increased batch size for smaller images
EPOCHS = 100
NOISE_DIM = 100
LEARNING_RATE = 0.0001
BETA1 = 0.5
BETA2 = 0.999
DATASET_PATH = r"../Gsoc/Rodrigo corpus 1.0.0.tar/Rodrigo corpus 1.0.0/images"
OUTPUT_PATH = "output_renaissance"
CHECKPOINT_PATH = "checkpoints_renaissance"
TRANSCRIPTIONS_PATH = r"../Gsoc/Rodrigo corpus 1.0.0.tar/Rodrigo corpus 1.0.0/text/transcriptions.txt"

# Ensure directories exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Dataset class for Renaissance manuscript images
class RenaissanceTextDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.image_files = [f for f in os.listdir(images_folder) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform
        print(f"Loaded {len(self.image_files)} images from {images_folder}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform if none provided
                image = transforms.ToTensor()(image)
            
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image instead of crashing
            blank = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 255)
            if self.transform:
                return self.transform(blank)
            else:
                return transforms.ToTensor()(blank)

# Enhanced augmentation functions for Renaissance printing imperfections
def apply_renaissance_augmentation(image_array):
    """Apply realistic Renaissance printing imperfections to the image"""
    # Convert to PIL Image for augmentation
    if torch.is_tensor(image_array):
        image_array = image_array.squeeze().cpu().numpy()
    
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    
    # 1. Paper texture and aging
    # Create a parchment-like texture
    paper_color = random.randint(220, 245)  # Slightly off-white
    paper_texture = Image.new('L', img.size, color=paper_color)
    
    # Add noise to simulate paper grain
    grain_noise = np.random.normal(0, 5, (img.size[1], img.size[0]))
    grain = Image.fromarray((grain_noise + 128).clip(0, 255).astype(np.uint8))
    paper_texture = Image.blend(paper_texture, grain, alpha=0.3)
    
    # 2. Ink bleed effect - common in Renaissance printing
    if random.random() > 0.3:  # High chance of ink bleed
        # Fix: Use valid filter size (must be odd integer >= 3)
        bleed_severity = random.choice([3, 5])  # Use only valid filter sizes
        img = img.filter(ImageFilter.MaxFilter(bleed_severity))
    
    # 3. Ink smudging and uneven distribution
    if random.random() > 0.4:
        # Create smudge effect
        smudge_mask = Image.new('L', img.size, 0)
        smudge_draw = ImageDraw.Draw(smudge_mask)
        
        # Add random smudges
        for _ in range(random.randint(2, 5)):
            x1 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            x2 = x1 + random.randint(10, 50)
            y2 = y1 + random.randint(5, 20)
            smudge_opacity = random.randint(50, 150)
            smudge_draw.ellipse([x1, y1, x2, y2], fill=smudge_opacity)
        
        # Apply smudge with blur
        smudge_mask = smudge_mask.filter(ImageFilter.GaussianBlur(radius=3))
        # Convert the mask to a numpy array and use as alpha values instead of using the image itself
        smudge_alpha_array = np.array(ImageOps.invert(smudge_mask)) / 255.0
        smudge_alpha = np.mean(smudge_alpha_array)  # Use average value for alpha
        img = Image.blend(img, img.filter(ImageFilter.GaussianBlur(1)), alpha=smudge_alpha)
    
    # 4. Faded text in random areas - common in old prints
    if random.random() > 0.5:
        fade_mask = Image.new('L', img.size, 255)
        fade_draw = ImageDraw.Draw(fade_mask)
        
        # Create random fade patterns
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, img.width)
            y1 = random.randint(0, img.height)
            x2 = x1 + random.randint(50, 200)
            y2 = y1 + random.randint(30, 100)
            fade_opacity = random.randint(50, 200)
            fade_draw.rectangle([x1, y1, x2, y2], fill=fade_opacity)
        
        # Blur the fade mask for a more natural transition
        fade_mask = fade_mask.filter(ImageFilter.GaussianBlur(radius=20))
        
        # Apply the fade effect
        fade_alpha_array = np.array(ImageOps.invert(fade_mask)) / 255.0
        fade_alpha = float(np.mean(fade_alpha_array))  # Average value between 0-1
        img = Image.blend(paper_texture, img, alpha=1.0 - fade_alpha)
    
    # 5. Slight rotation to simulate misalignment in printing process
    if random.random() > 0.6:
        angle = random.uniform(-1.5, 1.5)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # 6. Uneven lighting/shadows from book binding
    if random.random() > 0.4:
        gradient = Image.new('L', img.size)
        draw = ImageDraw.Draw(gradient)
        
        # Gradient for binding shadow (usually darker on one side)
        shadow_side = random.choice(['left', 'right'])
        if shadow_side == 'left':
            for x in range(img.width):
                brightness = int(255 * (0.7 + 0.3 * x / img.width))
                draw.line([(x, 0), (x, img.height)], fill=brightness)
        else:
            for x in range(img.width):
                brightness = int(255 * (1.0 - 0.3 * x / img.width))
                draw.line([(x, 0), (x, img.height)], fill=brightness)
        
        # Apply the gradient
        img = Image.blend(img, gradient, alpha=random.uniform(0.1, 0.25))
    
    # Convert back to numpy array and normalize
    img_array = np.array(img) / 255.0
    return img_array

# Part 2: Define the GAN architecture in PyTorch with WGAN-GP approach
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        
        # Initial size: 8x8
        self.init_size = IMAGE_SIZE // 16  # For 128x128 output, start with 8x8
        self.l1 = nn.Linear(noise_dim, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer with tanh activation
            nn.Conv2d(16, CHANNELS, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Simple CNN without fixed-size normalization
        self.model = nn.Sequential(
            # Layer 1: 128x128 -> 64x64
            nn.Conv2d(CHANNELS, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64x64 -> 32x32
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.InstanceNorm2d(32),  # Instance norm doesn't depend on spatial dimensions
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 16x16 -> 8x8
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final classification layer
        self.adv_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1)  # 8x8 feature maps from final conv layer
        )

    def forward(self, img):
        features = self.model(img)
        validity = self.adv_layer(features)
        return validity

# WGAN training function with gradient penalty
def train_gan(generator, discriminator, dataloader, epochs=EPOCHS):
    # Optimizers with adjusted learning rates
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    
    # For tracking progress
    g_losses = []
    d_losses = []
    
    # Lambda for gradient penalty
    lambda_gp = 10
    
    # Function to compute gradient penalty
    def compute_gradient_penalty(real_samples, fake_samples):
        # Random weight for interpolation
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=DEVICE)
        # Interpolate between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        # Get critic scores for interpolated images
        d_interpolates = discriminator(interpolates)
        # Get gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=DEVICE),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # Compute gradient penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # Training loop
    for epoch in range(epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        batches_processed = 0
        
        for i, imgs in enumerate(dataloader):
            batches_processed += 1
            
            # Configure input (make sure it's properly normalized)
            real_imgs = imgs.to(DEVICE)
            batch_size = real_imgs.size(0)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Generate a batch of fake images
            z = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
            fake_imgs = generator(z)
            
            # WGAN loss with gradient penalty (critic outputs raw scores, not probabilities)
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            gradient_penalty = compute_gradient_penalty(real_imgs, fake_imgs.detach())
            
            # Discriminator/critic loss: maximize real - fake with gradient penalty
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            
            # Train generator every 5 discriminator iterations
            if i % 5 == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                
                # Generate new batch of images
                z = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
                gen_imgs = generator(z)
                
                # Generator's WGAN loss: maximize fake scores
                fake_validity = discriminator(gen_imgs)
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()
                
                # Track generator loss
                epoch_g_loss += g_loss.item()
            
            # Track discriminator loss
            epoch_d_loss += d_loss.item()
            
            # Print status
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item() if i % 5 == 0 else 'N/A'}]")
        
        # Calculate and log average epoch losses (average G loss only for steps where G was trained)
        avg_d_loss = epoch_d_loss / batches_processed
        generator_updates = batches_processed // 5
        avg_g_loss = epoch_g_loss / max(1, generator_updates)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch} - Average G loss: {avg_g_loss:.4f}, Average D loss: {avg_d_loss:.4f}")
        
        # Save sample images
        if epoch % 2 == 0:  # More frequent saving to track progress
            generate_and_save_samples(generator, epoch)
            # Plot loss curves
            plot_losses(g_losses, d_losses, save_path=os.path.join(OUTPUT_PATH, f"losses_epoch_{epoch}.png"))
            
        # Save model checkpoints
        if epoch % 5 == 0 or epoch == epochs - 1:
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_PATH, f"generator_epoch_{epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_PATH, f"discriminator_epoch_{epoch}.pt"))
        
    return generator, discriminator

def generate_and_save_samples(generator, epoch, n_row=3, n_col=3):
    """Generate sample images and save them as a grid"""
    z = torch.randn(n_row * n_col, NOISE_DIM, device=DEVICE, dtype=torch.float)
    with torch.no_grad():
        gen_imgs = generator(z)
    
    # Convert from [-1,1] to [0,1] for saving images
    gen_imgs_normalized = (gen_imgs + 1) / 2
    save_image(gen_imgs_normalized, os.path.join(OUTPUT_PATH, f"renaissance_text_{epoch:03d}.png"), 
               nrow=n_row, normalize=False)
    
    # Save individual images with additional Renaissance post-processing
    for i in range(n_row * n_col):
        # Convert from [-1,1] to [0,1] range
        img = (gen_imgs[i].detach().cpu().squeeze().numpy() + 1) / 2
        # Apply additional Renaissance style degradation
        img = apply_renaissance_augmentation(img)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(OUTPUT_PATH, f"sample_epoch_{epoch:03d}_{i+1}.jpg"))

def plot_losses(g_losses, d_losses, save_path):
    """Plot and save loss curves during training"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('GAN Loss During Training')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to load historical Spanish text from Rodrigo corpus
def load_spanish_text():
    """Load historical Spanish text from the Rodrigo corpus transcriptions"""
    try:
        with open(TRANSCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract the actual text content, removing line IDs and metadata
        text_content = []
        for line in lines:
            if line.strip():
                parts = line.strip().split(' ', 1)
                if len(parts) > 1:
                    text_content.append(parts[1])
        
        return ' '.join(text_content)
    except Exception as e:
        print(f"Error loading Spanish text: {e}")
        # Sample text if file not available (from Don Quixote)
        return """
        En un lugar de la Mancha, de cuyo nombre no quiero acordarme, 
        no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, 
        adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, 
        salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, 
        algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.
        """ 