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

IMAGE_SIZE = 512
CHANNELS = 1
BATCH_SIZE = 8
EPOCHS = 100
NOISE_DIM = 100
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
DATASET_PATH = "renaissance_samples/"
OUTPUT_PATH = "synthetic_output/"
CHECKPOINT_PATH = "checkpoints/"

# Ensure directories exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Part 1: Dataset preparation and augmentation
class RenaissanceTextDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Convert numpy array to PIL Image for transformations
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.ToTensor()(image)
            
        return image

def load_and_preprocess_images(dataset_path):
    images = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(dataset_path, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img)
    return np.array(images)

# Data augmentation to introduce realistic printing imperfections
def apply_renaissance_augmentation(image_array):
    # Convert to PIL Image for augmentation
    if torch.is_tensor(image_array):
        image_array = image_array.squeeze().cpu().numpy()
    
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    
    # Apply various Renaissance printing imperfections
    
    # 1. Randomized noise for paper texture
    noise = np.random.normal(0, 0.05, (img.size[1], img.size[0]))
    noise = Image.fromarray((noise * 255).astype(np.uint8))
    img = Image.blend(img, noise.convert('L'), alpha=0.2)
    
    # 2. Ink bleed effect
    if random.random() > 0.5:
        img = img.filter(ImageFilter.MaxFilter(3))
    
    # 3. Uneven ink distribution
    if random.random() > 0.5:
        enhancer = ImageOps.autocontrast(img, cutoff=(10, 90))
        img = Image.blend(img, enhancer, alpha=random.uniform(0.2, 0.5))
    
    # 4. Paper aging (slight yellowing/staining)
    if random.random() > 0.5:
        sepia_tone = Image.new('L', img.size, color=random.randint(180, 220))
        img = Image.blend(img, sepia_tone, alpha=random.uniform(0.05, 0.15))
    
    # 5. Slight rotation to simulate misalignment
    if random.random() > 0.7:
        angle = random.uniform(-2, 2)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # 6. Uneven lighting and shadows
    if random.random() > 0.6:
        gradient = Image.new('L', img.size)
        draw = ImageDraw.Draw(gradient)
        
        # Random gradient direction
        if random.random() > 0.5:
            # Horizontal gradient
            for x in range(img.width):
                brightness = int(255 * (0.7 + 0.3 * x / img.width))
                draw.line([(x, 0), (x, img.height)], fill=brightness)
        else:
            # Vertical gradient
            for y in range(img.height):
                brightness = int(255 * (0.7 + 0.3 * y / img.height))
                draw.line([(0, y), (img.width, y)], fill=brightness)
                
        img = Image.blend(img, gradient, alpha=random.uniform(0.1, 0.3))
    
    # Convert back to numpy array and normalize
    img_array = np.array(img) / 255.0
    return img_array

# Synthetic baseline dataset for training
def generate_baseline_data(num_samples=200):
    # Load a Renaissance font (assumed to be in project directory)
    try:
        # You would need to provide an appropriate font
        font = ImageFont.truetype("EB_Garamond/EBGaramond-Regular.ttf", 24)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    sample_texts = [
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme,",
        "no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero,",
        "adarga antigua, rocín flaco y galgo corredor.",
        "Una olla de algo más vaca que carnero,",
        "salpicón las más noches, duelos y quebrantos los sábados,",
        "lantejas los viernes, algún palomino de añadidura los domingos,",
        "consumían las tres partes de su hacienda.",
        "El resto della concluían sayo de velarte,",
        "calzas de velludo para las fiestas, con sus pantuflos de lo mesmo,",
        "y los días de entresemana se honraba con su vellorí de lo más fino."
    ]
    
    images = []
    for i in range(num_samples):
        # Create blank image
        img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=245)
        draw = ImageDraw.Draw(img)
        
        # Randomly select and position text
        y_position = 50
        for _ in range(random.randint(4, 8)):
            text = random.choice(sample_texts)
            draw.text((30, y_position), text, font=font, fill=0)
            y_position += random.randint(35, 45)
        
        # Apply Renaissance degradation effects
        img_array = np.array(img) / 255.0
        img_array = apply_renaissance_augmentation(img_array)
        images.append(img_array)
        
        # Save some examples for reference
        if i < 10:
            Image.fromarray((img_array * 255).astype(np.uint8)).save(
                os.path.join(DATASET_PATH, f"baseline_sample_{i+1}.jpg")
            )
    
    return np.array(images)

# Part 2: Define the GAN architecture in PyTorch
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        
        self.init_size = IMAGE_SIZE // 32  # Initial size before upsampling
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim, 256 * self.init_size * self.init_size)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, CHANNELS, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.model = nn.Sequential(
            *discriminator_block(CHANNELS, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        # The height and width of downsampled image
        ds_size = IMAGE_SIZE // 32
        self.adv_layer = nn.Linear(256 * ds_size * ds_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        validity = self.sigmoid(validity)
        return validity

# Part 3: Training function
def train_gan(generator, discriminator, dataloader, epochs=EPOCHS):
    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # Training loop
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            # Convert tensors to the correct format and send to device
            imgs = imgs.float().to(DEVICE)  # Ensure float type consistency
            
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1, device=DEVICE, dtype=torch.float)
            fake = torch.zeros(imgs.size(0), 1, device=DEVICE, dtype=torch.float)
            
            # Label smoothing for better training
            valid = valid * 0.9  

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.size(0), NOISE_DIM, device=DEVICE, dtype=torch.float)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Print status
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            
        # Save sample images
        if epoch % 5 == 0:
            generate_and_save_samples(generator, epoch)
            
        # Save model checkpoints
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_PATH, f"generator_epoch_{epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_PATH, f"discriminator_epoch_{epoch}.pt"))
        
    return generator, discriminator

def generate_and_save_samples(generator, epoch, n_row=3, n_col=3):
    """Generate sample images and save them as a grid"""
    z = torch.randn(n_row * n_col, NOISE_DIM, device=DEVICE, dtype=torch.float)
    gen_imgs = generator(z)
    save_image(gen_imgs, os.path.join(OUTPUT_PATH, f"renaissance_text_{epoch:03d}.png"), 
               nrow=n_row, normalize=True)
    
    # Save individual images
    for i in range(n_row * n_col):
        img = gen_imgs[i].detach().cpu().squeeze().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(OUTPUT_PATH, f"sample_epoch_{epoch:03d}_{i+1}.jpg"))

# Part 4: Function to generate the required 5 pages from Spanish text
def generate_specific_pages(generator, text_file="spanish_text.txt", num_pages=5):
    # Load Spanish text
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except:
        # Sample text if file not available
        text_content = """
        En un lugar de la Mancha, de cuyo nombre no quiero acordarme, 
        no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, 
        adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, 
        salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, 
        algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.
        """
    
    paragraphs = text_content.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Create baseline pages
    pages = []
    for page_num in range(num_pages):
        # Create a blank page
        img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=245)
        draw = ImageDraw.Draw(img)
        
        # Add text from the file
        try:
            font = ImageFont.truetype("EB_Garamond/EBGaramond-Regular.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        y_position = 50
        start_idx = page_num * 7  # Approximately 7 paragraphs per page
        
        for i in range(min(7, len(paragraphs) - start_idx)):
            text = paragraphs[start_idx + i] if start_idx + i < len(paragraphs) else ""
            if not text:
                continue
                
            # Text wrapping for long paragraphs
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                text_width = draw.textlength(test_line, font=font)
                
                if text_width <= IMAGE_SIZE - 60:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                draw.text((30, y_position), line, font=font, fill=0)
                y_position += 35
            
            y_position += 10  # Add space between paragraphs
        
        img_array = np.array(img) / 255.0
        pages.append(img_array)
    
    # Use the GAN to add Renaissance-style degradation
    enhanced_pages = []
    for i, page in enumerate(pages):
        # Create tensor from page image
        page_tensor = torch.FloatTensor(page).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Generate noise and create degradation pattern
        z = torch.randn(1, NOISE_DIM, device=DEVICE, dtype=torch.float)
        gen_noise = generator(z)
        
        # Mix base content with generated degradation
        # Perform blending directly in tensor space
        combined = page_tensor * 0.7 + gen_noise * 0.3
        combined_np = combined.squeeze().cpu().numpy()
        
        # Additional post-processing
        enhanced_page = apply_renaissance_augmentation(combined_np)
        
        # Save the enhanced page
        img = Image.fromarray((enhanced_page * 255).astype(np.uint8))
        img.save(os.path.join(OUTPUT_PATH, f"renaissance_page_{i+1}.jpg"))
        enhanced_pages.append(enhanced_page)
        
        # Print progress
        print(f"Generated Renaissance page {i+1}/{num_pages}")
    
    return enhanced_pages

# Part 5: Evaluation metrics
def evaluate_model(generator, real_samples):
    """Evaluate the model using various metrics"""
    
    # Prepare real samples as tensors
    if not torch.is_tensor(real_samples):
        real_tensors = []
        for img in real_samples:
            tensor = torch.FloatTensor(img).unsqueeze(0)
            real_tensors.append(tensor)
        real_samples_tensor = torch.cat(real_tensors, 0)
    else:
        real_samples_tensor = real_samples
    
    real_samples_tensor = real_samples_tensor.to(DEVICE)
    
    # Generate samples
    batch_size = real_samples_tensor.size(0)
    z = torch.randn(batch_size, NOISE_DIM, device=DEVICE, dtype=torch.float)
    with torch.no_grad():
        generated_samples = generator(z)
    
    # Convert to numpy for evaluation
    real_np = [sample.squeeze().cpu().numpy() if torch.is_tensor(sample) else sample for sample in real_samples]
    gen_np = [img.squeeze().cpu().numpy() for img in generated_samples]
    
    # 1. Structural Similarity Index (SSIM)
    ssim_scores = []
    for i in range(len(real_np)):
        real_img = (real_np[i] * 255).astype(np.uint8)
        gen_img = (gen_np[i] * 255).astype(np.uint8)
        
        # Calculate SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            score = ssim(real_img, gen_img)
        except:
            # Fallback if skimage not available
            score = cv2.matchTemplate(real_img, gen_img, cv2.TM_CCORR_NORMED)[0][0]
        ssim_scores.append(score)
    
    # 2. Histogram comparison (for texture similarity)
    hist_scores = []
    for i in range(len(real_np)):
        real_img = (real_np[i] * 255).astype(np.uint8)
        gen_img = (gen_np[i] * 255).astype(np.uint8)
        
        real_hist = cv2.calcHist([real_img], [0], None, [256], [0, 256])
        gen_hist = cv2.calcHist([gen_img], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(real_hist, real_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(gen_hist, gen_hist, 0, 1, cv2.NORM_MINMAX)
        
        # Compare histograms
        score = cv2.compareHist(real_hist, gen_hist, cv2.HISTCMP_CORREL)
        hist_scores.append(score)
    
    # 3. Textural variance analysis (for degradation effects)
    variance_real = [np.var(img) for img in real_np]
    variance_gen = [np.var(img) for img in gen_np]
    
    variance_similarity = 1 - abs(np.mean(variance_real) - np.mean(variance_gen)) / max(np.mean(variance_real), np.mean(variance_gen))
    
    # 4. Custom Renaissance Impression Score
    # Analyzing specific Renaissance printing artifacts
    
    # Ink distribution unevenness
    ink_unevenness_real = [np.std(img) for img in real_np]
    ink_unevenness_gen = [np.std(img) for img in gen_np]
    
    ink_similarity = 1 - abs(np.mean(ink_unevenness_real) - np.mean(ink_unevenness_gen)) / max(np.mean(ink_unevenness_real), np.mean(ink_unevenness_gen))
    
    # Edge roughness analysis
    def edge_roughness(img):
        # Apply Canny edge detection
        img = (img * 255).astype(np.uint8)
        edges = cv2.Canny(img, 100, 200)
        return np.sum(edges) / (img.shape[0] * img.shape[1])
    
    edge_real = [edge_roughness(img) for img in real_np]
    edge_gen = [edge_roughness(img) for img in gen_np]
    
    edge_similarity = 1 - abs(np.mean(edge_real) - np.mean(edge_gen)) / max(np.mean(edge_real), np.mean(edge_gen))
    
    # Aggregate metrics into a Renaissance Authenticity Score
    renaissance_score = (np.mean(ssim_scores) * 0.3 + 
                         np.mean(hist_scores) * 0.2 + 
                         variance_similarity * 0.2 + 
                         ink_similarity * 0.15 + 
                         edge_similarity * 0.15)
    
    results = {
        'ssim_score': float(np.mean(ssim_scores)),
        'histogram_similarity': float(np.mean(hist_scores)),
        'variance_similarity': float(variance_similarity),
        'ink_distribution_similarity': float(ink_similarity),
        'edge_roughness_similarity': float(edge_similarity),
        'renaissance_authenticity_score': float(renaissance_score)
    }
    
    return results

# Main execution flow
def main():
    print("Starting Renaissance Text GAN project (PyTorch version)")
    
    # 1. Data preparation
    print("Generating baseline dataset...")
    image_data = generate_baseline_data(num_samples=200)
    
    # Create PyTorch dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = RenaissanceTextDataset(image_data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Build models
    print("Building GAN models...")
    generator = Generator(NOISE_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters() if p.requires_grad)}")
    
    # 3. Train models
    print("Training GAN...")
    generator, discriminator = train_gan(generator, discriminator, dataloader, epochs=EPOCHS)
    
    # 4. Generate specific pages
    print("Generating 5 pages of Renaissance-style Spanish text...")
    enhanced_pages = generate_specific_pages(generator, num_pages=5)
    
    # 5. Evaluate model
    print("Evaluating model performance...")
    evaluation_results = evaluate_model(generator, image_data[:5])
    
    # 6. Print evaluation results
    print("\nModel Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nProject completed successfully!")
    print(f"Output images saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()