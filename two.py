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
DATASET_PATH = r"C:\Users\Ismail Qayyum\Desktop\Gsoc\Rodrigo corpus 1.0.0.tar\Rodrigo corpus 1.0.0\images"
OUTPUT_PATH = "out"
CHECKPOINT_PATH = "checkpoints/renaissance_text/"

# Ensure directories exist
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Part 1: Dataset preparation and augmentation
class RenaissanceTextDataset(Dataset):
    def __init__(self, images_folder=None, images_array=None, transform=None):
        """
        Dataset class that can handle either:
        1. A folder path containing image files
        2. A numpy array of images
        """
        self.transform = transform
        
        if images_array is not None:
            self.images = images_array
            self.from_folder = False
        elif images_folder is not None:
            self.images_folder = images_folder
            self.image_files = [f for f in os.listdir(images_folder) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.from_folder = True
        else:
            raise ValueError("Either images_folder or images_array must be provided")

    def __len__(self):
        if self.from_folder:
            return len(self.image_files)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.from_folder:
            img_path = os.path.join(self.images_folder, self.image_files[idx])
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            image_array = np.array(image) / 255.0
        else:
            image_array = self.images[idx]
            # Convert to PIL Image for transformations if needed
            image = Image.fromarray((image_array * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.ToTensor()(image)
            
        return image

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
        bleed_severity = random.randint(2, 4)
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
        img = Image.blend(img, img.filter(ImageFilter.GaussianBlur(1)), 
                          alpha=ImageOps.invert(smudge_mask).point(lambda p: p/255))
    
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
        img = Image.blend(paper_texture, img, alpha=ImageOps.invert(fade_mask).point(lambda p: p/255))
    
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
    
    # 7. Stains and marks
    if random.random() > 0.7:
        stain_img = img.copy()
        stain_draw = ImageDraw.Draw(stain_img)
        
        # Add a few random stains/marks
        for _ in range(random.randint(1, 3)):
            x = random.randint(0, img.width)
            y = random.randint(0, img.height)
            radius = random.randint(5, 30)
            stain_color = random.randint(180, 220)
            stain_draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=stain_color)
        
        # Blur the stains for a more natural look
        stain_img = stain_img.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Blend with the original image
        img = Image.blend(img, stain_img, alpha=random.uniform(0.1, 0.3))
    
    # 8. Bleed-through from reverse side (common in old books)
    if random.random() > 0.6:
        bleed_through = Image.new('L', img.size, color=250)
        bleed_draw = ImageDraw.Draw(bleed_through)
        
        # Add random text-like shapes to simulate text from reverse side
        for _ in range(random.randint(3, 7)):
            x = random.randint(0, img.width - 100)
            y = random.randint(0, img.height - 20)
            length = random.randint(50, 150)
            bleed_draw.line([(x, y), (x+length, y)], fill=200, width=random.randint(2, 5))
        
        # Blur significantly to make it appear like it's bleeding through
        bleed_through = bleed_through.filter(ImageFilter.GaussianBlur(radius=8))
        
        # Apply bleed-through effect subtly
        img = Image.blend(img, bleed_through, alpha=random.uniform(0.05, 0.15))
    
    # Convert back to numpy array and normalize
    img_array = np.array(img) / 255.0
    return img_array

# Generate synthetic Renaissance text data
def generate_baseline_data(num_samples=200):
    """Generate synthetic baseline data with Renaissance-style Spanish text"""
    # Load a Renaissance-appropriate font
    try:
        # Try to load a period-appropriate font (EB Garamond is a good choice for Renaissance text)
        font = ImageFont.truetype("EB_Garamond/EBGaramond-Regular.ttf", 24)
    except IOError:
        try:
            # Try alternate font locations
            font = ImageFont.truetype("fonts/EBGaramond-Regular.ttf", 24)
        except IOError:
            # Fallback to default if Renaissance font not available
            font = ImageFont.load_default()
    
    # Sample Spanish text from the 17th century (Don Quixote)
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
        "y los días de entresemana se honraba con su vellorí de lo más fino.",
        "Tenía en su casa una ama que pasaba de los cuarenta,",
        "y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza,",
        "que así ensillaba el rocín como tomaba la podadera.",
        "Frisaba la edad de nuestro hidalgo con los cincuenta años."
    ]
    
    images = []
    for i in range(num_samples):
        # Create blank parchment-colored image
        base_color = random.randint(235, 245)
        img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=base_color)
        draw = ImageDraw.Draw(img)
        
        # Randomly select and position text in a layout typical of Renaissance printed books
        y_position = random.randint(40, 60)  # Start position varies slightly
        
        # Add a decorative header or first letter (common in Renaissance books)
        if random.random() > 0.7:
            first_letter_size = random.randint(36, 48)
            try:
                header_font = ImageFont.truetype("EB_Garamond/EBGaramond-Bold.ttf", first_letter_size)
            except:
                header_font = font
            
            # Draw a large first letter
            first_letter = random.choice("ABCDEFGHILMNOPQRST")
            draw.text((30, y_position-10), first_letter, font=header_font, fill=0)
            
            # Adjust starting position for text that follows the large first letter
            x_offset = 30 + header_font.getsize(first_letter)[0] + 5
            y_position += 10
        else:
            x_offset = 30
        
        # Add paragraphs of text
        for _ in range(random.randint(4, 8)):
            text = random.choice(sample_texts)
            
            # Text wrapping for long paragraphs
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                try:
                    text_width = draw.textlength(test_line, font=font)
                except AttributeError:
                    # For older PIL versions
                    text_width = font.getsize(test_line)[0]
                
                if text_width <= IMAGE_SIZE - 60:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw each line
            line_y = y_position
            for j, line in enumerate(lines):
                x_pos = x_offset if j == 0 and random.random() > 0.8 else 30
                draw.text((x_pos, line_y), line, font=font, fill=0)
                line_y += 35
            
            y_position = line_y + random.randint(10, 20)  # Space between paragraphs
            x_offset = 30  # Reset x_offset for new paragraphs
        
        # Add page numbers (common in Renaissance books)
        if random.random() > 0.5:
            page_num = str(random.randint(1, 200))
            try:
                text_width = draw.textlength(page_num, font=font)
            except AttributeError:
                text_width = font.getsize(page_num)[0]
            
            # Center the page number at the bottom
            draw.text(((IMAGE_SIZE - text_width) // 2, IMAGE_SIZE - 40), 
                      page_num, font=font, fill=0)
        
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
    with torch.no_grad():
        gen_imgs = generator(z)
    
    save_image(gen_imgs, os.path.join(OUTPUT_PATH, f"renaissance_text_{epoch:03d}.png"), 
               nrow=n_row, normalize=True)
    
    # Save individual images with additional Renaissance post-processing
    for i in range(n_row * n_col):
        img = gen_imgs[i].detach().cpu().squeeze().numpy()
        # Apply additional Renaissance style degradation
        img = apply_renaissance_augmentation(img)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(OUTPUT_PATH, f"sample_epoch_{epoch:03d}_{i+1}.jpg"))

# Part 4: Function to generate the required 5 pages from Spanish text
def generate_specific_pages(generator, text_file="spanish_text.txt", num_pages=5):
    """Generate specific pages of Renaissance-style Spanish text with degradation effects"""
    # Load Spanish text
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except:
        # Sample text if file not available (from Don Quixote)
        text_content = """
        En un lugar de la Mancha, de cuyo nombre no quiero acordarme, 
        no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, 
        adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, 
        salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, 
        algún palomino de añadidura los domingos, consumían las tres partes de su hacienda.
        El resto della concluían sayo de velarte, calzas de velludo para las fiestas,
        con sus pantuflos de lo mesmo, y los días de entresemana se honraba con su vellorí de lo más fino.
        Tenía en su casa una ama que pasaba de los cuarenta, y una sobrina que no llegaba a los veinte,
        y un mozo de campo y plaza, que así ensillaba el rocín como tomaba la podadera.
        Frisaba la edad de nuestro hidalgo con los cincuenta años. Era de complexión recia,
        seco de carnes, enjuto de rostro, gran madrugador y amigo de la caza.
        """
    
    paragraphs = text_content.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Try to load a Renaissance-appropriate font
    try:
        font = ImageFont.truetype("EB_Garamond/EBGaramond-Regular.ttf", 24)
    except IOError:
        try:
            font = ImageFont.truetype("fonts/EBGaramond-Regular.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
    
    # Create baseline pages
    pages = []
    for page_num in range(num_pages):
        # Create a blank parchment-colored page
        base_color = random.randint(235, 245)
        img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=base_color)
        draw = ImageDraw.Draw(img)
        
        # Add decorative elements typical of Renaissance books
        if page_num == 0 and random.random() > 0.5:
            # First page might have a decorative header
            header = "CAPÍTULO PRIMERO"
            try:
                header_width = draw.textlength(header, font=font)
            except AttributeError:
                header_width = font.getsize(header)[0]
            
            # Center the header
            draw.text(((IMAGE_SIZE - header_width) // 2, 30), header, font=font, fill=0)
            y_position = 80
        else:
            y_position = 50
        
        # Add text from the file
        start_idx = page_num * 5  # Fewer paragraphs per page for readability
        
        for i in range(min(5, len(paragraphs) - start_idx)):
            text = paragraphs[start_idx + i] if start_idx + i < len(paragraphs) else ""
            if not text:
                continue
                
            # Text wrapping for long paragraphs
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                try:
                    text_width = draw.textlength(test_line, font=font)
                except AttributeError:
                    text_width = font.getsize(test_line)[0]
                
                if text_width <= IMAGE_SIZE - 60:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Add paragraph indentation (common in Renaissance books)
            first_line_indent = 50 if random.random() > 0.5 else 30
            
            for j, line in enumerate(lines):
                x_pos = first_line_indent if j == 0 else 30
                draw.text((x_pos, y_position), line, font=font, fill=0)
                y_position += 35
            
            y_position += 15  # Space between paragraphs
        
        # Add page number at the bottom
        page_number = str(page_num + 1)
        try:
            text_width = draw.textlength(page_number, font=font)
        except AttributeError:
            text_width = font.getsize(page_number)[0]
        
        draw.text(((IMAGE_SIZE - text_width) // 2, IMAGE_SIZE - 40), 
                  page_number, font=font, fill=0)
        
        # Convert to array
        img_array = np.array(img) / 255.0
        pages.append(img_array)
    
    # Use the GAN to add Renaissance-style degradation
    enhanced_pages = []
    for i, page in enumerate(pages):
        # Create tensor from page image
        page_tensor = torch.FloatTensor(page).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Generate noise and create degradation pattern
        z = torch.randn(1, NOISE_DIM, device=DEVICE, dtype=torch.float)
        with torch.no_grad():
            gen_noise = generator(z)
        
        # Mix base content with generated degradation pattern
        alpha = random.uniform(0.2, 0.4)  # Vary the amount of degradation
        combined = page_tensor * (1 - alpha) + gen_noise * alpha
        combined_np = combined.squeeze().cpu().numpy()
        
        # Additional post-processing with Renaissance-specific effects
        enhanced_page = apply_renaissance_augmentation(combined_np)
        
        # Save the enhanced page
        img = Image.fromarray((enhanced_page * 255).astype(np.uint8))
        img.save(os.path.join(OUTPUT_PATH, f"renaissance_page_{i+1}.jpg"))
        enhanced_pages.append(enhanced_page)
        
        # Print progress
        print(f"Generated Renaissance page {i+1}/{num_pages}")
    
    return enhanced_pages

# # Part 5: Evaluation metrics
# def evaluate_model(generator, real_samples):
#     """Evaluate the model using various metrics specific to Renaissance text degradation"""
    
#     # Prepare real samples as tensors
#     if not torch.is_tensor(real_samples):
#         real_tensors = []
#         for img in real_samples:
#             tensor = torch.FloatTensor(img).unsqueeze(0).unsque