import os
import random
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import glob
import cv2

# Configuration
OUTPUT_DIR = "synthetic_renaissance_data"
NUM_SAMPLES = 500
IMAGE_SIZE = 128  # Keep consistent with the GAN architecture

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Renaissance Text Samples (Spanish)
RENAISSANCE_TEXTS = [
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
    "La edad de este caballero andaba entre los cincuenta y cinco años.",
    "Era de complexión recia, seco de carnes, enjuto de rostro,",
    "gran madrugador y amigo de la caza.",
    "Quieren decir que tenía el sobrenombre de «Quijada», o «Quesada»,",
    "que en esto hay alguna diferencia en los autores que deste caso escriben,",
    "aunque por conjeturas verosímiles se deja entender que se llama «Quijana»."
]

def create_paper_background(size):
    """Create a realistic parchment/paper background with aging effects"""
    # Base color (slightly off-white)
    paper_color = random.randint(220, 245)
    background = Image.new('L', size, color=paper_color)
    
    # Add random noise to simulate paper texture
    noise = np.random.normal(0, 5, size)
    noise_img = Image.fromarray((noise + 128).clip(0, 255).astype(np.uint8))
    background = Image.blend(background, noise_img, alpha=0.3)
    
    # Add age stains and marks
    for _ in range(random.randint(2, 5)):
        stain_size = random.randint(10, 40)
        stain_x = random.randint(0, size[0] - stain_size)
        stain_y = random.randint(0, size[1] - stain_size)
        stain_color = random.randint(180, 210)  # Darker than paper for stains
        stain = Image.new('L', (stain_size, stain_size), color=stain_color)
        stain = stain.filter(ImageFilter.GaussianBlur(radius=stain_size/3))
        background.paste(Image.blend(background.crop((stain_x, stain_y, 
                                                      stain_x + stain_size, 
                                                      stain_y + stain_size)), 
                                     stain, 0.5), 
                         (stain_x, stain_y))
    
    # Add paper grain with a high-frequency noise
    grain = np.random.normal(0, 2, size).astype(np.uint8)
    grain_img = Image.fromarray(grain + 128)
    background = Image.blend(background, grain_img, alpha=0.1)
    
    return background

def apply_renaissance_effects(img):
    """Apply Renaissance-era printing degradation effects"""
    # Convert to numpy for processing
    img_array = np.array(img)
    
    # 1. Ink bleeding
    # First create a mask of just the text
    _, text_mask = cv2.threshold(img_array, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate the text to simulate ink spread
    kernel_size = random.choice([2, 3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_text = cv2.dilate(text_mask, kernel, iterations=1)
    
    # Blur the dilated text for a more natural bleed
    bleed_amount = random.uniform(0.5, 1.5)
    blurred_text = cv2.GaussianBlur(dilated_text, (3, 3), bleed_amount)
    
    # Use the blurred text as a mask
    new_array = img_array.copy()
    indices = blurred_text > 50
    new_array[indices] = np.minimum(new_array[indices], 255 - blurred_text[indices])
    
    # 2. Uneven ink application
    if random.random() > 0.5:
        noise = np.random.normal(0, 10, img_array.shape) 
        text_indices = new_array < 180  # Identify text pixels
        new_array[text_indices] = np.clip(new_array[text_indices] + noise[text_indices] * 0.5, 0, 255)
    
    # 3. Faded areas
    if random.random() > 0.6:
        fade_mask = np.ones_like(new_array) * 255
        # Add a gradient fade effect in a random direction
        direction = random.choice(['left', 'right', 'top', 'bottom'])
        h, w = fade_mask.shape
        
        if direction == 'left':
            for x in range(w):
                fade_mask[:, x] = min(255, 200 + int((x / w) * 55))
        elif direction == 'right':
            for x in range(w):
                fade_mask[:, x] = min(255, 200 + int(((w-x) / w) * 55))
        elif direction == 'top':
            for y in range(h):
                fade_mask[y, :] = min(255, 200 + int((y / h) * 55))
        else:  # bottom
            for y in range(h):
                fade_mask[y, :] = min(255, 200 + int(((h-y) / h) * 55))
                
        # Apply the fade mask
        new_array = cv2.multiply(new_array.astype(np.float32), 
                                fade_mask.astype(np.float32) / 255)
        new_array = np.clip(new_array, 0, 255).astype(np.uint8)
    
    # 4. Slight rotation/misalignment
    if random.random() > 0.7:
        h, w = new_array.shape
        angle = random.uniform(-1.0, 1.0)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        new_array = cv2.warpAffine(new_array, M, (w, h), 
                                   borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(new_array)

def generate_synthetic_renaissance_text():
    """Generate a synthetic Renaissance text image"""
    # Create the base background
    img = create_paper_background((IMAGE_SIZE, IMAGE_SIZE))
    draw = ImageDraw.Draw(img)
    
    # Try to load Renaissance-style font
    try:
        font = ImageFont.truetype("EB_Garamond/EBGaramond-Regular.ttf", 12)
        emphasis_font = ImageFont.truetype("EB_Garamond/EBGaramond-Bold.ttf", 14)
    except:
        try:
            # Font files might be in a different location
            font_paths = glob.glob("**/EBGaramond*.ttf", recursive=True)
            if font_paths:
                font = ImageFont.truetype(font_paths[0], 12)
                emphasis_font = font
            else:
                font = ImageFont.load_default()
                emphasis_font = font
        except:
            font = ImageFont.load_default()
            emphasis_font = font
    
    # Choose how much text to add (line snippets, partial paragraph, or full paragraph)
    text_style = random.choice(['line', 'partial', 'paragraph'])
    
    # Position the text with a random starting position
    y_position = random.randint(10, 30)
    
    if text_style == 'line':
        # Just a single line of text (like the Rodrigo corpus samples)
        text = random.choice(RENAISSANCE_TEXTS)
        draw.text((20, y_position), text, font=font, fill=0)
    
    elif text_style == 'partial':
        # A few lines of text
        num_lines = random.randint(2, 4)
        for i in range(num_lines):
            text = random.choice(RENAISSANCE_TEXTS)
            draw.text((20, y_position), text, font=font, fill=0)
            y_position += random.randint(16, 20)
    
    else:  # Full paragraph layout
        # First line might have a large capital letter
        if random.random() > 0.6:
            first_text = random.choice(RENAISSANCE_TEXTS)
            first_letter = first_text[0]
            rest_of_line = first_text[1:]
            
            # Draw the capital letter
            draw.text((20, y_position-2), first_letter, font=emphasis_font, fill=0)
            letter_width = draw.textlength(first_letter, font=emphasis_font)
            
            # Draw the rest of the first line
            draw.text((20 + letter_width + 2, y_position), rest_of_line, font=font, fill=0)
            y_position += random.randint(16, 20)
        
        # Add more text lines
        num_lines = random.randint(3, 6)
        for i in range(num_lines):
            text = random.choice(RENAISSANCE_TEXTS)
            indent = 20 if random.random() > 0.3 else 30  # Varying indentation
            draw.text((indent, y_position), text, font=font, fill=0)
            y_position += random.randint(16, 20)
    
    # Apply Renaissance-era effects
    final_img = apply_renaissance_effects(img)
    
    return final_img

def generate_dataset():
    """Generate the entire synthetic dataset"""
    print(f"Generating {NUM_SAMPLES} synthetic Renaissance text images...")
    
    for i in range(NUM_SAMPLES):
        img = generate_synthetic_renaissance_text()
        img.save(os.path.join(OUTPUT_DIR, f"renaissance_text_{i+1:04d}.png"))
        
        if (i+1) % 50 == 0:
            print(f"Generated {i+1}/{NUM_SAMPLES} images")
    
    # Display some samples
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    sample_indices = random.sample(range(NUM_SAMPLES), 6)
    for i, idx in enumerate(sample_indices):
        sample_path = os.path.join(OUTPUT_DIR, f"renaissance_text_{idx+1:04d}.png")
        sample_img = Image.open(sample_path)
        axes[i].imshow(sample_img, cmap='gray')
        axes[i].set_title(f"Sample {idx+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "samples.png"))
    plt.close()
    
    print(f"Dataset generated successfully in {OUTPUT_DIR}")
    print(f"Sample visualization saved as {os.path.join(OUTPUT_DIR, 'samples.png')}")

if __name__ == "__main__":
    generate_dataset() 