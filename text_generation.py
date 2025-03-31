import os
import random
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps
import cv2
from skimage.metrics import structural_similarity as ssim
from renaissance_gan import apply_renaissance_augmentation, Generator, DEVICE, NOISE_DIM, OUTPUT_PATH, IMAGE_SIZE

# Function to generate the required 5 pages from Spanish text
def generate_specific_pages(generator, text_content, num_pages=5):
    """Generate specific pages of Renaissance-style Spanish text with degradation effects"""
    # Break text content into paragraphs
    paragraphs = text_content.split('.')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Try to load a Renaissance-appropriate font
    try:
        font = ImageFont.truetype("EB_Garamond/EBGaramond-Regular.ttf", 24)
    except IOError:
        try:
            font = ImageFont.truetype("fonts/EBGaramond-Regular.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
    
    print(f"Generating {num_pages} pages of Renaissance-style Spanish text...")
    
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
            header = "HISTORIA DE ESPAÑA"
            try:
                header_width = draw.textlength(header, font=font)
            except AttributeError:
                header_width = font.getsize(header)[0]
            
            # Center the header
            draw.text(((IMAGE_SIZE - header_width) // 2, 30), header, font=font, fill=0)
            y_position = 80
        else:
            y_position = 50
        
        # Add text from the corpus
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

# Evaluation metrics
def evaluate_model(generator, real_samples):
    """Evaluate the model using various metrics specific to Renaissance text degradation"""
    
    # Prepare real samples as tensors
    if not torch.is_tensor(real_samples[0]):
        real_tensors = []
        for img in real_samples:
            tensor = torch.FloatTensor(img).unsqueeze(0)
            real_tensors.append(tensor)
        real_samples_tensor = torch.cat(real_tensors, 0)
    else:
        real_samples_tensor = torch.cat(real_samples, 0)
    
    real_samples_tensor = real_samples_tensor.to(DEVICE)
    
    # Generate samples
    batch_size = real_samples_tensor.size(0)
    z = torch.randn(batch_size, NOISE_DIM, device=DEVICE, dtype=torch.float)
    with torch.no_grad():
        generated_samples = generator(z)
    
    # Convert to numpy for evaluation
    real_np = [sample.squeeze().cpu().numpy() for sample in real_samples_tensor]
    gen_np = [img.squeeze().cpu().numpy() for img in generated_samples]
    
    # 1. Structural Similarity Index (SSIM)
    ssim_scores = []
    for i in range(len(real_np)):
        real_img = (real_np[i] * 255).astype(np.uint8)
        gen_img = (gen_np[i] * 255).astype(np.uint8)
        
        # Calculate SSIM
        try:
            score = ssim(real_img, gen_img)
        except:
            # Fallback if ssim is not available
            score = np.mean(np.abs(real_img - gen_img)) / 255.0
            score = 1.0 - score  # Convert to similarity (higher is better)
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
    
    # 4. Renaissance-specific metrics
    
    # 4.1 Ink distribution unevenness
    ink_unevenness_real = [np.std(img) for img in real_np]
    ink_unevenness_gen = [np.std(img) for img in gen_np]
    
    ink_similarity = 1 - abs(np.mean(ink_unevenness_real) - np.mean(ink_unevenness_gen)) / max(np.mean(ink_unevenness_real), np.mean(ink_unevenness_gen))
    
    # 4.2 Edge roughness (typical of Renaissance printing)
    def edge_roughness(img):
        # Apply Canny edge detection
        img = (img * 255).astype(np.uint8)
        edges = cv2.Canny(img, 100, 200)
        return np.sum(edges) / (img.shape[0] * img.shape[1])
    
    edge_real = [edge_roughness(img) for img in real_np]
    edge_gen = [edge_roughness(img) for img in gen_np]
    
    edge_similarity = 1 - abs(np.mean(edge_real) - np.mean(edge_gen)) / max(np.mean(edge_real), np.mean(edge_gen))
    
    # 4.3 Ink bleed analysis
    def measure_ink_bleed(img):
        # Apply dilation to estimate ink bleed
        img_uint8 = (img * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(img_uint8, kernel, iterations=1)
        bleed = np.sum(dilated - img_uint8) / (img.shape[0] * img.shape[1])
        return bleed
    
    bleed_real = [measure_ink_bleed(img) for img in real_np]
    bleed_gen = [measure_ink_bleed(img) for img in gen_np]
    
    bleed_similarity = 1 - abs(np.mean(bleed_real) - np.mean(bleed_gen)) / max(np.mean(bleed_real), np.mean(bleed_gen))
    
    # 5. Combined Renaissance Authenticity Score
    renaissance_score = (
        np.mean(ssim_scores) * 0.25 +        # Visual similarity
        np.mean(hist_scores) * 0.15 +        # Texture similarity
        variance_similarity * 0.15 +         # Overall variance matching
        ink_similarity * 0.15 +              # Ink distribution matching
        edge_similarity * 0.15 +             # Edge characteristics
        bleed_similarity * 0.15              # Ink bleed effects
    )
    
    results = {
        'ssim_score': np.mean(ssim_scores),
        'histogram_similarity': np.mean(hist_scores),
        'variance_similarity': variance_similarity,
        'ink_distribution_similarity': ink_similarity,
        'edge_roughness_similarity': edge_similarity,
        'ink_bleed_similarity': bleed_similarity,
        'renaissance_authenticity_score': renaissance_score
    }
    
    return results

def print_evaluation_report(results):
    """Print a formatted evaluation report for the generated Renaissance text"""
    print("\n" + "="*50)
    print("RENAISSANCE TEXT GENERATION EVALUATION REPORT")
    print("="*50)
    
    print("\nQUANTITATIVE METRICS:")
    print(f"1. Structural Similarity (SSIM): {results['ssim_score']:.4f}/1.00")
    print(f"   - Measures how visually similar the generated images are to real historical texts")
    
    print(f"\n2. Texture Similarity: {results['histogram_similarity']:.4f}/1.00")
    print(f"   - Evaluates how well the model reproduces Renaissance paper and ink textures")
    
    print(f"\n3. Variance Similarity: {results['variance_similarity']:.4f}/1.00")
    print(f"   - Quantifies how well the model captures overall light/dark distribution")
    
    print(f"\n4. Ink Distribution Similarity: {results['ink_distribution_similarity']:.4f}/1.00")
    print(f"   - Measures the model's ability to reproduce uneven Renaissance ink application")
    
    print(f"\n5. Edge Roughness Similarity: {results['edge_roughness_similarity']:.4f}/1.00")
    print(f"   - Evaluates the reproduction of characteristic rough edges in Renaissance printing")
    
    print(f"\n6. Ink Bleed Similarity: {results['ink_bleed_similarity']:.4f}/1.00")
    print(f"   - Measures how well the model reproduces ink bleeding effects")
    
    print("\n" + "-"*50)
    print(f"RENAISSANCE AUTHENTICITY SCORE: {results['renaissance_authenticity_score']:.4f}/1.00")
    print("-"*50)
    
    # Interpretation
    score = results['renaissance_authenticity_score']
    if score >= 0.85:
        quality = "Excellent"
    elif score >= 0.75:
        quality = "Very Good"
    elif score >= 0.65:
        quality = "Good"
    elif score >= 0.5:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"\nOVERALL QUALITY: {quality}")
    print("\nThe model's ability to create authentic Renaissance-style text with")
    print("appropriate degradation effects (ink bleed, smudging, faded text) is")
    print(f"considered {quality.upper()} based on quantitative evaluation metrics.")
    
    print("\n" + "="*50) 