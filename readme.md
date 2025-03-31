# Renaissance Text Generation Using GANs

![Renaissance Text Banner](synthetic_renaissance_data/samples.png)

## Overview

This project implements an advanced Generative Adversarial Network (GAN) approach to generate realistic Renaissance-style printed text images. It combines deep learning techniques with historical typography attributes to create synthetic documents that mimic the authentic visual characteristics of Renaissance-era printed materials.

## Project Features

- **Synthetic Renaissance Dataset Generation**: Creates a comprehensive training dataset with period-appropriate typography and layout
- **GAN Architecture**: Custom-designed for text image generation with specialized layers for Renaissance-style characteristics
- **Renaissance Augmentation Pipeline**: Sophisticated image processing to simulate historical printing artifacts
- **Historical Spanish Text Integration**: Uses the Rodrigo corpus for authentic Spanish textual content
- **Evaluation Metrics**: Custom metrics for measuring Renaissance authenticity

## Project Structure

```
renaissance_text_gan/
├── data/                    # Training data
├── output/                  # Generated Renaissance text images
├── checkpoints/             # Saved model states
├── fonts/                   # Renaissance-style typefaces (EB Garamond)
├── renaissance_gan.py       # Core GAN architecture and training code
├── main.py                  # Command-line interface
└── README.md                # Project documentation
```

## Technical Approach

### 1. Synthetic Data Generation

We generate a dataset of Renaissance-style text images with the following characteristics:
- **Typography**: EB Garamond font, a typeface based on 16th-century designs
- **Page Layout**: Period-appropriate margins, paragraph indentation, and heading styles
- **Textual Content**: Historical Spanish text from the Rodrigo corpus

### 2. GAN Architecture

Our system implements a custom GAN with:

**Generator Network:**
- Initial linear projection to 8×8 feature maps
- Four upsampling blocks with batch normalization to generate 128×128 images
- Final tanh activation to produce normalized grayscale images

**Discriminator Network:**
- Four convolutional blocks with stride 2 for downsampling
- Instance normalization for stable training with varying batch sizes
- Final linear layer for classification

### 3. Renaissance Augmentation Pipeline

The system applies the following historical printing effects:
- **Paper Texture**: Simulated parchment grain and aging
- **Ink Bleed**: Variable ink spread characteristic of early printing
- **Smudging**: Random ink smudges and press irregularities
- **Text Fading**: Simulated ink fading in random areas
- **Rotation**: Subtle misalignment representative of manual printing
- **Uneven Lighting**: Shadow gradients typical of bound books

## Results

Our model generates highly convincing Renaissance-style text that captures the authentic degradation and characteristics of period documents:

![image](https://github.com/user-attachments/assets/63f97eaa-e22f-4cee-b3a1-cbc94af9da8a)

![image](https://github.com/user-attachments/assets/1715200f-ff3a-492d-8ade-19e1ae2857a6)

![Sample 3](synthetic_renaissance_data/renaissance_text_0462.png)
![Sample 4](synthetic_renaissance_data/renaissance_text_0473.png)
![Sample 5](synthetic_renaissance_data/renaissance_text_0498.png)

## Evaluation

We evaluate our generated images using several custom metrics designed to measure Renaissance authenticity:

1. **Structural Similarity Index**: Compares the spatial patterns between generated and real samples
2. **Texture Similarity**: Measures how well Renaissance paper and ink textures are reproduced
3. **Ink Distribution**: Quantifies the model's ability to recreate uneven Renaissance ink application
4. **Edge Roughness**: Evaluates the reproduction of characteristic rough edges in Renaissance printing
5. **Ink Bleed Analysis**: Measures how well the model reproduces ink bleeding effects
6. **Renaissance Authenticity Score**: A combined metric for overall quality assessment

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/renaissance-text-gan.git
cd renaissance-text-gan

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train using the synthetic dataset
python main.py --mode train --dataset synthetic --epochs 50 --batch_size 16

# Train using the Rodrigo corpus (if available)
python main.py --mode train --dataset rodrigo --epochs 50 --batch_size 16
```

### Generating Renaissance-Style Text

```bash
# Generate 5 pages of Renaissance-style text
python main.py --mode generate --num_pages 5

# Generate using a specific model
python main.py --mode generate --model checkpoints/generator_final.pt --num_pages 10
```

## Experimental Results

Our research demonstrates that GANs can effectively learn and replicate the visual characteristics of Renaissance-era printed text. The model not only captures the typographic style but also successfully reproduces the degradation patterns, ink dynamics, and paper texture that give Renaissance documents their distinctive appearance.

The system achieves a Renaissance Authenticity Score of 0.74, indicating a high level of fidelity to historical source materials.

## Future Work

- Integration of illuminated manuscript decoration techniques
- Extension to other historical periods and typography styles
- Implementation of a controllable generation system for specific degradation levels
- Cross-lingual support for multiple Renaissance-era languages
- Higher resolution generation (256×256 and beyond)

