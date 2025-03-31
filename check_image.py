from PIL import Image
import os
import sys

def check_images(folder_path):
    """Check image dimensions and statistics for folder"""
    try:
        # Get list of image files
        files = [f for f in os.listdir(folder_path) 
                if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Total images found: {len(files)}")
        
        # Check dimensions of sample images
        sample_size = min(10, len(files))
        print(f"\nSample of {sample_size} images:")
        
        dimensions = {}
        for i, f in enumerate(files[:sample_size]):
            try:
                img_path = os.path.join(folder_path, f)
                img = Image.open(img_path)
                size = img.size
                print(f"{i+1}. {f}: {size}")
                
                # Track dimension frequencies
                dim_key = f"{size[0]}x{size[1]}"
                if dim_key in dimensions:
                    dimensions[dim_key] += 1
                else:
                    dimensions[dim_key] = 1
            except Exception as e:
                print(f"Error with {f}: {e}")
        
        # Count all dimensions
        print("\nCounting all image dimensions...")
        all_dimensions = {}
        for f in files:
            try:
                img_path = os.path.join(folder_path, f)
                size = Image.open(img_path).size
                dim_key = f"{size[0]}x{size[1]}"
                if dim_key in all_dimensions:
                    all_dimensions[dim_key] += 1
                else:
                    all_dimensions[dim_key] = 1
            except:
                pass
        
        # Display dimension statistics
        print("\nImage dimension statistics:")
        for dim, count in sorted(all_dimensions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(files)) * 100
            print(f"{dim}: {count} images ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = "../Gsoc/Rodrigo corpus 1.0.0.tar/Rodrigo corpus 1.0.0/images"
    
    check_images(folder_path) 