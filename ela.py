from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def convert_to_ela(image_path, quality=90, target_size=(128, 128)):
    # 1. Standardize the image (converts PNG/WebP to JPEG compatible RGB)
    original = Image.open(image_path).convert('RGB')
    
    # 2. Use a lower quality (85) to force artifacts if 90 is too subtle
    temp_path = "temp_processing.jpg"
    original.save(temp_path, 'JPEG', quality=85)
    compressed = Image.open(temp_path)
    
    # 3. Calculate difference
    ela = ImageChops.difference(original, compressed)
    
    # 4. ENHANCED SCALING (The Fix)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    # If the image is "too perfect" (like a PNG), force a baseline diff
    if max_diff < 1:
        max_diff = 10 
        
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    
    # 5. Prepare for Model
    ela = ela.resize(target_size)
    return np.array(ela).astype('float32') / 255.0
