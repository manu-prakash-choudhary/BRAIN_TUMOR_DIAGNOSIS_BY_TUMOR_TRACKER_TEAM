import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import random

def generate_mri_sample(output_path, tumor_type=None, add_noise=True, img_size=(224, 224)):
    """
    Generate a synthetic MRI brain scan image
    
    Parameters:
    -----------
    output_path : str
        Path to save the generated image
    tumor_type : str, optional
        Type of tumor to generate ('glioma', 'meningioma', 'pituitary', or None for no tumor)
    add_noise : bool
        Whether to add noise to the image
    img_size : tuple
        Image size (height, width)
        
    Returns:
    --------
    numpy.ndarray
        Generated MRI image
    """
    # Create a blank grayscale image
    height, width = img_size
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create a PIL Image for easier drawing
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw brain outline (ellipse)
    brain_width = int(width * 0.7)
    brain_height = int(height * 0.8)
    center_x = width // 2
    center_y = height // 2
    
    # Draw the elliptical brain
    draw.ellipse(
        [(center_x - brain_width//2, center_y - brain_height//2),
         (center_x + brain_width//2, center_y + brain_height//2)],
        fill=180
    )
    
    # Add ventricles (darker regions in center)
    ventricle_width = int(brain_width * 0.15)
    ventricle_height = int(brain_height * 0.3)
    ventricle_spacing = int(brain_width * 0.15)
    
    draw.ellipse(
        [(center_x - ventricle_spacing//2 - ventricle_width//2, center_y - ventricle_height//2),
         (center_x - ventricle_spacing//2 + ventricle_width//2, center_y + ventricle_height//2)],
        fill=50
    )
    
    draw.ellipse(
        [(center_x + ventricle_spacing//2 - ventricle_width//2, center_y - ventricle_height//2),
         (center_x + ventricle_spacing//2 + ventricle_width//2, center_y + ventricle_height//2)],
        fill=50
    )
    
    # Add cerebellum (bottom part of brain)
    cerebellum_width = int(brain_width * 0.7)
    cerebellum_height = int(brain_height * 0.2)
    
    draw.ellipse(
        [(center_x - cerebellum_width//2, center_y + brain_height//2 - cerebellum_height//2),
         (center_x + cerebellum_width//2, center_y + brain_height//2 + cerebellum_height//2)],
        fill=150
    )
    
    # Add a tumor if specified
    if tumor_type:
        # Different tumor types have different characteristics
        if tumor_type.lower() == 'glioma':
            # Gliomas can be irregular shaped and often in cerebral hemispheres
            tumor_x = center_x + random.randint(-brain_width//3, brain_width//3)
            tumor_y = center_y + random.randint(-brain_height//3, brain_height//3)
            tumor_size = random.randint(int(brain_width * 0.1), int(brain_width * 0.2))
            tumor_brightness = 220  # Brighter than brain tissue
            
            # Draw irregular shape
            points = []
            for i in range(8):
                angle = i * (2 * np.pi / 8)
                radius = tumor_size * (0.7 + 0.3 * random.random())
                x = tumor_x + int(radius * np.cos(angle))
                y = tumor_y + int(radius * np.sin(angle))
                points.append((x, y))
            
            draw.polygon(points, fill=tumor_brightness)
            
        elif tumor_type.lower() == 'meningioma':
            # Meningiomas are often well-defined and near the skull
            angle = random.random() * 2 * np.pi
            distance = int(brain_width * 0.4)
            tumor_x = center_x + int(distance * np.cos(angle))
            tumor_y = center_y + int(distance * np.sin(angle))
            tumor_size = random.randint(int(brain_width * 0.05), int(brain_width * 0.15))
            
            draw.ellipse(
                [(tumor_x - tumor_size//2, tumor_y - tumor_size//2),
                 (tumor_x + tumor_size//2, tumor_y + tumor_size//2)],
                fill=200
            )
            
        elif tumor_type.lower() == 'pituitary':
            # Pituitary tumors are centered at the bottom middle of the brain
            tumor_x = center_x
            tumor_y = center_y + int(brain_height * 0.2)
            tumor_width = random.randint(int(brain_width * 0.1), int(brain_width * 0.15))
            tumor_height = random.randint(int(brain_height * 0.1), int(brain_height * 0.15))
            
            draw.ellipse(
                [(tumor_x - tumor_width//2, tumor_y - tumor_height//2),
                 (tumor_x + tumor_width//2, tumor_y + tumor_height//2)],
                fill=210
            )
    
    # Convert back to numpy array
    image = np.array(pil_img)
    
    # Apply Gaussian blur to smooth the image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, 15, (height, width)).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    # Add scan lines for more realism
    for i in range(0, height, 4):
        brightness_change = random.randint(-10, 10)
        image[i:i+1, :] = np.clip(image[i:i+1, :].astype(np.int16) + brightness_change, 0, 255).astype(np.uint8)
    
    # Save the image if output path is provided
    if output_path:
        cv2.imwrite(output_path, image)
    
    return image

def generate_sample_dataset(output_dir, samples_per_class=2):
    """
    Generate a sample dataset of synthetic MRI images
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the generated images
    samples_per_class : int
        Number of samples to generate per class
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define classes
    classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    
    for class_name in classes:
        for i in range(samples_per_class):
            tumor_type = None if class_name == 'no_tumor' else class_name
            filename = f"{class_name}_{i+1}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            generate_mri_sample(output_path, tumor_type=tumor_type)
            print(f"Generated {output_path}")

def simulate_mri_modalities(base_image_path, output_dir):
    """
    Generate different MRI modalities from a base image
    
    Parameters:
    -----------
    base_image_path : str
        Path to the base MRI image
    output_dir : str
        Directory to save the generated modality images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the base image
    base_image = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
    
    if base_image is None:
        print(f"Error: Could not load image from {base_image_path}")
        return
    
    # Get filename without extension
    base_filename = os.path.splitext(os.path.basename(base_image_path))[0]
    
    # Define modality functions
    def create_t1(img):
        # T1: Good contrast between gray and white matter
        contrast = cv2.convertScaleAbs(img, alpha=1.2, beta=-10)
        return contrast
    
    def create_t2(img):
        # T2: CSF is bright, gray and white matter less contrast
        # Invert the image to make dark areas bright
        inverted = 255 - img
        # Adjust contrast
        contrast = cv2.convertScaleAbs(inverted, alpha=0.9, beta=20)
        return contrast
    
    def create_flair(img):
        # FLAIR: Suppress CSF signal, highlight lesions
        # Apply a more complex transformation
        contrast = cv2.convertScaleAbs(img, alpha=1.3, beta=-30)
        # Add some noise and blur
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        flair = np.clip(contrast.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        flair = cv2.GaussianBlur(flair, (3, 3), 0)
        return flair
    
    def create_t1_contrast(img):
        # T1 with contrast: Enhances tumors
        # Start with T1
        t1 = create_t1(img)
        # Find bright spots (potential tumors) and enhance them
        _, binary = cv2.threshold(t1, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        # Add the enhancement to the original
        enhanced = cv2.addWeighted(t1, 0.7, dilated, 0.3, 0)
        return enhanced
    
    # Generate and save each modality
    modalities = {
        'T1': create_t1,
        'T2': create_t2,
        'FLAIR': create_flair,
        'T1_contrast': create_t1_contrast
    }
    
    for modality_name, modality_func in modalities.items():
        modality_image = modality_func(base_image)
        output_path = os.path.join(output_dir, f"{base_filename}_{modality_name}.jpg")
        cv2.imwrite(output_path, modality_image)
        print(f"Generated {output_path}")

if __name__ == "__main__":
    # Generate sample dataset in the data/sample_images directory
    output_dir = "data/sample_images"
    generate_sample_dataset(output_dir, samples_per_class=2)
    
    # Generate different modalities for one example
    sample_path = os.path.join(output_dir, "glioma_1.jpg")
    modalities_dir = os.path.join(output_dir, "modalities")
    simulate_mri_modalities(sample_path, modalities_dir)