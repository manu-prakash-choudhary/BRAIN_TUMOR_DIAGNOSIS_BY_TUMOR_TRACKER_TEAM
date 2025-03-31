import cv2
import numpy as np

def preprocess_image(image, modality, target_size=(224, 224)):
    """
    Preprocess MRI images for feature extraction and classification
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input MRI image
    modality : str
        MRI modality (T1-weighted, T2-weighted, FLAIR, T1-weighted with contrast)
    target_size : tuple
        Target size for resizing (height, width)
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed image ready for model input
    """
    # Convert to grayscale if image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to target size
    resized = cv2.resize(image, target_size)
    
    # Apply modality-specific preprocessing
    if modality == "T1-weighted":
        # Contrast enhancement for T1 images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized.astype(np.uint8))
    elif modality == "T2-weighted":
        # Histogram equalization for T2 images
        enhanced = cv2.equalizeHist(resized.astype(np.uint8))
    elif modality == "FLAIR":
        # Adaptive histogram equalization for FLAIR images
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized.astype(np.uint8))
    elif modality == "T1-weighted with contrast":
        # Enhance contrast and brightness for contrast-enhanced T1 images
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized.astype(np.uint8))
    else:
        # Default preprocessing
        enhanced = resized
    
    # Normalize pixel values to [0, 1]
    normalized = enhanced.astype(np.float32) / 255.0
    
    # Remove noise using Gaussian blur
    denoised = cv2.GaussianBlur(normalized, (3, 3), 0)
    
    # Expand dimensions for model input (add batch and channel dimensions)
    prepared = np.expand_dims(denoised, axis=-1)
    prepared = np.repeat(prepared, 3, axis=-1)  # Convert to 3 channels for pre-trained models
    
    return prepared

def skull_strip(image):
    """
    Remove skull from brain MRI images
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input MRI image
        
    Returns:
    --------
    numpy.ndarray
        Skull-stripped image
    """
    # Convert to 8-bit grayscale
    img_8bit = (image * 255).astype(np.uint8)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for brain region
    mask = np.zeros_like(img_8bit)
    if contours:
        # Find the largest contour (brain)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Apply mask to original image
    skull_stripped = cv2.bitwise_and(img_8bit, img_8bit, mask=mask)
    
    # Normalize back to [0, 1]
    return skull_stripped.astype(np.float32) / 255.0

def apply_data_augmentation(image):
    """
    Apply data augmentation to MRI images
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input MRI image
        
    Returns:
    --------
    numpy.ndarray
        Augmented image
    """
    # Random rotation
    angle = np.random.uniform(-10, 10)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    brightened = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)
    
    # Random flip
    if np.random.random() > 0.5:
        flipped = cv2.flip(brightened, 1)  # Horizontal flip
    else:
        flipped = brightened
    
    return flipped
