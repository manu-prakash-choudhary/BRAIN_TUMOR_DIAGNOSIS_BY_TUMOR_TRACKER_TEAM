
import numpy as np
import cv2
from PIL import Image
import io

def validate_image(image):
    """
    Validate that the uploaded image is suitable for processing
    """
    # Check if image is valid
    if image is None:
        return False
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Check dimensions
    if img_array.ndim < 2 or img_array.shape[0] < 100 or img_array.shape[1] < 100:
        return False
    
    # Simple check for blank images
    if img_array.std() < 10:  # Very low variance typically means blank image
        return False
    
    return True

def preprocess_image(image):
    """
    Preprocess the input image for brain tumor detection
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if not already
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Resize to standard dimensions
    resized = cv2.resize(gray, (224, 224))
    
    # Apply preprocessing techniques
    # Normalization
    normalized = resized / 255.0
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(np.uint8(normalized * 255))
    normalized_enhanced = enhanced / 255.0
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(normalized_enhanced, (5, 5), 0)
    
    # Apply edge enhancement
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    return sharpened

def segment_brain(image):
    """
    Segment the brain region from the background
    """
    # Threshold the image
    _, thresh = cv2.threshold(image, 0.2, 1, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(
        np.uint8(thresh * 255), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Create a mask
    mask = np.zeros_like(image)
    
    # Fill the largest contour (should be the brain)
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 1, -1)
        
        # Apply mask to original image
        segmented = image * mask
        return segmented
    
    # If no contours found, return original image
    return image

def extract_features(image):
    """
    Extract texture and statistical features from the image
    """
    features = {}
    
    # Basic statistical features
    features['mean'] = np.mean(image)
    features['std'] = np.std(image)
    features['min'] = np.min(image)
    features['max'] = np.max(image)
    
    # Gradient features (for texture analysis)
    gradient_x = np.gradient(image, axis=1)
    gradient_y = np.gradient(image, axis=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    features['gradient_mean'] = np.mean(gradient_magnitude)
    features['gradient_std'] = np.std(gradient_magnitude)
    
    # GLCM features (for texture analysis)
    # In a full implementation, you would compute GLCM features here
    
    return features
