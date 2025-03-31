import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image

def generate_gradcam(img, model, class_idx=None, layer_name=None):
    """
    Generate Grad-CAM visualization for model decision
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image
    model : object
        Model object (mock implementation)
    class_idx : int, optional
        Class index to visualize, if None uses predicted class
    layer_name : str, optional
        Name of the layer to use for Grad-CAM (unused in mock implementation)
        
    Returns:
    --------
    numpy.ndarray
        Simulated Grad-CAM visualization
    """
    # Ensure image has batch dimension and is in the right format
    if len(img.shape) == 3:
        img_tensor = np.expand_dims(img, axis=0)
    else:
        img_tensor = img
    
    # Set default class index if not provided
    if class_idx is None:
        class_idx = 0  # Default to first class
    
    # Create a simple heatmap for visualization (random pattern)
    height, width = img.shape[:2]
    
    # Generate a simulated heatmap with highest activation in center
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    
    # Create a radial gradient with some noise
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Scale distance to [0,1] range and invert (closer to center = higher value)
    heatmap = 1 - (dist_from_center / max_dist)
    
    # Add some randomness for realism
    noise = np.random.rand(height, width) * 0.2
    heatmap = heatmap + noise
    heatmap = np.clip(heatmap, 0, 1)
    
    # Resize heatmap to match input image size
    heatmap = cv2.resize(heatmap, (width, height))
    
    # Apply colormap to heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    img_rgb = (img * 255).astype(np.uint8)
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    superimposed_img = heatmap * 0.4 + img_rgb
    superimposed_img = superimposed_img.astype(np.uint8)
    
    return superimposed_img

def create_tumor_region_visualization(image, segmentation_mask, alpha=0.5):
    """
    Create visualization of tumor region using segmentation mask
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    segmentation_mask : numpy.ndarray
        Segmentation mask
    alpha : float
        Transparency value for overlay
        
    Returns:
    --------
    numpy.ndarray
        Visualization of tumor region
    """
    # Ensure image is in RGB format
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        img_rgb = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = np.uint8(image * 255)
    
    # Create colored mask (red for tumor)
    mask_rgb = np.zeros_like(img_rgb)
    mask_rgb[segmentation_mask > 0] = [255, 0, 0]  # Red for tumor
    
    # Create overlay
    visualization = cv2.addWeighted(img_rgb, 1, mask_rgb, alpha, 0)
    
    return visualization

def plot_mri_with_tumor_overlay(image, boxes, figsize=(10, 8)):
    """
    Create a figure with MRI image and tumor overlay
    
    Parameters:
    -----------
    image : numpy.ndarray
        MRI image
    boxes : list
        List of bounding boxes in format [x1, y1, x2, y2]
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with MRI and tumor overlay
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display MRI image
    ax.imshow(image, cmap='gray')
    
    # Add bounding boxes
    height, width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.set_title('MRI with Tumor Detection')
    ax.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = np.array(Image.open(buf))
    plt.close(fig)
    
    return img

def create_multimodal_visualization(images, titles):
    """
    Create visualization for multiple MRI modalities
    
    Parameters:
    -----------
    images : list
        List of images from different modalities
    titles : list
        List of titles for each modality
        
    Returns:
    --------
    numpy.ndarray
        Combined visualization
    """
    # Create figure with subplots
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    # Handle case with single image
    if n_images == 1:
        axes = [axes]
    
    # Display each image
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    combined_img = np.array(Image.open(buf))
    plt.close(fig)
    
    return combined_img
