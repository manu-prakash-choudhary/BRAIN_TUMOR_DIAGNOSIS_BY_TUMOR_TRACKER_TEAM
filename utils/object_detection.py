import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_tumor(image, model, confidence_threshold=0.25):
    """
    Detect tumor location in MRI image using YOLO object detection
    
    Parameters:
    -----------
    image : numpy.ndarray
        Preprocessed MRI image
    model : object
        Object detection model
    confidence_threshold : float
        Confidence threshold for detections
        
    Returns:
    --------
    tuple
        (detection_results, annotated_image)
    """
    # Ensure image is the right format
    if len(image.shape) == 3 and image.shape[2] == 3:
        input_image = image
    else:
        # Convert to 3 channels if needed
        input_image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
    
    # Make predictions using model
    # Note: This is a simplified representation
    predictions = model.predict(np.expand_dims(input_image, axis=0))
    
    # Process predictions to get bounding boxes
    # This is a placeholder for actual YOLO processing
    boxes, scores, classes = process_yolo_predictions(predictions, confidence_threshold)
    
    # Create annotated image with bounding boxes
    annotated_image = draw_bounding_boxes(input_image, boxes, scores, classes)
    
    # Create detection results dictionary
    detection_results = {
        'boxes': boxes,
        'scores': scores,
        'classes': classes
    }
    
    return detection_results, annotated_image

def process_yolo_predictions(predictions, confidence_threshold=0.25):
    """
    Process YOLO predictions to get bounding boxes
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Raw predictions from YOLO model
    confidence_threshold : float
        Confidence threshold for detections
        
    Returns:
    --------
    tuple
        (boxes, scores, classes)
    """
    # This is a simplified implementation to simulate YOLO output
    # In a real implementation, this would parse actual YOLO output format
    
    # Simulated outputs for demonstration
    # In this example, I'll simulate finding 1-2 tumor regions
    num_detections = np.random.randint(1, 3)
    
    boxes = []
    scores = []
    classes = []
    
    for _ in range(num_detections):
        # Generate random box (x1, y1, x2, y2) in normalized coordinates
        x1 = np.random.uniform(0.2, 0.6)
        y1 = np.random.uniform(0.2, 0.6)
        width = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.1, 0.3)
        x2 = min(x1 + width, 1.0)
        y2 = min(y1 + height, 1.0)
        
        # Generate random confidence score
        confidence = np.random.uniform(confidence_threshold, 1.0)
        
        # Class is always tumor (class 0)
        class_id = 0
        
        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        classes.append(class_id)
    
    return np.array(boxes), np.array(scores), np.array(classes)

def draw_bounding_boxes(image, boxes, scores, classes):
    """
    Draw bounding boxes on the image
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    boxes : numpy.ndarray
        Bounding boxes in format [x1, y1, x2, y2]
    scores : numpy.ndarray
        Confidence scores for each box
    classes : numpy.ndarray
        Class IDs for each box
        
    Returns:
    --------
    numpy.ndarray
        Image with bounding boxes
    """
    # Create a copy of the image
    annotated_image = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Class labels
    class_labels = ['Tumor']
    
    # Define colors for different classes
    colors = [(255, 0, 0)]  # Red for tumor
    
    # Draw each bounding box
    for box, score, class_id in zip(boxes, scores, classes):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = box
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # Get color for this class
        color = colors[int(class_id) % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label = f"{class_labels[int(class_id)]}: {score:.2f}"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            annotated_image,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return annotated_image

def calculate_segmentation_metrics(ground_truth, prediction):
    """
    Calculate segmentation metrics (Dice, IoU)
    
    Parameters:
    -----------
    ground_truth : numpy.ndarray
        Ground truth segmentation mask
    prediction : numpy.ndarray
        Predicted segmentation mask
        
    Returns:
    --------
    dict
        Metrics dictionary
    """
    # Ensure binary masks
    gt = (ground_truth > 0).astype(np.float32)
    pred = (prediction > 0).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection) / (np.sum(gt) + np.sum(pred))
    
    # Calculate IoU (Jaccard index)
    iou = intersection / union if union > 0 else 0
    
    # Calculate pixel-wise accuracy
    accuracy = np.sum((gt == pred).astype(np.float32)) / gt.size
    
    metrics = {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy
    }
    
    return metrics
