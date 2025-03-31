import numpy as np
import tensorflow as tf

def classify_tumor(features, model):
    """
    Classify tumor type based on extracted features
    
    Parameters:
    -----------
    features : numpy.ndarray
        Extracted features from MRI image
    model : tf.keras.Model
        Classification model
        
    Returns:
    --------
    tuple
        (prediction_label, class_probabilities)
    """
    # Ensure correct input shape
    if len(features.shape) == 2 and features.shape[0] == 1:
        # Already in batch format with batch size 1
        input_features = features
    else:
        # Add batch dimension
        input_features = np.expand_dims(features, axis=0)
    
    # Get predictions from model
    predictions = model.predict(input_features)
    
    # Get probabilities for each class
    probabilities = predictions[0]
    
    # Map class indices to labels
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class, probabilities

def get_confidence_score(probabilities):
    """
    Calculate confidence score from prediction probabilities
    
    Parameters:
    -----------
    probabilities : numpy.ndarray
        Class probabilities
        
    Returns:
    --------
    float
        Confidence score
    """
    # Get highest probability as confidence score
    confidence = np.max(probabilities)
    
    return confidence

def evaluate_classification(true_labels, predicted_labels):
    """
    Evaluate classification performance
    
    Parameters:
    -----------
    true_labels : list
        True class labels
    predicted_labels : list
        Predicted class labels
        
    Returns:
    --------
    dict
        Performance metrics
    """
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Calculate accuracy
    accuracy = sum([1 for t, p in zip(true_labels, predicted_labels) if t == p]) / len(true_labels)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    
    # Fill confusion matrix
    for t, p in zip(true_labels, predicted_labels):
        t_idx = class_labels.index(t)
        p_idx = class_labels.index(p)
        confusion_matrix[t_idx, p_idx] += 1
    
    # Calculate precision, recall, and F1 score for each class
    precision = []
    recall = []
    f1_score = []
    
    for i in range(len(class_labels)):
        true_positive = confusion_matrix[i, i]
        false_positive = sum(confusion_matrix[:, i]) - true_positive
        false_negative = sum(confusion_matrix[i, :]) - true_positive
        
        # Calculate precision
        if true_positive + false_positive == 0:
            class_precision = 0
        else:
            class_precision = true_positive / (true_positive + false_positive)
        
        # Calculate recall
        if true_positive + false_negative == 0:
            class_recall = 0
        else:
            class_recall = true_positive / (true_positive + false_negative)
        
        # Calculate F1 score
        if class_precision + class_recall == 0:
            class_f1 = 0
        else:
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
        
        precision.append(class_precision)
        recall.append(class_recall)
        f1_score.append(class_f1)
    
    # Calculate macro average metrics
    macro_precision = sum(precision) / len(precision)
    macro_recall = sum(recall) / len(recall)
    macro_f1 = sum(f1_score) / len(f1_score)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }
    
    return metrics

def ensemble_classification(features, models):
    """
    Perform ensemble classification using multiple models
    
    Parameters:
    -----------
    features : numpy.ndarray
        Extracted features from MRI image
    models : list
        List of classification models
        
    Returns:
    --------
    tuple
        (prediction_label, class_probabilities)
    """
    # Collect predictions from each model
    all_predictions = []
    
    for model in models:
        predictions = model.predict(np.expand_dims(features, axis=0))[0]
        all_predictions.append(predictions)
    
    # Average predictions from all models
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Map class indices to labels
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predicted_class_idx = np.argmax(ensemble_predictions)
    predicted_class = class_labels[predicted_class_idx]
    
    return predicted_class, ensemble_predictions
