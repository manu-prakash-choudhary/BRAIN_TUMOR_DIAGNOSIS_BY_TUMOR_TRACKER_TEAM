import numpy as np

def extract_features(preprocessed_image, model):
    """
    Extract features from preprocessed MRI images using a pre-trained model
    
    Parameters:
    -----------
    preprocessed_image : numpy.ndarray
        Preprocessed MRI image
    model : object
        Model for feature extraction
        
    Returns:
    --------
    numpy.ndarray
        Extracted features
    """
    # Add batch dimension if needed
    if len(preprocessed_image.shape) == 3:
        image_batch = np.expand_dims(preprocessed_image, axis=0)
    else:
        image_batch = preprocessed_image
        
    # Extract features
    features = model.predict(image_batch)
    
    return features

def extract_deep_features(preprocessed_image, model, layer_name=None):
    """
    Extract deep features from a specific layer of a deep learning model
    
    Parameters:
    -----------
    preprocessed_image : numpy.ndarray
        Preprocessed MRI image
    model : object
        Model for feature extraction
    layer_name : str, optional
        Name of the layer to extract features from (unused in mock implementation)
        
    Returns:
    --------
    numpy.ndarray
        Extracted deep features
    """
    # Add batch dimension if needed
    if len(preprocessed_image.shape) == 3:
        image_batch = np.expand_dims(preprocessed_image, axis=0)
    else:
        image_batch = preprocessed_image
    
    # In this simplified implementation, we just return features from the model
    # Ignoring the layer_name parameter since we're not using real TensorFlow models
    features = model.predict(image_batch)
    
    return features

def combine_multimodal_features(features_list):
    """
    Combine features from multiple MRI modalities
    
    Parameters:
    -----------
    features_list : list
        List of feature arrays from different MRI modalities
        
    Returns:
    --------
    numpy.ndarray
        Combined features
    """
    # Concatenate features from different modalities
    combined_features = np.concatenate(features_list, axis=-1)
    
    return combined_features

def select_relevant_features(features, n_features=100):
    """
    Select most relevant features using PCA-like dimensionality reduction
    
    Parameters:
    -----------
    features : numpy.ndarray
        Input features array
    n_features : int
        Number of features to select
        
    Returns:
    --------
    numpy.ndarray
        Selected features
    """
    # Flatten features
    flattened = features.reshape(features.shape[0], -1)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(flattened, rowvar=False)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top eigenvectors
    selected_eigenvectors = eigenvectors[:, :n_features]
    
    # Project data onto selected eigenvectors
    reduced_features = np.dot(flattened, selected_eigenvectors)
    
    return reduced_features
