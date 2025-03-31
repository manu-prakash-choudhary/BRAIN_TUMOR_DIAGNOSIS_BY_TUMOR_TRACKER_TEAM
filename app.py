import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import time

from utils.preprocessing import preprocess_image
from utils.feature_extraction import extract_features
from utils.classification import classify_tumor
from utils.object_detection import detect_tumor
from utils.visualization import generate_gradcam
from utils.report_generation import generate_report
from models.model_loader import load_models

# Set page config
st.set_page_config(
    page_title="BrainScanAI - Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models at startup
@st.cache_resource
def initialize_models():
    return load_models()

# Initialize models
models = initialize_models()

# Main application header
st.title("🧠 BrainScanAI - Brain Tumor Detection System")
st.markdown("""
This application uses deep learning to detect and classify brain tumors from MRI images.
Upload your MRI scan to get a comprehensive analysis.
""")

# Sidebar for navigation and settings
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Analyze", "About"])

if page == "Home":
    st.header("Welcome to BrainScanAI")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Features")
        st.markdown("""
        - 🔍 MRI image preprocessing
        - 🧬 Feature extraction using deep learning
        - 🔎 Tumor classification into four categories
        - 📍 Object detection to locate tumors
        - 📊 Visualization with GradCAM
        - 📝 Comprehensive diagnosis reports
        """)
    
    with col2:
        st.subheader("Supported MRI Modalities")
        st.markdown("""
        - T1-weighted
        - T2-weighted
        - FLAIR (Fluid Attenuated Inversion Recovery)
        - T1-weighted with contrast enhancement
        """)
    
    st.markdown("---")
    st.subheader("How It Works")
    
    # Create a simple flowchart using markdown
    st.markdown("""
    ```
    MRI Image → Preprocessing → Feature Extraction → Classification/Detection → Visualization → Report
    ```
    """)
    
    st.markdown("""
    **Step 1**: Upload your MRI scan\n
    **Step 2**: Select the MRI modality\n
    **Step 3**: Choose analysis options\n
    **Step 4**: View results and diagnosis report
    """)

elif page == "Upload & Analyze":
    st.header("Upload & Analyze MRI Scan")
    
    # File uploader for MRI images
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    
    # MRI modality selection
    mri_modality = st.selectbox(
        "Select MRI Modality",
        ["T1-weighted", "T2-weighted", "FLAIR", "T1-weighted with contrast"]
    )
    
    # Analysis options
    st.subheader("Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        run_classification = st.checkbox("Tumor Classification", value=True)
        run_detection = st.checkbox("Object Detection", value=True)
    
    with col2:
        generate_gradcam_vis = st.checkbox("Generate GradCAM Visualization", value=True)
        generate_diagnosis = st.checkbox("Generate Diagnosis Report", value=True)
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Add analyze button
        if st.button("Analyze"):
            with st.spinner("Processing MRI scan..."):
                # Progress bar for visual feedback
                progress_bar = st.progress(0)
                
                # Preprocess image
                progress_bar.progress(10)
                st.info("Preprocessing image...")
                preprocessed_img = preprocess_image(img_array, mri_modality)
                time.sleep(0.5)  # Simulate processing time
                
                # Extract features
                progress_bar.progress(30)
                st.info("Extracting features...")
                features = extract_features(preprocessed_img, models['feature_extractor'])
                time.sleep(0.5)  # Simulate processing time
                
                # Results section
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                # Classification
                if run_classification:
                    progress_bar.progress(50)
                    st.info("Classifying tumor...")
                    classification_result, class_probabilities = classify_tumor(
                        features, 
                        models['classifier']
                    )
                    time.sleep(0.5)  # Simulate processing time
                    
                    with col1:
                        st.markdown("### Tumor Classification")
                        st.markdown(f"**Detected tumor type:** {classification_result}")
                        
                        # Display probabilities
                        fig, ax = plt.figure(figsize=(10, 4)), plt.axes()
                        classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
                        ax.bar(classes, class_probabilities, color=colors)
                        ax.set_ylabel('Probability')
                        ax.set_title('Tumor Classification Probabilities')
                        
                        st.pyplot(fig)
                
                # Object Detection
                if run_detection:
                    progress_bar.progress(70)
                    st.info("Detecting tumor location...")
                    detection_result, detection_img = detect_tumor(
                        preprocessed_img, 
                        models['object_detector']
                    )
                    time.sleep(0.5)  # Simulate processing time
                    
                    with col2:
                        st.markdown("### Tumor Detection")
                        st.image(detection_img, caption="Tumor Detection Result", use_column_width=True)
                
                # GradCAM Visualization
                if generate_gradcam_vis and run_classification:
                    progress_bar.progress(85)
                    st.info("Generating GradCAM visualization...")
                    gradcam_img = generate_gradcam(
                        preprocessed_img,
                        models['classifier'],
                        class_idx=np.argmax(class_probabilities)
                    )
                    time.sleep(0.5)  # Simulate processing time
                    
                    st.subheader("GradCAM Visualization")
                    st.image(gradcam_img, caption="GradCAM: Regions of Interest", use_column_width=True)
                
                # Diagnosis Report
                if generate_diagnosis and run_classification:
                    progress_bar.progress(95)
                    st.info("Generating diagnosis report...")
                    report = generate_report(
                        classification_result, 
                        class_probabilities,
                        mri_modality
                    )
                    time.sleep(0.5)  # Simulate processing time
                    
                    st.subheader("Diagnosis Report")
                    st.markdown(report)
                
                progress_bar.progress(100)
                st.success("Analysis completed successfully!")

elif page == "About":
    st.header("About BrainScanAI")
    
    st.markdown("""
    ## About This Application
    
    BrainScanAI is an advanced brain tumor detection and classification system that uses deep learning techniques to analyze MRI images.
    
    ### Technologies Used
    - **Frontend**: Streamlit
    - **Deep Learning**: TensorFlow/Keras
    - **Image Processing**: OpenCV
    - **Visualization**: Matplotlib, Plotly
    - **Object Detection**: YOLO (via Ultralytics)
    
    ### Supported Tumor Types
    - Glioma
    - Meningioma
    - Pituitary
    - No tumor (normal scan)
    
    ### Deep Learning Models
    - **Feature Extraction**: ResNet50, EfficientNet
    - **Classification**: Custom CNN architectures
    - **Segmentation**: U-Net, SegNet
    - **Object Detection**: YOLOv8
    
    ### Disclaimer
    This tool is intended for research and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 BrainScanAI | Developed for medical research purposes")
