import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import time
import glob

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
Upload your MRI scan or use our example images to get a comprehensive analysis.
""")

# Sidebar for navigation and settings
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Analyze", "Example Images", "About"])

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
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        
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
                        st.image(detection_img, caption="Tumor Detection Result", use_container_width=True, clamp=True)
                
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
                    st.image(gradcam_img, caption="GradCAM: Regions of Interest", use_container_width=True, clamp=True)
                
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

elif page == "Example Images":
    st.header("Example MRI Images")
    
    st.markdown("""
    Use these example MRI images to test the system without having to upload your own files.
    These are synthetic MRI scans generated for demonstration purposes.
    """)
    
    # Get all example images
    example_images = glob.glob("data/sample_images/*.jpg")
    # Filter out modality directory images
    example_images = [img for img in example_images if "modalities" not in img]
    
    if not example_images:
        st.warning("No example images found. Please run the sample generator first.")
    else:
        # Group images by tumor type
        tumor_types = ["glioma", "meningioma", "pituitary", "no_tumor"]
        
        for tumor_type in tumor_types:
            st.subheader(f"{tumor_type.replace('_', ' ').title()} Examples")
            
            # Filter images for this tumor type
            type_images = [img for img in example_images if tumor_type in os.path.basename(img).lower()]
            
            if type_images:
                # Create columns for each image
                columns = st.columns(len(type_images))
                
                for i, img_path in enumerate(type_images):
                    with columns[i]:
                        # Load and display image
                        img = Image.open(img_path)
                        st.image(img, caption=os.path.basename(img_path), use_container_width=True)
                        
                        # Add a button to analyze this image
                        if st.button(f"Analyze {os.path.basename(img_path)}", key=f"analyze_{img_path}"):
                            # Load the image
                            img_array = np.array(img)
                            
                            # Run analysis with default options
                            mri_modality = "T1-weighted"  # Default modality
                            
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
                                progress_bar.progress(70)
                                st.info("Detecting tumor location...")
                                detection_result, detection_img = detect_tumor(
                                    preprocessed_img, 
                                    models['object_detector']
                                )
                                time.sleep(0.5)  # Simulate processing time
                                
                                with col2:
                                    st.markdown("### Tumor Detection")
                                    st.image(detection_img, caption="Tumor Detection Result", use_container_width=True, clamp=True)
                                
                                # GradCAM Visualization
                                progress_bar.progress(85)
                                st.info("Generating GradCAM visualization...")
                                gradcam_img = generate_gradcam(
                                    preprocessed_img,
                                    models['classifier'],
                                    class_idx=np.argmax(class_probabilities)
                                )
                                time.sleep(0.5)  # Simulate processing time
                                
                                st.subheader("GradCAM Visualization")
                                st.image(gradcam_img, caption="GradCAM: Regions of Interest", use_container_width=True, clamp=True)
                                
                                # Diagnosis Report
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
            else:
                st.info(f"No {tumor_type} example images found.")
        
        # Show modality examples if available
        modality_dir = "data/sample_images/modalities"
        if os.path.exists(modality_dir):
            modality_images = glob.glob(f"{modality_dir}/*.jpg")
            
            if modality_images:
                st.markdown("---")
                st.subheader("MRI Modality Examples")
                st.markdown("Different MRI modalities provide complementary information for diagnosis.")
                
                # Display all modalities of the same image
                modality_cols = st.columns(len(modality_images))
                
                for i, img_path in enumerate(modality_images):
                    modality_name = os.path.basename(img_path).split('_')[-1].split('.')[0]
                    with modality_cols[i]:
                        img = Image.open(img_path)
                        st.image(img, caption=f"{modality_name} Modality", use_container_width=True)
                        
                        # Add a button to analyze this modality image
                        if st.button(f"Analyze {modality_name}", key=f"analyze_mod_{img_path}"):
                            # Similar analysis logic as above
                            img_array = np.array(img)
                            
                            with st.spinner(f"Processing {modality_name} scan..."):
                                # Abbreviated analysis for modality examples
                                # (This could be expanded similar to the main analysis)
                                preprocessed_img = preprocess_image(img_array, modality_name.replace('_', '-'))
                                features = extract_features(preprocessed_img, models['feature_extractor'])
                                classification_result, class_probabilities = classify_tumor(features, models['classifier'])
                                
                                st.subheader(f"{modality_name} Analysis Results")
                                st.markdown(f"**Detected tumor type:** {classification_result}")
                                
                                # Display detection result
                                _, detection_img = detect_tumor(preprocessed_img, models['object_detector'])
                                st.image(detection_img, caption=f"Tumor Detection on {modality_name}", use_container_width=True, clamp=True)
                                
                                st.success(f"{modality_name} analysis completed!")

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
