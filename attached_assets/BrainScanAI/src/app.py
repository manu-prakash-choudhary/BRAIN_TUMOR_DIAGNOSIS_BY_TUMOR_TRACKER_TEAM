
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
import plotly.express as px
import plotly.graph_objects as go

from utils.image_processing import preprocess_image, validate_image, segment_brain, extract_features
from utils.model import predict_tumor
from utils.visualization import create_visualization, create_comparison_visualization
from utils.report import generate_report

# Configure page
st.set_page_config(
    page_title="CNN-Based Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state['processed_image'] = None
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "efficientnet"
if 'show_comparison' not in st.session_state:
    st.session_state['show_comparison'] = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .container {
        background-color: #F0F2F6;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.15rem 0.3rem rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-text {
        color: #00CC96;
        font-weight: bold;
    }
    .warning-text {
        color: #EF553B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<p class="main-header">CNN-Based Brain Tumor Detection</p>', unsafe_allow_html=True)
    st.markdown(
        """
        This application uses state-of-the-art Convolutional Neural Networks (CNNs) to detect and classify brain tumors 
        from MRI images. Select a model and upload an image to get started.
        """
    )

    # Sidebar
    with st.sidebar:
        st.header("Tool Controls")
        
        # Model selection
        st.markdown('<p class="sub-header">Model Selection</p>', unsafe_allow_html=True)
        model_option = st.selectbox(
            "Select CNN Model",
            options=["EfficientNet", "VGG16", "ResNet", "Compare All Models"],
            index=0,
            help="Choose which deep learning model to use for tumor detection"
        )
        
        # Map selection to model names
        model_mapping = {
            "EfficientNet": "efficientnet",
            "VGG16": "vgg16",
            "ResNet": "resnet",
            "Compare All Models": "compare"
        }
        
        # Update selected model in session state
        st.session_state['selected_model'] = model_mapping[model_option]
        st.session_state['show_comparison'] = (model_option == "Compare All Models")
        
        # Show model information
        with st.expander("About the selected model"):
            if model_option == "EfficientNet":
                st.write("""
                **EfficientNet** scales the model's depth, width, and resolution efficiently using compound scaling.
                - High accuracy with fewer parameters
                - State-of-the-art performance
                - Efficient resource utilization
                """)
            elif model_option == "VGG16":
                st.write("""
                **VGG16** is a 16-layer CNN architecture that won the ImageNet competition in 2014.
                - Simple but effective architecture
                - Good feature extraction capabilities
                - Widely used in medical imaging
                """)
            elif model_option == "ResNet":
                st.write("""
                **ResNet** uses skip connections to solve the vanishing gradient problem in deep networks.
                - Can train very deep networks
                - High accuracy on complex tasks
                - Used in many medical imaging applications
                """)
            elif model_option == "Compare All Models":
                st.write("""
                **Compare All Models** runs the image through EfficientNet, VGG16, and ResNet models, 
                allowing you to compare results across different architectures.
                """)
        
        # File upload
        st.markdown('<p class="sub-header">Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a brain MRI image for analysis"
        )
        
        # Process image button
        if uploaded_file and st.button("Analyze Image", use_container_width=True):
            try:
                with st.spinner("Processing MRI image..."):
                    # Display original image
                    image = Image.open(uploaded_file)
                    
                    # Validate image
                    if not validate_image(image):
                        st.error("Invalid image format or dimensions")
                        return
                    
                    # Preprocess image
                    processed_img = preprocess_image(image)
                    st.session_state['processed_image'] = processed_img
                    
                    # Segment brain region
                    segmented_img = segment_brain(processed_img)
                    
                    # Run selected model(s)
                    if st.session_state['show_comparison']:
                        # Run all models
                        models = ["efficientnet", "vgg16", "resnet"]
                        predictions = {}
                        
                        for model_name in models:
                            predictions[model_name] = predict_tumor(segmented_img, model_name)
                        
                        st.session_state['predictions'] = predictions
                    else:
                        # Run selected model
                        model_name = st.session_state['selected_model']
                        prediction = predict_tumor(segmented_img, model_name)
                        st.session_state['predictions'] = {model_name: prediction}
                    
                    time.sleep(1)  # Simulate processing time
                    st.success("Analysis complete!")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
    
    # Main content area
    if st.session_state['processed_image'] is not None and st.session_state['predictions'] is not None:
        # Create tabs for visualization and details
        tab1, tab2, tab3 = st.tabs(["Visualization", "Detailed Analysis", "Technical Information"])
        
        with tab1:
            # Display visualizations
            if st.session_state['show_comparison']:
                # Show comparison visualizations
                st.markdown('<p class="sub-header">Model Comparison</p>', unsafe_allow_html=True)
                comparison_viz = create_comparison_visualization(
                    st.session_state['processed_image'],
                    st.session_state['predictions']
                )
                st.plotly_chart(comparison_viz, use_container_width=True)
                
                # Create a feature comparison table
                st.markdown('<p class="sub-header">Model Prediction Comparison</p>', unsafe_allow_html=True)
                
                # Extract data for comparison table
                model_names = []
                tumor_types = []
                confidences = []
                
                for model_name, prediction in st.session_state['predictions'].items():
                    model_names.append(model_name.upper())
                    tumor_types.append(prediction['type'])
                    confidences.append(f"{prediction['confidence']:.2f}%")
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Model': model_names,
                    'Predicted Type': tumor_types,
                    'Confidence': confidences
                })
                
                # Show table
                st.table(comparison_df)
                
                # Create a bar chart for confidence comparison
                confidences_numeric = [float(conf.replace('%', '')) for conf in confidences]
                confidence_df = pd.DataFrame({
                    'Model': model_names,
                    'Confidence': confidences_numeric
                })
                
                confidence_chart = px.bar(
                    confidence_df, 
                    x='Model', 
                    y='Confidence',
                    color='Model',
                    labels={'Confidence': 'Confidence (%)'},
                    title='Model Confidence Comparison'
                )
                
                st.plotly_chart(confidence_chart, use_container_width=True)
                
            else:
                # Show single model visualization
                model_name = list(st.session_state['predictions'].keys())[0]
                prediction = st.session_state['predictions'][model_name]
                
                st.markdown(f'<p class="sub-header">{model_name.upper()} Detection Results</p>', unsafe_allow_html=True)
                
                # Create visualization
                visualization = create_visualization(
                    st.session_state['processed_image'],
                    prediction
                )
                st.plotly_chart(visualization, use_container_width=True)
                
                # Display prediction results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"### Detected Type: {prediction['type']}")
                    st.metric("Confidence Score", f"{prediction['confidence']:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Show appropriate card based on result
                    if prediction['type'] == 'No Tumor':
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<p class="success-text">✓ No tumor detected</p>', unsafe_allow_html=True)
                        st.markdown('Neural network analysis indicates no significant abnormalities in the brain MRI.')
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<p class="warning-text">⚠ Tumor detected</p>', unsafe_allow_html=True)
                        st.markdown(f'The analysis indicates potential presence of a {prediction["type"]} tumor.')
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            # Detailed analysis
            if st.session_state['show_comparison']:
                # Let user select which model's details to display
                selected_detail_model = st.selectbox(
                    "Select model for detailed analysis",
                    options=list(st.session_state['predictions'].keys()),
                    format_func=lambda x: x.upper()
                )
                prediction = st.session_state['predictions'][selected_detail_model]
            else:
                # Use the selected model
                model_name = list(st.session_state['predictions'].keys())[0]
                prediction = st.session_state['predictions'][model_name]
            
            # Display detailed information
            st.markdown('<p class="sub-header">Tumor Characteristics</p>', unsafe_allow_html=True)
            
            if prediction['type'] == 'No Tumor':
                st.info("No tumor characteristics available as no tumor was detected.")
            else:
                # Display tumor details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Tumor Size", f"{prediction['features']['size']:.1f} cm")
                    st.metric("Intensity", f"{prediction['features']['intensity']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"### Location: {prediction['features']['location']}")
                    st.markdown(f"Common symptoms associated with {prediction['type']} tumors in this region may include headaches, vision changes, and cognitive disturbances.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display tumor type information
                tumor_info = {
                    'Glioma': """
                    **Glioma** tumors develop from glial cells and can be found in the brain and spinal cord.
                    - **Grade**: Varies from low (I) to high (IV)
                    - **Treatment**: Surgery, radiation therapy, chemotherapy
                    - **Prognosis**: Depends on grade, location, and patient age
                    """,
                    'Meningioma': """
                    **Meningioma** tumors form in the meninges, the membranes surrounding the brain and spinal cord.
                    - **Grade**: Usually benign (Grade I)
                    - **Treatment**: Surgery, sometimes radiation therapy
                    - **Prognosis**: Generally good with complete removal
                    """,
                    'Pituitary': """
                    **Pituitary** tumors develop in the pituitary gland at the base of the brain.
                    - **Type**: Usually adenomas (non-cancerous)
                    - **Treatment**: Medication, surgery, radiation therapy
                    - **Prognosis**: Generally good with proper treatment
                    """
                }
                
                st.markdown('<p class="sub-header">About This Tumor Type</p>', unsafe_allow_html=True)
                st.markdown(tumor_info.get(prediction['type'], ""))
        
        with tab3:
            # Technical information
            st.markdown('<p class="sub-header">Technical Information</p>', unsafe_allow_html=True)
            
            # Model architecture information
            if st.session_state['show_comparison']:
                # Show multiple models
                for model_name, prediction in st.session_state['predictions'].items():
                    with st.expander(f"{model_name.upper()} Model Details"):
                        st.markdown(f"**Model**: {model_name.upper()}")
                        st.markdown(f"**Confidence**: {prediction['confidence']:.2f}%")
                        st.markdown(f"**Prediction**: {prediction['type']}")
                        
                        # Create feature importance chart (simulated)
                        features = {
                            'Texture': np.random.uniform(0.5, 0.9),
                            'Intensity': np.random.uniform(0.6, 0.95),
                            'Shape': np.random.uniform(0.55, 0.85),
                            'Location': np.random.uniform(0.5, 0.8),
                            'Contrast': np.random.uniform(0.45, 0.75)
                        }
                        
                        features_df = pd.DataFrame({
                            'Feature': list(features.keys()),
                            'Importance': list(features.values())
                        })
                        
                        features_df = features_df.sort_values('Importance', ascending=False)
                        
                        feature_chart = px.bar(
                            features_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance (Simulated)',
                            labels={'Importance': 'Relative Importance'},
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        
                        st.plotly_chart(feature_chart, use_container_width=True)
            else:
                # Show single model
                model_name = list(st.session_state['predictions'].keys())[0]
                prediction = st.session_state['predictions'][model_name]
                
                st.markdown(f"**Model**: {model_name.upper()}")
                st.markdown(f"**Architecture**: Deep Convolutional Neural Network")
                st.markdown(f"**Input Size**: 224x224 pixels")
                st.markdown(f"**Classes**: No Tumor, Glioma, Meningioma, Pituitary")
                
                # Create feature importance chart (simulated)
                features = {
                    'Texture': np.random.uniform(0.5, 0.9),
                    'Intensity': np.random.uniform(0.6, 0.95),
                    'Shape': np.random.uniform(0.55, 0.85),
                    'Location': np.random.uniform(0.5, 0.8),
                    'Contrast': np.random.uniform(0.45, 0.75)
                }
                
                features_df = pd.DataFrame({
                    'Feature': list(features.keys()),
                    'Importance': list(features.values())
                })
                
                features_df = features_df.sort_values('Importance', ascending=False)
                
                feature_chart = px.bar(
                    features_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance (Simulated)',
                    labels={'Importance': 'Relative Importance'},
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(feature_chart, use_container_width=True)
            
            # Image preprocessing details
            with st.expander("Image Preprocessing Pipeline"):
                st.markdown("""
                1. **Image Validation** - Check if the uploaded file is a valid MRI image
                2. **Conversion to Grayscale** - Simplify input for the neural network
                3. **Resizing** - Standardize to 224×224 pixels for CNN input
                4. **Normalization** - Scale pixel values to [0,1] range
                5. **CLAHE** - Apply Contrast Limited Adaptive Histogram Equalization
                6. **Noise Reduction** - Apply Gaussian blur
                7. **Edge Enhancement** - Sharpen edges for better feature detection
                8. **Brain Segmentation** - Isolate brain region from background
                """)
            
            # CNN architectures information
            with st.expander("CNN Architecture Information"):
                st.markdown("""
                **VGG16**
                - 16 layers (13 convolutional, 3 fully connected)
                - Used 3×3 convolutional filters
                - 138 million parameters
                
                **ResNet**
                - Uses residual connections to enable training of very deep networks
                - Skip connections help with gradient flow
                - Typically 50+ layers
                
                **EfficientNet**
                - Uses compound scaling to balance depth, width, and resolution
                - Achieves state-of-the-art accuracy with fewer parameters
                - Optimized for efficiency
                """)
        
        # Generate report button
        if st.button("Generate PDF Report"):
            try:
                with st.spinner("Generating comprehensive PDF report..."):
                    # If comparison is shown, use the first model for the report
                    if st.session_state['show_comparison']:
                        model_name = "efficientnet"  # Default to EfficientNet for report
                        report_prediction = st.session_state['predictions'][model_name]
                    else:
                        model_name = list(st.session_state['predictions'].keys())[0]
                        report_prediction = st.session_state['predictions'][model_name]
                    
                    report_bytes = generate_report(
                        image=st.session_state['processed_image'],
                        prediction=report_prediction
                    )
                    
                    st.download_button(
                        label="Download Report",
                        data=report_bytes,
                        file_name=f"cnn_tumor_analysis_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
                    st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    else:
        # Show welcome message
        st.markdown(
            """
            <div class="container">
                <h3>🧠 Welcome to the CNN-Based Brain Tumor Detection Tool</h3>
                <p>This application uses state-of-the-art deep learning models to analyze brain MRI images and detect tumors.</p>
                <p>To get started:</p>
                <ol>
                    <li>Select a model from the sidebar (EfficientNet, VGG16, ResNet, or Compare All)</li>
                    <li>Upload a brain MRI image</li>
                    <li>Click "Analyze Image" to process the image</li>
                </ol>
                <p>The system will analyze the image and provide detailed results including:</p>
                <ul>
                    <li>Tumor detection results with confidence scores</li>
                    <li>Visual analysis with tumor localization</li>
                    <li>Detailed tumor characteristics (if detected)</li>
                    <li>Option to generate a PDF report</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Add sample images section
        st.markdown('<p class="sub-header">Sample Information</p>', unsafe_allow_html=True)
        st.markdown(
            """
            The system can detect and classify three types of brain tumors:
            
            1. **Glioma** - Tumors that occur in the brain and spinal cord
            2. **Meningioma** - Tumors that arise from the membranes surrounding the brain and spinal cord
            3. **Pituitary** - Tumors that develop in the pituitary gland
            
            Please upload a brain MRI image to start the analysis.
            """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>⚠️ This tool is for educational purposes only. Always consult with qualified medical professionals for diagnosis and treatment decisions.</p>
            <p>Powered by TensorFlow, Streamlit, and Plotly</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
