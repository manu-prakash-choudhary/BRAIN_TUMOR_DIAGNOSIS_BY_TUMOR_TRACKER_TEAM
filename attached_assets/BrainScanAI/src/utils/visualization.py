
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_visualization(processed_image, prediction):
    """
    Create interactive visualization of the detection results with CNN model outputs
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=("MRI Scan", "Detection Results"),
        column_widths=[0.5, 0.5]
    )
    
    # Add original image
    fig.add_trace(
        go.Heatmap(
            z=processed_image,
            colorscale='Gray',
            showscale=False,
            name='MRI Scan'
        ),
        row=1, col=1
    )
    
    # Add original image to second plot for overlay
    fig.add_trace(
        go.Heatmap(
            z=processed_image,
            colorscale='Gray',
            showscale=False,
            name='MRI Scan'
        ),
        row=1, col=2
    )
    
    # Only add tumor overlay if a tumor is detected
    if prediction['type'] != 'No Tumor' and 'bbox' in prediction['features']:
        # Get bounding box from prediction
        bbox = prediction['features']['bbox']
        x, y = bbox['x'], bbox['y']
        w, h = bbox['width'], bbox['height']
        
        # Create rectangle shape for bounding box
        fig.add_shape(
            type="rect",
            x0=x - w/2,
            y0=y - h/2,
            x1=x + w/2,
            y1=y + h/2,
            line=dict(
                color="red",
                width=2,
            ),
            fillcolor="rgba(255, 0, 0, 0.2)",
            row=1, col=2
        )
        
        # Add tumor probability heatmap
        tumor_mask = np.zeros_like(processed_image)
        
        # Create a circle in the tumor region
        for i in range(processed_image.shape[0]):
            for j in range(processed_image.shape[1]):
                # Calculate distance from center of tumor
                dist = np.sqrt((i - y)**2 + (j - x)**2)
                if dist < max(w, h)/2:
                    # Decreasing probability with distance from center
                    tumor_mask[i, j] = max(0, 1 - dist/(max(w, h)/2))
        
        # Add tumor overlay
        fig.add_trace(
            go.Heatmap(
                z=tumor_mask,
                colorscale='Reds',
                opacity=0.6,
                showscale=True,
                colorbar=dict(
                    title='Tumor Probability',
                    len=0.8,
                    thickness=20,
                    bgcolor='rgba(255,255,255,0.8)',
                    borderwidth=1,
                    x=1.1
                ),
                name='Tumor Detection'
            ),
            row=1, col=2
        )
        
        # Add annotation for tumor type
        fig.add_annotation(
            x=x,
            y=y,
            text=prediction['type'],
            showarrow=True,
            arrowhead=1,
            row=1, col=2
        )
        
    # Update layout
    fig.update_layout(
        title=f"Brain Tumor Analysis - {prediction['model_used'].upper()} Model",
        title_x=0.5,
        height=600,
        width=1100,
        showlegend=False,
    )
    
    # Update x and y axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    # Add model information
    confidence = prediction['confidence']
    tumor_type = prediction['type']
    model_used = prediction['model_used'].upper()
    
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        text=f"Model: {model_used} | Prediction: {tumor_type} | Confidence: {confidence:.2f}%",
        showarrow=False,
        font=dict(size=14),
    )
    
    return fig

def create_comparison_visualization(processed_image, predictions):
    """
    Create visualization comparing results from multiple models
    """
    # Create figure with subplots - one for each model
    models = list(predictions.keys())
    num_models = len(models)
    
    fig = make_subplots(
        rows=1, 
        cols=num_models,
        subplot_titles=[model.upper() for model in models],
        column_widths=[1/num_models] * num_models
    )
    
    # For each model, add visualization
    for i, model_name in enumerate(models):
        prediction = predictions[model_name]
        col = i + 1
        
        # Add original image
        fig.add_trace(
            go.Heatmap(
                z=processed_image,
                colorscale='Gray',
                showscale=False,
                name=f'MRI Scan ({model_name})'
            ),
            row=1, col=col
        )
        
        # Only add tumor overlay if a tumor is detected
        if prediction['type'] != 'No Tumor' and 'bbox' in prediction['features']:
            # Get bounding box from prediction
            bbox = prediction['features']['bbox']
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']
            
            # Create rectangle shape for bounding box
            fig.add_shape(
                type="rect",
                x0=x - w/2,
                y0=y - h/2,
                x1=x + w/2,
                y1=y + h/2,
                line=dict(
                    color="red",
                    width=2,
                ),
                fillcolor="rgba(255, 0, 0, 0.2)",
                row=1, col=col
            )
            
            # Add annotation for tumor type and confidence
            fig.add_annotation(
                x=x,
                y=y + h/2 + 20,
                text=f"{prediction['type']} ({prediction['confidence']:.1f}%)",
                showarrow=False,
                font=dict(color="red", size=12),
                row=1, col=col
            )
    
    # Update layout
    fig.update_layout(
        title="Model Comparison for Brain Tumor Detection",
        title_x=0.5,
        height=500,
        width=200 * num_models,
        showlegend=False,
    )
    
    # Update x and y axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig
