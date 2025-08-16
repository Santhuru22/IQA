import json
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
import tempfile
import requests
import os

# Only import what's absolutely necessary
REQUIRED_MODULES = {
    'tensorflow': False,
    'matplotlib': False,
}

@st.cache_data
def check_dependencies():
    """Check which dependencies are available"""
    results = {}
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        results['tensorflow'] = (True, tf)
    except ImportError as e:
        results['tensorflow'] = (False, str(e))
    
    # Check matplotlib (optional)
    try:
        import matplotlib.pyplot as plt
        results['matplotlib'] = (True, plt)
    except ImportError:
        results['matplotlib'] = (False, "Not available")
    
    return results

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Image Quality Classifier",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check dependencies
    deps = check_dependencies()
    
    # Check TensorFlow (required)
    if not deps['tensorflow'][0]:
        st.error("❌ **TensorFlow is required but not available!**")
        st.error(f"Error details: {deps['tensorflow'][1]}")
        
        st.markdown("""
        ### 🛠️ Fix TensorFlow Installation
        
        **For Streamlit Cloud deployment**, update your `requirements.txt`:
        ```txt
        streamlit
        tensorflow-cpu==2.13.1
        numpy==1.24.3
        Pillow
        requests
        ```
        
        **For local development**:
        ```bash
        pip install tensorflow-cpu==2.13.1
        ```
        
        **Alternative lightweight requirements.txt**:
        ```txt
        streamlit==1.28.0
        tensorflow-cpu==2.13.1
        numpy==1.24.3
        Pillow==9.5.0
        requests==2.31.0
        ```
        
        ### 🔄 After updating requirements.txt:
        1. Push changes to GitHub
        2. Redeploy your Streamlit app
        3. Wait for installation to complete
        """)
        st.stop()
    
    # Get TensorFlow module
    tf = deps['tensorflow'][1]
    
    # Show dependency status
    with st.sidebar:
        st.markdown("### 📦 Dependencies")
        st.success("✅ TensorFlow: Available")
        if deps['matplotlib'][0]:
            st.success("✅ Matplotlib: Available")
        else:
            st.info("ℹ️ Matplotlib: Not needed")

    class ImageQualityApp:
        def __init__(self):
            self.model = None
            self.model_config = None
            
            # GitHub repository settings
            self.github_user = "Santhuru22"
            self.github_repo = "IQA"
            self.github_branch = "main"
            self.model_filename = "image_quality_model.h5"
            self.config_filename = "model_config.json"
        
        def get_github_raw_url(self, filename):
            """Generate GitHub raw file URL"""
            return f"https://raw.githubusercontent.com/{self.github_user}/{self.github_repo}/{self.github_branch}/{filename}"
        
        def download_file_from_github(self, filename):
            """Download file from GitHub repository"""
            url = self.get_github_raw_url(filename)
            
            try:
                with st.spinner(f"Downloading {filename}..."):
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                return True, response.content
            except requests.exceptions.RequestException as e:
                return False, f"Failed to download {filename}: {str(e)}"
        
        @st.cache_resource
        def load_model_from_github(_self):
            """Load model and config from GitHub with caching"""
            try:
                # Download model file
                success, model_content = _self.download_file_from_github(_self.model_filename)
                if not success:
                    return False, model_content
                
                # Download configuration
                success, config_content = _self.download_file_from_github(_self.config_filename)
                if not success:
                    return False, config_content
                
                # Create temporary file for model
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                    tmp_file.write(model_content)
                    tmp_path = tmp_file.name
                
                # Load model
                model = tf.keras.models.load_model(tmp_path)
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                # Parse configuration
                config = json.loads(config_content.decode('utf-8'))
                
                return True, (model, config)
                
            except Exception as e:
                return False, f"Failed to load model: {str(e)}"
        
        def preprocess_image(self, image, config):
            """Preprocess image for model prediction"""
            img_size = tuple(config.get('img_size', [224, 224]))
            
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Handle different image formats
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]  # Convert to RGB
                elif img_array.shape[2] == 1:  # Grayscale with single channel
                    img_array = np.stack([img_array[:, :, 0]] * 3, axis=-1)
            elif len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Resize using PIL (more reliable than cv2)
            pil_img = Image.fromarray(img_array.astype('uint8'))
            pil_img = pil_img.resize(img_size, Image.Resampling.LANCZOS)
            
            # Convert back to numpy and normalize
            processed = np.array(pil_img, dtype=np.float32) / 255.0
            
            # Add batch dimension
            return np.expand_dims(processed, axis=0)
        
        def predict_quality(self, image, model, config):
            """Predict image quality"""
            try:
                # Preprocess image
                processed_image = self.preprocess_image(image, config)
                
                # Make prediction
                with st.spinner("Making prediction..."):
                    prediction = model.predict(processed_image, verbose=0)[0][0]
                
                # Interpret results
                class_names = config.get('class_names', ['bad', 'good'])
                predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
                confidence = prediction if prediction > 0.5 else 1 - prediction
                
                return True, {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'raw_score': prediction,
                    'class_names': class_names
                }
                
            except Exception as e:
                return False, f"Prediction failed: {str(e)}"

    # Initialize app
    if 'app_instance' not in st.session_state:
        st.session_state.app_instance = ImageQualityApp()

    app = st.session_state.app_instance

    # Main title
    st.title("🖼️ Image Quality Classifier")
    st.markdown("*Automatically detects good vs bad quality images using AI*")
    st.markdown("---")

    # Sidebar for model management
    with st.sidebar:
        st.header("🤖 Model Management")
        
        # Model loading button
        if st.button("📥 Load Model from GitHub", type="primary", use_container_width=True):
            success, result = app.load_model_from_github()
            
            if success:
                st.session_state.model, st.session_state.config = result
                st.success("✅ Model loaded successfully!")
                st.rerun()
            else:
                st.error(f"❌ {result}")

        # Model status
        model_loaded = hasattr(st.session_state, 'model') and st.session_state.model is not None
        
        if model_loaded:
            st.success("✅ **Model Status:** Ready")
            
            # Model information
            config = st.session_state.config
            st.markdown("**Model Details:**")
            st.write(f"• Classes: {', '.join(config.get('class_names', ['bad', 'good']))}")
            st.write(f"• Input Size: {config.get('img_size', [224, 224])}")
            
            # Performance metrics
            metrics = config.get('metrics', {})
            if metrics:
                st.markdown("**Performance:**")
                if 'accuracy' in metrics:
                    st.write(f"• Accuracy: {metrics['accuracy']:.3f}")
                if 'val_accuracy' in metrics:
                    st.write(f"• Val Accuracy: {metrics['val_accuracy']:.3f}")
        else:
            st.warning("❌ **Model Status:** Not loaded")

        st.markdown("---")
        st.markdown(f"**Repository:** [{app.github_user}/{app.github_repo}](https://github.com/{app.github_user}/{app.github_repo})")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file to analyze",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            help="Supported formats: JPG, PNG, BMP, TIFF, WebP"
        )

        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.image(
                image, 
                caption=f"📁 {uploaded_file.name} ({image.size[0]}×{image.size[1]})",
                use_column_width=True
            )

    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded_file and model_loaded:
            if st.button("🚀 Analyze Image Quality", type="primary", use_container_width=True):
                success, result = app.predict_quality(
                    image, 
                    st.session_state.model, 
                    st.session_state.config
                )
                
                if success:
                    pred_class = result['predicted_class']
                    confidence = result['confidence']
                    raw_score = result['raw_score']
                    
                    # Results header
                    st.markdown("### 📊 Prediction Results")
                    
                    # Quality status
                    if pred_class.lower() == 'good':
                        st.success(f"✅ **Image Quality: {pred_class.upper()}**")
                        quality_color = "🟢"
                    else:
                        st.error(f"❌ **Image Quality: {pred_class.upper()}**")
                        quality_color = "🔴"
                    
                    # Metrics display
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.1%}",
                            help="How confident the model is in its prediction"
                        )
                    with metric_col2:
                        st.metric(
                            label="Raw Score",
                            value=f"{raw_score:.4f}",
                            help="Raw prediction score (0=bad, 1=good)"
                        )
                    
                    # Confidence bar
                    st.markdown("**Confidence Level:**")
                    st.progress(confidence)
                    
                    # Detailed interpretation
                    st.markdown("### 💡 Interpretation")
                    st.markdown(f"""
                    {quality_color} **Analysis:** The model predicts this image has **{pred_class.upper()}** quality.
                    
                    **Confidence:** The model is **{confidence:.1%}** confident in this assessment.
                    
                    **Technical Details:**
                    - Raw prediction score: `{raw_score:.4f}`
                    - Threshold: `0.5` (scores > 0.5 = good, ≤ 0.5 = bad)
                    - Model classes: `{', '.join(result['class_names'])}`
                    """)
                    
                    # Additional info
                    if confidence < 0.7:
                        st.warning("⚠️ **Note:** Low confidence prediction. The image quality may be borderline.")
                    elif confidence > 0.9:
                        st.info("🎯 **Note:** High confidence prediction. The model is very certain about this assessment.")
                
                else:
                    st.error(f"❌ **Error:** {result}")
                    st.markdown("Please try with a different image or reload the model.")
        
        elif uploaded_file and not model_loaded:
            st.warning("⚠️ **Please load the model first** using the button in the sidebar.")
            
        elif not uploaded_file:
            st.info("📝 **Upload an image** on the left to see analysis results here.")

    # Auto-load model on first visit
    if not model_loaded and 'auto_load_attempted' not in st.session_state:
        st.session_state.auto_load_attempted = True
        
        with st.spinner("🔄 Auto-loading model from GitHub..."):
            success, result = app.load_model_from_github()
            
            if success:
                st.session_state.model, st.session_state.config = result
                st.success("🎉 Model auto-loaded successfully!")
                st.rerun()
            else:
                st.info("ℹ️ Auto-load failed. Please use the 'Load Model' button in the sidebar.")

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; padding: 10px;'>
            <p>
                🖼️ <strong>Image Quality Classifier</strong> | 
                Developed by <strong>Santhuru S</strong> | 
                <a href="https://github.com/{app.github_user}/{app.github_repo}" target="_blank">View on GitHub</a>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
