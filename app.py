import json
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import requests
import os

# Check for TensorFlow availability
@st.cache_resource
def check_tensorflow():
    try:
        import tensorflow as tf
        return True, tf
    except ImportError as e:
        return False, str(e)

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Image Quality Classifier",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check TensorFlow availability
    has_tf, tf_result = check_tensorflow()
    
    if not has_tf:
        st.error("❌ TensorFlow is not available!")
        st.error(f"Error: {tf_result}")
        st.markdown("""
        ### Possible Solutions:
        1. **For Streamlit Cloud**: Try using `tensorflow-cpu` instead of `tensorflow`
        2. **Check your requirements.txt**:
        ```
        streamlit==1.29.0
        tensorflow-cpu==2.13.1
        numpy==1.24.3
        Pillow==10.0.1
        requests==2.31.0
        ```
        3. **For local development**: Install TensorFlow:
        ```bash
        pip install tensorflow-cpu==2.13.1
        ```
        """)
        st.stop()
    
    tf = tf_result

    class ImageQualityApp:
        def __init__(self):
            self.model = None
            self.model_config = None
            self.github_user = "Santhuru22"
            self.github_repo = "IQA"
            self.github_branch = "main"
            self.model_filename = "image_quality_model.h5"
            self.config_filename = "model_config.json"
        
        def get_github_raw_url(self, filename):
            return f"https://raw.githubusercontent.com/{self.github_user}/{self.github_repo}/{self.github_branch}/{filename}"
        
        def download_file_from_github(self, filename):
            url = self.get_github_raw_url(filename)
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                return True, response.content
            except requests.exceptions.RequestException as e:
                return False, f"Failed to download {filename}: {str(e)}"
        
        @st.cache_resource
        def load_model_from_github(_self):
            try:
                # Download model file
                success, model_content = _self.download_file_from_github(_self.model_filename)
                if not success:
                    return False, model_content
                
                # Download configuration file
                success, config_content = _self.download_file_from_github(_self.config_filename)
                if not success:
                    return False, config_content
                
                # Save model to temporary file and load it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                    tmp_file.write(model_content)
                    tmp_path = tmp_file.name
                
                # Load the model
                model = tf.keras.models.load_model(tmp_path)
                
                # Clean up
                os.unlink(tmp_path)
                
                # Load configuration
                config = json.loads(config_content.decode('utf-8'))
                
                return True, (model, config)
                
            except Exception as e:
                return False, f"Failed to load model: {str(e)}"
        
        def preprocess_image(self, image, config):
            img_size = tuple(config.get('img_size', [224, 224]))
            
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Ensure RGB
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            elif len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Resize using PIL
            pil_img = Image.fromarray(img_array.astype('uint8'))
            pil_img = pil_img.resize(img_size, Image.Resampling.LANCZOS)
            
            # Normalize and add batch dimension
            processed = np.array(pil_img).astype(np.float32) / 255.0
            return np.expand_dims(processed, axis=0)
        
        def predict_quality(self, image, model, config):
            try:
                processed = self.preprocess_image(image, config)
                prediction = model.predict(processed, verbose=0)[0][0]
                
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
    if 'app' not in st.session_state:
        st.session_state.app = ImageQualityApp()

    app = st.session_state.app

    # Title
    st.title("🖼️ Image Quality Classifier")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Model Status")
        
        if st.button("📥 Load Model from GitHub", type="primary"):
            with st.spinner("Loading model from GitHub..."):
                success, result = app.load_model_from_github()
                
                if success:
                    st.session_state.model, st.session_state.config = result
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error(f"❌ {result}")

        # Check if model is loaded
        model_loaded = hasattr(st.session_state, 'model') and st.session_state.model is not None
        
        if model_loaded:
            st.success("✅ Model Ready")
            config = st.session_state.config
            
            with st.expander("📊 Model Info"):
                st.write(f"**Classes:** {', '.join(config.get('class_names', ['bad', 'good']))}")
                st.write(f"**Image Size:** {config.get('img_size', [224, 224])}")
                metrics = config.get('metrics', {})
                if metrics:
                    st.write(f"**Accuracy:** {metrics.get('accuracy', 0):.3f}")
        else:
            st.warning("❌ No model loaded")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

    with col2:
        st.subheader("📊 Prediction Results")
        
        if uploaded_file and model_loaded:
            if st.button("🔮 Predict Quality", type="primary"):
                with st.spinner("Analyzing..."):
                    success, result = app.predict_quality(
                        image, 
                        st.session_state.model, 
                        st.session_state.config
                    )
                    
                    if success:
                        pred_class = result['predicted_class']
                        confidence = result['confidence']
                        raw_score = result['raw_score']
                        
                        st.markdown("### Results")
                        
                        if pred_class.lower() == 'good':
                            st.success(f"✅ **Quality: {pred_class.upper()}**")
                        else:
                            st.error(f"❌ **Quality: {pred_class.upper()}**")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        with col_b:
                            st.metric("Raw Score", f"{raw_score:.4f}")
                        
                        st.progress(confidence)
                        
                        st.markdown("### Interpretation")
                        quality_emoji = "🟢" if pred_class.lower() == 'good' else "🔴"
                        st.markdown(f"""
                        {quality_emoji} The image appears to be of **{pred_class.upper()}** quality.
                        
                        The model is **{confidence*100:.1f}%** confident in this prediction.
                        """)
                    else:
                        st.error(result)
        
        elif uploaded_file and not model_loaded:
            st.warning("⚠️ Please load the model first.")
        else:
            st.info("📝 Upload an image to see results.")

    # Auto-load model on startup
    if not model_loaded and 'auto_load_tried' not in st.session_state:
        st.session_state.auto_load_tried = True
        with st.spinner("Auto-loading model..."):
            success, result = app.load_model_from_github()
            if success:
                st.session_state.model, st.session_state.config = result
                st.success("✅ Model auto-loaded!")
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666;'>
            <p>🖼️ Image Quality Analyser | <a href="https://github.com/{app.github_user}/{app.github_repo}">GitHub</a></p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
