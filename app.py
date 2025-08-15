import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from pathlib import Path
import os
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

class ImageQualityStreamlitApp:
    def __init__(self):
        """Initialize the Streamlit Image Quality Classifier"""
        self.model = None
        self.model_config = None

    def load_model_from_path(self, model_dir):
        """Load the trained model and configuration from specified path"""
        try:
            if not os.path.exists(model_dir):
                return False, f"Model directory not found: {model_dir}"
            model_path = Path(model_dir) / 'image_quality_model.h5'
            config_path = Path(model_dir) / 'model_config.json'
            if not model_path.exists():
                return False, "Model file 'image_quality_model.h5' not found in the specified directory!"
            if not config_path.exists():
                return False, "Model configuration file 'model_config.json' not found!"
            @st.cache_resource
            def load_tf_model(model_path):
                return tf.keras.models.load_model(str(model_path))
            self.model = load_tf_model(model_path)
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            return True, f"Model loaded successfully from: {model_dir}"
        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    # ... [other class methods unchanged] ...

def main():
    MODEL_DIR = "./IQA"  # Set fixed model path

    st.set_page_config(
        page_title="Image Quality Classifier",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize the app if not already present in session_state
    if 'app' not in st.session_state:
        st.session_state.app = ImageQualityStreamlitApp()
        # Attempt to load model automatically on startup
        success, message = st.session_state.app.load_model_from_path(MODEL_DIR)
        st.session_state.model_loaded = success
        st.session_state.model_load_message = message

    app = st.session_state.app

    st.title("🖼️ Image Quality Classifier")
    st.markdown("---")

    # Sidebar for model configuration
    with st.sidebar:
        st.header("📁 Model Configuration")
        # Automatically show model load status/message
        if st.session_state.model_loaded:
            st.success("✅ Model loaded successfully!")
        else:
            st.error(st.session_state.model_load_message)

        if app.model_config:
            with st.expander("📊 Model Information", expanded=False):
                st.markdown(app.get_model_info())

    # Main content area (unchanged)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to classify its quality"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

    with col2:
        st.subheader("📊 Prediction Results")
        if uploaded_file is not None and app.model is not None:
            if st.button("🔮 Predict Quality", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    success, result = app.predict_image_quality(image)
                    if success:
                        predicted_class = result['predicted_class']
                        confidence = result['confidence']
                        raw_score = result['raw_score']
                        st.markdown("### Results")
                        if predicted_class.lower() == 'good':
                            st.success(f"✅ **Quality: {predicted_class.upper()}**")
                        else:
                            st.error(f"❌ **Quality: {predicted_class.upper()}**")
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        with metric_col2:
                            st.metric("Raw Score", f"{raw_score:.4f}")
                        st.markdown("**Confidence Level:**")
                        st.progress(confidence)
                        st.markdown("### Interpretation")
                        if predicted_class.lower() == 'good':
                            st.markdown(f"""
                            🟢 The image appears to be of **GOOD** quality.
                            The model is **{confidence*100:.1f}%** confident in this prediction.
                            """)
                        else:
                            st.markdown(f"""
                            🔴 The image appears to be of **BAD** quality.
                            The model is **{confidence*100:.1f}%** confident in this prediction.
                            """)
                        st.info("📝 Note: Scores closer to 1.0 indicate good quality, while scores closer to 0.0 indicate bad quality.")
                    else:
                        st.error(result)
        elif uploaded_file is not None and app.model is None:
            st.warning("⚠️ Model failed to load. See sidebar for details.")
        elif uploaded_file is None:
            st.info("📝 Upload an image to see prediction results here.")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🖼️ Image Quality Classifier | Built with Streamlit & TensorFlow</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
