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

            # Load model with caching for better performance
            @st.cache_resource
            def load_tf_model(model_path):
                return tf.keras.models.load_model(str(model_path))

            self.model = load_tf_model(model_path)

            # Load configuration
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)

            return True, f"Model loaded successfully from: {model_dir}"

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def get_model_info(self):
        """Get model information as formatted text"""
        if not self.model_config:
            return "No model configuration available"

        info_text = "**MODEL INFORMATION**\n\n"
        info_text += f"- **Training Date:** {self.model_config.get('training_date', 'Unknown')}\n"
        info_text += f"- **Image Size:** {self.model_config.get('img_size', [224, 224])}\n"
        info_text += f"- **Classes:** {', '.join(self.model_config.get('class_names', ['bad', 'good']))}\n\n"

        metrics = self.model_config.get('metrics', {})
        info_text += "**MODEL PERFORMANCE**\n\n"
        info_text += f"- **Accuracy:** {metrics.get('accuracy', 0):.4f}\n"
        info_text += f"- **Validation Accuracy:** {metrics.get('val_accuracy', 0):.4f}\n"
        info_text += f"- **Validation Loss:** {metrics.get('val_loss', 0):.4f}\n\n"

        if 'classification_report' in metrics:
            report = metrics['classification_report']
            info_text += "**CLASSIFICATION REPORT**\n\n"
            for class_name in ['bad', 'good']:
                if class_name in report:
                    class_metrics = report[class_name]
                    info_text += f"**{class_name.upper()}:**\n"
                    info_text += f"- Precision: {class_metrics.get('precision', 0):.4f}\n"
                    info_text += f"- Recall: {class_metrics.get('recall', 0):.4f}\n"
                    info_text += f"- F1-Score: {class_metrics.get('f1-score', 0):.4f}\n\n"

        return info_text

    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        if not self.model_config:
            raise ValueError("Model configuration not loaded")

        img_size = tuple(self.model_config.get('img_size', [224, 224]))

        # Convert PIL image to numpy array
        image_array = np.array(image)
        
        # Ensure RGB format
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = image_array
        else:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
        # Resize image
        image_resized = cv2.resize(image_rgb, img_size)
        image_processed = image_resized.astype(np.float32) / 255.0
        image_processed = np.expand_dims(image_processed, axis=0)  # Add batch dimension

        return image_processed

    def predict_image_quality(self, image):
        """Predict image quality"""
        if not self.model:
            return False, "Model not loaded!"

        try:
            processed_image = self.preprocess_image(image)
            prediction = self.model.predict(processed_image, verbose=0)[0][0]

            class_names = self.model_config.get('class_names', ['bad', 'good'])
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

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Image Quality Classifier",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize the app and auto-load model
    if 'app' not in st.session_state:
        st.session_state.app = ImageQualityStreamlitApp()
        st.session_state.model_loaded = False

    app = st.session_state.app

    # Auto-load model if not already loaded
    if not st.session_state.model_loaded:
        model_dir = "./"  # Model files are in the root directory
        with st.spinner("Loading model..."):
            success, message = app.load_model_from_path(model_dir)
            if success:
                st.session_state.model_loaded = True
                st.session_state.model_load_message = message
                st.session_state.model_load_success = True
            else:
                st.session_state.model_loaded = False
                st.session_state.model_load_message = message
                st.session_state.model_load_success = False

    # Main title
    st.title("🖼️ Image Quality Classifier")
    st.markdown("---")

    # Sidebar for model information
    with st.sidebar:
        st.header("📊 Model Status")
        
        # Model status display
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
            st.info(f"📁 Loaded from: ./")
        else:
            st.error("❌ Model Load Failed")
            st.error(st.session_state.model_load_message)

        # Model info
        if app.model_config:
            with st.expander("📊 Model Information", expanded=False):
                st.markdown(app.get_model_info())

    # Show initial load status message
    if hasattr(st.session_state, 'model_load_message'):
        if st.session_state.model_load_success:
            st.success(st.session_state.model_load_message)
        else:
            st.error(st.session_state.model_load_message)
            st.stop()  # Stop execution if model failed to load

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Image Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to classify its quality"
        )

        # Display uploaded image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

    with col2:
        st.subheader("📊 Prediction Results")
        
        if uploaded_file is not None:
            # Predict button
            if st.button("🔮 Predict Quality", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    success, result = app.predict_image_quality(image)
                    
                    if success:
                        # Display prediction results
                        predicted_class = result['predicted_class']
                        confidence = result['confidence']
                        raw_score = result['raw_score']
                        
                        # Create metrics display
                        st.markdown("### Results")
                        
                        # Quality indicator
                        if predicted_class.lower() == 'good':
                            st.success(f"✅ **Quality: {predicted_class.upper()}**")
                        else:
                            st.error(f"❌ **Quality: {predicted_class.upper()}**")
                        
                        # Metrics
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        with metric_col2:
                            st.metric("Raw Score", f"{raw_score:.4f}")
                        
                        # Progress bar for confidence
                        st.markdown("**Confidence Level:**")
                        st.progress(confidence)
                        
                        # Interpretation
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
        
        elif uploaded_file is None:
            st.info("📝 Upload an image to see prediction results here.")

    # Footer
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
