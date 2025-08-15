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
            model_path = Path(model_dir) / 'image_quality_model.h5'
            config_path = Path(model_dir) / 'model_config.json'

            if not model_path.exists():
                return False, "Model file 'image_quality_model.h5' not found in the current directory!"

            if not config_path.exists():
                return False, "Configuration file 'model_config.json' not found in the current directory!"

            # Use st.cache_resource to load and cache the model only once
            @st.cache_resource
            def load_tf_model(path):
                return tf.keras.models.load_model(str(path))

            self.model = load_tf_model(model_path)

            with open(config_path, 'r') as f:
                self.model_config = json.load(f)

            return True, "Model loaded successfully!"

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
        image_array = np.array(image)
        
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = image_array
        else:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
        image_resized = cv2.resize(image_rgb, img_size)
        image_processed = image_resized.astype(np.float32) / 255.0
        image_processed = np.expand_dims(image_processed, axis=0)

        return image_processed

    def predict_image_quality(self, image):
        """Predict image quality"""
        if not self.model:
            return False, "Model is not available for prediction."

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

    # Initialize the app class in session state if it doesn't exist
    if 'app' not in st.session_state:
        st.session_state.app = ImageQualityStreamlitApp()
    app = st.session_state.app

    # --- AUTOMATIC MODEL LOADING ---
    # This block runs only once per session when the app starts.
    if 'model_loaded' not in st.session_state:
        with st.spinner("Loading model... Please wait."):
            # Load model from the current directory "."
            success, message = app.load_model_from_path(".")
            st.session_state.model_loaded = success
            st.session_state.load_message = message

    # Main title
    st.title("🖼️ Image Quality Classifier")
    st.markdown("---")

    # Sidebar for model status and information
    with st.sidebar:
        st.header("📊 Model Status")
        
        # Display the outcome of the automatic loading attempt
        if st.session_state.model_loaded:
            st.success("✅ Model loaded automatically!")
            # Display model info if loaded successfully
            if app.model_config:
                with st.expander("Show Model Information", expanded=True):
                    st.markdown(app.get_model_info())
        else:
            st.error(f"❌ Model Failed to Load\n\n**Reason:** {st.session_state.load_message}")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload an Image")
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
        st.subheader("📈 Prediction Results")
        
        # Display results only if model is loaded and file is uploaded
        if st.session_state.model_loaded and uploaded_file is not None:
            if st.button("🔮 Predict Quality", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    success, result = app.predict_image_quality(image)
                    
                    if success:
                        predicted_class = result['predicted_class']
                        confidence = result['confidence']
                        
                        if predicted_class.lower() == 'good':
                            st.success(f"**Quality: GOOD** (Confidence: {confidence:.2%})")
                        else:
                            st.error(f"**Quality: BAD** (Confidence: {confidence:.2%})")
                        
                        st.progress(confidence)
                        
                        st.info(f"The model's raw score is **{result['raw_score']:.4f}**. Scores closer to 1.0 indicate good quality, while scores closer to 0.0 indicate bad quality.")
                        
                    else:
                        st.error(result) # Display prediction error
        
        elif not st.session_state.model_loaded:
            st.warning("⚠️ Cannot predict because the model failed to load. Please check the files and restart the app.")
        
        else:
            st.info("📝 Upload an image and click 'Predict Quality' to see the results.")

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Built with Streamlit & TensorFlow</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
