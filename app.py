import json
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
import tempfile
import requests
import os
import warnings
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

# Suppress all warnings and errors from displaying to users
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stderr to suppress TensorFlow warnings
sys.stderr = StringIO()

# Try to import optional dependencies silently
def silent_import(module_name):
    try:
        return __import__(module_name), True
    except ImportError:
        return None, False

# Import TensorFlow silently
tf, HAS_TF = silent_import('tensorflow')
if HAS_TF:
    # Suppress TensorFlow logging
    tf.get_logger().setLevel('ERROR')

# Import CV2 silently  
cv2, HAS_CV2 = silent_import('cv2')

class ImageQualityStreamlitApp:
    def __init__(self):
        """Initialize the Streamlit Image Quality Analyser"""
        self.model = None
        self.model_config = None
        
        # GitHub repository configuration
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
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return True, response.content
        except requests.exceptions.RequestException:
            return False, f"Unable to download {filename}. Please try manual upload."
    
    @st.cache_resource
    def load_model_from_github(_self):
        """Load the trained model and configuration from GitHub repository"""
        if not HAS_TF:
            return False, "TensorFlow is required but not available. Please check your installation."
            
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model_file:
                tmp_model_file.write(model_content)
                tmp_model_path = tmp_model_file.name
            
            # Load the model with error suppression
            with redirect_stderr(StringIO()):
                model = tf.keras.models.load_model(tmp_model_path)
            
            # Clean up temporary file
            os.unlink(tmp_model_path)
            
            # Load configuration
            config = json.loads(config_content.decode('utf-8'))
            
            return True, (model, config)
            
        except Exception:
            return False, "Failed to load model. Please try manual upload."
    
    def load_model_from_upload(self, model_file, config_file):
        """Load the trained model and configuration from uploaded files"""
        if not HAS_TF:
            return False, "TensorFlow is required but not available."
            
        try:
            # Save the uploaded model file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_model_file:
                tmp_model_file.write(model_file.getvalue())
                tmp_model_path = tmp_model_file.name
            
            # Load the model with error suppression
            with redirect_stderr(StringIO()):
                self.model = tf.keras.models.load_model(tmp_model_path)
            
            # Clean up the temporary model file
            os.unlink(tmp_model_path)

            # Load configuration from uploaded .json file
            config_buffer = BytesIO(config_file.getvalue())
            config_buffer.seek(0)
            self.model_config = json.load(config_buffer)

            return True, "Model loaded successfully from uploaded files"

        except Exception:
            return False, "Failed to load model from uploaded files."

    def get_model_info(self):
        """Get model information as formatted text"""
        if not self.model_config:
            return "No model configuration available"

        try:
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
        except Exception:
            return "Error displaying model information"

    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            if not self.model_config:
                raise ValueError("Model configuration not loaded")

            img_size = tuple(self.model_config.get('img_size', [224, 224]))

            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Ensure RGB format
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 4:  # RGBA
                    image_rgb = image_array[:, :, :3]
                elif image_array.shape[2] == 3:  # RGB
                    image_rgb = image_array
                else:
                    image_rgb = image_array
            elif len(image_array.shape) == 2:  # Grayscale
                image_rgb = np.stack([image_array] * 3, axis=-1)
            else:
                image_rgb = image_array
                
            # Resize image
            if HAS_CV2:
                image_resized = cv2.resize(image_rgb, img_size)
            else:
                # Use PIL for resizing (fallback)
                pil_img = Image.fromarray(image_rgb.astype('uint8'))
                pil_img = pil_img.resize(img_size, Image.Resampling.LANCZOS)
                image_resized = np.array(pil_img)
            
            # Normalize to [0,1] and add batch dimension
            image_processed = image_resized.astype(np.float32) / 255.0
            image_processed = np.expand_dims(image_processed, axis=0)

            return image_processed
            
        except Exception:
            raise ValueError("Failed to preprocess image")

    def predict_image_quality(self, image):
        """Predict image quality"""
        if not self.model:
            return False, "Please load a model first"

        try:
            processed_image = self.preprocess_image(image)
            
            # Make prediction with error suppression
            with redirect_stderr(StringIO()):
                prediction = self.model.predict(processed_image, verbose=0)[0][0]

            class_names = self.model_config.get('class_names', ['bad', 'good'])
            predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
            confidence = float(prediction if prediction > 0.5 else 1 - prediction)

            return True, {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'raw_score': float(prediction),
                'class_names': class_names
            }

        except Exception:
            return False, "Prediction failed. Please try with a different image."

def main():
    # Hide TensorFlow availability check from users
    if not HAS_TF:
        st.error("⚠️ This application requires TensorFlow to function properly.")
        st.markdown("""
        ### Please contact the administrator
        
        The required dependencies are not properly installed. 
        
        **For administrators:** Ensure your requirements.txt includes:
        ```
        tensorflow-cpu>=2.13.0
        streamlit
        numpy
        Pillow
        requests
        ```
        """)
        st.stop()

    # Configure Streamlit page
    st.set_page_config(
        page_title="Image Quality Classifier",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize the app
    if 'app' not in st.session_state:
        st.session_state.app = ImageQualityStreamlitApp()

    app = st.session_state.app

    # Main title
    st.title("🖼️ Image Quality Classifier")
    st.markdown("*Analyze your images to determine if they are good or bad quality*")
    st.markdown("---")

    # Sidebar for model loading
    with st.sidebar:
        st.header("🔧 Model Management")
        
        # GitHub auto-load section
        st.subheader("🚀 Auto-load from GitHub")
        st.markdown(f"**Repository:** `{app.github_user}/{app.github_repo}`")
        
        # Auto-load model button
        if st.button("📥 Load Model from GitHub", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                success, result = app.load_model_from_github()
                
                if success:
                    st.session_state.model, st.session_state.model_config = result
                    st.session_state.model_loaded = True
                    st.session_state.load_source = "github"
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load model from GitHub")
                    st.info("💡 Try the manual upload option below")
        
        st.markdown("---")
        
        # Manual upload fallback
        st.subheader("📁 Manual Upload")
        
        # File uploaders for model and config
        model_file = st.file_uploader("Upload Model File (.h5)", type=['h5'])
        config_file = st.file_uploader("Upload Configuration File (.json)", type=['json'])
        
        # Load model button
        if st.button("🔄 Load from Files", use_container_width=True) and model_file and config_file:
            with st.spinner("Loading model from files..."):
                success, message = app.load_model_from_upload(model_file, config_file)
                
                if success:
                    st.session_state.model_loaded = True
                    st.session_state.load_source = "upload"
                    st.success("✅ Model loaded from files!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load model from files")

        # Model status
        model_loaded = (hasattr(st.session_state, 'model_loaded') and 
                       st.session_state.model_loaded and 
                       hasattr(st.session_state, 'model') and 
                       st.session_state.model is not None)
        
        if model_loaded:
            source = getattr(st.session_state, 'load_source', 'unknown')
            st.success(f"✅ Model Status: Ready ({source})")
            
            # Update app instance
            if hasattr(st.session_state, 'model'):
                app.model = st.session_state.model
                app.model_config = st.session_state.model_config
        else:
            st.warning("❌ Model Status: Not loaded")

        # Model info
        if model_loaded and app.model_config:
            with st.expander("📊 Model Information", expanded=False):
                st.markdown(app.get_model_info())

    # Auto-load model on first run (silently)
    if 'auto_load_attempted' not in st.session_state:
        st.session_state.auto_load_attempted = True
        if not model_loaded:
            success, result = app.load_model_from_github()
            if success:
                st.session_state.model, st.session_state.model_config = result
                st.session_state.model_loaded = True
                st.session_state.load_source = "github"
                app.model = st.session_state.model
                app.model_config = st.session_state.model_config
                st.success("🎉 Model auto-loaded successfully!")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file to analyze",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            help="Upload an image to classify its quality"
        )

        # Display uploaded image
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                st.image(image, caption=f"📁 {uploaded_file.name}", use_column_width=True)
            except Exception:
                st.error("❌ Failed to load image. Please try a different file.")
                uploaded_file = None

    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded_file is not None and model_loaded and app.model is not None:
            # Predict button
            if st.button("🚀 Analyze Quality", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    success, result = app.predict_image_quality(image)
                    
                    if success:
                        try:
                            # Display prediction results
                            predicted_class = result['predicted_class']
                            confidence = result['confidence']
                            raw_score = result['raw_score']
                            
                            # Ensure confidence is a valid float for progress bar
                            confidence_value = max(0.0, min(1.0, float(confidence)))
                            
                            # Create metrics display
                            st.markdown("### 📊 Results")
                            
                            # Quality indicator
                            if predicted_class.lower() == 'good':
                                st.success(f"✅ **Quality: {predicted_class.upper()}**")
                            else:
                                st.error(f"❌ **Quality: {predicted_class.upper()}**")
                            
                            # Metrics
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("Confidence", f"{confidence_value*100:.1f}%")
                            with metric_col2:
                                st.metric("Raw Score", f"{raw_score:.4f}")
                            
                            # Progress bar for confidence (with safe value)
                            st.markdown("**Confidence Level:**")
                            st.progress(confidence_value)
                            
                            # Interpretation
                            st.markdown("### 💡 Interpretation")
                            if predicted_class.lower() == 'good':
                                st.markdown(f"""
                                🟢 The image appears to be of **GOOD** quality.
                                
                                The model is **{confidence_value*100:.1f}%** confident in this prediction.
                                """)
                            else:
                                st.markdown(f"""
                                🔴 The image appears to be of **BAD** quality.
                                
                                The model is **{confidence_value*100:.1f}%** confident in this prediction.
                                """)
                            
                            st.info("📝 Note: Scores closer to 1.0 indicate good quality, while scores closer to 0.0 indicate bad quality.")
                            
                        except Exception:
                            st.error("❌ Error displaying results. Please try again.")
                    else:
                        st.error(f"❌ {result}")
        
        elif uploaded_file is not None and not model_loaded:
            st.warning("⚠️ Please load a model first using the sidebar options.")
        
        elif uploaded_file is None:
            st.info("📝 Upload an image on the left to see analysis results here.")

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🖼️ Image Quality Classifier | 
            <a href="https://github.com/{app.github_user}/{app.github_repo}" target="_blank" style="text-decoration: none;">
            GitHub Repository</a></p>
            <p><strong>Developed by Santhuru S</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
