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
        """Load the trained model and configuration from specified path with improved error handling"""
        try:
            if not os.path.exists(model_dir):
                return False, f"Model directory not found: {model_dir}"

            model_path = Path(model_dir) / 'image_quality_model.h5'
            config_path = Path(model_dir) / 'model_config.json'

            if not model_path.exists():
                return False, "Model file 'image_quality_model.h5' not found in the specified directory!"

            if not config_path.exists():
                return False, "Model configuration file 'model_config.json' not found!"

            # Load configuration first
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)

            # Try different model loading approaches
            success, model, error_msg = self._try_load_model(model_path)
            
            if success:
                self.model = model
                return True, f"Model loaded successfully from: {model_dir}"
            else:
                return False, f"Failed to load model: {error_msg}"

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def _try_load_model(self, model_path):
        """Try different approaches to load the model"""
        
        # Approach 1: Standard loading with compile=False
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return True, model, None
        except Exception as e1:
            st.warning(f"Standard loading failed: {str(e1)}")
        
        # Approach 2: Load with custom objects (if needed)
        try:
            model = tf.keras.models.load_model(str(model_path), custom_objects={}, compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return True, model, None
        except Exception as e2:
            st.warning(f"Custom objects loading failed: {str(e2)}")
        
        # Approach 3: Try loading weights only (if this fails, we'll create a reconstruction method)
        try:
            # This approach would require the model architecture to be defined
            # For now, we'll return the error
            return False, None, f"All loading approaches failed. Last error: {str(e2)}"
        except Exception as e3:
            return False, None, str(e3)

    def create_model_from_config(self):
        """Create a new model from configuration (if available)"""
        # This is a placeholder for creating a model from scratch
        # You would implement this based on your specific model architecture
        st.info("Model reconstruction from config is not yet implemented. Please ensure your .h5 file is compatible.")
        return None

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

        # Add model architecture info if available
        if self.model:
            info_text += "**MODEL ARCHITECTURE**\n\n"
            info_text += f"- **Total Parameters:** {self.model.count_params():,}\n"
            info_text += f"- **Input Shape:** {self.model.input_shape}\n"
            info_text += f"- **Output Shape:** {self.model.output_shape}\n"
            info_text += f"- **Number of Layers:** {len(self.model.layers)}\n\n"

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
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # Convert RGBA to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
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
            return False, "Please load a model first!"

        try:
            processed_image = self.preprocess_image(image)
            
            # Make prediction with error handling
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Handle different output shapes
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Multi-class output
                prediction_value = np.max(prediction[0])
                predicted_class_idx = np.argmax(prediction[0])
            else:
                # Binary output
                prediction_value = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]
                predicted_class_idx = 1 if prediction_value > 0.5 else 0

            class_names = self.model_config.get('class_names', ['bad', 'good'])
            predicted_class = class_names[predicted_class_idx] if predicted_class_idx < len(class_names) else 'unknown'
            confidence = prediction_value if predicted_class_idx == 1 else 1 - prediction_value

            return True, {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'raw_score': float(prediction_value),
                'class_names': class_names,
                'prediction_array': prediction.tolist()
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

    # Initialize the app
    if 'app' not in st.session_state:
        st.session_state.app = ImageQualityStreamlitApp()

    app = st.session_state.app

    # Main title
    st.title("🖼️ Image Quality Classifier")
    st.markdown("---")

    # Sidebar for model configuration
    with st.sidebar:
        st.header("📁 Model Configuration")
        
        # Model path input with suggestions
        st.markdown("**Common paths:**")
        st.markdown("- `./` (current directory)")
        st.markdown("- `./models` (models subfolder)")
        st.markdown("- `./IQA` (IQA subfolder)")
        
        model_dir = st.text_input(
            "Model Directory Path",
            value="./",
            help="Path to directory containing image_quality_model.h5 and model_config.json"
        )
        
        # Load model button
        if st.button("🔄 Load Model", type="primary", use_container_width=True):
            with st.spinner("Loading model..."):
                success, message = app.load_model_from_path(model_dir)
                
                if success:
                    st.success(message)
                    st.session_state.model_loaded = True
                    st.rerun()  # Refresh to show model info
                else:
                    st.error(message)
                    st.session_state.model_loaded = False

        # Model status
        if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
            st.success("✅ Model loaded successfully!")
        else:
            st.warning("❌ No model loaded")

        # Debug information
        with st.expander("🔧 Debug Information", expanded=False):
            st.write("**Directory Contents:**")
            try:
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    for file in files:
                        file_path = os.path.join(model_dir, file)
                        if os.path.isfile(file_path):
                            st.write(f"📄 {file}")
                        else:
                            st.write(f"📁 {file}/")
                else:
                    st.write("Directory does not exist")
            except Exception as e:
                st.write(f"Error reading directory: {e}")

        # Model info
        if app.model_config:
            with st.expander("📊 Model Information", expanded=False):
                st.markdown(app.get_model_info())

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
            try:
                image = Image.open(uploaded_file)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                
                # Show image info
                st.markdown("**Image Information:**")
                st.write(f"- **Size:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"- **Mode:** {image.mode}")
                st.write(f"- **Format:** {uploaded_file.type}")
                
            except Exception as e:
                st.error(f"Error loading image: {e}")

    with col2:
        st.subheader("📊 Prediction Results")
        
        if uploaded_file is not None and app.model is not None:
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
                        
                        # Advanced results
                        with st.expander("🔬 Advanced Results", expanded=False):
                            st.json({
                                "predicted_class": predicted_class,
                                "confidence": confidence,
                                "raw_score": raw_score,
                                "class_names": result['class_names'],
                                "prediction_array": result['prediction_array']
                            })
                        
                        # Interpretation
                        st.markdown("### Interpretation")
                        if predicted_class.lower() == 'good':
                            st.markdown(f"""
                            🟢 The image appears to be of **GOOD** quality.
                            
                            The model is **{confidence*100:.1f}%** confident in this prediction.
                            
                            This suggests the image has good clarity, proper exposure, and minimal artifacts.
                            """)
                        else:
                            st.markdown(f"""
                            🔴 The image appears to be of **BAD** quality.
                            
                            The model is **{confidence*100:.1f}%** confident in this prediction.
                            
                            This may indicate issues like blur, poor lighting, compression artifacts, or other quality problems.
                            """)
                        
                        # Confidence interpretation
                        if confidence > 0.9:
                            confidence_text = "Very High Confidence"
                            confidence_color = "🟢"
                        elif confidence > 0.7:
                            confidence_text = "High Confidence"
                            confidence_color = "🟡"
                        elif confidence > 0.5:
                            confidence_text = "Moderate Confidence"
                            confidence_color = "🟠"
                        else:
                            confidence_text = "Low Confidence"
                            confidence_color = "🔴"
                        
                        st.info(f"{confidence_color} **{confidence_text}** - The model's certainty level for this prediction.")
                        
                    else:
                        st.error(f"Prediction failed: {result}")
        
        elif uploaded_file is not None and app.model is None:
            st.warning("⚠️ Please load a model first to make predictions.")
            st.info("Use the sidebar to specify your model directory path and click 'Load Model'.")
        
        elif uploaded_file is None:
            st.info("📝 Upload an image to see prediction results here.")
            
            # Show sample instructions
            st.markdown("### How to use:")
            st.markdown("""
            1. **Load Model:** Set the correct path to your model files in the sidebar
            2. **Upload Image:** Choose an image file (JPG, PNG, etc.)
            3. **Get Prediction:** Click the 'Predict Quality' button
            4. **View Results:** See the quality classification and confidence score
            """)

    # Troubleshooting section
    if not hasattr(st.session_state, 'model_loaded') or not st.session_state.model_loaded:
        st.markdown("---")
        with st.expander("🆘 Troubleshooting Guide", expanded=False):
            st.markdown("""
            ### Common Issues and Solutions:
            
            **1. "Layer 'dense' expects 1 input(s), but received 2"**
            - This error occurs with model architecture mismatches
            - Try loading with `compile=False` (handled automatically in this version)
            - Ensure TensorFlow versions match between training and inference
            
            **2. "Model directory not found"**
            - Check if the path is correct
            - Use `./` for current directory
            - Ensure both .h5 and .json files are present
            
            **3. "Model file not found"**
            - Verify file names: `image_quality_model.h5` and `model_config.json`
            - Check file permissions
            
            **4. Loading is slow**
            - Large models take time to load
            - Consider using model quantization for faster loading
            
            ### File Structure Examples:
            ```
            Option 1 (same directory):
            ├── app.py
            ├── image_quality_model.h5
            ├── model_config.json
            
            Option 2 (subdirectory):
            ├── app.py
            ├── models/
            │   ├── image_quality_model.h5
            │   └── model_config.json
            ```
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🖼️ Image Quality Classifier | Built with Streamlit & TensorFlow</p>
            <p><em>Enhanced with improved error handling and debugging features</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
