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
import glob

class ImageQualityStreamlitApp:
    def __init__(self):
        """Initialize the Streamlit Image Quality Classifier"""
        self.model = None
        self.model_config = None
        self.model_path = None
        self.config_path = None
        
    def find_model_files(self):
        """Automatically find model files in the repository"""
        search_paths = [
            ".",  # Current directory
            "./models",
            "./IQA", 
            "../models",
            "../IQA",
            "./src",
            "./app"
        ]
        
        model_candidates = []
        config_candidates = []
        
        # Search in common directories
        for search_path in search_paths:
            if os.path.exists(search_path):
                # Look for .h5 files
                h5_files = glob.glob(os.path.join(search_path, "*.h5"))
                h5_files.extend(glob.glob(os.path.join(search_path, "**/*.h5"), recursive=True))
                
                # Look for config files
                json_files = glob.glob(os.path.join(search_path, "*config*.json"))
                json_files.extend(glob.glob(os.path.join(search_path, "**/*config*.json"), recursive=True))
                
                model_candidates.extend(h5_files)
                config_candidates.extend(json_files)
        
        # Filter for likely model files
        model_files = [f for f in model_candidates if any(keyword in os.path.basename(f).lower() 
                      for keyword in ['model', 'quality', 'image'])]
        
        config_files = [f for f in config_candidates if any(keyword in os.path.basename(f).lower() 
                       for keyword in ['model', 'config'])]
        
        return model_files, config_files
    
    def auto_load_models(self):
        """Automatically find and load models from the repository"""
        try:
            st.info("🔍 Scanning repository for model files...")
            
            model_files, config_files = self.find_model_files()
            
            if not model_files:
                return False, "No model (.h5) files found in repository"
            
            if not config_files:
                return False, "No configuration (.json) files found in repository"
            
            # Display found files
            st.success(f"📁 Found {len(model_files)} model file(s) and {len(config_files)} config file(s)")
            
            # Try to match model and config files
            best_model = None
            best_config = None
            
            # Look for exact matches first
            for model_file in model_files:
                model_name = Path(model_file).stem
                for config_file in config_files:
                    config_name = Path(config_file).stem
                    if model_name.lower() in config_name.lower() or config_name.lower() in model_name.lower():
                        best_model = model_file
                        best_config = config_file
                        break
                if best_model:
                    break
            
            # If no exact match, use the first available files
            if not best_model:
                best_model = model_files[0]
                best_config = config_files[0]
            
            st.info(f"📄 Using model: {os.path.basename(best_model)}")
            st.info(f"📄 Using config: {os.path.basename(best_config)}")
            
            # Load the configuration first
            with open(best_config, 'r') as f:
                self.model_config = json.load(f)
            
            # Load the model with error handling
            success, model, error_msg = self._safe_load_model(best_model)
            
            if success:
                self.model = model
                self.model_path = best_model
                self.config_path = best_config
                return True, f"Models loaded successfully!\nModel: {os.path.basename(best_model)}\nConfig: {os.path.basename(best_config)}"
            else:
                return False, f"Failed to load model: {error_msg}"
                
        except Exception as e:
            return False, f"Auto-loading failed: {str(e)}"
    
    def _safe_load_model(self, model_path):
        """Safely load model with multiple approaches"""
        
        # Approach 1: Load with compile=False (safest)
        try:
            st.info("🔄 Trying safe model loading...")
            model = tf.keras.models.load_model(str(model_path), compile=False)
            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return True, model, None
        except Exception as e1:
            st.warning(f"Safe loading failed: {str(e1)}")
        
        # Approach 2: Try with custom objects
        try:
            st.info("🔄 Trying custom objects loading...")
            model = tf.keras.models.load_model(str(model_path), custom_objects={}, compile=False)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return True, model, None
        except Exception as e2:
            st.warning(f"Custom objects loading failed: {str(e2)}")
        
        # Approach 3: Create compatible model and load weights
        try:
            st.info("🔄 Trying weight-only loading with compatible architecture...")
            compatible_model = self._create_compatible_model()
            if compatible_model:
                compatible_model.load_weights(str(model_path))
                return True, compatible_model, None
        except Exception as e3:
            st.warning(f"Weight loading failed: {str(e3)}")
        
        return False, None, f"All loading methods failed. Last error: {str(e3) if 'e3' in locals() else str(e2)}"
    
    def _create_compatible_model(self):
        """Create a compatible model architecture"""
        try:
            if not self.model_config:
                return None
                
            img_size = self.model_config.get('img_size', [224, 224])
            
            # Try MobileNetV2 architecture (common for image quality)
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(*img_size, 3),
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            
            base_model.trainable = False
            
            # Add custom head
            inputs = tf.keras.Input(shape=(*img_size, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            
            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            st.warning(f"Compatible model creation failed: {e}")
            return None

    def get_model_info(self):
        """Get model information as formatted text"""
        if not self.model_config:
            return "No model configuration available"

        info_text = "**MODEL INFORMATION**\n\n"
        info_text += f"- **Model Path:** {os.path.basename(self.model_path) if self.model_path else 'Unknown'}\n"
        info_text += f"- **Config Path:** {os.path.basename(self.config_path) if self.config_path else 'Unknown'}\n"
        info_text += f"- **Training Date:** {self.model_config.get('training_date', 'Unknown')}\n"
        info_text += f"- **Image Size:** {self.model_config.get('img_size', [224, 224])}\n"
        info_text += f"- **Classes:** {', '.join(self.model_config.get('class_names', ['bad', 'good']))}\n\n"

        metrics = self.model_config.get('metrics', {})
        if metrics:
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
            try:
                info_text += f"- **Total Parameters:** {self.model.count_params():,}\n"
                info_text += f"- **Input Shape:** {self.model.input_shape}\n"
                info_text += f"- **Output Shape:** {self.model.output_shape}\n"
                info_text += f"- **Number of Layers:** {len(self.model.layers)}\n\n"
            except:
                info_text += "- **Architecture details:** Unable to retrieve\n\n"

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
            return False, "Please wait for model to load automatically!"

        try:
            processed_image = self.preprocess_image(image)
            
            # Make prediction with error handling
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Handle different output shapes
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Multi-class output
                prediction_value = np.max(prediction[0])
                predicted_class_idx = 1 if prediction_value > 0.5 else 0
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
                'prediction_array': prediction.tolist() if hasattr(prediction, 'tolist') else prediction
            }

        except Exception as e:
            return False, f"Prediction failed: {str(e)}"

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Image Quality Classifier - Auto Loading",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize the app
    if 'app' not in st.session_state:
        st.session_state.app = ImageQualityStreamlitApp()
        st.session_state.auto_loaded = False

    app = st.session_state.app

    # Main title
    st.title("🖼️ Image Quality Classifier")
    st.subheader("🤖 Automatic Model Loading")
    st.markdown("---")

    # Auto-load models on startup
    if not st.session_state.auto_loaded:
        with st.spinner("🔍 Auto-loading models from repository..."):
            success, message = app.auto_load_models()
            
            if success:
                st.success("✅ " + message)
                st.session_state.model_loaded = True
                st.session_state.auto_loaded = True
                st.rerun()
            else:
                st.error("❌ " + message)
                st.session_state.model_loaded = False
                st.session_state.auto_loaded = True

    # Sidebar for model information and manual controls
    with st.sidebar:
        st.header("📊 Model Status")
        
        # Model status
        if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
            st.success("✅ Models loaded automatically!")
            
            # Show loaded files
            if app.model_path and app.config_path:
                st.info(f"📄 **Model:** {os.path.basename(app.model_path)}")
                st.info(f"📄 **Config:** {os.path.basename(app.config_path)}")
        else:
            st.error("❌ Auto-loading failed")
        
        # Manual reload button
        st.markdown("### 🔄 Manual Controls")
        if st.button("🔄 Reload Models", use_container_width=True):
            st.session_state.auto_loaded = False
            st.rerun()
        
        # File browser for debugging
        with st.expander("🔍 Repository Files", expanded=False):
            st.markdown("**Found in repository:**")
            try:
                model_files, config_files = app.find_model_files()
                
                st.markdown("**Model files (.h5):**")
                for f in model_files:
                    st.write(f"📄 {os.path.relpath(f)}")
                
                st.markdown("**Config files (.json):**")
                for f in config_files:
                    st.write(f"📄 {os.path.relpath(f)}")
                    
                if not model_files and not config_files:
                    st.write("❌ No model files found")
                    
            except Exception as e:
                st.write(f"❌ Error scanning files: {e}")

        # Model information
        if app.model_config:
            with st.expander("📊 Model Information", expanded=False):
                st.markdown(app.get_model_info())

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Image Upload")
        
        # Instructions
        if not (hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded):
            st.warning("⏳ Please wait for models to load automatically...")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to classify its quality",
            disabled=not (hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded)
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
        
        if uploaded_file is not None and hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded and app.model is not None:
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
                        
                        # Quality indicator with enhanced styling
                        if predicted_class.lower() == 'good':
                            st.success(f"✅ **Quality: {predicted_class.upper()}**")
                            quality_color = "🟢"
                        else:
                            st.error(f"❌ **Quality: {predicted_class.upper()}**")
                            quality_color = "🔴"
                        
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
                                "model_used": os.path.basename(app.model_path) if app.model_path else "Unknown"
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
                            confidence_emoji = "🟢"
                        elif confidence > 0.7:
                            confidence_text = "High Confidence" 
                            confidence_emoji = "🟡"
                        elif confidence > 0.5:
                            confidence_text = "Moderate Confidence"
                            confidence_emoji = "🟠"
                        else:
                            confidence_text = "Low Confidence"
                            confidence_emoji = "🔴"
                        
                        st.info(f"{confidence_emoji} **{confidence_text}** - The model's certainty level for this prediction.")
                        
                    else:
                        st.error(f"Prediction failed: {result}")
        
        elif uploaded_file is not None and not (hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded):
            st.warning("⚠️ Models are still loading. Please wait...")
            
        elif uploaded_file is None:
            st.info("📝 Upload an image to see prediction results here.")
            
            # Show usage instructions
            st.markdown("### 🚀 How it works:")
            st.markdown("""
            1. **Auto-Loading**: Models are automatically detected and loaded from the repository
            2. **Upload Image**: Choose any image file (JPG, PNG, etc.)
            3. **Get Results**: Click 'Predict Quality' for instant analysis
            4. **View Details**: Expand 'Advanced Results' for technical details
            
            **✨ Features:**
            - Automatic model detection and loading
            - Multiple file format support
            - Detailed confidence scoring
            - Advanced result analysis
            """)

    # Repository info and troubleshooting
    if not (hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded):
        st.markdown("---")
        with st.expander("🆘 Troubleshooting Auto-Loading", expanded=True):
            st.markdown("""
            ### Expected Repository Structure:
            
            The app automatically searches for these files:
            
            ```
            your_repository/
            ├── app.py (this file)
            ├── image_quality_model.h5 ⭐
            ├── model_config.json ⭐
            └── other files...
            ```
            
            **OR in subdirectories:**
            ```
            your_repository/
            ├── app.py
            ├── models/
            │   ├── image_quality_model.h5 ⭐
            │   └── model_config.json ⭐
            └── other files...
            ```
            
            ### Auto-Loading Process:
            1. 🔍 Scans common directories (., ./models, ./IQA, etc.)
            2. 📄 Finds .h5 model files and .json config files
            3. 🔗 Matches related files by name similarity
            4. 🤖 Loads models automatically with error handling
            
            ### If Auto-Loading Fails:
            - ✅ Ensure both files are in the repository
            - ✅ Check file names contain keywords: 'model', 'quality', 'config'
            - ✅ Verify file permissions are readable
            - 🔄 Try the 'Reload Models' button in the sidebar
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🖼️ Image Quality Classifier | Built with Streamlit & TensorFlow</p>
            <p><em>🤖 Featuring Automatic Model Detection and Loading</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
