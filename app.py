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
        
        # Approach 1: Load weights only and reconstruct architecture
        try:
            st.info("🔄 Trying to reconstruct model from weights...")
            model = self._reconstruct_model_architecture()
            if model is not None:
                model.load_weights(str(model_path))
                return True, model, None
        except Exception as e0:
            st.warning(f"Architecture reconstruction failed: {str(e0)}")
        
        # Approach 2: Standard loading with compile=False
        try:
            st.info("🔄 Trying standard model loading...")
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
        
        # Approach 3: Load with safe_mode (TF 2.11+)
        try:
            st.info("🔄 Trying safe mode loading...")
            if hasattr(tf.keras.models, 'load_model'):
                # Try with safe_mode if available
                model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                return True, model, None
        except Exception as e2:
            st.warning(f"Safe mode loading failed: {str(e2)}")
        
        # Approach 4: Try to load just the weights and create a simple model
        try:
            st.info("🔄 Trying simple model reconstruction...")
            model = self._create_simple_model()
            if model is not None:
                # Try to load weights - this might fail if architectures don't match
                model.load_weights(str(model_path))
                return True, model, None
        except Exception as e3:
            st.warning(f"Simple model reconstruction failed: {str(e3)}")
        
        # Approach 5: Manual architecture fix
        try:
            st.info("🔄 Trying manual architecture fix...")
            return self._try_manual_fix(model_path)
        except Exception as e4:
            return False, None, f"All loading approaches failed. Last error: {str(e4)}"

    def _reconstruct_model_architecture(self):
        """Reconstruct a common image quality model architecture"""
        try:
            if not self.model_config:
                return None
                
            img_size = self.model_config.get('img_size', [224, 224])
            
            # Common architecture for image quality assessment
            # Base model (typically a pre-trained CNN)
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(*img_size, 3),
                weights='imagenet',
                include_top=False,
                pooling='avg'
            )
            
            # Freeze base model
            base_model.trainable = False
            
            # Add custom classification head
            inputs = tf.keras.Input(shape=(*img_size, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='quality_prediction')(x)
            
            model = tf.keras.Model(inputs, outputs)
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            st.warning(f"Failed to reconstruct MobileNetV2 architecture: {e}")
            return None

    def _create_simple_model(self):
        """Create a simple CNN model for image quality"""
        try:
            if not self.model_config:
                return None
                
            img_size = self.model_config.get('img_size', [224, 224])
            
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(*img_size, 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except Exception as e:
            st.warning(f"Failed to create simple model: {e}")
            return None

    def _try_manual_fix(self, model_path):
        """Try to manually fix the architecture issue"""
        try:
            import h5py
            
            # Try to inspect the model file structure
            with h5py.File(model_path, 'r') as f:
                # Get model config from the h5 file
                if 'model_config' in f.attrs:
                    model_config_str = f.attrs['model_config']
                    if isinstance(model_config_str, bytes):
                        model_config_str = model_config_str.decode('utf-8')
                    
                    model_config_dict = json.loads(model_config_str)
                    
                    # Try to fix the architecture by removing problematic connections
                    fixed_config = self._fix_model_config(model_config_dict)
                    
                    # Create model from fixed config
                    model = tf.keras.models.model_from_json(json.dumps(fixed_config))
                    
                    # Load weights (this might still fail, but worth trying)
                    model.load_weights(str(model_path))
                    
                    return True, model, None
                    
        except Exception as e:
            st.warning(f"Manual fix attempt failed: {e}")
            
        return False, None, "Manual fix failed"

    def _try_weights_only_loading(self, model_dir, architecture_choice):
        """Try to load only weights into a new architecture"""
        try:
            model_path = Path(model_dir) / 'image_quality_model.h5'
            img_size = self.model_config.get('img_size', [224, 224])
            
            # Create model based on architecture choice
            if architecture_choice == "MobileNetV2":
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(*img_size, 3),
                    weights='imagenet',
                    include_top=False
                )
                base_model.trainable = False
                inputs = tf.keras.Input(shape=(*img_size, 3))
                x = base_model(inputs, training=False)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                model = tf.keras.Model(inputs, outputs)
                
            elif architecture_choice == "EfficientNetB0":
                base_model = tf.keras.applications.EfficientNetB0(
                    input_shape=(*img_size, 3),
                    weights='imagenet',
                    include_top=False
                )
                base_model.trainable = False
                inputs = tf.keras.Input(shape=(*img_size, 3))
                x = base_model(inputs, training=False)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                model = tf.keras.Model(inputs, outputs)
                
            elif architecture_choice == "ResNet50":
                base_model = tf.keras.applications.ResNet50(
                    input_shape=(*img_size, 3),
                    weights='imagenet',
                    include_top=False
                )
                base_model.trainable = False
                inputs = tf.keras.Input(shape=(*img_size, 3))
                x = base_model(inputs, training=False)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                model = tf.keras.Model(inputs, outputs)
                
            else:  # Simple CNN
                model = self._create_simple_model()
            
            # Try to load weights
            model.load_weights(str(model_path))
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            return True
            
        except Exception as e:
            st.error(f"Weight loading failed: {e}")
            return False

    def _fix_model_config(self, config):
        """Attempt to fix problematic model configurations"""
        # This is a basic fix - you might need to customize based on your specific model
        if 'config' in config and 'layers' in config['config']:
            layers = config['config']['layers']
            
            # Look for problematic dense layer connections
            for layer in layers:
                if layer.get('class_name') == 'Dense':
                    # Ensure dense layer only has one input
                    if 'config' in layer and 'batch_input_shape' in layer['config']:
                        # Fix input shape issues
                        pass
                    
                    # Fix inbound nodes if they have multiple inputs
                    if 'inbound_nodes' in layer:
                        for node in layer['inbound_nodes']:
                            if len(node) > 1 and len(node[0]) > 1:
                                # Keep only the first input
                                node[0] = [node[0][0]]
        
        return config

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

        # Alternative loading method for problematic models
        st.markdown("### 🛠️ Alternative Loading Method")
        if st.button("🔧 Try Alternative Model Creation", use_container_width=True):
            with st.spinner("Creating alternative model..."):
                try:
                    # Create a new model with common architecture
                    if app.model_config:
                        img_size = app.model_config.get('img_size', [224, 224])
                        
                        # Create a working model
                        base_model = tf.keras.applications.MobileNetV2(
                            input_shape=(*img_size, 3),
                            weights='imagenet',
                            include_top=False
                        )
                        base_model.trainable = False
                        
                        inputs = tf.keras.Input(shape=(*img_size, 3))
                        x = base_model(inputs, training=False)
                        x = tf.keras.layers.GlobalAveragePooling2D()(x)
                        x = tf.keras.layers.Dropout(0.2)(x)
                        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                        
                        new_model = tf.keras.Model(inputs, outputs)
                        new_model.compile(
                            optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        # Set the model (without trained weights)
                        app.model = new_model
                        st.session_state.model_loaded = True
                        
                        st.success("✅ Alternative model created! Note: This uses pre-trained ImageNet weights, not your trained weights.")
                        st.info("This model will give predictions but may not match your trained model's performance.")
                        
                    else:
                        st.error("Model config not available for alternative creation")
                        
                except Exception as e:
                    st.error(f"Alternative model creation failed: {e}")

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
