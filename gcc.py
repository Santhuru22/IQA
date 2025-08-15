import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from pathlib import Path
import os
import base64
from io import BytesIO
from IPython.display import display, HTML
from google.colab import files, output
import matplotlib.pyplot as plt

class ImageQualityColabGUI:
    def __init__(self):
        """Initialize the Colab-compatible Image Quality Classifier"""
        self.model = None
        self.model_config = None
        self.current_image = None
        self.current_image_name = None

        # Setup the interface
        self.setup_colab_interface()

        # Automatically load the model after setting up the interface
        # Removed automatic call to load_model_callback here

    def setup_colab_interface(self):
        """Setup the Colab web interface"""
        html_interface = """
        <div id="image-quality-app" style="max-width: 1200px; margin: 20px auto; padding: 20px; font-family: Arial, sans-serif;">
            <h1 style="color: #2c3e50; text-align: center; margin-bottom: 30px;">
                🖼️ Image Quality Classifier
            </h1>

            <!-- Model Status Section -->
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                <h3 style="color: #495057; margin-top: 0;">📁 Model Configuration</h3>
                <div id="model-status" style="color: #dc3545; font-weight: bold;">
                    ❌ No model loaded
                </div>
                <button id="load-model-btn" style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-top: 10px;">
                    Load Model
                </button>
            </div>

            <!-- Image Upload Section -->
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
                <h3 style="color: #495057; margin-top: 0;">📤 Image Upload & Prediction</h3>
                <button id="upload-image-btn" style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                    Upload Image
                </button>
                <button id="predict-btn" style="background: #ffc107; color: #212529; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                    Predict Quality
                </button>
                <button id="clear-btn" style="background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Clear
                </button>
            </div>

            <!-- Results Section -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <!-- Image Display -->
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                    <h3 style="color: #495057; margin-top: 0;">🖼️ Image Preview</h3>
                    <div id="image-display" style="text-align: center; min-height: 300px; display: flex; align-items: center; justify-content: center; background: white; border: 2px dashed #dee2e6; border-radius: 5px;">
                        <span style="color: #6c757d;">No image selected</span>
                    </div>
                </div>

                <!-- Results Display -->
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px;">
                    <h3 style="color: #495057; margin-top: 0;">📊 Results</h3>
                    <div id="results-display" style="background: white; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; min-height: 300px; font-family: 'Courier New', monospace; font-size: 12px; overflow-y: auto; white-space: pre-wrap; color: #000000;">
                        <span style="color: #6c757d;">No results yet</span>
                    </div>
                </div>
            </div>

            <!-- Status Bar -->
            <div id="status-bar" style="background: #e9ecef; padding: 10px 20px; margin-top: 20px; border-radius: 5px; color: #495057;">
                Ready
            </div>
        </div>
        """

        display(HTML(html_interface))

        # Register the callback functions immediately after displaying HTML
        self.register_callbacks()

    def register_callbacks(self):
        """Register JavaScript callbacks with Python functions"""
        # Register callbacks using output.register_callback
        output.register_callback('notebook.load_model', self.load_model_callback)
        output.register_callback('notebook.upload_image', self.upload_image_callback)
        output.register_callback('notebook.predict_quality', self.predict_image_quality)
        output.register_callback('notebook.clear_results', self.clear_all)

        # Add JavaScript to handle button clicks
        js_code = """
        <script>
        (function() {
            function updateStatus(message, color = '#495057') {
                const statusBar = document.getElementById('status-bar');
                if (statusBar) {
                    statusBar.innerHTML = message;
                    statusBar.style.color = color;
                }
            }

            function updateModelStatus(message, isLoaded = false) {
                const statusElement = document.getElementById('model-status');
                if (statusElement) {
                    statusElement.innerHTML = message;
                    statusElement.style.color = isLoaded ? '#28a745' : '#dc3545';
                }
            }

            function displayImage(imageData, filename) {
                const imageDisplay = document.getElementById('image-display');
                if (imageDisplay) {
                    imageDisplay.innerHTML = `
                        <div>
                            <img src="data:image/jpeg;base64,${imageData}"
                                 style="max-width: 100%; max-height: 300px; border-radius: 5px;"
                                 alt="${filename}">
                            <div style="margin-top: 10px; color: #6c757d; font-size: 14px;">${filename}</div>
                        </div>
                    `;
                }
            }

            function displayResults(results) {
                const resultsDisplay = document.getElementById('results-display');
                if (resultsDisplay) {
                    resultsDisplay.innerHTML = results;
                    resultsDisplay.style.color = '#000000'; // Set text color to black
                }
            }

            function clearDisplay() {
                const imageDisplay = document.getElementById('image-display');
                const resultsDisplay = document.getElementById('results-display');
                if (imageDisplay) {
                    imageDisplay.innerHTML = '<span style="color: #6c757d;">No image selected</span>';
                }
                if (resultsDisplay) {
                    resultsDisplay.innerHTML = '<span style="color: #6c757d;">No results yet</span>';
                    resultsDisplay.style.color = '#6c757d'; // Reset color to default gray
                }
            }

            // Make functions globally available
            window.updateStatus = updateStatus;
            window.updateModelStatus = updateModelStatus;
            window.displayImage = displayImage;
            window.displayResults = displayResults;
            window.clearDisplay = clearDisplay;

            // Add event listeners to buttons
            const loadModelBtn = document.getElementById('load-model-btn');
            const uploadImageBtn = document.getElementById('upload-image-btn');
            const predictBtn = document.getElementById('predict-btn');
            const clearBtn = document.getElementById('clear-btn');

            if (loadModelBtn) {
                loadModelBtn.addEventListener('click', function() {
                    updateStatus('Loading model...', '#007bff');
                    google.colab.kernel.invokeFunction('notebook.load_model', [], {});
                });
            }

            if (uploadImageBtn) {
                uploadImageBtn.addEventListener('click', function() {
                    updateStatus('Preparing file upload...', '#28a745');
                    google.colab.kernel.invokeFunction('notebook.upload_image', [], {});
                });
            }

            if (predictBtn) {
                predictBtn.addEventListener('click', function() {
                    updateStatus('Making prediction...', '#ffc107');
                    google.colab.kernel.invokeFunction('notebook.predict_quality', [], {});
                });
            }

            if (clearBtn) {
                clearBtn.addEventListener('click', function() {
                    clearDisplay();
                    updateStatus('Cleared results', '#6c757d');
                    google.colab.kernel.invokeFunction('notebook.clear_results', [], {});
                });
            }

            console.log('Image Quality Classifier interface ready!');
        })();
        </script>
        """

        display(HTML(js_code))

    def load_model_from_path(self, model_dir="/content/IQA"):
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

            # Load model
            self.model = tf.keras.models.load_model(str(model_path))

            # Load configuration
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)

            return True, f"Model loaded successfully from: {model_dir}"

        except Exception as e:
            return False, f"Failed to load model: {str(e)}"

    def load_model_callback(self):
        """Callback function to load model"""
        success, message = self.load_model_from_path()

        if success:
            model_info = self.get_model_info()
            # Use output.eval_js to execute JavaScript
            output.eval_js(f'''
                updateModelStatus("✅ Model loaded successfully", true);
                updateStatus("{message}", "#28a745");
                displayResults(`{model_info}`);
            ''')
        else:
            output.eval_js(f'''
                updateModelStatus("❌ {message}", false);
                updateStatus("Failed to load model", "#dc3545");
            ''')


    def get_model_info(self):
        """Get model information as formatted text"""
        if not self.model_config:
            return "No model configuration available"

        info_text = "=== MODEL INFORMATION ===\\n\\n"
        info_text += f"Training Date: {self.model_config.get('training_date', 'Unknown')}\\n"
        info_text += f"Image Size: {self.model_config.get('img_size', [224, 224])}\\n"
        info_text += f"Classes: {', '.join(self.model_config.get('class_names', ['bad', 'good']))}\\n\\n"

        metrics = self.model_config.get('metrics', {})
        info_text += "=== MODEL PERFORMANCE ===\\n\\n"
        info_text += f"Accuracy: {metrics.get('accuracy', 0):.4f}\\n"
        info_text += f"Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}\\n"
        info_text += f"Validation Loss: {metrics.get('val_loss', 0):.4f}\\n\\n"

        if 'classification_report' in metrics:
            report = metrics['classification_report']
            info_text += "=== CLASSIFICATION REPORT ===\\n\\n"
            for class_name in ['bad', 'good']:
                if class_name in report:
                    class_metrics = report[class_name]
                    info_text += f"{class_name.upper()}:\\n"
                    info_text += f"  Precision: {class_metrics.get('precision', 0):.4f}\\n"
                    info_text += f"  Recall: {class_metrics.get('recall', 0):.4f}\\n"
                    info_text += f"  F1-Score: {class_metrics.get('f1-score', 0):.4f}\\n\\n"

        return info_text

    def upload_image_callback(self):
        """Callback function to upload image"""
        try:
            print("📤 Please select an image file to upload...")
            uploaded = files.upload()

            if not uploaded:
                output.eval_js('updateStatus("No file uploaded", "#dc3545");')
                return

            # Get the first uploaded file
            filename = list(uploaded.keys())[0]
            file_content = uploaded[filename]

            # Save temporarily and load with PIL
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(file_content)

            # Load and convert to base64 for display
            image = Image.open(temp_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize for display (maintain aspect ratio)
            display_size = (400, 300)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Store for prediction
            self.current_image = temp_path
            self.current_image_name = filename

            output.eval_js(f'''
                displayImage("{img_base64}", "{filename}");
                updateStatus("Image uploaded: {filename}", "#28a745");
            ''')

        except Exception as e:
            output.eval_js(f'updateStatus("Error uploading image: {str(e)}", "#dc3545");')

    def preprocess_image(self, image_path):
        """Preprocess image for model prediction"""
        if not self.model_config:
            raise ValueError("Model configuration not loaded")

        img_size = tuple(self.model_config.get('img_size', [224, 224]))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        return image

    def predict_image_quality(self):
        """Predict image quality"""
        if not self.model:
            # Use output.eval_js to update status
            output.eval_js('updateStatus("Please load a model first!", "#dc3545");')
            return False, "Please load a model first!"

        if not self.current_image:
            # Use output.eval_js to update status
            output.eval_js('updateStatus("Please upload an image first!", "#dc3545");')
            return False, "Please upload an image first!"

        try:
            processed_image = self.preprocess_image(self.current_image)
            prediction = self.model.predict(processed_image, verbose=0)[0][0]

            class_names = self.model_config.get('class_names', ['bad', 'good'])
            predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
            confidence = prediction if prediction > 0.5 else 1 - prediction

            # Format results
            results_text = "=== PREDICTION RESULTS ===\n\n"
            results_text += f"Image: {self.current_image_name}\n\n"
            results_text += f"Predicted Quality: {predicted_class.upper()}\n"
            results_text += f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)\n"
            results_text += f"Raw Score: {prediction:.6f}\n\n"

            results_text += "=== INTERPRETATION ===\n\n"
            if predicted_class.lower() == 'good':
                results_text += f"✅ The image appears to be of GOOD quality.\n"
                results_text += f"The model is {confidence*100:.1f}% confident in this prediction.\n"
            else:
                results_text += f"❌ The image appears to be of BAD quality.\n"
                results_text += f"The model is {confidence*100:.1f}% confident in this prediction.\n"

            results_text += f"\nNote: Scores closer to 1.0 indicate good quality,\n"
            results_text += f"while scores closer to 0.0 indicate bad quality.\n"

            # Use output.eval_js to display results
            output.eval_js(f'displayResults(`{results_text}`);')
            output.eval_js('updateStatus("Prediction completed successfully", "#28a745");')

            return True, results_text

        except Exception as e:
            # Use output.eval_js to update status
            output.eval_js(f'updateStatus("Prediction failed: {str(e)}", "#dc3545");')
            return False, f"Prediction failed: {str(e)}"


    def clear_all(self):
        """Callback function to clear results"""
        self.current_image = None
        self.current_image_name = None

        info_text = self.get_model_info() if self.model_config else "No results yet"

        output.eval_js('clearDisplay();')
        output.eval_js(f'displayResults(`{info_text}`);')
        output.eval_js('updateStatus("Cleared all results", "#6c757d");')

        return True, info_text # Returning a tuple for consistency, though only info_text is used currently


# Create and initialize the GUI
print("🚀 Initializing Image Quality Classifier for Google Colab...")
print("=" * 60)

# Create the GUI instance
colab_gui = ImageQualityColabGUI()

print("\n📋 Setup Instructions:")
print("1. Make sure you have your model files in /content/IQA/")
print("   - image_quality_model.h5")
print("   - model_config.json")
print("2. Click 'Load Model' to load your trained model")
print("3. Click 'Upload Image' to select an image for prediction")
print("4. Click 'Predict Quality' to classify the image")
print("=" * 60)
print("✅ Interface loaded successfully! Buttons are now functional.")
