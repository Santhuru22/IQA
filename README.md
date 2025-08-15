# Image Quality Analyzer

A machine learning-powered tool for automatically assessing and classifying image quality. This project uses deep learning techniques to distinguish between good and bad quality images, making it useful for content moderation, quality control, and automated image filtering.

## 🚀 Features

- **Automated Image Quality Assessment**: Classify images as good or bad quality
- **Deep Learning Model**: Trained neural network for accurate quality prediction
- **Batch Processing**: Analyze multiple images efficiently
- **GUI Interface**: User-friendly graphical interface for easy interaction
- **Training History Visualization**: Monitor model performance during training
- **Configurable Parameters**: Customizable model settings via JSON configuration

## 📁 Project Structure

```
dataset/
├── bad/                    # Low quality image samples
├── good/                   # High quality image samples
├── gcc.py                  # Streamlit deploy to host
├── data_collection.py      # Data preprocessing and collection utilities
├── gui.py                  # Graphical user interface
├── image_quality_model.h5  # Trained Keras/TensorFlow model
├── model.py               # Model architecture and training logic
├── model_config.json      # Configuration parameters
└── training_history.png   # Training performance visualization
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/image-quality-analyzer.git
   cd image-quality-analyzer
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Required Python packages**:
   - TensorFlow/Keras
   - OpenCV (cv2)
   - NumPy
   - Matplotlib
   - PIL (Pillow)
   - tkinter (for GUI)

## 🚀 Quick Start

### Using the GUI Application

Run the graphical interface for easy image quality analysis:

```bash
python gui.py
```

### Using the Model Programmatically

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('image_quality_model.h5')

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Adjust size based on your model
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Predict image quality
image = preprocess_image('path/to/your/image.jpg')
prediction = model.predict(image)
quality = "Good" if prediction[0] > 0.5 else "Bad"
print(f"Image quality: {quality}")
```

## 📊 Training Your Own Model

1. **Prepare your dataset**:
   - Place good quality images in the `good/` folder
   - Place bad quality images in the `bad/` folder

2. **Configure training parameters**:
   Edit `model_config.json` to adjust model settings:
   ```json
   {
     "image_size": [224, 224],
     "batch_size": 32,
     "epochs": 50,
     "learning_rate": 0.001,
     "validation_split": 0.2
   }
   ```

3. **Run data collection and preprocessing**:
   ```bash
   python data_collection.py
   ```

4. **Train the model**:
   ```bash
   python model.py
   ```

## 📈 Model Performance

The trained model's performance metrics and training history are visualized in `training_history.png`. This includes:
- Training and validation accuracy over epochs
- Training and validation loss curves
- Model convergence analysis

## 🔧 Configuration

Customize the model behavior by modifying `model_config.json`:

- `image_size`: Input image dimensions for the model
- `batch_size`: Number of images processed in each training batch
- `epochs`: Number of training iterations
- `learning_rate`: Optimizer learning rate
- `validation_split`: Percentage of data used for validation

## 📝 Usage Examples

### Batch Processing Multiple Images

```python
import os
import glob

# Process all images in a directory
image_folder = "path/to/images/"
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

for extension in image_extensions:
    for image_path in glob.glob(os.path.join(image_folder, extension)):
        image = preprocess_image(image_path)
        prediction = model.predict(image)
        quality = "Good" if prediction[0] > 0.5 else "Bad"
        print(f"{os.path.basename(image_path)}: {quality}")
```

### Quality Threshold Adjustment

```python
# Adjust confidence threshold for classification
def classify_quality(prediction, threshold=0.5):
    confidence = prediction[0]
    if confidence > threshold:
        return "Good", confidence
    else:
        return "Bad", 1 - confidence

prediction = model.predict(image)
quality, confidence = classify_quality(prediction, threshold=0.7)
print(f"Quality: {quality} (Confidence: {confidence:.2f})")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Thanks to the open-source community for the machine learning frameworks
- Dataset contributors for providing quality image samples
- TensorFlow/Keras team for the deep learning framework

## 📞 Contact

For questions, suggestions:
- Email: redsweety202@gmail.com
- GitHub: Santhuru22
- LinkedIn: https://www.linkedin.com/in/santhuru-s-05398b265

---

⭐ If you find this project helpful, please consider giving it a star on GitHub!
