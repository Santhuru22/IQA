import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from pathlib import Path
import os

class ImageQualityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Quality Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.model_config = None
        self.current_image_path = None
        self.processed_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_label = tk.Label(
            self.root, 
            text="Image Quality Classifier", 
            font=("Arial", 20, "bold"),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Model loading frame
        model_frame = ttk.LabelFrame(self.root, text="Model Configuration", padding=10)
        model_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(
            model_frame, 
            text="Load Model", 
            command=self.load_model
        ).pack(side='left', padx=5)
        
        self.model_status_label = tk.Label(
            model_frame, 
            text="No model loaded", 
            fg='red',
            bg='#f0f0f0'
        )
        self.model_status_label.pack(side='left', padx=20)
        
        # Image upload frame
        upload_frame = ttk.LabelFrame(self.root, text="Image Upload", padding=10)
        upload_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(
            upload_frame, 
            text="Select Image", 
            command=self.select_image
        ).pack(side='left', padx=5)
        
        ttk.Button(
            upload_frame, 
            text="Predict Quality", 
            command=self.predict_quality
        ).pack(side='left', padx=5)
        
        ttk.Button(
            upload_frame, 
            text="Clear", 
            command=self.clear_results
        ).pack(side='left', padx=5)
        
        # Image display frame
        display_frame = ttk.LabelFrame(self.root, text="Image Preview", padding=10)
        display_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create canvas for image display
        self.image_canvas = tk.Canvas(
            display_frame, 
            bg='white', 
            width=400, 
            height=300
        )
        self.image_canvas.pack(side='left', padx=10, pady=10)
        
        # Results frame
        results_frame = ttk.Frame(display_frame)
        results_frame.pack(side='right', fill='both', expand=True, padx=10)
        
        # Prediction result
        ttk.Label(
            results_frame, 
            text="Prediction Results:", 
            font=("Arial", 14, "bold")
        ).pack(anchor='w', pady=(0, 10))
        
        self.result_text = tk.Text(
            results_frame, 
            height=15, 
            width=30,
            font=("Consolas", 10),
            wrap=tk.WORD
        )
        self.result_text.pack(fill='both', expand=True)
        
        # Scrollbar for text
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root, 
            text="Ready", 
            relief=tk.SUNKEN, 
            anchor='w',
            bg='#e0e0e0'
        )
        self.status_bar.pack(side='bottom', fill='x')
        
    def load_model(self):
        """Load the trained model and configuration from a hardcoded path."""
        try:
            # --- EDIT THIS LINE ---
            # Replace the placeholder path with the actual path to your model directory.
            # Use forward slashes '/' even on Windows, or double backslashes '\\'.
            
            # Example for Windows: model_dir = "C:\\Users\\YourUser\\Desktop\\MyModelFolder"
            # Example for macOS/Linux: model_dir = "/home/user/documents/my_model"
            
            model_dir = "D:/vislona/dataset"
            
            # --- END OF EDIT ---

            if not model_dir or not os.path.exists(model_dir):
                messagebox.showerror("Error", f"Model directory not found: {model_dir}\nPlease edit the path in the load_model function.")
                return
            
            model_path = Path(model_dir) / 'image_quality_model.h5'
            config_path = Path(model_dir) / 'model_config.json'
            
            if not model_path.exists():
                messagebox.showerror("Error", "Model file 'image_quality_model.h5' not found in the specified directory!")
                return
            
            if not config_path.exists():
                messagebox.showerror("Error", "Model configuration file 'model_config.json' not found!")
                return
            
            # Load model
            self.model = tf.keras.models.load_model(str(model_path))
            
            # Load configuration
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            
            self.model_status_label.config(text="Model loaded successfully", fg='green')
            self.status_bar.config(text=f"Model loaded from: {model_dir}")
            
            self.display_model_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def display_model_info(self):
        """Display model information in results area"""
        if not self.model_config:
            return
            
        info_text = "=== MODEL INFORMATION ===\n\n"
        info_text += f"Training Date: {self.model_config.get('training_date', 'Unknown')}\n"
        info_text += f"Image Size: {self.model_config.get('img_size', [224, 224])}\n"
        info_text += f"Classes: {', '.join(self.model_config.get('class_names', ['bad', 'good']))}\n\n"
        
        metrics = self.model_config.get('metrics', {})
        info_text += "=== MODEL PERFORMANCE ===\n\n"
        info_text += f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
        info_text += f"Validation Accuracy: {metrics.get('val_accuracy', 0):.4f}\n"
        info_text += f"Validation Loss: {metrics.get('val_loss', 0):.4f}\n\n"
        
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            info_text += "=== CLASSIFICATION REPORT ===\n\n"
            for class_name in ['bad', 'good']:
                if class_name in report:
                    class_metrics = report[class_name]
                    info_text += f"{class_name.upper()}:\n"
                    info_text += f"  Precision: {class_metrics.get('precision', 0):.4f}\n"
                    info_text += f"  Recall: {class_metrics.get('recall', 0):.4f}\n"
                    info_text += f"  F1-Score: {class_metrics.get('f1-score', 0):.4f}\n\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, info_text)
        
    def select_image(self):
        """Select an image file for prediction"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.status_bar.config(text=f"Image loaded: {Path(file_path).name}")
            
    def display_image(self, image_path):
        """Display the selected image on canvas"""
        try:
            image = Image.open(image_path)
            
            canvas_width = 400
            canvas_height = 300
            
            img_width, img_height = image.size
            aspect_ratio = img_width / img_height
            
            if aspect_ratio > canvas_width / canvas_height:
                display_width = canvas_width
                display_height = int(canvas_width / aspect_ratio)
            else:
                display_height = canvas_height
                display_width = int(canvas_height * aspect_ratio)
            
            display_image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(display_image)
            
            self.image_canvas.delete("all")
            
            x = (canvas_width - display_width) // 2
            y = (canvas_height - display_height) // 2
            
            self.image_canvas.create_image(x, y, anchor='nw', image=self.photo)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
            
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
        
    def predict_quality(self):
        """Predict image quality"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first!")
            return
            
        if not self.current_image_path:
            messagebox.showerror("Error", "Please select an image first!")
            return
            
        try:
            self.status_bar.config(text="Making prediction...")
            self.root.update()
            
            processed_image = self.preprocess_image(self.current_image_path)
            
            prediction = self.model.predict(processed_image, verbose=0)[0][0]
            
            class_names = self.model_config.get('class_names', ['bad', 'good'])
            predicted_class = class_names[1] if prediction > 0.5 else class_names[0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            self.display_prediction_results(predicted_class, confidence, prediction)
            
            self.status_bar.config(text="Prediction completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.status_bar.config(text="Prediction failed")
            
    def display_prediction_results(self, predicted_class, confidence, raw_score):
        """Display prediction results"""
        results_text = "=== PREDICTION RESULTS ===\n\n"
        results_text += f"Image: {Path(self.current_image_path).name}\n\n"
        results_text += f"Predicted Quality: {predicted_class.upper()}\n"
        results_text += f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)\n"
        results_text += f"Raw Score: {raw_score:.6f}\n\n"
        
        results_text += "=== INTERPRETATION ===\n\n"
        if predicted_class.lower() == 'good':
            results_text += f"✅ The image appears to be of GOOD quality.\n"
            results_text += f"The model is {confidence*100:.1f}% confident in this prediction.\n"
        else:
            results_text += f"❌ The image appears to be of BAD quality.\n"
            results_text += f"The model is {confidence*100:.1f}% confident in this prediction.\n"
            
        results_text += f"\nNote: Scores closer to 1.0 indicate good quality,\n"
        results_text += f"while scores closer to 0.0 indicate bad quality.\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, results_text)
        
    def clear_results(self):
        """Clear all results and reset interface"""
        self.current_image_path = None
        self.processed_image = None
        self.image_canvas.delete("all")
        self.result_text.delete(1.0, tk.END)
        
        if self.model_config:
            self.display_model_info()
            
        self.status_bar.config(text="Ready")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = ImageQualityGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
