import os
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

class ImageQualityClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the Image Quality Classifier
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = ['bad', 'good']  # 0: bad, 1: good
        
    def load_and_preprocess_data(self):
        """Load and preprocess images from the directory structure"""
        print("Loading and preprocessing data...")
        
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        self.train_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            classes=['bad', 'good']
        )
        
        self.val_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            classes=['bad', 'good']
        )
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.val_generator.samples} validation images")
        print(f"Class indices: {self.train_generator.class_indices}")
        
    def build_model(self):
        """Build the CNN model using transfer learning"""
        print("Building model...")
        
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        base_model.trainable = False
        
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.summary()
        
    def train_model(self, epochs=20):
        """Train the model"""
        print(f"Training model for {epochs} epochs...")
        
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.val_generator.samples // self.batch_size
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        print("Training completed!")
        
    def evaluate_model(self):
        """Evaluate the model and return metrics"""
        print("Evaluating model...")
        
        val_loss, val_accuracy = self.model.evaluate(self.val_generator, verbose=0)
        self.val_generator.reset()
        predictions = self.model.predict(self.val_generator, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        true_classes = self.val_generator.classes
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        cm = confusion_matrix(true_classes, predicted_classes)
        
        metrics = {
            'accuracy': float(accuracy),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        return metrics
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model_and_results(self, model_dir='model_output'):
        """Save the trained model and results"""
        print("Saving model and results...")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / 'image_quality_model.h5'
        self.model.save(model_path)
        
        metrics = self.evaluate_model()
        
        if self.history:
            history_data = {
                'history': {
                    'accuracy': [float(x) for x in self.history.history['accuracy']],
                    'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
                    'loss': [float(x) for x in self.history.history['loss']],
                    'val_loss': [float(x) for x in self.history.history['val_loss']]
                }
            }
        else:
            history_data = {'history': None}
            
        model_config = {
            'model_path': str(model_path),
            'img_size': self.img_size,
            'class_names': self.class_names,
            'training_date': datetime.now().isoformat(),
            'metrics': metrics,
            **history_data
        }
        
        config_path = model_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2, default=str)
            
        print(f"Model saved to: {model_path}")
        print(f"Configuration saved to: {config_path}")
        
        return str(model_path), str(config_path)

def main():
    """Main training pipeline"""
    DATA_DIR = "D:/vislona/dataset"  # <-- Your dataset path
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} does not exist!")
        return
        
    if not (os.path.exists(os.path.join(DATA_DIR, 'good')) and 
            os.path.exists(os.path.join(DATA_DIR, 'bad'))):
        print("Error: Dataset should contain 'good' and 'bad' folders!")
        return
    
    classifier = ImageQualityClassifier(DATA_DIR)
    classifier.load_and_preprocess_data()
    classifier.build_model()
    
    epochs = 20  # Default epochs
    classifier.train_model(epochs=epochs)
    classifier.plot_training_history()
    model_path, config_path = classifier.save_model_and_results()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Model saved at: {model_path}")
    print(f"Config saved at: {config_path}")

if __name__ == "__main__":
    main()
