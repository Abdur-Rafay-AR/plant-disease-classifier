"""
Plant Disease Classification Model Training Script

This script provides a template for training a plant disease classification model.
Currently contains placeholder code for the training pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
from datetime import datetime

class PlantDiseaseTrainer:
    """Plant Disease Model Training Class"""
    
    def __init__(self, data_dir: str = "data", img_size: tuple = (224, 224), batch_size: int = 32):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing training data
            img_size: Target image size (width, height)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        self.class_indices = None
        
    def create_data_generators(self):
        """Create data generators for training, validation, and testing."""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # No augmentation for validation/test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        self.validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Test generator (if test directory exists)
        test_dir = os.path.join(self.data_dir, 'test')
        if os.path.exists(test_dir):
            self.test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
        
        # Store class indices
        self.class_indices = self.train_generator.class_indices
        
        print(f"Found {self.train_generator.samples} training samples")
        print(f"Found {self.validation_generator.samples} validation samples")
        if self.test_generator:
            print(f"Found {self.test_generator.samples} test samples")
        print(f"Number of classes: {len(self.class_indices)}")
    
    def create_model(self, num_classes: int):
        """
        Create the CNN model architecture.
        
        Args:
            num_classes: Number of output classes
        """
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created successfully!")
        print(self.model.summary())
    
    def train_model(self, epochs: int = 50, save_path: str = "model"):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            save_path: Directory to save the trained model
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(save_path, 'Plant_Disease_Detection.h5')
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save class indices
        class_indices_path = os.path.join(save_path, 'class_indices.json')
        with open(class_indices_path, 'w') as f:
            json.dump(self.class_indices, f, indent=2)
        print(f"Class indices saved to: {class_indices_path}")
        
        # Plot training history
        self.plot_training_history(history, save_path)
        
        return history
    
    def plot_training_history(self, history, save_path: str):
        """Plot and save training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.show()
    
    def evaluate_model(self):
        """Evaluate the model on test data."""
        if self.test_generator is None:
            print("No test data available for evaluation.")
            return
        
        # Predictions
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # True classes
        true_classes = self.test_generator.classes
        class_labels = list(self.test_generator.class_indices.keys())
        
        # Classification report
        print("Classification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=class_labels))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        self.plot_confusion_matrix(cm, class_labels)
    
    def plot_confusion_matrix(self, cm, class_labels):
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

def main():
    """Main training function."""
    print("Plant Disease Classification Model Training")
    print("=" * 50)
    
    # NOTE: This is a template script. You need to:
    # 1. Download and organize your dataset in the 'data' directory
    # 2. Structure should be: data/train/class_name/images.jpg
    # 3. Optionally: data/test/class_name/images.jpg
    
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please create the data directory and organize your dataset.")
        print("Expected structure:")
        print("data/")
        print("  train/")
        print("    class1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    class2/")
        print("      image1.jpg")
        print("      ...")
        print("  test/ (optional)")
        print("    class1/")
        print("      ...")
        return
    
    # Initialize trainer
    trainer = PlantDiseaseTrainer(data_dir=data_dir)
    
    # Create data generators
    print("Creating data generators...")
    trainer.create_data_generators()
    
    # Create model
    num_classes = len(trainer.class_indices)
    print(f"Creating model for {num_classes} classes...")
    trainer.create_model(num_classes)
    
    # Train model
    print("Starting training...")
    history = trainer.train_model(epochs=50)
    
    # Evaluate model
    print("Evaluating model...")
    trainer.evaluate_model()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
