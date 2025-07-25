
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os
from typing import Tuple, Optional

class PlantDiseasePredictor:
    """Plant Disease Prediction class with enhanced functionality."""
    
    def __init__(self, model_path: str = "model/Plant_Disease_Detection.h5", 
                 class_indices_path: str = "assets/class_indices.json"):
        """
        Initialize the predictor with model and class indices.
        
        Args:
            model_path: Path to the trained model file
            class_indices_path: Path to the class indices JSON file
        """
        self.model_path = model_path
        self.class_indices_path = class_indices_path
        self.model = None
        self.idx_to_label = None
        self._load_model_and_indices()
    
    def _load_model_and_indices(self):
        """Load the model and class indices."""
        try:
            # Load model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            
            # Load label mapping
            if not os.path.exists(self.class_indices_path):
                raise FileNotFoundError(f"Class indices file not found: {self.class_indices_path}")
            
            with open(self.class_indices_path, "r") as f:
                class_indices = json.load(f)
            
            # Convert string keys to integers for proper indexing
            self.idx_to_label = {int(k): v for k, v in class_indices.items()}
            print(f"Class indices loaded successfully. Total classes: {len(self.idx_to_label)}")
            
        except Exception as e:
            print(f"Error loading model or class indices: {str(e)}")
            raise
    
    def preprocess_image(self, img_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess the input image for prediction.
        
        Args:
            img_path: Path to the image file
            target_size: Target size for the image (width, height)
            
        Returns:
            Preprocessed image array
        """
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Load and preprocess image
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_disease(self, img_path: str) -> Tuple[str, float, dict]:
        """
        Predict plant disease from an image.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Tuple containing (class_name, confidence, prediction_details)
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(img_path)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            
            # Get results
            class_index = int(np.argmax(prediction))  # Convert np.int64 to regular int
            class_name = self.idx_to_label[class_index]
            confidence = float(np.max(prediction))
            
            # Get top 3 predictions for additional insight
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_predictions = {
                self.idx_to_label[int(idx)]: float(prediction[0][idx])  # Convert idx to int
                for idx in top_3_indices
            }
            
            prediction_details = {
                'top_predictions': top_3_predictions,
                'all_probabilities': prediction[0].tolist(),
                'image_path': img_path
            }
            
            return class_name, confidence, prediction_details
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def format_disease_name(self, disease_name: str) -> str:
        """
        Format disease name for better readability.
        
        Args:
            disease_name: Raw disease name from prediction
            
        Returns:
            Formatted disease name
        """
        # Replace underscores with spaces and format properly
        formatted = disease_name.replace('_', ' ')
        
        # Handle specific formatting cases
        formatted = formatted.replace('(', ' (').replace(')', ') ')
        formatted = ' '.join(formatted.split())  # Remove extra spaces
        
        return formatted
    
    def get_disease_info(self, disease_name: str) -> dict:
        """
        Get additional information about the detected disease.
        
        Args:
            disease_name: Name of the detected disease
            
        Returns:
            Dictionary with disease information
        """
        # Extract plant type and condition
        parts = disease_name.split('___')
        if len(parts) == 2:
            plant_type = parts[0].replace('_', ' ').title()
            condition = parts[1].replace('_', ' ').title()
        else:
            plant_type = "Unknown"
            condition = disease_name.replace('_', ' ').title()
        
        is_healthy = 'healthy' in disease_name.lower()
        
        return {
            'plant_type': plant_type,
            'condition': condition,
            'is_healthy': is_healthy,
            'formatted_name': self.format_disease_name(disease_name)
        }

# Global predictor instance
_predictor = None

def get_predictor() -> PlantDiseasePredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PlantDiseasePredictor()
    return _predictor

def predict_disease(img_path: str) -> Tuple[str, float]:
    """
    Legacy function for backward compatibility.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Tuple containing (class_name, confidence)
    """
    predictor = get_predictor()
    class_name, confidence, _ = predictor.predict_disease(img_path)
    return class_name, confidence

def predict_disease_detailed(img_path: str) -> Tuple[str, float, dict, dict]:
    """
    Enhanced prediction function with detailed results.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Tuple containing (class_name, confidence, prediction_details, disease_info)
    """
    predictor = get_predictor()
    class_name, confidence, prediction_details = predictor.predict_disease(img_path)
    disease_info = predictor.get_disease_info(class_name)
    
    return class_name, confidence, prediction_details, disease_info
