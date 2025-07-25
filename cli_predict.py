"""
Command Line Interface for Plant Disease Classification

Simple CLI tool for quick plant disease predictions.
"""

import argparse
import os
import sys
from utils.predict import PlantDiseasePredictor

def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("🌿 PLANT DISEASE CLASSIFIER - CLI")
    print("=" * 60)

def format_prediction_output(class_name: str, confidence: float, disease_info: dict):
    """Format prediction output for CLI display."""
    print("\n" + "🔍 PREDICTION RESULTS" + "=" * 40)
    print(f"📸 Image Analysis Complete!")
    print("-" * 50)
    
    # Basic prediction
    print(f"🩺 Diagnosis: {disease_info['formatted_name']}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")
    print(f"🌱 Plant Type: {disease_info['plant_type']}")
    print(f"🏥 Condition: {disease_info['condition']}")
    
    # Health status with color coding
    if disease_info['is_healthy']:
        print("✅ Health Status: HEALTHY")
        print("💚 Great news! Your plant appears to be in good health.")
    else:
        print("⚠️  Health Status: DISEASE DETECTED")
        print("🟡 Recommendation: Consult with a plant expert for proper treatment.")
    
    # Confidence interpretation
    if confidence >= 0.9:
        print("🎯 Confidence Level: Very High")
    elif confidence >= 0.7:
        print("🎯 Confidence Level: High")
    elif confidence >= 0.5:
        print("🎯 Confidence Level: Moderate")
    else:
        print("🎯 Confidence Level: Low (consider retaking the image)")
    
    print("=" * 60)

def predict_single_image(image_path: str, model_path: str = None, class_indices_path: str = None):
    """
    Predict disease for a single image.
    
    Args:
        image_path: Path to the image file
        model_path: Optional custom model path
        class_indices_path: Optional custom class indices path
    """
    try:
        # Use default paths if not provided
        if model_path is None:
            model_path = "model/Plant_Disease_Detection.h5"
        if class_indices_path is None:
            class_indices_path = "assets/class_indices.json"
        
        # Check if files exist
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file '{image_path}' not found!")
            return False
        
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file '{model_path}' not found!")
            return False
        
        if not os.path.exists(class_indices_path):
            print(f"❌ Error: Class indices file '{class_indices_path}' not found!")
            return False
        
        print(f"📂 Loading image: {image_path}")
        print(f"🤖 Using model: {model_path}")
        print("⏳ Processing...")
        
        # Initialize predictor
        predictor = PlantDiseasePredictor(model_path, class_indices_path)
        
        # Make prediction
        class_name, confidence, prediction_details = predictor.predict_disease(image_path)
        disease_info = predictor.get_disease_info(class_name)
        
        # Display results
        format_prediction_output(class_name, confidence, disease_info)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return False

def interactive_mode():
    """Run in interactive mode."""
    print_banner()
    print("🔄 Interactive Mode - Enter image paths to analyze")
    print("💡 Type 'quit' or 'exit' to stop")
    print("-" * 60)
    
    predictor = None
    
    while True:
        try:
            # Get image path from user
            image_path = input("\n📁 Enter image path: ").strip()
            
            # Check for exit commands
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not image_path:
                continue
            
            # Initialize predictor if not already done
            if predictor is None:
                print("🤖 Loading model (this may take a moment)...")
                try:
                    predictor = PlantDiseasePredictor()
                    print("✅ Model loaded successfully!")
                except Exception as e:
                    print(f"❌ Error loading model: {str(e)}")
                    break
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue
            
            print("⏳ Analyzing image...")
            
            # Make prediction
            class_name, confidence, prediction_details = predictor.predict_disease(image_path)
            disease_info = predictor.get_disease_info(class_name)
            
            # Display results
            format_prediction_output(class_name, confidence, disease_info)
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Plant Disease Classification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_predict.py image.jpg                    # Predict single image
  python cli_predict.py -i                          # Interactive mode
  python cli_predict.py image.jpg -m custom_model.h5 # Use custom model
        """
    )
    
    parser.add_argument("image", nargs='?', help="Path to image file")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-m", "--model", help="Path to model file (default: model/Plant_Disease_Detection.h5)")
    parser.add_argument("-c", "--classes", help="Path to class indices file (default: assets/class_indices.json)")
    parser.add_argument("--version", action="version", version="Plant Disease Classifier CLI v1.0")
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Single image prediction
    if args.image:
        print_banner()
        success = predict_single_image(args.image, args.model, args.classes)
        sys.exit(0 if success else 1)
    
    # No arguments provided
    print_banner()
    print("❓ No image provided. Use -h for help or -i for interactive mode.")
    print("\nQuick start:")
    print("  python cli_predict.py your_image.jpg")
    print("  python cli_predict.py -i")

if __name__ == "__main__":
    main()
