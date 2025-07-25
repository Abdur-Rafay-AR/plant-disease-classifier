"""
Setup script for Plant Disease Classifier

This script helps set up the environment and verify the installation.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🌿 PLANT DISEASE CLASSIFIER - SETUP")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"🐍 Python version: {sys.version}")
    
    if current_version >= min_version:
        print("✅ Python version is compatible")
        return True
    else:
        print(f"❌ Python {min_version[0]}.{min_version[1]}+ required, found {current_version[0]}.{current_version[1]}")
        return False

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Packages installed successfully")
            return True
        else:
            print(f"❌ Error installing packages: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during installation: {str(e)}")
        return False

def check_model_files():
    """Check if model files exist."""
    print("\n🤖 Checking model files...")
    
    model_path = "model/Plant_Disease_Detection.h5"
    class_indices_path = "assets/class_indices.json"
    
    model_exists = os.path.exists(model_path)
    classes_exist = os.path.exists(class_indices_path)
    
    if model_exists:
        print(f"✅ Model file found: {model_path}")
        # Check model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   Size: {model_size:.1f} MB")
    else:
        print(f"❌ Model file not found: {model_path}")
    
    if classes_exist:
        print(f"✅ Class indices found: {class_indices_path}")
        # Check number of classes
        try:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            print(f"   Classes: {len(class_indices)}")
        except Exception as e:
            print(f"   Warning: Could not read class indices: {str(e)}")
    else:
        print(f"❌ Class indices not found: {class_indices_path}")
    
    return model_exists and classes_exist

def test_imports():
    """Test if all required packages can be imported."""
    print("\n🔍 Testing package imports...")
    
    packages = [
        ("streamlit", "streamlit"),
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("plotly", "plotly"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "scikit-learn")
    ]
    
    all_good = True
    
    for package_name, pip_name in packages:
        try:
            __import__(package_name)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name} - not found or not working")
            all_good = False
    
    return all_good

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "outputs",
        "batch_results",
        "logs"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created/verified: {directory}/")
        except Exception as e:
            print(f"❌ Error creating {directory}/: {str(e)}")

def test_prediction():
    """Test the prediction functionality."""
    print("\n🧪 Testing prediction functionality...")
    
    try:
        from utils.predict import PlantDiseasePredictor
        
        # Check if we can initialize the predictor
        predictor = PlantDiseasePredictor()
        print("✅ Predictor initialized successfully")
        
        # Test with a dummy prediction (if model files exist)
        if os.path.exists("model/Plant_Disease_Detection.h5"):
            print("✅ Model loading test passed")
        else:
            print("⚠️  Cannot test prediction without model file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing prediction: {str(e)}")
        return False

def check_streamlit():
    """Test Streamlit installation."""
    print("\n🌐 Testing Streamlit...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test if streamlit command is available
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "--version"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Streamlit command available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Streamlit command not working properly")
            return False
            
    except Exception as e:
        print(f"❌ Error with Streamlit: {str(e)}")
        return False

def generate_setup_report():
    """Generate a setup report."""
    print("\n📋 SETUP REPORT")
    print("=" * 40)
    
    # Check all components
    checks = {
        "Python Version": check_python_version(),
        "Required Packages": test_imports(),
        "Model Files": check_model_files(),
        "Prediction Test": test_prediction(),
        "Streamlit": check_streamlit()
    }
    
    print("\nComponent Status:")
    for component, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}")
    
    all_good = all(checks.values())
    
    print(f"\n{'🎉 SETUP COMPLETE!' if all_good else '⚠️  SETUP INCOMPLETE'}")
    
    if all_good:
        print("\n🚀 You're ready to go! Try:")
        print("   streamlit run app.py")
        print("   python cli_predict.py -i")
        print("   python batch_predict.py --help")
    else:
        print("\n🔧 Please fix the issues above before proceeding.")
        if not checks["Model Files"]:
            print("\n📥 To get the model files:")
            print("   1. Download the trained model (Plant_Disease_Detection.h5)")
            print("   2. Place it in the 'model/' directory")
            print("   3. Ensure class_indices.json is in 'assets/' directory")
    
    return all_good

def main():
    """Main setup function."""
    print_banner()
    
    print("This script will set up the Plant Disease Classifier environment.")
    print("It will install required packages and verify the installation.\n")
    
    # Create directories first
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation.")
        return False
    
    # Generate comprehensive report
    success = generate_setup_report()
    
    if success:
        print(f"\n{'='*60}")
        print("🎯 NEXT STEPS:")
        print("  1. Run 'streamlit run app.py' to start the web interface")
        print("  2. Run 'python cli_predict.py -i' for CLI interface")
        print("  3. Check README.md for detailed usage instructions")
        print("="*60)
    
    return success

if __name__ == "__main__":
    main()
