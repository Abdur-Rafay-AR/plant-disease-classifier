# 🌿 Plant Disease Classifier

A deep learning-based web application for identifying plant diseases from leaf images using TensorFlow and Streamlit.

## 📋 Overview

This application uses a pre-trained Convolutional Neural Network (CNN) to classify plant diseases from uploaded leaf images. It can identify 38 different classes of plant diseases across various crops including Apple, Corn, Grape, Tomato, Potato, and more.

## 🚀 Features

- **Web Interface**: User-friendly Streamlit web application
- **Image Upload**: Support for JPG, JPEG, and PNG image formats
- **Real-time Prediction**: Instant disease classification with confidence scores
- **Multiple Plant Types**: Supports classification for various crops
- **High Accuracy**: Uses a trained deep learning model for reliable predictions

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdur-Rafay-AR/plant-disease-classifier.git
cd plant-disease-classifier
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a leaf image using the file uploader

4. View the prediction results with confidence scores

## 🗂️ Project Structure

```
plant-disease-classifier/
├── app.py                      # Main Streamlit application
├── utils/
│   └── predict.py             # Prediction utility functions
├── model/
│   └── Plant_Disease_Detection.h5  # Pre-trained model
├── assets/
│   └── class_indices.json     # Class label mappings
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore rules
```

## 🎯 Supported Plant Diseases

The model can classify the following plant diseases:

### Apple
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

### Corn (Maize)
- Cercospora Leaf Spot / Gray Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy

### Grape
- Black Rot
- Esca (Black Measles)
- Leaf Blight (Isariopsis Leaf Spot)
- Healthy

### Tomato
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted Spider Mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

### Other Crops
- Blueberry (Healthy)
- Cherry (Powdery Mildew, Healthy)
- Orange (Huanglongbing/Citrus Greening)
- Peach (Bacterial Spot, Healthy)
- Pepper Bell (Bacterial Spot, Healthy)
- Potato (Early Blight, Late Blight, Healthy)
- Raspberry (Healthy)
- Soybean (Healthy)
- Squash (Powdery Mildew)
- Strawberry (Leaf Scorch, Healthy)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Mukku27](https://github.com/Mukku27)** for developing the plant disease detection model, which served as the foundation for this project.
- Dataset and model training based on plant disease classification datasets.
- Streamlit for providing an excellent web app framework.
- TensorFlow/Keras for deep learning capabilities.
