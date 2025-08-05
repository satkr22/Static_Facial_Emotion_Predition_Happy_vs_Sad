# Static Facial Emotion Prediction (Happy vs Sad)

This project is a deep learning-based facial emotion classification system that predicts whether a person is **happy** or **sad** from static images. The model is trained on facial expression data and built using TensorFlow and Keras.

## Features

- Binary classification: **Happy** vs **Sad**
- Preprocessing with OpenCV (grayscale conversion, resizing)
- CNN-based architecture with dropout and data augmentation
- Evaluation using accuracy, loss plots, and confusion matrix

##  Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

- 3 convolutional blocks (Conv2D + MaxPooling2D)
- Flatten layer
- Dense layers with Dropout
- Output layer with sigmoid activation

## Folder Structure

```
Static_Facial_Emotion_Prediction_Happy_vs_Sad/
│
├── dataset/                     # Preprocessed dataset (happy/sad images)
├── model/                       # Trained model file (H5 format)
├── saved_plots/                # Accuracy/loss and confusion matrix plots
├── test_images/                # Test images for predictions
├── Emotion_detection.ipynb     # Main training + prediction notebook
├── requirements.txt            # Required libraries
└── README.md                   # This file
```

## Installation

1. **Clone the repo**
```bash
git clone https://github.com/satkr22/Static_Facial_Emotion_Predition_Happy_vs_Sad.git
cd Static_Facial_Emotion_Predition_Happy_vs_Sad
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the notebook**
Open the Jupyter Notebook:  
```bash
jupyter notebook Emotion_detection.ipynb
```

## Results

- Accuracy: ~**97.5%** (replace with your final accuracy)
- Confusion Matrix and Accuracy/Loss curves are available in `saved_plots/`

##  Sample Prediction

You can test images from the `test_images/` folder in the notebook and view predictions.

## Dependencies

- Python 3.x  
- TensorFlow  
- Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  

Full list in `requirements.txt`

## Author

**Satyam Kumar**  
🔗 [GitHub](https://github.com/satkr22)

## License

This project is open source and available under the [MIT License](LICENSE).Dependencies
