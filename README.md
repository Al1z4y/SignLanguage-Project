# Pakistani Sign Language Detection System

A real-time Pakistani Sign Language (PSL) gesture recognition system using MediaPipe and TensorFlow. This system can detect and classify 37 different Pakistani Sign Language alphabets with high accuracy.

## 🌟 Features

- **Real-time Detection**: Live gesture recognition using webcam
- **37 PSL Alphabets**: Supports all Pakistani Sign Language alphabets
- **High Accuracy**: 96.8% accuracy on test data
- **Normalized Coordinates**: Resolution-independent using MediaPipe's coordinate system
- **Smooth Predictions**: Temporal smoothing for stable gesture recognition
- **Adjustable Confidence**: Real-time threshold adjustment

## 📋 Requirements

### System Requirements
- Python 3.7+
- Webcam/Camera
- macOS, Windows, or Linux

### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `tensorflow>=2.10.0`
- `opencv-python>=4.6.0`
- `mediapipe>=0.9.0`
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `tqdm>=4.62.0`

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SignLanguage-Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pre-trained Model (if available)
```bash
python3 real_time_detector.py
```

### 4. Train Your Own Model (optional)
```bash
python3 hand_gesture_trainer.py
```

## 📊 Dataset

The system uses a CSV file `psl_sign_language_keypoints.csv` containing:
- **1,720 samples** across 37 PSL alphabets
- **Hand landmark coordinates** (21 landmarks per hand)
- **Normalized coordinates** (0-1 range) for resolution independence

### Dataset Structure
```
Columns: 128 total
- alphabet: Target label (37 classes)
- sample_file: Original filename
- left_hand_0_x, left_hand_0_y, left_hand_0_confidence: Left hand landmarks
- right_hand_0_x, right_hand_0_y, right_hand_0_confidence: Right hand landmarks
- ... (21 landmarks × 3 coordinates × 2 hands)
```

### Supported Gestures
```
ain, alif, bari_yay, bay, chay, choti_yay, daal, ddaal, fay, gaaf, 
ghain, hamza, hay, hay_do_chashmi, jeem, kaaf, khay, laam, meem, 
noon, pay, qaf, ray, rray, say, seen, sheen, swad, tay, toay, 
ttay, wao, zaal, zay, zhay, zoay, zwad
```

## 🔧 Training Process

### 1. Prepare Dataset
Ensure you have `psl_sign_language_keypoints.csv` in the project directory.

### 2. Run Training Script
```bash
python3 hand_gesture_trainer.py
```

### Training Options
- **CSV file path**: Default `psl_sign_language_keypoints.csv`
- **Epochs**: Default 100 (recommend 10-50 for quick training)
- **Batch size**: Default 32
- **Remove confidence**: Default 'y' (recommended)

### Training Process
1. **Data Loading**: Loads CSV and displays dataset info
2. **Coordinate Normalization**: Converts pixel coordinates to 0-1 range
3. **Feature Scaling**: StandardScaler normalization
4. **Model Creation**: Neural network with dropout and batch normalization
5. **Training**: With early stopping and learning rate reduction
6. **Evaluation**: Test accuracy and classification report
7. **Model Saving**: Saves model and preprocessing objects

### Generated Files
- `hand_gesture_model.h5`: Trained TensorFlow model
- `label_encoder.pkl`: Label encoder for gesture names
- `feature_scaler.pkl`: Feature scaler for normalization
- `feature_names.pkl`: Feature column names
- `training_results.png`: Training history plots
- `confusion_matrix.png`: Confusion matrix visualization

## 🎥 Real-time Detection

### 1. Run Detection System
```bash
python3 real_time_detector.py
```

### 2. Controls
- **'q'**: Quit the application
- **'t'**: Increase confidence threshold
- **'g'**: Decrease confidence threshold  
- **'r'**: Reset prediction buffer

### 3. Usage Tips
- **Lighting**: Ensure good lighting conditions
- **Background**: Use contrasting background for better hand detection
- **Distance**: Keep hand at appropriate distance from camera
- **Stability**: Hold gesture steady for better recognition

### Detection Display
- **Gesture Name**: Currently detected gesture
- **Confidence Score**: Prediction confidence (0-1)
- **Stability**: Temporal consistency of predictions
- **FPS**: Real-time processing speed
- **Threshold**: Current confidence threshold

## 📁 Project Structure

```
SignLanguage-Project/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── psl_sign_language_keypoints.csv     # Training dataset
├── hand_gesture_trainer.py             # Main training script
├── real_time_detector.py               # Real-time detection system
├── hand_gesture_model.h5               # Trained model (generated)
├── label_encoder.pkl                   # Label encoder (generated)
├── feature_scaler.pkl                  # Feature scaler (generated)
├── feature_names.pkl                   # Feature names (generated)
├── training_results.png                # Training plots (generated)
└── confusion_matrix.png                # Confusion matrix (generated)
```

## 🔧 Technical Details

### Coordinate System
- **Input Format**: MediaPipe normalized coordinates (0-1 range)
- **Training Data**: Pixel coordinates normalized to 0-1 range
- **Compatibility**: Perfect match between training and inference

### Model Architecture
```
Input Layer (126 features)
├── Dense(512) + BatchNorm + Dropout(0.3)
├── Dense(256) + BatchNorm + Dropout(0.3)
├── Dense(128) + BatchNorm + Dropout(0.2)
├── Dense(64) + Dropout(0.2)
└── Dense(37) + Softmax
```

### Feature Processing
1. **Hand Detection**: MediaPipe hand landmark detection
2. **Coordinate Extraction**: 21 landmarks × 2 coordinates × 2 hands = 84 features
3. **Confidence Features**: Optional confidence values (42 additional features)
4. **Normalization**: StandardScaler for feature scaling
5. **Prediction**: Softmax classification over 37 classes

## 🐛 Troubleshooting

### Common Issues

#### 1. "No gesture detected"
- **Check lighting**: Ensure good lighting conditions
- **Hand visibility**: Make sure your hand is clearly visible
- **Distance**: Adjust distance from camera
- **Threshold**: Lower confidence threshold with 'g' key

#### 2. "Model files not found"
```bash
# Run training first
python3 hand_gesture_trainer.py
```

#### 3. "Camera not accessible"
- **Permissions**: Check camera permissions
- **Other apps**: Close other apps using camera
- **Camera index**: Try different camera indices if multiple cameras

#### 4. Poor accuracy
- **Lighting**: Improve lighting conditions
- **Background**: Use contrasting background
- **Hand positioning**: Follow gesture guidelines
- **Model retraining**: Retrain with more epochs

### Performance Optimization
- **Reduce resolution**: Lower camera resolution for faster processing
- **Adjust smoothing**: Modify `smoothing_window` parameter
- **GPU acceleration**: Use TensorFlow-GPU for faster inference

## 📈 Model Performance

### Current Results
- **Test Accuracy**: 96.8%
- **Top-3 Accuracy**: 99.42%
- **Top-5 Accuracy**: 99.71%
- **Training Time**: ~2-5 minutes (10 epochs)
- **Inference Speed**: 15-30 FPS (depending on hardware)

### Best Performing Gestures
- Most alphabets achieve >95% accuracy
- Excellent performance on distinct gestures
- Some confusion between similar hand shapes (normal behavior)

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/improvement`
3. **Commit changes**: `git commit -am 'Add new feature'`
4. **Push to branch**: `git push origin feature/improvement`
5. **Create Pull Request**

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Hand landmark detection
- **TensorFlow**: Machine learning framework
- **PSL Community**: Pakistani Sign Language dataset and guidance
- **OpenCV**: Computer vision utilities

## 📞 Support

For issues and questions:
1. **Check troubleshooting section** above
2. **Open GitHub issue** with detailed description
3. **Include error messages** and system information

---

**Happy Gesture Recognition! 🤟**
