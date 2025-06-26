# AI-Powered Vehicle Safety Monitoring System

A real-time computer vision system that automatically detects and monitors safety alerts (fire, smoke, and other hazardous conditions) in vehicle environments using webcam feeds and MobileNetV2 deep learning model with transfer learning.

## 🎯 Features

- **Real-time Alert Detection**: Continuous monitoring with 0.1-second intervals using Google Colab environment
- **Multi-class Classification**: Detects multiple safety hazards with confidence scoring
- **Advanced Data Augmentation**: 20 variations per original image with sophisticated transformations
- **Transfer Learning**: Fine-tuned MobileNetV2 with custom architecture for automotive safety
- **Crash-resistant Logging**: Advanced SafetyAlertLogger with atomic writes and backup systems
- **Browser Integration**: Failsafe camera handler with JavaScript webcam access
- **Production Ready**: Complete error handling, graceful shutdown, and automated checkpointing

## 🏗️ System Architecture

```
Data Preprocessing → Model Training → Real-time Detection
      ↓                    ↓              ↓
Image Augmentation → MobileNetV2 → Browser Camera Capture
      ↓                    ↓              ↓
Dataset Split     → Custom Layers → JavaScript Frame Processing
      ↓                    ↓              ↓
Train/Val Folders → Model Weights → Python Backend Inference
                                   ↓
                            SafetyAlertLogger → CSV Export
```

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras, MobileNetV2, Transfer Learning
- **Computer Vision**: OpenCV (cv2), Image Processing, Real-time inference
- **Development Environment**: Google Colab, Jupyter Notebooks
- **Frontend**: JavaScript (webcam integration), HTML5 Canvas
- **Data Processing**: NumPy, Pandas, ImageDataGenerator
- **Logging**: Custom SafetyAlertLogger with atomic file operations

## 📋 Requirements

```python
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
pillow>=8.3.0
```

*Note: This system is designed to run in Google Colab environment*

## 🚀 Quick Start

### 1. Prepare Your Environment
```bash
# Upload to Google Colab or local Jupyter environment
# Ensure GPU runtime is enabled in Colab for faster training
```

### 2. Prepare Your Dataset
```python
# Place your original alert images in 'alerts/' folder
# Images should be named by class (e.g., fire.jpg, smoke.png)
python data_preprocessing.py
```

### 3. Train the Model
```python
# Run the training script
python model_training.py
```

### 4. Run Real-time Detection
```python
# Execute the main detection system
python real_time_detection.py
```

## 📊 Model Architecture Details

### Base Model: MobileNetV2
- **Pre-trained weights**: ImageNet
- **Fine-tuning**: Last 50 layers trainable
- **Input shape**: (224, 224, 3)

### Custom Classification Head
```python
GlobalAveragePooling2D()
Dense(512, activation='relu', kernel_regularizer=l2(0.001))
BatchNormalization()
Dropout(0.5)
Dense(256, activation='relu', kernel_regularizer=l2(0.001))
BatchNormalization()
Dropout(0.3)
Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))
```

### Training Configuration
- **Optimizer**: Adam (learning_rate=1e-4)
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Batch Size**: 32
- **Epochs**: 30 (with early stopping)

## 📁 Project Structure

```
vehicle-safety-monitoring/
│
├── data_preprocessing.py    # Dataset preparation and augmentation
├── model_training.py       # Model training with transfer learning
├── real_time_detection.py  # Main detection system
├── requirements.txt        # Dependencies
├── README.md              # This file
│
├── alerts/                # Original alert images
│   ├── fire.jpg
│   ├── smoke.png
│   └── normal.jpg
│
├── dataset/               # Processed dataset (auto-generated)
│   ├── train/
│   │   ├── fire/
│   │   ├── smoke/
│   │   └── normal/
│   └── val/
│       ├── fire/
│       ├── smoke/
│       └── normal/
│
├── models/
│   ├── best_model.h5      # Trained model weights
│   └── label_map.txt      # Class mappings
│
└── logs/
    ├── alert_logs_verified.csv
    └── /tmp/alert_backup.csv
```

## 🎮 Usage Guide

### Step 1: Data Preparation
```python
# Configure your dataset
source_folder = "alerts"        # Your original images
output_folder = "dataset"       # Processed dataset location
train_split = 0.8              # 80% for training
aug_per_image = 20             # 20 variations per image
img_size = (224, 224)          # Standard input size
```

### Step 2: Model Training
```python
# Key training parameters
batch_size = 32
epochs = 30
base_model.trainable = True
# Freeze all layers except last 50
for layer in base_model.layers[:-50]:
    layer.trainable = False
```

### Step 3: Real-time Detection
```python
# The system automatically:
# 1. Loads trained model weights
# 2. Initializes SafetyAlertLogger
# 3. Sets up FailsafeCamera
# 4. Starts continuous monitoring
```

## 🔧 Key Components

### SafetyAlertLogger Class
- **Alert Buffering**: 5-frame buffer for stable detection
- **Atomic Writes**: Crash-resistant logging to `/tmp/alert_backup.csv`
- **Duration Tracking**: Measures alert duration with 1-second minimum
- **Confidence Scoring**: Tracks average confidence per alert

### FailsafeCamera Class
- **Retry Mechanism**: 3 attempts with 2-second delays
- **JavaScript Integration**: Browser-based webcam access
- **Base64 Encoding**: Frame transmission between frontend/backend
- **Error Recovery**: Graceful handling of camera failures

### Advanced Data Augmentation
```python
rotation_range=30           # ±30 degree rotation
width_shift_range=0.2       # ±20% horizontal shift
height_shift_range=0.2      # ±20% vertical shift
zoom_range=0.3             # ±30% zoom
brightness_range=(0.3, 1.2) # 30% to 120% brightness
shear_range=0.2            # ±20 degree shear
horizontal_flip=True        # Random horizontal flips
```

## 📈 Training Features

### Callbacks & Optimization
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Prevents overfitting (patience=15)
- **ReduceLROnPlateau**: Dynamic learning rate adjustment
- **TensorBoard**: Training visualization and monitoring
- **Class Weights**: Automatic balancing for imbalanced datasets

### Regularization Techniques
- **L2 Regularization**: Applied to all dense layers (0.001)
- **Dropout**: 50% and 30% dropout rates
- **BatchNormalization**: Stabilizes training

## 🔍 Real-time Processing Pipeline

1. **Frame Capture**: JavaScript captures webcam frames at 0.1s intervals
2. **Preprocessing**: Resize to 224×224, apply MobileNetV2 preprocessing
3. **Inference**: Model prediction with confidence scoring
4. **Alert Logic**: 5-frame buffer with majority voting
5. **Logging**: Timestamp tracking with 35% confidence threshold
6. **Display**: Real-time overlay with alert status and confidence

## 🎯 Performance Optimization

- **Efficient Architecture**: MobileNetV2 optimized for mobile/edge deployment
- **Batch Processing**: 32-image batches for training efficiency
- **Memory Management**: Proper cleanup and resource management
- **Error Recovery**: Robust exception handling throughout pipeline

## 🛡️ Safety & Reliability

- **Crash-resistant Logging**: Dual logging system (memory + file backup)
- **Graceful Shutdown**: Proper resource cleanup on interruption
- **Camera Resource Management**: Automatic stream cleanup
- **Validation Checks**: System verification before execution
- **Error Boundaries**: Isolated error handling per component

## 📊 Output & Monitoring

### Real-time Display
- Live video feed with alert overlays
- Current alert status and confidence percentage
- Total logged alerts counter
- System status indicator

### Log Export
```csv
timestamp,alert_type,confidence,duration
2025-06-26 14:30:15,fire,0.876,3.2
2025-06-26 14:32:45,smoke,0.654,1.8
```

## 🔧 Configuration Options

### Model Parameters
```python
# In real_time_detection.py
confidence_threshold = 0.35    # Minimum confidence for logging
buffer_size = 5               # Frame buffer for stability
monitoring_interval = 0.1     # Seconds between captures
display_update = 0.5          # UI update frequency
```

### Training Parameters
```python
# In model_training.py
learning_rate = 1e-4          # Adam optimizer learning rate
fine_tune_layers = 50         # Last N layers to fine-tune
regularization = 0.001        # L2 regularization strength
```

## 🐛 Troubleshooting

### Common Issues

**Camera Access Denied:**
```javascript
// Ensure HTTPS or localhost for camera permissions
// Check browser security settings
```

**Model Loading Errors:**
```python
# Verify model architecture matches training
# Check TensorFlow version compatibility
# Ensure label_map.txt exists
```

**Memory Issues:**
```python
# Reduce batch_size in training
# Clear TensorFlow session periodically
# Monitor GPU memory usage in Colab
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Test your changes thoroughly
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Developed during internship at Onward Technologies, Chennai**
- **Mentored by Mr. Saravanan Guruswami and Mr. RajanRaju Chiluvuri**
- **MobileNetV2 architecture by Google Research**
- **TensorFlow team for the excellent deep learning framework**

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]
- **Project Link**: [https://github.com/yourusername/vehicle-safety-monitoring](https://github.com/yourusername/vehicle-safety-monitoring)

---

⭐ **If you found this project helpful, please give it a star!** ⭐

## 🎥 Demo

*Add screenshots or GIF demonstrations of your system in action*

## 📚 Additional Resources

- [Transfer Learning Guide](https://www.tensorflow.org/guide/transfer_learning)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Google Colab Tips](https://colab.research.google.com/)
- [OpenCV Documentation](https://opencv.org/)
