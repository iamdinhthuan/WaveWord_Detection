
# Wake Word Detection - "Hey DT"

The "Hey DT" wake word detection system uses deep learning with real-time processing capabilities and advanced audio augmentation.

## 🎯 Overview

This project builds a wake word detection system that can recognize the phrase "Hey DT" in real time. It utilizes a CNN-LSTM model combined with Voice Activity Detection (VAD) to achieve high performance and reduce false positives.

### ✨ Key Features

- **Real-time Detection**: Recognizes the wake word in real time  
- **Advanced Audio Augmentation**: Data augmentation using real noise, speed perturbation, and volume control  
- **Voice Activity Detection**: Uses WebRTC VAD to optimize performance  
- **Enhanced CNN-LSTM Architecture**: Deep learning architecture with BatchNormalization and Dropout  
- **Comprehensive Evaluation**: Detailed assessment using confusion matrix and performance metrics  

## 📋 System Requirements

### Dependencies

```bash
pip install -r requirements.txt
```

### Directory Structure

```
project/
├── train.py                 # Script to train the model
├── predict.py              # Script for real-time detection
├── dir_folder_data/        # Directory for training data
│   ├── wakeword/          # "Hey DT" audio files (.wav)
│   └── non-wakeword/      # Non-wake word audio files (.wav)
├── dir_noise_audio/       # Directory for noise files (optional)
├── wakeword_model.h5      # Trained model (output)
└── README.md
```

## 🚀 Usage

### 1. Prepare Data

#### Wake Word Data ("Hey DT")
- Create the `dir_folder_data/wakeword/` directory  
- Collect audio files containing the phrase "Hey DT"  
- Format: `.wav`, 16kHz sample rate, mono  
- Recommended: At least 100–200 samples from diverse speakers  

#### Non-Wake Word Data
- Create the `dir_folder_data/non-wakeword/` directory  
- Collect audio files that do not contain "Hey DT"  
- Can include: noise, environmental sounds, other speech  
- Recommended: 2–3x the number of wake word samples  

#### Noise Files (Optional)
- Create the `dir_noise_audio/` directory  
- Include noise audio files for augmentation  
- Format: `.wav`, `.mp3`, `.flac`, etc.  

### 2. Train the Model

```bash
python train.py
```

#### Training Configuration

Edit parameters in `train.py`:

```python
# Configuration
AUDIO_DIR = "dir_folder_data"      # Data directory
NOISE_DIR = "dir_noise_audio"      # Noise directory (optional)
MODEL_PATH = "wakeword_model.h5"   # Model save path

# Augmentation parameters
augmentation_factor = 2            # Augmented samples per original
```

#### Training Process

1. **Data Loading**: Load and process audio files  
2. **Audio Augmentation**: Apply augmentation techniques  
3. **Feature Extraction**: Extract MFCC features (13 coefficients)  
4. **Model Training**: Train CNN-LSTM model with early stopping  
5. **Evaluation**: Assess model using confusion matrix and metrics  

#### Training Outputs

- Trained model file: `wakeword_model.h5`  
- Training log: `training_log.csv`  
- Analysis plot: `enhanced_training_analysis.png`  
- Console output with detailed metrics  

### 3. Real-time Detection

```bash
python predict.py
```

#### Detection Configuration

Edit parameters in `predict.py`:

```python
MODEL_PATH = "wakeword_model.h5"   # Path to model
SAMPLE_RATE = 16000                # Sample rate
WAKE_THRESHOLD = 0.5               # Detection threshold (0.0–1.0)
VAD_AGGRESSIVENESS = 1             # VAD sensitivity (0–3)
```

#### Operation

1. **Audio Streaming**: Capture audio from microphone  
2. **Voice Activity Detection**: Identify speech segments  
3. **Feature Extraction**: Extract MFCC features  
4. **Wake Word Detection**: Predict confidence score  
5. **Threshold Check**: Compare score against threshold  

## ⚙️ System Architecture

### Audio Augmentation

- **Real Noise Addition**: Add real noise with SNR control  
- **Gaussian Noise**: Add white noise  
- **Time Shift**: Temporally shift audio  
- **Speed Perturbation**: Adjust speed (0.9x – 1.1x)  
- **Volume Perturbation**: Adjust volume (0.7x – 1.3x)  

### Model Architecture

```
Input: MFCC Features (time_steps, 13)
├── Conv1D(16, kernel_size=5) + BatchNorm + MaxPool + Dropout
├── Conv1D(32, kernel_size=5) + BatchNorm + MaxPool + Dropout
├── LSTM(32, return_sequences=True) + Dropout
├── LSTM(16) + Dropout
├── Dense(32, relu) + Dropout
└── Dense(1, sigmoid) -> Wake Word Probability
```

### Real-time Processing

- **Audio Streaming**: 480 samples per chunk (30ms at 16kHz)  
- **VAD Processing**: WebRTC VAD for speech detection  
- **Buffer Management**: 3-second sliding window  
- **Threading**: Separate threads for audio capture and processing  

## 📊 Performance Evaluation

### Metrics

- **Accuracy**: Overall correctness  
- **Precision**: True positives among predicted positives  
- **Recall**: True positives among actual positives  
- **Specificity**: True negatives correctly identified  
- **F1 Score**: Harmonic mean of precision and recall  

### Confusion Matrix

```
                Predicted
              No    Yes
Actual No   [TN]  [FP]
       Yes  [FN]  [TP]
```

## 🔧 Optimization

### Improve Accuracy

1. **Increase training data**:  
   - Gather more samples from diverse speakers  
   - Vary recording environments  
   - Include variations of "Hey DT"  

2. **Adjust threshold**:  
   - Lower threshold: Increase recall, decrease precision  
   - Raise threshold: Increase precision, decrease recall  

3. **Fine-tune model**:  
   - Modify architecture (layers, units)  
   - Tweak learning rate and optimizer  
   - Apply regularization techniques  

### Reduce Latency

1. **Optimize audio processing**:  
   - Reduce chunk size  
   - Optimize feature extraction  

2. **Model optimization**:  
   - Quantize model  
   - Convert to TensorFlow Lite  

## 🎛️ Configuration Options

### Training Configuration

```python
class AugmentationConfig:
    def __init__(self):
        # Noise augmentation
        self.noise_snr_range = [5, 20]        # SNR range in dB
        self.gaussian_noise_level = 0.005     # Gaussian noise level

        # Time and speed augmentation
        self.time_shift_max = 0.1             # 10% of audio length
        self.speed_range = [0.9, 1.1]         # Speed factor range
        self.volume_range = [0.7, 1.3]        # Volume factor range

        # Augmentation probability
        self.augmentation_prob = 0.5          # Probability per augmentation
```

### Detection Configuration

```python
# Real-time detection parameters
sample_rate = 16000          # Audio sample rate
chunk_size = 480            # Audio chunk size (30ms)
wake_threshold = 0.5        # Wake word detection threshold
vad_aggressiveness = 1      # VAD sensitivity (0–3)
```

### Debug Mode

Enable logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Notes

- The model works best with 16kHz mono audio  
- Using a headset microphone is recommended to reduce echo  
- Default threshold 0.5 is suitable in most cases  
- Higher VAD aggressiveness (3) increases sensitivity but may raise false positives  
