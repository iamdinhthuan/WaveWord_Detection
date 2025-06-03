# Wake Word Detection - "Hey DT"

Hệ thống nhận diện từ đánh thức "Hey DT" sử dụng deep learning với khả năng xử lý real-time và audio augmentation tiên tiến.

## 🎯 Tổng quan

Dự án này xây dựng một hệ thống wake word detection có thể nhận diện cụm từ "Hey DT" trong thời gian thực. Hệ thống sử dụng mô hình CNN-LSTM kết hợp với Voice Activity Detection (VAD) để đạt hiệu suất cao và giảm false positive.

### ✨ Tính năng chính

- **Real-time Detection**: Nhận diện wake word trong thời gian thực
- **Advanced Audio Augmentation**: Tăng cường dữ liệu với noise thực tế, speed perturbation, volume control
- **Voice Activity Detection**: Sử dụng WebRTC VAD để tối ưu hiệu suất
- **Enhanced CNN-LSTM Architecture**: Kiến trúc deep learning với BatchNormalization và Dropout
- **Comprehensive Evaluation**: Đánh giá chi tiết với confusion matrix và metrics

## 📋 Yêu cầu hệ thống

### Dependencies

```bash
pip install -r requirements.txt
```

### Cấu trúc thư mục

```
project/
├── train.py                 # Script training model
├── predict.py              # Script real-time detection
├── dir_folder_data/        # Thư mục chứa dữ liệu training
│   ├── wakeword/          # Audio files "Hey DT" (.wav)
│   └── non-wakeword/      # Audio files không phải wake word (.wav)
├── dir_noise_audio/       # Thư mục chứa noise files (optional)
├── wakeword_model.h5      # Model đã train (output)
└── README.md
```

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu

#### Dữ liệu Wake Word ("Hey DT")
- Tạo thư mục `dir_folder_data/wakeword/`
- Thu thập audio files chứa cụm từ "Hey DT"
- Format: `.wav`, sample rate 16kHz, mono
- Khuyến nghị: ít nhất 100-200 samples từ nhiều người nói khác nhau

#### Dữ liệu Non-Wake Word
- Tạo thư mục `dir_folder_data/non-wakeword/`
- Thu thập audio files không chứa "Hey DT"
- Có thể bao gồm: tiếng ồn, âm thanh môi trường, các từ khác
- Khuyến nghị: gấp 2-3 lần số lượng wake word samples

#### Noise Files (Tùy chọn)
- Tạo thư mục `dir_noise_audio/`
- Chứa các file audio noise để augmentation
- Format: `.wav`, `.mp3`, `.flac`, etc.

### 2. Training Model

```bash
python train.py
```

#### Cấu hình Training

Chỉnh sửa các tham số trong `train.py`:

```python
# Configuration
AUDIO_DIR = "dir_folder_data"      # Thư mục chứa dữ liệu
NOISE_DIR = "dir_noise_audio"      # Thư mục noise (có thể để None)
MODEL_PATH = "wakeword_model.h5"   # Đường dẫn lưu model

# Augmentation parameters
augmentation_factor = 2            # Số augmented samples cho mỗi original
```

#### Quá trình Training

1. **Data Loading**: Load và xử lý audio files
2. **Audio Augmentation**: Áp dụng các kỹ thuật augmentation
3. **Feature Extraction**: Trích xuất MFCC features (13 coefficients)
4. **Model Training**: Train CNN-LSTM model với early stopping
5. **Evaluation**: Đánh giá model với confusion matrix và metrics

#### Kết quả Training

- Model file: `wakeword_model.h5`
- Training log: `training_log.csv`
- Analysis plot: `enhanced_training_analysis.png`
- Console output với detailed metrics

### 3. Real-time Detection

```bash
python predict.py
```

#### Cấu hình Detection

Chỉnh sửa các tham số trong `predict.py`:

```python
MODEL_PATH = "wakeword_model.h5"   # Đường dẫn model
SAMPLE_RATE = 16000                # Sample rate
WAKE_THRESHOLD = 0.5               # Ngưỡng detection (0.0-1.0)
VAD_AGGRESSIVENESS = 1             # Độ nhạy VAD (0-3)
```

#### Hoạt động

1. **Audio Streaming**: Capture audio từ microphone
2. **Voice Activity Detection**: Phát hiện phần audio có voice
3. **Feature Extraction**: Trích xuất MFCC features
4. **Wake Word Detection**: Predict confidence score
5. **Threshold Check**: So sánh với ngưỡng để quyết định

## ⚙️ Kiến trúc hệ thống

### Audio Augmentation

- **Real Noise Addition**: Thêm noise thực tế với SNR control
- **Gaussian Noise**: Thêm white noise
- **Time Shift**: Dịch chuyển temporal
- **Speed Perturbation**: Thay đổi tốc độ (0.9x - 1.1x)
- **Volume Perturbation**: Thay đổi âm lượng (0.7x - 1.3x)

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

- **Audio Streaming**: 480 samples/chunk (30ms at 16kHz)
- **VAD Processing**: WebRTC VAD cho speech detection
- **Buffer Management**: 3-second sliding window
- **Threading**: Separate threads cho audio capture và processing

## 📊 Đánh giá hiệu suất

### Metrics

- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Tỷ lệ true positive trong predicted positive
- **Recall**: Tỷ lệ true positive được detect
- **Specificity**: Tỷ lệ true negative được detect chính xác
- **F1 Score**: Harmonic mean của precision và recall

### Confusion Matrix

```
                Predicted
              No    Yes
Actual No   [TN]  [FP]
       Yes  [FN]  [TP]
```

## 🔧 Tối ưu hóa

### Cải thiện độ chính xác

1. **Tăng dữ liệu training**:
   - Thu thập thêm samples từ nhiều người nói
   - Đa dạng hóa môi trường ghi âm
   - Thêm variations của "Hey DT"

2. **Điều chỉnh threshold**:
   - Giảm threshold: Tăng recall, giảm precision
   - Tăng threshold: Tăng precision, giảm recall

3. **Fine-tuning model**:
   - Điều chỉnh architecture (số layers, units)
   - Thay đổi learning rate và optimizer
   - Thêm regularization techniques

### Giảm latency

1. **Tối ưu audio processing**:
   - Giảm chunk size
   - Tối ưu feature extraction

2. **Model optimization**:
   - Model quantization
   - TensorFlow Lite conversion

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
vad_aggressiveness = 1      # VAD sensitivity (0-3)
```

### Debug Mode

Thêm logging để debug:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Notes

- Model hoạt động tốt nhất với audio 16kHz, mono
- Khuyến nghị sử dụng headset microphone để giảm echo
- Threshold mặc định 0.5 phù hợp cho hầu hết trường hợp
- VAD aggressiveness cao (3) nhạy hơn nhưng có thể tăng false positive


