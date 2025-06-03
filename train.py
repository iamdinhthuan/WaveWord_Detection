import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import random
from pathlib import Path
import torch
import torchaudio

warnings.filterwarnings('ignore')


class AugmentationConfig:
    """Configuration cho audio augmentation"""

    def __init__(self):
        # Noise augmentation
        self.noise_dir = "noise"  # Thư mục chứa noise files
        self.noise_snr_range = [5, 20]  # SNR range in dB
        self.gaussian_noise_level = 0.005

        # Time and speed augmentation
        self.time_shift_max = 0.1  # 10% of audio length
        self.speed_range = [0.9, 1.1]  # Speed factor range
        self.volume_range = [0.7, 1.3]  # Volume factor range

        # Augmentation probability
        self.augmentation_prob = 0.5  # Probability to apply each augmentation


class AudioAugmentation:
    """Audio augmentation với real noise và các kỹ thuật khác"""

    def __init__(self, config):
        self.config = config
        self.apply_prob = config.augmentation_prob
        self.noise_files = []
        self.noise_cache = {}

        # Load noise files
        if hasattr(config, 'noise_dir') and config.noise_dir and os.path.exists(config.noise_dir):
            self.load_noise_files(config.noise_dir)
            print(f"🔊 Loaded {len(self.noise_files)} noise files from {config.noise_dir}")
        else:
            print(f"⚠️ No noise directory provided or not found. Using synthetic noise only.")

    def load_noise_files(self, noise_dir):
        """Load tất cả audio files từ noise directory"""
        noise_dir = Path(noise_dir)
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}

        for ext in audio_extensions:
            self.noise_files.extend(list(noise_dir.glob(f'*{ext}')))
            self.noise_files.extend(list(noise_dir.glob(f'*{ext.upper()}')))

        self.noise_files = list(set(self.noise_files))

    def get_noise_segment(self, target_length, sample_rate=16000):
        """Lấy noise segment ngẫu nhiên"""
        if not self.noise_files:
            return None

        noise_file = random.choice(self.noise_files)

        try:
            cache_key = f"{noise_file}_{sample_rate}"
            if cache_key not in self.noise_cache:
                # Try loading with librosa first (more compatible)
                try:
                    waveform, sr = librosa.load(str(noise_file), sr=sample_rate, mono=True)
                    waveform = torch.tensor(waveform, dtype=torch.float32)
                except:
                    # Fallback to torchaudio
                    waveform, sr = torchaudio.load(str(noise_file))

                    # Resample if needed
                    if sr != sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, sample_rate)
                        waveform = resampler(waveform)

                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    waveform = waveform.squeeze()

                self.noise_cache[cache_key] = waveform

                # Limit cache size
                if len(self.noise_cache) > 30:
                    oldest_key = next(iter(self.noise_cache))
                    del self.noise_cache[oldest_key]

            noise = self.noise_cache[cache_key]

            # Handle length
            if len(noise) >= target_length:
                start_idx = random.randint(0, len(noise) - target_length)
                return noise[start_idx:start_idx + target_length]
            else:
                repeat_times = (target_length // len(noise)) + 1
                repeated_noise = noise.repeat(repeat_times)
                return repeated_noise[:target_length]

        except Exception as e:
            print(f"Error loading noise file {noise_file}: {e}")
            return None

    def add_real_noise(self, waveform):
        """Thêm real noise với SNR control"""
        if np.random.random() > self.apply_prob or not self.noise_files:
            return waveform

        noise = self.get_noise_segment(len(waveform))
        if noise is None:
            return waveform

        # Convert to tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # Calculate powers
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        if noise_power == 0:
            return waveform

        # Random SNR
        target_snr_db = random.uniform(self.config.noise_snr_range[0], self.config.noise_snr_range[1])
        target_snr_linear = 10 ** (target_snr_db / 10)

        # Scale noise
        noise_scale = torch.sqrt(signal_power / (noise_power * target_snr_linear))

        result = waveform + noise_scale * noise
        return result.numpy() if isinstance(waveform, torch.Tensor) else result

    def add_gaussian_noise(self, waveform):
        """Thêm Gaussian noise"""
        if np.random.random() > self.apply_prob:
            return waveform

        if isinstance(waveform, np.ndarray):
            noise = np.random.randn(len(waveform)) * self.config.gaussian_noise_level
            return waveform + noise
        else:
            noise = torch.randn_like(waveform) * self.config.gaussian_noise_level
            return waveform + noise

    def time_shift(self, waveform):
        """Time shift augmentation"""
        if np.random.random() > self.apply_prob:
            return waveform

        shift = int(self.config.time_shift_max * len(waveform))
        shift_amount = np.random.randint(-shift, shift)

        if isinstance(waveform, np.ndarray):
            return np.roll(waveform, shift_amount)
        else:
            return torch.roll(waveform, shift_amount, dims=-1)

    def speed_perturbation(self, waveform):
        """Speed perturbation"""
        if np.random.random() > self.apply_prob:
            return waveform

        speed_factor = random.uniform(self.config.speed_range[0], self.config.speed_range[1])

        if speed_factor != 1.0:
            original_length = len(waveform)
            new_length = int(original_length / speed_factor)

            if isinstance(waveform, np.ndarray):
                indices = np.linspace(0, original_length - 1, new_length)
                new_waveform = np.interp(indices, np.arange(original_length), waveform)
            else:
                indices = torch.linspace(0, original_length - 1, new_length)
                waveform_np = waveform.numpy()
                new_waveform = np.interp(indices.numpy(), np.arange(original_length), waveform_np)
                new_waveform = torch.tensor(new_waveform).float()

            # Pad or crop
            if len(new_waveform) > original_length:
                return new_waveform[:original_length]
            elif len(new_waveform) < original_length:
                padding = original_length - len(new_waveform)
                if isinstance(new_waveform, np.ndarray):
                    return np.pad(new_waveform, (0, padding), mode='constant')
                else:
                    return torch.nn.functional.pad(new_waveform, (0, padding))
            else:
                return new_waveform

        return waveform

    def volume_perturbation(self, waveform):
        """Volume perturbation"""
        if np.random.random() > self.apply_prob:
            return waveform

        volume_factor = random.uniform(self.config.volume_range[0], self.config.volume_range[1])
        return waveform * volume_factor

    def apply_augmentations(self, waveform):
        """Apply tất cả augmentations"""
        augmentations = [
            self.add_real_noise,
            self.add_gaussian_noise,
            self.time_shift,
            self.speed_perturbation,
            self.volume_perturbation
        ]

        # Random order
        random.shuffle(augmentations)

        augmented = waveform
        for aug_func in augmentations:
            augmented = aug_func(augmented)

        # Convert back to numpy if needed
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.numpy()

        return augmented


class EnhancedWakeWordDataset:
    def __init__(self, audio_dir, sample_rate=16000, max_length=3.0, use_augmentation=True, noise_dir=None):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(sample_rate * max_length)
        self.use_augmentation = use_augmentation

        # Initialize augmentation
        if use_augmentation:
            aug_config = AugmentationConfig()
            if noise_dir:
                aug_config.noise_dir = noise_dir
            self.augmenter = AudioAugmentation(aug_config)
        else:
            self.augmenter = None

    def extract_features(self, file_path, apply_augmentation=False):
        """Trích xuất MFCC features từ audio file với augmentation option"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate)

            # Apply augmentation if requested
            if apply_augmentation and self.augmenter:
                audio = self.augmenter.apply_augmentations(audio)

            # Pad hoặc truncate để có độ dài cố định
            if len(audio) < self.max_samples:
                # Pad với zeros
                audio = np.pad(audio, (0, self.max_samples - len(audio)), mode='constant')
            else:
                # Truncate
                audio = audio[:self.max_samples]

            # Trích xuất MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)

            # Normalize
            mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)

            return mfccs.T  # Transpose để có shape (time_steps, features)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def load_data(self, augmentation_factor=2):
        """
        Load và xử lý tất cả audio files với augmentation

        Args:
            augmentation_factor: Số lượng augmented samples cho mỗi original sample
        """
        X = []
        y = []

        # Load wakeword samples (label = 1)
        wakeword_dir = os.path.join(self.audio_dir, 'wakeword')
        if os.path.exists(wakeword_dir):
            files = [f for f in os.listdir(wakeword_dir) if f.endswith('.wav')]
            print(f"Loading {len(files)} wakeword samples...")

            for file in tqdm(files, desc="Processing wakeword"):
                file_path = os.path.join(wakeword_dir, file)

                # Original sample
                features = self.extract_features(file_path, apply_augmentation=False)
                if features is not None:
                    X.append(features)
                    y.append(1)

                # Augmented samples
                if self.use_augmentation and self.augmenter:
                    for _ in range(augmentation_factor):
                        features_aug = self.extract_features(file_path, apply_augmentation=True)
                        if features_aug is not None:
                            X.append(features_aug)
                            y.append(1)

        # Load non-wakeword samples (label = 0)
        non_wakeword_dir = os.path.join(self.audio_dir, 'non-wakeword')
        if os.path.exists(non_wakeword_dir):
            files = [f for f in os.listdir(non_wakeword_dir) if f.endswith('.wav')]
            print(f"Loading {len(files)} non-wakeword samples...")

            for file in tqdm(files, desc="Processing non-wakeword"):
                file_path = os.path.join(non_wakeword_dir, file)

                # Original sample
                features = self.extract_features(file_path, apply_augmentation=False)
                if features is not None:
                    X.append(features)
                    y.append(0)

                # Augmented samples
                if self.use_augmentation and self.augmenter:
                    for _ in range(augmentation_factor):
                        features_aug = self.extract_features(file_path, apply_augmentation=True)
                        if features_aug is not None:
                            X.append(features_aug)
                            y.append(0)

        return np.array(X), np.array(y)


def create_enhanced_model(input_shape, dropout_rate=0.5):
    """Tạo enhanced CNN-LSTM model với regularization tốt hơn"""
    model = keras.Sequential([
        # CNN layers với BatchNorm và Dropout
        keras.layers.Conv1D(16, 5, activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(dropout_rate * 0.5),

        keras.layers.Conv1D(32, 5, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(dropout_rate * 0.5),

        # LSTM layers với Dropout
        keras.layers.LSTM(32, return_sequences=True, dropout=dropout_rate * 0.5),
        keras.layers.LSTM(16, dropout=dropout_rate * 0.5),
        keras.layers.Dropout(dropout_rate * 0.6),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(dropout_rate * 0.6),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


def train_wakeword_model(audio_dir, noise_dir=None, model_save_path='wakeword_model.h5'):
    """Main function để train enhanced wake word detection model"""
    print("start training Wake Word Detection Model...")
    print("Features: Audio Augmentation + Enhanced Architecture")

    # 1. Load và xử lý dữ liệu với augmentation
    dataset = EnhancedWakeWordDataset(
        audio_dir,
        use_augmentation=True,
        noise_dir=noise_dir
    )

    # Load với augmentation factor
    augmentation_factor = 2  # Mỗi sample tạo 3 augmented versions
    X, y = dataset.load_data(augmentation_factor=augmentation_factor)

    if len(X) == 0:
        print("❌ Không tìm thấy dữ liệu audio!")
        return None

    print(f"\n📊 Dataset Summary:")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Wakeword samples: {np.sum(y == 1)}")
    print(f"   - Non-wakeword samples: {np.sum(y == 0)}")
    print(f"   - Feature shape: {X[0].shape}")
    print(f"   - Augmentation factor: {augmentation_factor}")

    # 2. Chia train/validation set với stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"\n📈 Data Split:")
    print(f"   - Train set: {len(X_train)} samples")
    print(f"   - Validation set: {len(X_val)} samples")
    print(f"   - Train wakeword ratio: {np.sum(y_train == 1) / len(y_train):.3f}")
    print(f"   - Val wakeword ratio: {np.sum(y_val == 1) / len(y_val):.3f}")

    # 3. Tạo enhanced model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_enhanced_model(input_shape, dropout_rate=0.5)

    # 4. Compile với advanced optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    print("\️  Enhanced Model Architecture:")
    model.summary()

    # 5. Advanced callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=7,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger('training_log.csv')
    ]

    # 6. Class weight balancing (nếu dataset không balanced)
    unique, counts = np.unique(y_train, return_counts=True)
    class_weight = {
        0: len(y_train) / (2 * counts[0]),
        1: len(y_train) / (2 * counts[1])
    }
    print(f"\n  Class weights: {class_weight}")

    # 7. Training với enhanced parameters
    print("\n Bắt đầu enhanced training...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        batch_size=64,  # Larger batch size
        epochs=150,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time / 60:.1f} minutes")

    # 8. Comprehensive evaluation
    print("\n📊 Model Evaluation:")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)

    # Calculate additional metrics
    val_predictions = model.predict(X_val, verbose=0)
    val_pred_binary = (val_predictions > 0.5).astype(int).flatten()

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_val, val_pred_binary)

    # Calculate specificity (true negative rate)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (
                                                                                              val_precision + val_recall) > 0 else 0

    print(f"   - Validation Accuracy: {val_acc:.4f}")
    print(f"   - Validation Precision: {val_precision:.4f}")
    print(f"   - Validation Recall: {val_recall:.4f}")
    print(f"   - Validation Specificity: {specificity:.4f}")
    print(f"   - F1 Score: {f1_score:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {tn}, FP: {fp}")
    print(f"   FN: {fn}, TP: {tp}")

    # 9. Enhanced visualization
    plt.figure(figsize=(15, 10))

    # Training history
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Precision and Recall
    plt.subplot(2, 3, 3)
    plt.plot(history.history['precision'], label='Training Precision', linewidth=2)
    plt.plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
    plt.title('Model Precision', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.plot(history.history['recall'], label='Training Recall', linewidth=2)
    plt.plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    plt.title('Model Recall', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confusion Matrix
    plt.subplot(2, 3, 5)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Prediction distribution
    plt.subplot(2, 3, 6)
    plt.hist(val_predictions[y_val == 0], bins=30, alpha=0.7, label='Non-wakeword', density=True)
    plt.hist(val_predictions[y_val == 1], bins=30, alpha=0.7, label='Wakeword', density=True)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.title('Prediction Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 10. Model analysis
    print(f"\n✅ Enhanced model training completed!")
    print(f"📁 Model saved: {model_save_path}")
    print(f"📊 Training log: training_log.csv")
    print(f"📈 Analysis plot: enhanced_training_analysis.png")

    return model, history


# Sử dụng
if __name__ == "__main__":
    import time

    # Configuration
    AUDIO_DIR = "dir_folder_data"
    NOISE_DIR = "dir_noise_audio"
    MODEL_PATH = "wakeword_model.h5"

    print("Wake Word Detection Training")
    print("=" * 50)
    print(f"📂 Audio directory: {AUDIO_DIR}")
    print(f"🔊 Noise directory: {NOISE_DIR}")
    print(f"💾 Model will be saved to: {MODEL_PATH}")

    # Train model
    model, history = train_wakeword_model(
        audio_dir=AUDIO_DIR,
        noise_dir=NOISE_DIR,
        model_save_path=MODEL_PATH
    )

    print("\n🎉 Training completed successfully!")
    print("🔄 You can now use this model with the real-time detection script.")