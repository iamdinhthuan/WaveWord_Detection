import pyaudio
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import webrtcvad
import collections
import threading
import time
import queue
import wave
from datetime import datetime
from scipy.signal import resample


class RealTimeWakeWordDetector:
    def __init__(self, model_path, sample_rate=16000, chunk_size=480,
                 vad_aggressiveness=3, wake_threshold=0.5):
        """
        Real-time WakeWord Detector với VAD

        Args:
            model_path: Đường dẫn đến model đã train
            sample_rate: Sample rate (16kHz cho WebRTC VAD)
            chunk_size: Kích thước chunk audio (30ms cho VAD)
            vad_aggressiveness: Độ nhạy VAD (0-3, 3 là nhạy nhất)
            wake_threshold: Ngưỡng để quyết định wake word
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # 480 samples = 30ms at 16kHz
        self.wake_threshold = wake_threshold

        # Load model
        print("🔄 Đang load model...")
        self.model = keras.models.load_model(model_path)
        print("✅ Model đã được load!")

        # Warmup model
        self._warmup_model()

        # Initialize VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # Audio buffer cho wake word detection
        self.audio_buffer = collections.deque(maxlen=int(3 * sample_rate))  # 3 seconds buffer
        self.voice_frames = collections.deque(maxlen=50)  # Buffer cho voice frames

        # Threading
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.detection_active = True

        # Statistics
        self.detection_count = 0
        self.last_detection_time = None

    def _warmup_model(self):
        """Warmup model với dummy data để tăng tốc inference đầu tiên"""
        print("🔄 Warming up model...")
        try:
            # Tạo dummy features với shape tương tự thực tế
            dummy_features = np.random.random((1, 94, 13))  # (batch, time_steps, features)

            # Chạy inference dummy
            _ = self.model.predict(dummy_features, verbose=0)

            print("✅ Model warmup completed!")
        except Exception as e:
            print(f"⚠️ Warmup failed: {e}")

    def extract_features(self, audio_data):
        """Trích xuất MFCC features từ audio data"""
        try:
            # Ensure audio has correct length (3 seconds)
            target_length = int(self.sample_rate * 3)

            if len(audio_data) < target_length:
                # Pad with zeros
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
            else:
                # Take last 3 seconds
                audio_data = audio_data[-target_length:]

            # Extract MFCC
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )

            # Normalize
            mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)

            return mfccs.T  # Shape: (time_steps, features)

        except Exception as e:
            print(f"❌ Lỗi extract features: {e}")
            return None

    def is_speech(self, audio_chunk):
        """Kiểm tra chunk audio có chứa speech không sử dụng VAD"""
        try:
            # Convert to bytes for VAD
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            return self.vad.is_speech(audio_bytes, self.sample_rate)
        except:
            return False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function cho audio stream"""
        if status:
            print(f"Audio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Add to queue for processing
        self.audio_queue.put(audio_data)

        return (None, pyaudio.paContinue)

    def process_audio(self):
        """Xử lý audio từ queue"""
        speech_frames = []
        non_speech_count = 0

        while self.is_listening:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)

                # Check if chunk contains speech
                has_speech = self.is_speech(audio_chunk)

                if has_speech:
                    # Add to speech buffer
                    speech_frames.extend(audio_chunk)
                    non_speech_count = 0

                    # Add to main audio buffer
                    self.audio_buffer.extend(audio_chunk)

                    print("🎤 Speech detected", end='\r')

                else:
                    non_speech_count += 1

                    # If we have accumulated speech and now silence
                    if len(speech_frames) > 0 and non_speech_count > 5:  # ~150ms silence

                        # Process accumulated speech for wake word detection
                        if len(speech_frames) > self.sample_rate * 0.5:  # At least 0.5s of speech
                            self.detect_wake_word(np.array(speech_frames))

                        # Clear speech buffer
                        speech_frames = []
                        non_speech_count = 0

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Lỗi xử lý audio: {e}")

    def detect_wake_word(self, audio_segment):
        """Detect wake word từ audio segment"""
        if not self.detection_active:
            return

        try:
            # Extract features
            features = self.extract_features(audio_segment)
            if features is None:
                return

            # Predict
            features_batch = np.expand_dims(features, axis=0)
            prediction = self.model.predict(features_batch, verbose=0)[0][0]

            # Check threshold
            if prediction > self.wake_threshold:
                self.detection_count += 1
                self.last_detection_time = datetime.now()

                print(f"\n🎯 WAKE WORD DETECTED! (Confidence: {prediction:.4f})")
                print(f"   Detection #{self.detection_count} at {self.last_detection_time.strftime('%H:%M:%S')}")

                # Optional: Pause detection for a short time to avoid multiple triggers
                self.detection_active = False
                threading.Timer(2.0, self.reactivate_detection).start()

                # Save detection audio for analysis (optional)
                self.save_detection_audio(audio_segment)

            else:
                print(f"🔍 Checking... (Confidence: {prediction:.4f})", end='\r')

        except Exception as e:
            print(f"❌ Lỗi detection: {e}")

    def reactivate_detection(self):
        """Kích hoạt lại detection sau khi detect wake word"""
        self.detection_active = True
        print("✅ Detection reactivated")

    def save_detection_audio(self, audio_data):
        """Lưu audio khi detect wake word để phân tích"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.wav"

            # Resample to 16kHz if needed
            if len(audio_data) > 0:
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                print(f"💾 Saved detection audio: {filename}")

        except Exception as e:
            print(f"❌ Lỗi lưu audio: {e}")

    def start_listening(self):
        """Bắt đầu listening"""
        print("🎙️ Initializing audio stream...")

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        try:
            # Open audio stream
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )

            print("✅ Audio stream initialized!")
            print(f"🎯 Wake word threshold: {self.wake_threshold}")
            print("🎤 Listening for wake word... (Press Ctrl+C to stop)")
            print("-" * 50)

            # Start audio processing thread
            self.is_listening = True
            processing_thread = threading.Thread(target=self.process_audio)
            processing_thread.daemon = True
            processing_thread.start()

            # Start audio stream
            stream.start_stream()

            # Keep main thread alive
            try:
                while stream.is_active():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n⏹️ Stopping...")

        except Exception as e:
            print(f"❌ Lỗi audio stream: {e}")

        finally:
            # Cleanup
            self.is_listening = False
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()

            print(f"\n📊 Session Statistics:")
            print(f"   - Total detections: {self.detection_count}")
            if self.last_detection_time:
                print(f"   - Last detection: {self.last_detection_time.strftime('%H:%M:%S')}")


def main():
    """Main function"""
    print("🎤 Real-time Wake Word Detection")
    print("=" * 40)

    # Configuration
    MODEL_PATH = "wakeword_model.h5"  # Đường dẫn model
    SAMPLE_RATE = 16000
    WAKE_THRESHOLD = 0.5  # Điều chỉnh threshold này
    VAD_AGGRESSIVENESS = 1  # 0-3, 3 là nhạy nhất

    try:
        # Initialize detector
        detector = RealTimeWakeWordDetector(
            model_path=MODEL_PATH,
            sample_rate=SAMPLE_RATE,
            wake_threshold=WAKE_THRESHOLD,
            vad_aggressiveness=VAD_AGGRESSIVENESS
        )

        # Start listening
        detector.start_listening()

    except FileNotFoundError:
        print(f"❌ Không tìm thấy model file: {MODEL_PATH}")
        print("Hãy đảm bảo bạn đã train model trước!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    main()