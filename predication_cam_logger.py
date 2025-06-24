# ===========================================
# 🚨 ALERT DETECTION SYSTEM - DOCUMENTED VERSION
# ===========================================
# Description: This script uses a pre-trained CNN to classify webcam input
# and logs high-confidence alerts with a fallback mechanism for robustness.

# === Step 1: Load class names from label_map.txt ===
class_names = []  # List to store class names read from label_map.txt
with open("label_map.txt", "r") as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            class_names.append(parts[1])  # Extract class label
NUM_CLASSES = len(class_names)
print(f"✓ Loaded {NUM_CLASSES} classes:", class_names)

# === Step 2: Import necessary libraries ===
from datetime import datetime
import numpy as np
import cv2
from base64 import b64decode, b64encode
import time
from google.colab.output import eval_js
from IPython.display import display, HTML, Javascript
import pandas as pd
import traceback
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# === Step 3: Build the model architecture ===
# Load MobileNetV2 without top classification layer
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))(x)

model = Model(inputs=base_model.input, outputs=output)

# === Step 4: Load trained weights ===
model.load_weights('best_model.h5')
print("✓ Model architecture rebuilt and weights loaded")

# === Step 5: Sanity check before running prediction ===
def verify_system():
    checks = {
        'model_loaded': 'model' in globals() and hasattr(model, 'predict'),
        'class_names_exists': 'class_names' in globals() and len(class_names) > 0,
        'preprocess_input': 'preprocess_input' in globals(),
        'img_to_array': 'img_to_array' in globals()
    }
    if not all(checks.values()):
        missing = [k for k,v in checks.items() if not v]
        raise RuntimeError(f"Critical missing components: {missing}")

# ========== LOGGER FOR HIGH-CONFIDENCE ALERTS ==========
class SafetyAlertLogger:
    def __init__(self):
        self.log = pd.DataFrame(columns=['timestamp', 'alert_type', 'confidence', 'duration'])
        self.current_alert = None
        self.alert_start_time = None
        self.alert_buffer = []
        self._backup_file = '/tmp/alert_backup.csv'

    def _atomic_write(self, data):
        try:
            with open(self._backup_file, 'a') as f:
                f.write(','.join(map(str, data)) + '\n')
        except Exception as e:
            print(f"BACKUP FAILED: {str(e)}")

    def update(self, alert_type, confidence):
        try:
            timestamp = datetime.now()
            if not isinstance(confidence, (float, np.floating)):
                raise ValueError(f"Invalid confidence: {confidence}")
            if not isinstance(alert_type, str):
                alert_type = str(alert_type)

            self.alert_buffer.append((alert_type, min(max(confidence, 0.0), 1.0)))
            if len(self.alert_buffer) > 5:
                self.alert_buffer.pop(0)

            alerts = [x[0] for x in self.alert_buffer]
            most_common = max(set(alerts), key=alerts.count, default=None)
            conf_values = [x[1] for x in self.alert_buffer if x[0] == most_common]
            avg_conf = np.mean(conf_values) if conf_values else 0.0

            if most_common != self.current_alert:
                if self.current_alert and self.alert_start_time:
                    duration = (timestamp - self.alert_start_time).total_seconds()
                    if duration >= 1.0:
                        log_entry = [
                            self.alert_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                            self.current_alert,
                            f"{avg_conf:.3f}",
                            f"{duration:.2f}"
                        ]
                        self.log = pd.concat([
                            self.log,
                            pd.DataFrame([dict(zip(['timestamp', 'alert_type', 'confidence', 'duration'], log_entry))])
                        ], ignore_index=True)
                        self._atomic_write(log_entry)
                        display(HTML(
                            f"<div style='color:green;font-weight:bold'>"
                            f"✔ LOGGED: {self.current_alert} ({duration:.1f}s, {avg_conf*100:.1f}%)"
                            f"</div>"
                        ))
                self.current_alert = most_common if avg_conf > 0.35 else None
                self.alert_start_time = timestamp if self.current_alert else None

        except Exception as e:
            print(f"⚠ LOGGER ERROR: {str(e)}")
            traceback.print_exc()

    def get_logs(self):
        try:
            try:
                backup = pd.read_csv(self._backup_file)
                self.log = pd.concat([self.log, backup]).drop_duplicates()
            except FileNotFoundError:
                pass
            return self.log.sort_values('timestamp', ascending=False)
        except Exception as e:
            print(f"LOG RETRIEVAL ERROR: {str(e)}")
            return pd.DataFrame()

# ========== CAMERA HANDLER FOR IMAGE INPUT ==========
class FailsafeCamera:
    def __init__(self):
        self._setup_display()
        self.max_retries = 3
        self.retry_delay = 2

    def _setup_display(self):
        display(HTML('''
        <div id="camera-container">
          <video id="webcam" width="640" height="480" autoplay playsinline style="display:none;"></video>
          <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
          <img id="output" width="640" height="480">
          <div id="status" style="margin:10px;padding:10px;background:#111;color:#eee;font-family:monospace"></div>
        </div>
        '''))

    def _js(self, code):
        for i in range(self.max_retries):
            try:
                return eval_js(code)
            except Exception:
                if i == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)

    def start(self):
        return self._js('''
            (async function() {
                const video = document.getElementById('webcam');
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({video: true});
                    video.srcObject = stream;
                    await video.play();
                    return true;
                } catch (e) {
                    console.error("Camera error:", e);
                    return false;
                }
            })()
        ''')

    def capture_frame(self):
        return self._js('''
            (function() {
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                ctx.drawImage(document.getElementById('webcam'), 0, 0, 640, 480);
                return canvas.toDataURL('image/jpeg', 0.8);
            })()
        ''')

# ========== MAIN LOGIC ==========
try:
    verify_system()
    print("✓ System verification passed")

    logger = SafetyAlertLogger()
    camera = FailsafeCamera()

    if not camera.start():
        raise RuntimeError("Failed to initialize camera")

    display(HTML('<div id="alert-display" style="margin-top:10px;padding:10px;background:#f5f5f5;border:1px solid #ddd"></div>'))

    last_update = time.time()
    while True:
        try:
            frame_data = camera.capture_frame()
            img_bytes = b64decode(frame_data.split(',')[1])
            frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Empty frame captured")

            img = cv2.resize(frame.copy(), (224, 224))
            img = preprocess_input(img_to_array(img))
            preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
            class_idx = np.argmax(preds)
            alert_type = class_names[class_idx]
            confidence = float(preds[class_idx])

            logger.update(alert_type, confidence)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            status_color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"{alert_type} ({confidence*100:.1f}%)", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            if time.time() - last_update > 0.5:
                _, jpeg = cv2.imencode('.jpg', frame)
                current_alert = logger.current_alert if logger.current_alert else "None"
                display(HTML(f'''
                <script>
                document.getElementById("output").src = 'data:image/jpeg;base64,{b64encode(jpeg).decode()}';
                document.getElementById("alert-display").innerHTML =
                    'LAST ALERT: <b>{current_alert}</b> | ' +
                    'CONFIDENCE: <b>{(confidence*100):.1f}%</b> | ' +
                    'LOGGED: <b>{logger.log.shape[0]}</b> | ' +
                    'STATUS: <span style="color:green">ACTIVE</span>';
                </script>
                '''))
                last_update = time.time()

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 Graceful shutdown initiated")
            break
        except Exception as e:
            print(f"⚠ Processing error: {str(e)}")
            traceback.print_exc()
            time.sleep(1)

except Exception as e:
    print(f"🚨 CRITICAL FAILURE: {str(e)}")
    traceback.print_exc()

finally:
    try:
        camera._js('''
        const video = document.getElementById('webcam');
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        ''')
    except:
        pass

    print("\n=== FINAL ALERT LOG ===")
    logs = logger.get_logs()
    display(logs)

    try:
        logs.to_csv('alert_logs_verified.csv', index=False)
        print("✓ Logs saved to 'alert_logs_verified.csv'")
    except Exception as e:
        print(f"⚠ Failed to save logs: {str(e)}")
        print("⚠ Backup available in /tmp/alert_backup.csv")
