# ===============================================================================
# REAL-TIME SAFETY ALERT DETECTION SYSTEM
# ===============================================================================
# WORKFLOW: Initialization → Model Setup → Camera Setup → Real-time Processing → Logging
# ===============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: CLASS LABEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
# Load safety alert class names from configuration file
# INPUT: label_map.txt (format: "id,class_name" per line)
# OUTPUT: class_names list, NUM_CLASSES count
# ═══════════════════════════════════════════════════════════════════════════════

class_names = []
with open("label_map.txt", "r") as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            class_names.append(parts[1])  # Extract class name (second column)

NUM_CLASSES = len(class_names)
print(f"Loaded {NUM_CLASSES} classes:", class_names)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: DEPENDENCY IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
# Import all required libraries for the system components
# - Computer Vision: cv2, numpy
# - Deep Learning: tensorflow.keras
# - Web Interface: google.colab, IPython
# - Data Handling: pandas, base64
# - Utilities: datetime, time, traceback
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: NEURAL NETWORK MODEL ARCHITECTURE SETUP
# ═══════════════════════════════════════════════════════════════════════════════
# Build custom MobileNetV2-based classifier for safety alert detection
# ARCHITECTURE FLOW:
# Input(224x224x3) → MobileNetV2 Base → GlobalAvgPool → Dense(512) → 
# BatchNorm → Dropout → Dense(256) → BatchNorm → Dropout → Dense(NUM_CLASSES)
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize MobileNetV2 base model without pretrained weights
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)                                    # Reduce spatial dimensions
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x) # First dense layer with L2 regularization
x = BatchNormalization()(x)                                        # Normalize for stable training
x = Dropout(0.5)(x)                                               # Prevent overfitting (50% dropout)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x) # Second dense layer
x = BatchNormalization()(x)                                        # Second normalization
x = Dropout(0.3)(x)                                               # Lighter dropout (30%)
output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.001))(x) # Final classification layer

# Create the complete model
model = Model(inputs=base_model.input, outputs=output)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: TRAINED MODEL WEIGHTS LOADING
# ═══════════════════════════════════════════════════════════════════════════════
# Load previously trained weights into the model architecture
# REQUIREMENT: 'best_model.h5' must exist and match the architecture
# ═══════════════════════════════════════════════════════════════════════════════

model.load_weights('best_model.h5')
print("Model architecture rebuilt and weights loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: SYSTEM VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
# Validate all critical components before starting main execution
# CHECKS: Model loading, class names, preprocessing functions
# FAILURE MODE: Raise RuntimeError if any component is missing
# ═══════════════════════════════════════════════════════════════════════════════

def verify_system():
    """Validate all critical components before starting execution."""
    checks = {
        'model_loaded': 'model' in globals() and hasattr(model, 'predict'),
        'class_names_exists': 'class_names' in globals() and len(class_names) > 0,
        'preprocess_input': 'preprocess_input' in globals(),
        'img_to_array': 'img_to_array' in globals()
    }

    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Critical missing components: {missing}")

# ═══════════════════════════════════════════════════════════════════════════════
# CLASS DEFINITION: SAFETY ALERT LOGGER
# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW: Buffer Predictions → Detect State Changes → Log Persistent Alerts → Backup Data
# PURPOSE: Track and log safety alerts with confidence levels and durations
# FEATURES: Alert buffering, state change detection, crash-resistant backup
# ═══════════════════════════════════════════════════════════════════════════════

class SafetyAlertLogger:
    def __init__(self):
        """Initialize logger with empty dataframe and state tracking variables."""
        self.log = pd.DataFrame(columns=['timestamp', 'alert_type', 'confidence', 'duration'])
        self.current_alert = None           # Currently active alert type
        self.alert_start_time = None        # When current alert began
        self.alert_buffer = []              # Rolling buffer of recent predictions
        self._backup_file = '/tmp/alert_backup.csv'  # Crash-resistant backup location

    def _atomic_write(self, data):
        """
        CRASH-RESISTANT BACKUP: Write alert data to backup file
        PURPOSE: Preserve data even if main process crashes
        INPUT: data - list of values to write as CSV row
        """
        try:
            with open(self._backup_file, 'a') as f:
                f.write(','.join(map(str, data)) + '\n')
        except Exception as e:
            print(f"Backup write failed: {str(e)}")

    def update(self, alert_type, confidence):
        """
        MAIN LOGGING WORKFLOW:
        1. Validate input data
        2. Add to rolling buffer
        3. Determine most common alert in buffer
        4. Detect alert state changes
        5. Log completed alerts with duration
        6. Update current alert state
        
        INPUT: alert_type (str), confidence (float 0-1)
        SIDE EFFECTS: Updates internal state, writes to backup, displays logs
        """
        try:
            timestamp = datetime.now()

            # ─── INPUT VALIDATION ───
            if not isinstance(confidence, (float, np.floating)):
                raise ValueError(f"Invalid confidence: {confidence}")
            if not isinstance(alert_type, str):
                alert_type = str(alert_type)

            # ─── ROLLING BUFFER MANAGEMENT ───
            # Add current prediction to buffer, maintain buffer size of 5
            self.alert_buffer.append((alert_type, min(max(confidence, 0.0), 1.0)))
            if len(self.alert_buffer) > 5:
                self.alert_buffer.pop(0)

            # ─── CONSENSUS DETECTION ───
            # Determine most common alert type in buffer for stability
            if self.alert_buffer:
                alerts = [x[0] for x in self.alert_buffer]
                most_common = max(set(alerts), key=alerts.count, default=None)
                conf_values = [x[1] for x in self.alert_buffer if x[0] == most_common]
                avg_conf = np.mean(conf_values) if conf_values else 0.0
            else:
                most_common, avg_conf = None, 0.0

            # ─── ALERT STATE CHANGE DETECTION ───
            # If alert type changed, log the previous alert if it was significant
            if most_common != self.current_alert:
                if self.current_alert and self.alert_start_time:
                    duration = (timestamp - self.alert_start_time).total_seconds()
                    # Only log alerts that lasted at least 1 second
                    if duration >= 1.0:
                        log_entry = [
                            self.alert_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                            self.current_alert,
                            f"{avg_conf:.3f}",
                            f"{duration:.2f}"
                        ]
                        # Add to main log dataframe
                        self.log = pd.concat([
                            self.log,
                            pd.DataFrame([dict(zip(['timestamp', 'alert_type', 'confidence', 'duration'], log_entry))])
                        ], ignore_index=True)
                        # Write to backup file
                        self._atomic_write(log_entry)
                        # Display confirmation
                        display(HTML(f"<div style='color:green;font-weight:bold'>Logged: {self.current_alert} ({duration:.1f}s, {avg_conf*100:.1f}%)</div>"))

                # ─── NEW ALERT STATE SETUP ───
                # Start tracking new alert only if confidence is above threshold
                self.current_alert = most_common if avg_conf > 0.35 else None
                self.alert_start_time = timestamp if self.current_alert else None

        except Exception as e:
            print(f"Logger error: {str(e)}")
            traceback.print_exc()

    def get_logs(self):
        """
        RETRIEVE ALL LOGS: Combine main log with backup data
        RECOVERY: Load backup file if available and merge with current logs
        OUTPUT: Complete sorted DataFrame of all logged alerts
        """
        try:
            # Try to load backup data and merge
            try:
                backup = pd.read_csv(self._backup_file)
                self.log = pd.concat([self.log, backup]).drop_duplicates()
            except FileNotFoundError:
                pass  # No backup file exists yet

            return self.log.sort_values('timestamp', ascending=False)
        except Exception as e:
            print(f"Log retrieval error: {str(e)}")
            return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════════════════════
# CLASS DEFINITION: FAILSAFE CAMERA HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW: Setup HTML Interface → Initialize Camera → Capture Frames → Handle Errors
# PURPOSE: Robust camera interface with retry logic and error handling
# FEATURES: Automatic retry, graceful error handling, HTML5 video integration
# ═══════════════════════════════════════════════════════════════════════════════

class FailsafeCamera:
    def __init__(self):
        """Initialize camera with HTML interface and retry parameters."""
        self._setup_display()
        self.max_retries = 3    # Maximum retry attempts for failed operations
        self.retry_delay = 2    # Seconds to wait between retries

    def _setup_display(self):
        """
        HTML INTERFACE SETUP: Create camera display elements
        ELEMENTS:
        - webcam: Hidden video element for camera stream
        - canvas: Hidden canvas for frame capture
        - output: Visible image element for processed frames
        - status: Status display area
        """
        display(HTML('''
        <div id="camera-container">
          <video id="webcam" width="640" height="480" autoplay playsinline style="display:none;"></video>
          <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
          <img id="output" width="640" height="480">
          <div id="status" style="margin:10px;padding:10px;background:#111;color:#eee;font-family:monospace"></div>
        </div>
        '''))

    def _js(self, code):
        """
        JAVASCRIPT EXECUTION WITH RETRY LOGIC
        PURPOSE: Execute JavaScript code with automatic retry on failure
        INPUT: code - JavaScript code string to execute
        OUTPUT: Result of JavaScript execution
        RETRY LOGIC: Up to max_retries attempts with delay between failures
        """
        for i in range(self.max_retries):
            try:
                return eval_js(code)
            except Exception as e:
                if i == self.max_retries - 1:  # Last attempt failed
                    raise
                time.sleep(self.retry_delay)

    def start(self):
        """
        CAMERA INITIALIZATION WORKFLOW:
        1. Request camera permissions from browser
        2. Create video stream from camera
        3. Start video playback
        4. Return success/failure status
        
        OUTPUT: Boolean indicating camera initialization success
        ERROR HANDLING: Graceful failure with error logging
        """
        try:
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
        except Exception as e:
            raise RuntimeError(f"Camera initialization failed: {str(e)}")

    def capture_frame(self):
        """
        FRAME CAPTURE WORKFLOW:
        1. Draw current video frame to hidden canvas
        2. Convert canvas to base64 JPEG data
        3. Return data URL for processing
        
        OUTPUT: Base64-encoded JPEG image data
        ERROR HANDLING: Raise RuntimeError on capture failure
        """
        try:
            return self._js('''
            (function() {
                const canvas = document.getElementById('canvas');
                const ctx = canvas.getContext('2d');
                ctx.drawImage(document.getElementById('webcam'), 0, 0, 640, 480);
                return canvas.toDataURL('image/jpeg', 0.8);
            })()
            ''')
        except Exception as e:
            raise RuntimeError(f"Frame capture failed: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════
# FLOW: System Verification → Component Initialization → Real-time Processing Loop → Cleanup
# COMPONENTS: Logger, Camera, Model Inference, Display Updates
# ERROR HANDLING: Graceful degradation, comprehensive logging, proper cleanup
# ═══════════════════════════════════════════════════════════════════════════════

try:
    # ─── PHASE 1: SYSTEM VERIFICATION ───
    verify_system()
    print("System verification passed")

    # ─── PHASE 2: COMPONENT INITIALIZATION ───
    logger = SafetyAlertLogger()  # Initialize alert logging system
    camera = FailsafeCamera()     # Initialize camera interface

    # ─── PHASE 3: CAMERA STARTUP ───
    if not camera.start():
        raise RuntimeError("Failed to initialize camera")

    # ─── PHASE 4: UI SETUP ───
    # Create alert display area in the interface
    display(HTML('<div id="alert-display" style="margin-top:10px;padding:10px;background:#f5f5f5;border:1px solid #ddd"></div>'))

    # ─── PHASE 5: REAL-TIME PROCESSING LOOP ───
    last_update = time.time()  # Track display update timing
    
    while True:  # Main processing loop
        try:
            # ═══ FRAME ACQUISITION ═══
            frame_data = camera.capture_frame()                              # Get base64 frame data
            img_bytes = b64decode(frame_data.split(',')[1])                  # Decode base64 to bytes
            frame = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)  # Convert to OpenCV format

            if frame is None:
                raise ValueError("Empty frame captured")

            # ═══ IMAGE PREPROCESSING ═══
            img = cv2.resize(frame.copy(), (224, 224))        # Resize to model input size
            img = preprocess_input(img_to_array(img))         # Apply MobileNetV2 preprocessing
            
            # ═══ MODEL INFERENCE ═══
            preds = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]  # Get model predictions
            class_idx = np.argmax(preds)                      # Find highest confidence class
            alert_type = class_names[class_idx]               # Map to class name
            confidence = float(preds[class_idx])              # Extract confidence score

            # ═══ ALERT LOGGING ═══
            logger.update(alert_type, confidence)             # Update logger with current prediction

            # ═══ FRAME ANNOTATION ═══
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # Convert to RGB for display
            # Determine status color based on confidence level
            status_color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
            # Add prediction text overlay
            cv2.putText(frame, f"{alert_type} ({confidence*100:.1f}%)", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # ═══ DISPLAY UPDATE (THROTTLED) ═══
            # Update display every 0.5 seconds to avoid overwhelming the interface
            if time.time() - last_update > 0.5:
                _, jpeg = cv2.imencode('.jpg', frame)         # Encode frame as JPEG
                current_alert = logger.current_alert if logger.current_alert else "None"
                
                # Update HTML display with current frame and status
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

            time.sleep(0.1)  # Brief pause to prevent excessive CPU usage

        except KeyboardInterrupt:
            # ─── GRACEFUL SHUTDOWN ───
            print("Graceful shutdown initiated")
            break
        except Exception as e:
            # ─── ERROR HANDLING ───
            print(f"Processing error: {str(e)}")
            traceback.print_exc()
            time.sleep(1)  # Wait before retrying

except Exception as e:
    # ─── CRITICAL FAILURE HANDLING ───
    print(f"Critical system failure: {str(e)}")
    traceback.print_exc()

finally:
    # ═══════════════════════════════════════════════════════════════════════════════
    # CLEANUP AND SHUTDOWN SEQUENCE
    # ═══════════════════════════════════════════════════════════════════════════════
    # WORKFLOW: Stop Camera → Display Final Logs → Save Data → Report Status
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # ─── CAMERA CLEANUP ───
    try:
        camera._js('''
        const video = document.getElementById('webcam');
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        ''')
    except:
        pass  # Ignore cleanup failures

    # ─── FINAL LOGGING AND DATA EXPORT ───
    print("\nFinal Alert Log")
    logs = logger.get_logs()      # Retrieve all logged alerts
    display(logs)                 # Display final log summary

    # ─── DATA PERSISTENCE ───
    try:
        logs.to_csv('alert_logs_verified.csv', index=False)  # Save to CSV file
        print("Logs saved to 'alert_logs_verified.csv'")
    except Exception as e:
        print(f"Failed to save logs: {str(e)}")
        print("Backup available in /tmp/alert_backup.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# END OF SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
