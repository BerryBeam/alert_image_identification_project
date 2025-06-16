import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the model
model = load_model('car_symbol_transfer.h5')

# Get class labels
# This must match the training class order
class_names = sorted(os.listdir('dataset/train'))# Adjust if folder path is different

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set model input size
img_size = (224, 224)  # Change if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    image = cv2.resize(frame, img_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalization
    image = np.expand_dims(image, axis=0)

    # Make prediction
    preds = model.predict(image)
    class_index = np.argmax(preds[0])
    confidence = preds[0][class_index]
    label = class_names[class_index]

    # Show result
    text = f"{label} ({confidence * 100:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Car Symbol Detector", frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
