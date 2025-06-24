# Import necessary libraries
import os  # For file/folder operations
import numpy as np  # For numerical operations
import tensorflow as tf  # Main deep learning framework
from tensorflow.keras.applications import MobileNetV2  # Pre-trained model we'll use
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Special image formatting
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For loading/processing images
from tensorflow.keras.models import Model  # For building our model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dropout,
                                   Dense, BatchNormalization)  # Layers we'll add
from tensorflow.keras.optimizers import Adam  # Optimization algorithm
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                      ReduceLROnPlateau, TensorBoard)  # Training helpers
from tensorflow.keras.regularizers import l2  # Helps prevent overfitting

# ===== DATA SETUP =====
# Where our training and validation images are stored
train_dir = 'dataset/train'  # Training images folder
val_dir = 'dataset/val'      # Validation images folder
img_size = (224, 224)        # All images will be resized to 224x224 pixels
batch_size = 32              # Number of images processed at once
epochs = 30                  # How many times we'll go through all the data

# ===== IMAGE AUGMENTATION =====
# Creates modified versions of training images to help model learn better
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # MobileNet-specific formatting
    rotation_range=40,        # Rotate images up to 40 degrees
    zoom_range=0.3,           # Zoom in/out up to 30%
    brightness_range=(0.2, 1.3),  # Make images darker (0.2) or brighter (1.3)
    width_shift_range=0.25,   # Shift images left/right by up to 25%
    height_shift_range=0.25,  # Shift images up/down by up to 25%
    shear_range=0.2,          # Tilt images by up to 20 degrees
    horizontal_flip=True,     # Flip images horizontally
    vertical_flip=True,       # Flip images vertically
    fill_mode="nearest"       # How to fill empty spaces during transforms
)

# For validation data, we only do basic preprocessing (no augmentation)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# ===== DATA GENERATORS =====
# These will load images from folders and prepare them for training
train_gen = train_datagen.flow_from_directory(
    train_dir,                # Where training images are
    target_size=img_size,     # Resize all images
    batch_size=batch_size,    # Process 32 images at a time
    class_mode='categorical', # For multi-class classification
    shuffle=True              # Mix up the order of images
)

val_gen = val_datagen.flow_from_directory(
    val_dir,                  # Where validation images are
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False             # Keep in order for evaluation
)

# ===== SAVE LABEL MAPPINGS =====
# Create a file that remembers which number represents which alert type
with open("label_map.txt", "w") as f:
    for label, index in train_gen.class_indices.items():
        f.write(f"{index},{label}\n")  # Format: "0,fire", "1,smoke", etc.

# ===== MODEL SETUP =====
# Start with MobileNetV2 (pre-trained on ImageNet)
base_model = MobileNetV2(
    weights='imagenet',       # Use pre-trained weights
    include_top=False,        # Don't use the original classification layer
    input_shape=(224, 224, 3) # Expects 224x224 color images
)

# ===== FINE-TUNING SETUP =====
# Allow the whole model to be trainable
base_model.trainable = True
# But freeze (make untrainable) all layers except the last 50
# This helps adapt the model to our specific task without forgetting everything
for layer in base_model.layers[:-50]:
    layer.trainable = False

# ===== BUILD OUR CUSTOM CLASSIFIER =====
# Start with the MobileNet's output
x = base_model.output
# Convert the 2D features to 1D
x = GlobalAveragePooling2D()(x)
# Add new layers for our specific task:
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)  # 512-neuron layer
x = BatchNormalization()(x)  # Helps stabilize training
x = Dropout(0.5)(x)         # Randomly turn off 50% of neurons to prevent overfitting
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)  # 256-neuron layer
x = BatchNormalization()(x)
x = Dropout(0.3)(x)         # Randomly turn off 30% of neurons
# Final layer with one output per class (e.g., fire, smoke, etc.)
output = Dense(train_gen.num_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)

# Combine the base model with our new layers
model = Model(inputs=base_model.input, outputs=output)

# ===== OPTIMIZER =====
# Adam optimizer with a low learning rate (so we don't change weights too drastically)
optimizer = Adam(learning_rate=1e-4)

# ===== CLASS WEIGHTS =====
# Helps balance training if some classes have fewer examples
class_counts = np.bincount(train_gen.classes)  # Count images in each class
class_weights = {}
for i in range(len(class_counts)):
    if class_counts[i] > 0:  # Avoid division by zero
        # Give more weight to classes with fewer examples
        class_weights[i] = (1./class_counts[i]) * train_gen.num_classes / sum(1./c for c in class_counts if c > 0)

# ===== COMPILE THE MODEL =====
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Standard loss for classification
    metrics=['accuracy',              # Track accuracy
            tf.keras.metrics.Precision(name='precision'),  # How many alerts were correct
            tf.keras.metrics.Recall(name='recall')]  # How many real alerts we caught
)

# ===== TRAINING HELPERS (CALLBACKS) =====
callbacks = [
    # Save the best model we find
    ModelCheckpoint('best_model.h5',
                  monitor='val_accuracy',  # Watch validation accuracy
                  save_best_only=True,     # Only keep the best
                  mode='max'),            # Higher accuracy is better
    # Stop early if we're not improving
    EarlyStopping(monitor='val_accuracy',
                 patience=15,             # Wait 15 epochs without improvement
                 min_delta=0.001,         # Minimum change to count as improvement
                 mode='max',
                 restore_best_weights=True),  # Keep the best weights
    # Reduce learning rate if stuck
    ReduceLROnPlateau(monitor='val_loss',
                     factor=0.2,          # Reduce learning rate by 80%
                     patience=5,          # Wait 5 epochs before reducing
                     min_lr=1e-6,         # Minimum learning rate
                     verbose=1),          # Show messages
    # Log training progress for TensorBoard
    TensorBoard(log_dir='./logs')
]

# ===== TRAIN THE MODEL =====
history = model.fit(
    train_gen,                      # Training data
    steps_per_epoch=len(train_gen), # How many batches per epoch
    validation_data=val_gen,        # Validation data
    validation_steps=len(val_gen),  # How many validation batches
    epochs=epochs,                 # Total training cycles
    callbacks=callbacks,           # Our helpers from above
    class_weight=class_weights,    # Balance for unequal classes
    verbose=1                      # Show progress
)
