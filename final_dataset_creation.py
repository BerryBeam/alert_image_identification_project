# Import necessary libraries
import os  # For working with files/folders
import cv2  # OpenCV for image processing
import random  # For shuffling data randomly
import shutil  # For deleting/creating folders
import numpy as np  # For numerical operations on images
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image augmentation

# ===== CONFIGURATION SETTINGS =====
# Folder where original alert images are stored
source_folder = "alerts"
# Folder where augmented dataset will be saved
output_folder = "dataset"
# 80% of images go to training, 20% to validation
train_split = 0.8
# How many augmented versions to create per original image
aug_per_image = 20
# All images will be resized to this (width, height)
img_size = (224, 224)

# ===== IMAGE AUGMENTATION SETUP =====
# Creates variations of images to simulate different conditions
# (like how a webcam might see the alert differently each time)
augmenter = ImageDataGenerator(
    rotation_range=30,  # Rotate image up to 30 degrees left/right
    width_shift_range=0.2,  # Shift image left/right by up to 20%
    height_shift_range=0.2,  # Shift image up/down by up to 20%
    zoom_range=0.3,  # Zoom in/out by up to 30%
    brightness_range=(0.3, 1.2),  # Make darker (0.3) or brighter (1.2)
    shear_range=0.2,  # Tilt image by up to 20 degrees
    horizontal_flip=True,  # Flip image left-right sometimes
    fill_mode="nearest"  # How to fill empty spaces during transforms
)

# ===== PREPARE OUTPUT FOLDERS =====
# Remove old dataset folder if it exists (start fresh)
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
# Create new empty folders for training and validation data
os.makedirs(os.path.join(output_folder, "train"))
os.makedirs(os.path.join(output_folder, "val"))

# ===== PROCESS EACH ALERT IMAGE =====
# Loop through all files in the alerts folder
for filename in os.listdir(source_folder):
    # Skip files that aren't images (like .txt files)
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Get the class name from filename (without extension)
    # Example: "fire.png" becomes class "fire"
    class_name = os.path.splitext(filename)[0]
    
    # Full path to the original image
    image_path = os.path.join(source_folder, filename)
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Resize image to standard size (224x224 pixels)
    image = cv2.resize(image, img_size)
    
    # Add extra dimension (required by the augmenter)
    # Changes shape from (224,224,3) to (1,224,224,3)
    image = np.expand_dims(image, 0)

    # ===== CREATE AUGMENTED IMAGES =====
    aug_images = []  # Will store all generated images
    
    # Create 'aug_per_image' (20) new versions of each original image
    for _ in range(aug_per_image):
        # augmenter.flow() generates one augmented image at a time
        for batch in augmenter.flow(image, batch_size=1):
            # Get the augmented image and convert to normal 0-255 values
            aug_images.append(batch[0].astype(np.uint8))
            break  # Only take one image per loop

    # ===== SPLIT INTO TRAIN & VALIDATION SETS =====
    # Shuffle all augmented images randomly
    random.shuffle(aug_images)
    
    # Calculate where to split (80% for training, 20% for validation)
    split_index = int(train_split * len(aug_images))
    
    # Split the list into two parts
    train_imgs = aug_images[:split_index]  # First 80%
    val_imgs = aug_images[split_index:]    # Remaining 20%

    # ===== SAVE IMAGES TO FOLDERS =====
    # Create separate folders for each class (like "fire/train", "fire/val")
    train_class_dir = os.path.join(output_folder, "train", class_name)
    val_class_dir = os.path.join(output_folder, "val", class_name)
    os.makedirs(train_class_dir)
    os.makedirs(val_class_dir)

    # Save all training images
    for idx, img in enumerate(train_imgs):
        cv2.imwrite(
            os.path.join(train_class_dir, f"{class_name}_train_{idx}.png"),
            img
        )

    # Save all validation images
    for idx, img in enumerate(val_imgs):
        cv2.imwrite(
            os.path.join(val_class_dir, f"{class_name}_val_{idx}.png"),
            img
        )

# Final success message
print("Dataset successfully created in 'dataset/' with augmentation and validation split.")
