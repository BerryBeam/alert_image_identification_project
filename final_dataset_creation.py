import os
import cv2
import random
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

source_folder = "alerts"
output_folder = "dataset"
train_split = 0.8
aug_per_image = 20
img_size = (224, 224)

# Augmentation generator (simulating webcam-like conditions)
augmenter = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=(0.3, 1.2),
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Clear and recreate output directories
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(os.path.join(output_folder, "train"))
os.makedirs(os.path.join(output_folder, "val"))

# Process each image in the alerts folder
for filename in os.listdir(source_folder):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    class_name = os.path.splitext(filename)[0]
    image_path = os.path.join(source_folder, filename)
    image = cv2.imread(image_path)
    image = cv2.resize(image, img_size)
    image = np.expand_dims(image, 0)

    # Generate augmented images
    aug_images = []
    for _ in range(aug_per_image):
        for batch in augmenter.flow(image, batch_size=1):
            aug_images.append(batch[0].astype(np.uint8))
            break

    # Shuffle and split into train and val sets
    random.shuffle(aug_images)
    split_index = int(train_split * len(aug_images))
    train_imgs = aug_images[:split_index]
    val_imgs = aug_images[split_index:]

    # Create class directories and save images
    train_class_dir = os.path.join(output_folder, "train", class_name)
    val_class_dir = os.path.join(output_folder, "val", class_name)
    os.makedirs(train_class_dir)
    os.makedirs(val_class_dir)

    for idx, img in enumerate(train_imgs):
        cv2.imwrite(os.path.join(train_class_dir, f"{class_name}_train_{idx}.png"), img)

    for idx, img in enumerate(val_imgs):
        cv2.imwrite(os.path.join(val_class_dir, f"{class_name}_val_{idx}.png"), img)

print("Dataset successfully created in 'dataset/' with augmentation and validation split.")
