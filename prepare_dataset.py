import os
from PIL import Image
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
print("data set started")

# Paths
SOURCE_DIR = "alerts"
DEST_DIR = "dataset/train"
AUGMENTATIONS_PER_IMAGE = 20
IMAGE_SIZE = (224, 224)

# Create destination directory
os.makedirs(DEST_DIR, exist_ok=True)

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Process each image
for filename in os.listdir(SOURCE_DIR):
    if filename.lower().endswith(".png"):
        class_name = os.path.splitext(filename)[0]
        class_dir = os.path.join(DEST_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Load image
        img_path = os.path.join(SOURCE_DIR, filename)
        img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Save original
        img.save(os.path.join(class_dir, "original.png"))

        # Generate augmentations
        aug_iter = datagen.flow(img_array, batch_size=1)
        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = next(aug_iter)[0].astype('uint8')
            aug_pil = array_to_img(aug_img)
            aug_pil.save(os.path.join(class_dir, f"aug_{i+1}.png"))

print("✅ Dataset prepared and saved to:", DEST_DIR)
