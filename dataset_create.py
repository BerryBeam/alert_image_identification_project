import os
import shutil
import random

# 📁 Original alerts folder
source_dir = 'alerts'

# 📁 New dataset structure
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# 🔧 Split ratio
val_split = 0.2  # 20% for validation

# 🧹 Clean previous runs
for folder in [train_dir, val_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# 🔁 For each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    val_count = int(len(images) * val_split)
    train_images = images[val_count:]
    val_images = images[:val_count]

    # 🗂️ Create class folders in train and val
    for split, split_images in [('train', train_images), ('val', val_images)]:
        split_class_dir = os.path.join(f'dataset/{split}', class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for img_name in split_images:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(split_class_dir, img_name)
            shutil.copy2(src, dst)

print("✅ Dataset split into 'train' and 'val' folders.")
