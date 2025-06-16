import os
import shutil
import random

def split_data(source_dir, val_dir, val_split=0.2):
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    classes = os.listdir(source_dir)
    for cls in classes:
        src_path = os.path.join(source_dir, cls)
        dst_path = os.path.join(val_dir, cls)

        if not os.path.isdir(src_path):
            continue

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        files = os.listdir(src_path)
        val_size = int(len(files) * val_split)
        val_files = random.sample(files, val_size)

        for f in val_files:
            shutil.move(os.path.join(src_path, f), os.path.join(dst_path, f))

    print("✅ Validation split complete.")

if __name__ == "__main__":
    split_data("dataset/train", "dataset/val", val_split=0.2)
