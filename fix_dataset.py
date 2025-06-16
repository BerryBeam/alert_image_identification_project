import os
import shutil

def clean_empty_folders(path):
    for root, dirs, _ in os.walk(path, topdown=False):
        for d in dirs:
            full_path = os.path.join(root, d)
            if not os.listdir(full_path):
                print(f"Deleting empty folder: {full_path}")
                os.rmdir(full_path)

def main():
    base_dir = "dataset"
    val_path = os.path.join(base_dir, "val")

    # Delete unnecessary nested folders inside dataset/val/
    nested_paths = [os.path.join(val_path, "train"), os.path.join(val_path, "val")]
    for p in nested_paths:
        if os.path.exists(p):
            print(f"Removing: {p}")
            shutil.rmtree(p)

    # Clean any other empty folders
    clean_empty_folders(base_dir)
    print("✅ Dataset folder structure fixed.")

if __name__ == "__main__":
    main()
