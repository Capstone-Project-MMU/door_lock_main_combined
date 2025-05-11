import os

BASE_PATH = "dataset"
SPLITS = ["train", "val"]
CLASSES = ["monkey", "cat", "squirrel", "snake"]

def count_images_in_folder(path):
    return len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

for split in SPLITS:
    print(f"\n{split.upper()} SET:")
    for cls in CLASSES:
        dir_path = os.path.join(BASE_PATH, split, cls)
        count = count_images_in_folder(dir_path) if os.path.exists(dir_path) else 0
        print(f"  {cls}: {count}")
