import os
import shutil
import random


def create_dataset_structure(base_path="dataset"):
    splits = ["train", "val"]
    classes = ["human", "animal", "none"]

    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_path, split, cls)
            os.makedirs(dir_path, exist_ok=True)

    print(f"Dataset structure created under '{base_path}'")

def copy_human_images():
    source_root = "/Volumes/main/Capstone/mobilenetv2_finetune/archive/lfw-deepfunneled/lfw-deepfunneled"
    train_dir = "/Volumes/main/Capstone/mobilenetv2_finetune/dataset/train/human"
    val_dir = "/Volumes/main/Capstone/mobilenetv2_finetune/dataset/val/human"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_images = []
    for person_folder in os.listdir(source_root):
        person_path = os.path.join(source_root, person_folder)
        if os.path.isdir(person_path):
            for file in os.listdir(person_path):
                if file.lower().endswith(".jpg"):
                    src = os.path.join(person_path, file)
                    prefixed_name = f"{person_folder}_{file}"
                    all_images.append((src, prefixed_name))

    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    for src, name in train_images:
        shutil.copy(src, os.path.join(train_dir, name))

    for src, name in val_images:
        shutil.copy(src, os.path.join(val_dir, name))

    print(f"Copied {len(train_images)} to train and {len(val_images)} to val for human class.")


def copy_animal_images():

    train_src = "/Volumes/main/Capstone/mobilenetv2_finetune/afhq/train"
    val_src = "/Volumes/main/Capstone/mobilenetv2_finetune/afhq/val"

    train_dst = "/Volumes/main/Capstone/mobilenetv2_finetune/dataset/train/animal"
    val_dst = "/Volumes/main/Capstone/mobilenetv2_finetune/dataset/val/animal"

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    def copy_images(src_root, dst_root):
        for subfolder in os.listdir(src_root):
            full_subfolder = os.path.join(src_root, subfolder)
            if os.path.isdir(full_subfolder):
                for file in os.listdir(full_subfolder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_file = os.path.join(full_subfolder, file)
                        dst_file = os.path.join(dst_root, f"{subfolder}_{file}")
                        shutil.copy(src_file, dst_file)

    copy_images(train_src, train_dst)
    copy_images(val_src, val_dst)

    print("Animal images copied to dataset/train/animal and dataset/val/animal")

def copy_none_images():
    src_root = "/Volumes/main/Capstone/mobilenetv2_finetune/House_Room_Dataset"
    train_dst = "/Volumes/main/Capstone/mobilenetv2_finetune/dataset/train/none"
    val_dst = "/Volumes/main/Capstone/mobilenetv2_finetune/dataset/val/none"

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    image_paths = []
    for room_type in os.listdir(src_root):
        room_path = os.path.join(src_root, room_type)
        if os.path.isdir(room_path):
            for file in os.listdir(room_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(room_path, file)
                    image_paths.append((full_path, room_type)) 
    # Shuffle and split 80/20
    random.shuffle(image_paths)
    split_idx = int(0.8 * len(image_paths))
    train_images = image_paths[:split_idx]
    val_images = image_paths[split_idx:]

    for src, prefix in train_images:
        dst_file = os.path.join(train_dst, f"{prefix}_{os.path.basename(src)}")
        shutil.copy(src, dst_file)

    for src, prefix in val_images:
        dst_file = os.path.join(val_dst, f"{prefix}_{os.path.basename(src)}")
        shutil.copy(src, dst_file)

    print("'none' class images copied to dataset/train/none and dataset/val/none")

if __name__ == "__main__":
    create_dataset_structure()
    choice = input("Copy human images (h) or animal images (a) or none images (n) or all (al)? ").strip().lower()
    if choice == 'a':
        copy_animal_images()
    elif choice == 'h':
        copy_human_images()
    elif choice == 'n':
        copy_none_images()
    elif choice == 'al':
        copy_human_images()
        copy_animal_images()
        copy_none_images()
    