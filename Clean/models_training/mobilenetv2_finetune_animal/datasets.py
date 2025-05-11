import os
import shutil
import random

def create_dataset_structure(base_path="dataset"):
    splits = ["train", "val"]
    classes = ["monkey", "cat", "squirrel", "snake"]

    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_path, split, cls)
            os.makedirs(dir_path, exist_ok=True)

    print(f"Dataset structure created under '{base_path}'")


def copy_images_for_class(
    class_name,
    structure,  # "flat" or "nested"
    train_src,
    val_src,
    dst_base="/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/dataset"
):
    train_dst = os.path.join(dst_base, "train", class_name)
    val_dst = os.path.join(dst_base, "val", class_name)

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)

    def copy_flat(src, dst):
        all_images = [
            os.path.join(src, f) for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(all_images)
        split_idx = int(0.8 * len(all_images))
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        for src_path in train_images:
            shutil.copy(src_path, os.path.join(train_dst, os.path.basename(src_path)))

        for src_path in val_images:
            shutil.copy(src_path, os.path.join(val_dst, os.path.basename(src_path)))

        print(f"Copied {len(train_images)} to train and {len(val_images)} to val for '{class_name}' (flat structure)")

    def copy_nested(src_root, dst_root):
        for subfolder in os.listdir(src_root):
            subfolder_path = os.path.join(src_root, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(subfolder_path, file)
                    new_name = f"{subfolder}_{file}"
                    dst_file = os.path.join(dst_root, new_name)
                    shutil.copy2(src_file, dst_file)

    if structure == "flat":
        copy_flat(train_src, train_dst)
        copy_flat(val_src, val_dst)
    elif structure == "nested":
        copy_nested(train_src, train_dst)
        copy_nested(val_src, val_dst)
        print(f"Copied images for '{class_name}' (nested structure)")
    else:
        raise ValueError(f"Unknown structure type: {structure}")


if __name__ == "__main__":
    create_dataset_structure()

    # Class config: structure type and source paths
    config = {
        "cat": {
            "structure": "flat",
            "train_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/cats/Data",
            "val_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/cats/Data"
        },
        "monkey": {
            "structure": "nested",
            "train_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/monkey/training/training",
            "val_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/monkey/validation/validation"
        },
        "squirrel": {
            "structure": "flat",
            "train_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/squirrel",
            "val_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/squirrel"
        },
        "snake": {
            "structure": "nested",
            "train_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/snake/train",
            "val_src": "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/snake/test"
        }
    }

    choice = input("Copy which images? (c=cat, m=monkey, q=squirrel, s=snake, a=all): ").strip().lower()

    alias_map = {"c": "cat", "m": "monkey", "q": "squirrel", "s": "snake"}

    if choice == "a":
        for cls, settings in config.items():
            copy_images_for_class(cls, settings["structure"], settings["train_src"], settings["val_src"])
    elif choice in alias_map:
        cls = alias_map[choice]
        settings = config[cls]
        copy_images_for_class(cls, settings["structure"], settings["train_src"], settings["val_src"])
    else:
        print("Invalid choice.")
