import os
import random
import shutil
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks
from keras.applications import MobileNetV2
from keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from keras.models import load_model


# Configuration
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 100

# Directories
# train_dir = "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/dataset/train"
# val_dir = "/Volumes/main/Capstone/Clean/models_training/mobilenetv2_finetune_animal/dataset/val"
# balanced_train_dir = "balanced_data/train"

train_dir = "/home/ahmed/animal_train"
val_dir = "/home/ahmed/animal_train"
balanced_train_dir = train_dir
os.makedirs(balanced_train_dir, exist_ok=True)

# Remove Non-Image Files
def remove_non_image_files(root_dir):
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    deleted_files = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(dirpath, file)

            if ext not in allowed_extensions:
                try:
                    os.remove(file_path)
                    deleted_files += 1
                    print(f"[EXT] Deleted non-image: {file_path}")
                except Exception as e:
                    print(f"[EXT] Failed to delete {file_path}: {e}")
                continue

            # Try to open file as an image
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check for corrupt file
            except (UnidentifiedImageError, OSError, ValueError):
                try:
                    os.remove(file_path)
                    deleted_files += 1
                    print(f"[CORRUPT] Deleted invalid image: {file_path}")
                except Exception as e:
                    print(f"[CORRUPT] Failed to delete {file_path}: {e}")

    print(f"Final cleanup done: {deleted_files} bad file(s) removed from '{root_dir}'")


# Clean source train/val directories first
# for directory in [train_dir, val_dir]:
#     remove_non_image_files(directory)

# Balance Training Data
def balance_dataset(original_path, target_path):
    categories = [
        d for d in os.listdir(original_path)
        if os.path.isdir(os.path.join(original_path, d))
    ]
    counts = {}

    for cls in categories:
        src = os.path.join(original_path, cls)
        dst = os.path.join(target_path, cls)
        os.makedirs(dst, exist_ok=True)
        remove_non_image_files(src)
        counts[cls] = len(os.listdir(src))

    target_size = counts["monkey"]  # use monkey as reference

    for cls in categories:
        src = os.path.join(original_path, cls)
        dst = os.path.join(target_path, cls)
        files = os.listdir(src)

        if counts[cls] > target_size:
            files = random.sample(files, target_size)  # downsample

        for f in files:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))

    print(f"Balanced data copied to '{target_path}'")

# Balance the dataset
# balance_dataset(train_dir, balanced_train_dir)

# Clean the balanced directory after copying
# remove_non_image_files(balanced_train_dir)

# Load Datasets
# train_ds_raw = image_dataset_from_directory(
#     balanced_train_dir,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode='categorical'
# )

# val_ds_raw = image_dataset_from_directory(
#     val_dir,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     label_mode='categorical',
#     shuffle=False
# )

train_ds_raw = image_dataset_from_directory(
    balanced_train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds_raw = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    shuffle=False
)

# Extract class names
class_labels = val_ds_raw.class_names

# Cache and prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)

# Build Model
base_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=IMG_SIZE + (3,))
base_model.trainable = True

model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1./255),
    base_model,
    layers.Dense(1, activation='sigmoid')  # ✅ Right for binary
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model = models.Sequential([
#     layers.Input(shape=IMG_SIZE + (3,)),
#     layers.Rescaling(1./255),
#     base_model,
#     layers.Dense(1, activation='sigmoid')  # Binary classifier
# ])
# model = load_model("mobilenetv2_animal.h5")
# model.trainable = True
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Callbacks
cb = [
    # callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint("cat_best_model.h5", save_best_only=True)
]

# ----------------------
# Train Model
# ----------------------
with tf.device('/GPU:0'):
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# Evaluate
# y_true = np.concatenate([y.numpy() for _, y in val_ds_raw])
# y_true = np.argmax(y_true, axis=1)

# y_pred_probs = model.predict(val_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y.numpy() for _, y in val_ds_raw])
y_pred_probs = model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
# Classification report
print(confusion_matrix(y_true, y_pred))
# print(classification_report(y_true, y_pred, target_names=class_labels))
print(classification_report(y_true, y_pred))


# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Training Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Save Final Model
model.save("mobilenetv2_animal_cat.h5")
