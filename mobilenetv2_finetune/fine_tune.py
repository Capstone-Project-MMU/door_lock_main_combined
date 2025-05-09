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

# Config
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 13

# Directories
train_dir = "dataset/train"
val_dir = "dataset/val"
balanced_train_dir = "balanced_data/train"
os.makedirs(balanced_train_dir, exist_ok=True)

# Balance training data
def balance_dataset(original_path, target_path):
    categories = ["human", "animal", "none"]
    counts = {}

    for cls in categories:
        src = os.path.join(original_path, cls)
        dst = os.path.join(target_path, cls)
        os.makedirs(dst, exist_ok=True)
        counts[cls] = len(os.listdir(src))

    target_size = counts["human"]

    for cls in categories:
        src = os.path.join(original_path, cls)
        dst = os.path.join(target_path, cls)
        files = os.listdir(src)

        if cls == "animal":
            files = random.sample(files, target_size)
        for f in files:
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))

balance_dataset(train_dir, balanced_train_dir)

# Load datasets
train_ds_raw = image_dataset_from_directory(
    balanced_train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds_raw = image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Extract class names BEFORE caching
class_labels = val_ds_raw.class_names

# Cache and prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)

# Model
base_model = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=IMG_SIZE + (3,))
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    layers.Rescaling(1./255),
    base_model,
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
]

# Train
with tf.device('/GPU:0'):
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# Evaluation
y_true = np.concatenate([y.numpy() for _, y in val_ds_raw])
y_true = np.argmax(y_true, axis=1)

y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print classification results
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Plot training curves
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Save final model
model.save("mobilenetv2_human_animal_none_balanced.h5")
