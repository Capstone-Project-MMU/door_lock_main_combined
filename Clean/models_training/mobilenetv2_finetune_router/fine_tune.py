import os
import random
import shutil
import numpy as np
import tensorflow as tf
from keras import layers, models, callbacks, regularizers
from keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Config
IMG_SIZE = (512, 512)
BATCH_SIZE = 32
EPOCHS = 25
CATEGORIES = ["animal", "human", "none"]
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
BALANCED_TRAIN_DIR = "balanced_data/train"


# Utility Functions
def balance_dataset(original_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    counts = {}
    for cls in CATEGORIES:
        cls_dir = os.path.join(original_path, cls)
        counts[cls] = len(os.listdir(cls_dir))
    min_count = min(counts.values())
    for cls in CATEGORIES:
        src_dir = os.path.join(original_path, cls)
        dst_dir = os.path.join(target_path, cls)
        os.makedirs(dst_dir, exist_ok=True)
        sampled_files = random.sample(os.listdir(src_dir), min_count)
        for f in sampled_files:
            src_path = os.path.join(src_dir, f)
            dst_path = os.path.join(dst_dir, f)

            # Resize with high-quality resampling
            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                img.save(dst_path)
            except Exception as e:
                print(f"Skipping corrupted image {src_path}: {e}")
    print("Dataset balanced and resized to:", IMG_SIZE, "with", min_count, "images per class")



def categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            cross_entropy *= alpha_tensor
        weight = tf.pow(1 - y_pred, gamma)
        focal_loss = weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return loss


def load_datasets():
    train_ds_raw = image_dataset_from_directory(
        BALANCED_TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        class_names=CATEGORIES
    )

    val_ds_raw = image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
        class_names=CATEGORIES
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, train_ds_raw, val_ds_raw


def compute_class_weights(train_ds_raw):
    labels = [np.argmax(y.numpy()) for _, y in train_ds_raw.unbatch()]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(CATEGORIES)), y=labels)
    return dict(enumerate(class_weights))


def get_model(name):
    if name == 'mobilenetv2':
        from keras.applications import MobileNetV2
        return MobileNetV2
    elif name == 'mobilenetv3':
        from keras.applications import MobileNetV3Large
        return MobileNetV3Large
    elif name == 'efficientnetb0':
        from keras.applications import EfficientNetB0
        return EfficientNetB0
    elif name == 'efficientnetv2b0':
        from keras.applications import EfficientNetV2B0
        return EfficientNetV2B0
    else:
        raise ValueError("Unknown model name")


def build_model(model_name):
    base_model_class = get_model(model_name)
    base_model = base_model_class(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SIZE + (3,),
        pooling='avg'
    )
    base_model.trainable = True
    for layer in base_model.layers[:200]:
        layer.trainable = False

    data_augmentation = tf.keras.Sequential([layers.RandomZoom(0.1)])

    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        data_augmentation,
        layers.Rescaling(1./255),
        base_model,
        layers.Dropout(0.3),
        layers.Dense(len(CATEGORIES), activation='softmax', kernel_regularizer=regularizers.l2(0.01))
    ])

    model.compile(
        optimizer='adam',
        loss=categorical_focal_loss(gamma=2.0, alpha=[1.0] * len(CATEGORIES)),
        metrics=['accuracy']
    )
    return model


def train_and_evaluate(model_name):
    balance_dataset(TRAIN_DIR, BALANCED_TRAIN_DIR)
    train_ds, val_ds, train_ds_raw, val_ds_raw = load_datasets()
    class_weights = compute_class_weights(train_ds_raw)

    model = build_model(model_name)
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(f"best_model_{model_name}.keras", save_best_only=True)
    ]

    with tf.device('/GPU:0'):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=cb,
            class_weight=class_weights
        )

    y_true = np.concatenate([y.numpy() for _, y in val_ds_raw])
    y_true = np.argmax(y_true, axis=1)
    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=CATEGORIES))

        # Save confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

    # Save training curves
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
    plt.savefig(f"{model_name}_training_curves.png")
    plt.close()
    print(f"Model {model_name} trained and evaluated successfully.")
if __name__ == "__main__":
    model_name = input("Which model you want to train? (mobilenetv2, mobilenetv3, efficientnetb0, efficientnetv2b0), all: ")
    if model_name == 'all':
        for name in ['mobilenetv2', 'mobilenetv3', 'efficientnetb0', 'efficientnetv2b0']:
            train_and_evaluate(name)
    else:
        train_and_evaluate(model_name)