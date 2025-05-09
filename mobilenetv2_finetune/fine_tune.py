import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing import image_dataset_from_directory
import os
from PIL import Image


def clean_invalid_images(dataset_path):
    removed_files = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        img.verify()  # validate the image
                except Exception as e:
                    print(f"Removing invalid image: {path} ({e})")
                    os.remove(path)
                    removed_files += 1
    print(f"Done. Removed {removed_files} invalid images.")

clean_invalid_images("/Volumes/main/Capstone/mobilenetv2_finetune/dataset")

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
DATA_DIR = "dataset"
NUM_CLASSES = 3

train_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)
val_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
print("Class labels:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

initial_epochs = 2
model.fit(train_ds, validation_data=val_ds, epochs=initial_epochs)

base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 3
model.fit(train_ds, validation_data=val_ds,
          epochs=fine_tune_epochs + initial_epochs,
          initial_epoch=initial_epochs)

model.save("mobilenetv2_human_animal_none.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("mobilenetv2_model.tflite", "wb") as f:
    f.write(tflite_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

with open("mobilenetv2_model_quant.tflite", "wb") as f:
    f.write(quantized_model)
