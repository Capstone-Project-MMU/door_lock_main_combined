import tensorflow as tf
import os

# Path to the trained Keras model
keras_model_path = "mobilenetv2_human_animal_none_balanced.h5"
tflite_model_path = "mobilenetv2_quantized.tflite"

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

# Convert the model to TensorFlow Lite with dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable default optimizations

# Convert and save
tflite_model = converter.convert()
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_model_path}")
