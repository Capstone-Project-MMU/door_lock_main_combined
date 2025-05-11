import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "mobilenetv2_quantized.tflite"
IMG_PATH = "image.png"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["animal", "human", "none"]

def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image, dtype=np.float32) / 255.0  # Match training normalization
    image = np.expand_dims(image, axis=0)
    return image

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess input image
input_data = load_and_preprocess_image(IMG_PATH)

# Cast if needed
if input_details[0]['dtype'] == np.uint8:
    input_scale, input_zero_point = input_details[0]['quantization']
    input_data = input_data / input_scale + input_zero_point
    input_data = input_data.astype(np.uint8)

# Set tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output and process
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Dequantize output if necessary
if output_details[0]['dtype'] == np.uint8:
    output_scale, output_zero_point = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

predicted_class = np.argmax(output_data)
confidence = np.max(output_data)

print(f"Prediction: {CLASS_NAMES[predicted_class]} (confidence: {confidence:.2f})")
