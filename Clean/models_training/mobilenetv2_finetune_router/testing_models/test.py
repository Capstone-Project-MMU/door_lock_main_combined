from keras.models import load_model
from tensorflow import keras
import numpy as np
from PIL import Image
from keras.applications.mobilenet_v2 import preprocess_input


# focal loss function
def categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = keras.backend.clip(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * keras.backend.log(y_pred)
        if alpha is not None:
            alpha_tensor = keras.backend.constant(alpha, dtype=keras.backend.floatx())
            cross_entropy *= alpha_tensor
        weight = keras.backend.pow(1 - y_pred, gamma)
        focal_loss = weight * cross_entropy
        return keras.backend.sum(focal_loss, axis=-1)
    return loss


MODEL_PATH = "best_model_mobilenetv2.keras"
IMG_PATH = "image.png"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["animal", "human", "none"]

# Pass custom_objects explicitly
model = load_model(
    MODEL_PATH,
    custom_objects={'loss': categorical_focal_loss(gamma=2.0, alpha=[1.0, 1.0, 1.0])}
)


def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    processed_image = preprocess_input(image.copy())
    input_tensor = np.expand_dims(processed_image, axis=0)
    debug_image = ((processed_image + 1.0) * 127.5).astype(np.uint8)
    Image.fromarray(debug_image).save("debug_model_input.png")

    return input_tensor



input_data = load_and_preprocess_image(IMG_PATH)

predictions = model.predict(input_data)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Prediction: {CLASS_NAMES[predicted_class]} (confidence: {confidence:.2f})")
