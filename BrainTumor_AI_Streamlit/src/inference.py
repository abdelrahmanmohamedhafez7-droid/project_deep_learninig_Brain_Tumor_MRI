from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "vgg16_model_final.keras"
IMG_SIZE = 128

@st.cache_resource(show_spinner=False)
def load_brain_tumor_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, image: Image.Image) -> dict:
    x = preprocess_image(image)
    probs = np.asarray(model.predict(x, verbose=0)).reshape(-1)

    if len(probs) != 2:
        raise ValueError(
            f"Expected a 2-class softmax output, but received shape {probs.shape}."
        )

    # The original notebook's LabelEncoder is fitted on ['no', 'yes'],
    # so index 0 = no tumor and index 1 = tumor.
    no_tumor_probability = float(probs[0])
    tumor_probability = float(probs[1])

    label = "Tumor Detected" if tumor_probability >= no_tumor_probability else "No Tumor Detected"

    return {
        "label": label,
        "tumor_probability": tumor_probability,
        "no_tumor_probability": no_tumor_probability,
        "model_name": "VGG16 Transfer Learning",
    }
