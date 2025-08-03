import modal

# Define your custom Modal image and include all required dependencies.
image = (
    modal.Image.debian_slim()
    .pip_install("tensorflow", "pillow", "fastapi[standard]", "numpy")
    .add_local_file("cat_vs_dogs_model.keras", "/model/cat_vs_dogs_model.keras")
)

import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile

app = modal.App("cats-vs-dogs-inference")

# Load the model only once per container for efficiency.
MODEL_PATH = "/model/cat_vs_dogs_model.keras"
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def classify_image(file: UploadFile):
    try:
        image_bytes = file.file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((160, 160))
        arr = np.array(img) / 255.0
        arr = arr[np.newaxis, ...]
        model = get_model()
        pred = model.predict(arr)
        return {
            "class": "dog" if pred[0][0] > 0.5 else "cat",
            "confidence": float(pred[0][0])
        }
    except Exception as e:
        return {"error": str(e)}


    
