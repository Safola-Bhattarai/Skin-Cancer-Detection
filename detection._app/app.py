import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (128, 128)   # must match training (Section 6.1 in your notebook)

# Class names in the exact order your model was trained on
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Human-readable labels (optional, but nicer for the API response)
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevus (common mole)",
    "vasc":  "Vascular lesion",
}

model = load_model("mobilenetv2_model.keras")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32")
    img_array = preprocess_input(img_array)   # MobileNetV2-specific preprocessing (NOT /255)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


app = FastAPI(
    title="Skin Cancer Detection API",
    description="Upload a dermoscopic skin lesion image to classify it into one of 7 categories using a MobileNetV2 transfer-learning CNN.",
    version="1.0.0"
)


@app.post(
    "/predict",
    summary="Classify a skin lesion image",
    description="Upload a skin lesion image (JPG/PNG). Returns the predicted class, its confidence, and the full probability breakdown across all 7 classes.",
    response_description="Prediction result with confidence and per-class probabilities"
)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    pred_probs = model.predict(x)[0]              # shape: (7,)
    pred_idx = int(np.argmax(pred_probs))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(pred_probs[pred_idx])

    all_probs = {
        CLASS_NAMES[i]: round(float(pred_probs[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": pred_label,
        "description": CLASS_DESCRIPTIONS[pred_label],
        "confidence": round(confidence * 100, 2),
        "all_class_probabilities": all_probs
    }


@app.get("/")
async def root():
    return {"message": "Skin Cancer Detection API is running. POST an image to /predict"}