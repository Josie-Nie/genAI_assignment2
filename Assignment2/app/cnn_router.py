"""Extend the existing API with a CIFAR-10 image classifier."""
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model import AssignmentCNN, CLASSES
from app.preprocess import image_transform

router = APIRouter(tags=["CNN classifier"])
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def model_path():
    return Path(os.getenv("CNN_MODEL_PATH", "models/cifar10_cnn.pth"))


@lru_cache(maxsize=1)
def load_model():
    checkpoint = model_path()
    if not checkpoint.is_file():
        raise HTTPException(status_code=503, detail="Trained CNN weights missing. Run python train.py first.")
    model = AssignmentCNN()
    try:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Invalid or incompatible CNN checkpoint.") from exc
    model.eval()
    return model


@router.get("/cnn-health")
def cnn_health():
    return {"weights_available": model_path().is_file(), "classes": list(CLASSES)}


@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPEG, or WebP image.")
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB.")
    try:
        with Image.open(BytesIO(payload)) as raw:
            image = raw.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Image cannot be decoded.") from exc

    model = load_model()
    batch = image_transform(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)[0]
        predicted = int(torch.argmax(probabilities).item())
    return {
        "class_id": predicted,
        "class_name": CLASSES[predicted],
        "confidence": round(float(probabilities[predicted]), 6),
    }
