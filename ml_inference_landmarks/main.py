from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import keras
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import cv2
import os
import time

CLASSES = ["like", "stop", "two_up"]

# ENSURE THE CORRECT MODEL NAME EXISTS IN shared_artifacts/models
base_path = os.getenv('MODEL_PATH', '')
MODEL_PATH = Path(base_path) / "gesture_model_20251221_184630.keras"


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = keras.models.load_model(path)
    print(f"Model loaded successfully from {path}")
    print("Input shape:", model.input_shape)
    return model


def predict(model, landmarks: np.ndarray) -> dict:
    # landmarks shape should be (21,2)
    landmarks = landmarks.reshape(1, -1)

    prediction = model.predict(landmarks, verbose=0)
    predicted_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return {
        "predicted_class": CLASSES[predicted_idx],
        "confidence": confidence,
        "timestamp": time.time()
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ML model...")
    try:
        app.state.model = load_model(MODEL_PATH)
    except Exception as e:
        print("ERROR loading model:", e)
        raise

    yield

    print("Shutting down... unloading model")
    app.state.model = None


app = FastAPI(lifespan=lifespan)


@app.post("/inference")
async def inference(request: Request):
    json_data = await request.json()
    
    # Validate and extract landmarks
    landmarks_list = json_data.get("landmarks")
    if not landmarks_list or not isinstance(landmarks_list, list) or len(landmarks_list) != 21:
        raise HTTPException(status_code=400, detail="Invalid or missing 'landmarks' in request")

    try:
        # Extract x,y coords into np array of shape (21, 2)
        landmarks = np.array([[pt["x"], pt["y"]] for pt in landmarks_list], dtype=np.float32)
    except (Exception) as e:
        raise HTTPException(status_code=400, detail=f"Malformed landmarks data: {e}")

    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        result = predict(app.state.model, landmarks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return JSONResponse(content=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-inference-landmarks"}
