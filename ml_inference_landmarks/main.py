from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import keras
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import json
import os
import time

ACTIVE_MODEL_PATH = os.getenv('ACTIVE_MODEL_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
MODEL_DIR = os.getenv('MODEL_PATH')

def load_active_model_info():
    active_json_path = Path(ACTIVE_MODEL_PATH)
    
    if not active_json_path.exists():
        raise FileNotFoundError(f"Active model descriptor not found at {active_json_path}")
    
    with open(active_json_path, 'r') as f:
        active_data = json.load(f)
    
    model_file = active_data.get("model_file")
    class_names = active_data.get("class_names")
    
    if not model_file:
        raise ValueError("Model file not specified in active_model.json")
    
    model_path = Path(MODEL_DIR) / model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist")
    
    return model_path, class_names


MODEL_PATH, CLASSES = load_active_model_info()


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = keras.models.load_model(path)
    print(f"Model loaded successfully from {path}")
    print("Input shape:", model.input_shape)
    return model


def predict(model, landmarks: np.ndarray) -> dict:
    # landmarks shape should be (21,2)

    input_vector = np.array(landmarks, dtype=np.float32).flatten() # (42,)

    input_vector = np.expand_dims(input_vector, axis=0) # (1, 42)

    prediction = model.predict(input_vector, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_idx]

    predicted_gesture = CLASSES[predicted_idx]

    return {
        "predicted_class": predicted_gesture,
        "confidence": float(confidence),
        "timestamp": time.time()
    }

def normalize_landmarks(landmarks, handedness):
    landmarks = np.array(landmarks)

    # Translate so that wrist is at origin
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale so that distance between wrist and middle finger MCP is 1
    mcp_index = 9  # Middle finger MCP landmark index
    scale = np.linalg.norm(landmarks[mcp_index]) # euclidean distance from the origin (wrist)
    if scale > 0:
        landmarks = landmarks / scale
    
    # Mirror left hands
    if handedness == "Left":
        landmarks[:, 0]  =  -landmarks[:, 0]

    rotated_landmarks = normalize_rotation(landmarks)

    return rotated_landmarks

def normalize_rotation(landmarks):
    # Reference vector: from wrist (now at origin) to middle finger MCP
    reference_vector = landmarks[9]  # Middle finger MCP (wrist is at origin)
    
    # Current angle of reference vector
    current_angle = np.arctan2(reference_vector[1], reference_vector[0])
    
    # Target angle (pointing up in image coordinates = -90 degrees = -pi/2)
    # Note: In image coordinates, Y increases downward, so "up" is negative Y
    target_angle = -np.pi / 2
    
    # Calculate rotation needed
    rotation_angle = target_angle - current_angle
    
    # Apply rotation
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated_landmarks = (rotation_matrix @ landmarks.T).T
    
    return rotated_landmarks

def compute_direction(landmark_list):
    wrist = np.array(landmark_list[0])
    index_mcp = np.array(landmark_list[5])
    middle_mcp = np.array(landmark_list[9])
    index_tip = np.array(landmark_list[8])
    middle_tip = np.array(landmark_list[12])

    palm_center = (wrist + index_mcp + middle_mcp) / 3
    finger_tip_avg = (index_tip + middle_tip) / 2
    finger_dir = finger_tip_avg - palm_center

    angle = np.arctan2(finger_dir[1], finger_dir[0])  # radians, range [-pi, pi]
    return angle

def retrieve_direction(angle):
    # Normalize angle to [-pi, pi]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    
    # Define thresholds in radians
    right_thresh = np.pi / 4          # 45 degrees
    left_thresh = 3 * np.pi / 4       # 135 degrees
    
    if -right_thresh <= angle <= right_thresh:
        return "Right"
    elif angle >= left_thresh or angle <= -left_thresh:
        return "Left"
    elif right_thresh < angle < left_thresh:
        return "Down"
    else:
        return "Up"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading ML model at {MODEL_PATH}")
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
    
    handedness = json_data.get("handedness")

    try:
        # Extract x,y coords into np array of shape (21, 2)
        landmarks = np.array([[pt["x"], pt["y"]] for pt in landmarks_list], dtype=np.float32)
    except (Exception) as e:
        raise HTTPException(status_code=400, detail=f"Malformed landmarks data: {e}")

    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # compute direction based on original landmarks
        angle = compute_direction(landmarks)
        direction = retrieve_direction(angle)

        normalized = normalize_landmarks(landmarks, handedness)

        result = predict(app.state.model, normalized)
        result["direction"] = direction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return JSONResponse(content=result)


@app.post("/reload")
async def reload_model():
    print("Reloading model...")
    try:
        global MODEL_PATH, CLASSES
        MODEL_PATH, CLASSES = load_active_model_info()
        app.state.model = load_model(MODEL_PATH)
        print(f"Model reloaded. Classes: {CLASSES}")
        return {"status": "reloaded", "classes": CLASSES}
    except Exception as e:
        print(f"Reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-inference-landmarks"}