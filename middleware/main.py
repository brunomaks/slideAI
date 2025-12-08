from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import Response
import os
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional
from pathlib import Path

app = FastAPI()

MARGIN = 20  # px
MEDIA_ROOT = "../web_app/media"

# ensure dir exists
os.makedirs(MEDIA_ROOT, exist_ok=True)

# STEP 0: Resolve detector path and initialize detector
detector_path = Path('../shared_artifacts/models') / 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=detector_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

@app.post("/crop")
async def crop_view(request: Request):
    """
    Process an image to detect and crop a single hand.
    """
    print("Crop service received a request")
    
    body = await request.body()

    print("Received a body")
    nparr = np.frombuffer(body, np.uint8)

    print("Converted to np arayy")
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("Decoded the image")
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    cropped = process_image(img)

    print("Cropped the image")
    
    if cropped is None:
        raise HTTPException(status_code=400, detail="Could not process image (no hand, multiple hands, or invalid crop)")
    
    _, buffer = cv2.imencode('.jpg', cropped)
    return Response(content=buffer.tobytes(), media_type='image/jpeg')


def process_image(img):
    """
    Process image to detect hand landmarks and crop to hand bounding box.
    
    Returns:
        Cropped image in RGB format, or None if processing failed
    """
    # STEP 1: Load the input image
    image = mp.Image(mp.ImageFormat.SRGB, np.asarray(img))
    
    # STEP 2: Detect hand landmarks from the input image
    detection_result = detector.detect(image)
    
    # STEP 3: Process and crop image
    # slice to get rid of alpha channel if present
    annotated_image = np.copy(image.numpy_view()[:, :, :3])
    hand_landmarks_list = detection_result.hand_landmarks
    
    if not hand_landmarks_list:
        print("No hands detected in the image. Skipping.")
        return None
    
    if len(hand_landmarks_list) > 1:
        print("Multiple hands detected in the image. Skipping.")
        return None
    
    # Extract the first and only detected hand
    hand_landmarks = hand_landmarks_list[0]
    
    # Get the detected hand's bounding box
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    
    box_x_min = int(min(x_coordinates) * width) - MARGIN
    box_y_min = int(min(y_coordinates) * height) - MARGIN
    box_x_max = int(max(x_coordinates) * width) + MARGIN
    box_y_max = int(max(y_coordinates) * height) + MARGIN
    
    # Ensure the bounding box is within the image dimensions
    box_x_min = max(box_x_min, 0)
    box_y_min = max(box_y_min, 0)
    box_x_max = min(box_x_max, width)
    box_y_max = min(box_y_max, height)
    
    if box_x_min >= box_x_max or box_y_min >= box_y_max:
        print("Invalid bounding box dimensions. Skipping this image.")
        return None
    
    # Crop the image by the bounding box
    cropped_img = annotated_image[box_y_min:box_y_max, box_x_min:box_x_max]
    
    if cropped_img.size == 0:
        print("Empty crop, skipping...")
        return None
    
    # Convert back to RGB
    to_save_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return to_save_rgb


@app.get("/health")
async def root():
    return {"status": "ok", "service": "hand detection"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)