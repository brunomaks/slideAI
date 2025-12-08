from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from tensorflow import keras
from pathlib import Path
from contextlib import asynccontextmanager


MARGIN = 20  # px
_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    """
    Lifespan event handler - runs on startup and shutdown
    """
    print("Loading ML model...")
    try:
        model_path = Path('../shared_artifacts/models') / 'gesture_model_20251206_201021.keras'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        _model = keras.models.load_model(model_path)
        print(_model.input_shape)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise
    
    yield  # App runs here
    
    # Shutdown: Clean up
    print("Shutting down... Unloading model")
    _model = None

app = FastAPI(lifespan=lifespan)

@app.post("/inference")
async def inference(request: Request):
    """
    Process an image to detect and crop a single hand.
    """
    
    body = await request.body()

    nparr = np.frombuffer(body, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("IMAGE SHAPE:")
    print(img.shape)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    prediction = process_image(img)

    
    if prediction is None:
        raise HTTPException(status_code=400, detail="Could not make a prediction")
    
    return JSONResponse(content=prediction)


def process_image(img):
    """
    Process image to detect hand landmarks and crop to hand bounding box.
    
    Returns:
        Cropped image in RGB format, or None if processing failed
    """
    # STEP 1: Load the input image

    img_batch = np.expand_dims(img, axis=0)

    prediction = _model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return {
        predicted_class: predicted_class,
        confidence: confidence
    }

@app.get("/health")
async def root():
    return {"status": "ok", "service": "hand detection"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)