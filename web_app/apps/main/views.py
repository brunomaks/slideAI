import base64
import aiohttp
import cv2
import numpy as np
import os
from django.http import JsonResponse
from apps.core.utils import encode_jpg, decode_jpg

INFERENCE_URL = os.getenv('INFERENCE_URL')
RESIZE_URL = os.getenv('RESIZE_URL')
DEBUG_SAVE = os.getenv('DEBUG_SAVE', 'false').lower() == 'true'


def is_frame_empty(img, pixel_threshold=10, ratio=0.99):
    """
    Consider a frame empty if > ratio of pixels have intensity below pixel_threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    low_pixels = np.sum(gray < pixel_threshold)
    return (low_pixels / gray.size) > ratio


async def process_frame(session: aiohttp.ClientSession, frame_b64: str):
    try:
        jpg_bytes = base64.b64decode(frame_b64)
        img = decode_jpg(jpg_bytes)

        if img is None:
            print("Failed to decode frame")
            return None

        if is_frame_empty(img):
            print("Skipping empty frame")
            return {
                "type": "frame",
                "frame": frame_b64,
                "prediction": {
                    "predicted_class": "empty",
                    "confidence": 100.0
                }
            }

        headers = {'X-Debug-Save': '1'} if DEBUG_SAVE else {}

        async with session.post(RESIZE_URL, data=jpg_bytes, headers=headers) as resp:
            if resp.status != 200:
                print(f"Resize service error: {resp.status}")
                return None
            resized_jpg_bytes = await resp.read()

        async with session.post(INFERENCE_URL, data=resized_jpg_bytes, headers=headers) as resp:
            if resp.status != 200:
                print(f"Inference service error: {resp.status}")
                return None
            prediction = await resp.json()

        processed_b64 = base64.b64encode(resized_jpg_bytes).decode('utf-8')

        return {
            "type": "frame",
            "frame": processed_b64,
            "prediction": prediction
        }

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return None

