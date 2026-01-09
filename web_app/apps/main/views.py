# Contributors:
# - Pavlo
# - Ahmet

import aiohttp
import os
from asgiref.sync import sync_to_async

INFERENCE_URL = os.getenv('INFERENCE_URL')


async def process_frame(session, landmarks, handedness):
    try:
        payload = {
            "landmarks": landmarks,
            "handedness": handedness
        }
        
        async with session.post(INFERENCE_URL, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result
            else:
                print(f"Inference service returned status {resp.status}")
                return None
    except Exception as e:
        print(f"Error calling inference service: {e}")
        return None

@sync_to_async
def log_prediction(result, request_id, landmarks, handedness):
    from apps.core.models import Prediction
    from apps.core.models import ModelVersion


    try:
        # Prediction result details
        predicted_class = result['predicted_class']
        confidence = result['confidence'] * 100
        inference_time_ms = result['timestamp']
        direction = result['direction']
        
        # Query to get the current active version
        active_version = ModelVersion.objects.filter(is_active=True).first()


        return Prediction.objects.create(
            request_id=request_id,
            predicted_class=predicted_class,
            model_version=active_version,
            confidence=confidence,
            landmarks=landmarks,
            handedness=handedness,
            inference_time_ms=inference_time_ms,
            direction=direction,
        )

    except Exception as e:
        print(f"Unexpected error while logging prediction: {e}")
        return None
