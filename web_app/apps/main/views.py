import aiohttp
import os

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
