import json
import asyncio
import aiohttp
from channels.generic.websocket import AsyncWebsocketConsumer
from .views import process_frame

GLOBAL_INFERENCE_SEMAPHORE = asyncio.Semaphore(4)

class LandmarksConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session = aiohttp.ClientSession()
        await self.accept()
        print("WS connection accepted")

    async def disconnect(self, close_code):
        if self.session:
            await self.session.close()
            self.session = None
        print(f"WS connection closed: {close_code}")

    async def receive(self, text_data=None, bytes_data=None):
        try:
            data = json.loads(text_data)
            await self.handle_landmarks(data)
        except Exception as e:
            print(f"Error processing message: {e}")

    async def handle_landmarks(self, data):
        try:
            request_id = data["request_id"]
            landmarks = data["landmarks"]
            handedness = data["handedness"]

            if request_id is None or landmarks is None or handedness is None:
                print("Missing required fields in landmarks message")
                return

            if GLOBAL_INFERENCE_SEMAPHORE.locked():
                print(f"All inference slots busy, skipping request {request_id}")
                return
            
            asyncio.create_task(self.process_inference_request(request_id, landmarks, handedness))

        except Exception as e:
            print(f"Error handling landmarks: {e}")

    async def process_inference_request(self, request_id, landmarks, handedness):
        async with GLOBAL_INFERENCE_SEMAPHORE:
            try:
                result = await process_frame(self.session, landmarks, handedness)
                
                if result:
                    await self.send(text_data=json.dumps(result))
            except Exception as e:
                print(f"Error processing inference request {request_id}: {e}")
