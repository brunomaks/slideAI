import json
import asyncio
import aiohttp
from channels.generic.websocket import AsyncWebsocketConsumer
from .views import process_frame

class StreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session = aiohttp.ClientSession()
        self.inference_semaphore = asyncio.Semaphore(3)
        await self.accept()
        print("WS connection accepted")

    async def disconnect(self, close_code):
        if self.session:
            await self.session.close()
            self.session = None
        print(f"WS connection closed: {close_code}")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                data = json.loads(text_data)
                msg_type = data.get("type")

                if msg_type == "landmarks":
                    await self.handle_landmarks(data)
                elif msg_type == "ping":
                    await self.send(text_data=json.dumps({"type": "pong"}))
            except Exception as e:
                print(f"Error processing message: {e}")

    async def handle_landmarks(self, data):
        try:
            request_id = data.get("request_id")
            landmarks = data.get("landmarks")
            handedness = data.get("handedness")

            if request_id is None or landmarks is None or handedness is None:
                print("Missing required fields in landmarks message")
                return

            if self.inference_semaphore.locked() and self.inference_semaphore._value == 0:
                print(f"All 3 inference slots busy, skipping request {request_id}")
                return
            
            asyncio.create_task(self.process_inference_request(request_id, landmarks, handedness))

        except Exception as e:
            print(f"Error handling landmarks: {e}")

    async def process_inference_request(self, request_id, landmarks, handedness):
        async with self.inference_semaphore:
            try:
                result = await process_frame(self.session, landmarks, handedness)
                
                if result:
                    response = {
                        "type": "inference_result",
                        "request_id": request_id,
                        "result": result
                    }
                    await self.send(text_data=json.dumps(response))
            except Exception as e:
                print(f"Error processing inference request {request_id}: {e}")
