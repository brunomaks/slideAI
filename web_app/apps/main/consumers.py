import json
import aiohttp
from channels.generic.websocket import AsyncWebsocketConsumer
from .views import process_frame

class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session = aiohttp.ClientSession()
        self.frame_count = 0
        self.processing = False
        await self.accept()
        print("WS connection accepted")

    async def disconnect(self, close_code):
        if self.session:
            await self.session.close()
            self.session = None

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                data = json.loads(text_data)
                msg_type = data.get("type")

                if msg_type == "frame":
                    await self.handle_frame(data)
                elif msg_type == "ping":
                    await self.send(text_data=json.dumps({"type": "pong"}))
            except Exception as e:
                print(f"Error processing message: {e}")

    async def handle_frame(self, data):
        if self.processing:
            return

        self.processing = True
        try:
            frame_b64 = data.get("frame")
            if not frame_b64:
                return
            response = await process_frame(self.session, frame_b64)
            if response:
                await self.send(text_data=json.dumps(response))
            self.frame_count += 1

        except Exception as e:
            print(f"Error processing frame: {e}")
        finally:
            self.processing = False
