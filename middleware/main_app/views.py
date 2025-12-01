import aiohttp
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# receive rtc track from the web app
import cv2
import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder
from av import VideoFrame

import requests
from dotenv import load_dotenv
import os
import numpy as np

class ProcessedVideoTrack(VideoStreamTrack):

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(5)

    async def recv(self):
        frame = await self.queue.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame

    async def add_frame(self, jpg):
        nparr = np.frombuffer(jpg, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.queue.put(frame)

pcs = set()

@csrf_exempt
async def main_view(request):
    """Handle WebRTC offer from client"""
    params = json.loads(request.body)
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    # load environment variables
    load_dotenv()
    GRAYSCALE_URL = os.getenv('GRAYSCALE_URL')
    FLIP_URL = os.getenv('FLIP_URL')
    HOST_URL = os.getenv('HOST_URL')
    DEBUG_SAVE = os.getenv('DEBUG_SAVE', 'false').lower() == 'true'
    
    return_track = ProcessedVideoTrack()
    pc.addTrack(return_track)

    print("Configuration as follows:")
    print(f"GRAYSCALE_URL: {GRAYSCALE_URL}")
    print(f"FLIP_URL: {FLIP_URL}")
    print(f"HOST_URL: {HOST_URL}")
    print(f"DEBUG_SAVE: {DEBUG_SAVE}")

    @pc.on("track")
    async def on_track(track):
        print(f"Track received: {track.kind}")

        if track.kind == "video":
            print("Video track started - processing frames")

            async with aiohttp.ClientSession() as session:
                try:
                    while True:
                        print("Receiving frame")
                        frame = await track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        print("image was converted to ndarray")
                        jpgImg = cv2.imencode('.jpg', img)[1].tobytes()
                        
                        async with session.post(
                            GRAYSCALE_URL,
                            data=jpgImg,
                            headers={"Content-Type": "image/jpeg"} | ({"X-Debug-Save": "True"} if DEBUG_SAVE else {})
                        ) as grayscale_resp:
                            grayscaled_content = await grayscale_resp.read()

                        async with session.post(
                            FLIP_URL,
                            data=grayscaled_content,
                            headers={"Content-Type": "image/jpeg"} | ({"X-Debug-Save": "True"} if DEBUG_SAVE else {})
                        ) as flip_resp:
                            flipped_content = await flip_resp.read()
                            
                        await return_track.add_frame(flipped_content)
                                                
                except Exception as e:
                    print(f"Track ended or error: {e}")

            @track.on("ended")
            async def on_ended():
                print("Video track ended")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    # Set remote description and create answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JsonResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })
