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
                        var = cv2.imencode('.jpg', img)[1].tobytes()
                        
                        async with session.post(
                            GRAYSCALE_URL,
                            data=var,
                            headers={"Content-Type": "image/jpeg"}
                        ) as grayscale_resp:
                            content = await grayscale_resp.read()
                            print(content)

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
