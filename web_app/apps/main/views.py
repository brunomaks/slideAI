import aiohttp
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# receive rtc track from the web app
import cv2
import asyncio
import json
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceCandidate
from aiortc.contrib.media import MediaRecorder
from av import VideoFrame

import requests
from dotenv import load_dotenv
import os
import re
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

data_channel = None

@csrf_exempt
async def main_view(request):
    """Handle WebRTC offer from client"""
    params = json.loads(request.body)
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)

    pcs.add(pc)


    @pc.on("datachannel")
    def on_datachannel(channel):
        print("DataChannel received:", channel.label)
        global data_channel
        data_channel = channel

        @channel.on("message")
        def on_message(message):
            print("Received message:", message)



    # load environment variables
    load_dotenv()
    CROP_URL = os.getenv('CROP_URL')
    FLIP_URL = os.getenv('FLIP_URL')
    HOST_URL = os.getenv('HOST_URL')
    DEBUG_SAVE = os.getenv('DEBUG_SAVE', 'false').lower() == 'true'
    
    return_track = ProcessedVideoTrack()
    pc.addTrack(return_track)

    print("Configuration as follows:")
    print(f"CROP_URL: {CROP_URL}")
    print(f"FLIP_URL: {FLIP_URL}")
    print(f"HOST_URL: {HOST_URL}")
    print(f"DEBUG_SAVE: {DEBUG_SAVE}")
    
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

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
                        if frame:
                            print("Frame received")
                            asyncio.create_task(process_video_frame(frame, session, return_track, CROP_URL, FLIP_URL, DEBUG_SAVE))
                            print("Frame processing task created")

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
            
    await pc.setRemoteDescription(offer)

    print(f'Number of Candidates: {len(params["candidates"])}')

    for candidate in params["candidates"]:
        parsed = parse_client_candidate(candidate)
        if parsed is not None:
            await pc.addIceCandidate(parsed)
    
    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    server_candidates = []
    @pc.on("icecandidate")
    def on_candidate(candidate):
        if candidate:
            server_candidates.append(candidate.toJSON())
    
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)

    return JsonResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "candidates": server_candidates
    })

def parse_client_candidate(candidate_dict):
    """
    Parse client candidate dictionary into aiortc.RTCIceCandidate parameters
    
    Input format:
    {
        'candidate': 'candidate:0 1 UDP 2122055935 192.168.10.60 48393 typ host',
        'sdpMLineIndex': 0,
        'sdpMid': '0',
        'usernameFragment': '9c9437c2'
    }
    """
    candidate_str = candidate_dict['candidate']

    # Parse the candidate string using regex
    # Format: candidate:foundation component transport priority ip port typ type
    pattern = r'candidate:(\S+) (\d+) (\S+) (\d+) (\S+) (\d+) typ (\S+)'
    match = re.match(pattern, candidate_str)
    
    if not match:
        # raise ValueError(f"Invalid candidate format: {candidate_str}")
        print(f"Invalid candidate format: {candidate_str}")
        return None

    foundation, component, transport, priority, ip, port, cand_type = match.groups()
    
    # Convert types
    component = int(component)
    priority = int(priority)
    port = int(port)
    
    candidate = RTCIceCandidate(
        component=component,
        foundation=foundation,
        ip=ip,
        port=port,
        priority=priority,
        protocol=transport,  # Note: parameter name is 'protocol' not 'transport'
        type=cand_type,
        sdpMid=candidate_dict['sdpMid'],
        sdpMLineIndex=candidate_dict['sdpMLineIndex'])

    print("Candidate object was created")
    
    return candidate

async def process_video_frame(frame, session, return_track, CROP_URL, FLIP_URL, DEBUG_SAVE):
    img = frame.to_ndarray(format="bgr24")

    jpgImg = cv2.imencode('.jpg', img)[1].tobytes()

    headers = {}
    if DEBUG_SAVE:
        headers['X-Debug-Save'] = '1'

    # resizing
    async with session.post(FLIP_URL, data=jpgImg, headers=headers) as flip_response:
        flippedJpg = await flip_response.read()

    # inference
    async with session.post(CROP_URL, data=flippedJpg, headers=headers) as crop_response:
        prediction = await crop_response.json()

    if data_channel and data_channel.readyState == "open":
        data_channel.send(str(prediction))
        print("Sent to channel: ", str(prediction))
    else:
        print("Channel not ready")

    await return_track.add_frame(flippedJpg)
