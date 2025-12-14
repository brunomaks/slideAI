import aiohttp
import asyncio
import json
import os
import re
import cv2
import numpy as np

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, VideoStreamTrack
from av import VideoFrame

from apps.core.utils import encode_jpg, decode_jpg


INFERENCE_URL = os.getenv('INFERENCE_URL')
RESIZE_URL = os.getenv('RESIZE_URL')
DEBUG_SAVE = os.getenv('DEBUG_SAVE', 'false').lower() == 'true'

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

    async def add_frame(self, jpg_bytes):
        img = decode_jpg(jpg_bytes)
        if img is None:
            return

        frame = VideoFrame.from_ndarray(img, format="bgr24")

        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await self.queue.put(frame)


def parse_client_candidate(candidate_dict):
    pattern = r'candidate:(\S+) (\d+) (\S+) (\d+) (\S+) (\d+) typ (\S+)'
    candidate_str = candidate_dict.get('candidate', '')
    match = re.match(pattern, candidate_str)

    if not match:
        print(f"Invalid candidate format: {candidate_str}")
        return None

    foundation, component, protocol, priority, ip, port, cand_type = match.groups()

    return RTCIceCandidate(
        foundation=foundation,
        component=int(component),
        protocol=protocol,
        priority=int(priority),
        ip=ip,
        port=int(port),
        type=cand_type,
        sdpMid=candidate_dict.get('sdpMid'),
        sdpMLineIndex=candidate_dict.get('sdpMLineIndex')
    )

def is_frame_empty(img, pixel_threshold=10, ratio=0.99):
    """
    Consider a frame empty if > ratio of pixels have intensity below pixel_threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    low_pixels = np.sum(gray < pixel_threshold)
    return (low_pixels / gray.size) > ratio

async def process_video_frame(frame, session, return_track, inference_url, resize_url, debug_save, data_channel):
    # opencv works with BGR instead of RGB
    img = frame.to_ndarray(format="bgr24")

    if is_frame_empty(img):
        print("Skipping empty frame")
        data_channel.send(str({"predicted_class": "empty", "confidence": 100.0}))
        return

    jpg_bytes = encode_jpg(img)

    headers = {'X-Debug-Save': '1'} if debug_save else {}

    async with session.post(resize_url, data=jpg_bytes, headers=headers) as resp:
        resized_jpg_bytes = await resp.read()

    async with session.post(inference_url, data=resized_jpg_bytes, headers=headers) as resp:
        prediction_json = await resp.json()

    if data_channel and data_channel.readyState == "open":
        data_channel.send(str(prediction_json))
        print("Sent to data channel:", prediction_json)
    else:
        print("Data channel not ready")

    await return_track.add_frame(resized_jpg_bytes)


@csrf_exempt
async def main_view(request):
    """Handle incoming WebRTC offer, establish connection, process video frames."""
    params = json.loads(request.body)
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)

    return_track = ProcessedVideoTrack()
    pc.addTrack(return_track)

    data_channel_container = {"channel": None}  # mutable container for closure access

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"DataChannel received: {channel.label}")
        data_channel_container["channel"] = channel

        @channel.on("message")
        def on_message(message):
            print(f"Received message on data channel: {message}")

    @pc.on("track")
    async def on_track(track):
        print(f"Track received: {track.kind}")

        if track.kind == "video":
            print("Starting video track processing")

            async with aiohttp.ClientSession() as session:
                try:
                    while True:
                        frame = await track.recv()
                        asyncio.create_task(
                            process_video_frame(
                                frame, session, return_track,
                                INFERENCE_URL, RESIZE_URL, DEBUG_SAVE,
                                data_channel_container["channel"]
                            )
                        )
                except Exception as e:
                    print(f"Track processing ended: {e}")

            @track.on("ended")
            async def on_ended():
                print("Video track ended")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state changed: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()

    await pc.setRemoteDescription(offer)

    # Add ICE candidates
    for candidate_dict in params.get("candidates", []):
        candidate = parse_client_candidate(candidate_dict)
        if candidate:
            await pc.addIceCandidate(candidate)

    # Create and set local answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Gather ICE candidates from server side
    server_candidates = []

    @pc.on("icecandidate")
    def on_icecandidate(candidate):
        if candidate:
            server_candidates.append(candidate.toJSON())

    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.5)

    return JsonResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "candidates": server_candidates
    })
