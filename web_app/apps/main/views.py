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

import time


INFERENCE_URL = os.getenv('INFERENCE_URL')
RESIZE_URL = os.getenv('RESIZE_URL')
DEBUG_SAVE = os.getenv('DEBUG_SAVE', 'false').lower() == 'true'

class ProcessedVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(1)
        
        # Debugging counters
        self.frames_added = 0
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_log_time = time.time()
        
    async def recv(self):
        recv_start = time.time()
        
        # Measure queue wait time
        queue_wait_start = time.time()
        frame = await self.queue.get()
        queue_wait_time = time.time() - queue_wait_start
        
        # Add timestamps
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        
        self.frames_received += 1
        recv_time = time.time() - recv_start
        
        # Log every 30 frames
        if self.frames_received % 30 == 0:
            print(f"[RECV] Total received: {self.frames_received}, "
                  f"Queue wait: {queue_wait_time*1000:.1f}ms, "
                  f"Total recv time: {recv_time*1000:.1f}ms, "
                  f"Queue size: {self.queue.qsize()}")
        
        return frame
    
    async def add_frame(self, jpg_bytes):
        add_start = time.time()
        
        # Check if queue is full
        if self.queue.full():
            self.frames_dropped += 1
            # Log drops
            if self.frames_dropped % 10 == 0:
                print(f"[ADD] Dropped {self.frames_dropped} frames so far")
            return
        
        # Measure decode time
        decode_start = time.time()
        img = decode_jpg(jpg_bytes)
        decode_time = time.time() - decode_start
        
        if img is None:
            return
        
        # Measure frame conversion time
        convert_start = time.time()
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        convert_time = time.time() - convert_start
        
        # Measure queue put time
        put_start = time.time()
        await self.queue.put(frame)
        put_time = time.time() - put_start
        
        self.frames_added += 1
        total_add_time = time.time() - add_start
        
        # Log stats every second
        now = time.time()
        if now - self.last_log_time >= 1.0:
            fps_added = self.frames_added / (now - self.last_log_time)
            fps_received = self.frames_received / (now - self.last_log_time)
            
            print(f"\n[STATS] Add FPS: {fps_added:.1f}, Recv FPS: {fps_received:.1f}, "
                  f"Dropped: {self.frames_dropped}, Queue: {self.queue.qsize()}")
            print(f"[TIMING] Decode: {decode_time*1000:.1f}ms, "
                  f"Convert: {convert_time*1000:.1f}ms, "
                  f"Put: {put_time*1000:.1f}ms, "
                  f"Total add: {total_add_time*1000:.1f}ms\n")
            
            self.last_log_time = now
            self.frames_added = 0
            self.frames_received = 0

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

    jpg_bytes = encode_jpg(img)

    if is_frame_empty(img):
        print("Skipping empty frame")
        data_channel.send(str({"predicted_class": "empty", "confidence": 100.0}))
        await return_track.add_frame(jpg_bytes)
        return

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
                        await process_video_frame(
                                frame, session, return_track,
                                INFERENCE_URL, RESIZE_URL, DEBUG_SAVE,
                                data_channel_container["channel"]
                            )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"Track processing ended: {e}")

            @track.on("ended")
            async def on_ended():
                print("Video track ended")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state changed: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
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
