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

INFERENCE_URL = os.getenv('INFERENCE_URL')

# Global threee-item queue for inference requests
inference_queue = asyncio.Queue(maxsize=3)

async def inference_worker(channel):
    while True:
        request = await inference_queue.get()
        request_id = request["request_id"]
        landmarks = request["landmarks"]
        handedness = request["handedness"]

        async with aiohttp.ClientSession() as session:
            async with session.post(INFERENCE_URL, json={"landmarks": landmarks, "handedness": handedness}) as resp:
                result = await resp.json()

        # Send response only if channel is open
        if channel.readyState == "open":
            response = {"request_id": request_id, "result": result}
            channel.send(json.dumps(response))

        inference_queue.task_done()


@csrf_exempt
async def main_view(request):
    params = json.loads(request.body)
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)

    data_channel_container = { "channel": None }

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"DataChannel received: {channel.label}")
        data_channel_container["channel"] = channel

        # Start the worker task once per channel
        asyncio.create_task(inference_worker(channel))

        @channel.on("message")
        async def on_message(message):
            try:
                data = json.loads(message)

                request_id = data.get("request_id")
                landmarks = data.get("landmarks")
                handedness = data.get("handedness")

                if request_id is None or landmarks is None or handedness is None:
                    return

                # If queue is full, remove the oldest one
                if inference_queue.full():
                    try:
                        inference_queue.get_nowait()
                        inference_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass

                # Put the latest request in the queue
                request = {
                    "request_id": request_id,
                    "landmarks": landmarks,
                    "handedness": handedness
                }
                await inference_queue.put(request)

            except Exception as e:
                print("Error in on_message:", e)

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