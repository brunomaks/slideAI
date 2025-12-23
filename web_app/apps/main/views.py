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

        @channel.on("message")
        async def on_message(message):
            try:
                data = json.loads(message)
            except Exception as e:
                print(f"Failed to parse JSON message: {e}")
                return

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(INFERENCE_URL, json=data) as resp:
                        response_json = await resp.json()
                except Exception as e:
                    print(f"Inference server request failed: {e}")
                    response_json = {"error": "inference request failed"}

            # Send response back via data channel
            if channel.readyState == "open":
                channel.send(json.dumps(response_json))
                print("Sent response back via datachannel:", response_json)
            else:
                print("Data channel not open to send response")

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