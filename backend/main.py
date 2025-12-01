# MOCK WEBRTC BACKEND SERVER
import asyncio
import json
import cv2
import numpy as np
from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder
from av import VideoFrame

pcs = set()

class VideoTransformTrack(VideoStreamTrack):
    """
    A video track that receives frames and can display them
    """
    def __init__(self):
        super().__init__()
        self.frame_count = 0

    async def recv(self):
        frame = await super().recv()
        self.frame_count += 1
        
        img = frame.to_ndarray(format="bgr24")

        cv2.imshow('Received Video', img)
        cv2.waitKey(1)
        
        if self.frame_count % 30 == 0:
            print(f"Received frame {self.frame_count}: {img.shape}")
        
        return frame


async def offer(request):
    """Handle WebRTC offer from client"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    async def on_track(track):
        print(f"Track received: {track.kind}")
        
        if track.kind == "video":
            print("Video track started - processing frames")
            
            try:
                while True:
                    frame = await track.recv()
                    
                    # Convert frame to numpy array for display
                    img = frame.to_ndarray(format="bgr24")
                    
                    # Display the frame using OpenCV
                    cv2.imshow('Received Video', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    if hasattr(on_track, 'frame_count'):
                        on_track.frame_count += 1
                    else:
                        on_track.frame_count = 1
                    
                    if on_track.frame_count % 30 == 0:
                        print(f"Received frame {on_track.frame_count}: {img.shape}")
                        
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

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def on_shutdown(app):
    """Close all peer connections on shutdown"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    cv2.destroyAllWindows()


def main():
    app = web.Application()
    
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })
    
    cors.add(app.router.add_post("/offer", offer))
    app.on_shutdown.append(on_shutdown)
    
    print("WebRTC server starting on http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
