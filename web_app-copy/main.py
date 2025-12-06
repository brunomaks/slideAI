from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from contextlib import asynccontextmanager
import asyncio

# Peer connection
pc = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pc
    pc = None

    yield # divides between startup and shutdown phases

    # Shutdown
    if pc is not None:
        await pc.close()


app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


relay = MediaRelay()


class OfferRequest(BaseModel):
    sdp: str
    type: str

# WARNING: No error handling
@app.post("/offer")
async def offer(request: OfferRequest):
    """
    Receives WebRTC offer from client and returns answer
    """
    global pc

    # Create new peer connection
    pc = RTCPeerConnection()

    signaling_state_change = asyncio.Future()
    
    # Handle incoming tracks (audio/video from client)
    @pc.on("track")
    async def on_track(track):
        print(f"Track received: {track.kind}")
        
        # Relay the track back to client (echo example)
        # Remove this if you don't want to echo media back
        pc.addTrack(relay.subscribe(track))
        
        @track.on("ended")
        async def on_ended():
            print(f"Track ended: {track.kind}")
    
    # Handle connection state changes
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()

    # Handle signalling connection state change
    @pc.on("signalingstatechange")
    async def on_signalingstatechange():
        if pc.signalingState == "have-remote-offer":
            signaling_state_change.set_result("have-remote-offer")
    
    # Set remote description (client's offer)
    offer_sdp = RTCSessionDescription(sdp=request.sdp, type=request.type)
    await pc.setRemoteDescription(offer_sdp)

    # ensure the state changed before creating answer
    await signaling_state_change
    print(f"Current state: {pc.signalingState}")
    
    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    # Return answer to client
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)