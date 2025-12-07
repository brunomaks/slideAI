from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaRelay
from contextlib import asynccontextmanager
import asyncio
import re

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
    candidates: list[dict]

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

    print(len(request.candidates))

    for candidate in request.candidates:
        parsed = parse_client_candidate(candidate)
        if parsed is not None:
            await pc.addIceCandidate(parsed)

    # ensure the signaling state changed before creating answer
    await signaling_state_change
    print(f"Current state: {pc.signalingState}")
    
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
    
    # Return answer to client
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "candidates": server_candidates
    }


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)