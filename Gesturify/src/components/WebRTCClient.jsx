import React, { useRef, useEffect, useState } from "react";
import axios from "axios";

const WEBRTC_SERVER_URL = import.meta.env.VITE_WEBRTC_SERVER_URL || "http://localhost:8080/offer";

export default function WebRTCClient({ stream }) {
    const [connectionState, setConnectionState] = useState("disconnected");
    const pcRef = useRef(null);

    useEffect(() => {
        if (!stream) return;

        const setupWebRTC = async () => {
            try {
                const pc = new RTCPeerConnection({
                    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
                });

                pcRef.current = pc;
                stream.getTracks().forEach((track) => {
                    pc.addTrack(track);
                });

                pc.onconnectionstatechange = () => {
                    setConnectionState(pc.connectionState);
                };

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                const { data: answer } = await axios.post(WEBRTC_SERVER_URL, {
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                });

                await pc.setRemoteDescription(
                    new RTCSessionDescription({
                        sdp: answer.sdp,
                        type: answer.type,
                    })
                );

                console.log("WebRTC conn established");
            } catch (error) {
                console.error("WebRTC error:", error);
                setConnectionState("failed");
            }
        };

        setupWebRTC();

        return () => {
            if (pcRef.current) {
                pcRef.current.close();
                console.log("WebRTC closed");
            }
        };
    }, [stream]);

    return (
        <div style={{ padding: "9px", fontSize: "9px" }}>
            <p>
                WebRTC Status: {connectionState}
            </p>
        </div>
    );
}
