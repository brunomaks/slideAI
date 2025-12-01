import React, { createContext, useContext, useRef, useEffect, useState } from "react";
import axios from "axios";

const WebRTCContext = createContext(null);

const WEBRTC_SERVER_URL = import.meta.env.VITE_WEBRTC_SERVER_URL || "http://localhost:8001/offer/";

export function WebRTCProvider({ children }) {
    const [stream, setStream] = useState(null);
    const [remoteStream, setRemoteStream] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const pcRef = useRef(null);

    useEffect(() => {
        if (!stream) {
            return;
        }

        const setupWebRTC = async () => {
            try {
                const pc = new RTCPeerConnection({
                    iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
                });

                pcRef.current = pc;
                stream.getTracks().forEach((track) => {
                    pc.addTrack(track);
                });

                pc.ontrack = (event) => {
                    if (event.streams && event.streams[0]) {
                        setRemoteStream(event.streams[0]);
                        console.log("Remote stream received");
                    }
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

                setIsConnected(true);
                console.log("WebRTC connection established");
            } catch (error) {
                console.error("WebRTC error:", error);
                setIsConnected(false);
            }
        };

        setupWebRTC();

        return () => {
            if (pcRef.current) {
                pcRef.current.close();
                console.log("WebRTC closed");
            }
            setIsConnected(false);
            setRemoteStream(null);
        };
    }, [stream]);

    const connectStream = (mediaStream) => {
        setStream(mediaStream);
    };

    const disconnectStream = () => {
        if (pcRef.current) {
            pcRef.current.close();
            pcRef.current = null;
        }
        setStream(null);
        setRemoteStream(null);
    };

    const value = {
        stream,
        remoteStream,
        isConnected,
        connectStream,
        disconnectStream,
    };

    return (
        <WebRTCContext.Provider value={value}>
            {children}
        </WebRTCContext.Provider>
    );
}

export function useWebRTC() {
    const context = useContext(WebRTCContext);
    if (!context) {
        throw new Error("useWebRTC must be used within a WebRTCProvider");
    }
    return context;
}
