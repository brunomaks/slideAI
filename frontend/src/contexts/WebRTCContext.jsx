import React, { createContext, useContext, useRef, useEffect, useState } from "react";
import axios from "axios";

const WebRTCContext = createContext(null);

const WEBRTC_SERVER_URL = import.meta.env.VITE_WEBRTC_SERVER_URL || "http://localhost:8001/offer/";

export function WebRTCProvider({ children }) {
    const [stream, setStream] = useState(null);
    const [remoteStream, setRemoteStream] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const pcRef = useRef(null);
    const dataChannelRef = useRef(null);

    useEffect(() => {
        if (!stream) {
            return;
        }

        const setupWebRTC = async () => {
            try {
                const pc = new RTCPeerConnection({
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' }
                    ],
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

                let dataChannel = pc.createDataChannel("MyApp Channel");

                dataChannel.onopen = () => {
                    console.log("DataChannel open");
                };

                dataChannel.onclose = () => {
                    console.log("DataChannel closed");
                };

                dataChannel.onmessage = (event) => {
                    const jsonString = event.data.replace(/'/g, '"');
                    const parsedData = JSON.parse(jsonString);
                    // do not trigger react rerender on empty gesture
                    if (parsedData.predicted_class !== 'empty') {
                        setPrediction(parsedData);
                    }
                    console.log("Received:", parsedData);
                };

                dataChannelRef.current = dataChannel

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                // WAIT for all client candidates to be gathered
                const clientCandidates = [];
                await new Promise((resolve) => {
                    pc.onicecandidate = (event) => {
                        if (event.candidate) {
                            clientCandidates.push(event.candidate.toJSON());
                        } else {
                            // null candidate = gathering complete
                            resolve();
                        }
                    };

                    // Fallback timeout in case gathering takes too long
                    setTimeout(resolve, 500);
                });

                const { data: answer } = await axios.post(WEBRTC_SERVER_URL, {
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                    candidates: clientCandidates
                });

                await pc.setRemoteDescription(
                    new RTCSessionDescription({
                        sdp: answer.sdp,
                        type: answer.type,
                    })
                );

                if (answer.candidates) {
                    for (const candidate in answer.candidates) {
                        await pc.addIceCandidate(new RTCIceCandidate(candidate))
                    }
                }

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
                pcRef.current = null
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

    // send message over webrtc datachannel
    const sendMessage = (payload) => {
        const channel = dataChannelRef.current;

        if (!channel || channel.readyState !== "open") {
            console.log("Data channel not open");
            return;
        }

        channel.send(JSON.stringify(payload));
    };


    const value = {
        stream,
        remoteStream,
        isConnected,
        prediction,
        connectStream,
        disconnectStream,
        sendMessage
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
