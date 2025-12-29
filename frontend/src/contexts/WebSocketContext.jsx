import React, { createContext, useContext, useRef, useEffect, useState, useCallback } from "react";

const WebSocketContext = createContext(null);

const WS_VIDEO_URL = (() => {
    const envUrl = import.meta.env.VITE_WS_VIDEO_URL;
    if (envUrl) return envUrl;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/api/ws/video/`;
})();

const FRAME_CONFIG = {
    fps: parseInt(import.meta.env.VITE_MAX_STREAM_FPS || "5"),
    quality: parseFloat(import.meta.env.VITE_FRAME_QUALITY || "0.7"),
    width: parseInt(import.meta.env.VITE_MAX_STREAM_WIDTH || "1280"),
    height: parseInt(import.meta.env.VITE_MAX_STREAM_HEIGHT || "720"),
};
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY = 2000;

function WebSocketProviderComponent({ children }) {
    const [stream, setStream] = useState(null);
    const [processedFrame, setProcessedFrame] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const [connectionError, setConnectionError] = useState(null);
    const [reconnectTrigger, setReconnectTrigger] = useState(0);

    const wsRef = useRef(null);
    const canvasRef = useRef(null);
    const videoRef = useRef(null);
    const frameIntervalRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const reconnectAttemptsRef = useRef(0);
    const shouldReconnectRef = useRef(true);

    const cleanup = useCallback(() => {
        shouldReconnectRef.current = false;
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    }, []);

    useEffect(() => {
        if (!stream) {
            cleanup();
            return;
        }

        if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
            return;
        }

        console.log("Connecting to WebSocket:", WS_VIDEO_URL);

        try {
            const ws = new WebSocket(WS_VIDEO_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log("WebSocket connected");
                setIsConnected(true);
                setConnectionError(null);
                reconnectAttemptsRef.current = 0;
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.frame) {
                        setProcessedFrame(`data:image/jpeg;base64,${data.frame}`);
                        if (data.prediction?.predicted_class !== "empty") {
                            setPrediction(data.prediction);
                            console.log("Received:", data.prediction);
                        }

                    }
                } catch (e) {
                    console.error("Error parsing WebSocket while parsing the message:", e);
                }
            };

            ws.onclose = (event) => {
                setIsConnected(false);

                if (shouldReconnectRef.current && reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
                    reconnectAttemptsRef.current += 1;
                    reconnectTimeoutRef.current = setTimeout(() => {
                        if (shouldReconnectRef.current) setReconnectTrigger(t => t + 1);
                    }, RECONNECT_DELAY);
                }
            };

            ws.onerror = () => setConnectionError("WebSocket connection failed");
        } catch (error) {
            setConnectionError(error.message);
        }

        return cleanup;
    }, [stream, reconnectTrigger, cleanup]);

    const startFrameCapture = useCallback(() => {
        if (!stream || !wsRef.current) return;

        if (!videoRef.current) {
            videoRef.current = document.createElement("video");
            videoRef.current.autoplay = true;
            videoRef.current.playsInline = true;
            videoRef.current.muted = true;
        }

        if (!canvasRef.current) {
            canvasRef.current = document.createElement("canvas");
            canvasRef.current.width = FRAME_CONFIG.width;
            canvasRef.current.height = FRAME_CONFIG.height;
        }

        videoRef.current.srcObject = stream;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const frameInterval = 1000 / FRAME_CONFIG.fps;

        videoRef.current.onloadedmetadata = () => {
            frameIntervalRef.current = setInterval(() => {
                if (wsRef.current?.readyState !== WebSocket.OPEN) return;

                try {
                    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
                    const base64 = canvas.toDataURL("image/jpeg", FRAME_CONFIG.quality).split(",")[1];
                    wsRef.current.send(JSON.stringify({ type: "frame", frame: base64, timestamp: Date.now() }));
                } catch (e) {
                    console.error("Error capturing/sending frame:", e);
                }
            }, frameInterval);
        };
    }, [stream]);

    useEffect(() => {
        if (isConnected && stream) startFrameCapture();
    }, [isConnected, stream, startFrameCapture]);

    const connectStream = useCallback((mediaStream) => {
        shouldReconnectRef.current = true;
        reconnectAttemptsRef.current = 0;
        setStream(mediaStream);
    }, []);

    const disconnectStream = useCallback(() => {
        cleanup();
        setStream(null);
        setProcessedFrame(null);
        setIsConnected(false);
        setPrediction(null);
    }, [cleanup]);

    const value = {
        stream,
        processedFrame,
        isConnected,
        prediction,
        connectionError,
        connectStream,
        disconnectStream,
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
}

function useWebSocketHook() {
    const context = useContext(WebSocketContext);
    return context;
}

export const WebSocketProvider = WebSocketProviderComponent;
export const useWebSocket = useWebSocketHook;
