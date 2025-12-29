import React, { useRef, useEffect } from "react";
import { useWebSocket } from "../contexts/WebSocketContext";
import { CropperProvider } from "../contexts/MediaPipeCropper";

const streamConfig = {
    fps: parseInt(import.meta.env.VITE_MAX_STREAM_FPS) || 5,
    width: parseInt(import.meta.env.VITE_MAX_STREAM_WIDTH) || 1280,
    height: parseInt(import.meta.env.VITE_MAX_STREAM_HEIGHT) || 720,
    delay: parseInt(import.meta.env.VITE_ORIGINAL_STREAM_VIEW_DELAY_MS) || 2000,
};

export default function CameraStream({ onStreamReady }) {
    const videoRef = useRef(null);
    const rawStreamRef = useRef(null);
    const { connectStream, disconnectStream } = useWebSocket();

    useEffect(() => {
        const startCamera = async () => {
            try {
                const rawStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { max: streamConfig.width },
                        height: { max: streamConfig.height },
                        frameRate: { max: streamConfig.fps },
                    },
                    audio: false,
                });


                //TODO: mediapipe
                const croppedStream = CropperProvider(rawStream);

                if (videoRef.current) {
                    videoRef.current.srcObject = rawStream;
                    rawStreamRef.current = rawStream;

                    connectStream(croppedStream);

                    if (onStreamReady) {
                        onStreamReady(croppedStream);
                    }
                }
            } catch (error) {
                console.error("Camera access error:", error);
                if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
                    alert("Camera access was denied. Please allow camera permissions to use this feature.");
                } else {
                    alert(`Camera error: ${error.message}`);
                }
            }
        };

        startCamera();

        return () => {
            if (rawStreamRef.current) {
                rawStreamRef.current.getTracks().forEach((track) => track.stop());
            }
            disconnectStream();
        };
    }, []);

    return (
        <div className="camera-stream-wrapper">
            <video
                ref={videoRef}
                autoPlay
                playsInline
                disablePictureInPicture
            />
        </div>
    );
}
