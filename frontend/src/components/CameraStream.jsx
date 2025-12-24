import { useRef, useState, useEffect } from "react";
import { useWebRTC } from "../contexts/WebRTCContext";
import { useHandLandmarks } from "../hooks/useHandLandmarks";

const streamConfig = {
    fps: parseInt(import.meta.env.VITE_MAX_STREAM_FPS) || 5,
    width: parseInt(import.meta.env.VITE_MAX_STREAM_WIDTH) || 1280,
    height: parseInt(import.meta.env.VITE_MAX_STREAM_HEIGHT) || 720,
    bitrate: parseInt(import.meta.env.VITE_MAX_STREAM_BITRATE) || 2500000,
    delay: parseInt(import.meta.env.VITE_ORIGINAL_STREAM_VIEW_DELAY_MS) || 2000,
};

export default function CameraStream({ onStreamReady }) {
    const videoRef = useRef(null);
    const rawStreamRef = useRef(null);
    const { connectStream, disconnectStream, sendMessage } = useWebRTC();

    const [stream, setStream] = useState(null)
    const subscribeToLandmarks = useHandLandmarks(stream);

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

                setStream(rawStream)

                if (videoRef.current) {
                    videoRef.current.srcObject = rawStream;
                    rawStreamRef.current = rawStream;

                    connectStream(rawStream);

                    if (onStreamReady) {
                        onStreamReady(rawStream);
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

    let requestId = 0;

    useEffect(() => {
        if (!subscribeToLandmarks) return;

        const unsubscribe = subscribeToLandmarks((landmarks_obj) => {
            console.log("Sent landmarks")
            requestId += 1
            const message = {
                request_id: requestId,
                landmarks: landmarks_obj.landmarks,
                handedness: landmarks_obj.handedness
            }

            sendMessage(message)
        });

        return () => unsubscribe();
    }, [subscribeToLandmarks]);

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
