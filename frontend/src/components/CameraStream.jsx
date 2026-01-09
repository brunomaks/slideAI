/*
 Contributors:
- Yaroslav
- Mahmoud
- Mykhailo
- Ahmet

*/

import React, { useRef, useState, useEffect } from "react";
import { useWebSocket } from "../contexts/WebSocketContext";
import { useHandLandmarks } from "../hooks/useHandLandmarks";

const streamConfig = {
    fps: parseInt(import.meta.env.VITE_MAX_STREAM_FPS) || 24,
    width: parseInt(import.meta.env.VITE_MAX_STREAM_WIDTH) || 1280,
    height: parseInt(import.meta.env.VITE_MAX_STREAM_HEIGHT) || 720,
    delay: parseInt(import.meta.env.VITE_ORIGINAL_STREAM_VIEW_DELAY_MS) || 2000,
};

export default function CameraStream({ onStreamReady }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const rawStreamRef = useRef(null);
    const { connect, disconnect, send } = useWebSocket();
    const [stream, setStream] = useState(null)
    const { subscribeToLandmarks } = useHandLandmarks(stream);

    // draw bounding box on canvas
    const drawBoundingBox = (landmarks, canvas, video) => {
        const ctx = canvas.getContext("2d");

        // remove previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!landmarks || landmarks.length === 0) return;

        // use video's intrinsic dimensions
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        if (videoWidth === 0 || videoHeight === 0) return;

        // min/max x and y coordinates from all landmarks
        let minX = 1, minY = 1, maxX = 0, maxY = 0;

        landmarks.forEach(landmark => {
            minX = Math.min(minX, landmark.x);
            minY = Math.min(minY, landmark.y);
            maxX = Math.max(maxX, landmark.x);
            maxY = Math.max(maxY, landmark.y);
        });

        // add padding (10% of box size)
        const width = maxX - minX;
        const height = maxY - minY;
        const padding = 0.1;

        minX = Math.max(0, minX - width * padding);
        minY = Math.max(0, minY - height * padding);
        maxX = Math.min(1, maxX + width * padding);
        maxY = Math.min(1, maxY + height * padding);

        // convert normalized coordinates to VIDEO coordinates (not canvas)
        const x = minX * videoWidth;
        const y = minY * videoHeight;
        const boxWidth = (maxX - minX) * videoWidth;
        const boxHeight = (maxY - minY) * videoHeight;

        // draw bounding box
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, boxWidth, boxHeight);
    };

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

                setStream(rawStream);

                if (videoRef.current) {
                    videoRef.current.srcObject = rawStream;
                    rawStreamRef.current = rawStream;

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
        connect(); // connect ws

        return () => {
            if (rawStreamRef.current) {
                rawStreamRef.current.getTracks().forEach((track) => track.stop());
            }
            disconnect(); // disconnect ws
        };
    }, []);

    // set canvas size to match video's intrinsic dimensions
    useEffect(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (!video || !canvas) return;

        const updateCanvasSize = () => {
            // set canvas internal resolution to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            console.log(`Canvas sized: ${canvas.width}x${canvas.height}`);
        };

        // try multiple events to catch when video dimensions are ready
        video.addEventListener("loadedmetadata", updateCanvasSize);
        video.addEventListener("loadeddata", updateCanvasSize);

        // also try immediately in case video is already loaded
        if (video.videoWidth > 0) {
            updateCanvasSize();
        }

        return () => {
            video.removeEventListener("loadedmetadata", updateCanvasSize);
            video.removeEventListener("loadeddata", updateCanvasSize);
        };
    }, [stream]);

    let requestId = 0;

    useEffect(() => {
        if (!subscribeToLandmarks) return;

        let lastDetectionTime = Date.now();
        const CLEAR_TIMEOUT = 200; // clear box after 200ms of no detection

        // check periodically if we should clear the canvas
        const clearCheckInterval = setInterval(() => {
            if (Date.now() - lastDetectionTime > CLEAR_TIMEOUT) {
                if (canvasRef.current) {
                    const ctx = canvasRef.current.getContext("2d");
                    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                }
            }
        }, 100);

        const unsubscribe = subscribeToLandmarks((landmarks_obj) => {
            lastDetectionTime = Date.now();
            requestId += 1;
            const message = {
                request_id: requestId,
                landmarks: landmarks_obj.landmarks,
                handedness: landmarks_obj.handedness
            };

            send(message)

            // Draw bounding box
            if (canvasRef.current && videoRef.current) {
                drawBoundingBox(landmarks_obj.landmarks, canvasRef.current, videoRef.current);
            }
        });

        return () => {
            clearInterval(clearCheckInterval);
            unsubscribe();
        };
    }, [subscribeToLandmarks]);

    return (
        <div className="camera-stream-wrapper" style={{ position: "relative", width: "100%", height: "100%" }}>
            <video
                ref={videoRef}
                autoPlay
                playsInline
                disablePictureInPicture
                style={{
                    display: "block",
                    width: "100%",
                    height: "100%",
                    objectFit: "contain"
                }}
            />
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                    pointerEvents: "none",
                    objectFit: "contain"
                }}
            />
        </div>
    );
}
