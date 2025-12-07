import React, { useRef, useEffect } from "react";
import { useWebRTC } from "../contexts/WebRTCContext";

const streamConfig = {
  fps: parseInt(import.meta.env.VITE_MAX_STREAM_FPS) || 24,
  width: parseInt(import.meta.env.VITE_MAX_STREAM_WIDTH) || 1280,
  height: parseInt(import.meta.env.VITE_MAX_STREAM_HEIGHT) || 720,
  bitrate: parseInt(import.meta.env.VITE_MAX_STREAM_BITRATE) || 2500000,
  delay: parseInt(import.meta.env.VITE_ORIGINAL_STREAM_VIEW_DELAY_MS) || 2000,
};

export default function CameraStream({ onStreamReady }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const { connectStream, disconnectStream } = useWebRTC();

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { max: streamConfig.width },
            height: { max: streamConfig.height },
            frameRate: { max: streamConfig.fps },
          },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;

          connectStream(stream);

          if (onStreamReady) {
            onStreamReady(stream);
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
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
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