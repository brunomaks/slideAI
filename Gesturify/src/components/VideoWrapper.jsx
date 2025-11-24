import React, { useState } from "react"
import CameraStream from "./CameraStream"
import CameraOverlay from "./CameraOverlay"
import "./VideoWrapper.css"

export default function VideoWrapper() {
    const [stream, setStream] = useState(null);
    const [cameraEnabled, setCameraEnabled] = useState(false);

    const handleStreamReady = (mediaStream) => {
        setStream(mediaStream);
        setCameraEnabled(true);
    };

    const handleEnableCamera = () => {
        setCameraEnabled(true);
    };

    const handleDisableCamera = () => {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
        }
        setStream(null);
        setCameraEnabled(false);
    };

    return (
        <div>
            <div className="video-wrapper">
                {cameraEnabled && <CameraStream className="camera-stream" onStreamReady={handleStreamReady} />}
                <CameraOverlay
                    cameraEnabled={cameraEnabled}
                    onEnableCamera={handleEnableCamera}
                    onDisableCamera={handleDisableCamera}
                />
            </div>
        </div>
    )
}