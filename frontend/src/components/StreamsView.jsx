import React, { useState } from 'react';
import VideoWrapper from './VideoWrapper';
import CameraStream from './CameraStream';
import CameraOverlay from './CameraOverlay';
import ProcessedStream from './ProcessedStream';
import './StreamsView.css';

export default function StreamsView() {
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
        <div className="streams-view">
            <div className="control-topbar">
                {/* Control buttons will go here */}
            </div>
            <div className="video-container">
                <div className="selfie-video-wrapper-instance video-wrapper-instance">
                    <VideoWrapper>
                        {cameraEnabled && <CameraStream className="camera-stream" onStreamReady={handleStreamReady} />}
                        <CameraOverlay
                            cameraEnabled={cameraEnabled}
                            onEnableCamera={handleEnableCamera}
                            onDisableCamera={handleDisableCamera}
                        />
                    </VideoWrapper>
                </div>
                <div className="processed-video-wrapper-instance video-wrapper-instance">
                    <VideoWrapper>
                        <ProcessedStream />
                    </VideoWrapper>
                </div>
            </div>
        </div>
    );
}
