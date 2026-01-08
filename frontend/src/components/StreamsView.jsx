/*
 Contributors:
- Yaroslav
- Mahmoud

*/

import React, { useState } from 'react';
import VideoWrapper from './VideoWrapper';
import CameraStream from './CameraStream';
import CameraOverlay from './CameraOverlay';
import { useHandLandmarks } from '../hooks/useHandLandmarks.jsx';
import './StreamsView.css';

export default function StreamsView() {
    const [stream, setStream] = useState(null);
    const [cameraEnabled, setCameraEnabled] = useState(false);
    // Get mediapipe status from the hook
    const { mediapipeStatus } = useHandLandmarks(stream);

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
            <VideoWrapper>
                {cameraEnabled && <CameraStream className="camera-stream" onStreamReady={handleStreamReady} />}
                <CameraOverlay
                    cameraEnabled={cameraEnabled}
                    onEnableCamera={handleEnableCamera}
                    onDisableCamera={handleDisableCamera}
                    mediapipeStatus={mediapipeStatus} // Pass mediapipeStatus to overlay
                />
            </VideoWrapper>
        </div>
    );
}
