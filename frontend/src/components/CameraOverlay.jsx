/*
 Contributors:
- Yaroslav
- Mahmoud

*/


import React from "react";
import "./CameraOverlay.css";

export default function CameraOverlay({ cameraEnabled, onEnableCamera, onDisableCamera, mediapipeStatus }) {
    if (cameraEnabled) {
        return (
            <div className="camera-overlay-controls">
                <button
                    onClick={onDisableCamera}
                    className="camera-button camera-button-disable"
                >
                    Disable Camera
                </button>

                {/* MediaPipe Status Display */}
                {mediapipeStatus && (
                    <div className="mediapipe-status">
                        {mediapipeStatus.isLoading && (
                            <div className="status-loading">
                                <span>Loading...</span>
                            </div>
                        )}

                        {mediapipeStatus.isReady && (
                            <div className="status-ready">
                                <span>Hand Detection: On</span>
                            </div>
                        )}

                        {mediapipeStatus.error && (
                            <div className="status-error">
                                <span>Error: {mediapipeStatus.error}</span>
                            </div>
                        )}
                    </div>
                )}

            </div>
        );
    }

    return (
        <div className="camera-overlay-enable">
            <button
                onClick={onEnableCamera}
                className="camera-button camera-button-enable"
            >
                Enable Camera
            </button>
        </div>
    );
}
