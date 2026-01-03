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

                {mediapipeStatus && (
                    <div className="mediapipe-status">
                        {mediapipeStatus.isLoading && <span>Loading Mediapipe...</span>}
                        {mediapipeStatus.isReady && <span>Mediapipe Ready</span>}
                        {mediapipeStatus.error && <span className="error">Error: {mediapipeStatus.error}</span>}
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
