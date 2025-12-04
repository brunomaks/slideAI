import React from "react";
import "./CameraOverlay.css";

export default function CameraOverlay({ cameraEnabled, onEnableCamera, onDisableCamera }) {
    if (cameraEnabled) {
        return (
            <div className="camera-overlay-controls">
                <button
                    onClick={onDisableCamera}
                    className="camera-button camera-button-disable"
                >
                    Disable Camera
                </button>
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
