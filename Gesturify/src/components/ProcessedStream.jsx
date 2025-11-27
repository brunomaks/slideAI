import React, { useRef, useEffect } from "react";
import { useWebRTC } from "../contexts/WebRTCContext";
import "./ProcessedStream.css";

export default function ProcessedStream() {
    const videoRef = useRef(null);
    const { remoteStream } = useWebRTC();

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        if (remoteStream) {
            video.srcObject = remoteStream;
        } else {
            video.srcObject = null;
        }
    }, [remoteStream]);

    return (
        <div className="processed-stream-wrapper">
            <video
                ref={videoRef}
                autoPlay
                playsInline
                disablePictureInPicture
            />
        </div>
    );
}
