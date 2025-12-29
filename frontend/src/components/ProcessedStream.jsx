import React, { useRef, useEffect } from "react";
import { useWebSocket } from "../contexts/WebSocketContext";
import "./ProcessedStream.css";

export default function ProcessedStream() {
    const imgRef = useRef(null);
    const { processedFrame } = useWebSocket();

    useEffect(() => {
        const img = imgRef.current;
        if (!img) return;

        if (processedFrame) {
            img.src = processedFrame;
        }
    }, [processedFrame]);

    return (
        <div className="processed-stream-wrapper">
            <img
                ref={imgRef}
            />
        </div>
    );
}
