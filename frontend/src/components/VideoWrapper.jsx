import React from "react"
import "./VideoWrapper.css"

export default function VideoWrapper({ children }) {
    return (
        <div className="video-wrapper">
            {children}
        </div>
    )
}