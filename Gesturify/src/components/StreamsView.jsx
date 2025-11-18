import React from 'react';
import VideoWrapper from './VideoWrapper';
import './StreamsView.css';

export default function StreamsView() {
    return (
        <div className="streams-view">
            <div className="control-topbar">
                {/* Control buttons will go here */}
            </div>
            <div className="video-container">
                <div className="selfie-video-wrapper-instance video-wrapper-instance">
                    <VideoWrapper />
                </div>
                <div className="processed-video-wrapper-instance video-wrapper-instance">
                    <VideoWrapper />
                </div>
            </div>
        </div>
    );
}
