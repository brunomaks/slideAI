import React from "react";
import Webcam from "react-webcam";

export function CameraStream() {
  return <Webcam audio={false} />;
}