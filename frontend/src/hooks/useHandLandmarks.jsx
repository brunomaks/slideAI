import { useRef, useEffect, useCallback, useState } from "react";
import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let handLandmarker = null;
let handLandmarkerReadyPromise = null;

const initHandLandmarker = () => {
  if (!handLandmarkerReadyPromise) {
    handLandmarkerReadyPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });
      return handLandmarker;
    })();
  }
  return handLandmarkerReadyPromise;
};

export function useHandLandmarks(inputStream) {
  const onLandmarksRef = useRef(null);
  const videoRef = useRef(null);
  const animationIdRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastPredictionTimeMs = useRef(0);
  const PREDICTION_INTERVAL_MS = 100;

  const [mediapipeStatus, setMediapipeStatus] = useState({
    isLoading: true,
    isReady: false,
    error: null,
  });

  // Create or update the video element when inputStream changes
  useEffect(() => {
    if (!inputStream) return;

    // Create hidden video element
    const video = document.createElement("video");
    video.srcObject = inputStream;
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    videoRef.current = video;

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      video.srcObject = null;
      videoRef.current = null;
    };
  }, [inputStream]);

  // Main loop: process frames and call callback with landmarks
  useEffect(() => {
    const processFrame = () => {
      if (!videoRef.current || videoRef.current.readyState < 2 || !handLandmarker) {
        // Wait and try again
        animationIdRef.current = requestAnimationFrame(processFrame);
        return;
      }

      // Only process new frames
      if (lastVideoTimeRef.current !== videoRef.current.currentTime) {
        lastVideoTimeRef.current = videoRef.current.currentTime;
        const startTimeMs = performance.now();

        if ((startTimeMs - lastPredictionTimeMs.current) < PREDICTION_INTERVAL_MS) {
          animationIdRef.current = requestAnimationFrame(processFrame);
          return
        }

        lastPredictionTimeMs.current = startTimeMs

        const results = handLandmarker.detectForVideo(
          videoRef.current,
          startTimeMs
        );

        if (results.landmarks && results.landmarks.length > 0 && onLandmarksRef.current) {
          let handedness = results.handednesses[0][0].categoryName

          const flippedHandedness = handedness === "Left" ? "Right" : "Left";

          const message = {
            landmarks: results.landmarks[0],
            handedness: flippedHandedness
          }
          onLandmarksRef.current(message);
        }

      }

      animationIdRef.current = requestAnimationFrame(processFrame);
    };

    setMediapipeStatus({ isLoading: true, isReady: false, error: null });

    // Wait for handLandmarker to be ready then start processing
    initHandLandmarker().then(() => {
      setMediapipeStatus({ isLoading: false, isReady: true, error: null });
      animationIdRef.current = requestAnimationFrame(processFrame);
    }).catch((error) => {
      console.error("Error initializing Hand Landmarker:", error);
      setMediapipeStatus({ isLoading: false, isReady: false, error: error.message });
    });

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    };
  }, [inputStream]);

  const subscribeToLandmarks = useCallback((callback) => {
    onLandmarksRef.current = callback;

    return () => {
      onLandmarksRef.current = null;
    };
  }, []);

  return { mediapipeStatus, subscribeToLandmarks };
}

export default useHandLandmarks;