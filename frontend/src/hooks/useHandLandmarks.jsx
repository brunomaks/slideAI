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

  const lastTimestampMsRef = useRef(-1);
  const lastVideoTimeRef = useRef(-1);


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

    video.style.position = "fixed";
    video.style.top = "0";
    video.style.left = "-9999px";
    video.width = 640;
    video.height = 480;

    document.body.appendChild(video);
    video.play().catch((error) => {
      console.error("Error playing video:", error);
    });
    videoRef.current = video;

  
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      
      video.srcObject = null;
      document.body.removeChild(video);
      videoRef.current = null;
    };
  }, [inputStream]);

  // main loop for frame processing and callback with landmarks
  useEffect(() => {
    const processFrame = () => {
      const video = videoRef.current;

      if (!video || video.readyState < 2 || !handLandmarker) {
        // Wait and try again
        animationIdRef.current = requestAnimationFrame(processFrame);
        return;
      }


      if (video.videoWidth === 0 || video.videoHeight === 0) {
        animationIdRef.current = requestAnimationFrame(processFrame);
        return;
      }

      if (video.currentTime === lastVideoTimeRef.current) {
        animationIdRef.current = requestAnimationFrame(processFrame);
        return;
      }

      lastVideoTimeRef.current = video.currentTime;
      
      // Use performance.now() for reliable monotonically increasing timestamps
      const timestampMs = performance.now();
      
      // Ensure timestamp is strictly greater than last timestamp
      if (timestampMs <= lastTimestampMsRef.current) {
        // If performance.now() somehow didn't increase enough, force increment
        const adjustedTimestamp = lastTimestampMsRef.current + 1;
        lastTimestampMsRef.current = adjustedTimestamp;
        
        const results = handLandmarker.detectForVideo(
          video,
          adjustedTimestamp
        );
        
        if (results.landmarks && results.landmarks.length > 0 && onLandmarksRef.current) {
          let handedness = results.handednesses[0][0].categoryName;
          const flippedHandedness = handedness === "Left" ? "Right" : "Left";
          const message = {
            landmarks: results.landmarks[0],
            handedness: flippedHandedness
          };
          onLandmarksRef.current(message);
        }
      } else {
        lastTimestampMsRef.current = timestampMs;
        
        const results = handLandmarker.detectForVideo(
          video,
          timestampMs
        );

        if (results.landmarks && results.landmarks.length > 0 && onLandmarksRef.current) {
          let handedness = results.handednesses[0][0].categoryName;
          const flippedHandedness = handedness === "Left" ? "Right" : "Left";
          const message = {
            landmarks: results.landmarks[0],
            handedness: flippedHandedness
          };
          onLandmarksRef.current(message);
        }
      }

      animationIdRef.current = requestAnimationFrame(processFrame);
    };

    // Reset timestamp tracking when stream changes
    lastTimestampMsRef.current = -1;
    lastVideoTimeRef.current = -1;

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