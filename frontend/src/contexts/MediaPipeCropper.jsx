import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let handLandmarker = undefined;

let runningMode = "VIDEO";

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });
};
createHandLandmarker();

export function CropperProvider(inputStream) {
    const [inputTrack] = inputStream.getVideoTracks();
    const settings = inputTrack.getSettings();

    const video = document.createElement('video');
    video.srcObject = inputStream;
    video.autoplay = true;
    video.muted = true;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    let animationId;
    let lastVideoTime = -1;
    let results = undefined;

    async function processFrame() {
        if (!handLandmarker) {
            console.log("Wait! handLandmarker not loaded yet.");
            animationId = requestAnimationFrame(processFrame);
            return;
        }

        let startTimeMs = performance.now();
        if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            results = handLandmarker.detectForVideo(video, startTimeMs);
        }


        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.landmarks) {
            if (results.landmarks.length > 0) {
                const landmark_0 = results.landmarks[0]

                function cropHandBoundingBox(ctx, handLandmarks, MARGIN) {

                    const videoWidth = video.videoWidth;
                    const videoHeight = video.videoHeight;

                    // Extract x and y coordinates (normalized)
                    const xCoordinates = handLandmarks.map(lm => lm.x);
                    const yCoordinates = handLandmarks.map(lm => lm.y);

                    let boxXMin = Math.floor(Math.min(...xCoordinates) * videoWidth) - MARGIN;
                    let boxYMin = Math.floor(Math.min(...yCoordinates) * videoHeight) - MARGIN;
                    let boxXMax = Math.floor(Math.max(...xCoordinates) * videoWidth) + MARGIN;
                    let boxYMax = Math.floor(Math.max(...yCoordinates) * videoHeight) + MARGIN;

                    // Clamp bounding box to image dimensions
                    boxXMin = Math.max(boxXMin, 0);
                    boxYMin = Math.max(boxYMin, 0);
                    boxXMax = Math.min(boxXMax, videoWidth);
                    boxYMax = Math.min(boxYMax, videoHeight);

                    if (boxXMin >= boxXMax || boxYMin >= boxYMax) {
                        console.warn("Invalid bounding box dimensions. Skipping this frame.");
                        return null;
                    }

                    const cropWidth = boxXMax - boxXMin;
                    const cropHeight = boxYMax - boxYMin;

                    canvas.width = cropWidth;
                    canvas.height = cropHeight;

                    // Crop using drawImage
                    ctx.drawImage(
                        video,
                        boxXMin,
                        boxYMin,
                        cropWidth,
                        cropHeight,
                        0,
                        0,
                        cropWidth,
                        cropHeight
                    );
                }
                cropHandBoundingBox(ctx, landmark_0, 10)
            }
        }

        animationId = requestAnimationFrame(processFrame);
    }

    video.addEventListener('loadedmetadata', () => {
        processFrame();
    });

    const outputStream = canvas.captureStream(settings.frameRate || 24);

    outputStream.cleanup = () => {
        cancelAnimationFrame(animationId);
        video.srcObject = null;
        inputTrack.stop();
    };

    console.log("Hand landmarking stream created");
    return outputStream;
}

export default CropperProvider;
