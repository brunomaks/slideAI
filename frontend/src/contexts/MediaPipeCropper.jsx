import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
// import { drawConnectors, drawLandmarks } from "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1";
// import { HAND_CONNECTIONS } from "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1";

// Hand connections FOR DRAWING
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],           // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],           // Index
    [0, 9], [9, 10], [10, 11], [11, 12],      // Middle
    [0, 13], [13, 14], [14, 15], [15, 16],    // Ring
    [0, 17], [17, 18], [18, 19], [19, 20],    // Pinky
    [5, 9], [9, 13], [13, 17]                 // Palm
];
// Draw connectors FOR DRAWING
function drawConnectors(ctx, landmarks, connections, style) {
    ctx.strokeStyle = style.color;
    ctx.lineWidth = style.lineWidth;

    for (const connection of connections) {
        const [startIdx, endIdx] = connection;
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];

        ctx.beginPath();
        ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
        ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
        ctx.stroke();
    }
}
// Draw landmarks FOR DRAWING
function drawLandmarks(ctx, landmarks, style) {
    ctx.fillStyle = style.color;

    for (const landmark of landmarks) {
        const x = landmark.x * ctx.canvas.width;
        const y = landmark.y * ctx.canvas.height;

        ctx.beginPath();
        ctx.arc(x, y, style.lineWidth * 2, 0, 2 * Math.PI);
        ctx.fill();
    }
}

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

    canvas.width = settings.width;
    canvas.height = settings.height;

    let animationId;
    let lastVideoTime = -1;
    let results = undefined;

    async function processFrame() {
        if (!handLandmarker) {
            console.log("Wait! handLandmarker not loaded yet.");
            animationId = requestAnimationFrame(processFrame);
            return;
        }

        // Draw the video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Detect hands on video frame
        let startTimeMs = performance.now();
        if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            results = handLandmarker.detectForVideo(video, startTimeMs);
        }

        // Draw hand landmarks on top of the video
        if (results && results.landmarks) {
            for (const landmarks of results.landmarks) {
                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                    color: "#00FF00",
                    lineWidth: 5
                });
                drawLandmarks(ctx, landmarks, {
                    color: "#FF0000",
                    lineWidth: 2
                });
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
