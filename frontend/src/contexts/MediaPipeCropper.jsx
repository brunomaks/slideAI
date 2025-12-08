export function CropperProvider(inputStream) {
    const [inputTrack] = inputStream.getVideoTracks();
    const settings = inputTrack.getSettings();

    // Create video element to read the stream
    const video = document.createElement('video');
    video.srcObject = inputStream;
    video.autoplay = true;
    video.muted = true;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    canvas.width = settings.width;
    canvas.height = settings.height;

    let animationId;

    function processFrame() {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            // Draw the full video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            console.log("Processing frame here");
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

    console.log("Cropped stream created");
    return outputStream;
}
