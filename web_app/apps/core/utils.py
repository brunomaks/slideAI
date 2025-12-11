import cv2
import numpy as np

""" Purpose:
# HTTP gives us JPEG bytes, which are compressed data—not raw pixels.  
# To resize or pad the image, we first decode the JPEG into a pixel array.  
# After processing, we re-encode it back to JPEG bytes so it can be sent over HTTP """
# HTTP JPEG bytes (request) -> decode -> pixel array -> resize + pad -> encode -> HTTP JPEG bytes (respoens)

def encode_jpg(image: np.ndarray, quality: int = 95) -> bytes:
    """
    Encode an OpenCV image (numpy ndarray) to JPEG bytes.

    Args:
        image (np.ndarray): Image array in BGR format.
        quality (int): JPEG quality (0-100), opencv default is 95.

    Returns:
        bytes: JPEG encoded bytes.
    """
    success, encoded_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return encoded_img.tobytes()


def decode_jpg(jpg_bytes: bytes) -> np.ndarray:
    """
    Decode JPEG bytes to an OpenCV image (numpy ndarray).

    Args:
        jpg_bytes (bytes): JPEG image bytes.

    Returns:
        np.ndarray: Decoded image array in BGR format.
    """
    nparr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG bytes to image")
    return img
