from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import os
import time

from apps.core.utils import encode_jpg, decode_jpg

SIZE = 150

@csrf_exempt
def resize_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST an image file"}, status=400)

    try:
        img = decode_jpg(request.body)
    except ValueError as e:
        return JsonResponse({"error": f"Invalid image data: {str(e)}"}, status=400)

    resized = resize_to_square(img, SIZE)

    if request.headers.get("X-Debug-Save"):
        debug_save_image(resized)

    try:
        jpg_bytes = encode_jpg(resized)
    except ValueError as e:
        return JsonResponse({"error": f"Failed to encode image: {str(e)}"}, status=500)

    return HttpResponse(jpg_bytes, content_type="image/jpeg")

def resize_to_square(img, size: int):
    h, w = img.shape[:2]

    # scale resizing
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # compute padding
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )


def debug_save_image(img):
    filename = f"frame_resize_{time.time_ns()}.jpg"
    path = os.path.join(settings.MEDIA_ROOT, filename)
    cv2.imwrite(path, img)
