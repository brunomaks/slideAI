from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
# use opencv for image processing
import cv2
import numpy as np
import os
import time
import tqdm

#csrf_exempt is needed since without it we get a 403 error when posting the request
@csrf_exempt
def flip_view(request):
    print("Flip service received a request")
    
    if request.method != "POST":
        return JsonResponse({"error": "POST an image file"}, status=400)
    
    size = int(request.headers.get('X-Resize-Size', 94))
    
    # STEP 1: Load the input image. 
    nparr = np.frombuffer(request.body, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # STEP 2: Resize and save the image
    resized = resize_and_save(img, size)
    
    # save the resized image to the output folder if header is there
    if request.headers.get('X-Debug-Save'):
        filename = "frame" + "_resize_" + str(int(time.time_ns())) + ".jpg"
        path = os.path.join(settings.MEDIA_ROOT, filename)
        print("Saving resized image to disk for debugging at path:", path)
        cv2.imwrite(path, resized)

    # return the flipped image
    _, buffer = cv2.imencode('.jpg', resized)
    return HttpResponse(buffer.tobytes(), content_type='image/jpeg')

def resize_and_save(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # compute padding
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0,0,0]
    )

    return padded