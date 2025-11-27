from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
# use opencv for image processing
import cv2
import numpy as np
import os
import time

#csrf_exempt is needed since without it we get a 403 error when posting the request
@csrf_exempt
def flip_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST an image file"}, status=400)
    print("Flip service received a request")
    nparr = np.frombuffer(request.body, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # flip the image horizontally
    flipped = cv2.flip(img, 1)

    # save the flipped image to the output folder if header is there
    if request.headers.get('X-Debug-Save'):
        print("Saving flipped image to disk for debugging")
        filename = "frame" + "_flip_" + str(int(time.time_ns())) + ".jpg"
        path = os.path.join(settings.MEDIA_ROOT, filename)
        cv2.imwrite(path, flipped)

    # return the flipped image
    _, buffer = cv2.imencode('.jpg', flipped)
    return HttpResponse(buffer.tobytes(), content_type='image/jpeg')
