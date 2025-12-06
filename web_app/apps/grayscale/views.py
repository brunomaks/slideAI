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
def grayscale_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST an image file"}, status=400)
    
    print("Grayscale service received a request")

    nparr = np.frombuffer(request.body, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # convert the file to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save to output folder if header is present
    if request.headers.get('X-Debug-Save'):
        print("Saving grayscale image to disk for debugging")
        filename = "frame" + "_grayscale_" + str(int(time.time_ns())) + ".jpg"
        path = os.path.join(settings.MEDIA_ROOT, filename)
        cv2.imwrite(path, gray)

    # return the grayscale image
    _, buffer = cv2.imencode('.jpg', gray)
    return HttpResponse(buffer.tobytes(), content_type='image/jpeg')
