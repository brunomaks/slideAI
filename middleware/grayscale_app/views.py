from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# use opencv for image processing
import cv2
import numpy as np
import os

#csrf_exempt is needed since without it we get a 403 error when posting the request
@csrf_exempt
def grayscale_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST an image file"}, status=400)

    # read the uploaded file
    if "image" not in request.FILES:
        return JsonResponse({"error": "No image provided"}, status=400)

    file = request.FILES["image"]

    # read the uploaded file
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # convert the file to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save the grayscale image to the output folder
    filename = file.name
    path = os.path.join(settings.MEDIA_ROOT, filename)
    cv2.imwrite(path, gray)

    file_url = settings.MEDIA_URL + filename
    # return url to the updated file
    return JsonResponse({
            "message": "Grayscale image saved",
            "filename": filename,
            "url": file_url
        })
