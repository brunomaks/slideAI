from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import requests
from dotenv import load_dotenv
import os

#csrf_exempt is needed since without it we get a 403 error when posting the request
@csrf_exempt
def main_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST a frame request"}, status=400)

    # read the uploaded file
    if "frame" not in request.FILES:
        return JsonResponse({"error": "No frame provided"}, status=400)
    if "frame_number" not in request.POST:
        return JsonResponse({"error": "No frame order number provided"}, status=400)

    frame = request.FILES["frame"]
    frame_number = request.POST["frame_number"]

    # read the uploaded file
    frame_bytes = frame.read()

    # alternatively, if frame is sent as a url
    # frame_body_response = requests.get(frame)
    # files = {
    #     "image": (f"frame{frame_number}.png", frame_body_response.content, "image/png")
    # }

    files = {
        "image": (f"frame{frame_number}.png", frame_bytes, "image/png")
    }

    # load environment variables
    load_dotenv()
    GRAYSCALE_URL = os.getenv('GRAYSCALE_URL')
    FLIP_URL = os.getenv('FLIP_URL')
    HOST_URL = os.getenv('MAIN_URL')

    # post to grayscale service
    grayscale_resp = requests.post(GRAYSCALE_URL, files=files)

    # post to flip service
    frame_to_flip_url = grayscale_resp.json().get("url")
    frame_to_flip_name = grayscale_resp.json().get("filename")

    resp = requests.get(HOST_URL + frame_to_flip_url)
    if resp.status_code != 200:
        return JsonResponse({"error": "Could not load grayscale frame"}, status=500)

    files_to_flip = {
        "image": (frame_to_flip_name, resp.content, "image/png")
    }

    flip_resp = requests.post(FLIP_URL, files=files_to_flip)

    return JsonResponse(flip_resp.json(), safe=False)
