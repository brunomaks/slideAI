# use fastapi for convenience, since Django is quite heavy, we can use it for building the web interface later
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
# use opencv for image processing
import cv2
import numpy as np
import os

app = FastAPI()

OUTPUT_FOLDER = "output"

@app.post("/grayscale/")
async def convert_grayscale(file: UploadFile):
    # read the uploaded file
    file_bytes = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # convert the file to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save the grayscale image to the output folder
    filename = file.filename
    path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(path, gray)
    # return the saved file path as a response
    return {"message": f"Saved grayscale image as {path}"}
