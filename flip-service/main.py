# use fastapi for convenience, since Django is quite heavy, we can use it for building the web interface later
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
# use opencv for image processing
import cv2
import numpy as np
import os

app = FastAPI()

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/flip/")
async def flip_horizontal(file: UploadFile = File(...)):
    # read the uploaded file
    file_bytes = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # flip the image horizontally
    flipped = cv2.flip(img, 1)

    # save the flipped image to the output folder
    filename = file.filename
    out_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(out_path, flipped)
    # return the saved file path as a response
    return {"message": f"Saved flipped image as {out_path}"}
