import numpy as np
import cv2


NEW_WIDTH = 128
NEW_HEIGHT = 128

def resize_and_save(image, save_file_path):
    
    resized_img = cv2.resize(image, (NEW_WIDTH, NEW_HEIGHT))
    
    
    # Save cropped image
    cv2.imwrite(save_file_path, resized_img)



import os 

input_path = "./input/"
output_path = "./output/"



for subdir, dirs, files in os.walk(input_path, topdown=True):
    for dir in dirs:
        if not os.path.isdir(os.path.join(output_path, dir)):
            os.makedirs(os.path.join(output_path, dir))
    for file in files:
        if not file.endswith((".jpg", ".jpeg", ".png")):
            continue
        file = os.path.relpath(os.path.join(subdir, file), input_path)
        
# Load the input image.
        input_file_path = os.path.join(input_path, file)
        image = cv2.imread(input_file_path)

# Resize and save image 
        output_file_path = os.path.join(output_path, file)
        print("processing ", file, "...")
        resize_and_save(image, output_file_path)
