import numpy as np
import cv2
import argparse
import os 


parser = argparse.ArgumentParser()

#-h HEIGHT -w WIDTH -i INPUT -o OUTPUT 
parser.add_argument("-height", "--height", dest = "height", default = 128, help="Output image height", type=int)
parser.add_argument("-width", "--width", dest = "width", default = 128, help="Output image width", type=int)
parser.add_argument("-input", "--input", dest = "input", default = "./input/", help="Path to input folder")
parser.add_argument("-output", "--output", dest = "output", default = "./output/", help="Path to output folder")

args = parser.parse_args()

NEW_WIDTH = args.width
NEW_HEIGHT = args.height
input_path = args.input
output_path = args.output




def resize_and_save(image, save_file_path):
    
    resized_img = cv2.resize(image, (NEW_WIDTH, NEW_HEIGHT))
    
    
    # Save cropped image
    cv2.imwrite(save_file_path, resized_img)





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
