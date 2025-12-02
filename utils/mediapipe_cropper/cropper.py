import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import argparse
import tqdm

MARGIN = 10  # pixels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--input", dest = "input", default = "input/", help="Path to input folder")
    parser.add_argument("-output", "--output", dest = "output", default = "output/", help="Path to output folder")
    return parser.parse_args()

def save_detected_hand(rgb_image, detection_result, save_file_path):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    if not hand_landmarks_list:
        fname = os.path.basename(save_file_path)
        print(f"No hands detected in the image: {fname}")
        return

    # Skip if more than one hand is detected
    if len(hand_landmarks_list) > 1:
        fname = os.path.basename(save_file_path)
        print(f"Multiple hands detected in the image: {fname}. Skipping.")
        return

# Loop through the detected hands to visualize.
    hand_landmarks = hand_landmarks_list[0]

    # Get the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    box_x_min = int(min(x_coordinates) * width) - MARGIN
    box_y_min = int(min(y_coordinates) * height) - MARGIN
    box_x_max = int(max(x_coordinates) * width) + MARGIN
    box_y_max = int(max(y_coordinates) * height) + MARGIN

    # Ensure the bounding box is within the image dimensions
    box_x_min = max(box_x_min, 0)
    box_y_min = max(box_y_min, 0)
    box_x_max = min(box_x_max, width)
    box_y_max = min(box_y_max, height)

    if box_x_min >= box_x_max or box_y_min >= box_y_max:
      print(f"Invalid bounding box dimensions for {os.path.basename(save_file_path)}. Skipping this hand.")
      return

    # Crop the image by the bounding box
    cropped_img = annotated_image[box_y_min:box_y_max, box_x_min:box_x_max]

    if cropped_img.size == 0:
        print("Empty crop, skipping...")
        return

    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    
    # Save cropped image
    to_save_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, to_save_rgb)

args = parse_args()
input_path = args.input
output_path = args.output

print(f"Input path: {input_path}")
print(f"Output path: {output_path}")

# STEP 1: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

for subdir, dirs, files in os.walk(input_path, topdown=True):
    print(f"Processing directory: {subdir}")
    for file in tqdm.tqdm(files, unit="img"):
        if not file.endswith((".jpg", ".jpeg", ".png")):
            continue
        file = os.path.relpath(os.path.join(subdir, file), input_path)
# STEP 2: Load the input image.
        image = mp.Image.create_from_file(os.path.join(input_path, file))

# STEP 3: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)

# STEP 4: Save cropped image 
        save_detected_hand(image.numpy_view()[:,:,:3], detection_result, os.path.join(output_path, file))
