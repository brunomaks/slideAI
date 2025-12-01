from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 20  # pixels
def save_detected_hand(rgb_image, detection_result, save_file_path):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
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
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Get the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    box_x_l = int(min(x_coordinates) * width) - MARGIN
    box_y_l = int(min(y_coordinates) * height) - MARGIN
    box_x_r = int(max(x_coordinates) * width) + MARGIN
    box_y_r = int(max(y_coordinates) * height) + MARGIN

    # Ensure the bounding box is within the image dimensions
    box_x_l = max(box_x_l, 0)
    box_y_l = max(box_y_l, 0)
    box_x_r = min(box_x_r, width)
    box_y_r = min(box_y_r, height)

    if box_x_l >= box_x_r or box_y_l >= box_y_r:
      print(f"Invalid bounding box dimensions for {os.path.basename(save_file_path)}. Skipping this hand.")
      continue

    # Crop the image by the bounding box
    cropped_img = annotated_image[box_y_l:box_y_r, box_x_l:box_x_r]

    handedness_label = handedness[0].category_name.lower()
    base_name, ext = os.path.splitext(save_file_path)
    hand_save_path = f"{base_name}_{handedness_label}{ext}"
    
    # Save cropped image
    try:
        to_save_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(hand_save_path, to_save_rgb)
    except Exception as e:
        print(f"Error saving image {hand_save_path}: {e}")

# STEP 1: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

        
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
# STEP 2: Load the input image.
        image = mp.Image.create_from_file(os.path.join(input_path, file))

# STEP 3: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)

# STEP 4: Save cropped image 
        save_detected_hand(image.numpy_view()[:,:,:3], detection_result, os.path.join(output_path, file))
