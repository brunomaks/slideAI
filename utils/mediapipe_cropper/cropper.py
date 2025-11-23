from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def save_detected_hand(rgb_image, detection_result, save_file_path):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # List the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    # Get the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    box_x_l = int(min(x_coordinates) * width) - MARGIN
    box_y_l = int(min(y_coordinates) * height) - MARGIN
    box_x_r = int(max(x_coordinates) * width) + MARGIN
    box_y_r = int(max(y_coordinates) * height) + MARGIN


    # Crop the image by the bounding box
    cropped_img = annotated_image[box_y_l:box_y_r, box_x_l:box_x_r]

    handedness_label = handedness[0].category_name.lower()
    base_name, ext = os.path.splitext(save_file_path)
    hand_save_path = f"{base_name}_{handedness_label}{ext}"
    
    # Save cropped image
    to_save_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(hand_save_path, to_save_rgb)



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
        if not file.endswith((".jpg", ".jpeg")):
            continue
        file = os.path.relpath(os.path.join(subdir, file), input_path)
# STEP 2: Load the input image.
        image = mp.Image.create_from_file(os.path.join(input_path, file))

# STEP 3: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)

# STEP 4: Save cropped image 
        save_detected_hand(image.numpy_view()[:,:,:3], detection_result, os.path.join(output_path, file))
